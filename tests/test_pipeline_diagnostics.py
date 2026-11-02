"""Retrieval scope, reference-clock isolation and opt-in evaluation evidence."""
import json
from datetime import datetime, timezone
from types import SimpleNamespace as NS

import pytest

import core.pipeline as pipeline
from agents.runtime import ContextBudgetExceeded


def mock_filter_model(monkeypatch, payload):
    monkeypatch.setattr(pipeline, '_get_client', lambda: object())
    monkeypatch.setattr(pipeline, 'create_completion', lambda *args, **kwargs: NS(
        choices=[NS(finish_reason='stop', message=NS(content=json.dumps(payload)))],
    ))
    monkeypatch.setattr(pipeline, 'rewrite_query', lambda query: query)


def test_explicit_day_never_admits_next_day(monkeypatch, make_search_result):
    payload = {'query': '发票', 'sender': 'alice@example.test',
               'date_hint': '2026-03-26', 'labels': ['finance']}
    mock_filter_model(monkeypatch, payload)
    rows = [make_search_result(day, metadata={
        'sender': 'alice@example.test', 'date': day, 'labels': ['finance'],
    }) for day in ['2026-03-26', '2026-03-27']]
    # Deliberately return an out-of-scope candidate to exercise the real
    # defensive filter as well as the exact scope passed to hybrid search.
    seen = []
    def search(query, *, top_k, filters):
        seen.append(filters)
        return rows
    monkeypatch.setattr(pipeline, 'hybrid_search', search)
    monkeypatch.setattr(pipeline.cfg, 'ENABLE_RERANKER', False)
    details = {'stale': True}
    result = pipeline.retrieve(
        '只查 2026 年 3 月 26 日 alice@example.test 发来的、标签为 finance 的发票',
        diagnostics=details,
    )
    assert [row.chunk_id for row in result] == ['2026-03-26']
    assert seen[0].matches(rows[0].metadata)
    assert not seen[0].matches(rows[1].metadata)
    assert 'stale' not in details
    assert details['filters'] == payload
    assert details['hybrid_candidate_count'] == 2
    assert details['post_filter_candidate_count'] == 1
    assert details['rerank_candidate_ids'] == ['2026-03-26']
    assert details['returned_count'] == 1
    assert details['reranker']['status'] == 'disabled'


def test_valid_date_with_no_matches_is_empty_not_error(monkeypatch, make_search_result):
    monkeypatch.setattr(pipeline, 'hybrid_search', lambda *a, **kw: [
        make_search_result('january_report', metadata={'date': '2026-01-02'}),
    ])
    details = {}
    assert pipeline.retrieve('December report', filters={'date_hint': '2025-12'},
                             diagnostics=details) == []
    assert details['status'] == 'empty'
    assert details['post_filter_candidate_count'] == 0
    assert details['reranker']['reason'] == 'empty_candidates'
    assert 'error_type' not in details


def test_diagnostics_preserve_results_and_one_model_call_per_request(monkeypatch, make_search_result):
    monkeypatch.setattr(pipeline, '_get_client', lambda: object())
    monkeypatch.setattr(pipeline, 'rewrite_query', lambda query: query)
    monkeypatch.setattr(pipeline.cfg, 'ENABLE_RERANKER', False)
    calls = []
    def complete(*args, **kwargs):
        calls.append(kwargs)
        return NS(choices=[NS(finish_reason='stop', message=NS(content='{"query":"invoice"}'))])
    monkeypatch.setattr(pipeline, 'create_completion', complete)
    expected = [make_search_result('invoice_0')]
    monkeypatch.setattr(pipeline, 'hybrid_search', lambda *args, **kwargs: expected)
    reference = datetime(2026, 11, 2, tzinfo=timezone.utc)
    plain = pipeline.retrieve('invoice', reference_time=reference)
    assert len(calls) == 1
    assert pipeline._reference_time.get() is None
    details = {}
    observed = pipeline.retrieve('invoice', reference_time=reference, diagnostics=details)
    assert observed == plain == expected
    assert len(calls) == 2
    assert details['extraction']['attempt_count'] == 1
    assert pipeline._reference_time.get() is None


def test_reference_time_uses_retrieval_timezone_and_resets_after_error(monkeypatch):
    monkeypatch.setattr(pipeline.cfg, 'RETRIEVAL_TIMEZONE', '')
    monkeypatch.setattr(pipeline.cfg, 'RETRIEVAL_TIMEZONE_OFFSET_HOURS', 8)
    reference = datetime(2026, 3, 25, 18, tzinfo=timezone.utc)
    seen = []
    def search(query, *, top_k, filters):
        seen.append((pipeline._now(), filters))
        raise RuntimeError('synthetic search failure')
    monkeypatch.setattr(pipeline, 'hybrid_search', search)
    details = {}
    with pytest.raises(RuntimeError, match='synthetic search failure'):
        pipeline.retrieve('invoice', filters={'date_hint': '今天'},
                          reference_time=reference, diagnostics=details)
    assert seen[0][0].isoformat() == '2026-03-26T02:00:00+08:00'
    assert seen[0][1].start.isoformat() == '2026-03-26T00:00:00+08:00'
    assert details['reference_time'] == seen[0][0].isoformat()
    assert details['status'] == 'error'
    assert details['stage'] == 'hybrid_search'
    assert pipeline._reference_time.get() is None


def test_fixed_clock_reaches_extraction_without_becoming_a_missing_year_default(monkeypatch):
    mock_filter_model(monkeypatch, {'query': 'invoice', 'date_hint': '3月26日'})
    reference = datetime(2026, 11, 2, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match='无法可靠解析'):
        pipeline.retrieve('只查 2025 年 3 月 26 日的邮件', reference_time=reference)
    assert pipeline._reference_time.get() is None


@pytest.mark.parametrize('reference', [datetime(2026, 11, 2), '2026-11-02'])
def test_invalid_reference_time_is_rejected_before_model_call(monkeypatch, reference):
    def unexpected(*args, **kwargs):
        pytest.fail('Invalid clock must not start model work')
    monkeypatch.setattr(pipeline, 'rewrite_query', unexpected)
    with pytest.raises(ValueError, match='timezone-aware'):
        pipeline.retrieve('invoice', reference_time=reference)


@pytest.mark.parametrize('bad_field', [
    {'sender': 42}, {'query': []}, {'labels': 'finance'},
    {'labels': ['']}, {'unknown': True}, {'date_hint': None},
])
def test_invalid_fields_remain_errors_and_diagnostics_keep_the_input(monkeypatch, bad_field):
    payload = {'query': 'invoice', 'sender': '', 'labels': [], 'date_hint': 'some day', **bad_field}
    mock_filter_model(monkeypatch, payload)
    details = {}
    with pytest.raises(ValueError, match='无法可靠解析'):
        pipeline.extract_filters('invoice', diagnostics=details)
    assert details['status'] == 'error'
    assert details['parsed_filters'] == payload
    assert details['attempt_count'] == 1


def test_message_budget_failure_is_recorded_without_attempt_or_retry(monkeypatch):
    error = ContextBudgetExceeded('synthetic context budget')
    def build(*args, **kwargs):
        raise error
    def unexpected(*args, **kwargs):
        pytest.fail('No model call is allowed after a message budget failure')
    monkeypatch.setattr(pipeline, 'build_model_messages', build)
    monkeypatch.setattr(pipeline, 'create_completion', unexpected)
    details = {}
    with pytest.raises(ContextBudgetExceeded) as caught:
        pipeline.extract_filters('invoice', diagnostics=details)
    assert caught.value is error
    assert details == {'status': 'error', 'attempt_count': 0, 'error_type': 'ContextBudgetExceeded'}


def test_filter_model_receives_frozen_local_reference_time(monkeypatch):
    monkeypatch.setattr(pipeline.cfg, 'RETRIEVAL_TIMEZONE', '')
    monkeypatch.setattr(pipeline.cfg, 'RETRIEVAL_TIMEZONE_OFFSET_HOURS', 8)
    monkeypatch.setattr(pipeline, '_get_client', lambda: object())
    monkeypatch.setattr(pipeline, 'rewrite_query', lambda query: query)
    monkeypatch.setattr(pipeline, 'hybrid_search', lambda *args, **kwargs: [])
    calls = []
    def complete(*args, **kwargs):
        calls.append(kwargs)
        return NS(choices=[NS(finish_reason='stop', message=NS(content='{"date_hint":"今天"}'))])
    monkeypatch.setattr(pipeline, 'create_completion', complete)
    pipeline.retrieve('今天发来的邮件', reference_time=datetime(2026, 11, 2, 20, tzinfo=timezone.utc))
    assert len(calls) == 1
    assert '2026-11-03T04:00:00+08:00' in json.dumps(calls[0]['messages'], ensure_ascii=False)


@pytest.mark.parametrize('number', ['NaN', 'Infinity', '1e999'])
def test_nonfinite_model_output_cannot_corrupt_diagnostic_json(monkeypatch, number):
    monkeypatch.setattr(pipeline, '_get_client', lambda: object())
    monkeypatch.setattr(pipeline, 'create_completion', lambda *args, **kwargs: NS(
        choices=[NS(finish_reason='stop', message=NS(content='{"date_hint":' + number + '}'))],
    ))
    details = {}
    with pytest.raises(ValueError, match='无法可靠解析'):
        pipeline.extract_filters('invoice', diagnostics=details)
    assert details['status'] == 'error'
    json.dumps(details, allow_nan=False)
