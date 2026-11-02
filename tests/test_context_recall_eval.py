from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json

import pytest

from models.schemas import SearchResult


def _result(chunk_id: str) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        email_id=chunk_id.split("_")[0],
        content=f"content {chunk_id}",
        score=1.0,
        metadata={},
    )


def test_build_gold_template_keeps_question_and_leaves_chunk_ids_blank():
    from scripts import evaluate_context_recall as recall

    template = recall.build_gold_template(
        [
            {
                "question": "When is the budget meeting?",
                "ground_truth": "Monday",
                "email_ids": ["email_1"],
            }
        ],
        limit=1,
    )

    assert template == [
        {
            "id": "gold_001",
            "question": "When is the budget meeting?",
            "ground_truth": "Monday",
            "source_email_ids": ["email_1"],
            "gold_chunk_ids": [],
            "notes": "Fill gold_chunk_ids after inspecting retrieved/source chunks.",
        }
    ]


def test_score_context_recall_case_counts_hits_against_gold_chunks():
    from scripts import evaluate_context_recall as recall

    scored = recall.score_context_recall_case(
        {
            "id": "gold_001",
            "question": "q",
            "gold_chunk_ids": ["c1", "c3"],
        },
        [_result("c1"), _result("c2")],
    )

    assert scored["context_recall"] == 0.5
    assert scored["hit_count"] == 1
    assert scored["gold_count"] == 2
    assert scored["retrieved_chunk_ids"] == ["c1", "c2"]
    assert scored["hit_chunk_ids"] == ["c1"]
    assert scored["has_any_hit"] is True
    assert scored["has_perfect_recall"] is False


def test_evaluate_context_recall_cases_aggregates_recall_rates():
    from scripts import evaluate_context_recall as recall

    cases = [
        {"id": "a", "question": "qa", "gold_chunk_ids": ["a1", "a2"]},
        {"id": "b", "question": "qb", "gold_chunk_ids": ["b1"]},
    ]

    def fake_retrieve(question: str):
        if question == "qa":
            return [_result("a1"), _result("x")]
        return [_result("b1")]

    report = recall.evaluate_context_recall_cases(cases, retrieve_fn=fake_retrieve)

    assert report["summary"] == {
        "n": 2,
        "mean_context_recall": 0.75,
        "chunk_hit_rate": 1.0,
        "perfect_recall_rate": 0.5,
        "ok_count": 2,
        "empty_count": 0,
        "error_count": 0,
    }
    assert [record["context_recall"] for record in report["records"]] == [0.5, 1.0]


def test_single_case_error_keeps_denominator_and_continues_without_error_text():
    from scripts import evaluate_context_recall as recall

    cases = [
        {"id": name, "question": name, "gold_chunk_ids": [name]}
        for name in ("first", "broken", "empty", "last")
    ]
    seen = []

    def retrieve(question):
        seen.append(question)
        if question == "broken":
            raise ValueError("sensitive model response") from TypeError("sensitive underlying error")
        return [] if question == "empty" else [_result(question)]

    report = recall.evaluate_context_recall_cases(cases, retrieve)

    assert seen == [case["question"] for case in cases]
    assert report["summary"] == {
        "n": 4, "mean_context_recall": 0.5, "chunk_hit_rate": 0.5, "perfect_recall_rate": 0.5,
        "ok_count": 2, "empty_count": 1, "error_count": 1,
    }
    assert [row["status"] for row in report["records"]] == ["ok", "error", "empty", "ok"]
    failed = report["records"][1]
    assert failed["retrieved_chunk_ids"] == []
    assert failed["context_recall"] == 0.0
    assert failed["error_type"] == "ValueError"
    assert failed["cause_type"] == "TypeError"
    assert "sensitive" not in json.dumps(report)


def test_reclassifying_existing_empty_cases_as_errors_does_not_change_recall():
    from scripts import evaluate_context_recall as recall

    cases = [{"id": str(i), "question": str(i), "gold_chunk_ids": [str(i)]} for i in range(30)]

    def retrieve(question, *, fail=False):
        index = int(question)
        if index < 22:
            return [_result(question)]
        if fail and index < 26:
            raise ValueError("invalid model filter")
        return []

    empty = recall.evaluate_context_recall_cases(cases, retrieve)
    failed = recall.evaluate_context_recall_cases(cases, lambda q: retrieve(q, fail=True))
    assert empty["summary"]["mean_context_recall"] == failed["summary"]["mean_context_recall"] == 0.7333
    assert failed["summary"]["n"] == 30
    assert failed["summary"]["empty_count"] == failed["summary"]["error_count"] == 4


@pytest.mark.parametrize("error_name", [
    "RunCancelled", "RunDeadlineExceeded", "ContextBudgetExceeded", "ModelBudgetExceeded",
    "KeyboardInterrupt", "SystemExit",
])
@pytest.mark.parametrize("wrapped", [False, True])
def test_control_errors_abort_evaluation(error_name, wrapped):
    from scripts import evaluate_context_recall as recall

    error_type = {"KeyboardInterrupt": KeyboardInterrupt, "SystemExit": SystemExit}.get(error_name)
    if error_type is None:
        error_type = getattr(recall, error_name)
    seen = []

    def retrieve(question):
        seen.append(question)
        error = error_type("stop")
        if wrapped:
            raise ValueError("wrapper") from error
        raise error

    cases = [{"question": "first", "gold_chunk_ids": ["a"]}, {"question": "never", "gold_chunk_ids": ["b"]}]
    with pytest.raises(ValueError if wrapped else error_type):
        recall.evaluate_context_recall_cases(cases, retrieve)
    assert seen == ["first"]


def test_cyclic_exception_causes_do_not_hang_evaluation():
    from scripts import evaluate_context_recall as recall

    error = ValueError("invalid")
    error.__cause__ = error

    def retrieve(_):
        raise error

    report = recall.evaluate_context_recall_cases([{"question": "q", "gold_chunk_ids": ["a"]}], retrieve)
    assert report["summary"]["error_count"] == 1


@pytest.mark.parametrize("error_name", [
    "RunCancelled", "RunDeadlineExceeded", "ContextBudgetExceeded", "ModelBudgetExceeded",
])
@pytest.mark.parametrize("explicit_other_cause", [False, True])
def test_implicit_control_error_context_stops_before_next_case(error_name, explicit_other_cause):
    from scripts import evaluate_context_recall as recall

    error_type = getattr(recall, error_name)
    seen = []

    def retrieve(question):
        seen.append(question)
        try:
            raise error_type("stop")
        except error_type:
            if explicit_other_cause:
                # A separate explicit cause must not hide the control signal
                # retained in Python's implicit exception context.
                raise ValueError("wrapper") from TypeError("ordinary cause")
            raise ValueError("wrapper")

    cases = [{"question": "first", "gold_chunk_ids": ["a"]}, {"question": "never", "gold_chunk_ids": ["b"]}]
    with pytest.raises(ValueError):
        recall.evaluate_context_recall_cases(cases, retrieve)
    assert seen == ["first"]


def test_malformed_gold_is_not_reported_as_a_retrieval_failure():
    from scripts import evaluate_context_recall as recall

    with pytest.raises(KeyError):
        recall.evaluate_context_recall_cases([{"gold_chunk_ids": ["a"]}], lambda _: [])
    with pytest.raises(TypeError):
        recall.evaluate_context_recall_cases([{"question": "q", "gold_chunk_ids": None}], lambda _: [])


def test_version_records_failure_diagnostics_and_uses_one_reference_time(monkeypatch):
    from scripts import evaluate_context_recall as recall
    import core.pipeline as pipeline

    when = datetime(2026, 11, 2, 12, tzinfo=timezone.utc)
    seen_times = []
    captured = []

    def retrieve(query, *, diagnostics, reference_time):
        assert diagnostics == {}
        seen_times.append(reference_time)
        diagnostics.update(stage=query, filters={"date_hint": query})
        captured.append(diagnostics)
        if query == "broken":
            raise ValueError("invalid date")
        return [_result(query)]

    monkeypatch.setattr(pipeline, "retrieve", retrieve)
    monkeypatch.setattr(recall, "apply_flags", lambda _: None)
    cases = [{"question": q, "gold_chunk_ids": [q]} for q in ("broken", "valid")]
    report = recall.evaluate_version("V2", cases, reference_time=when)
    assert seen_times == [when, when]
    assert report["reference_time"] == when.isoformat()
    assert report["records"][0]["diagnostics"] == {"stage": "broken", "filters": {"date_hint": "broken"}}
    captured[-1]["filters"]["date_hint"] = "changed later"
    assert report["records"][1]["diagnostics"]["filters"]["date_hint"] == "valid"


@pytest.mark.parametrize("raw", ["2026-11-02", "2026-11-02T12:00:00", "not a date"])
def test_reference_time_requires_an_explicit_timezone(raw):
    from scripts import evaluate_context_recall as recall

    with pytest.raises(argparse.ArgumentTypeError):
        recall._parse_reference_time(raw)
    assert recall._parse_reference_time("2026-11-02T12:00:00Z") == datetime(2026, 11, 2, 12, tzinfo=timezone.utc)


def test_main_freezes_default_time_once_across_versions(monkeypatch, tmp_path):
    from scripts import evaluate_context_recall as recall
    import core.pipeline as pipeline

    gold = tmp_path / "gold.json"
    output = tmp_path / "report.json"
    gold.write_text('[{"question":"q","gold_chunk_ids":["a"]}]', encoding="utf-8")
    when = datetime(2026, 11, 2, 12, tzinfo=timezone.utc)
    clock_calls = []
    version_times = []

    def now():
        clock_calls.append(True)
        return when

    def evaluate(version, cases, limit=None, *, reference_time, progress_fn=None):
        version_times.append(reference_time)
        return {"version": version, "summary": {
            "n": 1, "mean_context_recall": 0, "chunk_hit_rate": 0, "perfect_recall_rate": 0,
            "ok_count": 0, "empty_count": 1, "error_count": 0,
        }, "records": []}

    monkeypatch.setattr(pipeline, "_now", now)
    monkeypatch.setattr(recall, "evaluate_version", evaluate)
    monkeypatch.setattr(recall.sys, "argv", ["evaluate_context_recall.py", "--gold", str(gold), "--output", str(output)])
    recall.main()
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert clock_calls == [True]
    assert version_times == [when, when]
    assert saved["schema_version"] == 2
    assert saved["run"]["reference_time"] == when.isoformat()
    assert len(saved["run"]["gold_sha256"]) == 64
    assert "API_KEY" not in json.dumps(saved)


def test_package_metadata_failure_does_not_block_run_metadata(monkeypatch, tmp_path):
    from scripts import evaluate_context_recall as recall

    def missing(_):
        raise OSError("broken optional package metadata")

    monkeypatch.setattr(recall.package_metadata, "version", missing)
    metadata = recall._run_metadata(datetime(2026, 11, 2, tzinfo=timezone.utc), tmp_path / "missing.json")
    assert metadata["gold_hash_error_type"] == "FileNotFoundError"
    assert all(version is None for version in metadata["packages"].values())


def test_provider_error_stops_and_preserves_previous_case_checkpoint(monkeypatch, tmp_path):
    from scripts import evaluate_context_recall as recall
    from core.model_errors import PermanentModelProviderError
    import core.pipeline as pipeline
    gold = tmp_path / 'gold.json'
    gold.write_text(json.dumps([{'question': q, 'gold_chunk_ids': [q]} for q in ('a', 'b', 'c')]), encoding='utf-8')
    output = tmp_path / 'result.json'
    calls = []
    def retrieve(query, **kwargs):
        calls.append(query)
        if query == 'b':
            raise ValueError('private wrapper') from PermanentModelProviderError(401, 'filter')
        return [_result(query)]
    monkeypatch.setattr(pipeline, 'retrieve', retrieve)
    monkeypatch.setattr(recall, 'apply_flags', lambda _: None)
    monkeypatch.setattr(recall.sys, 'argv', ['recall', '--gold', str(gold), '--output', str(output), '--reference-time', '2026-11-02T12:00:00+08:00'])
    assert recall.main() == 2
    saved = json.loads(output.read_text(encoding='utf-8'))
    assert calls == ['a', 'b']
    assert saved['complete'] is False and saved['batch_status'] == 'stopped_provider_error'
    assert saved['provider_error']['http_status'] == 401
    assert saved['results'][0]['summary'] is None
    assert len(saved['results'][0]['records']) == 1
    assert 'private wrapper' not in output.read_text(encoding='utf-8')
    before = output.read_bytes()
    with pytest.raises(SystemExit):
        recall.main()
    assert output.read_bytes() == before


def test_plain_http_status_is_not_a_provider_abort():
    from scripts import evaluate_context_recall as recall
    error = ValueError('business missing item')
    error.status_code = 404
    def retrieve(query):
        if query == 'bad':
            raise error
        return [_result(query)]
    report = recall.evaluate_context_recall_cases([{'question': q, 'gold_chunk_ids': [q]} for q in ('bad', 'good')], retrieve)
    assert report['summary']['n'] == 2
    assert report['summary']['error_count'] == 1
    assert report['summary']['mean_context_recall'] == 0.5


@pytest.mark.parametrize('count', [0,1,2])
def test_version_scores_real_reranker_fallback_as_error(monkeypatch, count):
    from scripts import evaluate_context_recall as recall
    from scripts.paired_acceptance import COMMON_FLAGS
    from core import pipeline, reranker
    from types import SimpleNamespace
    from unittest.mock import Mock
    for name in (*COMMON_FLAGS, 'ENABLE_RERANKER', 'RERANKER_BACKEND'):
        monkeypatch.setattr(reranker.cfg, name, getattr(reranker.cfg, name))
    rows = [_result('a'), _result('b')][:count]
    predict = Mock(side_effect=ValueError('model failure'))
    monkeypatch.setattr(reranker, '_get_cross_encoder', lambda: SimpleNamespace(predict=predict))
    def retrieve(query, *, diagnostics, **kwargs):
        diagnostics.update(rerank_candidate_count=len(rows), reranker={})
        return reranker.rerank(query, rows, top_n=3, diagnostics=diagnostics['reranker'])
    monkeypatch.setattr(pipeline, 'retrieve', retrieve)
    try:
        report = recall.evaluate_version('V7', [{'question':'q', 'gold_chunk_ids':['a']}],
                                         reference_time=datetime(2026,11,2,tzinfo=timezone.utc))
        assert report['summary']['n'] == 1
        assert report['summary']['error_count'] == int(count == 2)
        assert report['summary']['mean_context_recall'] == (1 if count == 1 else 0)
        assert predict.call_count == int(count == 2)
    finally:
        reranker.reset_circuit_breaker()


@pytest.mark.parametrize('details', [None, {}, {'status':'succeeded','backend':'llm'}, {'status':'skipped','reason':'circuit_open'}])
def test_version_never_scores_missing_or_wrong_backend_diagnostics_as_success(monkeypatch, details):
    from scripts import evaluate_context_recall as recall
    from core import pipeline
    def retrieve(query, *, diagnostics, **kwargs):
        diagnostics.update(rerank_candidate_count=2, reranker=details)
        return [_result('a')]
    monkeypatch.setattr(pipeline, 'retrieve', retrieve)
    monkeypatch.setattr(recall, 'apply_flags', lambda _: None)
    report = recall.evaluate_version('V7', [{'question':'q','gold_chunk_ids':['a']}],
                                     reference_time=datetime(2026,11,2,tzinfo=timezone.utc))
    assert report['summary']['n'] == 1 and report['summary']['error_count'] == 1
    assert report['summary']['mean_context_recall'] == 0
