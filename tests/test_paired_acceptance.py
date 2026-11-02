from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import pytest

from core.model_errors import PermanentModelProviderError
from scripts import paired_acceptance as paired
from scripts.evaluate_context_recall import score_context_recall_case


WHEN = datetime(2026, 11, 2, 12, tzinfo=timezone.utc)


def cases():
    return [{'id': key, 'question': key, 'gold_chunk_ids': [key]} for key in ('a', 'b', 'c')]


def runtime(tmp_path):
    cfg = SimpleNamespace(**paired.COMMON_FLAGS, ENABLE_RERANKER=False,
                          RERANKER_BACKEND='llm', TOP_K=5, RERANK_TOP_N=3)
    def retrieve(query, *, diagnostics, **kwargs):
        rows = [SimpleNamespace(chunk_id=query), SimpleNamespace(chunk_id='other')]
        diagnostics.update(rerank_candidate_ids=[r.chunk_id for r in rows], scoring_query=query)
        return rows
    def rerank(query, rows, *, diagnostics, **kwargs):
        diagnostics.update(status='succeeded', backend='cross_encoder')
        return rows[:1]
    corpus = tmp_path / 'emails.json'
    corpus.write_text('[]', encoding='utf-8')
    return SimpleNamespace(cfg=cfg, RunContext=lambda **kw: SimpleNamespace(**kw),
        use_run_context=lambda _: nullcontext(), control_types=(),
        model_metrics_snapshot=lambda _: {'call_count': 1}, retrieve=retrieve,
        rerank=rerank, score=score_context_recall_case, corpus_path=corpus,
        reset_circuit_breaker=lambda: None, index_snapshot=nullcontext)


def test_portable_root_and_reject_already_imported_foreign_project(tmp_path, monkeypatch):
    assert paired.DEFAULT_ROOT == Path(paired.__file__).resolve().parents[1]
    monkeypatch.setitem(sys.modules, 'core.foreign', SimpleNamespace(__file__=str(tmp_path / 'foreign.py')))
    with pytest.raises(paired.AcceptanceInvariantError, match='core.foreign'):
        paired._validate_import_sources(paired.DEFAULT_ROOT)


def test_provenance_covers_scope_provider_and_dependency_changes(tmp_path):
    for name in ('agents/execution_scope.py', 'core/model_errors.py', 'scripts/paired_acceptance.py', 'requirements.txt'):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('original', encoding='utf-8')
    before = paired._source_hashes(tmp_path)
    assert set(before) == {'agents/execution_scope.py', 'core/model_errors.py', 'scripts/paired_acceptance.py', 'requirements.txt'}
    (tmp_path / 'core/model_errors.py').write_text('changed', encoding='utf-8')
    assert before != paired._source_hashes(tmp_path)


def test_same_candidate_pool_is_preserved_and_hidden_truncation_rejected(tmp_path):
    rt = runtime(tmp_path)
    seen = []
    def recording_rerank(query, rows, *, diagnostics, **kwargs):
        seen.append([row.chunk_id for row in rows])
        diagnostics.update(status='succeeded', backend='cross_encoder')
        return rows[:1]
    rt.rerank = recording_rerank
    report = paired.evaluate_pair(cases()[0], rt, WHEN, fetch_k=20, top_n=3, deadline_seconds=10)
    assert seen == [report['candidate_ids']] == [['a', 'other']]
    assert report['versions']['V2']['retrieved_chunk_ids'] == ['a', 'other']
    original = rt.retrieve
    def truncated(*args, **kwargs):
        return original(*args, **kwargs)[:1]
    rt.retrieve = truncated
    with pytest.raises(paired.AcceptanceInvariantError):
        paired.evaluate_pair(cases()[0], rt, WHEN, fetch_k=20, top_n=3, deadline_seconds=10)


def test_business_http_failure_is_scored_but_wrapped_provider_failure_stops(tmp_path):
    rt = runtime(tmp_path)
    business = ValueError('private')
    business.status_code = 404
    def fail(*args, **kwargs):
        raise business
    rt.retrieve = fail
    report = paired.evaluate_pair(cases()[0], rt, WHEN, fetch_k=20, top_n=3, deadline_seconds=10)
    assert report['complete']
    assert all(row['status'] == 'error' and row['context_recall'] == 0 for row in report['versions'].values())
    business.__cause__ = PermanentModelProviderError(402, 'filter')
    with pytest.raises(ValueError):
        paired.evaluate_pair(cases()[0], rt, WHEN, fetch_k=20, top_n=3, deadline_seconds=10)


def test_main_provider_abort_preserves_paid_cases_without_aggregate_or_next_call(tmp_path, monkeypatch):
    rt = runtime(tmp_path)
    gold = tmp_path / 'gold.json'
    gold.write_text(json.dumps(cases()), encoding='utf-8')
    output = tmp_path / 'result.json'
    called = []
    original = rt.retrieve
    def retrieve(query, **kwargs):
        called.append(query)
        if query == 'b':
            raise RuntimeError('sensitive wrapper') from PermanentModelProviderError(404, 'filter')
        return original(query, **kwargs)
    rt.retrieve = retrieve
    monkeypatch.setattr(paired, '_runtime', lambda _: rt)
    monkeypatch.setattr(paired, '_metadata', lambda *a: {})
    monkeypatch.chdir(paired.DEFAULT_ROOT)
    assert paired.main(['--gold', str(gold), '--output', str(output)]) == 2
    result = json.loads(output.read_text(encoding='utf-8'))
    assert called == ['a', 'b']
    assert result['completed_n'] == 1 and result['planned_n'] == 3
    assert result['complete'] is False and result['batch_status'] == 'stopped_provider_error'
    assert result['provider_error']['http_status'] == 404
    assert all(v['summary'] is None and len(v['records']) == 1 for v in result['results'])
    assert 'sensitive' not in output.read_text(encoding='utf-8')
    before = output.read_bytes()
    with pytest.raises(SystemExit):
        paired.main(['--gold', str(gold), '--output', str(output)])
    assert output.read_bytes() == before


def test_checkpoint_does_not_publish_partial_mean(tmp_path):
    path = tmp_path / 'result.json'
    report = dict(complete=False, paired_cases=[])
    paired._checkpoint(path, report)
    assert all(item['summary'] is None for item in json.loads(path.read_text())['results'])


def test_provider_endpoint_strips_credentials_and_queries():
    assert paired._provider_endpoint(SimpleNamespace(DEEPSEEK_BASE_URL='https://user:secret@example.org:8443/api?key=secret')) == 'https://example.org:8443'

@pytest.mark.parametrize('bad_labels', [[], [''], [' '], [1], [None], [['a']], [{'x':'a'}], ['a', 'a']])
@pytest.mark.parametrize('entrypoint', ['paired', 'recall'])
def test_cli_rejects_invalid_gold_labels_before_runtime(tmp_path, monkeypatch, bad_labels, entrypoint):
    from scripts import evaluate_context_recall as recall
    gold = tmp_path / 'bad.json'
    gold.write_text(json.dumps([{'id':'a','question':'q','gold_chunk_ids':bad_labels}]), encoding='utf-8')
    output = tmp_path / 'not-created.json'
    monkeypatch.setattr(paired, '_runtime', lambda _: pytest.fail('must not initialize runtime'))
    with pytest.raises(SystemExit) as error:
        if entrypoint == 'paired':
            paired.main(['--gold', str(gold), '--output', str(output)])
        else:
            monkeypatch.setattr(sys, 'argv', ['recall', '--gold', str(gold), '--output', str(output)])
            recall.main()
    assert error.value.code == 2 and not output.exists()


@pytest.mark.parametrize('field,value', [('question',''), ('question',' '), ('question',4), ('id',''), ('id',' '), ('id',1)])
def test_invalid_case_identity_and_question_rejected(field, value):
    sample = cases()[:1]
    sample[0][field] = value
    with pytest.raises(ValueError):
        paired.validate_gold_cases(sample)


def test_duplicate_case_id_and_optional_id_policy():
    with pytest.raises(ValueError):
        paired.validate_gold_cases([cases()[0], cases()[0]])
    no_id = [{'question':'valid', 'gold_chunk_ids':['chunk']}]
    paired.validate_gold_cases(no_id)
    with pytest.raises(ValueError):
        paired.validate_gold_cases(no_id, require_case_ids=True)


@pytest.mark.parametrize('count', [0, 1, 2])
def test_real_cross_encoder_fallback_is_failure_but_normal_skips_are_valid(tmp_path, monkeypatch, count):
    from core import reranker
    from models.schemas import SearchResult
    from unittest.mock import Mock
    rt = runtime(tmp_path)
    for name in (*paired.COMMON_FLAGS, 'ENABLE_RERANKER', 'RERANKER_BACKEND'):
        monkeypatch.setattr(reranker.cfg, name, getattr(reranker.cfg, name))
    rt.cfg = reranker.cfg
    rt.rerank = reranker.rerank
    rows = [SearchResult(chunk_id=name, email_id=name, content='fixture', score=1, metadata={})
            for name in ('a','other')][:count]
    def retrieve(query, *, diagnostics, **kwargs):
        diagnostics.update(rerank_candidate_ids=[row.chunk_id for row in rows], scoring_query=query)
        return rows
    rt.retrieve = retrieve
    predict = Mock(side_effect=RuntimeError('private model failure'))
    monkeypatch.setattr(reranker, '_get_cross_encoder', lambda: SimpleNamespace(predict=predict))
    reranker.reset_circuit_breaker()
    try:
        report = paired.evaluate_pair(cases()[0], rt, WHEN, fetch_k=20, top_n=3, deadline_seconds=10)
        v7 = report['versions']['V7']
        assert report['complete']
        assert v7['status'] == ('error' if count == 2 else 'ok' if count else 'empty')
        assert v7['context_recall'] == (1 if count == 1 else 0)
        if count == 2:
            assert report['versions']['V2']['context_recall'] == 1
            assert predict.call_count == 1 and report['v7_reranker']['status'] == 'failed'
        else:
            predict.assert_not_called()
    finally:
        reranker.reset_circuit_breaker()


@pytest.mark.parametrize('diagnostics', [{}, {'status':'succeeded','backend':'llm'}, {'status':'skipped','reason':'insufficient_candidates'}])
def test_pair_cannot_claim_cross_encoder_without_valid_diagnostics(tmp_path, diagnostics):
    rt = runtime(tmp_path)
    def rerank(query, rows, **kwargs):
        kwargs['diagnostics'].update(diagnostics)
        return rows
    rt.rerank = rerank
    with pytest.raises(paired.AcceptanceInvariantError):
        paired.evaluate_pair(cases()[0], rt, WHEN, fetch_k=20, top_n=3, deadline_seconds=10)


def test_pair_open_circuit_is_scored_error(tmp_path):
    rt = runtime(tmp_path)
    def rerank(query, rows, *, diagnostics, **kwargs):
        diagnostics.update(status='skipped', reason='circuit_open')
        return rows
    rt.rerank = rerank
    report = paired.evaluate_pair(cases()[0], rt, WHEN, fetch_k=20, top_n=3, deadline_seconds=10)
    assert report['versions']['V7']['status'] == 'error'
    assert report['versions']['V7']['context_recall'] == 0
