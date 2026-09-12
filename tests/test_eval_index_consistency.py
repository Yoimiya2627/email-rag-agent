"""A report's manifest must describe every task, even during index activation."""
import json
from types import SimpleNamespace
import pytest

from core import embedder, index_manifest
from scripts import run_agent_eval as runner


def test_manifest_accessor_cannot_mutate_pinned_generation(monkeypatch):
    manifest = {'generation':'A', 'metadata':{'count':1}}
    monkeypatch.setattr(index_manifest, 'read_active_manifest', lambda **kw: manifest)
    with embedder.index_snapshot():
        exposed = embedder.current_index_manifest()
        exposed['generation'] = 'B'
        exposed['metadata']['count'] = 2
        assert embedder.current_index_manifest() == {'generation':'A','metadata':{'count':1}}


@pytest.mark.parametrize('has_manifest', [True, False])
def test_batch_uses_one_snapshot_and_captures_that_same_manifest(tmp_path, monkeypatch, has_manifest):
    active = {'generation':'generation-A', 'collection':'synthetic-A'} if has_manifest else {}
    seen, reads = [], []
    cases = tmp_path / 'cases.json'
    cases.write_text(json.dumps([{'id':'one','task':'one'}, {'id':'two','task':'two'}]), encoding='utf-8')
    directory = tmp_path / 'evaluation'
    monkeypatch.setattr(runner, 'TESTSET_PATH', cases)
    monkeypatch.setattr(runner.sys, 'argv', ['run_agent_eval','--run-dir',str(directory)])
    monkeypatch.setattr(runner, 'build_provenance', lambda **kw: {'artifacts':{}})
    monkeypatch.setattr(runner, 'OpenAI', lambda **kw: SimpleNamespace())
    monkeypatch.setattr(runner.time, 'sleep', lambda duration: None)

    def read_manifest(**kw):
        captured = dict(active)
        reads.append(captured.get('generation'))
        # Activate B immediately after the first read, exercising capture/pin
        # ordering as well as activation between cases. No files or models.
        active.update(generation='generation-B', collection='synthetic-B')
        return captured or None

    monkeypatch.setattr(index_manifest, 'read_active_manifest', read_manifest)

    def evaluate(item, client):
        with embedder.index_snapshot():
            seen.append((embedder.current_index_manifest() or {}).get('generation'))
        return {'id':item['id'],'success':1,'tool_accuracy':True,'n_steps':0,
                'max_steps_reached':False,'forbidden_tool_violation':False,'task_type':'general'}

    monkeypatch.setattr(runner, 'evaluate_task', evaluate)
    runner.main()
    report = json.loads((directory / 'agent_eval.json').read_text(encoding='utf-8'))
    expected = 'generation-A' if has_manifest else None
    assert seen == [expected, expected]
    assert reads == [expected]
    if has_manifest:
        assert report['provenance']['active_index_manifest']['manifest']['generation'] == 'generation-A'
        assert report['provenance']['index_consistency'] == 'batch_snapshot'
    else:
        assert report['provenance']['active_index_manifest']['status'] == 'unavailable'
        assert report['provenance']['index_consistency'] == 'unavailable_no_immutable_generation'
    # ContextVar is reset after completion, so the next request sees B.
    assert embedder.current_index_manifest()['generation'] == 'generation-B'
