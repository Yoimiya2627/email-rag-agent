"""Completed jobs publish their result and recovery boundary atomically."""
from unittest.mock import patch

import pytest

from core.jobs import JobConflict, JobManager, JobStore


def test_completed_result_does_not_depend_on_a_separate_checkpoint_write(tmp_path):
    store = JobStore(tmp_path / 'jobs.sqlite3')
    committed = []
    result = {'answer': 'Synthetic complete answer',
              'metadata': {'status': 'success', 'completion_status': 'complete'}}
    original = store.checkpoint

    def checkpoint(owner, job_id, value):
        if value.get('kind') == 'completed_result':
            raise OSError('synthetic completion checkpoint failure')
        return original(owner, job_id, value)

    def runner(job, event, progress, save_checkpoint):
        save_checkpoint({'safe': True, 'kind': 'agent', 'next_round': 0})
        committed.append(result)
        return result

    manager = JobManager(store, runner)
    with patch.object(store, 'checkpoint', checkpoint):
        job = manager.submit('local', 'agent', {'query': 'synthetic'})
        assert manager.wait_idle(3)
    final = store.get('local', job['id'], private=True)
    assert final['status'] == 'succeeded'
    assert final['result'] == result
    assert final['checkpoint'] == {'safe': True, 'kind': 'completed_result', 'result': result}
    assert not final['resumable']
    with pytest.raises(JobConflict):
        manager.resume('local', job['id'])
    assert committed == [result]


def test_success_publishes_checkpoint_and_result_in_one_update(tmp_path):
    store = JobStore(tmp_path / 'jobs.sqlite3')
    job, _ = store.create('local', 'agent', {'query': 'synthetic'})
    assert store.claim('local', job['id'])
    store.checkpoint('local', job['id'], {'safe': True, 'kind': 'agent'})
    # A SQLite trigger checks the actual persisted transition, not a later read:
    # no successful row may retain a planning checkpoint or a different result.
    with store._db() as db:
        db.execute("""CREATE TRIGGER complete_together BEFORE UPDATE ON jobs
            WHEN NEW.status='succeeded' AND (
                json_extract(NEW.checkpoint,'$.kind') IS NOT 'completed_result'
                OR json_extract(NEW.checkpoint,'$.result') IS NOT NEW.result)
            BEGIN SELECT RAISE(ABORT,'completion not atomic'); END""")
    result = {'answer': 'Synthetic'}
    store.finish('local', job['id'], 'succeeded', result=result)
    assert store.get('local', job['id'])['result'] == result


def test_failed_completion_update_rolls_back_all_fields(tmp_path):
    import sqlite3
    store = JobStore(tmp_path / 'jobs.sqlite3')
    job, _ = store.create('local', 'agent', {'query': 'synthetic'})
    store.claim('local', job['id'])
    planning = {'safe': True, 'kind': 'agent'}
    store.checkpoint('local', job['id'], planning)
    with store._db() as db:
        db.execute("""CREATE TRIGGER fail_completion AFTER UPDATE ON jobs
            WHEN NEW.status='succeeded'
            BEGIN SELECT RAISE(ABORT,'synthetic write failure'); END""")
    with pytest.raises(sqlite3.IntegrityError, match='synthetic write failure'):
        store.finish('local', job['id'], 'succeeded', result={'answer': 'Synthetic'})
    row = store.get('local', job['id'], private=True)
    assert row['status'] == 'running'
    assert row['result'] is None
    assert row['checkpoint'] == planning


def test_real_agent_completion_keeps_one_model_call_and_transcript(tmp_path):
    from fastapi.testclient import TestClient
    import api.main as api
    import agents.agent_loop as loop
    from api.sessions import SessionStore
    from tests.test_agent_loop import _ScriptedClient, _response

    sessions = SessionStore(path=tmp_path / 'sessions.sqlite3')
    store = JobStore(tmp_path / 'jobs.sqlite3')
    manager = JobManager(store, api._run_background_job)
    client = _ScriptedClient([_response(content='Synthetic complete answer')])
    original = store.checkpoint

    def checkpoint(owner, job_id, value):
        if value.get('kind') == 'completed_result':
            raise OSError('synthetic completion checkpoint failure')
        return original(owner, job_id, value)

    with patch.object(api, 'sessions', sessions), patch.object(api, '_jobs', manager), \
         patch.object(loop, '_get_client', return_value=client), \
         patch.object(store, 'checkpoint', checkpoint), \
         patch.object(api.cfg, 'TOOL_RESULT_STORE_PATH', str(tmp_path / 'results.sqlite3')), \
         patch.object(api.cfg, 'AGENT_TRACE_LOG_PATH', str(tmp_path / 'trace.jsonl')):
        http = TestClient(api.app)
        submitted = http.post('/jobs/agent', json={'query': 'synthetic', 'session_id': 's'})
        assert submitted.status_code == 202
        job_id = submitted.json()['id']
        assert manager.wait_idle(5)
        assert http.get('/jobs/' + job_id).json()['status'] == 'succeeded'
        assert http.post('/jobs/' + job_id + '/resume').status_code == 409
    assert len(client.calls) == 1
    turns = sessions.history('local', 's')['turns']
    assert len(turns) == 1
    assert turns[0]['answer'] == 'Synthetic complete answer'
