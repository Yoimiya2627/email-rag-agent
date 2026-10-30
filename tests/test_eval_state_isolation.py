from types import SimpleNamespace
from unittest.mock import patch

import pytest

import config.settings as cfg
from agents.approvals import ApprovalStore
from agents.execution_scope import ExecutionScope, current_execution_scope, use_execution_scope
from agents.runtime import RunContext, use_run_context
from agents.tools import send_email
from scripts import run_agent_eval as evaluation


def test_evaluate_task_writes_only_isolated_pending_approvals(monkeypatch, tmp_path):
    daily = tmp_path / 'daily.sqlite3'
    monkeypatch.setattr(cfg, 'APPROVAL_STORE_PATH', str(daily))
    monkeypatch.setattr(cfg, 'AGENT_TOOL_BACKEND', 'local')
    monkeypatch.setattr(cfg, 'MAIL_PROVIDER', 'gmail')
    from agents.mail_providers import GmailDraftProvider
    monkeypatch.setattr(GmailDraftProvider, '_authorization_snapshot',
                        lambda *a: pytest.fail('evaluation must not read daily credentials'))
    observed = []
    def fake_loop(request, *, owner_id, session_id):
        observed.append((owner_id, session_id, current_execution_scope()))
        with use_run_context(RunContext(owner_id=owner_id, session_id=session_id)):
            result = send_email(['test@example.com'], 'test', 'test body', 'evaluation')
        return SimpleNamespace(answer='test', sources=[], metadata={
            'steps':[{'tool':'send_email','status':result['status']}],
            'status':'approval_required','trace_id':'synthetic'})
    with patch('agents.agent_loop.run_agent_loop', side_effect=fake_loop), \
         patch.object(evaluation, 'judge_success', return_value={'success':1, 'reason':'test'}):
        result = evaluation.evaluate_task({'id':'test', 'task':'test', 'expected_tools':['send_email']},
                                         object(), run_dir=tmp_path/'evaluation')
    assert result['success'] == 1 and not daily.exists()
    owner, session, scope = observed[0]
    assert owner.startswith('eval:') and session.startswith('eval:')
    store = ApprovalStore(scope.approval_store_path)
    assert store.list(owner_id='local') == []
    assert len(store.list(owner_id=owner)) == 1
    assert store.list(owner_id=owner)[0]['payload']['execution_binding'] == {'version': 1, 'provider': 'simulated'}
    assert cfg.APPROVAL_STORE_PATH == str(daily) and current_execution_scope() is None


def test_remote_evaluation_fails_before_agent_call(monkeypatch):
    monkeypatch.setattr(cfg, 'AGENT_TOOL_BACKEND', 'mcp')
    with patch('agents.agent_loop.run_agent_loop') as loop:
        with pytest.raises(ValueError, match='not isolated'):
            evaluation.evaluate_task({'id':'test','task':'test'}, object())
        loop.assert_not_called()


def test_explicit_run_directory_cannot_alias_application_store(monkeypatch, tmp_path):
    monkeypatch.setattr(cfg, 'APPROVAL_STORE_PATH', str(tmp_path / 'approvals.json'))
    monkeypatch.setattr(cfg, 'AGENT_TOOL_BACKEND', 'local')
    with pytest.raises(ValueError, match='differ'):
        with evaluation.evaluation_scope(tmp_path):
            pytest.fail('must not enter unsafe scope')


def test_reused_evaluation_scope_cannot_alias_daily_store(monkeypatch, tmp_path):
    daily=tmp_path/'daily.sqlite3'
    monkeypatch.setattr(cfg,'APPROVAL_STORE_PATH',str(daily))
    scope=ExecutionScope('eval:test',approval_store_path=daily,run_dir=tmp_path,evaluation=True)
    with use_execution_scope(scope), pytest.raises(ValueError,match='differ'):
        with evaluation.evaluation_scope():
            pytest.fail('must reject unsafe inherited scope')


def test_eval_cases_receive_isolated_durable_context_and_distinct_sessions(monkeypatch, tmp_path):
    observed = []
    def loop(request, *, owner_id, session_id):
        scope = current_execution_scope()
        assert scope.session_id == session_id
        assert scope.session_repository.validate_context_epoch(owner_id, session_id, scope.context_epoch)
        ref = scope.tool_result_store.put(owner=owner_id, session=session_id, run='fixture-run',
            epoch=scope.context_epoch, call_id='call-1', tool='email_stats', argument_hash='a'*64,
            value={'status':'success','data':{'count':7}})
        observed.append((owner_id, session_id, scope, ref))
        return SimpleNamespace(answer='7', sources=[], metadata={
            'steps':[{'tool':'email_stats','status':'success'}], 'status':'success', 'trace_id':'fixture'})
    monkeypatch.setattr(cfg, 'CONTEXT_TOOL_RESULTS_ENABLED', True)
    monkeypatch.setattr(evaluation, 'judge_success', lambda *a,**kw:{'success':1,'reason':'fixture'})
    with evaluation.evaluation_scope(tmp_path/'eval'), patch('agents.agent_loop.run_agent_loop', side_effect=loop):
        for identity in ('case-a','case-b'):
            evaluation.evaluate_task({'id':identity,'task':'count','expected_tools':['email_stats']},object())
    assert observed[0][0] == observed[1][0]
    assert observed[0][1] != observed[1][1]
    _, session, scope, ref = observed[0]
    assert scope.session_repository.path.parent == (tmp_path/'eval').resolve()
    assert scope.tool_result_store.path.parent == (tmp_path/'eval').resolve()
    from core.tool_results import ToolResultUnavailable
    with pytest.raises(ToolResultUnavailable):
        scope.tool_result_store.load(ref['result_id'], owner=scope.owner_id, session=observed[1][1],
                                    run='fixture-run', epoch=scope.context_epoch)
    assert current_execution_scope() is None


@pytest.mark.parametrize('filename,setting', [('sessions.sqlite3','SESSION_STORE_PATH'),
                                            ('tool_results.sqlite3','TOOL_RESULT_STORE_PATH')])
def test_eval_context_stores_cannot_alias_daily_state(monkeypatch,tmp_path,filename,setting):
    monkeypatch.setattr(cfg, setting, str(tmp_path/filename))
    with pytest.raises(ValueError,match='differ'):
        with evaluation.evaluation_scope(tmp_path):
            pytest.fail('must reject a daily store alias')
    assert not (tmp_path/filename).exists()
