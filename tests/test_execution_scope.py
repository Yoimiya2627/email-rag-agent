from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from agents.approvals import ApprovalStore
from agents.execution_scope import ExecutionScope, current_execution_scope, use_execution_scope
from agents.runtime import RunContext, use_run_context
from agents.tools import send_email


def send(scope, body='body'):
    with use_execution_scope(scope), use_run_context(RunContext(owner_id=scope.owner_id, session_id='session')):
        return send_email(['test@example.com'], 'subject', body, 'test')


def test_cross_run_operation_key_reuses_approval_and_rejects_changed_content(tmp_path):
    path = tmp_path / 'approvals.sqlite3'
    factory = lambda: ExecutionScope('local', operation_key='retry-1', approval_store_path=path)
    first, second = send(factory()), send(factory())
    assert first['approval_id'] == second['approval_id']
    with pytest.raises(ValueError, match='different approval content'):
        send(factory(), body='changed')
    assert len(ApprovalStore(path).list()) == 1
    another = send(ExecutionScope('local', operation_key='new-intent', approval_store_path=path))
    assert another['approval_id'] != first['approval_id']


def test_multiple_actions_keep_stable_slots_on_replay(tmp_path):
    path = tmp_path / 'approvals.sqlite3'
    def run():
        scope = ExecutionScope('local', operation_key='multi', approval_store_path=path)
        return [send(scope, body=body)['approval_id'] for body in ['one', 'two', 'one']]
    first, second = run(), run()
    assert first == second and first[0] == first[2] and first[0] != first[1]


def test_scope_owner_mismatch_is_rejected_before_store_write(tmp_path):
    path = tmp_path / 'never-created.sqlite3'
    with use_execution_scope(ExecutionScope('other', approval_store_path=path)), use_run_context(RunContext(owner_id='local')):
        with pytest.raises(PermissionError):
            send_email(['test@example.com'], 'subject', 'body', 'test')
    assert not path.exists()


def test_two_concurrent_scopes_do_not_share_state(tmp_path):
    barrier = threading.Barrier(2)
    def run(index):
        path = tmp_path / f'{index}.sqlite3'
        scope = ExecutionScope(f'eval:{index}', approval_store_path=path,
                               run_dir=tmp_path / str(index), evaluation=True)
        with use_execution_scope(scope):
            barrier.wait()
            assert current_execution_scope() is scope
            send(scope)
        return ApprovalStore(path).list(owner_id=scope.owner_id)
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(run, [1, 2]))
    assert [row[0]['owner_id'] for row in rows] == ['eval:1', 'eval:2']
    assert current_execution_scope() is None


@pytest.mark.parametrize('key', ['', ' ', 'line\nbreak', 'x'*129])
def test_operation_key_validation(key):
    with pytest.raises(ValueError):
        ExecutionScope('local', operation_key=key)


def test_resume_restores_slots_before_new_action():
    first=ExecutionScope('local',operation_key='run')
    original=first.approval_request_id('a'*64)
    restored=ExecutionScope('local',operation_key='run')
    restored.restore_slots(first.snapshot_slots())
    assert restored.approval_request_id('a'*64)==original
    assert restored.approval_request_id('b'*64).endswith(':send:2')
    for invalid in ({'x':1},{'a'*64:True},{'a'*64:2},{'a'*64:1,'b'*64:1}):
        with pytest.raises(ValueError):
            ExecutionScope('local').restore_slots(invalid)
