"""Clock recovery failures must remain actionable at the job/API boundary."""
import pytest

from core.jobs import JobConflict, JobManager, JobStore
from core.tool_results import ToolResultUnavailable
from tests.test_jobs_recovery import wait_done


def test_legacy_clock_job_failure_keeps_evidence_and_blocks_repeat_resume(tmp_path):
    store = JobStore(tmp_path / 'jobs.sqlite3')
    saved = {'kind': 'agent', 'safe': True, 'messages': ['completed evidence']}
    def runner(job, event, progress, checkpoint):
        checkpoint(saved)
        raise ToolResultUnavailable('checkpoint_reference_requires_restart')
    manager = JobManager(store, runner, max_workers=1)
    job = manager.submit('owner', 'agent', {'query': 'continue'})
    final = wait_done(store, 'owner', job['id'])
    assert final['status'] == 'failed'
    assert final['error_code'] == 'checkpoint_reference_requires_restart'
    assert final['resume_block_reason'] == 'checkpoint_reference_requires_restart'
    assert final['resumable'] is False
    assert store.get('owner', job['id'], private=True)['checkpoint'] == saved
    with pytest.raises(JobConflict, match='start a new task'):
        manager.resume('owner', job['id'])
    assert store.get('owner', job['id'])['attempt'] == final['attempt']


def test_unknown_job_error_text_is_not_persisted(tmp_path):
    store = JobStore(tmp_path / 'jobs.sqlite3')
    def runner(*args):
        raise ToolResultUnavailable('private checkpoint payload')
    manager = JobManager(store, runner, max_workers=1)
    job = manager.submit('owner', 'agent', {'query': 'continue'})
    final = wait_done(store, 'owner', job['id'])
    assert final['error_code'] == 'ToolResultUnavailable'
    assert 'private' not in str(final)


def test_api_clock_recovery_message_is_specific_and_safe():
    from api.main import _safe_error
    error = _safe_error(ToolResultUnavailable('checkpoint_reference_requires_restart'))
    assert error.status_code == 409 and 'start a new task' in error.detail
    conflict = _safe_error(ToolResultUnavailable('checkpoint_reference_conflict'))
    assert conflict.status_code == 409 and 'reference' in conflict.detail
    unknown = _safe_error(ToolResultUnavailable('private checkpoint payload'))
    assert unknown.status_code == 409 and 'private' not in unknown.detail


def test_api_unsupported_mcp_clock_reports_required_upgrade():
    from api.main import _safe_error
    from agents.mcp_adapter import MCPReferenceClockError
    error = _safe_error(MCPReferenceClockError())
    assert error.status_code == 409 and 'upgrade the server' in error.detail
