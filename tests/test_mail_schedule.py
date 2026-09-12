"""Deterministic offline scheduler recovery and configuration boundary tests."""
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.jobs import CapacityExceeded, JobStore
from core.mail_schedule import MailScheduler, MailScheduleStore


ACCOUNT = 'a' * 32


class Clock:
    def __init__(self):
        self.now = 1000

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class FakeManager:
    def __init__(self, path):
        self.store = JobStore(path)
        self.submissions = []
        self.full = False
        self.after_submit = None

    def submit(self, owner, kind, request, operation_key):
        assert kind == 'imap_sync'
        if self.full:
            raise CapacityExceeded('full')
        job, created = self.store.create(owner, kind, request, operation_key)
        if created:
            self.store.claim(owner, job['id'])
        self.submissions.append((job['id'], created))
        if self.after_submit:
            self.after_submit()
        return job

    def finish(self, owner, job_id, *, status='succeeded', remaining=0, failed=0, error=None):
        if status == 'interrupted':
            self.store.recover_interrupted()
            return
        self.store.finish(owner, job_id, status, result={'remaining_eligible':remaining,
                          'last_run':{'failed':failed}}, error_code=error)


@pytest.fixture
def env(tmp_path):
    store = MailScheduleStore(tmp_path/'schedule.sqlite3')
    manager = FakeManager(tmp_path/'jobs.sqlite3')
    clock = Clock()
    factory = Mock(side_effect=lambda owner, aid, folders, maximum:{'account_id':aid,
                   'credential_version':1, 'folders':folders, 'max_messages':maximum})
    scheduler = MailScheduler(store, lambda:manager, factory, clock=clock)
    return SimpleNamespace(store=store, manager=manager, clock=clock, factory=factory, scheduler=scheduler)


def enable(env, *, owner='alice', folders=None, **kwargs):
    return env.store.upsert(owner, ACCOUNT, enabled=True, folders=folders or ['INBOX'], **kwargs)


def current(env, owner='alice'):
    return env.store.get(owner, ACCOUNT)


def test_defaults_disabled_and_persistent_owner_isolation(env):
    assert current(env)['enabled'] is False
    assert current(env)['revision'] == 0
    env.scheduler.tick()
    env.factory.assert_not_called()
    enabled = enable(env)
    assert enabled['revision'] == 1 and enabled['next_due'] == 0
    assert current(env, 'bob')['enabled'] is False
    reopened = MailScheduleStore(env.store.path)
    assert reopened.get('alice', ACCOUNT) == enabled
    assert 'owner' not in enabled and 'pending_request' not in enabled


@pytest.mark.parametrize('kwargs', [
    {'enabled':1}, {'folders':[]}, {'folders':['INBOX']*2}, {'folders':['a']*31},
    {'folders':['line\nbreak']}, {'folders':[1]}, {'interval_seconds':59},
    {'interval_seconds':86401}, {'interval_seconds':True}, {'max_messages':0}, {'max_messages':2001},
])
def test_invalid_schedule_input_is_rejected_without_state(env, kwargs):
    values = dict(enabled=True, folders=['INBOX'])
    values.update(kwargs)
    with pytest.raises(ValueError):
        env.store.upsert('alice', ACCOUNT, **values)
    assert current(env)['revision'] == 0


def test_no_overlap_and_five_second_backfill_then_regular_interval(env):
    enable(env)
    env.scheduler.tick()
    first = current(env)['last_job_id']
    env.scheduler.tick()
    env.clock.advance(10000)
    env.scheduler.tick()
    assert len(env.manager.submissions) == 1
    env.manager.finish('alice', first, remaining=10)
    env.scheduler.tick()
    assert current(env)['next_due'] == env.clock()+5
    env.clock.advance(4)
    env.scheduler.tick()
    assert len(env.manager.submissions) == 1
    env.clock.advance(1)
    env.scheduler.tick()
    second = current(env)['last_job_id']
    assert second != first
    env.manager.finish('alice', second)
    env.scheduler.tick()
    assert current(env)['next_due'] == env.clock()+300
    assert current(env)['failure_count'] == 0


def test_disable_waits_for_active_batch_without_submitting_another(env):
    enable(env)
    env.scheduler.tick()
    first = current(env)['last_job_id']
    paused = env.store.upsert('alice', ACCOUNT, enabled=False, folders=['INBOX'])
    env.scheduler.tick()
    env.manager.finish('alice', first, remaining=100)
    env.scheduler.tick()
    env.clock.advance(100000)
    env.scheduler.tick()
    assert len(env.manager.submissions) == 1
    assert current(env)['revision'] == paused['revision']
    assert current(env)['next_due'] is None
    assert current(env)['state'] == 'disabled'


def test_config_change_preserves_new_scope_due_and_discards_old_completion(env):
    enable(env)
    env.scheduler.tick()
    first = current(env)['last_job_id']
    changed = enable(env, folders=['Archive'], interval_seconds=600, max_messages=25)
    env.scheduler.tick()
    assert len(env.manager.submissions) == 1
    env.manager.finish('alice', first, status='failed', error='authentication_failed')
    env.scheduler.tick()
    assert current(env)['revision'] == changed['revision']
    assert current(env)['failure_count'] == 0 and current(env)['next_due'] == 0
    env.scheduler.tick()
    job = env.manager.store.get('alice', current(env)['last_job_id'], private=True)
    assert job['request']['folders'] == ['Archive']
    assert job['request']['max_messages'] == 25
    assert len(env.manager.submissions) == 2


def test_capacity_backoff_reuses_operation_without_counting_failure(env):
    enable(env)
    env.manager.full = True
    env.scheduler.tick()
    assert current(env)['next_due'] == env.clock()+5
    assert current(env)['failure_count'] == 0
    env.manager.full = False
    env.scheduler.tick()
    assert env.manager.submissions == []
    env.clock.advance(5)
    env.scheduler.tick()
    assert len(env.manager.submissions) == 1
    assert env.factory.call_count == 1


@pytest.mark.parametrize('partial', [False, True])
def test_failed_or_partial_batches_backoff_then_block_without_busy_loop(env, partial):
    enable(env)
    for attempt in range(1, 9):
        env.scheduler.tick()
        job_id = current(env)['last_job_id']
        env.manager.finish('alice', job_id, status='succeeded' if partial else 'failed',
                           remaining=500, failed=1 if partial else 0, error='authentication_failed')
        env.scheduler.tick()
        record = current(env)
        assert record['failure_count'] == attempt
        if attempt < 8:
            delay = min(3600, 300 * 2**(attempt-1))
            assert record['next_due'] == env.clock()+delay
            env.clock.advance(delay)
    assert current(env)['state'] == 'blocked'
    assert current(env)['next_due'] is None
    env.clock.advance(100000)
    for _ in range(20):
        env.scheduler.tick()
    assert len(env.manager.submissions) == 8
    enable(env)
    env.scheduler.tick()
    assert len(env.manager.submissions) == 9


def test_unknown_job_error_is_sanitized_not_mail_content(env):
    enable(env)
    env.scheduler.tick()
    secret = 'private query body authorization-token'
    env.manager.finish('alice', current(env)['last_job_id'], status='failed', error=secret)
    env.scheduler.tick()
    record = current(env)
    assert record['last_error'] == 'schedule_operation_failed'
    assert secret not in json.dumps(record)
    assert secret.encode() not in env.store.path.read_bytes()


def test_factory_cannot_persist_secret_or_arbitrary_job_request(env):
    enable(env)
    secret = 'do-not-store-authorization-token'
    env.factory.side_effect = lambda *args:{'authorization_code':secret}
    env.scheduler.tick()
    assert env.manager.submissions == []
    assert current(env)['last_error'] == 'ValueError'
    assert secret.encode() not in env.store.path.read_bytes()


def test_restart_after_prepare_before_submit_uses_persisted_request(env, monkeypatch):
    enable(env)
    original = env.manager.submit
    def crash_before(*args):
        raise SystemExit('simulated process death')
    monkeypatch.setattr(env.manager, 'submit', crash_before)
    with pytest.raises(SystemExit):
        env.scheduler.tick()
    monkeypatch.setattr(env.manager, 'submit', original)
    factory = Mock(side_effect=AssertionError('persisted intent must be reused'))
    restarted = MailScheduler(MailScheduleStore(env.store.path), lambda:env.manager, factory, clock=env.clock)
    restarted.tick()
    assert len(env.manager.submissions) == 1
    factory.assert_not_called()


def test_restart_after_submit_before_tracking_finds_existing_job(env):
    enable(env)
    def crash_after():
        raise SystemExit('simulated process death')
    env.manager.after_submit = crash_after
    with pytest.raises(SystemExit):
        env.scheduler.tick()
    assert current(env)['last_job_id'] is None
    env.manager.after_submit = None
    restarted = MailScheduler(MailScheduleStore(env.store.path), lambda:env.manager, env.factory, clock=env.clock)
    restarted.tick()
    assert current(env)['last_job_id'] == env.manager.submissions[0][0]
    assert len(env.manager.submissions) == 1


def test_change_after_ambiguous_submit_tracks_old_job_without_replaying_old_scope(env):
    enable(env)
    def crash_after():
        raise SystemExit('crash')
    env.manager.after_submit = crash_after
    with pytest.raises(SystemExit):
        env.scheduler.tick()
    old_id = env.manager.submissions[0][0]
    enable(env, folders=['Archive'])
    env.manager.after_submit = None
    env.scheduler.tick()
    assert len(env.manager.submissions) == 1
    env.manager.finish('alice', old_id, remaining=200)
    env.scheduler.tick()
    env.scheduler.tick()
    new = env.manager.store.get('alice', current(env)['last_job_id'], private=True)
    assert new['request']['folders'] == ['Archive']


def test_disable_after_prepare_drops_unsubmitted_old_scope(env, monkeypatch):
    enable(env)
    def crash_before(*args):
        raise SystemExit('crash')
    monkeypatch.setattr(env.manager, 'submit', crash_before)
    with pytest.raises(SystemExit):
        env.scheduler.tick()
    env.store.upsert('alice', ACCOUNT, enabled=False, folders=['INBOX'])
    env.scheduler.tick()
    assert current(env)['state'] == 'disabled'
    assert env.manager.submissions == []


def test_interrupted_imap_job_schedules_fresh_readonly_batch_only(env):
    # Unrelated interrupted agent work must never be resumed by this timer.
    agent, _ = env.manager.store.create('alice', 'agent', {'query':'synthetic write task'})
    env.manager.store.claim('alice', agent['id'])
    env.manager.store.recover_interrupted()
    enable(env)
    env.scheduler.tick()
    env.manager.finish('alice', current(env)['last_job_id'], status='interrupted')
    env.scheduler.tick()
    env.clock.advance(300)
    env.scheduler.tick()
    assert len(env.manager.submissions) == 2
    assert env.manager.store.get('alice', agent['id'])['status'] == 'interrupted'


def test_manager_exception_backoff_does_not_consume_failures_each_tick(env):
    enable(env)
    calls = Mock(side_effect=RuntimeError('authorization-code-and-body-must-be-hidden'))
    env.scheduler.manager_factory = calls
    env.scheduler.tick()
    for _ in range(10):
        env.scheduler.tick()
    assert calls.call_count == 1
    assert current(env)['failure_count'] == 1
    assert current(env)['last_error'] == 'RuntimeError'


def test_two_scheduler_instances_share_durable_intent_without_overlap(env):
    enable(env)
    other = MailScheduler(MailScheduleStore(env.store.path), lambda:env.manager, env.factory, clock=env.clock)
    threads = [threading.Thread(target=scheduler.tick) for scheduler in [env.scheduler, other]]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive()
    assert len(env.manager.submissions) == 1


def test_stop_prevents_future_tick_submission_and_thread_survives_transient_error(env, monkeypatch):
    enable(env)
    env.scheduler.stop()
    env.scheduler.tick()
    assert env.manager.submissions == []
    observed = threading.Event()
    calls = []
    def occasionally_failing_tick():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError('transient database failure')
        observed.set()
    monkeypatch.setattr(env.scheduler, 'tick', occasionally_failing_tick)
    env.scheduler.poll_seconds = .01
    env.scheduler.start()
    assert observed.wait(2)
    assert env.scheduler.stop(timeout=2)
    assert len(calls) >= 2


def test_pending_intent_keeps_credential_version_across_capacity_retry(env):
    enable(env)
    env.manager.full = True
    env.scheduler.tick()
    env.factory.side_effect = lambda owner, aid, folders, maximum:{'account_id':aid,
        'credential_version':2, 'folders':folders, 'max_messages':maximum}
    env.manager.full = False
    env.clock.advance(5)
    env.scheduler.tick()
    job = env.manager.store.get('alice', current(env)['last_job_id'], private=True)
    assert job['request']['credential_version'] == 1
    assert env.factory.call_count == 1
    env.manager.finish('alice', job['id'], status='failed', error='credential_binding_changed')
    env.scheduler.tick()
    env.clock.advance(300)
    env.scheduler.tick()
    retry = env.manager.store.get('alice', current(env)['last_job_id'], private=True)
    assert retry['request']['credential_version'] == 2
    assert retry['operation_key'] != job['operation_key']


def test_scope_change_before_submission_discards_pending_request(env):
    enable(env)
    env.manager.full = True
    env.scheduler.tick()
    enable(env, folders=['Archive'])
    env.manager.full = False
    env.scheduler.tick()
    assert env.manager.submissions == []
    env.scheduler.tick()
    job = env.manager.store.get('alice', current(env)['last_job_id'], private=True)
    assert job['request']['folders'] == ['Archive']


def test_restart_recovers_interrupted_schedule_and_leaves_agent_untouched(env):
    enable(env)
    env.scheduler.tick()
    first = current(env)['last_job_id']
    agent, _ = env.manager.store.create('alice', 'agent', {'query':'synthetic write action'})
    env.manager.store.claim('alice', agent['id'])
    env.manager.store.recover_interrupted()
    restarted = MailScheduler(MailScheduleStore(env.store.path), lambda:env.manager, env.factory, clock=env.clock)
    restarted.tick()
    assert current(env)['failure_count'] == 1
    env.clock.advance(300)
    restarted.tick()
    assert current(env)['last_job_id'] != first
    assert env.manager.store.get('alice', agent['id'])['status'] == 'interrupted'


def test_stop_during_request_preparation_prevents_submission(env):
    enable(env)
    entered, release = threading.Event(), threading.Event()
    def prepare(owner, aid, folders, maximum):
        entered.set()
        assert release.wait(2)
        return {'account_id':aid, 'credential_version':1, 'folders':folders, 'max_messages':maximum}
    env.factory.side_effect = prepare
    worker = threading.Thread(target=env.scheduler.tick)
    worker.start()
    assert entered.wait(2)
    env.scheduler.stop()
    release.set()
    worker.join(2)
    assert not worker.is_alive()
    assert env.manager.submissions == []
