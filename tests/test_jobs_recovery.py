"""Recovery must not replay an uncertain tool or exceed local capacity."""
import threading
import time
from unittest.mock import patch

import pytest
from core.jobs import AdmissionGate, CapacityExceeded, JobConflict, JobManager, JobStore


def wait_done(store, owner, job_id):
    deadline = time.monotonic()+3
    while time.monotonic()<deadline:
        result = store.get(owner,job_id)
        if result['status'] not in {'queued','running'}:
            return result
        time.sleep(.01)
    raise AssertionError('Worker did not finish')


def test_recovery_never_replays_unsafe_checkpoint(tmp_path):
    store = JobStore(tmp_path/'jobs.sqlite3')
    row,_ = store.create('alice','agent',{'query':'private'},'operation')
    store.claim('alice',row['id'])
    store.checkpoint('alice',row['id'],{'safe':False,'messages':['private']})
    restored = JobStore(tmp_path/'jobs.sqlite3')
    restored.recover_interrupted()
    assert restored.get('alice',row['id'])['status']=='interrupted'
    assert 'checkpoint' not in restored.get('alice',row['id'])
    with pytest.raises(JobConflict):
        restored.resume('alice',row['id'])
    with pytest.raises(KeyError):
        restored.get('bob',row['id'])


def test_operation_key_is_bound_to_request_and_owner(tmp_path):
    store = JobStore(tmp_path/'jobs.sqlite3')
    first,_=store.create('alice','agent',{'q':'one'},'key')
    duplicate,created=store.create('alice','agent',{'q':'one'},'key')
    assert not created and first['id']==duplicate['id']
    with pytest.raises(JobConflict):
        store.create('alice','agent',{'q':'two'},'key')
    other,_=store.create('bob','agent',{'q':'two'},'key')
    assert other['id']!=first['id']


def test_cancel_then_explicit_resume_uses_saved_boundary(tmp_path):
    store = JobStore(tmp_path/'jobs.sqlite3')
    began = threading.Event()
    calls=[]
    def run(job,event,progress,checkpoint):
        calls.append(job['checkpoint'])
        progress('read',completed=1)
        checkpoint({'safe':True,'next_item':1})
        began.set()
        if job['attempt']==1:
            assert event.wait(2)
            return {'metadata':{'status':'cancelled','completion_status':'incomplete'}}
        return {'metadata':{'status':'success'}}
    manager=JobManager(store,run,max_workers=1)
    row=manager.submit('alice','agent',{'query':'q'})
    assert began.wait(2)
    manager.cancel('alice',row['id'])
    assert wait_done(store,'alice',row['id'])['status']=='cancelled'
    manager.resume('alice',row['id'])
    assert wait_done(store,'alice',row['id'])['status']=='succeeded'
    assert calls==[None,{'safe':True,'next_item':1}]


def test_capacity_rejects_without_queuing_or_losing_slot(tmp_path):
    gate=AdmissionGate(1)
    began,release=threading.Event(),threading.Event()
    def run(*args):
        began.set()
        assert release.wait(2)
        return {}
    store=JobStore(tmp_path/'jobs.sqlite3')
    manager=JobManager(store,run,gate=gate,max_workers=1)
    row=manager.submit('alice','agent',{'q':'first'})
    assert began.wait(2)
    try:
        with pytest.raises(CapacityExceeded):
            manager.submit('alice','agent',{'q':'second'})
        assert len(store.list('alice')['jobs'])==1
        with pytest.raises(CapacityExceeded):
            gate.acquire()
    finally:
        release.set()
    wait_done(store,'alice',row['id'])
    with gate.slot():
        assert gate.active==1


def test_checkpoint_disk_failure_stops_before_following_work(tmp_path):
    store=JobStore(tmp_path/'jobs.sqlite3')
    later=[]
    def run(job,event,progress,checkpoint):
        checkpoint({'safe':False})
        later.append('external-work')
    manager=JobManager(store,run)
    with patch.object(store,'checkpoint',side_effect=OSError('synthetic disk full')):
        row=manager.submit('alice','agent',{'q':'q'})
        final=wait_done(store,'alice',row['id'])
    assert final['status']=='failed' and final['error_code']=='OSError'
    assert not later


def test_recovery_attempt_limit_matches_public_resumable_state(tmp_path):
    store=JobStore(tmp_path/'jobs.sqlite3')
    job,_=store.create('alice','agent',{'q':'fixture'})
    for attempt in range(1,11):
        assert store.claim('alice',job['id'])
        store.checkpoint('alice',job['id'],{'safe':True,'next_step':1})
        store.finish('alice',job['id'],'incomplete')
        result=store.get('alice',job['id'])
        assert result['attempt']==attempt
        assert result['resumable']==(attempt<10)
        if attempt<10:
            store.resume('alice',job['id'])
    assert result['resume_block_reason']=='recovery_budget_exhausted'
    with pytest.raises(JobConflict):
        store.resume('alice',job['id'])
