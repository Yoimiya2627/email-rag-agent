import json
import sqlite3
import time

import pytest

from agents.approvals import ApprovalStore
from agents.log_storage import append_jsonl,read_jsonl_tail
from core.jobs import JobStore,JobConflict
from core.session_repository import SessionRepository
from scripts.state_maintenance import backup_states,restore_states,retain_private_state


def create(store,key,owner='local'):
    return store.create('send_email',{'to':['synthetic@example.test'],'subject':key,'body':'PRIVATE'},
                        owner_id=owner,request_id=key)


def test_approval_pages_bounded_and_bound_to_owner_filters(tmp_path):
    store=ApprovalStore(tmp_path/'approvals.sqlite3')
    for number in range(5): create(store,str(number))
    create(store,'other','another')
    first=store.list_page(limit=2)
    assert len(first['items'])==2 and 'payload' not in first['items'][0]
    second=store.list_page(limit=2,cursor=first['next_cursor'])
    third=store.list_page(limit=2,cursor=second['next_cursor'],include_payload=True)
    assert len({r['approval_id'] for page in (first,second,third) for r in page['items']})==5
    assert third['next_cursor'] is None and third['items'][0]['payload']['body']=='PRIVATE'
    with pytest.raises(ValueError): store.list_page(owner_id='another',cursor=first['next_cursor'])
    with pytest.raises(ValueError): store.list_page(limit=201)


def test_redaction_preserves_dedup_tombstone_and_uncertain_evidence(tmp_path):
    store=ApprovalStore(tmp_path/'approvals.sqlite3')
    done=create(store,'done')
    store.approve(done['approval_id'])
    unknown=create(store,'unknown')
    with pytest.raises(RuntimeError):
        store.approve(unknown['approval_id'],executor=lambda _:(_ for _ in ()).throw(RuntimeError('fake')))
    with sqlite3.connect(store.db_path) as db: db.execute('UPDATE approvals SET updated_at=1')
    preview=store.retain(before=time.time()-1)
    assert preview['eligible']==1 and preview['redacted']==0 and preview['uncertain_preserved']==1
    assert store.get(done['approval_id'])['payload']['body']=='PRIVATE'
    changed=store.retain(before=time.time()-1,apply=True)
    assert changed['redacted']==1 and changed['tombstones_deleted']==0
    assert store.get(done['approval_id'])['payload']=={}
    duplicate=create(store,'done')
    assert duplicate['approval_id']==done['approval_id'] and duplicate['status']=='approved'
    assert store.get(unknown['approval_id'])['payload']['body']=='PRIVATE'
    with pytest.raises(ValueError): store.create('send_email',{'body':'changed'},request_id='done')


def test_rotated_logs_have_bounded_tail_and_skip_malformed_rows(tmp_path):
    path=tmp_path/'events.jsonl'
    for index in range(40): append_jsonl(path,{'id':index,'tool':'test'},max_bytes=256,backups=2)
    assert len(list(tmp_path.iterdir()))==3
    assert all(p.stat().st_size<=256 for p in tmp_path.iterdir())
    assert read_jsonl_tail(path,limit=2)[-1]['id']==39
    with path.open('ab') as target: target.write(b'not-json\n')
    assert read_jsonl_tail(path,limit=1,max_bytes=80)[0]['id']==39
    assert read_jsonl_tail(path,limit=3,predicate=lambda r:r.get('tool')=='missing')==[]
    with pytest.raises(ValueError): read_jsonl_tail(path,limit=0)


def test_tail_preserves_complete_row_exactly_at_byte_boundary(tmp_path):
    path=tmp_path/'boundary.jsonl'
    first,last=b'{"old":1}\n',b'{"latest":2}\n'
    path.write_bytes(first+last)
    assert read_jsonl_tail(path,max_bytes=len(last))==[{'latest':2}]
    assert read_jsonl_tail(path,max_bytes=len(last)-1)==[]


def state_fixture(tmp_path):
    approvals=ApprovalStore(tmp_path/'approvals.sqlite3')
    pending=create(approvals,'pending')
    approved=create(approvals,'approved')
    approvals.approve(approved['approval_id'])
    jobs=JobStore(tmp_path/'jobs.sqlite3')
    job,_=jobs.create('local','agent',{'query':'PRIVATE'},'job-key')
    with sqlite3.connect(jobs.path) as db:
        db.execute("UPDATE jobs SET status='running',checkpoint=?",(json.dumps({'safe':True,'private':'PRIVATE'}),))
    sessions=SessionRepository(tmp_path/'sessions.sqlite3')
    sessions.append_turns('local','session',[{'turn_id':'turn','query':'PRIVATE','answer':'PRIVATE'}],expected_revision=0)
    return approvals,pending,approved,jobs,job,sessions


def test_backup_restore_quarantines_pending_and_disables_resume(tmp_path):
    approvals,pending,approved,jobs,job,sessions=state_fixture(tmp_path)
    sources={'approvals':approvals.db_path,'jobs':jobs.path,'sessions':sessions.path}
    with pytest.raises(ValueError): backup_states(sources,tmp_path/'bad')
    backup_states(sources,tmp_path/'backup',service_stopped=True)
    report=restore_states(tmp_path/'backup',tmp_path/'restored',service_stopped=True)
    assert report['quarantined_approvals']==1 and report['disabled_job_checkpoints']==1
    restored=ApprovalStore(tmp_path/'restored/approvals.sqlite3')
    calls=[]
    with pytest.raises(ValueError): restored.approve(pending['approval_id'],executor=lambda x:calls.append(x))
    restored.approve(approved['approval_id'],executor=lambda x:calls.append(x))
    assert calls==[] and restored.get(pending['approval_id'])['status']=='unknown'
    restored_jobs=JobStore(tmp_path/'restored/jobs.sqlite3')
    with pytest.raises(JobConflict): restored_jobs.resume('local',job['id'])
    assert approvals.get(pending['approval_id'])['status']=='pending'
    with pytest.raises(FileExistsError): restore_states(tmp_path/'backup',tmp_path/'restored',service_stopped=True)


def test_restore_checksum_failure_never_creates_destination(tmp_path):
    store=ApprovalStore(tmp_path/'approvals.sqlite3')
    create(store,'pending')
    backup_states({'approvals':store.db_path},tmp_path/'backup',service_stopped=True)
    with (tmp_path/'backup/approvals.sqlite3').open('ab') as target: target.write(b'tampered')
    with pytest.raises(ValueError,match='checksum'):
        restore_states(tmp_path/'backup',tmp_path/'restored',service_stopped=True)
    assert not (tmp_path/'restored').exists()


def test_offline_job_and_session_retention_is_explicit_and_cascades(tmp_path):
    approvals,pending,approved,jobs,job,sessions=state_fixture(tmp_path)
    with sqlite3.connect(jobs.path) as db: db.execute("UPDATE jobs SET status='failed',updated=1")
    with sqlite3.connect(sessions.path) as db: db.execute("UPDATE sessions SET updated_at='2000-01-01T00:00:00+00:00'")
    before=time.time()-1
    assert retain_private_state(jobs.path,'jobs',owner_id='local',before=before,service_stopped=True)['eligible']==1
    assert jobs.get('local',job['id'],private=True)['request']['query']=='PRIVATE'
    retain_private_state(jobs.path,'jobs',owner_id='local',before=before,apply=True,service_stopped=True)
    assert jobs.get('local',job['id'],private=True)['request']=={}
    assert jobs.create('local','agent',{'query':'PRIVATE'},'job-key')[0]['id']==job['id']
    assert retain_private_state(sessions.path,'sessions',owner_id='local',before=before,apply=True,service_stopped=True)['changed']==1
    with sqlite3.connect(sessions.path) as db:
        assert db.execute('SELECT COUNT(*) FROM session_turns').fetchone()[0]==0
