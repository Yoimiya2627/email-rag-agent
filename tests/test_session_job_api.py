import threading
import time
from unittest.mock import patch

from fastapi.testclient import TestClient
from api.readiness import Readiness
from api.sessions import SessionStore
from core.jobs import JobManager,JobStore
from core.model_outcomes import ModelOutputError
from models.schemas import AgentResponse,IntentType
import api.main as api


def test_partial_and_error_transcripts_survive_restart_without_becoming_context(tmp_path):
    path=tmp_path/'sessions.sqlite3'
    sessions=SessionStore(path=path)
    with patch.object(api,'sessions',sessions),patch.object(api,'route',return_value=AgentResponse(
        answer='half',metadata={'status':'incomplete','completion_status':'incomplete'})):
        result=TestClient(api.app).post('/chat',json={'query':'keep this question','session_id':'s'})
        assert result.status_code==200
    def partial(*args,**kwargs):
        yield 'stream half'
        raise ModelOutputError('stream half',finish_reason='length')
    with patch.object(api,'sessions',sessions),patch('agents.coordinator.classify_intent',return_value=IntentType.RETRIEVE), \
         patch('core.pipeline.retrieve',return_value=[]),patch('core.generator.stream_generate',side_effect=partial):
        TestClient(api.app).post('/chat/stream',json={'query':'second','session_id':'s'})
    restored=SessionStore(path=path)
    with patch.object(api,'sessions',restored):
        page=TestClient(api.app).get('/chat/history',params={'session_id':'s','limit':1}).json()
        assert page['has_more'] and page['turns'][0]['answer']=='half'
        next_page=TestClient(api.app).get('/chat/history',params={'session_id':'s','after':page['next_after']}).json()
        assert next_page['turns'][0]['answer']=='stream half'
    with restored.turn('local','s') as memory:
        assert not memory.to_messages()
    assert not restored.history('other','s')['turns']


def test_explicit_fact_source_and_conflict_are_enforced_at_api(tmp_path):
    sessions=SessionStore(path=tmp_path/'sessions.sqlite3')
    with sessions.turn('local','s') as memory:
        turn=memory.append_turn('Budget is 500','Understood')
    client=TestClient(api.app)
    with patch.object(api,'sessions',sessions):
        payload={'session_id':'s','key':'budget','value':'500','source_turn_id':turn,'expected_version':0}
        created=client.post('/chat/facts',json=payload)
        assert created.status_code==200 and created.json()['version']==1
        assert client.post('/chat/facts',json=payload).status_code==409
        assert client.post('/chat/facts',json={**payload,'session_id':'other'}).status_code==404
        captured=[]
        def runner(*args,**kwargs):
            from agents.runtime import current_run
            captured.append(current_run().task_context)
            return AgentResponse(answer='ok')
        with patch.object(api,'route',runner):
            assert client.post('/chat',json={'query':'continue','session_id':'s'}).status_code==200
        assert '500' in captured[0]['text'] and captured[0]['contains_execution_authority'] is False


def test_background_request_retry_reuses_job_even_while_busy(tmp_path):
    started,release=threading.Event(),threading.Event()
    calls=[]
    def runner(job,event,progress,checkpoint):
        calls.append(job)
        checkpoint({'safe':True,'kind':'agent','next_round':0})
        started.set()
        assert release.wait(2)
        return {}
    manager=JobManager(JobStore(tmp_path/'jobs.sqlite3'),runner,max_workers=1)
    client=TestClient(api.app)
    try:
        with patch.object(api,'_jobs',manager):
            body={'query':'q','operation_key':'logical-one'}
            first=client.post('/jobs/agent',json=body)
            assert started.wait(2)
            retry=client.post('/jobs/agent',json=body)
            assert retry.status_code==202 and retry.json()['id']==first.json()['id']
            assert client.get('/jobs/operations/logical-one').json()['id']==first.json()['id']
            assert client.post('/jobs/agent',json={**body,'query':'changed'}).status_code==409
            assert 'request' not in first.json() and 'checkpoint' not in first.json()
            assert len(calls)==1
    finally:
        release.set()


def test_warmup_is_singleflight_and_readiness_is_not_liveness():
    state=Readiness()
    started,release=threading.Event(),threading.Event()
    calls=[]
    def warm():
        calls.append(1);started.set()
        assert release.wait(2)
        return {'embedding':'initialized'}
    with patch.object(api,'readiness',state),patch.object(api,'_warm_components',warm), \
         patch.object(api,'verify_collection_readiness',return_value={'chunk_count':3}):
        client=TestClient(api.app)
        assert client.get('/health').status_code==200
        assert client.get('/ready').status_code==503
        try:
            assert client.post('/warmup').status_code==202
            assert started.wait(2)
            assert client.post('/warmup').status_code==202
            assert len(calls)==1
            assert client.get('/ready').json()['provider_network_verified'] is False
        finally:
            release.set()


def test_cancel_during_completed_transcript_commit_does_not_replay(tmp_path):
    sessions=SessionStore(path=tmp_path/'sessions.sqlite3')
    manager=JobManager(JobStore(tmp_path/'jobs.sqlite3'),api._run_background_job,max_workers=1)
    reached,release=threading.Event(),threading.Event()
    calls=[]
    original=sessions.repository.append_turns
    def persist(*args,**kwargs):
        reached.set()
        assert release.wait(3)
        return original(*args,**kwargs)
    def fake_loop(*args,**kwargs):
        from agents.runtime import current_run
        calls.append(1)
        current_run().checkpoint({'safe':True,'kind':'agent','next_round':0})
        return AgentResponse(answer='complete synthetic answer',metadata={'status':'success','completion_status':'complete'})
    with patch.object(api,'sessions',sessions),patch.object(api,'_jobs',manager), \
         patch.object(sessions.repository,'append_turns',persist),patch('agents.agent_loop.run_agent_loop',fake_loop):
        client=TestClient(api.app)
        job=client.post('/jobs/agent',json={'query':'fixture','session_id':'s','operation_key':'fixture-key'}).json()
        try:
            assert reached.wait(3)
            client.post('/jobs/'+job['id']+'/cancel')
        finally:
            release.set()
        assert manager.wait_idle(3)
        completed=client.get('/jobs/'+job['id']).json()
        assert completed['status']=='succeeded' and not completed['resumable']
        assert client.post('/jobs/'+job['id']+'/resume').status_code==409
        assert len(calls)==1 and len(sessions.history('local','s')['turns'])==1


def test_sync_operation_key_retry_without_session_reuses_approval(tmp_path):
    from agents.tools import send_email
    from agents.approvals import ApprovalStore
    sessions=SessionStore(path=tmp_path/'sessions.sqlite3')
    approvals=tmp_path/'approvals.sqlite3'
    def fake_loop(*args,**kwargs):
        action=send_email(['fixture@example.test'],'Synthetic','Fixture body','Fixture rationale')
        return AgentResponse(answer='Waiting',metadata={'status':'approval_required','approval_id':action['approval_id']})
    with patch.object(api,'sessions',sessions),patch.object(api.cfg,'APPROVAL_STORE_PATH',str(approvals)),patch('agents.agent_loop.run_agent_loop',fake_loop):
        client=TestClient(api.app)
        payload={'query':'Create fixture approval','operation_key':'same-logical-key'}
        first=client.post('/chat/agent',json=payload)
        second=client.post('/chat/agent',json=payload)
        assert first.status_code==second.status_code==200
        assert first.json()['metadata']['session_id']==second.json()['metadata']['session_id']
        assert len(ApprovalStore(approvals).list())==1


def test_sessions_api_exposes_all_pages_and_owner_isolation(tmp_path):
    sessions=SessionStore(path=tmp_path/'sessions.sqlite3')
    for index in range(115):
        sessions.record_result('local',f's{index}','query','answer')
    sessions.record_result('other','private','query','answer')
    with patch.object(api,'sessions',sessions):
        client=TestClient(api.app)
        ids=[];offset=0
        while True:
            page=client.get('/chat/sessions',params={'limit':50,'offset':offset}).json()
            ids.extend(row['session_id'] for row in page['sessions'])
            offset=page['next_offset']
            if offset is None:
                break
        assert len(ids)==len(set(ids))==115 and 'private' not in ids
        assert client.get('/chat/sessions',params={'offset':-1}).status_code==422


def test_only_actual_visible_ranges_are_saved_and_reread_via_api(tmp_path):
    from agents.runtime import current_run
    from core.evidence import evidence_reference
    from core.evidence_pages import read_email_page,reread_evidence
    from tests.test_evidence_pages import email
    from models.schemas import SearchResult
    doc=email('prefix evidence tail')
    actual=read_email_page('e1',chunk_id='c0',start=7,limit=8,loader=lambda _:doc)['chunks'][0]
    ref=evidence_reference(actual)
    sessions=SessionStore(path=tmp_path/'sessions.sqlite3')
    def runner(*args,**kwargs):
        current_run().generation_evidence_refs.append(ref)
        return AgentResponse(answer='Supported [e1#c0]',sources=[SearchResult(**doc['chunks'][0])])
    with patch.object(api,'sessions',sessions),patch.object(api,'route',runner):
        client=TestClient(api.app)
        result=client.post('/chat',json={'query':'fixture','session_id':'s'}).json()
        assert result['metadata']['model_visible_evidence']==[ref]
        saved=client.get('/chat/evidence',params={'session_id':'s'}).json()['evidence'][0]
        assert saved['visible_start']==7 and saved['visible_end']==15
        payload={name:saved[name] for name in ('email_id','chunk_id','source_version','source_sha256','visible_start','visible_end','visible_hash')}
        with patch('api.evidence_routes.reread_evidence',side_effect=lambda value,**kw:reread_evidence(value,loader=lambda _:doc,**kw)):
            page=client.post('/evidence/reread',json=payload)
            assert page.status_code==200 and page.json()['body']=='evidence'
            payload['source_version']='old-version'
            assert client.post('/evidence/reread',json=payload).status_code==409
            assert client.post('/evidence/reread',json={**payload,'unexpected':'field'}).status_code==422
