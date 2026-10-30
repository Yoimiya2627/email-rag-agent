import json
import pytest
from core.session_repository import SessionRepository
from tests.test_ctx_sessions import append


def good(payload, budget=None):
    turn=payload['turns'][0]
    return {'schema_version':1,'sections':{'constraints':[{'text':turn['query'],'source_quotes':[
        {'turn_id':turn['turn_id'],'field':'query','start':0,'end':len(turn['query']),'text':turn['query']}]}]}}


def test_summary_publish_cache_incremental_and_provenance(tmp_path):
    repo=SessionRepository(tmp_path/'s.db');append(repo)
    budget=object()
    def generate(*,payload,budget):
        # A second connection can commit during the model call: no write lock.
        with repo._connect() as db:db.execute('CREATE TABLE IF NOT EXISTS lock_probe(x)')
        return json.dumps(good(payload,budget))
    result=repo.generate_summary('o','s',generate=generate,budget=budget,min_turns=1)
    assert result['status']=='generated'
    summary=result['summary']
    assert summary['sections']['constraints'][0]['source_quotes'][0]['source_type']=='user_utterance'
    assert summary['input_sha256'] and summary['model_id']=='injected'
    append(repo,'第二轮')
    assert repo.get_semantic_summary('o','s') is not None
    assert repo.generate_summary('o','s',generate=lambda **kw:pytest.fail('should reuse'),min_turns=2)['status']=='cached'
    append(repo,'第三轮')
    assert repo.generate_summary('o','s',generate=good,min_turns=2)['status']=='generated'


def test_summary_failures_persistent_attempt_cap_and_bad_sources(tmp_path):
    repo=SessionRepository(tmp_path/'s.db');append(repo)
    for _ in range(2):
        assert repo.generate_summary('o','s',generate=lambda **kw:'invalid JSON',min_turns=1)['status']=='failed'
    assert repo.generate_summary('o','s',generate=good,min_turns=1)['status']=='attempt_limit'
    assert len(repo.summary_attempts('o','s'))==2
    assert repo.get_semantic_summary('o','s') is None
    append(repo)
    def forged(*,payload,budget):
        result=good(payload);result['sections']['constraints'][0]['source_quotes'][0]['turn_id']='other-owner'
        return result
    assert repo.generate_summary('o','s',generate=forged,min_turns=1)['status']=='failed'


@pytest.mark.parametrize('mutation',['append','delete','correct'])
def test_summary_cas_rejects_concurrent_mutations(tmp_path,mutation):
    repo=SessionRepository(tmp_path/'s.db');append(repo)
    def mutate(*,payload,budget):
        result=good(payload)
        if mutation=='append':append(repo)
        elif mutation=='delete':repo.delete('o','s')
        else:repo.record_current_request('o','s',text='预算改成3000',request_id='r')
        return result
    assert repo.generate_summary('o','s',generate=mutate,min_turns=1)['status']=='stale'
    assert repo.get_semantic_summary('o','s') is None


def test_summary_source_tamper_and_task_isolation(tmp_path):
    repo=SessionRepository(tmp_path/'s.db');tid=append(repo)
    repo.generate_summary('o','s',generate=good,min_turns=1)
    with repo._connect() as db:db.execute('UPDATE session_turns SET query=? WHERE turn_id=?',('changed',tid))
    assert repo.get_semantic_summary('o','s') is None
    repo.update_task('o','s',task_id='other',goal='new',source_turn_id=tid,expected_revision=repo.revision('o','s'),explicit_user=True)
    append(repo,'other task')
    result=repo.generate_summary('o','s',generate=good,min_turns=1)
    assert len(result['summary']['sources'])==1
    assert result['summary']['sources'][0]['turn_id']!=tid


def test_failed_summary_input_hash_cap_survives_request_revision(tmp_path):
    repo=SessionRepository(tmp_path/'s.db');append(repo)
    assert repo.generate_summary('o','s',generate=lambda **kw:'bad',min_turns=1,max_attempts=1)['status']=='failed'
    repo.record_current_request('o','s',text='继续',request_id='new-revision')
    assert repo.generate_summary('o','s',generate=good,min_turns=1,max_attempts=1)['status']=='attempt_limit'


def test_summary_minimal_snapshot_preserves_uncertainty_and_bounds_json(tmp_path):
    repo=SessionRepository(tmp_path/'s.db')
    metadata={'status':'partial','completion_status':'incomplete','coverage':{'attachment_inventory_status':'unknown',
        'unread_attachments':3,'partial':True,'scope':'indexed_text_only'},
        'model_usage':{'old_payload':'DO_NOT_REINJECT'*20000},'session_context':{'summary':'DO_NOT_REINJECT'*20000},
        'model_visible_evidence':[{'coverage':{'attachment_inventory_status':'available','unread_attachments':2,'truncated':True}}]}
    repo.append_turns('o','s',[{'turn_id':'source','query':'预算5000元','answer':'尚未读取附件','metadata':metadata}],expected_revision=0)
    captured=[]
    def generate(*,payload,budget):
        captured.append(payload)
        return good(payload)
    result=repo.generate_summary('o','s',generate=generate,min_turns=1,max_chars=2500)
    assert result['status']=='generated'
    raw=json.dumps(captured[0],ensure_ascii=False)
    assert len(raw)<=2500 and 'DO_NOT_REINJECT' not in raw and 'model_usage' not in raw
    state=captured[0]['turns'][0]['metadata']
    assert state['status']=='partial' and state['completion_status']=='incomplete'
    assert state['coverage']['unread_attachments']==3 and state['coverage']['attachment_inventory_status']=='unknown'
    assert state['source_coverages'][0]['truncated'] is True
    assert result['summary']['source_statuses'][0]['coverage']['partial'] is True
    # Full JSON envelope counts even when original query+answer are tiny.
    assert repo.generate_summary('o','s',generate=lambda **kw:pytest.fail('oversized envelope'),min_turns=1,max_chars=100,force=True)['status']=='below_threshold'


def test_twenty_real_api_turns_then_summary_and_candidates_fit_default_budget(tmp_path,monkeypatch):
    from types import SimpleNamespace as NS
    from unittest.mock import patch
    from fastapi.testclient import TestClient
    import api.main as api
    import config.settings as cfg
    from api.sessions import SessionStore
    from models.schemas import AgentResponse
    store=SessionStore(path=tmp_path/'api.db',max_history_turns=30)
    monkeypatch.setattr(api,'sessions',store)
    monkeypatch.setattr(cfg,'TOOL_RESULT_STORE_PATH',str(tmp_path/'results.db'))
    monkeypatch.setattr(cfg,'ENABLE_CONTEXT_OPTIMIZATION',True)
    monkeypatch.setattr(cfg,'ENABLE_CONTEXT_SUMMARY',False)
    monkeypatch.setattr(cfg,'ENABLE_CONTEXT_CANDIDATES',False)
    monkeypatch.setattr(cfg,'CONTEXT_SUMMARY_MIN_TURNS',20)
    monkeypatch.setattr(cfg,'MODEL_CONTEXT_TOKENS',32000)
    client=TestClient(api.app)
    def route(*args,**kwargs):
        return AgentResponse(answer='已核对可见正文，附件未读。',metadata={'status':'partial','completion_status':'incomplete',
            'coverage':{'attachment_inventory_status':'unknown','unread_attachments':2,'partial':True},
            'diagnostic_raw':'EXCLUDED_RAW_DEBUG'*5000})
    with patch.object(api,'route',route):
        for i in range(20):
            response=client.post('/chat',json={'session_id':'s','query':f'第{i}轮核对订单'})
            assert response.status_code==200,response.text
    old=store.history('local','s')['turns']
    assert len(old)==20 and all('session_context' in t['metadata'] and 'model_usage' in t['metadata'] for t in old)
    monkeypatch.setattr(cfg,'ENABLE_CONTEXT_SUMMARY',True)
    monkeypatch.setattr(cfg,'ENABLE_CONTEXT_CANDIDATES',True)
    calls=[]
    def create(**kwargs):
        payload=json.loads(kwargs['messages'][-1]['content']);calls.append(payload)
        assert len(payload['turns'])==20
        assert 'EXCLUDED_RAW_DEBUG' not in json.dumps(payload)
        turn=payload['turns'][0]
        quote={'turn_id':turn['turn_id'],'start':0,'end':len(turn['query']),'text':turn['query']}
        if 'sections' in payload:
            assert all(t['metadata']['coverage']['attachment_inventory_status']=='unknown' for t in payload['turns'])
            assert all(t['metadata']['coverage']['unread_attachments']==2 and t['metadata']['status']=='partial' for t in payload['turns'])
            value={'schema_version':1,'sections':{'goals':[{'text':'核对订单','source_quotes':[{**quote,'field':'query'}]}]}}
        else:
            assert all(set(t)=={'seq','turn_id','query'} for t in payload['turns'])
            value={'schema_version':1,'candidates':[]}
        return NS(choices=[NS(message=NS(content=json.dumps(value,ensure_ascii=False)),finish_reason='stop')],usage=NS(prompt_tokens=100,completion_tokens=30,total_tokens=130))
    provider=NS(chat=NS(completions=NS(create=create)))
    with patch('core.model_clients.get_model_client',return_value=provider),patch.object(api,'route',return_value=AgentResponse(answer='继续核对')):
        response=client.post('/chat',json={'session_id':'s','query':'继续核对'})
    assert response.status_code==200,response.text
    assert len(calls)==2
    assert store.get_semantic_summary('local','s') is not None
    assert response.json()['metadata']['model_usage']['call_count']==2



def test_summary_reference_without_stored_coverage_stays_unknown(tmp_path):
    repo=SessionRepository(tmp_path/'s.db')
    repo.append_turns('o','s',[{'turn_id':'source','query':'核对邮件','answer':'历史回答',
        'metadata':{'model_visible_evidence':[{'email_id':'e','chunk_id':'c'}]}}],expected_revision=0)
    captured=[]
    def generate(*,payload,budget):
        captured.append(payload);return good(payload)
    assert repo.generate_summary('o','s',generate=generate,min_turns=1)['status']=='generated'
    flags=captured[0]['turns'][0]['metadata']['source_coverages'][0]
    assert flags['attachment_inventory_status']=='unknown' and flags['unread_attachments'] is None
    assert flags['scope']=='historical_reference_only'
