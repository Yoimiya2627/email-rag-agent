import pytest
from core.session_repository import SessionRepository
from api.sessions import SessionStore
from tests.test_ctx_sessions import append


def valid(*,payload,budget=None):
    t=payload['turns'][-1]
    return {'schema_version':1,'candidates':[{'key':'budget','value':'5000美元','kind':'constraint',
        'source_quote':{'turn_id':t['turn_id'],'start':0,'end':len(t['query']),'text':t['query']}}]}


def test_candidates_provenance_explicit_confirmation_and_reuse(tmp_path):
    repo=SessionRepository(tmp_path/'s.db');tid=append(repo)
    budget=object()
    def call(*,payload,budget):
        with repo._connect() as db:db.execute('CREATE TABLE IF NOT EXISTS probe(x)')
        return valid(payload=payload,budget=budget)
    result=repo.extract_candidates('o','s',generate=call,budget=budget)
    assert result['status']=='generated' and result['candidates'][0]['status']=='candidate'
    assert repo.task_facts('o','s')==[]
    assert repo.extract_candidates('o','s',generate=lambda **kw:pytest.fail('must reuse'))['status']=='cached'
    candidate=result['candidates'][0]
    with pytest.raises(PermissionError):
        repo.confirm_task_fact('o','s','budget',expected_version=candidate['version'],source_turn_id=tid)
    repo.confirm_task_fact('o','s','budget',expected_version=candidate['version'],source_turn_id=tid,explicit_user=True)
    assert repo.task_facts('o','s')[0]['status']=='active'


@pytest.mark.parametrize('bad',['json','scope','assistant','range','permission'])
def test_invalid_candidates_atomic_and_capped(tmp_path,bad):
    repo=SessionRepository(tmp_path/'s.db');append(repo)
    def call(*,payload,budget):
        value=valid(payload=payload)
        if bad=='json':return 'not json'
        if bad=='scope':value['candidates'][0]['owner_id']='other'
        if bad=='assistant':value['candidates'][0]['source_quote']['field']='answer'
        if bad=='range':value['candidates'][0]['source_quote']['end']=999999
        if bad=='permission':value['candidates'][0]['key']='approval'
        return value
    assert repo.extract_candidates('o','s',generate=call)['status']=='failed'
    assert repo.extract_candidates('o','s',generate=valid)['status']=='attempt_limit'
    assert repo.task_facts('o','s',include_candidates=True)==[]


@pytest.mark.parametrize('mutation',['append','delete','task'])
def test_candidate_publish_cas(tmp_path,mutation):
    repo=SessionRepository(tmp_path/'s.db');tid=append(repo)
    def call(*,payload,budget):
        value=valid(payload=payload)
        if mutation=='append':append(repo)
        elif mutation=='delete':repo.delete('o','s')
        else:repo.update_task('o','s',task_id='new',source_turn_id=tid,expected_revision=repo.revision('o','s'),explicit_user=True)
        return value
    assert repo.extract_candidates('o','s',generate=call)['status']=='stale'
    assert repo.task_facts('o','s',include_candidates=True)==[]


def test_candidates_store_active_turn_revision_and_committed_only(tmp_path):
    store=SessionStore(path=tmp_path/'s.db');tid=append(store.repository)
    with store.turn('o','s') as memory:
        store.record_current_request('o','s',text='new current request',request_id='run')
        def call(*,payload,budget):
            assert [t['turn_id'] for t in payload['turns']]==[tid]
            return valid(payload=payload)
        assert store.extract_candidates('o','s',generate=call)['status']=='generated'
        memory.append_turn('new current request','done')
    assert len(store.history('o','s')['turns'])==2


def test_current_correction_prevents_reproposing_old_constraint(tmp_path):
    repo=SessionRepository(tmp_path/'s.db');append(repo,'预算5000美元')
    repo.record_current_request('o','s',text='预算改为7000美元',request_id='correct')
    result=repo.extract_candidates('o','s',generate=valid)
    assert result['status']=='generated' and result['candidates']==[]
    assert repo.task_facts('o','s',include_candidates=True)==[]


def test_candidate_json_envelope_and_escaping_count_toward_input_limit(tmp_path):
    import json
    repo=SessionRepository(tmp_path/'s.db');append(repo,'"\\'*100)
    captured=[]
    def callback(*,payload,budget):
        captured.append(payload)
        return {'schema_version':1,'candidates':[]}
    result=repo.extract_candidates('o','s',generate=callback,max_chars=1500)
    assert result['status']=='generated' and len(json.dumps(captured[0],ensure_ascii=False))<=1500
    # User text alone is below this cap, but the full envelope exceeds it.
    assert repo.extract_candidates('o','s',generate=lambda **kw:pytest.fail('must not call'),max_chars=300)['status']=='empty'
