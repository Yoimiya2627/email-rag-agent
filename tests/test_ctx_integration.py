"""Main-agent acceptance: real API/storage/selection flows with provider doubles."""
import json
import sqlite3
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient
import api.main as api
import config.settings as cfg
from api.sessions import SessionStore
from agents.runtime import current_run
from core.memory import build_model_messages
from core.jobs import JobStore,JobManager
from core.tool_results import ToolResultStore
from models.schemas import AgentResponse


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    store=SessionStore(path=tmp_path/'sessions.sqlite3',max_history_turns=30)
    monkeypatch.setattr(api,'sessions',store)
    monkeypatch.setattr(api,'_jobs',None)
    monkeypatch.setattr(cfg,'TOOL_RESULT_STORE_PATH',str(tmp_path/'results.sqlite3'))
    monkeypatch.setattr(cfg,'JOB_STORE_PATH',str(tmp_path/'jobs.sqlite3'))
    monkeypatch.setattr(cfg,'ENABLE_CONTEXT_SUMMARY',False)
    monkeypatch.setattr(cfg,'ENABLE_CONTEXT_CANDIDATES',False)
    monkeypatch.setattr(cfg,'ENABLE_CONTEXT_OPTIMIZATION',True)
    return store,TestClient(api.app)


def seed(store,query='预算5000，不得发送邮件。',sid='s',tid='source',owner='local',answer='已记录'):
    with store.turn(owner,sid) as memory:
        memory.append_turn(query,answer,turn_id=tid)
    return tid


def test_forty_chats_do_not_turn_old_ordinary_requests_into_required_overflow(fixture):
    store,client=fixture;captured=[]
    def runner(request,memory):
        messages=build_model_messages('助理',request.query,memory.to_messages(),stage='generate',max_output_tokens=100)
        captured.append(messages)
        return AgentResponse(answer='合成答复')
    with patch.object(api,'route',runner):
        for index in range(40):
            text=f'第{index}轮，'+'请核对普通事项。'*12
            result=client.post('/chat',json={'query':text,'session_id':'s'})
            assert result.status_code==200,result.text
            assert text in captured[-1][-1]['content']
    assert len(store.history('local','s')['turns'])==40


def test_history_hit_after_200_and_paged_original_never_crosses_owner(fixture):
    store,client=fixture
    answer='无关背景。'*100+'订单 Q-728 不含税，金额 8000 元。'
    seed(store,answer=answer)
    rows=client.get('/chat/history/search',params={'session_id':'s','query':'Q-728'}).json()['turns']
    assert rows and any('不含税' in hit['text'] for hit in rows[0]['hits'])
    page=client.get('/chat/history/turn',params={'session_id':'s','turn_id':'source','offset':490,'limit':100}).json()
    assert page['text']==answer[490:590]
    seed(store,'private-other',sid='other',tid='other-source',owner='other')
    assert client.get('/chat/history/turn',params={'session_id':'other','turn_id':'other-source'}).status_code==404


def test_task_and_fact_lifecycle_api_rejects_stale_versions(fixture):
    store,client=fixture;source=seed(store)
    first=client.post('/chat/facts',json={'session_id':'s','key':'budget','value':'5000','source_turn_id':source})
    assert first.status_code==200
    state=client.get('/chat/context',params={'session_id':'s'}).json()
    task={'session_id':'s','task_id':'order-b','goal':'处理第二笔订单','objects':['B-7'],
          'source_turn_id':source,'expected_revision':state['revision']}
    assert client.post('/chat/tasks',json={**task,'owner_id':'forged'}).status_code==422
    assert client.post('/chat/tasks',json=task).status_code==200
    assert client.post('/chat/tasks',json=task).status_code==409
    state=client.get('/chat/context',params={'session_id':'s'}).json()
    assert state['task_id']=='order-b' and not state['facts']
    assert state['execution_authority'] is False
    saved=client.post('/chat/facts',json={'session_id':'s','key':'date','value':'2026-10-01','source_turn_id':source}).json()
    action={'session_id':'s','key':'date','source_turn_id':source,'expected_version':saved['version']}
    assert client.post('/chat/facts/revoke',json=action).status_code==200
    assert not client.get('/chat/facts',params={'session_id':'s'}).json()['facts']
    assert client.post('/chat/facts/revoke',json=action).status_code==409


def test_current_correction_retained_without_second_confirmation(fixture):
    store,client=fixture;source=seed(store)
    store.set_task_fact('local','s','budget','5000',source_turn_id=source,expected_version=0,explicit_user=True)
    prompts=[]
    def runner(request,memory):
        prompts.append(build_model_messages('system',request.query,memory.to_messages(),stage='generate',max_output_tokens=100))
        return AgentResponse(answer='ok')
    with patch.object(api,'route',runner):
        assert client.post('/chat',json={'session_id':'s','query':'预算改为8000，仍然不得发送邮件。'}).status_code==200
        assert client.post('/chat',json={'session_id':'s','query':'继续执行查询'}).status_code==200
    assert '预算改为8000' in prompts[-1][-1]['content']
    assert not store.context_state('local','s')['facts']
    assert not store.context_state('local','s')['candidates']


def test_delete_hides_results_even_if_cross_store_cleanup_fails(fixture,monkeypatch):
    store,client=fixture;seed(store)
    jobs=JobStore(cfg.JOB_STORE_PATH)
    manager=JobManager(jobs,lambda *a:{})
    monkeypatch.setattr(api,'_jobs',manager)
    job,_=jobs.create('local','agent',{'session_id':'s','query':'secret','_session_epoch':0})
    jobs.claim('local',job['id']);jobs.finish('local',job['id'],'succeeded',result={'answer':'private-result'})
    results=ToolResultStore(cfg.TOOL_RESULT_STORE_PATH)
    ref=results.put(owner='local',session='s',run=job['id'],call_id='c',epoch=0,tool='email_stats',argument_hash='a'*64,value={'text':'private-result'})
    with patch.object(manager,'invalidate_session',side_effect=OSError('offline')),patch.object(ToolResultStore,'delete_session',side_effect=OSError('offline')):
        erased=client.delete('/chat/history',params={'session_id':'s'}).json()
    assert erased['history_inaccessible'] and len(erased['cleanup_pending'])==2
    visible=client.get('/jobs/'+job['id']).json()
    assert visible['result'] is None and visible['error_code']=='session_invalidated'
    assert client.post('/jobs/'+job['id']+'/resume').status_code==409
    assert client.get('/chat/tool-result',params={'session_id':'s','run_id':job['id'],'result_id':ref['result_id']}).status_code in (404,409)
    monkeypatch.setattr(api,'_jobs',None)
    # Restarted inspection must apply the same deletion epoch before a manager exists.
    progress=client.get('/chat/context',params={'session_id':'s'}).json()['execution_progress']
    assert progress[0]['status']=='cancelled' and progress[0]['progress']=={}
    assert client.delete('/chat/history',params={'session_id':'s'}).json()['success']


def test_summary_is_explicit_experiment_and_stateless_query_gets_no_session(fixture,monkeypatch):
    store,client=fixture;seed(store)
    assert client.post('/chat/summary/rebuild',json={'session_id':'s'}).status_code==409
    seen=[]
    def generate(*args,**kwargs):
        seen.append(current_run().task_context);return 'answer'
    with patch('core.pipeline.retrieve',return_value=[]),patch('core.generator.generate_answer',side_effect=generate):
        assert client.post('/query',json={'query':'hello'}).status_code==200
    assert not seen[0]


def test_maintenance_copy_backup_restore_and_cleanup_preserve_source(fixture,tmp_path):
    from scripts.context_maintenance import migrate_copy,cleanup_context
    from scripts.state_maintenance import backup_states,restore_states
    from core.session_repository import SessionRepository
    store,client=fixture;seed(store)
    report=migrate_copy(store.repository.path,tmp_path/'migrated.sqlite3',service_stopped=True)
    assert report['original_preserved'] and report['transcript']['turns']==1
    resultstore=ToolResultStore(cfg.TOOL_RESULT_STORE_PATH)
    ref=resultstore.put(owner='local',session='s',run='r',call_id='c',epoch=0,tool='email_stats',argument_hash='a'*64,value={'text':'private'})
    backup_states({'sessions':store.repository.path,'tool_results':resultstore.path},tmp_path/'backup',service_stopped=True)
    restore_states(tmp_path/'backup',tmp_path/'restore',service_stopped=True)
    restored=SessionRepository(tmp_path/'restore/sessions.sqlite3')
    assert restored.context_epoch('local','s')==1 and restored.history('local','s')['turns'][0]['query'].startswith('预算')
    assert store.repository.context_epoch('local','s')==0
    store.clear('local','s')
    preview=cleanup_context(store.repository.path,tool_results_path=resultstore.path,service_stopped=True)
    assert preview['tool_results_eligible']==1 and preview['changed']==0
    applied=cleanup_context(store.repository.path,tool_results_path=resultstore.path,service_stopped=True,apply=True)
    assert applied['changed']==1
    assert cleanup_context(store.repository.path,tool_results_path=resultstore.path,service_stopped=True,apply=True)['changed']==0


def test_context_path_probe_short_and_deep(tmp_path):
    from scripts.context_paths import check_context_path
    assert check_context_path(tmp_path/'sessions.sqlite3',probe=True)['write_probe']=='passed'
    deep=tmp_path/('x'*180)/'sessions.sqlite3'
    assert check_context_path(deep,windows=True)['status']=='warning'


def test_experimental_summary_and_candidates_share_real_call_accounting(fixture,monkeypatch):
    from types import SimpleNamespace as NS
    store,client=fixture;seed(store,query='预算8000元。')
    monkeypatch.setattr(cfg,'ENABLE_CONTEXT_SUMMARY',True)
    monkeypatch.setattr(cfg,'ENABLE_CONTEXT_CANDIDATES',True)
    monkeypatch.setattr(cfg,'CONTEXT_SUMMARY_MIN_TURNS',1)
    calls=[]
    def create(**kwargs):
        calls.append(kwargs)
        payload=json.loads(kwargs['messages'][-1]['content']);turn=payload['turns'][0]
        quote={'turn_id':turn['turn_id'],'start':0,'end':len(turn['query']),'text':turn['query']}
        if 'sections' in payload:
            value={'schema_version':1,'sections':{'goals':[{'text':'核对预算8000元','source_quotes':[{**quote,'field':'query'}]}]}}
        else:
            value={'schema_version':1,'candidates':[{'key':'budget','kind':'constraint','value':'8000元','source_quote':quote}]}
        return NS(choices=[NS(message=NS(content=json.dumps(value,ensure_ascii=False)),finish_reason='stop')],
                  usage=NS(prompt_tokens=100,completion_tokens=60,total_tokens=160))
    provider=NS(chat=NS(completions=NS(create=create)))
    with patch('core.model_clients.get_model_client',return_value=provider),patch.object(api,'route',return_value=AgentResponse(answer='合成答复')):
        response=client.post('/chat',json={'session_id':'s','query':'继续核对'} )
    assert response.status_code==200,response.text
    metadata=response.json()['metadata']
    assert metadata['model_usage']['call_count']==2 and metadata['model_usage']['actual_total_tokens']==320
    assert {call['stage'] for call in metadata['model_usage']['calls']}=={'session_summary','context_candidates'}
    state=store.context_state('local','s')
    assert state['summary'] and state['candidates'] and not state['facts']
    assert state['candidates'][0]['status']=='candidate'
    assert len(store.history('local','s')['turns'])==2


def test_valid_json_with_incomplete_model_finish_is_not_published(fixture,monkeypatch):
    from types import SimpleNamespace as NS
    store,client=fixture;seed(store)
    monkeypatch.setattr(cfg,'ENABLE_CONTEXT_SUMMARY',True)
    monkeypatch.setattr(cfg,'CONTEXT_SUMMARY_MIN_TURNS',1)
    provider=NS(chat=NS(completions=NS(create=lambda **kwargs:NS(
        choices=[NS(message=NS(content='{"schema_version":1,"sections":{}}'),finish_reason='length')],usage=None))))
    with patch('core.model_clients.get_model_client',return_value=provider),patch.object(api,'route',return_value=AgentResponse(answer='可用回退')):
        response=client.post('/chat',json={'session_id':'s','query':'继续'})
    assert response.status_code==200,response.text
    assert store.get_semantic_summary('local','s') is None
    assert response.json()['metadata']['context_metrics']['summary']['status']=='failed'
    assert response.json()['metadata']['model_usage']['unknown_usage_calls']==1


@pytest.mark.parametrize('mode',['graph','stream'])
def test_real_graph_and_sse_model_boundaries_retain_constraint_and_commit(fixture,monkeypatch,mode):
    from types import SimpleNamespace as NS
    from models.schemas import SearchResult
    import agents.graph_workflow as graph
    import core.generator as generator
    import agents.coordinator as coordinator
    store,client=fixture;source=seed(store)
    store.set_task_fact('local','s','budget','5000',source_turn_id=source,expected_version=0,explicit_user=True)
    monkeypatch.setattr(cfg,'ENABLE_QUERY_REWRITE',True)
    monkeypatch.setattr(graph,'_graph',None)
    captured=[]
    def create(**kwargs):
        captured.append(kwargs)
        usage=NS(prompt_tokens=100,completion_tokens=20,total_tokens=120)
        if kwargs.get('stream'):
            return iter([NS(choices=[NS(delta=NS(content='合成回答'),finish_reason=None)],usage=None),
                         NS(choices=[NS(delta=NS(content=''),finish_reason='stop')],usage=usage)])
        system=kwargs['messages'][0]['content']
        text=('[0]' if '候选邮件是参考数据' in system else
              '{"intent":"retrieve","reason":"合成检查"}' if '任务协调器' in system else
              '订单检索' if '将以下搜索查询改写' in system else '合成回答')
        return NS(choices=[NS(message=NS(content=text),finish_reason='stop')],usage=usage)
    provider=NS(chat=NS(completions=NS(create=create)))
    for module in (graph,generator,coordinator):monkeypatch.setattr(module,'_get_client',lambda:provider)
    hit=SearchResult(email_id='e',chunk_id='c',content='合成订单原文',score=1,metadata={})
    with patch.object(graph,'retrieve',return_value=[hit]),patch.object(graph,'extract_filters',return_value={}),\
         patch('core.pipeline.retrieve',return_value=[hit]):
        response=client.post('/chat/'+mode,json={'session_id':'s','query':'继续核对订单'})
    assert response.status_code==200,response.text
    if mode=='stream':
        assert 'data: [DONE]' in response.text
        events=[json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: {')]
        metadata=next(row['metadata'] for row in events if 'sources' in row)
        stages={'intent','stream_generate'}
    else:
        assert response.json()['answer']=='合成回答'
        metadata=response.json()['metadata'];stages={'intent','rewrite','graph_grade','generate'}
    assert {call['stage'] for call in metadata['model_usage']['calls']}==stages
    assert metadata['completion_status']=='complete'
    assert metadata['model_usage']['actual_total_tokens']==120*len(stages)
    assert all('5000' in call['messages'][-1]['content'] for call in captured)
    assert all('继续核对订单' in call['messages'][-1]['content'] for call in captured)
    assert len(store.history('local','s')['turns'])==2


def test_required_repository_overflow_stops_actual_api_before_provider(fixture):
    store,client=fixture;seed(store)
    store.repository.record_current_request('local','s',text='预算改为'+('长约束'*40000),request_id='large')
    with patch('agents.coordinator._get_client',side_effect=AssertionError('must not reach provider')):
        result=client.post('/chat',json={'session_id':'s','query':'继续核对'})
    assert result.status_code==413,result.text


def test_mcp_sdk_history_scope_is_unavailable_without_trusted_session(monkeypatch):
    import asyncio
    import mcp_server
    monkeypatch.setattr(cfg,'MCP_HOST','127.0.0.1')
    monkeypatch.setattr(cfg,'MCP_AUTH_TOKEN','')
    server=mcp_server.build_server()
    for name,arguments in [('search_history',{'query':'预算'}),('get_turn',{'turn_id':'source'}),
                           ('get_tool_result',{'result_id':'a'*32})]:
        blocks,value=asyncio.run(server.call_tool(name,arguments))
        assert value['status']=='error'
        assert value['error_code'] in ('history_unavailable','tool_result_unavailable')
        assert not value['evidence_refs']


def test_optional_langchain_demo_invoke_contract_and_missing_dependency(monkeypatch):
    from types import SimpleNamespace as NS
    import langchain_version.rag_chain as demo
    chain=NS(invoke=lambda payload:{'answer':payload['question'],'source_documents':[NS(page_content='source',metadata={})]})
    assert demo.ask(chain,'query')=={'answer':'query','sources':[{'content':'source','metadata':{}}]}
    monkeypatch.setattr(demo,'_HAS_LANGCHAIN',False)
    with pytest.raises(ImportError,match='LangChain not installed'):demo.build_chain()


def test_cleanup_paginates_mutating_jobs_without_skipping_rows(fixture):
    from scripts.context_maintenance import cleanup_context
    store,client=fixture;seed(store)
    jobs=JobStore(cfg.JOB_STORE_PATH)
    ids=[jobs.create('local','agent',{'session_id':'s','query':'synthetic','_session_epoch':0})[0]['id'] for _ in range(19)]
    store.clear('local','s')
    preview=cleanup_context(store.repository.path,jobs_path=jobs.path,service_stopped=True)
    assert preview['jobs_eligible']==19 and preview['changed']==0
    result=cleanup_context(store.repository.path,jobs_path=jobs.path,apply=True,service_stopped=True)
    assert result['changed']==19
    assert all(jobs.get('local',identifier,private=True)['request']=={'session_id':'s'} for identifier in ids)
    assert cleanup_context(store.repository.path,jobs_path=jobs.path,apply=True,service_stopped=True)['changed']==0


def test_feature_switch_rollback_preserves_transcripts_and_constraints(fixture,monkeypatch):
    store,client=fixture;source=seed(store)
    store.set_task_fact('local','s','budget','5000',source_turn_id=source,expected_version=0,explicit_user=True)
    for flag in ('ENABLE_CONTEXT_OPTIMIZATION','CONTEXT_TOOL_RESULTS_ENABLED','CONTEXT_TOOL_COMPACTION_ENABLED'):
        monkeypatch.setattr(cfg,flag,False)
    captured=[]
    def runner(request,memory):
        captured.append(build_model_messages('system',request.query,memory.to_messages(),stage='generate',max_output_tokens=100))
        assert current_run().tool_result_store is None
        return AgentResponse(answer='合成回退答复')
    with patch.object(api,'route',runner):
        response=client.post('/chat',json={'session_id':'s','query':'继续核对原预算'})
    assert response.status_code==200,response.text
    assert '5000' in captured[0][-1]['content'] and '继续核对原预算' in captured[0][-1]['content']
    assert len(store.history('local','s')['turns'])==2
    assert store.context_state('local','s')['facts'][0]['value']=='5000'
