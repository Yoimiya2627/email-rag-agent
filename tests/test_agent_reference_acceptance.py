"""Behavioral clock acceptance: actual loop/retrieval with offline data and durable state."""
import copy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS
from threading import Thread

import pytest

from agents.execution_scope import ExecutionScope, use_execution_scope
from agents.runtime import RunContext, use_run_context
from core.session_repository import SessionRepository
from core.tool_results import ToolResultStore, ToolResultUnavailable
from models.schemas import AgentRequest, SearchResult

T0 = datetime(2026, 4, 28, 23, 59, 59, tzinfo=timezone(timedelta(hours=8)))
T1 = T0 + timedelta(days=1, seconds=2)


def reply(identity=None):
    calls = [NS(id=identity,function=NS(name='search_emails',arguments='{}'))] if identity else []
    return NS(choices=[NS(message=NS(content='' if calls else 'done',tool_calls=calls),finish_reason='stop')])


@pytest.fixture
def harness(tmp_path, monkeypatch):
    import agents.agent_loop as loop
    import core.pipeline as pipeline
    import core.embedder as embedder
    wall = [T0]
    class ClockMeta(type):
        def __instancecheck__(cls, instance):
            return isinstance(instance, datetime)
    class Clock(datetime, metaclass=ClockMeta):
        @classmethod
        def now(cls, tz=None):
            return wall[0].astimezone(tz) if tz else wall[0].replace(tzinfo=None)
    monkeypatch.setattr(pipeline, 'datetime', Clock)
    monkeypatch.setattr(pipeline.cfg,'RETRIEVAL_TIMEZONE','')
    monkeypatch.setattr(pipeline.cfg,'RETRIEVAL_TIMEZONE_OFFSET_HOURS',8)
    monkeypatch.setattr(pipeline.cfg,'ENABLE_RERANKER',False)
    monkeypatch.setattr(pipeline.cfg,'ENABLE_QUERY_REWRITE',False)
    monkeypatch.setattr(pipeline.cfg,'AGENT_TOOL_BACKEND','local')
    monkeypatch.setattr(embedder.manifests,'read_active_manifest',lambda:None)
    rows=[SearchResult(email_id='mail',chunk_id='today',content='synthetic',score=1,
        metadata={'date':'2026-04-28T12:00:00+08:00'})]
    monkeypatch.setattr(pipeline,'hybrid_search',lambda *a,**kw:rows)
    owner='eval:reference-acceptance'; session='session'
    repo=SessionRepository(tmp_path/'sessions.db')
    repo.append_turns(owner,session,[{'turn_id':'old','query':'previous','answer':'previous','metadata':{'status':'failed'}}],expected_revision=0)
    store=ToolResultStore(tmp_path/'results.db')
    checkpoints=[]; observations=[]; planner=[]
    ctx=RunContext(owner_id=owner,session_id=session,run_id='durable-run',session_repository=repo,
        context_epoch=repo.context_epoch(owner,session),tool_result_store=store,
        checkpoint_callback=lambda value:checkpoints.append(copy.deepcopy(value)))
    scope=ExecutionScope(owner_id=owner,session_id=session,session_repository=repo,
        context_epoch=ctx.context_epoch,tool_result_store=store)
    schema=[{'type':'function','function':{'name':'search_emails',
        'parameters':{'type':'object','properties':{},'additionalProperties':False}}}]
    def search(*args):
        details={}
        found=pipeline.retrieve('today',filters={'date_hint':'今天'},diagnostics=details)
        observations.append({'reference_time':details['reference_time'],'ids':[r.chunk_id for r in found]})
        return {'ids':[r.chunk_id for r in found]}
    backend=NS(tool_schemas=lambda:schema,call_tool=search)
    monkeypatch.setattr(loop,'_get_tool_backend',lambda:backend)
    request=AgentRequest(query='retrieve today messages twice',session_id=session)
    def execute(sequence,checkpoint=None,execution_scope=None,advance_after_first=False):
        remaining=iter(sequence)
        def create(**kwargs):
            planner.append(copy.deepcopy(kwargs))
            if advance_after_first:wall[0]=T1
            value=next(remaining)
            if isinstance(value,BaseException):raise value
            return value
        monkeypatch.setattr(loop,'_get_client',lambda:NS(chat=NS(completions=NS(create=create))))
        ctx.model_usage=None
        with use_run_context(ctx),use_execution_scope(execution_scope or scope):
            return loop.run_agent_loop(request,owner_id=owner,session_id=session,checkpoint=checkpoint)
    return NS(loop=loop,pipeline=pipeline,wall=wall,ctx=ctx,scope=scope,owner=owner,session=session,
        request=request,execute=execute,checkpoints=checkpoints,observations=observations,planner=planner,
        tmp=tmp_path)


def initial_checkpoint(h):
    result=h.execute([reply('first'),TimeoutError('offline pause')])
    assert result.metadata['status']=='timeout'
    checkpoint=copy.deepcopy(h.checkpoints[-1])
    assert checkpoint['version']==2 and checkpoint['safe'] and checkpoint['tool_calls']==1
    assert checkpoint['tool_result_refs']['first']
    return checkpoint


def test_live_run_crosses_midnight_but_planner_and_real_retrieval_share_t0(harness):
    h=harness
    result=h.execute([reply('first'),reply('second'),reply()],advance_after_first=True)
    assert result.metadata['status']=='success'
    assert h.observations==[{'reference_time':T0.isoformat(),'ids':['today']}]*2
    assert all(T0.isoformat() in call['messages'][0]['content'] for call in h.planner)
    assert h.pipeline._now()==T1


@pytest.mark.parametrize('new_thread',[False,True])
def test_persistent_resume_uses_t0_and_does_not_replay_completed_tool(harness,new_thread):
    h=harness; checkpoint=initial_checkpoint(h);h.wall[0]=T1
    result=[]; errors=[]
    def work():
        try:result.append(h.execute([reply('second'),reply()],checkpoint))
        except BaseException as exc:errors.append(exc)
    if new_thread:
        worker=Thread(target=work);worker.start();worker.join(timeout=10)
        assert not worker.is_alive()
    else:work()
    assert not errors
    assert result[0].metadata['status']=='success'
    assert h.observations==[{'reference_time':T0.isoformat(),'ids':['today']}]*2
    assert T0.isoformat() in h.planner[-2]['messages'][0]['content']
    final=h.checkpoints[-1]
    assert final['tool_result_refs']['first']==checkpoint['tool_result_refs']['first']
    assert set(final['tool_result_refs'])=={'first','second'}
    for key in ('owner_id','session_id','run_id','context_epoch','task_memory_guard'):
        assert final[key]==checkpoint[key]
    assert h.pipeline._now()==T1


@pytest.mark.parametrize('condition',['conflicting_reference','legacy_checkpoint','wrong_owner'])
def test_invalid_resume_rejected_before_any_client_or_schema_lookup(harness,monkeypatch,condition):
    h=harness; checkpoint=initial_checkpoint(h);scope=h.scope
    if condition=='conflicting_reference':
        scope=ExecutionScope(owner_id=h.owner,session_id=h.session,session_repository=h.ctx.session_repository,
            context_epoch=h.ctx.context_epoch,tool_result_store=h.ctx.tool_result_store,evaluation=True,
            approval_store_path=h.tmp/'approval.db',run_dir=h.tmp,evaluation_reference_time=T1)
    elif condition=='wrong_owner':checkpoint['owner_id']='other-owner'
    else:
        # This is exactly the legacy v2 shape, independent of new clock field names.
        old_fields={'version','kind','safe','request_hash','schema_hash','messages','history_message_count',
            'steps','tool_calls','counts','had_error','pending','next_round','evidence','visible_evidence',
            'operation_slots','model_metrics','compaction_policy','context_fingerprint','task_memory_guard',
            'owner_id','session_id','run_id','context_epoch','tool_result_refs','generation_evidence_refs'}
        checkpoint={k:v for k,v in checkpoint.items() if k in old_fields}
    reached=[]
    def client():reached.append('client');raise AssertionError('client access before rejection')
    def backend():
        reached.append('backend')
        return NS(tool_schemas=lambda:reached.append('schemas') or [])
    monkeypatch.setattr(h.loop,'_get_client',client)
    monkeypatch.setattr(h.loop,'_get_tool_backend',backend)
    h.ctx.model_usage=None
    with use_run_context(h.ctx),use_execution_scope(scope),pytest.raises((ValueError,ToolResultUnavailable)):
        h.loop.run_agent_loop(h.request,owner_id=h.owner,session_id=h.session,checkpoint=checkpoint)
    assert reached==[]
    assert len(h.observations)==1


def test_changed_valid_checkpoint_instant_rejected_before_clients(harness,monkeypatch):
    h=harness;checkpoint=initial_checkpoint(h)
    checkpoint['reference_clock']['instant_utc']=T1.astimezone(timezone.utc).isoformat()
    reached=[]
    def forbidden():reached.append(True);raise AssertionError('client reached')
    monkeypatch.setattr(h.loop,'_get_client',forbidden)
    monkeypatch.setattr(h.loop,'_get_tool_backend',forbidden)
    h.ctx.model_usage=None
    with use_run_context(h.ctx),use_execution_scope(h.scope),pytest.raises((ValueError,ToolResultUnavailable)):
        h.loop.run_agent_loop(h.request,owner_id=h.owner,session_id=h.session,checkpoint=checkpoint)
    assert reached==[]


def test_equivalent_offset_checkpoint_instant_is_accepted(harness):
    h=harness;checkpoint=initial_checkpoint(h);h.wall[0]=T1
    checkpoint['reference_clock']['instant_utc']=T0.astimezone(timezone(timedelta(hours=-5))).isoformat()
    result=h.execute([reply('second'),reply()],checkpoint)
    assert result.metadata['status']=='success'
    assert h.observations==[{'reference_time':T0.isoformat(),'ids':['today']}]*2
