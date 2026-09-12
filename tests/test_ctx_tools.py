import copy
import hashlib
import json
import sqlite3

import pytest

from agents.runtime import RunContext, normalize_tool_result, use_run_context, bounded_json
from core.session_repository import SessionRepository
from core.tool_results import ToolResultStore, ToolResultUnavailable, trusted_result_scope
from core.tool_compaction import compact_tool_messages, validate_checkpoint_references, POLICY_VERSION
from core.tool_compaction import task_memory_guard, validate_task_memory_guard


def context(tmp_path, **limits):
    repo = SessionRepository(tmp_path / 's.db')
    repo.append_turns('owner', 'session', [{'turn_id':'turn', 'query':'原话', 'answer':'前言'*500+'订单 A-102 待确认', 'metadata':{'status':'failed'}}], expected_revision=0)
    return RunContext(owner_id='owner', session_id='session', run_id='run', session_repository=repo,
                      context_epoch=repo.context_epoch('owner','session'), tool_result_store=ToolResultStore(tmp_path/'r.db', **limits))


def put(ctx, call='call', value=None):
    return ctx.tool_result_store.put(**trusted_result_scope(ctx), call_id=call,
        tool='search_emails', argument_hash='a'*64, value=value or normalize_tool_result({'text':'正文'*3000}))


def test_immutable_scope_hash_capacity_and_paging(tmp_path):
    ctx = context(tmp_path)
    ref = put(ctx)
    assert put(ctx) == ref
    with pytest.raises(ToolResultUnavailable, match='immutable'):
        put(ctx, value={'different':True})
    scope = trusted_result_scope(ctx)
    for key in ('owner','session','run','epoch'):
        wrong = {**scope, key: 1 if key == 'epoch' else 'wrong'}
        with pytest.raises(ToolResultUnavailable):
            ctx.tool_result_store.load(ref['result_id'], **wrong)
    payload, _ = ctx.tool_result_store.load(ref['result_id'], **scope)
    text, start = '', 0
    while True:
        page = ctx.tool_result_store.page(ref['result_id'], start=start, limit=77, **scope)
        assert page['page_hash'] == hashlib.sha256(page['text'].encode()).hexdigest()
        text += page['text']
        if not page['has_more']: break
        start = page['next_start']
    assert text == payload
    with sqlite3.connect(ctx.tool_result_store.path) as db:
        db.execute("UPDATE tool_results SET payload='tampered'")
    with pytest.raises(ToolResultUnavailable, match='hash'):
        ctx.tool_result_store.load(ref['result_id'], **scope)


def test_store_limits_and_retention(tmp_path):
    ctx = context(tmp_path, max_object_bytes=200, max_run_bytes=250)
    with pytest.raises(ToolResultUnavailable, match='object_byte_limit'): put(ctx)
    put(ctx, 'a', {'text':'x'*150})
    with pytest.raises(ToolResultUnavailable, match='store_capacity'): put(ctx, 'b', {'text':'x'*150})
    assert ctx.tool_result_store.prune(now=10**12) == 1
    assert ctx.tool_result_store.delete_session('owner','session') == 0


def test_put_reclaims_expired_capacity_without_renewing_old_references(tmp_path):
    ctx=context(tmp_path,max_total_bytes=200,max_objects=1)
    ref=put(ctx,value={'text':'x'*150})
    scope=trusted_result_scope(ctx)
    with pytest.raises(ToolResultUnavailable,match='store_capacity'):
        ctx.tool_result_store.put(**{**scope,'run':'fresh-run'},call_id='fresh-call',tool='search_emails',
            argument_hash='a'*64,value={'text':'y'*150})
    with sqlite3.connect(ctx.tool_result_store.path) as db:
        db.execute('UPDATE tool_results SET expires=0 WHERE id=?',(ref['result_id'],))
    # A retry of the retained expired slot fails before pruning or writing.
    with pytest.raises(ToolResultUnavailable,match='immutable_call_conflict'):
        put(ctx,value={'text':'x'*150})
    with pytest.raises(ToolResultUnavailable,match='missing_or_expired'):
        ctx.tool_result_store.load(ref['result_id'],**scope)
    # Production new-write path reclaims both bytes and the object-count slot.
    replacement=ctx.tool_result_store.put(**{**scope,'run':'fresh-run'},call_id='fresh-call',tool='search_emails',
        argument_hash='a'*64,value={'text':'y'*150})
    assert replacement['result_id']!=ref['result_id']
    with sqlite3.connect(ctx.tool_result_store.path) as db:
        count,used=db.execute('SELECT count(*),sum(bytes) FROM tool_results').fetchone()
    assert count==1 and used<=200
    checkpoint={'version':2,'compaction_policy':POLICY_VERSION,'owner_id':'owner','session_id':'session',
        'context_epoch':ctx.context_epoch,'run_id':'run','tool_result_refs':{'protocol':ref},
        'steps':[{'tool_call_id':'protocol','result_ref':ref}]}
    with pytest.raises(ToolResultUnavailable,match='missing_or_expired'):
        validate_checkpoint_references(checkpoint,ctx)
    # No expired ID can resolve to the new payload after collection.
    with pytest.raises(ToolResultUnavailable,match='missing_or_expired'):
        ctx.tool_result_store.load(ref['result_id'],**scope)


def test_nested_credentials_removed_before_hash_and_global_capacity(tmp_path):
    from core.tool_results import sanitize_tool_result
    ctx=context(tmp_path, max_total_bytes=250)
    value={'status':'unknown','data':{'content':'mail body','debug':{'raw':'secret'},
        'nested':[{'api_key':'secret','access_token':'secret','coverage':{'unread_attachments':2}}]}}
    ref=put(ctx,value=value)
    raw,_=ctx.tool_result_store.load(ref['result_id'],**trusted_result_scope(ctx))
    assert 'secret' not in raw and json.loads(raw)==sanitize_tool_result(value)
    assert json.loads(raw)['status']=='unknown' and 'unread_attachments' in raw
    assert ref['content_hash']==hashlib.sha256(raw.encode()).hexdigest()
    scope={**trusted_result_scope(ctx),'run':'another-run'}
    with pytest.raises(ToolResultUnavailable,match='store_capacity'):
        ctx.tool_result_store.put(**scope,call_id='new',tool='search_emails',argument_hash='a'*64,value={'text':'x'*200})


def test_io_failure_leaves_no_reference(tmp_path,monkeypatch):
    ctx=context(tmp_path)
    monkeypatch.setattr(ctx.tool_result_store,'_db',lambda: (_ for _ in ()).throw(OSError('synthetic io')))
    with pytest.raises(OSError): put(ctx)
    assert ctx.tool_result_refs=={}


def test_history_tools_scoped_failed_claims_and_no_evidence(tmp_path):
    from agents.tools import search_history, get_turn, get_tool_result, call_tool
    ctx = context(tmp_path)
    assert search_history('订单')['error_code'] == 'history_unavailable'
    with use_run_context(ctx):
        hit = search_history('A-102')['items'][0]
        assert hit['metadata']['status'] == 'failed'
        assert any('A-102' in row['text'] for row in hit['hits'])
        page = get_turn('turn', offset=1000)
        assert 'A-102' in page['text']
        ref = put(ctx)
        ctx.tool_result_refs['protocol-call'] = ref
        assert get_tool_result(ref['result_id'])['material_type'] == 'historical_tool_result'
        assert call_tool('get_tool_result', {'result_id':ref['result_id'], 'owner':'other'})['error_code'] == 'validation_error'
        assert not ctx.evidence and not ctx.visible_evidence
        ctx.session_repository.delete('owner','session')
        assert get_tool_result(ref['result_id'])['error_code'] == 'tool_result_unavailable'
        assert get_turn('turn')['error_code'] == 'history_unavailable'


def group(call_id, result):
    return [{'role':'assistant','content':'','tool_calls':[{'id':call_id,'type':'function','function':{'name':'search_emails','arguments':'{}'}}]},
            {'role':'tool','tool_call_id':call_id,'content':json.dumps(result)}]


def test_compaction_closed_groups_preserves_current_tail_and_coverage(tmp_path):
    ctx = context(tmp_path)
    result = normalize_tool_result({'text':'mail'*2000,'coverage':{'attachment_inventory_status':'unknown','unread_attachments':None}})
    ref = put(ctx, value=result); ctx.tool_result_refs['call'] = ref
    messages = [{'role':'system','content':'system'},{'role':'user','content':'当前原话'}] + group('call',result) + group('tail',result)
    compact = compact_tool_messages(messages,ctx)
    assert compact[:2] == messages[:2] and compact[-2:] == messages[-2:]
    saved = json.loads(compact[3]['content'])
    assert saved['result_ref'] == ref and saved['source_coverage'][0]['attachment_inventory_status'] == 'unknown'
    assert saved['evidence_refs'] == []
    assert messages[3]['content'] != compact[3]['content']
    broken = messages[:3]
    assert compact_tool_messages(broken,ctx,keep_groups=0) == broken
    pending = copy.deepcopy(messages)
    pending[3]['content'] = json.dumps({'status':'unknown','side_effect_state':'unknown'})
    assert compact_tool_messages(pending,ctx,keep_groups=0) == pending


def test_checkpoint_v2_ref_binding_and_deleted_result(tmp_path):
    ctx = context(tmp_path); ref = put(ctx)
    checkpoint = {'version':2,'compaction_policy':POLICY_VERSION,'owner_id':'owner','session_id':'session',
        'context_epoch':ctx.context_epoch,'run_id':'run','tool_result_refs':{'protocol':ref},
        'steps':[{'tool_call_id':'protocol','result_ref':ref}]}
    validate_checkpoint_references(checkpoint,ctx)
    for key in ('owner_id','session_id','run_id','context_epoch','compaction_policy'):
        with pytest.raises(ToolResultUnavailable): validate_checkpoint_references({**checkpoint,key:'wrong'},ctx)
    with pytest.raises(ToolResultUnavailable,match='legacy_context_checkpoint_requires_restart'):
        validate_checkpoint_references({'version':1},ctx)
    validate_checkpoint_references({'version':1},RunContext())
    with pytest.raises(ToolResultUnavailable): validate_checkpoint_references({'version':3},ctx)
    ctx.tool_result_store.delete_session('owner','session')
    with pytest.raises(ToolResultUnavailable): validate_checkpoint_references(checkpoint,ctx)


@pytest.mark.parametrize('change',['fact','revoke','task','correction'])
def test_memory_guard_ignores_failed_append_but_rejects_user_state_change(tmp_path,change):
    ctx=context(tmp_path); repo=ctx.session_repository
    repo.set_task_fact('owner','session','budget',100,source_turn_id='turn',expected_version=0,explicit_user=True)
    guard=task_memory_guard(ctx)
    repo.append_turns('owner','session',[{'turn_id':'failed','query':'q','answer':'partial','metadata':{'status':'failed'}}],expected_revision=repo.revision('owner','session'))
    validate_task_memory_guard(guard,ctx)
    if change=='fact':
        repo.set_task_fact('owner','session','budget',200,source_turn_id='turn',expected_version=1,explicit_user=True)
    elif change=='revoke':
        repo.revoke_task_fact('owner','session','budget',source_turn_id='turn',expected_version=1,explicit_user=True)
    elif change=='task':
        repo.update_task('owner','session',task_id='next',goal='新任务',source_turn_id='turn',expected_revision=repo.revision('owner','session'),explicit_user=True)
    else:
        repo.record_current_request('owner','session',text='预算改为200',request_id='correction')
    with pytest.raises(ToolResultUnavailable,match='task_memory_changed'):
        validate_task_memory_guard(guard,ctx)


def test_memory_guard_pins_only_used_summary_and_checks_invalidation(tmp_path):
    ctx=context(tmp_path); repo=ctx.session_repository
    original=task_memory_guard(ctx)
    result=repo.generate_summary('owner','session',min_turns=1,
        generate=lambda **kwargs:{'schema_version':1,'sections':{'goals':[{'text':'原话', 'source_quotes':[{'turn_id':'turn','field':'query','start':0,'end':2,'text':'原话'}]}]}})
    assert result['status']=='generated'
    validate_task_memory_guard(original,ctx)  # unused new summary has no bearing on old messages
    ctx.task_context={'text':json.dumps({'derived_summaries':[{'summary':result['summary']}]})}
    guard=task_memory_guard(ctx)
    with repo._connect() as db:
        db.execute("UPDATE semantic_summaries SET status='stale'")
    with pytest.raises(ToolResultUnavailable,match='summary_invalidated'):
        validate_task_memory_guard(guard,ctx)


def test_restored_email_range_requires_current_matching_version(tmp_path):
    from core.tool_compaction import revalidate_restored_evidence
    from core.evidence import with_visible_reference
    ctx=context(tmp_path)
    row=with_visible_reference({'email_id':'e','chunk_id':'c','content':'original mail', 'metadata':{'source_version':'v1'}})
    ctx.visible_evidence[('e','c')]=copy.deepcopy(row)
    revalidate_restored_evidence(ctx,loader=lambda _: {'chunks':[row]})
    assert ('e','c') in ctx.visible_evidence
    changed={**row,'source_version':'v2','content':'changed mail'}
    revalidate_restored_evidence(ctx,loader=lambda _: {'chunks':[changed]})
    assert not ctx.visible_evidence and ctx.context_metrics['restored_evidence_invalidated']==1


def test_page_truncation_retains_correct_offsets_hashes(tmp_path):
    ctx = context(tmp_path); ref = put(ctx)
    page = ctx.tool_result_store.page(ref['result_id'], start=100,limit=4000,**trusted_result_scope(ctx))
    clipped = json.loads(bounded_json(normalize_tool_result(page), 2500))['data']
    assert clipped['page_end'] == clipped['page_start'] + len(clipped['text'])
    assert clipped['page_hash'] == hashlib.sha256(clipped['text'].encode()).hexdigest()
    assert clipped['next_start'] == clipped['page_end']


@pytest.mark.parametrize('inventory,expected', [(None, 'unknown'), ([{'name':'a.pdf','status':'unread'}], 'available')])
def test_k1_search_runtime_response_keeps_unread_inventory(tmp_path,inventory,expected):
    from agents.tools import _format_hit
    from agents.runtime import add_evidence
    from core.evidence import source_coverage
    from models.schemas import SearchResult
    metadata={} if inventory is None else {'attachments':inventory}
    source=SearchResult(email_id='e',chunk_id='c',content='text '*1000,metadata=metadata,score=.9)
    visible=json.loads(bounded_json(normalize_tool_result([_format_hit(source)]),2000))['data'][0]
    ctx=context(tmp_path)
    with use_run_context(ctx): add_evidence([visible])
    response_source=SearchResult(**ctx.evidence[('e','c')])
    coverage=source_coverage(response_source.metadata)
    assert coverage['attachment_inventory_status']==expected
    assert coverage['unread_attachments']==(None if inventory is None else 1)


def test_k1_email_page_chunk_keeps_attachment_coverage():
    from core.evidence_pages import _read_email_page
    page=_read_email_page({'email_id':'e','chunks':[{'email_id':'e','chunk_id':'c','content':'body',
        'metadata':{'attachments':[{'filename':'a.pdf','status':'unread'}]}}]},chunk_id='c')
    assert page['chunks'][0]['coverage']['unread_attachments']==1


def test_history_excerpt_clipping_refreshes_hash_and_end(tmp_path):
    from agents.tools import search_history
    ctx=context(tmp_path)
    with use_run_context(ctx): result=normalize_tool_result(search_history('A-102'))
    clipped=json.loads(bounded_json(result,1900))
    for item in clipped.get('data',{}).get('items',[]):
        for hit in item['hits']:
            assert hit['end']==hit['start']+len(hit['text'])
            assert hit['sha256']==hashlib.sha256(hit['text'].encode()).hexdigest()


def test_job_invalidation_blocks_late_checkpoint_and_finish(tmp_path):
    from core.jobs import JobStore, JobConflict, _encode
    store=JobStore(tmp_path/'jobs.db')
    job,_=store.create('owner','agent',{'session_id':'session','query':'private'})
    other,_=store.create('owner','agent',{'session_id':'other','query':'other'})
    store.claim('owner',job['id'])
    store.checkpoint('owner',job['id'],{'safe':True,'private':'text'})
    assert store.session_jobs('owner','session')[0]['id']==job['id']
    assert store.invalidate_session('owner','session')==[job['id']]
    with pytest.raises(JobConflict): store.checkpoint('owner',job['id'],{'safe':True,'private':'late'})
    store.finish('owner',job['id'],'succeeded',result={'private':'late'})
    current=store.get('owner',job['id'],private=True)
    assert current['request']=={'session_id':'session'} and current['checkpoint'] is None and current['result'] is None
    assert current['resume_block_reason']=='session_invalidated'
    with pytest.raises(JobConflict): store.resume('owner',job['id'])
    assert store.get('owner',other['id'])['status']=='queued'
    with pytest.raises(ValueError): _encode({'text':'x'*2_000_000})


def test_real_loop_stores_compacts_and_resumes_without_tool_replay(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS
    from agents.runtime import RunCancelled
    import agents.agent_loop as loop
    from models.schemas import AgentRequest
    ctx=context(tmp_path)
    checkpoints=[]
    ctx.checkpoint_callback=lambda value: checkpoints.append(copy.deepcopy(value))
    provider_calls=[]
    def create(**kwargs):
        provider_calls.append(copy.deepcopy(kwargs))
        if len(provider_calls)>2:
            raise RunCancelled('synthetic interruption')
        n=len(provider_calls)
        call=NS(id=f'provider-{n}',function=NS(name='search_emails',arguments=json.dumps({'query':f'q{n}'})))
        return NS(choices=[NS(message=NS(content='',tool_calls=[call]),finish_reason='tool_calls')])
    client=NS(chat=NS(completions=NS(create=create)))
    monkeypatch.setattr(loop,'_get_client',lambda:client)
    executions=[]
    monkeypatch.setattr(loop,'call_tool',lambda name,args: executions.append((name,args)) or {'text':'mail '*600})
    request=AgentRequest(query='CURRENT exact 原话',session_id='session')
    with use_run_context(ctx):
        result=loop.run_agent_loop(request,owner_id='owner',session_id='session')
    assert result.metadata['status']=='cancelled' and len(executions)==2
    checkpoint=checkpoints[-1]
    assert checkpoint['version']==2 and checkpoint['safe'] and checkpoint['tool_calls']==2
    assert checkpoint['messages'][1]['content']=='CURRENT exact 原话'
    assert json.loads(checkpoint['messages'][3]['content'])['compaction_policy']==POLICY_VERSION
    assert checkpoint['tool_result_refs']['provider-1']['call_id']=='step-1'
    def finish(**kwargs):
        return NS(choices=[NS(message=NS(content='done',tool_calls=[]),finish_reason='stop')])
    client.chat.completions.create=finish
    ctx.model_usage=None  # resume receives a fresh trusted ledger
    with use_run_context(ctx):
        restored=loop.run_agent_loop(request,owner_id='owner',session_id='session',checkpoint=checkpoint)
    assert restored.answer=='done' and len(executions)==2
    assert checkpoints[-1]['tool_calls']==2 and checkpoints[-1]['counts']==checkpoint['counts']
    ctx.session_repository.set_task_fact('owner','session','budget',200,source_turn_id='turn',expected_version=0,explicit_user=True)
    with use_run_context(ctx),pytest.raises(ToolResultUnavailable,match='task_memory_changed'):
        loop.run_agent_loop(request,owner_id='owner',session_id='session',checkpoint=checkpoint)
    assert len(executions)==2
    ctx.tool_result_store.delete_session('owner','session')
    with use_run_context(ctx), pytest.raises(ToolResultUnavailable):
        loop.run_agent_loop(request,owner_id='owner',session_id='session',checkpoint=checkpoint)
    assert len(executions)==2


def test_loop_storage_failure_remains_observable_without_invented_reference(tmp_path,monkeypatch):
    from types import SimpleNamespace as NS
    import agents.agent_loop as loop
    from models.schemas import AgentRequest
    ctx=context(tmp_path, max_object_bytes=100)
    ctx.context_metrics["parent_summary_degraded"]=True
    calls=[]
    def create(**kwargs):
        calls.append(kwargs)
        tool=NS(id='call',function=NS(name='email_stats',arguments='{}'))
        return NS(choices=[NS(message=NS(content='done' if len(calls)>1 else '',tool_calls=[] if len(calls)>1 else [tool]),finish_reason='stop')])
    monkeypatch.setattr(loop,'_get_client',lambda:NS(chat=NS(completions=NS(create=create))))
    monkeypatch.setattr(loop,'call_tool',lambda name,args:{'text':'x'*1000})
    with use_run_context(ctx):
        result=loop.run_agent_loop(AgentRequest(query='q',session_id='session'),owner_id='owner',session_id='session')
    step=result.metadata['steps'][0]
    assert step['status']=='success' and step['result_storage_error']=='object_byte_limit'
    assert 'result_ref' not in step
    assert result.metadata['context_metrics']['tool_result_storage_failures']==1
    assert result.metadata['context_metrics']['parent_summary_degraded'] is True


def test_runtime_profile_narrows_capacity_and_uses_shared_safety(monkeypatch):
    import config.settings as cfg
    from agents.runtime import ContextBudgetExceeded
    monkeypatch.setattr(cfg,'MODEL_CONTEXT_SAFETY_TOKENS',77)
    monkeypatch.setattr(cfg,'MODEL_CONTEXT_PROFILES',{'test-model':{'context_tokens':400,
        'revision':'r1','stages':{'summarize':{'context_tokens':200},'generate':{'context_tokens':800}}}})
    ctx=RunContext(context_token_limit=500,token_counter=lambda raw:10)
    ctx.check_context([{'role':'user','content':'q'}],model='test-model',stage='summarize',max_output_tokens=20)
    assert ctx.context_metrics['token_capacity']==200
    assert ctx.context_metrics['output_token_reserve']==97
    ctx.check_context([{'role':'user','content':'q'}],model='test-model',stage='generate',max_output_tokens=20)
    assert ctx.context_metrics['token_capacity']==400  # stage cannot enlarge model capacity
    ctx.context_token_limit=100
    with pytest.raises(ContextBudgetExceeded):
        ctx.check_context([{'role':'user','content':'q'}],model='test-model',stage='generate',max_output_tokens=20)
    assert ctx.context_metrics['token_capacity']==100  # model cannot enlarge run capacity
    with pytest.raises(ValueError):
        ctx.check_context([],max_output_tokens=True)
