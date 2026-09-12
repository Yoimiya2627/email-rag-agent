"""Synthetic deterministic context contracts, selection and call-boundary checks."""
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import config.settings as cfg
from agents.runtime import ContextBudgetExceeded, RunContext, use_run_context
from core.context_budget import measure_context, register_token_counter, unregister_token_counter
from core.context_contracts import TrustedScope
from core.memory import ConversationMemory, build_model_messages, conversation_pairs
from core.model_clients import create_completion
from core.session_context import assemble_session_context


def test_adapter_requires_matching_model_revision_stage_and_golden_counts():
    with pytest.raises(ValueError):
        register_token_counter('m', 'r', 'intent', len, validation_samples=[('abc', 4)])
    register_token_counter('m', 'r', 'intent', len, validation_samples=[('中英', 2)])
    try:
        exact = measure_context([{'role':'user','content':'中文'}], model='m', model_revision='r', stage='intent')
        fallback = measure_context([{'role':'user','content':'中文'}], model='m', model_revision='wrong', stage='intent')
        assert exact['token_estimation_method'] == 'validated_model_adapter'
        assert fallback['token_estimation_method'] == 'utf8_bytes_upper_bound'
        assert exact['estimated_input_tokens'] < fallback['estimated_input_tokens']
    finally:
        unregister_token_counter('m', 'r', 'intent')


def test_required_constraint_omission_stops_before_call_and_scope_cannot_be_forged():
    facts = [{'key':'limit','value':'x'*2000,'source_turn_id':'t','version':1}]
    context = assemble_session_context(facts, char_limit=500, token_limit=0, scope=TrustedScope('a','s'))
    assert context['required_omissions'][0]['reason'] == 'required_material_over_budget'
    with use_run_context(RunContext(task_context=context)), pytest.raises(ContextBudgetExceeded):
        build_model_messages('system','original request')
    forged = assemble_session_context([{**facts[0], 'owner_id':'b'}], scope=TrustedScope('a','s'))
    assert forged['required_omissions'][0]['reason'] == 'scope_mismatch'
    with pytest.raises(ValueError):
        assemble_session_context(scope={'owner_id':'b'})


def test_hit_window_hash_and_assistant_status_remain_exact():
    text = 'prefix'*100 + '订单A-102不延期' + 'tail'*100
    start, end = 600, 610
    hit = {'field':'answer','start':start,'end':end,'text':text[start:end],
           'sha256':hashlib.sha256(text[start:end].encode()).hexdigest()}
    row = {'turn_id':'t','query':'q','answer':text,'hits':[hit],'metadata':{'status':'error'}}
    context = assemble_session_context(history_matches=[row], char_limit=4000, token_limit=0)
    shown = json.loads(context['text'])['historical_excerpts'][0]
    assert shown['assistant_claim_not_source_evidence'] == text[start:end]
    assert shown['status'] == 'error'
    ref = next(x for x in context['material_manifest'] if x['kind']=='historical_assistant_claim')
    assert (ref['visible_start'],ref['visible_end'],ref['visible_hash']) == (start,end,hit['sha256'])


def test_current_request_and_recent_identity_dedup_without_losing_evicted_hit():
    memory = ConversationMemory()
    memory.append_turn('old','answer',turn_id='t')
    memory.append_turn('old','answer',turn_id='t')
    assert len(conversation_pairs(memory.to_messages())) == 2
    context = assemble_session_context(history_matches=[{'turn_id':'t','query':'old','answer':'answer'}], token_limit=0)
    with use_run_context(RunContext(task_context=context)):
        sent = build_model_messages('system','  current\nraw  ',memory.to_messages())
    assert '"turn_id":"t"' not in sent[-1]['content']
    assert sent[-1]['content'].endswith('  current\nraw  ')
    long_memory = ConversationMemory()
    long_memory.append_turn('old'*400, 'answer'*400, turn_id='t')
    with use_run_context(RunContext(task_context=context, context_char_limit=1500)):
        sent = build_model_messages('system','current',long_memory.to_messages())
    assert '"turn_id":"t"' in sent[-1]['content']
    assert len(sent) == 2


def test_real_output_and_tools_checked_before_provider_and_usage_error_observable():
    provider = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=Mock(return_value=
        SimpleNamespace(usage={'prompt_tokens':7,'completion_tokens':2,'total_tokens':9})))))
    with use_run_context(RunContext(context_token_limit=300, output_token_reserve=0)), pytest.raises(ContextBudgetExceeded):
        create_completion(provider, stage='intent', model='unknown', messages=[{'role':'user','content':'q'}], max_tokens=300)
    provider.chat.completions.create.assert_not_called()
    with use_run_context(RunContext(context_token_limit=300, output_token_reserve=0)), pytest.raises(ContextBudgetExceeded):
        create_completion(provider, stage='intent', model='unknown', messages=[{'role':'user','content':'q'}],
                          tools=[{'description':'x'*500}], max_tokens=1)
    run = RunContext(context_token_limit=3000)
    with use_run_context(run):
        create_completion(provider, stage='intent', model='unknown', messages=[{'role':'user','content':'q'}], max_tokens=10)
    assert run.context_metrics['provider_input_tokens'] == 7
    assert isinstance(run.context_metrics['input_estimate_error_tokens'], int)


def test_model_stage_capacity_can_only_narrow_and_current_event_not_duplicated():
    with patch.object(cfg,'MODEL_CONTEXT_PROFILES', {'m':{'context_tokens':1000,'stages':{'intent':{'context_tokens':200}}}}, create=True):
        with use_run_context(RunContext(context_token_limit=4000)), pytest.raises(ContextBudgetExceeded):
            build_model_messages('system','q'*100, model='m',stage='intent',max_output_tokens=10)
    context = assemble_session_context(user_events=[{'event_id':'e','text':'raw current','revision':1}], current_request='raw current')
    assert context['text'] == ''
    assert context['omissions'][0]['reason']=='current_request_duplicate'


def test_summary_model_view_preserves_original_source_uncertainty():
    summary={'summary_id':'s','sections':{'goals':[{'text':'核对订单','source_turn_ids':['t'],'source_quotes':[]}]},
             'source_statuses':[{'turn_id':'t','status':'partial','completion_status':'incomplete',
                                'coverage':{'attachment_inventory_status':'unknown','unread_attachments':2,'partial':True}}]}
    context=assemble_session_context(summary=summary)
    item=json.loads(context['text'])['derived_summaries'][0]['summary']['sections']['goals'][0]
    assert item['source_states'][0]['coverage']['unread_attachments']==2
    assert item['source_states'][0]['coverage']['attachment_inventory_status']=='unknown'
    assert item['source_states'][0]['completion_status']=='incomplete'
    assert not context['required_omissions']


def test_generator_propagates_unknown_inventory_and_unread_attachments():
    from core.generator import build_context
    from models.schemas import SearchResult
    for metadata,expected in [({},'unknown'), ({'attachments':[{'status':'unread'}]},'available')]:
        text,refs = build_context([SearchResult(email_id='e',chunk_id='c',content='body',score=1,metadata=metadata)],return_references=True)
        assert '"attachment_inventory_status": "'+expected+'"' in text
        assert refs[0]['visible_hash']==hashlib.sha256(b'body').hexdigest()


def test_partial_hit_does_not_erase_summary_of_unseen_tail():
    text = 'prefix'*100 + 'critical future decision'
    row = {'turn_id':'t','query':'q','answer':text}
    summary = {'summary_id':'s','sections':{'decisions':[{'text':'critical future decision',
        'source_turn_ids':['t'], 'source_quotes':[{'turn_id':'t','field':'answer','start':600,'end':len(text),
         'text':text[600:],'sha256':hashlib.sha256(text[600:].encode()).hexdigest()}]}]}}
    context = assemble_session_context(history_matches=[row], summary=summary,token_limit=0,char_limit=5000)
    assert 'critical future decision' in context['text']


def test_llm_rerank_uses_actual_output_budget_and_preserves_constraint():
    from core import reranker
    from models.schemas import SearchResult
    provider = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=Mock())))
    rows = [SearchResult(email_id='e',chunk_id='c',content='body',score=1,metadata={})]
    with use_run_context(RunContext(context_token_limit=3000)), patch.object(reranker,'_get_client',return_value=provider), pytest.raises(ContextBudgetExceeded):
        reranker._rerank_with_llm('current',rows,1)
    provider.chat.completions.create.assert_not_called()


def test_normal_prior_events_and_empty_task_state_cannot_exhaust_required_budget():
    events = [{'event_id':str(i),'event_type':'current_user_request','text':'普通聊天'*100,'revision':i} for i in range(5)]
    context = assemble_session_context(task_state={'task_id':'default','revision':0}, user_events=events)
    assert not context['required_omissions']
    assert 'task_state' not in json.loads(context['text'])
    with use_run_context(RunContext(task_context=context)):
        messages = build_model_messages('system','current',stage='intent',max_output_tokens=1500)
    assert messages[-1]['content'].endswith('current')


def test_purpose_soft_quota_reselection_and_unused_quota_borrowing():
    from core.session_context import context_for_stage
    events = [{'event_id':'e','event_type':'current_user_request','text':'e'*180,'revision':1}]
    history = [{'turn_id':'t','query':'h'*150,'answer':'a'*150}]
    weights = {'intent':{'user_events':1.0},'generate':{'historical_excerpts':1.0}}
    with patch.object(cfg,'CONTEXT_PURPOSE_WEIGHTS',weights,create=True):
        context = assemble_session_context(user_events=events,history_matches=history,char_limit=650,token_limit=0)
        intent = context_for_stage(context,'intent')
        assert 'user_events' in json.loads(intent['text'])
        assert json.loads(context['text'])['historical_excerpts']
        borrowed = assemble_session_context(user_events=events,char_limit=650,token_limit=0)
        assert json.loads(borrowed['text'])['user_events']
    assert 'selection_inputs' not in json.dumps(context)


def test_only_backend_marked_effective_corrections_are_required():
    event = {'event_id':'e','event_type':'current_user_correction','text':'更正预算'*1000,'revision':1,
             'requires_protection':False, 'status':'unresolved'}
    assert not assemble_session_context(user_events=[event])['required_omissions']
    event.update(requires_protection=True,status='active')
    assert assemble_session_context(user_events=[event])['required_omissions']
    event.update(status='superseded')
    assert not assemble_session_context(user_events=[event])['required_omissions']


@pytest.mark.parametrize('failure', ['negative','wrong_type','raises'])
def test_registered_adapter_runtime_failure_falls_back_without_weakening_budget(failure):
    def counter(raw):
        if raw == 'ok':
            return 2
        if failure == 'raises':
            raise RuntimeError('synthetic counter failure')
        return -1 if failure == 'negative' else 'invalid'
    register_token_counter('broken','r','intent',counter,validation_samples=[('ok',2)])
    try:
        measured = measure_context([{'role':'user','content':'真实输入'}],model='broken',model_revision='r',stage='intent')
        conservative = measure_context([{'role':'user','content':'真实输入'}])
        assert measured['estimated_input_tokens'] == conservative['estimated_input_tokens']
        assert measured['token_counter_degraded'] == ('RuntimeError' if failure == 'raises' else 'ValueError')
    finally:
        unregister_token_counter('broken','r','intent')


def test_stage_reselection_preserves_repository_required_omission_markers():
    from core.session_context import context_for_stage
    context=assemble_session_context()
    context['external_required_omissions']=[{'material_id':'user_events','reason':'active_user_event_count_limit'}]
    selected=context_for_stage(context,'intent')
    assert selected['required_omissions']==context['external_required_omissions']
    with use_run_context(RunContext(task_context=context)),pytest.raises(ContextBudgetExceeded):
        build_model_messages('system','current',stage='intent')
