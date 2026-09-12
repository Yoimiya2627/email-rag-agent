import copy
import json
import threading
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch

import pytest
import agents.agent_loop as loop
from agents.execution_scope import ExecutionScope,use_execution_scope
from agents.runtime import RunContext,use_run_context
from models.schemas import AgentRequest


def reply(text='done',calls=None,finish='stop'):
    return NS(choices=[NS(message=NS(content=text,tool_calls=calls),finish_reason=finish)])


def client(sequence):
    return NS(chat=NS(completions=NS(create=Mock(side_effect=sequence))))


def setup_backend():
    return NS(tool_schemas=lambda:[{'type':'function','function':{'name':'send_email',
        'parameters':{'type':'object','properties':{},'additionalProperties':False}}}],
        call_tool=Mock(return_value={'status':'pending_approval','approval_id':'approval'}))


def run(request,model,backend,snapshots,**kwargs):
    context=RunContext(session_id='s',checkpoint_callback=lambda value:snapshots.append(copy.deepcopy(value)))
    with patch.object(loop,'_get_client',return_value=model),patch.object(loop,'_get_tool_backend',return_value=backend), \
         use_run_context(context),use_execution_scope(ExecutionScope('local',operation_key='stable')):
        return loop.run_agent_loop(request,session_id='s',**kwargs)


def test_resume_after_complete_tool_round_does_not_replay_approval():
    req=AgentRequest(query='draft',session_id='s',operation_key='stable')
    backend=setup_backend()
    call=NS(id='call-1',function=NS(name='send_email',arguments='{}'))
    snapshots=[]
    first=run(req,client([reply('',[call],'tool_calls'),TimeoutError('synthetic')]),backend,snapshots)
    assert first.metadata['status']=='timeout'
    checkpoint=snapshots[-1]
    assert checkpoint['safe'] and checkpoint['tool_calls']==1
    assert checkpoint['model_metrics']['call_count']==2
    second=run(req,client([reply('Waiting for approval')]),backend,[],checkpoint=checkpoint)
    assert second.metadata['status']=='approval_required'
    assert backend.call_tool.call_count==1
    assert second.metadata['pending_approval_ids']==['approval']
    assert second.metadata['model_usage']['call_count']==3


def test_uncertain_tool_cannot_be_resumed_from_older_safe_boundary():
    req=AgentRequest(query='draft',session_id='s')
    backend=setup_backend()
    backend.call_tool.side_effect=RuntimeError('unknown external result')
    call=NS(id='call-1',function=NS(name='send_email',arguments='{}'))
    snapshots=[]
    with pytest.raises(RuntimeError):
        run(req,client([reply('',[call],'tool_calls')]),backend,snapshots)
    assert snapshots[-1]['safe'] is False
    with pytest.raises(ValueError):
        run(req,client([]),backend,[],checkpoint=snapshots[-1])
    assert backend.call_tool.call_count==1


def test_truncated_answer_continuation_has_no_tools_and_keeps_prefix():
    req=AgentRequest(query='read',session_id='s')
    backend=setup_backend()
    snapshots=[]
    first=run(req,client([reply('prefix ',finish='length')]),backend,snapshots)
    assert first.metadata['completion_status']=='incomplete'
    resumed_client=client([reply('suffix')])
    result=run(req,resumed_client,backend,[],checkpoint=snapshots[-1])
    assert result.answer=='prefix\nsuffix'
    assert 'tools' not in resumed_client.chat.completions.create.call_args.kwargs
    backend.call_tool.assert_not_called()


def test_pre_cancelled_run_starts_no_model_or_tool():
    event=threading.Event();event.set()
    model=client([])
    with patch.object(loop,'_get_client',return_value=model), \
         use_run_context(RunContext(session_id='s',cancel_event=event)):
        result=loop.run_agent_loop(AgentRequest(query='q',session_id='s'),session_id='s')
    assert result.metadata['status']=='cancelled'
    model.chat.completions.create.assert_not_called()
