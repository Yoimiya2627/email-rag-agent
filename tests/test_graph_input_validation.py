"""Self-RAG final-output validation and non-retrieval conversation paths."""
import threading
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest

from agents import graph_workflow as graph, agent_loop as loop
from agents.runtime import RunContext, use_run_context
from models.schemas import AgentRequest, SearchResult, IntentType


def reply(content, finish='stop', reasoning=None):
    return NS(choices=[NS(finish_reason=finish,
                         message=NS(content=content,reasoning_content=reasoning,tool_calls=None))])


def hit():
    return SearchResult(chunk_id='e_0',email_id='e',content='synthetic evidence',score=1,metadata={})


@pytest.mark.parametrize('content,finish,reasoning',[
    ('[0]','length',None), ('[0]','content_filter',None), ('[0]',None,None),
    ('','stop','Maybe [0], but not decided'), ('','length','[0]'),
])
def test_grader_rejects_incomplete_or_reasoning_only_output(content,finish,reasoning):
    with patch.object(graph,'_get_client',return_value=object()), \
         patch.object(graph,'create_completion',return_value=reply(content,finish,reasoning)):
        result=graph.node_grade_contexts({'query':'find the invoice','results':[hit()]})
    assert result['relevant_results']==[]
    assert result['grading_error']=='context_grading_failed'


@pytest.mark.parametrize('content,finish,reasoning',[
    ('changed scope','length',None), ('changed scope','content_filter',None),
    ('changed scope',None,None), ('','stop','rewrite from reasoning'),
])
def test_rewrite_preserves_original_on_incomplete_output(content,finish,reasoning):
    with patch.object(graph.cfg,'ENABLE_QUERY_REWRITE',True), \
         patch.object(graph,'_get_client',return_value=object()), \
         patch.object(graph,'create_completion',return_value=reply(content,finish,reasoning)):
        result=graph.node_rewrite({'query':'original constrained query'})
    assert result['rewritten_query']=='original constrained query'


def test_complete_final_grade_and_rewrite_remain_usable():
    source=hit()
    with patch.object(graph.cfg,'ENABLE_QUERY_REWRITE',True), \
         patch.object(graph,'_get_client',return_value=object()), \
         patch.object(graph,'create_completion',side_effect=[reply('[0]'),reply('resolved invoice')]):
        graded=graph.node_grade_contexts({'query':'invoice','results':[source]})
        rewritten=graph.node_rewrite({'query':'invoice'})
    assert graded['relevant_results']==[source] and graded['grading_error'] is None
    assert rewritten['rewritten_query']=='resolved invoice'


def test_graph_greeting_skips_classification_graph_and_history():
    memory=NS(to_messages=Mock(side_effect=AssertionError('Greeting must not load email history')))
    with patch('agents.coordinator.classify_intent') as classify,patch.object(graph,'get_graph') as build:
        result=graph.run_graph(AgentRequest(query='你好！'),memory)
    assert result.intent==IntentType.GENERAL and result.sources==[]
    assert result.metadata['retrieval_performed'] is False
    classify.assert_not_called()
    build.assert_not_called()


def test_graph_general_classification_is_once_and_never_builds_retrieval_graph():
    history=[{'role':'user','content':'synthetic context'}]
    with patch('agents.coordinator.classify_intent',return_value=IntentType.GENERAL) as classify, \
         patch.object(graph,'get_graph') as build:
        result=graph.run_graph(AgentRequest(query='请解释这个助手的使用范围'),NS(to_messages=lambda:history))
    classify.assert_called_once_with('请解释这个助手的使用范围',history=history)
    build.assert_not_called()
    assert result.sources==[] and result.metadata['retrieval_performed'] is False


def test_graph_mixed_greeting_mail_request_keeps_graph_path():
    engine=NS(invoke=Mock(return_value={'answer':'synthetic summary','relevant_results':[]}))
    with patch('agents.coordinator.classify_intent',return_value=IntentType.SUMMARIZE) as classify, \
         patch.object(graph,'get_graph',return_value=engine):
        result=graph.run_graph(AgentRequest(query='你好，请总结昨天的邮件'))
    assert classify.call_count==1 and engine.invoke.call_count==1
    assert result.answer=='synthetic summary'


def test_fresh_agent_greeting_keeps_identity_checks_but_skips_planner():
    with patch.object(loop,'_get_client') as model,patch.object(loop,'_get_tool_backend') as backend:
        result=loop.run_agent_loop(AgentRequest(query='你好'))
    assert result.sources==[] and result.metadata['actual_tool_calls']==0
    assert result.metadata['retrieval_performed'] is False and result.metadata['run_id']
    model.assert_not_called()
    backend.assert_not_called()
    with use_run_context(RunContext(owner_id='other')):
        with pytest.raises(ValueError,match='identity'):
            loop.run_agent_loop(AgentRequest(query='你好'))


def test_cancelled_agent_greeting_cannot_bypass_cancellation():
    event=threading.Event();event.set()
    with use_run_context(RunContext(cancel_event=event)), \
         patch('agents.general_agent.direct_general_response') as direct:
        result=loop.run_agent_loop(AgentRequest(query='你好'))
    assert result.metadata['status']=='cancelled'
    direct.assert_not_called()


def test_agent_greeting_cannot_bypass_checkpoint_validation():
    with patch.object(loop,'_get_client',return_value=object()), \
         patch.object(loop,'_get_tool_backend',return_value=NS(tool_schemas=lambda:[])), \
         patch('agents.general_agent.direct_general_response') as direct:
        with pytest.raises(ValueError,match='Checkpoint'):
            loop.run_agent_loop(AgentRequest(query='你好'),checkpoint={'safe':False})
    direct.assert_not_called()


def test_agent_mixed_request_uses_planner_without_extra_classification():
    with patch.object(loop,'_get_client',return_value=object()), \
         patch.object(loop,'_get_tool_backend',return_value=NS(tool_schemas=lambda:[])), \
         patch.object(loop,'create_completion',return_value=reply('synthetic answer')) as model, \
         patch('agents.coordinator.classify_intent') as classify:
        result=loop.run_agent_loop(AgentRequest(query='你好，请总结昨天的邮件'))
    assert model.call_count==1 and result.answer=='synthetic answer'
    classify.assert_not_called()
