"""A terminal transport marker must not turn incomplete output into success."""
import json
from unittest.mock import patch
from fastapi.testclient import TestClient
from models.schemas import AgentRequest, AgentResponse, IntentType
from core.model_outcomes import ModelOutputError
from api.sessions import SessionStore
import api.main as api
from frontend.client import StreamAccumulator


def test_incomplete_response_is_visible_but_not_successful_history():
    sessions = SessionStore()
    response = AgentResponse(answer='partial', metadata={'status':'incomplete','completion_status':'incomplete','finish_reason':'length'})
    with patch.object(api,'sessions',sessions), patch.object(api,'route',return_value=response):
        result = TestClient(api.app).post('/chat',json={'query':'q','session_id':'s'})
    assert result.json()['answer'] == 'partial'
    assert result.json()['metadata']['status'] == 'incomplete'
    with sessions.turn('local','s') as memory:
        assert memory.to_messages() == []


def test_empty_stream_does_not_commit_a_turn():
    sessions = SessionStore()
    with patch.object(api,'sessions',sessions), patch('agents.coordinator.classify_intent',return_value=IntentType.RETRIEVE), \
         patch('core.pipeline.retrieve',return_value=[]), patch('core.generator.stream_generate',return_value=iter([])):
        result = TestClient(api.app).post('/chat/stream',json={'query':'q','session_id':'s'})
    assert 'empty_model_response' in result.text
    with sessions.turn('local','s') as memory:
        assert memory.to_messages() == []


def test_truncated_stream_emits_terminal_metadata_and_retains_partial_tokens():
    def partial(*args,**kwargs):
        yield 'half answer'
        raise ModelOutputError('half answer',finish_reason='length')
    sessions = SessionStore()
    with patch.object(api,'sessions',sessions), patch('agents.coordinator.classify_intent',return_value=IntentType.RETRIEVE), \
         patch('core.pipeline.retrieve',return_value=[]), patch('core.generator.stream_generate',side_effect=partial):
        response = TestClient(api.app).post('/chat/stream',json={'query':'q','session_id':'s'})
    stream = StreamAccumulator()
    for line in response.text.splitlines():
        if line.startswith('data: '):
            data = line[6:]
            stream.feed(data if data == '[DONE]' else json.loads(data))
    result = stream.result()
    assert stream.done and result['answer'] == 'half answer'
    assert result['metadata']['status'] == 'incomplete'
    assert result['metadata']['finish_reason'] == 'length'
    with sessions.turn('local','s') as memory:
        assert memory.to_messages() == []


def test_lazy_graph_import_failure_is_503():
    with patch('agents.graph_workflow.run_graph',side_effect=ImportError('synthetic')):
        response = TestClient(api.app).post('/chat/graph',json={'query':'q'})
    assert response.status_code == 503


def test_disconnected_ui_stream_keeps_text_but_is_incomplete():
    stream = StreamAccumulator()
    stream.feed({'token':'partial'})
    result = stream.result()
    assert result['answer'] == 'partial'
    assert result['metadata']['status'] == 'incomplete'
    assert 'warning' in result
