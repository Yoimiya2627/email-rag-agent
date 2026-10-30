"""Greetings and clarification cannot implicitly read an indexed mailbox."""
import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import api.main as api
from api.sessions import SessionStore
from models.schemas import IntentType


def streamed(response):
    events = [json.loads(line[6:]) for line in response.text.splitlines()
              if line.startswith('data: ') and line != 'data: [DONE]']
    assert 'data: [DONE]' in response.text
    terminal = next(row for row in reversed(events) if 'metadata' in row)
    return {'answer': ''.join(row.get('token', '') for row in events), **terminal}


@pytest.mark.parametrize('endpoint', ['/chat', '/chat/stream', '/chat/graph', '/chat/agent'])
@pytest.mark.parametrize('prior', [False, True])
@pytest.mark.parametrize('query', ['你好！', '谢谢', '你能做什么？'])
def test_social_chat_skips_mail_model_and_derived_context(endpoint, prior, query, tmp_path):
    sessions = SessionStore(path=tmp_path/'sessions.sqlite3')
    if prior:
        with sessions.turn('local', 's') as memory:
            memory.append_turn('总结预算邮件', '敏感历史预算987654，来源[email_1#chunk_0]')
    with patch.object(api, 'sessions', sessions), \
         patch.object(api, '_prepare_session_context', side_effect=AssertionError('No derived context for social turns')), \
         patch('core.pipeline.retrieve', side_effect=AssertionError('No mailbox retrieval')) as retrieve, \
         patch('core.model_clients.create_completion', side_effect=AssertionError('No provider request')) as model:
        response = TestClient(api.app).post(endpoint, json={'query':query, 'session_id':'s'})
    assert response.status_code == 200
    result = streamed(response) if endpoint.endswith('/stream') else response.json()
    assert result['answer'] and '987654' not in result['answer'] and 'email_1' not in result['answer']
    assert result['sources'] == []
    assert result['metadata']['model_visible_evidence'] == []
    assert result['metadata']['retrieval_performed'] is False
    assert result['metadata']['model_usage']['call_count'] == 0
    assert result['metadata']['status'] == 'success'
    assert len(sessions.history('local','s')['turns']) == (2 if prior else 1)
    retrieve.assert_not_called()
    model.assert_not_called()


@pytest.mark.parametrize('endpoint', ['/chat', '/chat/stream'])
def test_unclear_general_turn_requests_details_without_retrieval(endpoint, tmp_path):
    sessions = SessionStore(path=tmp_path/'sessions.sqlite3')
    with patch.object(api, 'sessions', sessions), \
         patch('agents.coordinator.classify_intent', return_value=IntentType.GENERAL), \
         patch('core.pipeline.retrieve', side_effect=AssertionError('No mail request')) as retrieve:
        response = TestClient(api.app).post(endpoint, json={'query':'帮我处理一下', 'session_id':'s'})
    assert response.status_code == 200
    result = streamed(response) if endpoint.endswith('/stream') else response.json()
    assert '请具体说明' in result['answer'] and result['sources'] == []
    assert result['metadata']['retrieval_performed'] is False
    retrieve.assert_not_called()


@pytest.mark.parametrize('endpoint', ['/chat', '/chat/stream'])
def test_context_preparation_failure_keeps_original_error_and_failed_history(endpoint, tmp_path):
    sessions = SessionStore(path=tmp_path/'sessions.sqlite3')
    with patch.object(api, 'sessions', sessions), \
         patch.object(api, '_prepare_session_context', side_effect=TimeoutError('synthetic timeout')), \
         patch.object(api, 'route') as runner:
        response = TestClient(api.app).post(endpoint, json={'query':'查找项目邮件', 'session_id':'s'})
    if endpoint.endswith('/stream'):
        result = streamed(response)
        assert 'timed out' in result['error']
        assert result['metadata']['status'] == 'error'
        assert result['metadata']['session_context'] == {}
    else:
        assert response.status_code == 504
    turns = sessions.history('local','s')['turns']
    assert len(turns) == 1 and turns[0]['metadata']['status'] == 'error'
    with sessions.turn('local','s') as memory:
        assert memory.to_messages() == []
    runner.assert_not_called()
