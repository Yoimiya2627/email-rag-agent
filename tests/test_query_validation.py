"""Reject blank requests before retrieval, generation, session writes or queuing."""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import api.main as api
from models.schemas import AgentRequest, QueryRequest


@pytest.mark.parametrize('endpoint', ['/chat', '/chat/stream', '/query', '/jobs/agent'])
@pytest.mark.parametrize('query', ['', ' ', '\t\r\n', '\u3000\u00a0\u2003'])
def test_blank_query_rejected_before_execution(endpoint, query):
    with patch.object(api, '_prepare_session_context') as prepare, \
         patch.object(api, 'route') as route, \
         patch.object(api, '_job_manager') as jobs, \
         patch('core.pipeline.retrieve') as retrieve, \
         patch('core.generator.generate_answer') as generate, \
         patch('agents.coordinator.classify_intent') as classify:
        response = TestClient(api.app).post(endpoint, json={'query': query})
        assert response.status_code == 422
        assert response.json()['detail'][0]['loc'] == ['body', 'query']
        for operation in (prepare, route, jobs, retrieve, generate, classify):
            operation.assert_not_called()


@pytest.mark.parametrize('model', [AgentRequest, QueryRequest])
@pytest.mark.parametrize('query', ['你好', ' \t保留原文\n\u3000', '  hello  '])
def test_nonblank_query_keeps_original_text(model, query):
    assert model(query=query).query == query
