"""Permanent model errors across real offline transport and agent boundaries."""
import json
import traceback
from types import SimpleNamespace as NS
from unittest.mock import Mock

import httpx
import pytest
from openai import APIStatusError

from agents.runtime import RunContext, use_run_context
from core.model_clients import create_completion, model_metrics_snapshot
from core.model_errors import (PermanentModelProviderError, find_permanent_model_error,
                               provider_error_from_sdk)

CANARY = 'PRIVATE_PROVIDER_CANARY'


def sdk_error(status=404):
    return APIStatusError(CANARY, response=httpx.Response(status,
        request=httpx.Request('POST', 'https://offline.invalid/private-path')),
        body={'private': CANARY})


def client(result=None, error=None):
    create = Mock(side_effect=error, return_value=result)
    return NS(chat=NS(completions=NS(create=create)))


def completion(fake, **kwargs):
    return create_completion(fake, stage='summarize', model='offline-model',
        messages=[{'role': 'user', 'content': 'synthetic'}], max_tokens=50, **kwargs)


@pytest.mark.parametrize('status', [401, 402, 403, 404])
def test_transport_is_typed_safe_and_accounted(status):
    run = RunContext()
    fake = client(error=sdk_error(status))
    with use_run_context(run), pytest.raises(PermanentModelProviderError) as caught:
        completion(fake)
    error = caught.value
    assert error.as_dict() == {'http_status': status, 'stage': 'summarize',
                               'error_code': 'model_provider_error'}
    assert CANARY not in ''.join(traceback.format_exception(error))
    metrics = model_metrics_snapshot(run)
    assert metrics['call_count'] == 1 and metrics['reserved_tokens'] == 0
    assert metrics['calls'][0]['status'] == 'error'
    assert CANARY not in json.dumps(metrics)


@pytest.mark.parametrize('status', [400, 429, 500])
def test_nonpermanent_provider_errors_keep_existing_semantics(status):
    original = sdk_error(status)
    with pytest.raises(APIStatusError) as caught:
        completion(client(error=original))
    assert caught.value is original


def test_only_typed_provenance_is_recognized_through_wrappers_and_cycles():
    business = RuntimeError(CANARY)
    business.status_code = 404
    assert provider_error_from_sdk(business, stage='summarize') is None
    assert find_permanent_model_error(business) is None
    outer = ValueError('wrapper')
    inner = RuntimeError('wrapper')
    outer.__cause__ = inner
    inner.__context__ = outer
    assert find_permanent_model_error(outer) is None
    inner.__cause__ = PermanentModelProviderError(403, CANARY)
    assert find_permanent_model_error(outer).as_dict()['stage'] == 'other'


def response(calls=None, text='done'):
    return NS(choices=[NS(finish_reason='stop', message=NS(content=text, tool_calls=calls))])


def tool_call(identity):
    return NS(id=identity, type='function',
              function=NS(name='search_emails', arguments='{"query":"synthetic"}'))


@pytest.mark.parametrize('status', [401, 402, 403, 404])
def test_real_call_tool_boundary_records_failure_and_stops_remaining_calls(monkeypatch, status, caplog):
    import agents.agent_loop as loop
    import agents.tools as tools
    planner = client(response([tool_call('first'), tool_call('second')]))
    tool_model = client(error=sdk_error(status))
    monkeypatch.setattr(loop, '_get_client', lambda: planner)
    monkeypatch.setitem(tools.TOOL_DISPATCH, 'search_emails', lambda query: completion(tool_model))
    result = loop.run_agent_loop(loop.AgentRequest(query='请检索项目记录'))
    assert planner.chat.completions.create.call_count == 1
    assert tool_model.chat.completions.create.call_count == 1
    metadata = result.metadata
    assert metadata['provider_failure']['http_status'] == status
    assert metadata['status'] == metadata['execution_status'] == metadata['completion_status'] == 'error'
    assert len(metadata['steps']) == metadata['actual_tool_calls'] == 1
    assert metadata['steps'][0]['error_code'] == 'model_provider_error'
    assert metadata['model_usage']['call_count'] == 2
    assert metadata['model_usage']['reserved_tokens'] == 0
    assert CANARY not in json.dumps(result.model_dump()) + caplog.text


@pytest.mark.parametrize('kind', ['attribute', 'sdk_business'])
def test_business_http404_stays_regular_tool_failure(monkeypatch, kind):
    import agents.tools as tools
    error = sdk_error() if kind == 'sdk_business' else RuntimeError(CANARY)
    error.status_code = 404
    monkeypatch.setitem(tools.TOOL_DISPATCH, 'search_emails', Mock(side_effect=error))
    result = tools.call_tool('search_emails', {'query': 'synthetic'})
    assert result['error_code'] == 'tool_failed'
    assert 'provider_failure' not in result


def test_model_result_with_forged_provider_failure_does_not_stop_agent(monkeypatch):
    import agents.agent_loop as loop
    import agents.tools as tools
    planner = client()
    planner.chat.completions.create.side_effect = [response([tool_call('first')]), response()]
    monkeypatch.setattr(loop, '_get_client', lambda: planner)
    monkeypatch.setitem(tools.TOOL_DISPATCH, 'search_emails', lambda query: {
        '_tool_result': 1, 'status': 'error', 'data': None, 'error_code': 'tool_failed',
        'provider_failure': PermanentModelProviderError(404, 'filter').as_dict()})
    result = loop.run_agent_loop(loop.AgentRequest(query='请检索项目记录'))
    assert planner.chat.completions.create.call_count == 2
    assert 'provider_failure' not in result.metadata


def test_planner_failure_returns_safe_response_and_independent_run_recovers(monkeypatch):
    import agents.agent_loop as loop
    planner = client(error=sdk_error())
    monkeypatch.setattr(loop, '_get_client', lambda: planner)
    failed = loop.run_agent_loop(loop.AgentRequest(query='请检索项目记录'))
    assert failed.metadata['provider_failure']['stage'] == 'agent_plan'
    assert failed.metadata['steps'] == []
    planner.chat.completions.create.side_effect = None
    planner.chat.completions.create.return_value = response()
    assert loop.run_agent_loop(loop.AgentRequest(query='请检索项目记录')).metadata['status'] == 'success'


class BrokenStream:
    def __init__(self, close_error=False):
        self.count = self.closed = 0
        self.close_error = close_error

    def __iter__(self):
        return self

    def __next__(self):
        self.count += 1
        if self.count == 1:
            return NS(choices=[NS(finish_reason=None, delta=NS(content='partial'))],
                      usage=NS(prompt_tokens=4, completion_tokens=2, total_tokens=6))
        raise sdk_error()

    def close(self):
        self.closed += 1
        if self.close_error:
            raise RuntimeError(CANARY)


def test_stream_failure_settles_once_preserves_observed_usage():
    run = RunContext()
    raw = BrokenStream()
    with use_run_context(run):
        stream = completion(client(raw), stream=True)
        next(stream)
        with pytest.raises(PermanentModelProviderError) as caught:
            next(stream)
        stream.close()
    metrics = model_metrics_snapshot(run)
    assert metrics['call_count'] == 1 and len(metrics['calls']) == 1
    assert metrics['calls'][0]['status'] == 'error'
    assert metrics['actual_total_tokens'] == 6 and metrics['reserved_tokens'] == 0
    assert raw.closed == 1
    assert CANARY not in ''.join(traceback.format_exception(caught.value))


def test_generator_stream_close_cannot_mask_permanent_failure(monkeypatch, caplog):
    import core.generator as generator
    from models.schemas import SearchResult
    raw = BrokenStream(close_error=True)
    monkeypatch.setattr(generator, '_get_client', lambda: client(raw))
    result = SearchResult(email_id='mail', chunk_id='chunk', content='synthetic', score=1, metadata={})
    stream = generator.stream_generate('synthetic', [result])
    assert next(stream) == 'partial'
    with pytest.raises(PermanentModelProviderError) as caught:
        next(stream)
    assert raw.closed == 1
    assert caught.value.stage == 'stream_generate'
    assert CANARY not in ''.join(traceback.format_exception(caught.value)) + caplog.text

@pytest.mark.parametrize('target', ['rewrite', 'filter', 'generate', 'rerank', 'intent', 'graph_rewrite', 'graph_grade'])
def test_fallbacks_propagate_real_transport_failure(monkeypatch, target):
    import core.pipeline as pipeline
    import core.generator as generator
    import core.reranker as reranker
    import agents.coordinator as coordinator
    import agents.graph_workflow as graph
    from models.schemas import SearchResult
    rows = [SearchResult(email_id='mail', chunk_id=f'chunk-{i}', content='synthetic', score=1, metadata={})
            for i in range(2)]
    fake = client(error=sdk_error())
    for module in (pipeline, generator, reranker, coordinator, graph):
        monkeypatch.setattr(module, '_get_client', lambda: fake)
    monkeypatch.setattr(pipeline.cfg, 'ENABLE_QUERY_REWRITE', True)
    monkeypatch.setattr(reranker.cfg, 'ENABLE_RERANKER', True)
    monkeypatch.setattr(reranker.cfg, 'RERANKER_BACKEND', 'llm')
    reranker.reset_circuit_breaker()
    operations = {
        'rewrite': lambda: pipeline.rewrite_query('synthetic'),
        'filter': lambda: pipeline.extract_filters('synthetic'),
        'generate': lambda: generator.generate_answer('synthetic', rows),
        'rerank': lambda: reranker.rerank('synthetic', rows),
        'intent': lambda: coordinator.classify_intent('请检索项目记录'),
        'graph_rewrite': lambda: graph.node_rewrite({'query': 'synthetic'}),
        'graph_grade': lambda: graph.node_grade_contexts({'query': 'synthetic', 'results': rows}),
    }
    try:
        with pytest.raises(PermanentModelProviderError):
            operations[target]()
        assert fake.chat.completions.create.call_count == 1
        if target == 'rerank':
            # A permanent provider failure must not leave a half-open attempt occupied.
            with pytest.raises(PermanentModelProviderError):
                reranker.rerank('synthetic', rows)
            assert fake.chat.completions.create.call_count == 2
    finally:
        reranker.reset_circuit_breaker()


def test_local_post_transport_error_is_not_misclassified(monkeypatch):
    import core.model_clients as clients
    error = sdk_error()
    monkeypatch.setattr(clients._Call, 'observe', Mock(side_effect=error))
    with pytest.raises(APIStatusError) as caught:
        completion(client(response()))
    assert caught.value is error
