"""Offline client lifecycle/accounting regressions: no credentials or network."""
import json
import copy
import runpy
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

import config.settings as cfg
from agents.runtime import RunCancelled, RunContext, use_run_context
from core.model_clients import (ModelBudgetExceeded, close_model_clients,
                                create_completion, get_model_client, model_metrics_snapshot,
                                ensure_model_usage, restore_model_metrics)


class Client:
    def __init__(self, result=None):
        self.chat = NS(completions=NS(create=Mock(return_value=result or NS(choices=[]))))
        self.closed = 0

    def close(self):
        self.closed += 1


@pytest.fixture(autouse=True)
def isolated_clients(monkeypatch):
    close_model_clients()
    monkeypatch.setattr(cfg, "DEEPSEEK_API_KEY", "test-only-placeholder")
    monkeypatch.setattr(cfg, "DEEPSEEK_BASE_URL", "https://offline.invalid")
    monkeypatch.setattr(cfg, "DEEPSEEK_MODEL", "offline-model")
    monkeypatch.setattr(cfg, "MODEL_TOKEN_PRICES", {}, raising=False)
    for name, value in {"MODEL_RUN_TOKEN_LIMIT": 0, "MODEL_RUN_COST_LIMIT": 0,
                        "MODEL_INPUT_COST_PER_MILLION": None,
                        "MODEL_OUTPUT_COST_PER_MILLION": None}.items():
        monkeypatch.setattr(cfg, name, value, raising=False)
    yield
    close_model_clients()


def call(client, **kwargs):
    return create_completion(client, stage="generate", model="offline-model",
                             messages=[{"role": "user", "content": "private mail text"}],
                             max_tokens=100, **kwargs)


@pytest.mark.parametrize('stream',[False,True])
@pytest.mark.parametrize('mode',['disabled','enabled'])
@pytest.mark.parametrize('model',['deepseek-v4-flash','deepseek-v4-pro'])
def test_official_v4_thinking_mode_applies_to_stream_and_nonstream(monkeypatch,stream,mode,model):
    monkeypatch.setattr(cfg,'DEEPSEEK_THINKING_MODE',mode)
    client=Client(Stream([]) if stream else None)
    client.base_url='https://api.deepseek.com/v1/'
    result=create_completion(client,model=model,messages=[{'role':'user','content':'synthetic'}],max_tokens=50,stream=stream)
    if stream:
        assert list(result)==[]
    assert client.chat.completions.create.call_args.kwargs['extra_body']=={'thinking':{'type':mode}}


@pytest.mark.parametrize('endpoint,model',[
    ('https://other.example/v1/','deepseek-v4-flash'),
    ('https://api.deepseek.com.other.example/','deepseek-v4-flash'),
    ('https://api.deepseek.com@other.example/','deepseek-v4-flash'),
    ('http://api.deepseek.com/','deepseek-v4-flash'),
    ('https://api.deepseek.com:8443/','deepseek-v4-flash'),
    ('https://api.deepseek.com/proxy/','deepseek-v4-flash'),
    ('https://api.deepseek.com/?route=other','deepseek-v4-flash'),
    ('https://api.deepseek.com/','deepseek-chat'),
    ('https://api.deepseek.com/','gpt-fixture'),
    (None,'deepseek-v4-flash'),
])
def test_thinking_extension_never_leaks_to_other_client_or_model(monkeypatch,endpoint,model):
    monkeypatch.setattr(cfg,'DEEPSEEK_BASE_URL','https://api.deepseek.com')
    client=Client()
    if endpoint is not None:
        client.base_url=endpoint
    create_completion(client,model=model,messages=[],max_tokens=50)
    assert 'extra_body' not in client.chat.completions.create.call_args.kwargs


@pytest.mark.parametrize('explicit',[True,False])
def test_thinking_body_preserves_explicit_choice_and_never_mutates_caller(monkeypatch,explicit):
    monkeypatch.setattr(cfg,'DEEPSEEK_THINKING_MODE','disabled')
    client=Client()
    client.base_url='https://api.deepseek.com/'
    body={'other':{'nested':[1]}}
    if explicit:
        body['thinking']={'type':'enabled'}
    original=copy.deepcopy(body)
    create_completion(client,model='deepseek-v4-flash',messages=[],max_tokens=50,extra_body=body)
    sent=client.chat.completions.create.call_args.kwargs['extra_body']
    assert sent['thinking']=={'type':'enabled' if explicit else 'disabled'}
    assert sent['other']==body['other'] and body==original
    sent['other']['nested'].append(2)
    sent['thinking']['type']='changed'
    assert body==original


def test_extra_body_model_override_is_respected():
    client=Client()
    client.base_url='https://api.deepseek.com/'
    create_completion(client,model='deepseek-v4-flash',messages=[],max_tokens=50,extra_body={'model':'other-model'})
    assert client.chat.completions.create.call_args.kwargs['extra_body']=={'model':'other-model'}


@pytest.mark.parametrize('mode',[None,'enabled','disabled','invalid'])
def test_thinking_mode_settings_validation_without_loading_env(monkeypatch,mode):
    import dotenv
    monkeypatch.setattr(dotenv,'load_dotenv',lambda *args,**kwargs:False)
    if mode is None:
        monkeypatch.delenv('DEEPSEEK_THINKING_MODE',raising=False)
    else:
        monkeypatch.setenv('DEEPSEEK_THINKING_MODE',mode)
    if mode=='invalid':
        with pytest.raises(ValueError,match='DEEPSEEK_THINKING_MODE'):
            runpy.run_path(cfg.__file__)
    else:
        assert runpy.run_path(cfg.__file__)['DEEPSEEK_THINKING_MODE']==(mode or 'disabled')


def test_client_initialization_singleflight_across_threads():
    factory = Mock(side_effect=lambda **kwargs: Client())
    gate = threading.Barrier(12)
    def get(_):
        gate.wait()
        return get_model_client(factory=factory)
    with ThreadPoolExecutor(max_workers=12) as executor:
        clients = list(executor.map(get, range(12)))
    assert len({id(client) for client in clients}) == 1
    factory.assert_called_once_with(api_key="test-only-placeholder", base_url="https://offline.invalid", max_retries=0)


def test_configuration_refresh_and_injected_client_remain_distinct(monkeypatch):
    factory = Mock(side_effect=lambda **kwargs: Client())
    first = get_model_client(factory=factory)
    monkeypatch.setattr(cfg, "DEEPSEEK_API_KEY", "second-placeholder")
    second = get_model_client(legacy=first, factory=factory)
    assert second is not first
    assert get_model_client(factory=factory) is second
    injected = Client()
    assert get_model_client(legacy=injected, factory=factory) is injected
    close_model_clients()
    assert first.closed == second.closed == 1 and injected.closed == 0
    assert get_model_client(legacy=second, factory=factory) is not second


def test_failed_initialization_can_retry():
    expected = Client()
    factory = Mock(side_effect=[RuntimeError("offline failure"), expected])
    with pytest.raises(RuntimeError):
        get_model_client(factory=factory)
    assert get_model_client(factory=factory) is expected


def test_modules_share_client_but_preserve_monkeypatch(monkeypatch):
    from core import generator
    from agents import coordinator, writer_agent
    factory = Mock(side_effect=lambda **kwargs: Client())
    for module in (generator, coordinator, writer_agent):
        monkeypatch.setattr(module, "OpenAI", factory)
        monkeypatch.setattr(module, "_client", None)
    client = generator._get_client()
    assert coordinator._get_client() is writer_agent._get_client() is client
    external = Client()
    monkeypatch.setattr(generator, "_client", external)
    assert generator._get_client() is external


def test_actual_usage_and_prices_are_separate_from_estimates(monkeypatch):
    monkeypatch.setattr(cfg, "MODEL_INPUT_COST_PER_MILLION", 1)
    monkeypatch.setattr(cfg, "MODEL_OUTPUT_COST_PER_MILLION", 2)
    response = NS(choices=[], usage=NS(prompt_tokens=12, completion_tokens=8, total_tokens=20))
    run = RunContext()
    with use_run_context(run):
        assert call(Client(response)) is response
    snapshot = model_metrics_snapshot(run)
    assert snapshot["call_count"] == 1 and snapshot["actual_total_tokens"] == 20
    assert snapshot["actual_cost"] == pytest.approx(28 / 1_000_000)
    assert snapshot["unknown_usage_calls"] == snapshot["unknown_cost_calls"] == 0
    record = snapshot["calls"][0]
    assert record["estimated_input_tokens"] > 12 and record["usage_source"] == "provider"
    assert record["duration_ms"] >= 0 and record["stage"] == "generate"
    assert "private mail text" not in json.dumps(snapshot)
    assert "test-only-placeholder" not in json.dumps(snapshot)


def test_missing_usage_is_unknown_and_charged_conservatively():
    run = RunContext()
    with use_run_context(run):
        call(Client())
    snapshot = model_metrics_snapshot(run)
    record = snapshot["calls"][0]
    assert snapshot["unknown_usage_calls"] == snapshot["unknown_cost_calls"] == 1
    assert record["total_tokens"] is record["actual_cost"] is None
    assert snapshot["budget_accounted_tokens"] == record["estimated_input_tokens"] + 100
    assert snapshot["reserved_tokens"] == 0


def test_token_budget_rejects_before_provider_request():
    client, run = Client(), RunContext(model_token_limit=10)
    with use_run_context(run), pytest.raises(ModelBudgetExceeded):
        call(client)
    client.chat.completions.create.assert_not_called()


def test_actual_usage_overrun_prevents_next_call():
    client = Client(NS(choices=[], usage=NS(prompt_tokens=300, completion_tokens=100, total_tokens=400)))
    run = RunContext(model_token_limit=300)
    with use_run_context(run):
        call(client)
        with pytest.raises(ModelBudgetExceeded):
            call(client)
    assert client.chat.completions.create.call_count == 1
    assert model_metrics_snapshot(run)["actual_total_tokens"] == 400


def test_unknown_cost_does_not_bypass_configured_limit():
    client, run = Client(), RunContext(model_cost_limit=1)
    with use_run_context(run), pytest.raises(ModelBudgetExceeded, match="requires configured"):
        call(client)
    client.chat.completions.create.assert_not_called()


def test_concurrent_calls_reserve_budget_atomically():
    entered, release = threading.Event(), threading.Event()
    client, run = Client(), RunContext(model_token_limit=300)
    def provider(**kwargs):
        entered.set()
        assert release.wait(5)
        return NS(choices=[])
    client.chat.completions.create.side_effect = provider
    def first():
        with use_run_context(run):
            call(client)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(first)
        assert entered.wait(5)
        try:
            with use_run_context(run), pytest.raises(ModelBudgetExceeded):
                call(client)
        finally:
            release.set()
        future.result()
    assert client.chat.completions.create.call_count == 1


class Stream:
    def __init__(self, chunks):
        self.chunks = iter(chunks)
        self.closed = False
    def __iter__(self):
        return self
    def __next__(self):
        item = next(self.chunks)
        if isinstance(item, BaseException):
            raise item
        return item
    def close(self):
        self.closed = True


def test_stream_usage_only_terminal_chunk_is_observed_and_closed_once():
    chunks = [NS(choices=[NS(delta=NS(content="a"))]),
              NS(choices=[], usage=NS(prompt_tokens=9, completion_tokens=1, total_tokens=10))]
    stream, run = Stream(chunks), RunContext()
    with use_run_context(run):
        result = call(Client(stream), stream=True)
        assert list(result) == chunks
        result.close()
    metrics = model_metrics_snapshot(run)
    assert stream.closed and metrics["actual_total_tokens"] == 10
    assert metrics["call_count"] == 1 and metrics["calls"][0]["status"] == "complete"


def test_stream_interruption_accounts_once_and_releases_reservation():
    stream, run = Stream([NS(choices=[]), TimeoutError("private provider payload")]), RunContext()
    with use_run_context(run):
        result = call(Client(stream), stream=True)
        next(result)
        with pytest.raises(TimeoutError):
            next(result)
        result.close()
    metrics = model_metrics_snapshot(run)
    assert metrics["call_count"] == 1 and metrics["reserved_tokens"] == 0
    assert metrics["calls"][0]["status"] == "error"
    assert "private provider" not in json.dumps(metrics)


def test_stream_early_close_does_not_assert_transport_success():
    stream, run = Stream([NS(choices=[])]), RunContext()
    with use_run_context(run):
        result = call(Client(stream), stream=True)
        result.close()
    assert model_metrics_snapshot(run)["calls"][0]["status"] == "closed"


def test_cancellation_and_errors_are_not_swallowed_or_retried():
    event, client = threading.Event(), Client()
    event.set()
    with use_run_context(RunContext(cancel_event=event)), pytest.raises(RunCancelled):
        call(client)
    client.chat.completions.create.assert_not_called()
    client.chat.completions.create.side_effect = TimeoutError("private content")
    run = RunContext()
    with use_run_context(run), pytest.raises(TimeoutError):
        call(client)
    assert client.chat.completions.create.call_count == 1
    assert model_metrics_snapshot(run)["calls"][0]["status"] == "error"


def test_calls_log_is_bounded_while_aggregates_remain_complete():
    run = RunContext()
    with use_run_context(run):
        for _ in range(270):
            call(Client(NS(usage=NS(prompt_tokens=1, completion_tokens=1, total_tokens=2))))
    metrics = model_metrics_snapshot(run)
    assert metrics["call_count"] == 270 and metrics["actual_total_tokens"] == 540
    assert len(metrics["calls"]) == 256 and metrics["calls_truncated"] is True


def test_parent_child_ledger_and_resume_preserve_cumulative_budget():
    parent = RunContext(model_token_limit=300)
    child = RunContext(model_token_limit=300, model_usage=ensure_model_usage(parent))
    with use_run_context(child):
        call(Client())
    snapshot = model_metrics_snapshot(parent)
    assert snapshot["call_count"] == 1
    resumed = RunContext(model_token_limit=300)
    restore_model_metrics(json.loads(json.dumps(snapshot)), resumed)
    client = Client()
    with use_run_context(resumed), pytest.raises(ModelBudgetExceeded):
        call(client)
    client.chat.completions.create.assert_not_called()
    assert model_metrics_snapshot(resumed) == snapshot
    with pytest.raises(ValueError, match="already used"):
        restore_model_metrics(snapshot, resumed)


@pytest.mark.parametrize("mutation", [
    lambda snapshot: snapshot.update(reserved_tokens=1),
    lambda snapshot: snapshot.update(call_count=True),
    lambda snapshot: snapshot.update(actual_cost=float("nan")),
    lambda snapshot: snapshot.update(calls=[{}] * 257),
    lambda snapshot: snapshot.update(untrusted_mail_text="must not persist"),
    lambda snapshot: snapshot.update(budget_accounted_tokens=-1),
])
def test_resume_rejects_invalid_or_in_flight_accounting(mutation):
    snapshot = model_metrics_snapshot(RunContext())
    mutation(snapshot)
    with pytest.raises(ValueError):
        restore_model_metrics(snapshot, RunContext())


def test_cost_threshold_reserves_maximum_and_blocks_before_send(monkeypatch):
    monkeypatch.setattr(cfg, "MODEL_INPUT_COST_PER_MILLION", 1)
    monkeypatch.setattr(cfg, "MODEL_OUTPUT_COST_PER_MILLION", 2)
    client, run = Client(), RunContext(model_cost_limit=0.0004)
    with use_run_context(run):
        call(client)
        with pytest.raises(ModelBudgetExceeded, match="cost budget exhausted"):
            call(client)
    assert client.chat.completions.create.call_count == 1


def test_enabling_cost_budget_does_not_ignore_previously_unpriced_calls(monkeypatch):
    client, run = Client(), RunContext()
    with use_run_context(run):
        call(client)
        monkeypatch.setattr(cfg, "MODEL_INPUT_COST_PER_MILLION", 1)
        monkeypatch.setattr(cfg, "MODEL_OUTPUT_COST_PER_MILLION", 2)
        run.model_cost_limit = 1
        with pytest.raises(ModelBudgetExceeded, match="previous run calls"):
            call(client)
    assert client.chat.completions.create.call_count == 1


def test_shutdown_attempts_all_client_closes_after_one_failure(monkeypatch):
    first, second = Client(), Client()
    first.close = Mock(side_effect=RuntimeError("private transport context"))
    factory = Mock(side_effect=[first, second])
    get_model_client(factory=factory)
    monkeypatch.setattr(cfg, "DEEPSEEK_BASE_URL", "https://second-offline.invalid")
    get_model_client(factory=factory)
    with pytest.raises(RuntimeError, match="one or more") as error:
        close_model_clients()
    assert second.closed == 1
    assert "private" not in str(error.value)


def test_wrapper_clamps_nested_timeout_to_remaining_deadline():
    client = Client()
    with use_run_context(RunContext(deadline=time.monotonic() + 2)):
        call(client, timeout=100)
    assert 0 < client.chat.completions.create.call_args.kwargs["timeout"] <= 2


def test_restore_cannot_reduce_totals_below_retained_call_usage():
    run = RunContext()
    with use_run_context(run):
        call(Client())
    snapshot = model_metrics_snapshot(run)
    snapshot["budget_accounted_tokens"] = 0
    with pytest.raises(ValueError, match="omit recorded usage"):
        restore_model_metrics(snapshot, RunContext())


def test_different_model_requires_its_own_price_entry(monkeypatch):
    monkeypatch.setattr(cfg, "MODEL_INPUT_COST_PER_MILLION", 1)
    monkeypatch.setattr(cfg, "MODEL_OUTPUT_COST_PER_MILLION", 2)
    client, run = Client(), RunContext(model_cost_limit=1)
    with use_run_context(run), pytest.raises(ModelBudgetExceeded, match="requires configured"):
        create_completion(client, model="different-planner", messages=[], max_tokens=100)
    client.chat.completions.create.assert_not_called()
    monkeypatch.setattr(cfg, "MODEL_TOKEN_PRICES", {"different-planner": {"input": 2, "output": 3}})
    with use_run_context(run):
        create_completion(client, model="different-planner", messages=[], max_tokens=100)
    assert client.chat.completions.create.call_count == 1


def test_stream_requests_usage_but_supports_explicit_compatibility_setting(monkeypatch):
    client = Client(Stream([]))
    stream = call(client, stream=True)
    assert client.chat.completions.create.call_args.kwargs["stream_options"] == {"include_usage": True}
    stream.close()
    monkeypatch.setattr(cfg, "MODEL_STREAM_INCLUDE_USAGE", False, raising=False)
    stream = call(client, stream=True)
    assert "stream_options" not in client.chat.completions.create.call_args.kwargs
    stream.close()
