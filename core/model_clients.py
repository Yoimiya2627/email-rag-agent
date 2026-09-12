"""Shared OpenAI client lifecycle and content-free per-run call accounting.

Prices are optional operator configuration, in one consistent currency per
million tokens. Estimates are conservative reservations, never billed usage.
Call ``close_model_clients`` only after stopping/waiting for active requests.
"""
from __future__ import annotations

import hashlib
import copy
import math
import threading
import time
import weakref
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Mapping
from urllib.parse import urlsplit

from openai import OpenAI

import config.settings as cfg
from core.context_budget import measure_context, model_profile


class ModelBudgetExceeded(RuntimeError):
    """A configured model budget prevents another provider request."""


_CLIENT_LOCK = threading.RLock()
_CLIENTS: dict[tuple, Any] = {}
_MANAGED = weakref.WeakValueDictionary()
_METER_LOCK = threading.RLock()
_STAGES = frozenset({"generate", "stream_generate", "intent", "write_reply", "summarize",
                     "analyze", "rewrite", "filter", "rerank", "graph_grade", "agent",
                     "agent_plan", "agent_finalize", "graph_generate", "session_summary", "context_compress", "context_candidates"})


def get_model_client(legacy=None, *, factory=OpenAI):
    """Singleflight construction keyed by SDK factory and connection settings.

    Caller-injected clients remain authoritative. A formerly managed client is
    refreshed on config changes; all modules sharing the same settings reuse it.
    At most eight configurations per SDK factory are retained until shutdown.
    """
    with _CLIENT_LOCK:
        if legacy is not None and _MANAGED.get(id(legacy)) is not legacy:
            return legacy
        api_key, base_url = str(cfg.DEEPSEEK_API_KEY), str(cfg.DEEPSEEK_BASE_URL)
        key = (factory, base_url, hashlib.sha256(api_key.encode()).digest())
        if key not in _CLIENTS:
            if sum(existing[0] is factory for existing in _CLIENTS) >= 8:
                raise RuntimeError("model client configuration capacity reached; restart after active requests finish")
            client = factory(api_key=api_key, base_url=base_url, max_retries=0)
            _CLIENTS[key] = client
            try:
                _MANAGED[id(client)] = client
            except TypeError:
                # Non-weak-referenceable injected factories are test adapters.
                pass
        return _CLIENTS[key]


def close_model_clients():
    """Close owned clients at shutdown; never close injected client doubles."""
    with _CLIENT_LOCK:
        clients = list({id(client): client for client in _CLIENTS.values()}.values())
        _CLIENTS.clear()
        # Keep identities until their legacy references disappear: a stale
        # module reference must obtain a new client after shutdown/reinitializing.
        failed = False
        for client in clients:
            close = getattr(client, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:
                    failed = True
        if failed:
            raise RuntimeError("one or more model clients could not close")


def _number(value, label, *, integer=False):
    if type(value) not in (int, float) or value < 0:
        raise ValueError(f"invalid {label}")
    if (type(value) is float and not math.isfinite(value)) or value > 2**63 - 1:
        raise ValueError(f"invalid {label}")
    if integer and type(value) is not int:
        raise ValueError(f"invalid {label}")
    return value


def _price(name):
    value = getattr(cfg, name, None)
    return None if value is None or value == "" else _number(float(value), name)


def _model_prices(model):
    prices = getattr(cfg, "MODEL_TOKEN_PRICES", {})
    if type(prices) is not dict:
        raise ValueError("model token prices must map model names to input/output prices")
    if model in prices:
        entry = prices[model]
        if type(entry) is not dict or set(entry) != {"input", "output"}:
            raise ValueError("model price entry requires input and output rates")
        return (_number(entry["input"], "model input price"),
                _number(entry["output"], "model output price"))
    if model == cfg.DEEPSEEK_MODEL:
        return _price("MODEL_INPUT_COST_PER_MILLION"), _price("MODEL_OUTPUT_COST_PER_MILLION")
    # A planner/reranker can use another model; never reuse its provider's
    # primary-model price silently for a differently priced model.
    return None, None


def _usage(response):
    usage = getattr(response, "usage", None)
    def read(name):
        value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
        return value if type(value) is int and value >= 0 else None
    prompt, completion, total = read("prompt_tokens"), read("completion_tokens"), read("total_tokens")
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    return {"input_tokens": prompt, "output_tokens": completion, "total_tokens": total}


@dataclass
class ModelUsage:
    """One ledger shared by nested calls, with atomic budget reservations."""
    calls: list = field(default_factory=list)
    call_count: int = 0
    actual_total_tokens: int = 0
    actual_cost: float = 0.0
    unknown_usage_calls: int = 0
    unknown_cost_calls: int = 0
    unpriced_calls: int = 0
    budget_tokens: int = 0
    budget_cost: float = 0.0
    reserved_tokens: int = 0
    reserved_cost: float = 0.0
    lock: Any = field(default_factory=threading.RLock, repr=False)


def _meter(run):
    if run is None:
        return ModelUsage()
    with _METER_LOCK:
        if run.model_usage is None:
            run.model_usage = ModelUsage()
        return run.model_usage


def ensure_model_usage(run=None):
    """Initialize before constructing a nested RunContext; pass the same ledger."""
    from agents.runtime import current_run
    return _meter(run or current_run())


def model_metrics_snapshot(run=None):
    """Safe JSON data only; unknown provider usage/cost stays explicitly unknown."""
    from agents.runtime import current_run
    run = run or current_run()
    meter = _meter(run)
    with meter.lock:
        return {"call_count": meter.call_count, "actual_total_tokens": meter.actual_total_tokens,
                "actual_cost": meter.actual_cost, "unknown_usage_calls": meter.unknown_usage_calls,
                "unknown_cost_calls": meter.unknown_cost_calls,
                "unpriced_calls": meter.unpriced_calls,
                "budget_accounted_tokens": meter.budget_tokens,
                "budget_accounted_cost": meter.budget_cost,
                "reserved_tokens": meter.reserved_tokens, "reserved_cost": meter.reserved_cost,
                "calls": [dict(item) for item in meter.calls],
                "calls_truncated": meter.call_count > len(meter.calls)}


def restore_model_metrics(snapshot, run=None):
    """Restore a validated checkpoint only into a fresh ledger at a safe boundary.

    This validates representation, not ownership/authenticity. The caller must
    load its own trusted owner-bound checkpoint, never client-supplied counters.
    """
    from agents.runtime import current_run
    run = run or current_run()
    if run is None:
        raise ValueError("a run context is required to restore model accounting")
    integer_fields = ("call_count", "actual_total_tokens", "unknown_usage_calls", "unknown_cost_calls", "unpriced_calls",
                      "budget_accounted_tokens", "reserved_tokens")
    cost_fields = ("actual_cost", "budget_accounted_cost", "reserved_cost")
    expected = set(integer_fields + cost_fields + ("calls", "calls_truncated"))
    if type(snapshot) is not dict or set(snapshot) != expected:
        raise ValueError("invalid model accounting checkpoint fields")
    for name in integer_fields:
        _number(snapshot[name], name, integer=True)
        if snapshot[name] > 2**63 - 1:
            raise ValueError("model accounting integer exceeds storage range")
    for name in cost_fields:
        _number(snapshot[name], name)
    if snapshot["reserved_tokens"] or snapshot["reserved_cost"]:
        raise ValueError("cannot restore a model call in flight")
    calls = snapshot["calls"]
    if (type(calls) is not list or len(calls) != min(256, snapshot["call_count"])
            or type(snapshot["calls_truncated"]) is not bool
            or snapshot["calls_truncated"] != (snapshot["call_count"] > len(calls))
            or snapshot["unknown_usage_calls"] > snapshot["call_count"]
            or snapshot["unknown_cost_calls"] > snapshot["call_count"]
            or snapshot["unpriced_calls"] > snapshot["unknown_cost_calls"]
            or snapshot["budget_accounted_tokens"] < snapshot["actual_total_tokens"]
            or snapshot["budget_accounted_cost"] < snapshot["actual_cost"]):
        raise ValueError("inconsistent model accounting checkpoint")
    call_keys = {"stage", "duration_ms", "status", "input_tokens", "output_tokens", "total_tokens",
                 "actual_cost", "estimated_input_tokens", "reserved_output_tokens", "estimated_max_cost",
                 "token_estimation_method", "usage_source"}
    for item in calls:
        if type(item) is not dict or set(item) != call_keys:
            raise ValueError("invalid model call checkpoint fields")
        if (any(type(item[name]) is not str for name in ("stage", "status", "token_estimation_method", "usage_source"))
                or item["stage"] not in _STAGES | {"other"} or item["status"] not in {"complete", "closed", "error"}
                or item["token_estimation_method"] not in {"utf8_bytes_upper_bound", "injected_tokenizer", "validated_model_adapter"}
                or item["usage_source"] not in {"provider", "unavailable"}):
            raise ValueError("invalid model call checkpoint labels")
        for name in ("input_tokens", "output_tokens", "total_tokens"):
            if item[name] is not None:
                _number(item[name], name, integer=True)
        for name in ("estimated_input_tokens", "reserved_output_tokens"):
            _number(item[name], name, integer=True)
        _number(item["duration_ms"], "duration_ms")
        for name in ("actual_cost", "estimated_max_cost"):
            if item[name] is not None:
                _number(item[name], name)
    if (snapshot["actual_total_tokens"] < sum(item["total_tokens"] or 0 for item in calls)
            or snapshot["actual_cost"] + 1e-12 < sum(item["actual_cost"] or 0 for item in calls)
            or snapshot["budget_accounted_tokens"] < sum(item["total_tokens"] if item["total_tokens"] is not None
                    else item["estimated_input_tokens"] + item["reserved_output_tokens"] for item in calls)
            or snapshot["budget_accounted_cost"] + 1e-12 < sum(item["actual_cost"] if item["actual_cost"] is not None
                    else item["estimated_max_cost"] or 0 for item in calls)):
        raise ValueError("model checkpoint totals omit recorded usage")
    meter = ModelUsage(calls=[dict(item) for item in calls], call_count=snapshot["call_count"],
                       actual_total_tokens=snapshot["actual_total_tokens"], actual_cost=snapshot["actual_cost"],
                       unknown_usage_calls=snapshot["unknown_usage_calls"], unknown_cost_calls=snapshot["unknown_cost_calls"],
                       unpriced_calls=snapshot["unpriced_calls"],
                       budget_tokens=snapshot["budget_accounted_tokens"], budget_cost=snapshot["budget_accounted_cost"])
    with _METER_LOCK:
        existing = run.model_usage
        if existing is not None:
            with existing.lock:
                if existing.call_count or existing.reserved_tokens or existing.reserved_cost:
                    raise ValueError("cannot replace an active or already used model ledger")
        run.model_usage = meter
    return meter


class _Call:
    def __init__(self, run, stage, kwargs):
        self.run, self.meter = run, _meter(run)
        self.stage = stage if stage in _STAGES else "other"
        self.started = time.monotonic()
        self.finished = False
        self.usage = {"input_tokens": None, "output_tokens": None, "total_tokens": None}
        estimate = measure_context(kwargs.get("messages", []), kwargs.get("tools"),
                                   token_counter=getattr(run, "token_counter", None), model=kwargs.get("model"),
                                   model_revision=model_profile(kwargs.get("model"), stage)["revision"], stage=stage)
        if run is not None:
            run.context_metrics.update(provider_input_tokens=None, input_estimate_error_tokens=None,
                                       provider_usage_available=False)
        self.input_estimate = estimate["estimated_input_tokens"]
        self.method = estimate["token_estimation_method"]
        self.output_reserve = kwargs.get("max_completion_tokens", kwargs.get("max_tokens", 0))
        _number(self.output_reserve, "output reservation", integer=True)
        self.tokens = self.input_estimate + self.output_reserve
        self.input_price, self.output_price = _model_prices(kwargs.get("model"))
        self.cost = self._cost(self.input_estimate, self.output_reserve)
        token_limit = getattr(run, "model_token_limit", None)
        cost_limit = getattr(run, "model_cost_limit", None)
        token_limit = _number(token_limit if token_limit is not None else getattr(cfg, "MODEL_RUN_TOKEN_LIMIT", 0), "run token limit", integer=True)
        cost_limit = _number(cost_limit if cost_limit is not None else getattr(cfg, "MODEL_RUN_COST_LIMIT", 0), "run cost limit")
        if (token_limit or cost_limit) and not self.output_reserve:
            raise ModelBudgetExceeded("a bounded output reservation is required for model budgets")
        with self.meter.lock:
            if token_limit and self.meter.budget_tokens + self.meter.reserved_tokens + self.tokens > token_limit:
                raise ModelBudgetExceeded("run model token budget exhausted")
            if cost_limit and self.cost is None:
                raise ModelBudgetExceeded("model cost budget requires configured input and output prices")
            if cost_limit and self.meter.unpriced_calls:
                raise ModelBudgetExceeded("previous run calls have no configured price; cost budget cannot be verified")
            if cost_limit and self.meter.budget_cost + self.meter.reserved_cost + self.cost > cost_limit:
                raise ModelBudgetExceeded("run model cost budget exhausted")
            self.meter.reserved_tokens += self.tokens
            self.meter.reserved_cost += self.cost or 0.0

    def _cost(self, prompt, completion):
        if self.input_price is None or self.output_price is None or prompt is None or completion is None:
            return None
        return _number((prompt * self.input_price + completion * self.output_price) / 1_000_000,
                       "model cost")

    def observe(self, response):
        usage = _usage(response)
        if any(value is not None for value in usage.values()):
            self.usage = usage
            if self.run is not None:
                actual = usage['input_tokens']
                self.run.context_metrics.update(provider_input_tokens=actual,
                    input_estimate_error_tokens=(actual-self.input_estimate if actual is not None else None),
                    provider_usage_available=actual is not None)

    def finish(self, status):
        with self.meter.lock:
            if self.finished:
                return
            self.finished = True
            actual_cost = self._cost(self.usage["input_tokens"], self.usage["output_tokens"])
            actual_tokens = self.usage["total_tokens"]
            self.meter.reserved_tokens -= self.tokens
            self.meter.reserved_cost = max(0.0, self.meter.reserved_cost - (self.cost or 0.0))
            self.meter.budget_tokens += actual_tokens if actual_tokens is not None else self.tokens
            self.meter.budget_cost += actual_cost if actual_cost is not None else self.cost or 0.0
            self.meter.actual_total_tokens += actual_tokens or 0
            self.meter.actual_cost += actual_cost or 0.0
            self.meter.unknown_usage_calls += int(actual_tokens is None)
            self.meter.unknown_cost_calls += int(actual_cost is None)
            self.meter.unpriced_calls += int(self.cost is None)
            self.meter.call_count += 1
            self.meter.calls.append({"stage": self.stage, "duration_ms": round((time.monotonic() - self.started) * 1000, 3),
                                     "status": status, **self.usage, "actual_cost": actual_cost,
                                     "estimated_input_tokens": self.input_estimate,
                                     "reserved_output_tokens": self.output_reserve,
                                     "estimated_max_cost": self.cost, "token_estimation_method": self.method,
                                     "usage_source": "provider" if actual_tokens is not None else "unavailable"})
            del self.meter.calls[:-256]


class _MeasuredStream:
    def __init__(self, stream, call):
        self.stream, self.call, self.iterator = stream, call, iter(stream)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            if self.call.run is not None:
                self.call.run.remaining(1)
            chunk = next(self.iterator)
            self.call.observe(chunk)
            return chunk
        except StopIteration:
            self.call.finish("complete")
            raise
        except BaseException:
            self.call.finish("error")
            raise

    def close(self):
        try:
            close = getattr(self.stream, "close", None)
            if close is not None:
                close()
        except BaseException:
            self.call.finish("error")
            raise
        finally:
            self.call.finish("closed")


def _apply_deepseek_thinking(client, kwargs):
    """Scope provider extensions to the actual official V4 transport/model."""
    extra = kwargs.get('extra_body')
    if extra is not None and not isinstance(extra, Mapping):
        return  # Preserve SDK-specific sentinels and its argument validation.
    model = extra.get('model', kwargs.get('model')) if extra is not None else kwargs.get('model')
    if model not in {'deepseek-v4-flash', 'deepseek-v4-pro'}:
        return
    try:
        # Injected clients may point elsewhere even when application config is
        # DeepSeek. Never infer their destination from global settings.
        url = urlsplit(str(getattr(client, 'base_url', '')))
        official = (url.scheme == 'https' and url.hostname == 'api.deepseek.com'
                    and url.port in (None, 443) and not url.username and not url.password
                    and not url.query and not url.fragment
                    and url.path.rstrip('/') in ('', '/v1', '/beta'))
    except (TypeError, ValueError):
        return
    if not official:
        return
    body = copy.deepcopy(dict(extra)) if extra is not None else {}
    if 'thinking' not in body:
        mode = getattr(cfg, 'DEEPSEEK_THINKING_MODE', 'disabled')
        if mode not in {'enabled', 'disabled'}:
            raise ValueError('DEEPSEEK_THINKING_MODE must be enabled or disabled')
        body['thinking'] = {'type': mode}
    kwargs['extra_body'] = body


def create_completion(client, *, stage="other", evidence_refs=None, **kwargs):
    """One provider request, never retry/replay. Return the SDK response/stream.

    Transport success does not assert a complete model answer; callers must
    retain the ModelText/finish_reason checks. Missing usage reserves the full
    requested output budget; it is not reported as actual provider consumption.
    """
    from agents.runtime import current_run
    _apply_deepseek_thinking(client, kwargs)
    run = current_run()
    if run is not None:
        kwargs["timeout"] = run.remaining(float(kwargs.get("timeout", cfg.LLM_TIMEOUT)))
    if kwargs.get("stream") and getattr(cfg, "MODEL_STREAM_INCLUDE_USAGE", True):
        kwargs.setdefault("stream_options", {"include_usage": True})
    # The last shared boundary validates the actual model output reservation,
    # including tools. Earlier selection never grants a larger call budget.
    from core.memory import fit_messages_to_budget
    output = kwargs.get("max_completion_tokens", kwargs.get("max_tokens", 0))
    kwargs["messages"] = fit_messages_to_budget(kwargs.get("messages", []), schemas=kwargs.get("tools"),
        model=kwargs.get("model"), model_revision=model_profile(kwargs.get("model"), stage)["revision"], stage=stage,
        max_output_tokens=output)[0]
    call = _Call(run, stage, kwargs)
    try:
        if run is not None:
            kwargs["timeout"] = run.remaining(float(kwargs.get("timeout", cfg.LLM_TIMEOUT)))
        response = client.chat.completions.create(**kwargs)
        if evidence_refs:
            from agents.runtime import record_generation_evidence
            record_generation_evidence(evidence_refs)
        if kwargs.get("stream"):
            return _MeasuredStream(response, call)
        call.observe(response)
        if run is not None:
            run.remaining(1)
        call.finish("complete")
        return response
    except BaseException:
        call.finish("error")
        raise
