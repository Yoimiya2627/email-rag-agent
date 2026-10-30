"""Small per-run context and shared result semantics; no infrastructure imports."""
from __future__ import annotations

import hashlib
import json
import math
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


class RunDeadlineExceeded(TimeoutError):
    """The run deadline expired; no new operation may start."""


class ContextBudgetExceeded(RuntimeError):
    """The model context exceeds the explicit character budget."""


class RunCancelled(RuntimeError):
    """The caller requested cancellation; no new operation may start."""


@dataclass
class RunContext:
    owner_id: str = "local"
    session_id: str | None = None
    resuming: bool = False
    session_repository: Any = None
    context_epoch: int | None = None
    tool_result_store: Any = None
    tool_result_refs: dict = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    deadline: float | None = None
    max_tool_calls: int = 12
    context_char_limit: int = 60000
    context_token_limit: int | None = None
    output_token_reserve: int | None = None
    token_counter: Any = None
    context_metrics: dict = field(default_factory=dict)
    cancel_event: Any = None
    progress_callback: Any = None
    checkpoint_callback: Any = None
    task_context: dict | None = None
    model_usage: Any = None
    model_token_limit: int | None = None
    model_cost_limit: float | None = None
    generation_evidence_refs: list[dict] = field(default_factory=list)
    tool_calls: int = 0
    tool_call_id: str = ""
    evidence: dict[tuple[str, str], dict] = field(default_factory=dict)
    tool_evidence: dict[tuple[str, str], dict] = field(default_factory=dict)
    visible_evidence: dict[tuple[str, str], dict] = field(default_factory=dict)

    def remaining(self, default: float) -> float:
        if self.cancel_event is not None and self.cancel_event.is_set():
            raise RunCancelled("run cancelled")
        if self.deadline is None:
            return float(default)
        left = self.deadline - time.monotonic()
        if left <= 0:
            raise RunDeadlineExceeded("run deadline exceeded")
        return min(float(default), left)

    def progress(self, stage: str, **metrics) -> None:
        self.remaining(1)
        if self.progress_callback is not None:
            self.progress_callback(stage, **metrics)

    def checkpoint(self, payload: dict) -> None:
        # Persist a just-finished operation even if cancellation/deadline arrived
        # during it. Checkpoints do not authorize starting another operation.
        if self.checkpoint_callback is not None:
            self.checkpoint_callback(payload)

    def check_context(self, messages: list, schemas: list | None = None, *, model=None, model_revision=None, stage=None, max_output_tokens=None) -> None:
        from core.context_budget import measure_context, within_budget, model_profile
        import config.settings as cfg
        token_limit = (self.context_token_limit if self.context_token_limit is not None
                       else int(getattr(cfg, "MODEL_CONTEXT_TOKENS", 0)))
        reserve = (self.output_token_reserve if self.output_token_reserve is not None
                   else int(getattr(cfg, "MODEL_OUTPUT_RESERVE_TOKENS", 0)))
        profile = model_profile(model, stage)
        model_revision = profile['revision'] or model_revision
        if profile['context_tokens']:
            token_limit = min(token_limit, profile['context_tokens']) if token_limit else profile['context_tokens']
        if max_output_tokens is not None:
            safety = int(getattr(cfg, "MODEL_CONTEXT_SAFETY_TOKENS", 128))
            if type(max_output_tokens) is not int or max_output_tokens < 0 or safety < 0:
                raise ValueError('invalid output tokens or safety margin')
            reserve = max_output_tokens + safety
        usage = measure_context(messages, schemas, token_counter=self.token_counter,
                                model=model, model_revision=model_revision, stage=stage)
        self.context_metrics = {**self.context_metrics, **usage, "token_capacity": token_limit,
                                "output_token_reserve": reserve}
        if not within_budget(usage, char_limit=self.context_char_limit,
                             token_limit=token_limit, output_reserve=reserve):
            raise ContextBudgetExceeded("model context exceeds character or token budget")


_RUN: ContextVar[RunContext | None] = ContextVar("email_agent_run", default=None)


def current_run() -> RunContext | None:
    return _RUN.get()


def record_generation_evidence(references) -> None:
    """Record source ranges submitted in a successful provider request.

    These are model-visible material, not proof that the answer cited or follows
    from it. Callers prepare refs from the exact final rendered source text.
    """
    context = current_run()
    if context is None:
        return
    fields = ("email_id", "chunk_id", "source_version", "source_sha256", "chunk_sha256",
              "visible_start", "visible_end", "visible_hash", "offset_basis")
    for ref in references or []:
        value = {key: ref[key] for key in fields if key in ref}
        if value and value not in context.generation_evidence_refs:
            context.generation_evidence_refs.append(value)


@contextmanager
def use_run_context(context: RunContext):
    token = _RUN.set(context)
    try:
        yield context
    finally:
        _RUN.reset(token)


def remaining_timeout(default: float) -> float:
    """Nested LLM/transport calls use the same monotonic deadline."""
    context = current_run()
    return context.remaining(default) if context else float(default)


def check_model_context(messages: list, schemas: list | None = None) -> None:
    context = current_run()
    if context:
        context.check_context(messages, schemas)


def content_digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def argument_summary(arguments: Any) -> dict:
    """No argument values or caller-chosen field names enter persistent logs."""
    raw = json.dumps(arguments, ensure_ascii=False, default=str)
    return {"sha256": content_digest(arguments), "chars": len(raw),
            "field_count": len(arguments) if isinstance(arguments, dict) else 0}


def add_evidence(sources) -> None:
    context = current_run()
    if context is None:
        return
    for source in sources or []:
        if hasattr(source, "model_dump"):
            source = source.model_dump()
        elif hasattr(source, "dict"):
            source = source.dict()
        if not isinstance(source, dict):
            continue
        email_id, chunk_id = source.get("email_id"), source.get("chunk_id")
        if not isinstance(email_id, str) or not isinstance(chunk_id, str) or not email_id or not chunk_id:
            continue
        metadata = source.get("metadata") or {
            key: source[key] for key in ("subject", "sender", "date") if key in source
        }
        score = source.get("score", 0.0)
        score = float(score) if isinstance(score, (int, float)) and math.isfinite(score) else 0.0
        from core.evidence import evidence_reference, source_coverage
        ref = evidence_reference(source)
        candidate = {
            "email_id": email_id, "chunk_id": chunk_id,
            "content": str(source.get("content", source.get("snippet", ""))),
            "score": score, "metadata": {**metadata, **ref,
                "coverage": source.get("coverage") or source_coverage(metadata)},
            "coverage": source.get("coverage") or source_coverage(metadata), **ref,
        }
        context.tool_evidence[(email_id, chunk_id)] = candidate
        previous = context.evidence.get((email_id, chunk_id))
        if (previous is None or previous.get("source_version") != candidate["source_version"]
                or len(candidate["content"]) > len(previous["content"])):
            context.evidence[(email_id, chunk_id)] = candidate


def tool_error(code: str, message: str, *, unknown: bool = False) -> dict:
    return {"_tool_result": 1, "status": "unknown" if unknown else "error", "data": None,
            "error": message, "error_code": code,
            "side_effect_state": "unknown" if unknown else "none", "evidence_refs": []}


def normalize_tool_result(value: Any, *, protocol_error: bool = False) -> dict:
    """Shared local/MCP envelope. Legacy tool implementations may return raw data."""
    if protocol_error:
        return {**tool_error("mcp_tool_error", "The MCP server reported a tool error."), "isError": True}
    from core.model_outcomes import ModelText, outcome_metadata
    if isinstance(value, ModelText):
        metadata = outcome_metadata(value)
        if metadata["completion_status"] != "complete":
            result = tool_error(metadata["error_code"] or "model_output_incomplete",
                                "The model did not complete the requested output.")
            result.update(completion_status=metadata["completion_status"],
                          finish_reason=metadata["finish_reason"],
                          data={"partial_text": str(value)} if value.strip() else None)
            return result
    if isinstance(value, dict) and value.get("_tool_result") == 1:
        return value
    if isinstance(value, dict) and (value.get("error") or value.get("isError") or value.get("is_error")):
        # Do not repeat arbitrary exception/remote response text in API logs.
        return tool_error("tool_error", "The tool could not complete the request.",
                          unknown=value.get("status") == "unknown")
    pending = isinstance(value, dict) and value.get("status") == "pending_approval"
    return {"_tool_result": 1, "status": "approval_required" if pending else "success",
            "data": value, "error": None, "error_code": None,
            "side_effect_state": "pending_approval" if pending else "none", "evidence_refs": []}


def result_evidence(result: dict) -> list[dict]:
    """Evidence carried by this result, never references from earlier calls."""
    sources = list(result.get("candidate_sources") or [])
    data = result.get("data")
    if isinstance(data, list):
        sources.extend(data)
    elif isinstance(data, dict):
        sources.extend(data.get("chunks") or [])
        sources.extend(data.get("items") or [])
        sources.append(data)
    return [source for source in sources if isinstance(source, dict)
            and isinstance(source.get("email_id"), str) and source["email_id"]
            and isinstance(source.get("chunk_id"), str) and source["chunk_id"]
            and (source.get("content") or source.get("snippet"))]


def _result_view(value: dict) -> dict:
    """Avoid duplicate body/chunk/candidate text in a model tool message."""
    view = dict(value)
    data = view.get("data")
    if isinstance(data, dict) and any(
            isinstance(chunk, dict) and chunk.get("content") for chunk in data.get("chunks", [])):
        data = dict(data)
        data.pop("body", None)
        view["data"] = data
    data_pairs = {(row["email_id"], row["chunk_id"])
                  for row in result_evidence({"data": data})}
    if "candidate_sources" in view:
        view["candidate_sources"] = [row for row in view["candidate_sources"]
            if isinstance(row, dict) and (row.get("email_id"), row.get("chunk_id")) not in data_pairs]
        if not view["candidate_sources"]:
            view.pop("candidate_sources")
    if "evidence_refs" in view:
        from core.evidence import evidence_reference
        view["evidence_refs"] = [evidence_reference(row) for row in result_evidence(view)]
    return view


def bounded_json(value: Any, limit: int) -> str:
    """Keep readable prefixes and exact source IDs within a valid JSON budget.

    Truncation is explicit. Tiny budgets cannot hold a usable envelope; they
    return an error/marker (never a content-free success) and the loop records
    output_budget_exceeded separately from the actual tool execution outcome.
    """
    limit = max(2, int(limit))
    from core.evidence import refresh_visible_references
    value = refresh_visible_references(value)
    def encode(item):
        return json.dumps(item, ensure_ascii=False, default=str, separators=(",", ":"))
    original = encode(value)
    view = _result_view(value) if isinstance(value, dict) else value
    encoded = encode(view)
    if len(encoded) <= limit:
        return encoded

    def shorten(item, chars, count, key=""):
        if isinstance(item, str):
            # Identifiers and execution state must never become invented values.
            if key in {"email_id", "chunk_id", "approval_id", "status", "error_code", "side_effect_state",
                       "source_version", "source_sha256", "chunk_sha256", "visible_hash", "offset_basis",
                       "next_cursor", "resume_cursor", "continuation_cursor", "page_kind",
                       "attachment_inventory_status", "scope", "decode_status", "mailbox_sync_complete",
                       "result_id", "content_hash", "argument_hash", "call_id", "material_type", "tool"}:
                return item
            return item[:chars]
        if isinstance(item, list):
            return [shorten(row, chars, count) for row in item[:count]]
        if isinstance(item, dict):
            return {name: (row if name in {"coverage", "result_ref"} else shorten(row, chars, count, name)) for name, row in item.items()
                    if name != "evidence_refs"}
        return item

    # First reduce long passages, then the result count. A 2100-character
    # email normally fits intact after dropping the duplicated body field.
    for count in (50, 20, 10, 5, 2, 1):
        for chars in (2000, 1000, 500, 250, 120, 60):
            compact = shorten(view, chars, count)
            if isinstance(compact, dict):
                compact.update(truncated=True, original_chars=len(original),
                               truncation_note="Only retained items and text prefixes are shown.")
                if "evidence_refs" in value:
                    compact["evidence_refs"] = []
                compact = _result_view(refresh_visible_references(compact))
            else:
                compact = {"data": compact, "truncated": True, "original_chars": len(original)}
            encoded = encode(compact)
            if len(encoded) <= limit:
                return encoded
    error = {"status": "error", "error_code": "output_budget_exceeded", "truncated": True,
             "error": "Tool output cannot fit this budget. Increase the output limit."}
    for fallback in (error, {"error_code": "output_budget_exceeded"}, {"truncated": True}, {}):
        encoded = encode(fallback)
        if len(encoded) <= limit:
            return encoded


def validate_schema(value: Any, schema: dict, path: str = "arguments") -> None:
    """Validate the JSON Schema subset used by our registry (strict, no coercion).

    This is intentionally not advertised as a general JSON Schema engine.
    Unsupported remote schema features remain the remote server's responsibility.
    """
    kind = schema.get("type")
    checks = {"object": lambda v: isinstance(v, dict), "array": lambda v: isinstance(v, list),
              "string": lambda v: isinstance(v, str), "integer": lambda v: type(v) is int,
              "number": lambda v: type(v) in (int, float) and math.isfinite(v),
              "boolean": lambda v: type(v) is bool, "null": lambda v: v is None}
    kinds = kind if isinstance(kind, list) else [kind]
    if kind and not any(k in checks and checks[k](value) for k in kinds):
        raise ValueError(f"{path}: invalid type")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: invalid enum value")
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        missing = [key for key in schema.get("required", []) if key not in value]
        if missing:
            raise ValueError(f"{path}: missing required argument(s): {', '.join(missing)}")
        if schema.get("additionalProperties") is False and any(key not in properties for key in value):
            raise ValueError(f"{path}: unknown argument")
        for key, item in value.items():
            if key in properties:
                validate_schema(item, properties[key], f"{path}.{key}")
    elif isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", math.inf):
            raise ValueError(f"{path}: invalid list length")
        for item in value:
            validate_schema(item, schema.get("items", {}), f"{path}[]")
    elif isinstance(value, str):
        if len(value.strip()) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", math.inf):
            raise ValueError(f"{path}: invalid string length")
    elif type(value) in (int, float):
        if not math.isfinite(value) or value < schema.get("minimum", -math.inf) or value > schema.get("maximum", math.inf):
            raise ValueError(f"{path}: outside allowed range")
