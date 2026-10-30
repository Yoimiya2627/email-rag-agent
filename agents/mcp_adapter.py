"""Adapters between MCP tools and the existing agent loop.

The agent loop is synchronous today, so the live Streamable HTTP client runs
MCP SDK async calls inside a short-lived worker thread.  The default backend
remains local; MCP is only used when `AGENT_TOOL_BACKEND=mcp`.
"""
from __future__ import annotations

import asyncio
import contextvars
import copy
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import config.settings as cfg
from agents.runtime import (current_run, remaining_timeout, normalize_tool_result, tool_error, content_digest,
                            validate_schema, RunDeadlineExceeded, ContextBudgetExceeded)
from agents.tool_policy import ToolPolicy
from agents.tool_registry import TOOL_REGISTRY
from agents.tracing import safe_event


def _get_field(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def mcp_tool_to_openai_schema(tool: Any) -> dict:
    """Convert an MCP tool description to an OpenAI-compatible tool schema."""
    name = _get_field(tool, "name")
    description = _get_field(tool, "description", default="") or ""
    input_schema = _get_field(
        tool,
        "inputSchema",
        "input_schema",
        default={"type": "object", "properties": {}},
    )
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": input_schema,
        },
    }


def normalize_mcp_result(result: Any) -> Any:
    """Return the model-facing payload from an MCP call result."""
    if _get_field(result, "isError", "is_error", default=False):
        return normalize_tool_result(None, protocol_error=True)
    if isinstance(result, dict):
        if result.get("_tool_result") == 1:
            return normalize_tool_result(result)

    structured = _get_field(result, "structuredContent", "structured_content")
    if structured is not None:
        return normalize_tool_result(structured)

    content = _get_field(result, "content", default=None)
    if content:
        texts = []
        for item in content:
            text = _get_field(item, "text", default=None)
            if text is not None:
                texts.append(text)
        if len(texts) == 1:
            try:
                return normalize_tool_result(json.loads(texts[0]))
            except json.JSONDecodeError:
                return normalize_tool_result(texts[0])
        if texts:
            return normalize_tool_result("\n".join(texts))

    return normalize_tool_result(result)


class LocalToolBackend:
    """Adapter for the existing in-process tool schema + dispatch path."""

    def __init__(
        self,
        schemas_provider: Callable[[], list[dict]],
        call_tool_fn: Callable[[str, dict], Any],
    ):
        self._schemas_provider = schemas_provider
        self._call_tool_fn = call_tool_fn

    def tool_schemas(self) -> list[dict]:
        return self._schemas_provider()

    def call_tool(self, name: str, arguments: dict) -> Any:
        return normalize_tool_result(self._call_tool_fn(name, arguments))


class MCPToolBackend:
    """Adapter for a synchronous MCP client facade."""

    def __init__(
        self,
        client: Any,
        schema_cache_seconds: int | None = None,
        audit_logger: "MCPAuditLogger | None" = None,
    ):
        self.client = client
        self.schema_cache_seconds = (
            cfg.MCP_TOOL_SCHEMA_CACHE_SECONDS
            if schema_cache_seconds is None
            else schema_cache_seconds
        )
        self.audit_logger = audit_logger or MCPAuditLogger.from_settings()
        self._schema_cache: list[dict] | None = None
        self._schema_cache_at = 0.0
        # This backend normally lives for one run. Never infer cross-run/server idempotency.
        self._approval_results: dict[tuple[str, str, str, str], dict] = {}

    def tool_schemas(self) -> list[dict]:
        now = time.monotonic()
        if (
            self._schema_cache is not None
            and self.schema_cache_seconds > 0
            and now - self._schema_cache_at < self.schema_cache_seconds
        ):
            return self._schema_cache

        result = self.client.list_tools()
        tools = _get_field(result, "tools", default=result)
        visible = ToolPolicy.from_settings().visible_specs(TOOL_REGISTRY)
        self._schema_cache = [mcp_tool_to_openai_schema(tool) for tool in tools
                              if _get_field(tool, "name") in visible]
        self._schema_cache_at = now
        return self._schema_cache

    def refresh_tools(self) -> list[dict]:
        self._schema_cache = None
        self._schema_cache_at = 0.0
        return self.tool_schemas()

    def call_tool(self, name: str, arguments: dict) -> Any:
        started = time.perf_counter()
        request_id = str(uuid.uuid4())
        submitted = False
        cache_key = None
        context = current_run()
        cache_hit = False
        try:
            visible = ToolPolicy.from_settings().visible_specs(TOOL_REGISTRY)
            if name not in visible:
                normalized = tool_error("tool_not_allowed", "Tool is not allowed by policy.")
            else:
                validate_schema(arguments, visible[name].parameters)
                owner = getattr(cfg, "MCP_OWNER_ID", getattr(cfg, "API_OWNER_ID", "local"))
                if context and context.owner_id != owner:
                    normalized = tool_error("owner_mismatch", "MCP owner does not match this run.")
                else:
                    if context and visible[name].requires_approval:
                        cache_key = (context.owner_id, context.run_id, name, content_digest(arguments))
                    if cache_key is not None and cache_key in self._approval_results:
                        normalized = copy.deepcopy(self._approval_results[cache_key])
                        cache_hit = True
                    else:
                        submitted = True
                        result = self.client.call_tool(name, arguments)
                        normalized = normalize_mcp_result(result)
                        if normalized["status"] == "error" and visible[name].requires_approval:
                            normalized = tool_error("mcp_write_error", "Check the remote operation before retrying.", unknown=True)
                        self._cache_approval_result(cache_key, normalized, context)
            self.audit_logger.record(
                event="tool_call",
                tool=name,
                status=normalized["status"],
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                request_id=request_id,
                cache_hit=cache_hit,
            )
            return normalized
        except (RunDeadlineExceeded, ContextBudgetExceeded):
            raise
        except Exception as exc:
            spec = TOOL_REGISTRY.get(name)
            validation = not submitted and isinstance(exc, (ValueError, TypeError))
            normalized = tool_error("validation_error" if validation else "mcp_transport_error",
                "Invalid tool arguments." if validation else "MCP transport failed; operation outcome may be unknown.",
                unknown=not validation and (spec is None or spec.risk_level != "low"))
            self._cache_approval_result(cache_key, normalized, context)
            self.audit_logger.record(
                event="tool_call",
                tool=name,
                status=normalized["status"],
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                request_id=request_id,
                error_type=type(exc).__name__,
            )
            return normalized

    def _cache_approval_result(self, key, result: dict, context) -> None:
        if key is None or context is None:
            return
        capacity = max(1, min(context.max_tool_calls, 128))
        if key not in self._approval_results and len(self._approval_results) >= capacity:
            self._approval_results.pop(next(iter(self._approval_results)))
        self._approval_results[key] = copy.deepcopy(result)


class MCPAuditLogger:
    """Append-only JSONL audit logger for MCP tool calls."""

    def __init__(self, path: str | Path, enabled: bool = True):
        self.path = Path(path)
        self.enabled = enabled

    @classmethod
    def from_settings(cls) -> "MCPAuditLogger":
        from agents.execution_scope import current_execution_scope
        scope = current_execution_scope()
        if scope is not None and scope.evaluation:
            return cls(path=Path(scope.run_dir) / 'mcp-audit.jsonl', enabled=True)
        return cls(path=cfg.MCP_AUDIT_LOG_PATH, enabled=cfg.ENABLE_MCP_AUDIT)

    def record(self, **event: Any) -> None:
        if not self.enabled:
            return
        row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "event": "tool_call", **safe_event(event)}
        try:
            from agents.log_storage import append_jsonl
            append_jsonl(self.path,row,max_bytes=getattr(cfg,'LOG_MAX_BYTES',5_000_000),
                         backups=getattr(cfg,'LOG_BACKUP_COUNT',3))
        except (OSError, ValueError):
            # An audit failure must never turn a completed write into a retryable tool failure.
            import logging
            logging.getLogger(__name__).warning("MCP audit write failed")

    @staticmethod
    def load_events(
        path: str | Path | None = None,
        tool: str | None = None,
        status: str | None = None,
        limit: int = 100,
        max_bytes: int = 1_000_000,
    ) -> list[dict[str, Any]]:
        """Load MCP audit events, optionally filtering by tool and status."""
        audit_path = Path(path or cfg.MCP_AUDIT_LOG_PATH)
        from agents.log_storage import read_jsonl_tail
        return read_jsonl_tail(audit_path,limit=limit,max_bytes=max_bytes,
                                predicate=lambda row:(not tool or row.get('tool')==tool)
                                and (not status or row.get('status')==status))


class StreamableHttpMCPClient:
    """Small synchronous facade over the MCP SDK streamable HTTP client."""

    def __init__(self, server_url: str, timeout: int | None = None):
        self.server_url = server_url
        self.timeout = timeout or cfg.LLM_TIMEOUT

    def headers(self) -> dict[str, str]:
        headers = {"X-Email-Agent-Client": "email-rag-agent"}
        if cfg.MCP_AUTH_TOKEN:
            headers["Authorization"] = f"Bearer {cfg.MCP_AUTH_TOKEN}"
        return headers

    def _run(self, coro_factory: Callable[[], Any]) -> Any:
        box: dict[str, Any] = {}
        timeout = remaining_timeout(self.timeout)
        inherited = contextvars.copy_context()

        async def run_bounded():
            box["loop"] = asyncio.get_running_loop()
            box["task"] = asyncio.current_task()
            return await asyncio.wait_for(coro_factory(), timeout=timeout)

        def worker() -> None:
            try:
                box["result"] = asyncio.run(run_bounded())
            except BaseException as exc:
                box["error"] = exc

        thread = threading.Thread(target=lambda: inherited.run(worker), daemon=True)
        thread.start()
        thread.join(timeout + 0.05)
        if thread.is_alive():
            # Cooperative local cancellation; this is not proof a remote operation stopped.
            event_loop, task = box.get("loop"), box.get("task")
            if event_loop is not None and task is not None:
                try:
                    event_loop.call_soon_threadsafe(task.cancel)
                except RuntimeError:
                    pass
            raise TimeoutError("MCP wait timed out; remote outcome is unknown")
        if "error" in box:
            if isinstance(box["error"], (asyncio.CancelledError, asyncio.TimeoutError)):
                raise TimeoutError("MCP request cancelled; remote outcome is unknown")
            raise box["error"]
        return box.get("result")

    async def _list_tools_async(self) -> Any:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        async with streamablehttp_client(
            self.server_url,
            headers=self.headers(),
            timeout=remaining_timeout(self.timeout),
        ) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                return await session.list_tools()

    async def _call_tool_async(self, name: str, arguments: dict) -> Any:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        async with streamablehttp_client(
            self.server_url,
            headers=self.headers(),
            timeout=remaining_timeout(self.timeout),
        ) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                return await session.call_tool(name, arguments=arguments)

    def list_tools(self) -> Any:
        return self._run(self._list_tools_async)

    def call_tool(self, name: str, arguments: dict) -> Any:
        return self._run(lambda: self._call_tool_async(name, arguments))


def create_mcp_backend_from_settings() -> MCPToolBackend:
    return MCPToolBackend(StreamableHttpMCPClient(cfg.MCP_SERVER_URL))
