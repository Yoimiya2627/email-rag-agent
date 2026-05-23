"""Adapters between MCP tools and the existing agent loop.

The agent loop is synchronous today, so the live Streamable HTTP client runs
MCP SDK async calls inside a short-lived worker thread.  The default backend
remains local; MCP is only used when `AGENT_TOOL_BACKEND=mcp`.
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import config.settings as cfg


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
    if isinstance(result, dict):
        if "structuredContent" in result:
            return result["structuredContent"]
        if "structured_content" in result:
            return result["structured_content"]
        return result

    structured = _get_field(result, "structuredContent", "structured_content")
    if structured is not None:
        return structured

    content = _get_field(result, "content", default=None)
    if content:
        texts = []
        for item in content:
            text = _get_field(item, "text", default=None)
            if text is not None:
                texts.append(text)
        if len(texts) == 1:
            try:
                return json.loads(texts[0])
            except json.JSONDecodeError:
                return texts[0]
        if texts:
            return "\n".join(texts)

    return result


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
        return self._call_tool_fn(name, arguments)


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
        self._schema_cache = [mcp_tool_to_openai_schema(tool) for tool in tools]
        self._schema_cache_at = now
        return self._schema_cache

    def refresh_tools(self) -> list[dict]:
        self._schema_cache = None
        self._schema_cache_at = 0.0
        return self.tool_schemas()

    def call_tool(self, name: str, arguments: dict) -> Any:
        started = time.perf_counter()
        request_id = str(uuid.uuid4())
        try:
            result = self.client.call_tool(name, arguments or {})
            normalized = normalize_mcp_result(result)
            self.audit_logger.record(
                event="tool_call",
                tool=name,
                status="success",
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                request_id=request_id,
            )
            return normalized
        except Exception as exc:
            self.audit_logger.record(
                event="tool_call",
                tool=name,
                status="error",
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                request_id=request_id,
                error=str(exc),
            )
            return {"error": f"mcp tool {name!r} failed: {exc}"}


class MCPAuditLogger:
    """Append-only JSONL audit logger for MCP tool calls."""

    def __init__(self, path: str | Path, enabled: bool = True):
        self.path = Path(path)
        self.enabled = enabled

    @classmethod
    def from_settings(cls) -> "MCPAuditLogger":
        return cls(path=cfg.MCP_AUDIT_LOG_PATH, enabled=cfg.ENABLE_MCP_AUDIT)

    def record(self, **event: Any) -> None:
        if not self.enabled:
            return
        row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **event}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


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

        def worker() -> None:
            try:
                box["result"] = asyncio.run(coro_factory())
            except Exception as exc:
                box["error"] = exc

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(self.timeout)
        if thread.is_alive():
            raise TimeoutError(f"MCP request timed out after {self.timeout}s")
        if "error" in box:
            raise box["error"]
        return box.get("result")

    async def _list_tools_async(self) -> Any:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        async with streamablehttp_client(
            self.server_url,
            headers=self.headers(),
            timeout=self.timeout,
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
            timeout=self.timeout,
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
