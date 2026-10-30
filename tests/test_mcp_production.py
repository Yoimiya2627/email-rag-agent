"""Production-hardening tests for the MCP backend."""

import asyncio
import json
from types import SimpleNamespace

from agents.mcp_adapter import MCPAuditLogger, MCPToolBackend, StreamableHttpMCPClient


def test_streamable_http_client_sends_bearer_token_when_configured(monkeypatch):
    import config.settings as cfg

    monkeypatch.setattr(cfg, "MCP_AUTH_TOKEN", "secret-token", raising=False)

    client = StreamableHttpMCPClient("http://127.0.0.1:8001/mcp")

    assert client.headers()["Authorization"] == "Bearer secret-token"


def test_mcp_backend_caches_tool_schemas_until_refresh():
    class FakeClient:
        def __init__(self):
            self.calls = 0

        def list_tools(self):
            self.calls += 1
            return SimpleNamespace(tools=[
                {
                    "name": "email_stats",
                    "description": "Stats",
                    "inputSchema": {"type": "object", "properties": {}},
                }
            ])

        def call_tool(self, name, arguments):
            return {}

    client = FakeClient()
    backend = MCPToolBackend(client, schema_cache_seconds=60)

    first = backend.tool_schemas()
    second = backend.tool_schemas()
    backend.refresh_tools()
    third = backend.tool_schemas()

    assert first == second == third
    assert client.calls == 2


def test_mcp_audit_logger_writes_jsonl(tmp_path):
    path = tmp_path / "mcp_audit.jsonl"
    logger = MCPAuditLogger(path=path, enabled=True)

    logger.record(
        event="tool_call",
        tool="email_stats",
        status="success",
        latency_ms=12,
        request_id="req-1",
    )

    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert row["event"] == "tool_call"
    assert row["tool"] == "email_stats"
    assert row["status"] == "success"
    assert row["latency_ms"] == 12
    assert row["request_id"] == "req-1"
    assert "ts" in row


def test_static_bearer_token_verifier_accepts_only_configured_token():
    from mcp_server import StaticBearerTokenVerifier

    verifier = StaticBearerTokenVerifier(token="secret-token")

    valid = asyncio.run(verifier.verify_token("secret-token"))
    invalid = asyncio.run(verifier.verify_token("wrong-token"))

    assert valid is not None
    assert valid.client_id == "email-rag-agent"
    assert invalid is None


def test_build_server_enables_auth_when_token_configured(monkeypatch):
    import mcp_server

    class FakeFastMCP:
        def __init__(self, name: str, **kwargs):
            self.name = name
            self.kwargs = kwargs

        def tool(self, name=None, description=None):
            return lambda fn: fn

        def resource(self, uri):
            return lambda fn: fn

        def prompt(self, name=None, description=None):
            return lambda fn: fn

    monkeypatch.setattr(mcp_server.cfg, "MCP_AUTH_TOKEN", "secret-token", raising=False)

    server = mcp_server.build_server(server_factory=FakeFastMCP)

    assert server.kwargs["token_verifier"] is not None
    assert server.kwargs["auth"] is not None
