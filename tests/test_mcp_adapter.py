"""Tests for adapting MCP tools to the existing agent-loop tool backend."""

from types import SimpleNamespace

from agents.mcp_adapter import MCPToolBackend, mcp_tool_to_openai_schema


def test_mcp_tool_converts_to_openai_tool_schema():
    tool = SimpleNamespace(
        name="search_emails",
        description="Search emails",
        inputSchema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )

    assert mcp_tool_to_openai_schema(tool) == {
        "type": "function",
        "function": {
            "name": "search_emails",
            "description": "Search emails",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }


def test_mcp_backend_lists_tools_as_openai_schemas():
    class FakeClient:
        def list_tools(self):
            return SimpleNamespace(tools=[
                {
                    "name": "email_stats",
                    "description": "Stats",
                    "inputSchema": {"type": "object", "properties": {}},
                }
            ])

    backend = MCPToolBackend(FakeClient())

    assert backend.tool_schemas()[0]["function"]["name"] == "email_stats"


def test_mcp_backend_call_tool_returns_structured_content():
    class FakeClient:
        def call_tool(self, name, arguments):
            assert name == "email_stats"
            assert arguments == {}
            return SimpleNamespace(structuredContent={"total_emails": 3})

    backend = MCPToolBackend(FakeClient())

    out = backend.call_tool("email_stats", {})
    assert out["status"] == "success"
    assert out["data"] == {"total_emails": 3}


def test_mcp_backend_call_tool_captures_errors():
    class FakeClient:
        def call_tool(self, name, arguments):
            raise RuntimeError("server down")

    backend = MCPToolBackend(FakeClient())

    out = backend.call_tool("email_stats", {})
    assert out["status"] == "error"
    assert out["error_code"] == "mcp_transport_error"
    assert "server down" not in out["error"]
