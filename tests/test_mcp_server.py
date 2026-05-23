"""Tests for the MCP server registration layer.

The tests use a fake FastMCP-compatible object so they do not require the MCP
SDK to be installed and never open a transport.
"""

from agents.tool_registry import TOOL_REGISTRY


class FakeFastMCP:
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.kwargs = kwargs
        self.tools = []
        self.resources = []
        self.prompts = []

    def tool(self, name=None, description=None):
        def decorator(fn):
            self.tools.append({
                "name": name or fn.__name__,
                "description": description,
                "fn": fn,
            })
            return fn

        return decorator

    def resource(self, uri):
        def decorator(fn):
            self.resources.append({"uri": uri, "fn": fn})
            return fn

        return decorator

    def prompt(self, name=None, description=None):
        def decorator(fn):
            self.prompts.append({
                "name": name or fn.__name__,
                "description": description,
                "fn": fn,
            })
            return fn

        return decorator


def test_register_tools_adds_every_registry_tool():
    from mcp_server import register_tools

    server = FakeFastMCP("test")
    registered = register_tools(server)

    assert registered == list(TOOL_REGISTRY)
    assert {tool["name"] for tool in server.tools} == set(TOOL_REGISTRY)
    assert all(tool["description"] for tool in server.tools)


def test_build_server_accepts_injected_factory():
    from mcp_server import build_server

    server = build_server(server_factory=FakeFastMCP)

    assert server.name == "Email RAG Agent"
    assert server.kwargs["json_response"] is True
    assert server.kwargs["host"] == "127.0.0.1"
    assert server.kwargs["port"] == 8001
    assert {tool["name"] for tool in server.tools} == set(TOOL_REGISTRY)
    assert {resource["uri"] for resource in server.resources} == {
        "email://{email_id}",
        "email-corpus://stats",
    }
    assert {prompt["name"] for prompt in server.prompts} == {
        "draft_reply_prompt",
        "summarize_emails_prompt",
    }


def test_register_resources_adds_email_resource_handlers():
    from mcp_server import register_resources

    server = FakeFastMCP("test")
    registered = register_resources(server)

    assert registered == ["email://{email_id}", "email-corpus://stats"]
    assert {resource["uri"] for resource in server.resources} == set(registered)


def test_read_email_resource_serializes_tool_result(monkeypatch):
    import mcp_server

    monkeypatch.setattr(
        mcp_server,
        "get_email",
        lambda email_id: {"email_id": email_id, "subject": "Budget"},
    )

    assert mcp_server.read_email_resource("e1") == '{"email_id": "e1", "subject": "Budget"}'


def test_register_prompts_adds_reusable_prompt_helpers():
    from mcp_server import register_prompts

    server = FakeFastMCP("test")
    registered = register_prompts(server)

    assert registered == ["draft_reply_prompt", "summarize_emails_prompt"]
    assert {prompt["name"] for prompt in server.prompts} == set(registered)
