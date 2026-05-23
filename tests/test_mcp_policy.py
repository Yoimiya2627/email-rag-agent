"""Tests for MCP tool visibility and permission policy."""

from agents.tool_registry import TOOL_REGISTRY


def test_read_only_policy_hides_medium_high_and_approval_tools():
    from agents.tool_policy import ToolPolicy

    policy = ToolPolicy(read_only=True)
    visible = policy.visible_specs(TOOL_REGISTRY)

    assert "search_emails" in visible
    assert "get_email" in visible
    assert "summarize_emails" in visible
    assert "email_stats" in visible
    assert "draft_reply" not in visible
    assert "send_email" not in visible


def test_allowed_tools_policy_exposes_only_named_known_tools():
    from agents.tool_policy import ToolPolicy

    policy = ToolPolicy(allowed_tools={"search_emails", "email_stats", "missing_tool"})
    visible = policy.visible_specs(TOOL_REGISTRY)

    assert set(visible) == {"search_emails", "email_stats"}


def test_mcp_server_register_tools_respects_policy_filter():
    from agents.tool_policy import ToolPolicy
    from mcp_server import register_tools
    from tests.test_mcp_server import FakeFastMCP

    server = FakeFastMCP("test")
    registered = register_tools(
        server,
        policy=ToolPolicy(allowed_tools={"search_emails", "email_stats"}),
    )

    assert registered == ["search_emails", "email_stats"]
    assert {tool["name"] for tool in server.tools} == {"search_emails", "email_stats"}
