"""Tests for the shared agent tool registry.

The registry is the single source that will later feed both OpenAI-style
function calling and MCP tool registration.
"""

import agents.tools as tools_mod
from agents.tool_registry import TOOL_REGISTRY, openai_tool_schemas, tool_dispatch


def test_registry_contains_existing_email_tools():
    assert set(TOOL_REGISTRY) == {
        "search_emails",
        "get_email",
        "summarize_emails",
        "draft_reply",
        "send_email",
        "email_stats",
    }


def test_registry_marks_send_email_as_high_risk_approval_tool():
    spec = TOOL_REGISTRY["send_email"]

    assert spec.requires_approval is True
    assert spec.risk_level == "high"


def test_openai_schemas_preserve_existing_function_calling_shape():
    schemas = openai_tool_schemas()
    schema_names = {schema["function"]["name"] for schema in schemas}

    assert schema_names == set(TOOL_REGISTRY)
    for schema in schemas:
        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["name"] in TOOL_REGISTRY
        assert fn["description"]
        assert fn["parameters"]["type"] == "object"
        assert "properties" in fn["parameters"]


def test_tools_module_uses_registry_outputs():
    assert tools_mod.TOOL_SCHEMAS == openai_tool_schemas()
    assert tools_mod.TOOL_DISPATCH == tool_dispatch()
