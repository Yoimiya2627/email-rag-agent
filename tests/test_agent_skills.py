import pytest

from agents.tool_registry import openai_tool_schemas


def test_skill_registry_defines_task_oriented_tool_sets():
    from agents.skills import SKILL_REGISTRY

    assert set(SKILL_REGISTRY) == {"general", "mail_search", "reply_drafting", "mail_digest"}
    assert SKILL_REGISTRY["mail_search"].tool_names == (
        "search_emails",
        "get_email",
        "email_stats",
    )
    assert "send_email" in SKILL_REGISTRY["reply_drafting"].tool_names
    assert "summarize_emails" in SKILL_REGISTRY["mail_digest"].tool_names


def test_resolve_skill_prefers_explicit_context_over_query_heuristics():
    from agents.skills import resolve_agent_skill

    skill = resolve_agent_skill(
        "帮我起草一封回复",
        context={"skill": "mail_search"},
    )

    assert skill.name == "mail_search"


def test_resolve_skill_uses_safe_query_heuristics_when_no_context():
    from agents.skills import resolve_agent_skill

    assert resolve_agent_skill("帮我起草一封回复确认收到").name == "reply_drafting"
    assert resolve_agent_skill("总结一下最近项目进展邮件").name == "mail_digest"
    assert resolve_agent_skill("Q3 预算会议是谁发的").name == "mail_search"


def test_filter_tool_schemas_keeps_only_skill_allowed_tools():
    from agents.skills import SKILL_REGISTRY, filter_tool_schemas

    schemas = filter_tool_schemas(openai_tool_schemas(), SKILL_REGISTRY["mail_search"])
    names = {schema["function"]["name"] for schema in schemas}

    assert names == {"search_emails", "get_email", "email_stats"}


def test_skill_tool_backend_blocks_disallowed_tool_calls():
    from agents.skills import SKILL_REGISTRY, SkillToolBackend

    class FakeBackend:
        def tool_schemas(self):
            return openai_tool_schemas()

        def call_tool(self, name, arguments):
            return {"called": name, "arguments": arguments}

    backend = SkillToolBackend(FakeBackend(), SKILL_REGISTRY["mail_search"])

    assert backend.call_tool("email_stats", {}) == {"called": "email_stats", "arguments": {}}
    blocked = backend.call_tool("send_email", {"to": ["a@example.com"]})
    assert blocked["error"].startswith("tool 'send_email' is not allowed")
