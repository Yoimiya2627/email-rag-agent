"""Task-oriented skill profiles for the function-calling email agent.

Skills sit above tools: a skill is a named bundle of allowed tools plus a short
planner instruction.  The agent loop still uses the same local/MCP tool
backend; this layer only narrows the exposed tool set and gives the planner a
task-specific operating mode.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from agents.tool_registry import TOOL_REGISTRY


@dataclass(frozen=True)
class AgentSkill:
    name: str
    description: str
    tool_names: tuple[str, ...]
    planner_instruction: str

    def allows(self, tool_name: str) -> bool:
        return tool_name in self.tool_names


SKILL_REGISTRY: dict[str, AgentSkill] = {
    "general": AgentSkill(
        name="general",
        description="Default agent mode with all email tools available.",
        tool_names=tuple(TOOL_REGISTRY.keys()),
        planner_instruction="当前 Skill=general：根据用户任务自行选择检索、摘要、起草、统计或发信申请工具。",
    ),
    "mail_search": AgentSkill(
        name="mail_search",
        description="Read-only search/detail/statistics mode for factual email questions.",
        tool_names=("search_emails", "get_email", "email_stats"),
        planner_instruction=(
            "当前 Skill=mail_search：只做只读查询。优先 search_emails；摘要不足时再 get_email；"
            "统计类问题使用 email_stats；不要起草或申请发送邮件。"
        ),
    ),
    "reply_drafting": AgentSkill(
        name="reply_drafting",
        description="Find relevant emails and draft or request replies through the approval path.",
        tool_names=("search_emails", "get_email", "draft_reply", "send_email"),
        planner_instruction=(
            "当前 Skill=reply_drafting：先检索并确认目标邮件，再起草回复。"
            "send_email 是高风险工具，只能创建 pending approval，不会直接发送。"
        ),
    ),
    "mail_digest": AgentSkill(
        name="mail_digest",
        description="Summarize a topic across multiple emails and optionally inspect stats.",
        tool_names=("search_emails", "summarize_emails", "email_stats"),
        planner_instruction=(
            "当前 Skill=mail_digest：面向主题汇总。优先 summarize_emails；需要范围或数量时可使用"
            " search_emails / email_stats；不要申请发送邮件。"
        ),
    ),
}


def list_agent_skills() -> list[dict[str, Any]]:
    """Return public skill metadata for docs, UI, or MCP resources."""
    return [
        {
            "name": skill.name,
            "description": skill.description,
            "tool_names": list(skill.tool_names),
        }
        for skill in SKILL_REGISTRY.values()
    ]


def get_agent_skill(name: str | None) -> AgentSkill:
    if not name:
        return SKILL_REGISTRY["general"]
    return SKILL_REGISTRY.get(str(name).strip().lower(), SKILL_REGISTRY["general"])


def resolve_agent_skill(query: str, context: Mapping[str, Any] | None = None) -> AgentSkill:
    """Resolve a skill from explicit context first, then conservative heuristics."""
    explicit = (context or {}).get("skill")
    if explicit:
        return get_agent_skill(str(explicit))

    text = (query or "").lower()
    if any(token in text for token in ("起草", "回复", "回信", "draft", "reply", "发送")):
        return SKILL_REGISTRY["reply_drafting"]
    if any(token in text for token in ("总结", "摘要", "汇总", "digest", "summary", "summarize")):
        return SKILL_REGISTRY["mail_digest"]
    if any(token in text for token in ("谁发", "多少封", "统计", "查", "找", "检索", "search")):
        return SKILL_REGISTRY["mail_search"]
    return SKILL_REGISTRY["general"]


def filter_tool_schemas(tool_schemas: Sequence[dict], skill: AgentSkill) -> list[dict]:
    """Filter OpenAI-compatible tool schemas by skill-allowed tool names."""
    allowed = set(skill.tool_names)
    return [
        schema
        for schema in tool_schemas
        if schema.get("function", {}).get("name") in allowed
    ]


class SkillToolBackend:
    """Restrict an existing local or MCP tool backend to one skill profile."""

    def __init__(self, backend: Any, skill: AgentSkill):
        self.backend = backend
        self.skill = skill

    def tool_schemas(self) -> list[dict]:
        schemas = self.backend.tool_schemas()
        if self.skill.name == "general":
            return schemas
        return filter_tool_schemas(schemas, self.skill)

    def call_tool(self, name: str, arguments: dict) -> Any:
        if not self.skill.allows(name):
            return {
                "error": (
                    f"tool {name!r} is not allowed by skill {self.skill.name!r}; "
                    f"allowed tools: {list(self.skill.tool_names)}"
                )
            }
        return self.backend.call_tool(name, arguments)
