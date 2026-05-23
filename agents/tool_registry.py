"""Shared registry for email agent tools.

The registry is intentionally metadata-first: tool implementations stay in
``agents.tools`` so existing tests and local function-calling behavior remain
stable.  This module turns the same metadata into OpenAI-style tool schemas
today, and MCP tool registrations in the next phase.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]
    function_name: str
    output_schema: Dict[str, Any] | None = None
    risk_level: str = "low"
    requires_approval: bool = False

    def as_openai_tool(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


TOOL_REGISTRY: Dict[str, ToolSpec] = {
    "search_emails": ToolSpec(
        name="search_emails",
        function_name="search_emails",
        description="在邮件库中按语义+关键词混合检索邮件，可选按发件人、相对日期、标签过滤。返回匹配邮件的摘要列表。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索查询（自然语言或关键词）"},
                "sender": {"type": "string", "description": "发件人过滤关键词，可选"},
                "date_hint": {"type": "string", "description": "相对日期，如 '本周' '上月' '最近'，可选"},
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "标签过滤列表，可选",
                },
                "limit": {"type": "integer", "description": "返回结果数上限，可选"},
            },
            "required": ["query"],
        },
    ),
    "get_email": ToolSpec(
        name="get_email",
        function_name="get_email",
        description="按 email_id 获取一封邮件的完整内容（当检索摘要不足以回答时使用）。",
        parameters={
            "type": "object",
            "properties": {
                "email_id": {"type": "string", "description": "邮件 id（来自 search_emails 结果）"},
            },
            "required": ["email_id"],
        },
    ),
    "summarize_emails": ToolSpec(
        name="summarize_emails",
        function_name="summarize_emails",
        description="检索与某主题相关的多封邮件并生成结构化综合摘要。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "要摘要的主题或问题"},
            },
            "required": ["query"],
        },
    ),
    "draft_reply": ToolSpec(
        name="draft_reply",
        function_name="draft_reply",
        description="起草一封回信。已知具体邮件时传 email_id（精确，多步任务推荐）；否则传 query 让系统检索目标邮件。",
        parameters={
            "type": "object",
            "properties": {
                "instruction": {"type": "string", "description": "回信要求，如 '礼貌拒绝' '确认参会'"},
                "email_id": {"type": "string", "description": "要回复的邮件 id（来自 search_emails 结果），可选"},
                "query": {"type": "string", "description": "不知道 email_id 时，用于检索目标邮件的描述，可选"},
            },
            "required": ["instruction"],
        },
        risk_level="medium",
    ),
    "send_email": ToolSpec(
        name="send_email",
        function_name="send_email",
        description=(
            "申请发送邮件。高风险工具：只会创建待人工审批的 pending action，"
            "不会直接发送邮件；审批通过后由人工确认链路执行。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "to": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "收件人邮箱列表",
                },
                "subject": {"type": "string", "description": "邮件主题"},
                "body": {"type": "string", "description": "邮件正文"},
                "rationale": {
                    "type": "string",
                    "description": "为什么需要发送这封邮件，供人工审批时查看",
                },
            },
            "required": ["to", "subject", "body", "rationale"],
        },
        risk_level="high",
        requires_approval=True,
    ),
    "email_stats": ToolSpec(
        name="email_stats",
        function_name="email_stats",
        description="返回邮件库的聚合统计（发件人 Top5、标签分布、每日邮件量）。无需参数。",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
    ),
}


def openai_tool_schemas() -> list[dict]:
    """Return OpenAI/DeepSeek-compatible tool schemas."""
    return [spec.as_openai_tool() for spec in TOOL_REGISTRY.values()]


def tool_dispatch() -> dict[str, Callable[..., Any]]:
    """Resolve registry entries to the local Python implementations."""
    from agents import tools as tools_mod

    return {
        name: getattr(tools_mod, spec.function_name)
        for name, spec in TOOL_REGISTRY.items()
    }
