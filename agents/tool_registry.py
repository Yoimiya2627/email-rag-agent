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
        description="在邮件库中按语义+关键词混合检索邮件，可选按发件人、相对日期、标签过滤。返回固定候选快照的分页 items、next_cursor 和覆盖范围；耗尽候选不代表全邮箱。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索查询（自然语言或关键词）"},
                "sender": {"type": "string", "description": "发件人过滤关键词，可选"},
                "date_hint": {"type": "string", "description": "相对日期，如 '本周' '上月' '最近'，可选"},
                "labels": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                    "description": "标签过滤列表，可选；null 表示不按标签过滤",
                },
                "cursor": {"type": ["string", "null"], "maxLength": 180, "description": "上一页 next_cursor；续页须保持同样 query 和过滤条件"},
                "limit": {"type": ["integer", "null"], "minimum": 1, "maximum": 50, "description": "返回结果数上限，可选（1–50）；null 使用配置默认值"},
            },
            "required": ["query"],
        },
    ),
    "get_email": ToolSpec(
        name="get_email",
        function_name="get_email",
        description="按 email_id 和可选 chunk_id 分页读取索引文本；未读尾部按 next_start 续读并传 source_version；不是原始 MIME 或附件内容。",
        parameters={
            "type": "object",
            "properties": {
                "email_id": {"type": "string", "description": "邮件 id（来自 search_emails 结果）"},
                "chunk_id": {"type": ["string", "null"], "maxLength": 500, "description": "准确的引用片段 ID；不传时按邮件索引正文分页"},
                "start": {"type": "integer", "minimum": 0, "maximum": 100000000, "description": "当前正文/片段字符起点，默认 0；续页用 next_start"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 4000, "description": "本页字符上限，默认 1200"},
                "source_version": {"type": ["string", "null"], "maxLength": 500, "description": "上一页或引用版本；变更时明确失败"},
                "source_sha256": {"type": ["string", "null"], "maxLength": 64, "description": "已知的原索引正文 hash，用于版本核验"},
            },
            "required": ["email_id"],
        },
    ),
    "get_thread": ToolSpec(
        name="get_thread", function_name="get_thread",
        description="按 thread_id 分页查看当前索引中的线程顺序和回复关系。日期最新不自动代表结论有效；用 get_email 读取所需正文后核对冲突。",
        parameters={"type": "object", "properties": {
            "thread_id": {"type": "string", "description": "来自邮件读取结果的 thread_id"},
            "start": {"type": "integer", "minimum": 0, "maximum": 100000000},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            "source_version": {"type": ["string", "null"], "maxLength": 500},
        }, "required": ["thread_id"]},
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
        description="对明确筛选范围或指定 email_ids 作确定性统计；返回总计、展示上限与覆盖范围。统计只覆盖当前索引，不能推断全部邮箱或未回复业务状态。",
        parameters={"type": "object", "properties": {
            "sender": {"type": "string", "description": "明确的发件人范围"},
            "date_hint": {"type": "string", "description": "日历或 ISO 日期范围"},
            "labels": {"type": ["array", "null"], "items": {"type": "string"}, "description": "需要全部满足的标签"},
            "email_ids": {"type": ["array", "null"], "items": {"type": "string"}, "description": "可选，已选邮件 ID；不传则对全部符合过滤的索引邮件计算"},
        }, "additionalProperties": False},
    ),
}


TOOL_REGISTRY.update({
    "search_history": ToolSpec(name="search_history", function_name="search_history",
        description="只读检索当前可信会话历史；历史助手陈述不是已验证邮件证据；无可信会话时不可用。",
        parameters={"type":"object", "properties":{"query":{"type":"string"}, "limit":{"type":"integer","minimum":1,"maximum":10}}, "required":["query"]}),
    "get_turn": ToolSpec(name="get_turn", function_name="get_turn",
        description="按 turn_id 分页回读当前会话用户原话或历史答复；保留失败/完成状态，不授予邮件证据或操作权限。",
        parameters={"type":"object", "properties":{"turn_id":{"type":"string"}, "field":{"type":"string","enum":["query","answer"]}, "offset":{"type":"integer","minimum":0,"maximum":100000000}, "limit":{"type":"integer","minimum":1,"maximum":4000}}, "required":["turn_id"]}),
    "get_tool_result": ToolSpec(name="get_tool_result", function_name="get_tool_result",
        description="分页回读本次运行的不可变工具结果 JSON 文本快照；不是新邮件证据，引用原邮件需 get_email 重新核对版本。",
        parameters={"type":"object", "properties":{"result_id":{"type":"string"}, "start":{"type":"integer","minimum":0,"maximum":100000000}, "limit":{"type":"integer","minimum":1,"maximum":4000}}, "required":["result_id"]}),
})


# These constraints are enforced by the dispatcher as well as shown to models.
for _spec in TOOL_REGISTRY.values():
    _spec.parameters["additionalProperties"] = False
    for _name, _parameter in _spec.parameters.get("properties", {}).items():
        if _parameter.get("type") == "string":
            _parameter["maxLength"] = 20000 if _name == "body" else 4000
            if _name in _spec.parameters.get("required", []):
                _parameter["minLength"] = 1
        elif _parameter.get("type") == "array" or "array" in _parameter.get("type", []):
            _parameter["maxItems"] = 50
            _parameter["items"].update({"minLength": 1, "maxLength": 320})
            if _name == "to":
                _parameter["minItems"] = 1


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
