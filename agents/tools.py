"""
Tool layer for the email agent.

Each tool is a plain Python function plus an OpenAI-format JSON schema:
  - TOOL_SCHEMAS  — passed to the LLM as the `tools` param
  - TOOL_DISPATCH — maps a tool name to its implementation
  - call_tool()   — name + parsed-args → result

The agent loop (agents/agent_loop.py, Step 3) selects and invokes tools
through this layer.  Tools wrap existing capabilities — `search_emails` reuses
core.retriever / core.pipeline, the rest reuse the specialist-agent logic — so
the agent and the legacy /chat agents stay behind one implementation.

call_tool() here is intentionally minimal; argument validation, error-feedback
and loop-safety guardrails are added in Step 5.
"""
import logging
from typing import Any, List

from core.retriever import hybrid_search
from core.reranker import rerank
from core.pipeline import apply_post_filters
from core.embedder import get_all_chunks
from agents.summarizer_agent import SummarizerAgent
from agents.writer_agent import WriterAgent
from agents.analyzer_agent import compute_email_stats
from models.schemas import AgentRequest, SearchResult
import config.settings as cfg

logger = logging.getLogger(__name__)


# ── Tool implementations ────────────────────────────────────────────────────

def _format_hit(r: SearchResult) -> dict:
    """Compact, JSON-serializable view of a search result for the LLM."""
    m = r.metadata
    return {
        "email_id": r.email_id,
        "chunk_id": r.chunk_id,
        "subject": m.get("subject", ""),
        "sender": m.get("sender", ""),
        "date": m.get("date", ""),
        "snippet": r.content[:300],
        "score": round(float(r.score), 4),
    }


def search_emails(
    query: str,
    sender: str = "",
    date_hint: str = "",
    labels: List[str] = None,
    limit: int = None,
) -> List[dict]:
    """Hybrid (vector + BM25) search over the email corpus, with optional
    sender / date / label post-filters.  Returns compact hit dicts.
    """
    limit = cfg.RERANK_TOP_N if limit is None else limit
    results = hybrid_search(query, top_k=cfg.TOP_K * 4)
    results = apply_post_filters(
        results, {"sender": sender, "date_hint": date_hint, "labels": labels or []}
    )
    reranked = rerank(query, results, top_n=limit)
    return [_format_hit(r) for r in reranked]


def get_email(email_id: str) -> dict:
    """Fetch one full email by id (all chunks joined in order)."""
    chunks = [c for c in get_all_chunks() if c["metadata"].get("email_id") == email_id]
    if not chunks:
        return {"error": f"email_id {email_id!r} not found"}
    chunks.sort(key=lambda c: c["metadata"].get("chunk_index", 0))
    m = chunks[0]["metadata"]
    return {
        "email_id": email_id,
        "subject": m.get("subject", ""),
        "sender": m.get("sender", ""),
        "date": m.get("date", ""),
        "body": "\n".join(c["content"] for c in chunks),
    }


def summarize_emails(query: str) -> str:
    """Retrieve emails relevant to `query` and return a structured summary."""
    resp = SummarizerAgent().run(AgentRequest(query=query))
    return resp.answer


def draft_reply(query: str) -> str:
    """Find the most relevant original email and draft a reply per `query`."""
    resp = WriterAgent().run(AgentRequest(query=query))
    return resp.answer


def email_stats() -> dict:
    """Aggregate corpus statistics (top senders, label distribution, daily volume)."""
    return compute_email_stats()


# ── Schemas + dispatch ──────────────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_emails",
            "description": "在邮件库中按语义+关键词混合检索邮件，可选按发件人、相对日期、标签过滤。返回匹配邮件的摘要列表。",
            "parameters": {
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
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_email",
            "description": "按 email_id 获取一封邮件的完整内容（当检索摘要不足以回答时使用）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "email_id": {"type": "string", "description": "邮件 id（来自 search_emails 结果）"},
                },
                "required": ["email_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_emails",
            "description": "检索与某主题相关的多封邮件并生成结构化综合摘要。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要摘要的主题或问题"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "draft_reply",
            "description": "找到最相关的原始邮件并起草一封回信。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "回信要求，如 '帮我回复 Bob 的询价邮件，礼貌拒绝'"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "email_stats",
            "description": "返回邮件库的聚合统计（发件人 Top5、标签分布、每日邮件量）。无需参数。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

TOOL_DISPATCH = {
    "search_emails": search_emails,
    "get_email": get_email,
    "summarize_emails": summarize_emails,
    "draft_reply": draft_reply,
    "email_stats": email_stats,
}


def call_tool(name: str, arguments: dict) -> Any:
    """Dispatch a tool call by name. Returns the tool result (JSON-serializable).

    Minimal for now — Step 5 adds argument validation and error-feedback so a
    bad call is fed back to the model instead of crashing the loop.
    """
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        return {"error": f"unknown tool: {name!r}"}
    return fn(**(arguments or {}))
