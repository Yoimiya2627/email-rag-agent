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
import inspect
import logging
from typing import Any, List

from core.retriever import hybrid_search
from core.reranker import rerank
from core.pipeline import apply_post_filters
from core.embedder import get_all_chunks
from agents.summarizer_agent import SummarizerAgent
from agents.writer_agent import WriterAgent, draft_reply_for_email
from agents.analyzer_agent import compute_email_stats
from agents.approvals import ApprovalStore
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


def draft_reply(instruction: str = "", email_id: str = "", query: str = "") -> Any:
    """Draft a reply.

    With `email_id` → draft against that exact email (precise; this is how a
    multi-step task drafts a reply per email found by search_emails).
    Otherwise → search by `query` (or `instruction`) for the target email.
    """
    if email_id:
        email = get_email(email_id)
        if isinstance(email, dict) and "error" in email:
            return email
        return draft_reply_for_email(email, instruction)
    resp = WriterAgent().run(AgentRequest(query=query or instruction))
    return resp.answer


def send_email(to: List[str], subject: str, body: str, rationale: str) -> dict:
    """Create a pending approval for a high-risk send-email action.

    The tool never sends directly.  It records the requested action and returns
    an approval id that a human can approve or reject through the API.
    """
    item = ApprovalStore().create(
        action_type="send_email",
        payload={
            "to": list(to or []),
            "subject": subject,
            "body": body,
            "rationale": rationale,
        },
        requested_by="agent",
        risk_level="high",
    )
    return {
        "status": "pending_approval",
        "approval_id": item["approval_id"],
        "message": "发送邮件属于高风险动作，已创建待审批请求，需要人工确认后才会执行。",
    }


def email_stats() -> dict:
    """Aggregate corpus statistics (top senders, label distribution, daily volume)."""
    return compute_email_stats()


# ── Schemas + dispatch ──────────────────────────────────────────────────────

from agents.tool_registry import openai_tool_schemas, tool_dispatch

TOOL_SCHEMAS = openai_tool_schemas()
TOOL_DISPATCH = tool_dispatch()


def call_tool(name: str, arguments: dict) -> Any:
    """Dispatch a tool call by name, with argument validation and error capture.

    Always returns a JSON-serializable result.  Failures — unknown tool,
    hallucinated/missing arguments, or an exception inside the tool — come back
    as ``{"error": ...}`` dicts so the agent loop can feed them to the model
    instead of crashing the whole request.
    """
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        return {"error": f"unknown tool: {name!r}; available: {sorted(TOOL_DISPATCH)}"}

    arguments = dict(arguments or {})
    params = inspect.signature(fn).parameters
    has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())

    # Drop hallucinated kwargs the function does not accept (unless it takes **kwargs).
    if not has_var_kw:
        unknown = set(arguments) - set(params)
        for k in unknown:
            arguments.pop(k, None)
        if unknown:
            logger.warning(f"tool {name!r}: dropped unknown args {sorted(unknown)}")

    # Required = positional-or-keyword / keyword-only params with no default.
    missing = [
        p
        for p, param in params.items()
        if param.default is inspect.Parameter.empty
        and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        and p not in arguments
    ]
    if missing:
        return {"error": f"tool {name!r} missing required argument(s): {missing}"}

    try:
        return fn(**arguments)
    except Exception as exc:
        logger.warning(f"tool {name!r} raised: {exc}")
        return {"error": f"tool {name!r} failed: {exc}"}
