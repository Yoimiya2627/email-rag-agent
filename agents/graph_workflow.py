"""
LangGraph-based Self-RAG workflow.

State machine:
  rewrite_query → retrieve → grade_contexts → generate
                                  ↓ (no relevant docs)
                              rewrite_query (retry, max 2)

Self-RAG checks:
  - grade_contexts: LLM scores each retrieved chunk (relevant/irrelevant)
  - If 0 relevant chunks → rewrite and retry (up to MAX_RETRIES)
  - After generation: grade_answer for hallucination; if bad → retry
"""
import json
import logging
from typing import Any, Dict, List, Optional, TypedDict

from openai import OpenAI
from langgraph.graph import StateGraph, END

import config.settings as cfg
from core.retriever import hybrid_search
from core.reranker import rerank
from core.generator import generate_answer, build_context
from models.schemas import AgentRequest, AgentResponse, SearchResult

logger = logging.getLogger(__name__)

MAX_RETRIES = 2


# ── State ─────────────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    query: str
    rewritten_query: str
    results: List[SearchResult]
    relevant_results: List[SearchResult]
    answer: str
    retry_count: int
    history: Optional[List[dict]]


# ── Node functions ─────────────────────────────────────────────────────────────

def _get_client() -> OpenAI:
    return OpenAI(api_key=cfg.DEEPSEEK_API_KEY, base_url=cfg.DEEPSEEK_BASE_URL)


def node_rewrite(state: RAGState) -> RAGState:
    """Rewrite the query to improve retrieval (skips on first pass if already rewritten)."""
    query = state["query"]
    retry = state.get("retry_count", 0)

    if not cfg.ENABLE_QUERY_REWRITE or (retry == 0 and state.get("rewritten_query")):
        state["rewritten_query"] = state.get("rewritten_query") or query
        return state

    system = (
        "将以下搜索查询改写，使其更适合语义检索。"
        + (f" 这是第{retry+1}次改写，请尝试不同角度。" if retry > 0 else "")
        + " 只返回改写后的查询，不要解释。"
    )
    try:
        resp = _get_client().chat.completions.create(
            model=cfg.DEEPSEEK_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": query}],
            temperature=0.3 * (retry + 1),
            # 推理模型预留充足空间；128 token 会被推理过程吃光导致 content 为空
            max_tokens=1500,
            timeout=cfg.LLM_TIMEOUT,
        )
        choice = resp.choices[0]
        rewritten = (choice.message.content or "").strip()
        if not rewritten:
            rc = getattr(choice.message, "reasoning_content", None) or ""
            lines = [ln.strip() for ln in rc.splitlines() if ln.strip()]
            rewritten = lines[-1] if lines else ""
        state["rewritten_query"] = rewritten or query
    except Exception as exc:
        logger.warning(f"Rewrite failed: {exc}")
        state["rewritten_query"] = query
    return state


def node_retrieve(state: RAGState) -> RAGState:
    """Hybrid search + rerank."""
    q = state.get("rewritten_query") or state["query"]
    results = hybrid_search(q, top_k=cfg.TOP_K * 4)
    reranked = rerank(state["query"], results, top_n=cfg.RERANK_TOP_N + 2)
    state["results"] = reranked
    return state


def node_grade_contexts(state: RAGState) -> RAGState:
    """LLM grades each chunk as relevant or not."""
    query = state["query"]
    results = state.get("results", [])
    if not results:
        state["relevant_results"] = []
        return state

    docs_text = "\n\n".join(f"[{i}] {r.content[:300]}" for i, r in enumerate(results))
    prompt = (
        f"问题：{query}\n\n候选段落：\n{docs_text}\n\n"
        "请判断哪些段落与问题相关，返回相关段落的索引列表（JSON数组，如[0,2,3]）。"
        "如果全部无关，返回[]。只返回JSON数组。"
    )
    try:
        resp = _get_client().chat.completions.create(
            model=cfg.DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            # 推理模型需要充足空间；64 token 会让 content 永远为空
            max_tokens=1500,
            timeout=cfg.LLM_TIMEOUT,
        )
        choice = resp.choices[0]
        raw = (choice.message.content or "").strip()
        if not raw:
            rc = getattr(choice.message, "reasoning_content", None) or ""
            if rc and "[" in rc and "]" in rc:
                raw = rc
            else:
                raise ValueError(f"Empty grade response (finish_reason={choice.finish_reason!r})")
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()
        s, e = raw.find("["), raw.rfind("]") + 1
        if s >= 0 and e > s:
            raw = raw[s:e]
        indices = json.loads(raw)
        state["relevant_results"] = [results[i] for i in indices if 0 <= i < len(results)]
    except Exception as exc:
        logger.warning(f"Context grading failed: {exc}, using all results")
        state["relevant_results"] = results
    return state


def node_generate(state: RAGState) -> RAGState:
    """Generate answer from relevant contexts."""
    results = state.get("relevant_results") or state.get("results", [])
    answer = generate_answer(state["query"], results, history=state.get("history"))
    state["answer"] = answer
    return state


def _bump_retry(state: RAGState) -> RAGState:
    """LangGraph node: increment retry counter before re-entering rewrite."""
    state["retry_count"] = state.get("retry_count", 0) + 1
    return state


def _should_retry(state: RAGState) -> str:
    """Conditional edge label: 'retry' if no relevant docs and retries remain."""
    retry = state.get("retry_count", 0)
    has_relevant = bool(state.get("relevant_results"))
    if not has_relevant and retry < MAX_RETRIES:
        return "retry"
    return "generate"


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_graph():
    """Build the Self-RAG state graph using langgraph.

    Flow:
        rewrite → retrieve → grade ─┬─relevant→ generate → END
                                    └─retry→ bump_retry → rewrite (loop)

    The conditional edge after `grade` checks both whether any relevant
    contexts were found and whether retry budget remains (MAX_RETRIES).
    """
    graph = StateGraph(RAGState)
    graph.add_node("rewrite", node_rewrite)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("grade", node_grade_contexts)
    graph.add_node("bump_retry", _bump_retry)
    graph.add_node("generate", node_generate)

    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        _should_retry,
        {"retry": "bump_retry", "generate": "generate"},
    )
    graph.add_edge("bump_retry", "rewrite")
    graph.add_edge("generate", END)

    return graph.compile()


# ── Public API ─────────────────────────────────────────────────────────────────

_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_graph(request: AgentRequest, memory=None) -> AgentResponse:
    """Run Self-RAG workflow and return AgentResponse."""
    history = memory.to_messages() if memory else None
    state: RAGState = {
        "query": request.query,
        "rewritten_query": "",
        "results": [],
        "relevant_results": [],
        "answer": "",
        "retry_count": 0,
        "history": history,
    }
    final_state = get_graph().invoke(state)
    sources = final_state.get("relevant_results") or final_state.get("results", [])
    return AgentResponse(answer=final_state["answer"], sources=sources[:cfg.RERANK_TOP_N])
