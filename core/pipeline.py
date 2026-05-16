"""
Shared retrieval pipeline — the single source of truth for
"query → grounded contexts".

Before this module the pipeline (rewrite → extract filters → hybrid search →
post-filter → rerank) was re-implemented in three places (RetrieverAgent,
run_ragas_eval, measure_latency), and two of them silently skipped the
sender/date/label post-filters — so the RAGAS evaluation was not measuring the
pipeline the product actually serves.  Everything now goes through `retrieve()`.
"""
import json
import logging
from datetime import datetime, timedelta
from typing import List

from openai import OpenAI

from models.schemas import SearchResult
from core.retriever import hybrid_search
from core.reranker import rerank
import config.settings as cfg

logger = logging.getLogger(__name__)

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=cfg.DEEPSEEK_API_KEY, base_url=cfg.DEEPSEEK_BASE_URL)
    return _client


# ── Query rewrite ───────────────────────────────────────────────────────────

_REWRITE_SYSTEM = """你是一个搜索查询优化专家。请将用户的口语化问题改写为更适合语义搜索的查询语句。
要求：
1. 展开缩写和指代词（如"它""这个项目"→具体名称）
2. 补充隐含的关键词（如"最近的问题"→"最近发生的技术问题或业务问题"）
3. 保持核心意图不变
4. 只返回改写后的查询语句，不要解释。"""


def rewrite_query(query: str) -> str:
    """Rewrite a colloquial query into a retrieval-friendly one.

    No-op when ENABLE_QUERY_REWRITE is off; falls back to the original query on
    any failure or empty response.
    """
    if not cfg.ENABLE_QUERY_REWRITE:
        return query
    try:
        resp = _get_client().chat.completions.create(
            model=cfg.DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": _REWRITE_SYSTEM},
                {"role": "user", "content": query},
            ],
            temperature=0,
            # 推理模型需要给推理过程 + 答案都留足空间，否则 content 空
            max_tokens=1500,
            timeout=cfg.LLM_TIMEOUT,
        )
        choice = resp.choices[0]
        rewritten = (choice.message.content or "").strip()
        if not rewritten:
            # 兜底：从 reasoning_content 取最后一行非空文本作为改写结果
            rc = getattr(choice.message, "reasoning_content", None) or ""
            lines = [ln.strip() for ln in rc.splitlines() if ln.strip()]
            rewritten = lines[-1] if lines else ""
        if rewritten:
            logger.debug(f"Query rewritten: {query!r} → {rewritten!r}")
            return rewritten
    except Exception as exc:
        logger.warning(f"Query rewrite failed: {exc}")
    return query


# ── Filter extraction ───────────────────────────────────────────────────────

_FILTER_SYSTEM = """从用户问题中提取邮件检索的过滤条件，以JSON格式返回：
{
  "query": "用于语义搜索的核心查询语句",
  "sender": "发件人过滤关键词（可选，没有则为空字符串）",
  "date_hint": "日期提示（可选，如'本周''上月'，没有则为空字符串）",
  "labels": ["标签列表（可选，没有则为空数组）"]
}
只返回JSON，不要解释。"""


def extract_filters(query: str) -> dict:
    """Extract structured retrieval filters (sender/date/labels) from a query.

    Falls back to an all-pass filter on any failure.
    """
    try:
        resp = _get_client().chat.completions.create(
            model=cfg.DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": _FILTER_SYSTEM},
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=1500,
            timeout=cfg.LLM_TIMEOUT,
        )
        choice = resp.choices[0]
        raw = (choice.message.content or "").strip()
        if not raw:
            rc = getattr(choice.message, "reasoning_content", None) or ""
            if rc and "{" in rc and "}" in rc:
                raw = rc
            else:
                raise ValueError(f"Empty filter response (finish_reason={choice.finish_reason!r})")
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()
        s, e = raw.find("{"), raw.rfind("}") + 1
        if s >= 0 and e > s:
            raw = raw[s:e]
        return json.loads(raw)
    except Exception as exc:
        logger.warning(f"Filter extraction failed: {exc}")
        return {"query": query, "sender": "", "date_hint": "", "labels": []}


# ── Post-filters ────────────────────────────────────────────────────────────

def _apply_sender_filter(results, sender_kw: str):
    kw = (sender_kw or "").strip().lower()
    if not kw:
        return results
    filtered = [r for r in results if kw in r.metadata.get("sender", "").lower()]
    return filtered or results  # fall back if filter removes everything


def _apply_label_filter(results, labels):
    if not labels:
        return results
    wanted = {str(l).strip().lower() for l in labels if str(l).strip()}
    if not wanted:
        return results

    def has_match(r: SearchResult) -> bool:
        raw = r.metadata.get("labels", "[]")
        try:
            actual = json.loads(raw) if isinstance(raw, str) else (raw or [])
        except Exception:
            actual = []
        return any(str(a).strip().lower() in wanted for a in actual)

    filtered = [r for r in results if has_match(r)]
    return filtered or results


_RELATIVE_DATE_KEYWORDS = {
    "今天": 1, "today": 1,
    "昨天": 2, "yesterday": 2,
    "本周": 7, "这周": 7, "this week": 7,
    "最近": 14, "recent": 14, "recently": 14,
    "上周": 14, "last week": 14,
    "本月": 30, "这个月": 30, "this month": 30,
    "上月": 60, "上个月": 60, "last month": 60,
    "今年": 365, "this year": 365,
}


def _apply_date_filter(results, date_hint: str):
    hint = (date_hint or "").strip().lower()
    if not hint:
        return results

    days = None
    for kw, d in _RELATIVE_DATE_KEYWORDS.items():
        if kw in hint:
            days = d
            break
    if days is None:
        return results

    cutoff = datetime.now() - timedelta(days=days)

    def in_window(r: SearchResult) -> bool:
        date_str = r.metadata.get("date", "")
        if not date_str or len(date_str) < 10:
            return False
        try:
            dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        except ValueError:
            return False
        return dt >= cutoff

    filtered = [r for r in results if in_window(r)]
    return filtered or results


def apply_post_filters(results, filters: dict):
    """Apply sender / label / date filters extracted from the query.

    Each filter falls back to the unfiltered list if it would remove every
    candidate (better to return loosely-relevant results than nothing).
    """
    results = _apply_sender_filter(results, filters.get("sender", ""))
    results = _apply_label_filter(results, filters.get("labels", []))
    results = _apply_date_filter(results, filters.get("date_hint", ""))
    return results


# ── Full pipeline ───────────────────────────────────────────────────────────

def retrieve(query: str, *, top_n: int = None, fetch_k: int = None) -> List[SearchResult]:
    """Full retrieval pipeline: rewrite → extract filters → hybrid search →
    post-filter → rerank.

    Args:
        query:   the original user query (rerank scores against this, not the
                 rewritten form, so user intent is not diluted).
        top_n:   results kept after rerank (default cfg.RERANK_TOP_N).
        fetch_k: candidates pulled from hybrid search before post-filter and
                 rerank get room to work (default cfg.TOP_K * 4).
    """
    top_n = cfg.RERANK_TOP_N if top_n is None else top_n
    fetch_k = cfg.TOP_K * 4 if fetch_k is None else fetch_k

    rewritten = rewrite_query(query)
    filters = extract_filters(rewritten)
    search_query = filters.get("query") or rewritten

    results = hybrid_search(search_query, top_k=fetch_k)
    results = apply_post_filters(results, filters)
    return rerank(query, results, top_n=top_n)
