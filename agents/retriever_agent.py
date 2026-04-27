"""
RetrieverAgent: extracts filter conditions from natural language,
performs hybrid retrieval + reranking, and generates a grounded answer.
"""
import json
import logging
from datetime import datetime, timedelta

from openai import OpenAI

from models.schemas import AgentRequest, AgentResponse, SearchResult
from core.retriever import hybrid_search
from core.reranker import rerank
from core.generator import generate_answer
import config.settings as cfg

logger = logging.getLogger(__name__)

_FILTER_SYSTEM = """从用户问题中提取邮件检索的过滤条件，以JSON格式返回：
{
  "query": "用于语义搜索的核心查询语句",
  "sender": "发件人过滤关键词（可选，没有则为空字符串）",
  "date_hint": "日期提示（可选，如'本周''上月'，没有则为空字符串）",
  "labels": ["标签列表（可选，没有则为空数组）"]
}
只返回JSON，不要解释。"""


class RetrieverAgent:
    def __init__(self):
        self._client = OpenAI(api_key=cfg.DEEPSEEK_API_KEY, base_url=cfg.DEEPSEEK_BASE_URL)

    def _extract_filters(self, query: str) -> dict:
        try:
            resp = self._client.chat.completions.create(
                model=cfg.DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": _FILTER_SYSTEM},
                    {"role": "user", "content": query},
                ],
                temperature=0,
                max_tokens=256,
            )
            raw = resp.choices[0].message.content.strip()
            if "```" in raw:
                raw = raw.split("```")[1].lstrip("json").strip()
            return json.loads(raw)
        except Exception as exc:
            logger.warning(f"Filter extraction failed: {exc}")
            return {"query": query, "sender": "", "date_hint": "", "labels": []}

    def run(self, request: AgentRequest) -> AgentResponse:
        filters = self._extract_filters(request.query)
        search_query = filters.get("query") or request.query

        # Fetch extra candidates so reranker (and post-filters) have room to work
        results = hybrid_search(search_query, top_k=cfg.TOP_K * 4)

        results = _apply_sender_filter(results, filters.get("sender", ""))
        results = _apply_label_filter(results, filters.get("labels", []))
        results = _apply_date_filter(results, filters.get("date_hint", ""))

        reranked = rerank(request.query, results, top_n=cfg.RERANK_TOP_N)
        answer = generate_answer(request.query, reranked)
        return AgentResponse(answer=answer, sources=reranked)


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
