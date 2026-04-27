"""
RetrieverAgent: extracts filter conditions from natural language,
performs hybrid retrieval + reranking, and generates a grounded answer.
"""
import json
import logging

from openai import OpenAI

from models.schemas import AgentRequest, AgentResponse
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

        # Fetch extra candidates so reranker has room to work
        results = hybrid_search(search_query, top_k=cfg.TOP_K * 2)

        # Optional post-filter by sender keyword
        sender_kw = filters.get("sender", "").strip().lower()
        if sender_kw:
            filtered = [r for r in results if sender_kw in r.metadata.get("sender", "").lower()]
            results = filtered or results  # fall back if filter removes everything

        reranked = rerank(request.query, results, top_n=cfg.RERANK_TOP_N)
        answer = generate_answer(request.query, reranked)
        return AgentResponse(answer=answer, sources=reranked)
