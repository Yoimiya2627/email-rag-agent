"""
SummarizerAgent: retrieves relevant emails and produces a structured summary.
"""
import logging

from openai import OpenAI

from models.schemas import AgentRequest, AgentResponse
from core.retriever import hybrid_search
from core.reranker import rerank
import config.settings as cfg

logger = logging.getLogger(__name__)

_SYSTEM = """你是一位专业的邮件摘要专家。根据检索到的邮件内容，生成清晰、结构化的综合摘要。

摘要结构要求：
1. **核心议题**：涉及的主要话题和背景
2. **关键信息**：重要决策、数字、人名、日期
3. **待办事项**：需要跟进或行动的事项（如有）
4. **结论**：整体情况概述"""


class SummarizerAgent:
    def __init__(self):
        self._client = OpenAI(api_key=cfg.DEEPSEEK_API_KEY, base_url=cfg.DEEPSEEK_BASE_URL)

    def run(self, request: AgentRequest, memory=None) -> AgentResponse:
        results = hybrid_search(request.query, top_k=cfg.TOP_K * 2)
        reranked = rerank(request.query, results, top_n=cfg.TOP_K)

        if not reranked:
            return AgentResponse(answer="未找到相关邮件，无法生成摘要。", sources=[])

        context_parts = []
        for i, r in enumerate(reranked):
            m = r.metadata
            context_parts.append(
                f"邮件{i + 1}（发件人: {m.get('sender','?')}，"
                f"日期: {m.get('date','?')}，主题: {m.get('subject','?')}）：\n{r.content}"
            )
        context = "\n\n".join(context_parts)

        resp = self._client.chat.completions.create(
            model=cfg.DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {
                    "role": "user",
                    "content": f"请综合摘要以下邮件，聚焦于：{request.query}\n\n{context}",
                },
            ],
            temperature=0.3,
            max_tokens=1500,
        )
        return AgentResponse(answer=resp.choices[0].message.content, sources=reranked)
