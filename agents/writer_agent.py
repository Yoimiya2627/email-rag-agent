"""
WriterAgent: finds the most relevant original email and drafts a reply
according to the user's intent.
"""
import logging

from openai import OpenAI

from models.schemas import AgentRequest, AgentResponse
from core.retriever import hybrid_search
from core.reranker import rerank
import config.settings as cfg

logger = logging.getLogger(__name__)

_SYSTEM = """你是一位专业的邮件撰写助手。根据原始邮件和用户要求，生成得体、专业的邮件回复草稿。

格式要求：
- 开头使用适当称呼（如"您好，[姓名]："）
- 正文清晰表达回复要点，段落分明
- 结尾使用礼貌用语（如"此致，[发件人]"）
- 默认使用中文，若原邮件为英文则用英文回复"""


class WriterAgent:
    def __init__(self):
        self._client = OpenAI(api_key=cfg.DEEPSEEK_API_KEY, base_url=cfg.DEEPSEEK_BASE_URL)

    def run(self, request: AgentRequest, memory=None) -> AgentResponse:
        results = hybrid_search(request.query, top_k=cfg.TOP_K)
        reranked = rerank(request.query, results, top_n=3)

        if not reranked:
            return AgentResponse(
                answer="未找到原始邮件，无法生成回复草稿。请先索引邮件数据。",
                sources=[],
            )

        original = reranked[0]
        m = original.metadata
        context = (
            f"原始邮件\n"
            f"发件人: {m.get('sender', '?')}\n"
            f"日期: {m.get('date', '?')}\n"
            f"主题: {m.get('subject', '?')}\n\n"
            f"{original.content}"
        )

        resp = self._client.chat.completions.create(
            model=cfg.DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": f"{context}\n\n用户要求：{request.query}"},
            ],
            temperature=0.5,
            max_tokens=1000,
        )
        return AgentResponse(
            answer=resp.choices[0].message.content,
            sources=reranked[:1],
        )
