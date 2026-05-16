"""
WriterAgent: finds the most relevant original email and drafts a reply.

`draft_reply_for_email()` is module-level so the agent tool layer can draft a
reply for a *specific, already-known* email — multi-step tasks pass an
email_id from search results straight into the draft step.
"""
import logging

from openai import OpenAI

from models.schemas import AgentRequest, AgentResponse
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


_SYSTEM = """你是一位专业的邮件撰写助手。根据原始邮件和用户要求，生成得体、专业的邮件回复草稿。

格式要求：
- 开头使用适当称呼（如"您好，[姓名]："）
- 正文清晰表达回复要点，段落分明
- 结尾使用礼貌用语（如"此致，[发件人]"）
- 默认使用中文，若原邮件为英文则用英文回复"""


def draft_reply_for_email(email: dict, instruction: str = "") -> str:
    """Draft a reply for an explicit original email.

    `email` keys used: sender / date / subject / body (or content).
    """
    body = email.get("body") or email.get("content", "")
    context = (
        f"原始邮件\n"
        f"发件人: {email.get('sender', '?')}\n"
        f"日期: {email.get('date', '?')}\n"
        f"主题: {email.get('subject', '?')}\n\n"
        f"{body}"
    )
    ask = (instruction or "").strip() or "请根据原始邮件内容起草一封得体的回复。"
    resp = _get_client().chat.completions.create(
        model=cfg.DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"{context}\n\n用户要求：{ask}"},
        ],
        temperature=0.5,
        max_tokens=1000,
    )
    return resp.choices[0].message.content


class WriterAgent:
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
        email = {
            "sender": m.get("sender", "?"),
            "date": m.get("date", "?"),
            "subject": m.get("subject", "?"),
            "content": original.content,
        }
        answer = draft_reply_for_email(email, request.query)
        return AgentResponse(answer=answer, sources=reranked[:1])
