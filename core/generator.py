import logging
from typing import List, Optional

from openai import OpenAI

from models.schemas import SearchResult
import config.settings as cfg

logger = logging.getLogger(__name__)

_client = None

_DEFAULT_SYSTEM = (
    "你是一个专业的邮件智能助手。请根据提供的邮件内容回答用户的问题。"
    "回答要准确、简洁。如果检索内容不足以回答问题，请明确说明。"
    "引用具体邮件时请注明发件人和日期。"
)


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=cfg.DEEPSEEK_API_KEY, base_url=cfg.DEEPSEEK_BASE_URL)
    return _client


def build_context(results: List[SearchResult]) -> str:
    parts = []
    for i, r in enumerate(results):
        m = r.metadata
        header = (
            f"【邮件{i + 1}】"
            f"发件人: {m.get('sender', '?')} | "
            f"日期: {m.get('date', '?')} | "
            f"主题: {m.get('subject', '?')}"
        )
        parts.append(f"{header}\n{r.content}")
    separator = f"\n\n{'—' * 40}\n\n"
    return separator.join(parts)


def generate_answer(
    query: str,
    results: List[SearchResult],
    system_prompt: Optional[str] = None,
) -> str:
    if not results:
        return "未找到相关邮件内容，无法回答该问题。"

    context = build_context(results)
    user_msg = f"参考邮件内容：\n{context}\n\n用户问题：{query}"

    client = _get_client()
    resp = client.chat.completions.create(
        model=cfg.DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system_prompt or _DEFAULT_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
        max_tokens=1500,
    )
    return resp.choices[0].message.content
