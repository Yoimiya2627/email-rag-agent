import logging
from typing import Generator, Iterator, List, Optional

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
    history: Optional[List[dict]] = None,
) -> str:
    if not results:
        return "未找到相关邮件内容，无法回答该问题。"

    context = build_context(results)
    user_msg = f"参考邮件内容：\n{context}\n\n用户问题：{query}"

    messages = [{"role": "system", "content": system_prompt or _DEFAULT_SYSTEM}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_msg})

    client = _get_client()
    try:
        resp = client.chat.completions.create(
            model=cfg.DEEPSEEK_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=1500,
            timeout=cfg.LLM_TIMEOUT,
        )
        return resp.choices[0].message.content
    except Exception as exc:
        # Degradation level 2: return a summary built directly from context (no LLM)
        logger.warning(f"LLM generate_answer failed ({exc}), falling back to context summary")
        if results:
            snippets = "\n".join(
                f"[{r.metadata.get('date','?')}] {r.metadata.get('subject','?')}: {r.content[:150]}"
                for r in results[:3]
            )
            return f"（LLM暂时不可用，以下是原始检索结果）\n\n{snippets}"
        return "服务暂时不可用，请稍后重试。"


def stream_generate(
    query: str,
    results: List[SearchResult],
    system_prompt: Optional[str] = None,
    history: Optional[List[dict]] = None,
) -> Iterator[str]:
    """Yields answer tokens one by one for SSE streaming."""
    if not results:
        yield "未找到相关邮件内容，无法回答该问题。"
        return

    context = build_context(results)
    user_msg = f"参考邮件内容：\n{context}\n\n用户问题：{query}"

    messages = [{"role": "system", "content": system_prompt or _DEFAULT_SYSTEM}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_msg})

    client = _get_client()
    stream = client.chat.completions.create(
        model=cfg.DEEPSEEK_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=1500,
        stream=True,
        timeout=cfg.LLM_TIMEOUT,
    )

    # 推理模型（deepseek-v4-flash）先输出 reasoning_content（思考过程）再输出
    # content（最终答案）。只 yield delta.content 的话，reasoning 期间前端会有一段
    # 空白、体验像非流式；两段都转发 + 用 markdown 标记区分，让用户看到模型在工作。
    sent_reasoning_marker = False
    sent_content_marker = False

    for chunk in stream:
        delta = chunk.choices[0].delta

        rc = getattr(delta, "reasoning_content", None)
        if rc:
            if not sent_reasoning_marker:
                yield "💭 **思考过程**\n\n"
                sent_reasoning_marker = True
            yield rc

        if delta.content:
            if not sent_content_marker:
                if sent_reasoning_marker:
                    yield "\n\n---\n\n📝 **答案**\n\n"
                sent_content_marker = True
            yield delta.content
