import json
import logging
from typing import List

from openai import OpenAI

from models.schemas import SearchResult
import config.settings as cfg

logger = logging.getLogger(__name__)

_client = None

# Degradation level tracking (module-level, reset on process start)
_consecutive_failures = 0
_FAILURE_THRESHOLD = 3  # disable reranker after this many consecutive failures


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=cfg.DEEPSEEK_API_KEY, base_url=cfg.DEEPSEEK_BASE_URL)
    return _client


_RERANK_PROMPT = """你是一个文本相关性评分器。给定用户问题和若干候选段落，请为每个段落与问题的相关程度打分（0-10的整数）。

用户问题：{query}

候选段落（共{n}个）：
{docs}

请严格按JSON格式返回，格式：{{"scores": [分数1, 分数2, ...]}}
只返回JSON，不要解释。"""


def rerank(query: str, results: List[SearchResult], top_n: int = None) -> List[SearchResult]:
    top_n = top_n or cfg.RERANK_TOP_N
    if not results:
        return []
    if not cfg.ENABLE_RERANKER:
        return results[:top_n]
    if len(results) <= 1:
        return results[:top_n]

    docs_text = "\n\n".join(
        f"[{i + 1}] {r.content[:400]}" for i, r in enumerate(results)
    )
    prompt = _RERANK_PROMPT.format(query=query, n=len(results), docs=docs_text)

    global _consecutive_failures
    # Circuit breaker: if too many consecutive failures, skip reranking
    if _consecutive_failures >= _FAILURE_THRESHOLD:
        logger.warning("Reranker circuit breaker open, skipping LLM rerank")
        return results[:top_n]

    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model=cfg.DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=256,
            timeout=cfg.LLM_TIMEOUT,
        )
        raw = resp.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()
        data = json.loads(raw)
        scores = data.get("scores", [])

        if len(scores) != len(results):
            raise ValueError(f"Score count mismatch: {len(scores)} vs {len(results)}")

        _consecutive_failures = 0  # reset on success
        scored = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
        return [
            SearchResult(
                chunk_id=r.chunk_id,
                email_id=r.email_id,
                content=r.content,
                score=float(s),
                metadata=r.metadata,
            )
            for r, s in scored[:top_n]
        ]
    except Exception as exc:
        _consecutive_failures += 1
        logger.warning(f"Reranker failed ({exc}), consecutive={_consecutive_failures}, returning original order")
        return results[:top_n]
