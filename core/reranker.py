from __future__ import annotations

import json
import logging
from typing import List

from openai import OpenAI

import config.settings as cfg
from models.schemas import SearchResult

logger = logging.getLogger(__name__)

_client = None
_cross_encoder = None

# Degradation level tracking (module-level, reset on process start).
_consecutive_failures = 0
_FAILURE_THRESHOLD = 3  # disable reranker after this many consecutive failures


def reset_circuit_breaker() -> None:
    """Reset failure counter between RAGAS ablation versions."""
    global _consecutive_failures
    _consecutive_failures = 0


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=cfg.DEEPSEEK_API_KEY, base_url=cfg.DEEPSEEK_BASE_URL)
    return _client


def _get_cross_encoder():
    """Lazy-load the local cross-encoder so normal imports stay lightweight."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        kwargs = {"device": cfg.CROSS_ENCODER_DEVICE}
        max_length = getattr(cfg, "CROSS_ENCODER_MAX_LENGTH", None)
        if max_length:
            kwargs["max_length"] = max_length
        try:
            _cross_encoder = CrossEncoder(cfg.CROSS_ENCODER_MODEL, **kwargs)
        except TypeError:
            # Older sentence-transformers releases do not accept max_length in
            # the constructor; the model still truncates internally.
            kwargs.pop("max_length", None)
            _cross_encoder = CrossEncoder(cfg.CROSS_ENCODER_MODEL, **kwargs)
    return _cross_encoder


def _truncate_for_rerank(text: str, limit: int = None) -> str:
    limit = int(limit if limit is not None else getattr(cfg, "RERANK_INPUT_CHAR_LIMIT", 1200))
    if limit <= 0:
        return text
    return text[:limit]


def _to_float_scores(raw_scores) -> List[float]:
    scores = raw_scores.tolist() if hasattr(raw_scores, "tolist") else raw_scores
    return [float(score) for score in scores]


def _copy_with_score(result: SearchResult, score: float) -> SearchResult:
    return SearchResult(
        chunk_id=result.chunk_id,
        email_id=result.email_id,
        content=result.content,
        score=float(score),
        metadata=result.metadata,
    )


def _extract_json_obj(text: str) -> str:
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1].lstrip("json").strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        return text[start:end]
    return text


_RERANK_PROMPT = """You are a text relevance scorer.
Given a user query and candidate email chunks, score each chunk's relevance
to the query with an integer from 0 to 10.

User query:
{query}

Candidate chunks ({n}):
{docs}

Return strict JSON only:
{{"scores": [score1, score2, ...]}}"""


def _rerank_with_cross_encoder(query: str, results: List[SearchResult], top_n: int) -> List[SearchResult]:
    model = _get_cross_encoder()
    pairs = [(query, _truncate_for_rerank(r.content)) for r in results]
    scores = _to_float_scores(model.predict(pairs))
    if len(scores) != len(results):
        raise ValueError(f"Score count mismatch: {len(scores)} vs {len(results)}")
    scored = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
    return [_copy_with_score(r, s) for r, s in scored[:top_n]]


def _rerank_with_llm(query: str, results: List[SearchResult], top_n: int) -> List[SearchResult]:
    docs_text = "\n\n".join(
        f"[{i + 1}] {_truncate_for_rerank(r.content, 400)}" for i, r in enumerate(results)
    )
    prompt = _RERANK_PROMPT.format(query=query, n=len(results), docs=docs_text)

    resp = _get_client().chat.completions.create(
        model=cfg.DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=3000,
        timeout=cfg.LLM_TIMEOUT,
    )
    choice = resp.choices[0]
    raw = (choice.message.content or "").strip()
    if not raw:
        reasoning = getattr(choice.message, "reasoning_content", None) or ""
        if reasoning and "{" in reasoning and "}" in reasoning:
            raw = reasoning.strip()
        else:
            raise ValueError(f"Empty rerank response (finish_reason={choice.finish_reason!r})")

    data = json.loads(_extract_json_obj(raw))
    scores = data.get("scores", [])
    if len(scores) != len(results):
        raise ValueError(f"Score count mismatch: {len(scores)} vs {len(results)}")

    scored = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
    return [_copy_with_score(r, s) for r, s in scored[:top_n]]


def rerank(query: str, results: List[SearchResult], top_n: int = None) -> List[SearchResult]:
    top_n = top_n or cfg.RERANK_TOP_N
    if not results:
        return []
    if not cfg.ENABLE_RERANKER:
        return results[:top_n]
    if len(results) <= 1:
        return results[:top_n]

    global _consecutive_failures
    if _consecutive_failures >= _FAILURE_THRESHOLD:
        logger.warning("Reranker circuit breaker open, skipping rerank")
        return results[:top_n]

    try:
        backend = getattr(cfg, "RERANKER_BACKEND", "cross_encoder").lower()
        if backend == "cross_encoder":
            reranked = _rerank_with_cross_encoder(query, results, top_n)
        elif backend == "llm":
            reranked = _rerank_with_llm(query, results, top_n)
        else:
            logger.warning("Unknown reranker backend %r, falling back to LLM rerank", backend)
            reranked = _rerank_with_llm(query, results, top_n)
        _consecutive_failures = 0
        return reranked
    except Exception as exc:
        _consecutive_failures += 1
        logger.warning(
            "Reranker failed (%s), consecutive=%s, returning original order",
            exc,
            _consecutive_failures,
        )
        return results[:top_n]
