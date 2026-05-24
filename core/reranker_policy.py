from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RerankerServingDecision:
    goal: str
    version: str
    enable_reranker: bool
    backend: str
    reason: str
    caveat: str

    def as_flags(self) -> dict:
        flags = {
            "ENABLE_BM25": True,
            "ENABLE_RRF": True,
            "ENABLE_RERANKER": self.enable_reranker,
            "ENABLE_QUERY_REWRITE": False,
        }
        if self.backend != "none":
            flags["RERANKER_BACKEND"] = self.backend
        return flags


def choose_reranker_policy(
    goal: str = "conversation",
    latency_budget_ms: Optional[int] = None,
    allow_llm_reranker: bool = False,
) -> RerankerServingDecision:
    """Choose a serving version for the current reranker trade-off.

    The policy encodes the current evaluation facts:
    - V2 is the low-latency default because reranker is disabled.
    - V7 is the deterministic quality mode using the cross-encoder backend.
    - V3 is only a high-precision reference when an extra LLM scorer is allowed.
    """
    normalized = (goal or "conversation").strip().lower()
    low_latency_budget = latency_budget_ms is not None and latency_budget_ms <= 10_000

    if normalized in {"conversation", "default", "latency", "chat"} or low_latency_budget:
        return RerankerServingDecision(
            goal=normalized,
            version="V2",
            enable_reranker=False,
            backend="none",
            reason="Use V2 for low latency conversation traffic: no reranker, no rewrite, shortest path.",
            caveat="Use V7 only when answer quality is worth the extra local rerank cost.",
        )

    if normalized in {"precision", "top_k_precision", "audit", "review"}:
        if allow_llm_reranker:
            return RerankerServingDecision(
                goal=normalized,
                version="V3",
                enable_reranker=True,
                backend="llm",
                reason="Use V3 only as a high context_precision reference when an extra LLM scorer is allowed.",
                caveat="V3 has the best current context_precision but adds LLM latency and scoring variance.",
            )
        return RerankerServingDecision(
            goal=normalized,
            version="V7",
            enable_reranker=True,
            backend="cross_encoder",
            reason="Use deterministic Cross-Encoder reranking for precision-oriented serving without an LLM scorer.",
            caveat="V7 precision improves over V2 but did not beat V3 in the 30-question run.",
        )

    if normalized in {"quality", "relevancy", "faithfulness", "high_quality"}:
        return RerankerServingDecision(
            goal=normalized,
            version="V7",
            enable_reranker=True,
            backend="cross_encoder",
            reason="Use V7 when answer_relevancy and faithfulness matter more than the default low latency path.",
            caveat="Formal end-to-end latency is still required before making V7 the global default.",
        )

    return RerankerServingDecision(
        goal=normalized,
        version="V2",
        enable_reranker=False,
        backend="none",
        reason="Unknown goal falls back to the low latency V2 default.",
        caveat="Pass goal='quality' or goal='precision' to opt into reranking explicitly.",
    )
