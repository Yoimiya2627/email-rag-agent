from __future__ import annotations

from core.reranker_policy import choose_reranker_policy


def test_conversation_policy_keeps_v2_for_low_latency_default():
    decision = choose_reranker_policy(goal="conversation", latency_budget_ms=9000)

    assert decision.version == "V2"
    assert decision.enable_reranker is False
    assert decision.backend == "none"
    assert "low latency" in decision.reason


def test_quality_policy_prefers_v7_cross_encoder_when_budget_allows():
    decision = choose_reranker_policy(goal="quality", latency_budget_ms=20000)

    assert decision.version == "V7"
    assert decision.enable_reranker is True
    assert decision.backend == "cross_encoder"
    assert "answer_relevancy" in decision.reason


def test_precision_policy_only_uses_llm_reranker_when_explicitly_allowed():
    deterministic = choose_reranker_policy(goal="precision", allow_llm_reranker=False)
    high_precision = choose_reranker_policy(goal="precision", allow_llm_reranker=True)

    assert deterministic.version == "V7"
    assert deterministic.backend == "cross_encoder"
    assert "deterministic" in deterministic.reason

    assert high_precision.version == "V3"
    assert high_precision.backend == "llm"
    assert "context_precision" in high_precision.reason
