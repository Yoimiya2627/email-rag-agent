"""Tests for the optional cross-encoder reranker backend."""
from __future__ import annotations

import pytest

import config.settings as cfg
import core.reranker as reranker_mod
from models.schemas import SearchResult


def _result(chunk_id: str, content: str, score: float = 0.0) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        email_id=f"email-{chunk_id}",
        content=content,
        score=score,
        metadata={"subject": f"subject-{chunk_id}"},
    )


class FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = scores
        self.pairs = None

    def predict(self, pairs):
        self.pairs = list(pairs)
        return self.scores


@pytest.fixture(autouse=True)
def _reset_reranker(monkeypatch):
    monkeypatch.setattr(cfg, "ENABLE_RERANKER", True)
    monkeypatch.setattr(cfg, "RERANKER_BACKEND", "cross_encoder", raising=False)
    monkeypatch.setattr(cfg, "RERANK_INPUT_CHAR_LIMIT", 1200, raising=False)
    reranker_mod.reset_circuit_breaker()
    monkeypatch.setattr(reranker_mod, "_cross_encoder", None, raising=False)
    yield
    reranker_mod.reset_circuit_breaker()


def test_cross_encoder_reranker_orders_by_predicted_score(monkeypatch):
    model = FakeCrossEncoder([0.1, 0.9, 0.4])
    monkeypatch.setattr(reranker_mod, "_get_cross_encoder", lambda: model)

    results = [
        _result("a", "报销流程说明"),
        _result("b", "预算审批邮件"),
        _result("c", "会议纪要"),
    ]

    out = reranker_mod.rerank("谁提到了预算审批？", results, top_n=2)

    assert [r.chunk_id for r in out] == ["b", "c"]
    assert [r.score for r in out] == [0.9, 0.4]
    assert model.pairs == [
        ("谁提到了预算审批？", "报销流程说明"),
        ("谁提到了预算审批？", "预算审批邮件"),
        ("谁提到了预算审批？", "会议纪要"),
    ]


def test_cross_encoder_reranker_falls_back_to_original_order_on_score_mismatch(monkeypatch):
    monkeypatch.setattr(reranker_mod, "_get_cross_encoder", lambda: FakeCrossEncoder([0.8]))
    results = [_result("a", "A"), _result("b", "B"), _result("c", "C")]

    out = reranker_mod.rerank("query", results, top_n=2)

    assert [r.chunk_id for r in out] == ["a", "b"]
    assert [r.score for r in out] == [0.0, 0.0]


def test_cross_encoder_truncates_model_input_but_preserves_result_content(monkeypatch):
    model = FakeCrossEncoder([1.0, 0.2])
    monkeypatch.setattr(reranker_mod, "_get_cross_encoder", lambda: model)
    monkeypatch.setattr(cfg, "RERANK_INPUT_CHAR_LIMIT", 10, raising=False)
    long_content = "x" * 50
    results = [_result("long", long_content), _result("short", "ok")]

    out = reranker_mod.rerank("query", results, top_n=1)

    assert model.pairs[0] == ("query", "x" * 10)
    assert out[0].chunk_id == "long"
    assert out[0].content == long_content
