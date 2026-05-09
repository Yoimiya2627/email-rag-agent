"""Tests for core/retriever.py — tokenizer, BM25 cache, hybrid search branches."""
from typing import List

import pytest

import config.settings as cfg
import core.retriever as retriever_mod
from core.retriever import (
    _tokenize,
    bm25_search,
    hybrid_search,
    invalidate_bm25_cache,
    vector_search,
)
from models.schemas import SearchResult


@pytest.fixture(autouse=True)
def _wipe_bm25_cache():
    invalidate_bm25_cache()
    yield
    invalidate_bm25_cache()


def _stub_chunks(n: int = 3) -> List[dict]:
    return [
        {
            "chunk_id": f"c{i}",
            "content": f"alpha beta gamma {i}",
            "metadata": {"email_id": f"e{i}", "subject": f"s{i}"},
        }
        for i in range(n)
    ]


def test_tokenize_handles_chinese_english_and_lowercases():
    assert _tokenize("OKR 会议 v2") == ["okr", "会", "议", "v2"]


def test_hybrid_search_skips_bm25_when_disabled(monkeypatch, make_search_result):
    monkeypatch.setattr(cfg, "ENABLE_BM25", False)
    vec = [make_search_result("A", score=3.0), make_search_result("B", score=2.0)]
    monkeypatch.setattr(retriever_mod, "search_similar", lambda q, top_k: [
        {"chunk_id": r.chunk_id, "email_id": r.email_id, "content": r.content,
         "score": r.score, "metadata": r.metadata} for r in vec
    ])

    bm25_calls = {"count": 0}
    monkeypatch.setattr(retriever_mod, "get_collection_count", lambda: bm25_calls.__setitem__("count", bm25_calls["count"] + 1) or 999)

    out = hybrid_search("anything", top_k=5)
    assert [r.chunk_id for r in out] == ["A", "B"]
    assert bm25_calls["count"] == 0, "BM25 path must not run when ENABLE_BM25=False"


def _patch_searches(monkeypatch, vec_results, bm25_results):
    """Stub vector_search and bm25_search inside retriever module."""
    monkeypatch.setattr(retriever_mod, "vector_search", lambda q, top_k=None: vec_results)
    monkeypatch.setattr(retriever_mod, "bm25_search", lambda q, top_k=None: bm25_results)


def test_hybrid_search_rrf_math_and_ordering(monkeypatch, make_search_result):
    monkeypatch.setattr(cfg, "ENABLE_BM25", True)
    monkeypatch.setattr(cfg, "ENABLE_RRF", True)
    monkeypatch.setattr(cfg, "VECTOR_WEIGHT", 0.7)
    monkeypatch.setattr(cfg, "BM25_WEIGHT", 0.3)

    vec = [make_search_result("A", score=10), make_search_result("B", score=9), make_search_result("C", score=8)]
    bm25 = [make_search_result("B", score=5), make_search_result("D", score=4), make_search_result("E", score=3)]
    _patch_searches(monkeypatch, vec, bm25)

    out = hybrid_search("q", top_k=5)
    ids = [r.chunk_id for r in out]
    # Expected fused scores (RRF_K=60, weights 0.7/0.3):
    #   B = 0.7/(1+60) + 0.3/(0+60) = 0.01148 + 0.005   = 0.01648
    #   A = 0.7/(0+60)                                  = 0.01167
    #   C = 0.7/(2+60)                                  = 0.01129
    #   D = 0.3/(1+60)                                  = 0.00492
    #   E = 0.3/(2+60)                                  = 0.00484
    assert ids == ["B", "A", "C", "D", "E"]
    assert out[0].score == pytest.approx(0.7 / 61 + 0.3 / 60, rel=1e-6)
    assert out[1].score == pytest.approx(0.7 / 60, rel=1e-6)


def test_hybrid_search_simple_merge_dedups_and_sorts_by_score(monkeypatch, make_search_result):
    monkeypatch.setattr(cfg, "ENABLE_BM25", True)
    monkeypatch.setattr(cfg, "ENABLE_RRF", False)

    vec = [make_search_result("A", score=10), make_search_result("B", score=7), make_search_result("C", score=5)]
    bm25 = [make_search_result("B", score=8), make_search_result("D", score=4), make_search_result("E", score=2)]
    _patch_searches(monkeypatch, vec, bm25)

    out = hybrid_search("q", top_k=10)
    ids = [r.chunk_id for r in out]
    # Each chunk_id must appear once; vec wins for B because dict.update overrides bm25.
    assert sorted(ids) == ["A", "B", "C", "D", "E"]
    # Sorted by raw score desc — vec's B (7) replaces bm25's B (8), so order is A(10),B(7),C(5),D(4),E(2).
    assert ids == ["A", "B", "C", "D", "E"]
    by_id = {r.chunk_id: r for r in out}
    assert by_id["B"].score == 7  # vec value, not 8 from bm25


def test_bm25_cache_reused_when_count_unchanged(monkeypatch):
    chunks = _stub_chunks(3)
    calls = {"count": 0, "fetch": 0}

    def fake_count():
        calls["count"] += 1
        return len(chunks)

    def fake_fetch():
        calls["fetch"] += 1
        return list(chunks)

    monkeypatch.setattr(retriever_mod, "get_collection_count", fake_count)
    monkeypatch.setattr(retriever_mod, "get_all_chunks", fake_fetch)

    bm25_search("alpha", top_k=3)
    bm25_search("beta", top_k=3)
    assert calls["fetch"] == 1, "BM25 should be cached on second call when count is unchanged"
    assert calls["count"] >= 2


def test_bm25_cache_rebuilds_on_count_drift(monkeypatch):
    chunks = _stub_chunks(3)
    counts = iter([3, 10])  # second call sees a larger collection
    fetches = {"n": 0}

    monkeypatch.setattr(retriever_mod, "get_collection_count", lambda: next(counts))

    def fake_fetch():
        fetches["n"] += 1
        return list(chunks)

    monkeypatch.setattr(retriever_mod, "get_all_chunks", fake_fetch)

    bm25_search("alpha", top_k=3)
    bm25_search("alpha", top_k=3)
    assert fetches["n"] == 2, "BM25 should rebuild when collection count drifts"


def test_invalidate_bm25_cache_forces_rebuild(monkeypatch):
    chunks = _stub_chunks(3)
    fetches = {"n": 0}
    monkeypatch.setattr(retriever_mod, "get_collection_count", lambda: 3)

    def fake_fetch():
        fetches["n"] += 1
        return list(chunks)

    monkeypatch.setattr(retriever_mod, "get_all_chunks", fake_fetch)

    bm25_search("alpha", top_k=3)
    invalidate_bm25_cache()
    bm25_search("alpha", top_k=3)
    assert fetches["n"] == 2


def test_bm25_search_empty_collection_returns_empty_list(monkeypatch):
    monkeypatch.setattr(retriever_mod, "get_collection_count", lambda: 0)
    # get_all_chunks should never be called when count is 0.
    monkeypatch.setattr(retriever_mod, "get_all_chunks", lambda: pytest.fail("unexpected fetch"))
    assert bm25_search("anything", top_k=5) == []


def test_bm25_search_filters_zero_scores(monkeypatch):
    chunks = _stub_chunks(3)
    monkeypatch.setattr(retriever_mod, "get_collection_count", lambda: len(chunks))
    monkeypatch.setattr(retriever_mod, "get_all_chunks", lambda: list(chunks))

    # Query with no overlap with the corpus → all BM25 scores are 0 → filtered out.
    out = bm25_search("zzz_no_match_token_xyz", top_k=5)
    assert out == []
