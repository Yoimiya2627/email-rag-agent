"""Tests for core/pipeline.py — post-filters and the unified retrieve() orchestration."""
from datetime import datetime, timedelta

import core.pipeline as pipeline_mod
from core.pipeline import (
    apply_metadata_boost,
    apply_post_filters,
    augment_filters_from_query,
    retrieve,
)


def test_apply_post_filters_passthrough_when_filters_empty(make_search_result):
    results = [make_search_result("a"), make_search_result("b")]
    out = apply_post_filters(results, {"sender": "", "labels": [], "date_hint": ""})
    assert out == results


def test_apply_post_filters_sender_keeps_only_matching(make_search_result):
    results = [
        make_search_result("a", metadata={"sender": "alice@corp.com"}),
        make_search_result("b", metadata={"sender": "bob@corp.com"}),
    ]
    out = apply_post_filters(results, {"sender": "alice"})
    assert [r.chunk_id for r in out] == ["a"]


def test_apply_post_filters_sender_falls_back_when_all_removed(make_search_result):
    """A filter that would remove every candidate must fall back to the full
    list — better loosely-relevant results than nothing."""
    results = [
        make_search_result("a", metadata={"sender": "alice@corp.com"}),
        make_search_result("b", metadata={"sender": "bob@corp.com"}),
    ]
    out = apply_post_filters(results, {"sender": "nobody"})
    assert out == results


def test_apply_post_filters_labels_keeps_only_matching(make_search_result):
    results = [
        make_search_result("a", metadata={"labels": ["finance", "urgent"]}),
        make_search_result("b", metadata={"labels": ["social"]}),
    ]
    out = apply_post_filters(results, {"labels": ["urgent"]})
    assert [r.chunk_id for r in out] == ["a"]


def test_apply_post_filters_date_window(make_search_result):
    recent = datetime.now().strftime("%Y-%m-%d")
    old = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
    results = [
        make_search_result("recent", metadata={"date": recent}),
        make_search_result("old", metadata={"date": old}),
    ]
    out = apply_post_filters(results, {"date_hint": "本周"})  # 7-day window
    assert [r.chunk_id for r in out] == ["recent"]


def test_apply_metadata_boost_prefers_exact_subject_and_sender(make_search_result):
    generic = make_search_result(
        "generic",
        score=1.0,
        metadata={"subject": "Weekly AI digest", "sender": "news@example.com"},
    )
    target = make_search_result(
        "target",
        score=0.95,
        metadata={
            "subject": "Gemini API billing alert",
            "sender": "alerts@example.com",
        },
    )

    out = apply_metadata_boost(
        [generic, target],
        'What did alerts@example.com say about "Gemini API billing alert"?',
        {"sender": "alerts@example.com", "labels": [], "date_hint": ""},
    )

    assert [r.chunk_id for r in out] == ["target", "generic"]
    assert out[0].score > target.score


def test_augment_filters_from_query_extracts_sender_and_subject():
    filters = {"query": "96 Let's get yo", "sender": "", "labels": [], "date_hint": ""}

    out = augment_filters_from_query(
        'What does the email from login@example.com about "Secure link to log in" say?',
        filters,
    )

    assert out["sender"] == "login@example.com"
    assert out["query"].startswith("Secure link to log in")
    assert "96 Let's get yo" in out["query"]


def test_retrieve_runs_full_pipeline_in_order(monkeypatch, make_search_result):
    """retrieve() must: search with filters['query'], rerank against the
    ORIGINAL query, and apply the post-filters."""
    calls = {}
    monkeypatch.setattr(pipeline_mod, "rewrite_query", lambda q: "REWRITTEN")
    monkeypatch.setattr(
        pipeline_mod, "extract_filters",
        lambda q: {"query": "SEARCH_Q", "sender": "alice", "labels": [], "date_hint": ""},
    )
    monkeypatch.setattr(pipeline_mod, "augment_filters_from_query", lambda q, f: f)
    raw = [
        make_search_result("a", metadata={"sender": "alice@corp.com"}),
        make_search_result("b", metadata={"sender": "bob@corp.com"}),
    ]

    def fake_hybrid(q, top_k=None):
        calls["hybrid_q"] = q
        calls["hybrid_k"] = top_k
        return raw

    def fake_rerank(q, results, top_n=None):
        calls["rerank_q"] = q
        return results[:top_n]

    monkeypatch.setattr(pipeline_mod, "hybrid_search", fake_hybrid)
    monkeypatch.setattr(pipeline_mod, "apply_metadata_boost", lambda rs, q, f: rs)
    monkeypatch.setattr(pipeline_mod, "rerank", fake_rerank)

    out = retrieve("ORIGINAL", top_n=5, fetch_k=20)

    assert calls["hybrid_q"] == "SEARCH_Q"    # hybrid search uses filters['query']
    assert calls["hybrid_k"] == 20            # fetch_k passed through
    assert calls["rerank_q"] == "ORIGINAL"    # rerank scores against the ORIGINAL query
    assert [r.chunk_id for r in out] == ["a"]  # sender post-filter applied


def test_retrieve_search_query_falls_back_to_rewritten(monkeypatch, make_search_result):
    """When extract_filters returns no 'query', hybrid search uses the rewritten query."""
    calls = {}
    monkeypatch.setattr(pipeline_mod, "rewrite_query", lambda q: "REWRITTEN_FORM")
    monkeypatch.setattr(
        pipeline_mod, "extract_filters",
        lambda q: {"sender": "", "labels": [], "date_hint": ""},  # no 'query' key
    )
    monkeypatch.setattr(pipeline_mod, "augment_filters_from_query", lambda q, f: f)
    monkeypatch.setattr(
        pipeline_mod, "hybrid_search",
        lambda q, top_k=None: calls.__setitem__("hybrid_q", q) or [],
    )
    monkeypatch.setattr(pipeline_mod, "rerank", lambda q, results, top_n=None: results)

    retrieve("ORIGINAL")
    assert calls["hybrid_q"] == "REWRITTEN_FORM"
