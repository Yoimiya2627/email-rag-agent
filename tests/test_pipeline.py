"""Tests for core/pipeline.py — post-filters and the unified retrieve() orchestration."""
from datetime import datetime, timedelta
import pytest

import core.pipeline as pipeline_mod
from core.pipeline import apply_post_filters, retrieve


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


def test_apply_post_filters_sender_returns_empty_when_all_removed(make_search_result):
    """An explicit hard filter must never fall back to unrelated candidates."""
    results = [
        make_search_result("a", metadata={"sender": "alice@corp.com"}),
        make_search_result("b", metadata={"sender": "bob@corp.com"}),
    ]
    out = apply_post_filters(results, {"sender": "nobody"})
    assert out == []


def test_apply_post_filters_labels_keeps_only_matching(make_search_result):
    results = [
        make_search_result("a", metadata={"labels": ["finance", "urgent"]}),
        make_search_result("b", metadata={"labels": ["social"]}),
    ]
    out = apply_post_filters(results, {"labels": ["urgent"]})
    assert [r.chunk_id for r in out] == ["a"]


def test_apply_post_filters_date_window(make_search_result):
    recent = pipeline_mod._now().strftime("%Y-%m-%d")
    old = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
    results = [
        make_search_result("recent", metadata={"date": recent}),
        make_search_result("old", metadata={"date": old}),
    ]
    out = apply_post_filters(results, {"date_hint": "本周"})  # calendar week
    assert [r.chunk_id for r in out] == ["recent"]


def test_retrieve_runs_full_pipeline_in_order(monkeypatch, make_search_result):
    """retrieve() must: search with filters['query'], rerank against the
    ORIGINAL query, and apply the post-filters."""
    calls = {}
    monkeypatch.setattr(pipeline_mod, "rewrite_query", lambda q: "REWRITTEN")
    monkeypatch.setattr(
        pipeline_mod, "extract_filters",
        lambda q: {"query": "SEARCH_Q", "sender": "alice", "labels": [], "date_hint": ""},
    )
    raw = [
        make_search_result("a", metadata={"sender": "alice@corp.com"}),
        make_search_result("b", metadata={"sender": "bob@corp.com"}),
    ]

    def fake_hybrid(q, top_k=None, filters=None):
        calls["hybrid_q"] = q
        calls["hybrid_k"] = top_k
        calls["scope"] = filters
        return raw

    def fake_rerank(q, results, top_n=None):
        calls["rerank_q"] = q
        return results[:top_n]

    monkeypatch.setattr(pipeline_mod, "hybrid_search", fake_hybrid)
    monkeypatch.setattr(pipeline_mod, "rerank", fake_rerank)

    out = retrieve("ORIGINAL", top_n=5, fetch_k=20)

    assert calls["hybrid_q"] == "SEARCH_Q"    # hybrid search uses filters['query']
    assert calls["hybrid_k"] == 20            # fetch_k passed through
    assert calls["scope"].sender == "alice"
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
    monkeypatch.setattr(
        pipeline_mod, "hybrid_search",
        lambda q, top_k=None: calls.__setitem__("hybrid_q", q) or [],
    )
    monkeypatch.setattr(pipeline_mod, "rerank", lambda q, results, top_n=None: results)

    retrieve("ORIGINAL")
    assert calls["hybrid_q"] == "REWRITTEN_FORM"
@pytest.mark.parametrize('query,invented_label', [
    ('请检索导入邮件，星舟项目批准的预算是多少元？', '导入邮件'),
    ('Search imported emails for the approved project budget.', 'imported'),
])
def test_source_description_does_not_become_a_hard_label_filter(monkeypatch,make_search_result,query,invented_label):
    import json
    from types import SimpleNamespace
    import core.pipeline as pipeline
    raw=json.dumps({'query':'project budget','sender':'','date_hint':'','labels':[invented_label]})
    monkeypatch.setattr(pipeline,'_get_client',lambda:object())
    monkeypatch.setattr(pipeline,'create_completion',lambda *a,**k:SimpleNamespace(choices=[
        SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content=raw))]))
    monkeypatch.setattr(pipeline,'rewrite_query',lambda text:text)
    hit=make_search_result('budget_0',metadata={'labels':['synthetic']})
    seen=[]
    def search(text,**kwargs):
        seen.append((text,kwargs))
        return [hit]
    monkeypatch.setattr(pipeline,'hybrid_search',search)
    monkeypatch.setattr(pipeline,'rerank',lambda query,results,**kwargs:results)
    assert pipeline.retrieve(query)==[hit]
    assert seen[0][0]==query
    assert 'filters' not in seen[0][1]


@pytest.mark.parametrize('query', ['查找标签为“导入邮件”的预算邮件', 'Find emails labeled imported'])
def test_explicit_label_request_still_keeps_the_hard_filter(monkeypatch,query):
    import json
    from types import SimpleNamespace
    import core.pipeline as pipeline
    raw={'query':'budget','sender':'','date_hint':'','labels':['imported']}
    monkeypatch.setattr(pipeline,'_get_client',lambda:object())
    monkeypatch.setattr(pipeline,'create_completion',lambda *a,**k:SimpleNamespace(choices=[
        SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content=json.dumps(raw)))]))
    assert pipeline.extract_filters(query)==raw
