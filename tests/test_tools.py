"""Tests for agents/tools.py — tool implementations, dispatch, schema consistency."""
from types import SimpleNamespace

import agents.tools as tools_mod
from agents.tools import call_tool


def test_search_emails_returns_formatted_hits(monkeypatch, make_search_result):
    hits = [
        make_search_result(
            "e1_0", score=0.91,
            content="Q3 预算评审会议纪要……",
            metadata={"email_id": "e1", "subject": "Q3 预算", "sender": "alice@x.com", "date": "2026-05-01"},
        ),
    ]
    monkeypatch.setattr(tools_mod, "hybrid_search", lambda q, top_k=None: hits)
    monkeypatch.setattr(tools_mod, "apply_post_filters", lambda r, f: r)
    monkeypatch.setattr(tools_mod, "rerank", lambda q, r, top_n=None: r[:top_n])

    out = tools_mod.search_emails(query="预算", limit=3)
    assert len(out) == 1
    hit = out[0]
    assert hit["email_id"] == "e1"
    assert hit["subject"] == "Q3 预算"
    assert hit["sender"] == "alice@x.com"
    assert "snippet" in hit and hit["snippet"]
    assert isinstance(hit["score"], float)


def test_search_emails_passes_filters_through(monkeypatch, make_search_result):
    captured = {}
    monkeypatch.setattr(tools_mod, "hybrid_search", lambda q, top_k=None: [])
    monkeypatch.setattr(
        tools_mod, "apply_post_filters",
        lambda r, f: captured.update(f) or r,
    )
    monkeypatch.setattr(tools_mod, "rerank", lambda q, r, top_n=None: r)

    tools_mod.search_emails(query="q", sender="bob", date_hint="本周", labels=["urgent"])
    assert captured == {"sender": "bob", "date_hint": "本周", "labels": ["urgent"]}


def test_get_email_joins_chunks_in_index_order(monkeypatch):
    chunks = [
        {"chunk_id": "e1_1", "content": "second part", "metadata": {"email_id": "e1", "chunk_index": 1, "subject": "S"}},
        {"chunk_id": "e1_0", "content": "first part", "metadata": {"email_id": "e1", "chunk_index": 0, "subject": "S"}},
        {"chunk_id": "e2_0", "content": "other email", "metadata": {"email_id": "e2", "chunk_index": 0}},
    ]
    monkeypatch.setattr(tools_mod, "get_all_chunks", lambda: chunks)
    out = tools_mod.get_email("e1")
    assert out["body"] == "first part\nsecond part"
    assert out["subject"] == "S"
    assert out["email_id"] == "e1"


def test_get_email_returns_error_when_not_found(monkeypatch):
    monkeypatch.setattr(tools_mod, "get_all_chunks", lambda: [])
    out = tools_mod.get_email("missing")
    assert "error" in out


def test_summarize_emails_delegates_to_summarizer_agent(monkeypatch):
    class FakeAgent:
        def run(self, request, memory=None):
            return SimpleNamespace(answer="STRUCTURED SUMMARY")

    monkeypatch.setattr(tools_mod, "SummarizerAgent", lambda: FakeAgent())
    assert tools_mod.summarize_emails(query="本周进展") == "STRUCTURED SUMMARY"


def test_draft_reply_delegates_to_writer_agent(monkeypatch):
    class FakeAgent:
        def run(self, request, memory=None):
            return SimpleNamespace(answer="Dear Bob, ...")

    monkeypatch.setattr(tools_mod, "WriterAgent", lambda: FakeAgent())
    assert tools_mod.draft_reply(query="回复 Bob") == "Dear Bob, ..."


def test_draft_reply_with_email_id_targets_specific_email(monkeypatch):
    """With email_id the tool drafts against that exact email (multi-step path)."""
    monkeypatch.setattr(
        tools_mod, "get_email",
        lambda eid: {"email_id": eid, "subject": "询价", "sender": "bob@x.com", "body": "请尽快报价"},
    )
    captured = {}
    monkeypatch.setattr(
        tools_mod, "draft_reply_for_email",
        lambda email, instruction: captured.update({"email": email, "instruction": instruction}) or "DRAFT",
    )
    out = tools_mod.draft_reply(instruction="礼貌报价", email_id="email_42")
    assert out == "DRAFT"
    assert captured["email"]["email_id"] == "email_42"
    assert captured["instruction"] == "礼貌报价"


def test_draft_reply_unknown_email_id_returns_error(monkeypatch):
    monkeypatch.setattr(tools_mod, "get_email", lambda eid: {"error": "not found"})
    out = tools_mod.draft_reply(instruction="x", email_id="missing")
    assert isinstance(out, dict) and "error" in out


def test_email_stats_delegates(monkeypatch):
    monkeypatch.setattr(tools_mod, "compute_email_stats", lambda: {"total_emails": 5000})
    assert tools_mod.email_stats() == {"total_emails": 5000}


def test_call_tool_invokes_named_tool_with_arguments(monkeypatch):
    recorded = {}
    monkeypatch.setitem(
        tools_mod.TOOL_DISPATCH, "search_emails",
        lambda **kw: recorded.update(kw) or ["hit"],
    )
    out = call_tool("search_emails", {"query": "预算", "sender": "alice"})
    assert out == ["hit"]
    assert recorded == {"query": "预算", "sender": "alice"}


def test_call_tool_unknown_tool_returns_error():
    out = call_tool("delete_everything", {})
    assert isinstance(out, dict) and "error" in out


def test_call_tool_handles_none_arguments(monkeypatch):
    monkeypatch.setitem(tools_mod.TOOL_DISPATCH, "email_stats", lambda: {"ok": True})
    assert call_tool("email_stats", None) == {"ok": True}


def test_call_tool_drops_hallucinated_kwargs(monkeypatch):
    """An argument the tool does not accept is dropped, not raised as TypeError."""
    monkeypatch.setitem(tools_mod.TOOL_DISPATCH, "search_emails", lambda query: {"q": query})
    out = call_tool("search_emails", {"query": "预算", "bogus_param": 123})
    assert out == {"q": "预算"}


def test_call_tool_missing_required_argument_returns_error(monkeypatch):
    monkeypatch.setitem(tools_mod.TOOL_DISPATCH, "get_email", lambda email_id: {"id": email_id})
    out = call_tool("get_email", {})
    assert isinstance(out, dict) and "error" in out
    assert "email_id" in out["error"]


def test_call_tool_captures_tool_exception(monkeypatch):
    """An exception inside a tool becomes an error result, not a crash."""
    def boom(**kw):
        raise RuntimeError("db down")

    monkeypatch.setitem(tools_mod.TOOL_DISPATCH, "email_stats", boom)
    out = call_tool("email_stats", {})
    assert isinstance(out, dict) and "error" in out
    assert "db down" in out["error"]


def test_tool_schemas_match_dispatch_table():
    schema_names = {s["function"]["name"] for s in tools_mod.TOOL_SCHEMAS}
    assert schema_names == set(tools_mod.TOOL_DISPATCH.keys())
    for s in tools_mod.TOOL_SCHEMAS:
        assert s["type"] == "function"
        fn = s["function"]
        assert fn["name"] and fn["description"]
        params = fn["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
