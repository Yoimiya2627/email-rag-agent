"""Offline evidence/page/range regressions with synthetic indexed text only."""
import copy
import json
from datetime import datetime, timezone
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from agents.runtime import RunContext, bounded_json, normalize_tool_result, use_run_context
from core.evidence import evidence_reference, text_hash, with_visible_reference
from core.evidence_pages import (EvidenceCursorError, EvidenceVersionMismatch, SearchPages,
                                 compute_scoped_stats, read_email_page, read_thread_evidence,
                                 reread_evidence)
from core.evidence_claims import validate_claim_records


def email(text="0123456789", *, generation="g1", overlap=False):
    spans = [(0, len(text))] if not overlap else [(0, 7), (5, len(text))]
    chunks = [{"email_id": "e1", "chunk_id": f"c{index}", "content": text[start:end], "score": 0,
               "metadata": {"email_id": "e1", "index_generation": generation, "source_sha256": text_hash(text),
                            "source_start": start, "source_end": end, "source_length": len(text),
                            "thread_id": "thread1", "subject": "synthetic", "date": "2026-09-10T00:00:00Z"}}
              for index, (start, end) in enumerate(spans)]
    return {"email_id": "e1", "body": text, "chunks": chunks, "reconstruction_exact": True}


def test_long_unicode_text_can_be_read_without_gaps_or_overlap():
    body = ("甲🙂乙\n" * 2000) + "TAIL_FACT_381"
    doc, start, parts = email(body), 0, []
    while True:
        page = read_email_page("e1", start=start, limit=713, source_version="g1", loader=lambda _: doc)
        parts.append(page["body"])
        assert page["read_end"] - page["read_start"] == len(page["body"])
        if not page["has_more"]:
            break
        assert page["next_start"] > start
        start = page["next_start"]
    assert "".join(parts) == body


def test_chunk_read_offsets_and_visible_hash_are_exact():
    doc = email(overlap=True)
    page = read_email_page("e1", chunk_id="c1", start=1, limit=3, loader=lambda _: doc)
    assert page["body"] == "678"
    ref = evidence_reference(page["chunks"][0])
    assert (ref["visible_start"], ref["visible_end"]) == (1, 4)
    assert ref["visible_hash"] == text_hash("678")
    assert reread_evidence(ref, loader=lambda _: doc)["validation_status"].startswith("version_and_visible_hash_match")


def test_full_body_reads_preserve_original_overlap_offsets():
    page = read_email_page("e1", start=6, limit=3, loader=lambda _: email(overlap=True))
    assert page["body"] == "678"
    assert page["chunks"][0]["content"] == "6"
    assert page["chunks"][1]["content"] == "678"
    assert page["chunks"][1]["visible_start"] == 1


@pytest.mark.parametrize("changed", ["version", "body", "wrong_chunk"])
def test_persisted_reference_never_silently_reads_changed_evidence(changed):
    old, new = email(), email()
    ref = evidence_reference(old["chunks"][0])
    if changed == "version":
        new = email(generation="g2")
    elif changed == "body":
        new = email("different text")
    else:
        ref["chunk_id"] = "missing"
    with pytest.raises(EvidenceVersionMismatch):
        reread_evidence(ref, loader=lambda _: new)


def test_session_sqlite_reference_rereads_exact_original_slice(tmp_path):
    from api.sessions import SessionStore
    doc = email("prefix evidence tail")
    page = read_email_page("e1", chunk_id="c0", start=7, limit=8, loader=lambda _: doc)
    store = SessionStore(path=tmp_path / "sessions.sqlite3")
    store.record_result("owner", "session", "q", "answer", evidence_refs=[evidence_reference(page["chunks"][0])])
    reopened = SessionStore(path=tmp_path / "sessions.sqlite3")
    ref = reopened.evidence_refs("owner", "session")[0]
    assert reread_evidence(ref, loader=lambda _: doc)["body"] == "evidence"


def test_output_clipping_moves_continuation_to_actually_visible_end():
    text = "ABCDE" * 1800 + "TAIL_81"
    doc, start, reconstructed = email(text), 0, ""
    for _ in range(100):
        page = read_email_page("e1", start=start, limit=4000, loader=lambda _: doc)
        visible = json.loads(bounded_json(normalize_tool_result(page), 4000))["data"]
        piece = visible["chunks"][0]
        reconstructed += piece["content"]
        assert piece["visible_hash"] == text_hash(piece["content"])
        assert visible["read_end"] == start + len(piece["content"])
        if not visible["has_more"]:
            break
        assert visible["next_start"] > start
        start = visible["next_start"]
    assert reconstructed == text


def test_clipped_legacy_source_retains_full_chunk_version_but_only_prefix_visibility():
    source = {"email_id": "e1", "chunk_id": "c1", "content": "A" * 10000 + "TAIL"}
    result = json.loads(bounded_json(normalize_tool_result([source]), 2000))
    clipped = result["data"][0]
    assert clipped["chunk_sha256"] == text_hash(source["content"])
    assert clipped["visible_end"] == len(clipped["content"]) < 10004
    assert clipped["visible_hash"] == text_hash(clipped["content"])
    assert "TAIL" not in clipped["content"]


def search_rows(count=12):
    return [with_visible_reference({"email_id": f"e{i}", "chunk_id": f"c{i}", "snippet": "A" * 300,
                                   "source_version": "g1"}) for i in range(count)]


def test_search_pages_fixed_snapshot_never_repeat_provider_ranking():
    pages, fetch = SearchPages(), Mock(return_value=search_rows())
    common = dict(query="query", filters={"sender": "alice"}, generation="g1", owner="owner", limit=5, fetch=fetch)
    first = pages.page(**common)
    second = pages.page(**common, cursor=first["next_cursor"])
    third = pages.page(**common, cursor=second["next_cursor"])
    assert [row["email_id"] for page in (first, second, third) for row in page["items"]] == [f"e{i}" for i in range(12)]
    assert third["has_more"] is False and third["remaining"] == 0
    assert third["coverage"]["semantic_coverage_complete"] is False
    fetch.assert_called_once()


@pytest.mark.parametrize("field,new_value", [("query", "other"), ("filters", {"sender": "bob"}),
                                             ("generation", "g2"), ("owner", "other")])
def test_search_cursor_is_bound_to_query_filters_generation_owner(field, new_value):
    pages = SearchPages()
    common = dict(query="query", filters={"sender": "alice"}, generation="g1", owner="owner", limit=5, fetch=lambda: search_rows())
    first = pages.page(**common)
    with pytest.raises(EvidenceCursorError):
        pages.page(**{**common, field: new_value}, cursor=first["next_cursor"])


def test_search_json_clipping_resumes_after_only_retained_items():
    pages = SearchPages()
    common = dict(query="q", filters={}, generation="g1", owner="owner", limit=12, fetch=lambda: search_rows())
    first = pages.page(**common)
    visible = json.loads(bounded_json(normalize_tool_result(first), 2500))["data"]
    assert len(visible["items"]) < 12
    following = pages.page(**common, cursor=visible["next_cursor"])
    assert following["items"][0]["email_id"] == f'e{len(visible["items"])}'
    assert visible["remaining"] == 12 - len(visible["items"])


def test_search_cache_eviction_and_cursor_tampering_fail_closed():
    pages = SearchPages(max_snapshots=1)
    common = dict(query="q", filters={}, generation="g1", owner="owner", limit=1, fetch=lambda: search_rows())
    first = pages.page(**common)
    pages.page(**{**common, "query": "replacement"})
    with pytest.raises(EvidenceCursorError):
        pages.page(**common, cursor=first["next_cursor"])
    with pytest.raises(EvidenceCursorError):
        pages.page(**common, cursor="tampered")


def test_statistics_cover_explicit_filter_set_beyond_top_k_and_deduplicate():
    rows = [{"chunk_id": f"c{i}_{j}", "metadata": {"email_id": f"e{i}", "sender": "alice" if i < 40 else "bob",
            "date": "2026-09-01T00:00:00Z", "labels": '["urgent"]'}} for i in range(70) for j in range(2)]
    result = compute_scoped_stats(rows, filters={"sender": "alice", "date_hint": "2026-09"},
                                  generation="g1", now=datetime(2026, 9, 10, tzinfo=timezone.utc))
    assert result["total_emails"] == 40 and result["top5_senders"] == [{"sender": "alice", "count": 40}]
    assert result["coverage"]["metadata_rows_scanned"] == 140
    assert result["coverage"]["complete_within_index_scope"] is True
    assert result["coverage"]["mailbox_complete"] is False
    selected = compute_scoped_stats(rows, email_ids=["e1", "e60", "missing"])
    assert selected["total_emails"] == 2 and selected["coverage"]["scope"] == "selected_email_ids"
    assert selected["coverage"]["missing_selected_ids"] == ["missing"]
    with pytest.raises(ValueError, match="semantic-query"):
        compute_scoped_stats(rows, filters={"query": "all unpaid orders"})


def test_thread_dates_reply_links_and_decoding_uncertainty_do_not_assert_latest_validity():
    rows = [{"chunk_id": f"c{i}", "metadata": {"email_id": f"e{i}", "thread_id": "t1",
            "message_id": f"m{i}", "in_reply_to": "m0" if i else "",
            "date": f"2026-09-{i+1:02}T00:00:00Z"}} for i in range(3)]
    rows[1]["metadata"].update(attachments='[{"filename":"budget.xlsx","status":"not_read"}]',
                                 decode_quality='{"body":{"status":"suspect","corrected":false}}')
    page = read_thread_evidence("t1", limit=2, metadata_loader=lambda: (rows, "g1"))
    assert page["latest_dated_indexed_ids"] == ["e2"]
    assert "not_proof" in page["conclusion_status"]
    assert page["items"][1]["reply_parent_present_in_index"] is True
    assert page["items"][1]["coverage"]["unread_attachments"] == 1
    assert page["items"][1]["coverage"]["decode_status"] == "suspect"
    tail = read_thread_evidence("t1", start=page["next_start"], source_version="g1", metadata_loader=lambda: (rows, "g1"))
    assert tail["items"][0]["email_id"] == "e2"
    with pytest.raises(EvidenceVersionMismatch):
        read_thread_evidence("t1", source_version="old", metadata_loader=lambda: (rows, "g1"))


def test_claim_format_accepts_sources_without_claiming_entailment_and_rejects_unseen_tail():
    ref = evidence_reference({"email_id": "e1", "chunk_id": "c1", "content": "cost is 10", "source_version": "g1"})
    fields = ("email_id", "chunk_id", "source_version", "visible_start", "visible_end", "visible_hash")
    minimal = {key: ref[key] for key in fields}
    record = {"claims": [{"claim_id": "a", "text": "cost is 10", "assessment": "supported", "evidence_refs": [minimal]}], "conflicts": []}
    result = validate_claim_records(record, [ref])
    assert result["all_references_visible"] and result["entailment_checked"] is False
    record["claims"][0]["evidence_refs"][0]["visible_end"] = 1000
    assert validate_claim_records(record, [ref])["claims_with_missing_evidence"] == ["a"]
    record["conflicts"] = [{"claim_ids": ["a", "missing"], "relation": "supersedes"}]
    with pytest.raises(ValueError):
        validate_claim_records(record, [ref])


def test_version_ambiguous_citation_is_marked_for_review():
    from agents.agent_loop import _response
    first = with_visible_reference({"email_id": "e1", "chunk_id": "c1", "content": "old", "source_version": "old"})
    second = with_visible_reference({"email_id": "e1", "chunk_id": "c1", "content": "new", "source_version": "new"})
    run = RunContext(visible_evidence={("e1", "c1"): {**second, "visible_ranges": [evidence_reference(first), evidence_reference(second)]}})
    result = _response("value[e1#c1]", run, [], "trace", "success")
    assert result.metadata["status"] == "needs_review"
    assert result.metadata["citation_version_conflict_count"] == 1
    assert result.metadata["cited_evidence"] == []


@pytest.mark.parametrize("persistent", [False, True])
def test_session_listing_pages_reach_older_than_first_hundred(tmp_path, persistent):
    from api.sessions import SessionStore
    store = SessionStore(path=tmp_path / "sessions.sqlite3" if persistent else None)
    for index in range(115):
        store.record_result("owner", f"s{index}", "q", "a")
    store.record_result("different-owner", "hidden", "q", "a")
    if persistent:
        store = SessionStore(path=tmp_path / "sessions.sqlite3")
    offset, ids = 0, []
    while True:
        page = store.list_sessions_page("owner", limit=50, offset=offset)
        ids.extend(row["session_id"] for row in page["sessions"])
        if page["next_offset"] is None:
            break
        offset = page["next_offset"]
    assert len(ids) == len(set(ids)) == 115 and "hidden" not in ids
    assert len(store.list_sessions("owner", limit=10)) == 10
    with pytest.raises(ValueError):
        store.list_sessions_page("owner", offset=-1)


def test_actual_loop_reads_tail_by_visible_next_start_and_keeps_both_ranges(monkeypatch):
    import agents.agent_loop as loop
    import agents.tools as tools
    import config.settings as cfg
    body = "A" * 3990 + "TAIL_47"
    doc = email(body)
    monkeypatch.setattr(tools, "get_indexed_email", lambda _: doc)
    monkeypatch.setattr(cfg, "AGENT_TOOL_OUTPUT_LIMIT", 4000)
    observed = []
    def create(**kwargs):
        messages = kwargs["messages"]
        visible = [json.loads(row["content"])["data"] for row in messages if row["role"] == "tool"]
        if visible:
            latest = visible[-1]
            observed.append(copy.deepcopy(latest))
            if not latest["has_more"]:
                return NS(choices=[NS(finish_reason="stop", message=NS(content="Tail is 47 [e1#c0]", tool_calls=None))])
            start = latest["next_start"]
        else:
            start = 0
        tool = NS(id=f"read{start}", type="function", function=NS(name="get_email", arguments=json.dumps({
            "email_id": "e1", "chunk_id": "c0", "start": start, "limit": 4000, "source_version": "g1"})))
        return NS(choices=[NS(finish_reason="tool_calls", message=NS(content=None, tool_calls=[tool]))])
    monkeypatch.setattr(loop, "_get_client", lambda: NS(chat=NS(completions=NS(create=create))))
    from models.schemas import AgentRequest
    result = loop.run_agent_loop(AgentRequest(query="read tail"))
    assert result.metadata["status"] == "success"
    assert len(observed) >= 2
    assert "TAIL_47" not in observed[0]["chunks"][0]["content"]
    assert "TAIL_47" in observed[-1]["chunks"][0]["content"]
    refs = result.metadata["cited_evidence"]
    assert refs[0]["visible_start"] == 0 and refs[-1]["visible_end"] == len(body)
    assert all(ref["visible_hash"] == text_hash(body[ref["visible_start"]:ref["visible_end"]]) for ref in refs)
    assert result.metadata["citation_check"] == "identity_only_not_entailment"


def test_invalid_read_parameters_rejected_before_source_loader():
    loader = Mock()
    with pytest.raises(ValueError):
        read_email_page("e1", start=-1, loader=loader)
    loader.assert_not_called()


def test_legacy_synthetic_separator_cannot_stall_continuation():
    doc = {"email_id": "e1", "body": "abc\n\ndef", "reconstruction_exact": False,
           "chunks": [{"email_id": "e1", "chunk_id": "c1", "content": "abc"},
                      {"email_id": "e1", "chunk_id": "c2", "content": "def"}]}
    page = read_email_page("e1", start=3, limit=5, loader=lambda _: doc)
    visible = json.loads(bounded_json(normalize_tool_result(page), 4000))["data"]
    assert visible["read_end"] == 8 and not visible["has_more"]


def test_tool_boundary_preserves_run_budget_and_cancellation(monkeypatch):
    import agents.tools as tools
    from agents.runtime import RunCancelled
    from core.model_clients import ModelBudgetExceeded
    for error in (RunCancelled("cancelled"), ModelBudgetExceeded("budget")):
        monkeypatch.setitem(tools.TOOL_DISPATCH, "email_stats", Mock(side_effect=error))
        with pytest.raises(type(error)):
            tools.call_tool("email_stats", {})


def test_reread_verifies_long_reference_in_bounded_pages_and_returns_only_preview():
    doc = email("A" * 9000 + "TAIL")
    ref = evidence_reference(doc["chunks"][0])
    loader = Mock(return_value=doc)
    page = reread_evidence(ref, loader=loader)
    assert page["verification_complete"] is True
    assert page["verified_end"] == 9004 and len(page["body"]) == 1200
    assert page["next_start"] == 1200
    loader.assert_called_once()
    tail = reread_evidence(ref, loader=loader, start=9000)
    assert tail["body"] == "TAIL" and tail["has_more"] is False


def test_reread_excessive_span_rejected_before_loading(monkeypatch):
    import config.settings as cfg
    doc, loader = email("A" * 1000), Mock()
    monkeypatch.setattr(cfg, "EVIDENCE_VERIFY_CHAR_LIMIT", 500, raising=False)
    with pytest.raises(ValueError, match="character budget"):
        reread_evidence(evidence_reference(doc["chunks"][0]), loader=loader)
    loader.assert_not_called()


def test_missing_attachment_inventory_is_unknown_instead_of_empty():
    from core.evidence import source_coverage
    assert source_coverage({})["unread_attachments"] is None
    assert source_coverage({"attachments": "not-json"})["attachment_inventory_status"] == "unknown"
    assert source_coverage({"attachments": "[]"})["attachment_count"] == 0
