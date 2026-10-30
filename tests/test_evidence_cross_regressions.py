import json
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from agents.runtime import bounded_json, normalize_tool_result
from core.evidence import evidence_reference, text_hash
from core.evidence_pages import read_email_page, reread_evidence, compute_scoped_stats, EvidenceVersionMismatch


def document(body="visible unseen-tail", **metadata):
    return {"email_id": "e", "body": body, "reconstruction_exact": True,
            "chunks": [{"email_id": "e", "chunk_id": "c", "content": body,
                "metadata": {"index_generation": "g1", "source_sha256": text_hash(body),
                             "source_start": 0, "source_end": len(body), "source_length": len(body), **metadata}}]}


def test_table_page_preserves_bounded_column_lineage_and_partial_marker():
    lineage = json.dumps([{"table_id": "t1", "row_id": "t1:r1", "status": "complete", "partial_row": False,
        "cells": [{"source_id": "t1:r1:c1", "column": 1, "headers": ["net total"]}]}])
    doc = document("header value=123.45 remaining source", table_context=lineage)
    page = read_email_page("e", chunk_id="c", start=7, limit=12, loader=lambda _: doc)
    visible = json.loads(bounded_json(normalize_tool_result(page), 4000))["data"]["chunks"][0]
    assert "net total" in visible["table_context"]
    assert '"partial_row":true' in visible["table_context"]
    assert visible["table_excerpt_truncated"] is True
    assert visible["visible_hash"] == text_hash(visible["content"])


def test_reread_rejects_changed_full_chunk_when_reference_includes_its_hash():
    old = document()
    ref = evidence_reference({**old["chunks"][0], "content": "visible", "chunk_sha256": text_hash(old["body"])})
    changed = document("visible MODIFIED TAIL", source_sha256=ref["source_sha256"])
    with pytest.raises(EvidenceVersionMismatch, match="chunk hash"):
        reread_evidence(ref, loader=lambda _: changed)
    assert reread_evidence(ref, loader=lambda _: old)["chunk_hash_match"] is True
    ref.pop("chunk_sha256")
    assert reread_evidence(ref, loader=lambda _: changed)["chunk_hash_match"] is None


def test_supplied_full_chunk_hash_cannot_bypass_verification_character_budget(monkeypatch):
    import config.settings as cfg
    doc = document("A" * 1000)
    ref = evidence_reference({**doc["chunks"][0], "content": "A", "chunk_sha256": text_hash(doc["body"])})
    monkeypatch.setattr(cfg, "EVIDENCE_VERIFY_CHAR_LIMIT", 500)
    with pytest.raises(ValueError, match="full chunk hash"):
        reread_evidence(ref, loader=lambda _: doc)


def test_statistics_filter_and_calendar_buckets_share_user_timezone():
    rows = [{"chunk_id": "c", "metadata": {"email_id": "e", "date": "2026-09-10T00:30:00+00:00"}}]
    result = compute_scoped_stats(rows, filters={"date_hint": "2026-09-09"},
                                 now=datetime(2026, 9, 10, tzinfo=ZoneInfo("America/New_York")))
    assert result["total_emails"] == 1
    assert result["daily_counts"] == [{"date": "2026-09-09", "count": 1}]
    assert result["date_bucketing"]["timezone"] == "America/New_York"


def test_statistics_unknown_or_ambiguous_dates_do_not_invent_daily_buckets():
    rows = [{"chunk_id": str(i), "metadata": {"email_id": str(i), "date": date}}
            for i, date in enumerate(["not-a-date", "2026-11-01T01:30:00", "2026-11-01T01:30:00-04:00"])]
    result = compute_scoped_stats(rows, now=datetime(2026, 11, 1, tzinfo=ZoneInfo("America/New_York")))
    assert result["total_emails"] == 3
    assert result["daily_counts"] == [{"date": "2026-11-01", "count": 1}]
    assert result["date_bucketing"]["unknown_date_emails"] == 2
