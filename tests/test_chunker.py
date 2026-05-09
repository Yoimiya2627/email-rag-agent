"""Tests for core/chunker.py — paragraph-aware chunker with overlap and min-size merge."""
import json

import pytest

from core.chunker import _force_split, chunk_email, chunk_text


def test_short_text_returns_single_chunk():
    out = chunk_text("hello world", chunk_size=100, chunk_overlap=10, min_chunk_size=5)
    assert out == ["hello world"]


def test_two_short_paragraphs_merge_into_one_chunk():
    text = "first line\n\nsecond line"
    out = chunk_text(text, chunk_size=100, chunk_overlap=10, min_chunk_size=5)
    assert len(out) == 1
    assert "first line" in out[0]
    assert "second line" in out[0]


def test_oversize_paragraph_force_splits_with_overlap():
    text = "A" * 35
    size, overlap = 10, 3
    out = chunk_text(text, chunk_size=size, chunk_overlap=overlap, min_chunk_size=1)

    assert len(out) >= 2
    for chunk in out:
        assert len(chunk) <= size

    for i in range(len(out) - 1):
        assert out[i + 1][:overlap] == out[i][-overlap:]


def test_zero_overlap_is_respected():
    text = "0123456789ABCDEFGHIJ"
    out = chunk_text(text, chunk_size=10, chunk_overlap=0, min_chunk_size=1)
    assert out == ["0123456789", "ABCDEFGHIJ"]


def test_force_split_helper_directly():
    chunks = _force_split("0123456789ABCDEF", size=6, overlap=2)
    assert chunks[0] == "012345"
    assert chunks[1] == "456789"
    assert chunks[2] == "89ABCD"
    for i in range(len(chunks) - 1):
        assert chunks[i + 1][:2] == chunks[i][-2:]


def test_short_tail_chunk_merges_into_previous():
    # Two paragraphs whose sizes force a flush; the second flushed chunk is
    # below min_chunk_size, so the merge step folds it into the first.
    text = ("A" * 15) + "\n\n" + "BB"
    out = chunk_text(text, chunk_size=15, chunk_overlap=2, min_chunk_size=8)
    assert len(out) == 1
    assert "A" * 15 in out[0]
    assert "BB" in out[0]


def test_empty_and_whitespace_input_returns_empty_list():
    assert chunk_text("", chunk_size=100, chunk_overlap=10, min_chunk_size=5) == []
    assert chunk_text("\n\n   \n\n", chunk_size=100, chunk_overlap=10, min_chunk_size=5) == []


def test_chunk_email_serializes_metadata_and_carries_subject(make_email):
    email = make_email(
        id="e42",
        subject="Quarterly Review",
        sender="boss@example.com",
        recipients=["alice@example.com", "bob@example.com"],
        labels=["work", "urgent"],
        thread_id=None,
        body="paragraph one body content for the chunker to handle.",
    )
    chunks = chunk_email(email)

    assert len(chunks) >= 1
    first = chunks[0]
    assert first.content.startswith("Subject: Quarterly Review")
    assert first.email_id == "e42"
    assert first.chunk_id == "e42_chunk_0"
    assert first.chunk_index == 0

    md = first.metadata
    assert md["subject"] == "Quarterly Review"
    assert md["sender"] == "boss@example.com"
    assert md["thread_id"] == ""
    assert json.loads(md["labels"]) == ["work", "urgent"]
    assert json.loads(md["recipients"]) == ["alice@example.com", "bob@example.com"]

    for i, chunk in enumerate(chunks):
        assert chunk.chunk_id == f"e42_chunk_{i}"
        assert chunk.chunk_index == i
