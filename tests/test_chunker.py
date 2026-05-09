"""Regression test for the chunker overlap=0 path."""
from core.chunker import chunk_text


def test_zero_overlap_is_respected():
    text = "0123456789ABCDEFGHIJ"
    out = chunk_text(text, chunk_size=10, chunk_overlap=0, min_chunk_size=1)
    assert out == ["0123456789", "ABCDEFGHIJ"]
