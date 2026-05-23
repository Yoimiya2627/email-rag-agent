"""Tests for generation-context assembly and token-budget guards."""
from __future__ import annotations

import config.settings as cfg
from core.generator import build_context


def test_build_context_caps_total_content_chars(monkeypatch, make_search_result):
    monkeypatch.setattr(cfg, "GENERATION_CONTEXT_CHAR_LIMIT", 12, raising=False)
    results = [
        make_search_result("a", content="a" * 10),
        make_search_result("b", content="b" * 10),
    ]

    context = build_context(results)

    assert "a" * 10 in context
    assert "bb" in context
    assert "bbb" not in context
