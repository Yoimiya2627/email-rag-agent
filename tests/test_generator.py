"""Tests for generation-context assembly and token-budget guards."""
from __future__ import annotations

import config.settings as cfg
from core.generator import build_context
import pytest


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


@pytest.mark.parametrize('offset,expected',[(8,'2026-08-17T05:01:00+08:00'),
                                          (-5,'2026-08-16T16:01:00-05:00')])
def test_model_mail_header_uses_retrieval_calendar_without_changing_source(monkeypatch,make_search_result,offset,expected):
    monkeypatch.setattr(cfg,'RETRIEVAL_TIMEZONE','')
    monkeypatch.setattr(cfg,'RETRIEVAL_TIMEZONE_OFFSET_HOURS',offset)
    original='2026-08-16T21:01:00+00:00'
    result=make_search_result('r',metadata={'date':original})
    assert expected in build_context([result])
    assert result.metadata['date']==original


@pytest.mark.parametrize('value',['2026-08-17','2026-08-17T05:01:00','unavailable'])
def test_naive_or_unknown_dates_never_acquire_invented_utc_offset(value):
    from core.generator import format_email_date
    rendered=format_email_date(value)
    assert value in rendered and '+00:00' not in rendered


def test_named_retrieval_timezone_observes_dst(monkeypatch):
    from core.generator import format_email_date
    monkeypatch.setattr(cfg,'RETRIEVAL_TIMEZONE','America/New_York')
    assert '2026-07-01T08:00:00-04:00' in format_email_date('2026-07-01T12:00:00Z')
    assert '2026-01-01T07:00:00-05:00' in format_email_date('2026-01-01T12:00:00Z')
