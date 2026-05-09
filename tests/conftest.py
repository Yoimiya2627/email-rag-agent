"""Shared test fixtures.

All tests run with mocked LLM clients and mocked ChromaDB — they never touch
real network or disk.  Each test gets its own fresh feature-flag state via the
`reset_cfg` autouse fixture so flag flips do not leak between cases.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, List, Optional

import pytest

# Ensure the project root is importable when pytest is invoked from anywhere.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# `sentence_transformers` is only used inside lazy `_get_model()` — under
# mocked tests we never call it, so stub the import to avoid a heavy dep.
if "sentence_transformers" not in sys.modules:
    _st = ModuleType("sentence_transformers")
    class _StubSentenceTransformer:  # noqa: D401
        def __init__(self, *a, **kw): pass
        def encode(self, *a, **kw): raise RuntimeError("stub: real model not loaded in tests")
    _st.SentenceTransformer = _StubSentenceTransformer
    sys.modules["sentence_transformers"] = _st

import config.settings as cfg
from models.schemas import Email, SearchResult


@pytest.fixture(autouse=True)
def reset_cfg(monkeypatch):
    """Snapshot the feature flags so each test starts from defaults."""
    for name in ("ENABLE_BM25", "ENABLE_RRF", "ENABLE_RERANKER", "ENABLE_QUERY_REWRITE"):
        monkeypatch.setattr(cfg, name, getattr(cfg, name), raising=True)
    yield


@pytest.fixture
def make_email():
    def _make(
        id: str = "e1",
        subject: str = "Hello",
        sender: str = "alice@example.com",
        recipients: Optional[List[str]] = None,
        date: str = "2026-05-09",
        body: str = "body text",
        labels: Optional[List[str]] = None,
        thread_id: Optional[str] = None,
    ) -> Email:
        return Email(
            id=id,
            subject=subject,
            sender=sender,
            recipients=recipients if recipients is not None else ["bob@example.com"],
            date=date,
            body=body,
            labels=labels if labels is not None else ["work"],
            thread_id=thread_id,
        )
    return _make


@pytest.fixture
def make_search_result():
    def _make(
        chunk_id: str,
        score: float = 1.0,
        email_id: Optional[str] = None,
        content: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> SearchResult:
        return SearchResult(
            chunk_id=chunk_id,
            email_id=email_id or chunk_id.split("_")[0],
            content=content or f"content of {chunk_id}",
            score=score,
            metadata=metadata or {"email_id": email_id or chunk_id.split("_")[0]},
        )
    return _make


@pytest.fixture
def fake_openai_response():
    """Build a fake OpenAI ChatCompletion response with optional reasoning_content.

    The real OpenAI SDK returns nested objects; tests only ever read
    ``resp.choices[0].message.content`` /  ``.reasoning_content`` and
    ``resp.choices[0].finish_reason``, so mimicking with SimpleNamespace is
    enough.
    """
    def _make(content: str = "", reasoning: Optional[str] = None, finish_reason: str = "stop") -> Any:
        message = SimpleNamespace(content=content, reasoning_content=reasoning)
        choice = SimpleNamespace(message=message, finish_reason=finish_reason)
        return SimpleNamespace(choices=[choice])
    return _make
