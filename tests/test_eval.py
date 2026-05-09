"""Tests for scripts/run_ragas_eval.py — flag application, JSON extraction,
LLM/embedding scoring fallbacks, and per-version circuit-breaker reset."""
import json
from types import SimpleNamespace
from typing import Dict, List

import pytest

import config.settings as cfg
import scripts.run_ragas_eval as rev


def _fake_client(content: str = "", reasoning: str = None, exc: Exception = None):
    """Build a fake OpenAI client with a tracked .chat.completions.create."""
    calls = {"n": 0}

    class _Completions:
        def create(self, **kwargs):
            calls["n"] += 1
            if exc is not None:
                raise exc
            message = SimpleNamespace(content=content, reasoning_content=reasoning)
            choice = SimpleNamespace(message=message, finish_reason="stop")
            return SimpleNamespace(choices=[choice])

    chat = SimpleNamespace(completions=_Completions())
    return SimpleNamespace(chat=chat, _calls=calls)


def test_apply_flags_writes_back_to_cfg():
    rev.apply_flags(rev.VERSION_FLAGS["V4"])
    assert cfg.ENABLE_BM25 is True
    assert cfg.ENABLE_RRF is True
    assert cfg.ENABLE_RERANKER is True
    assert cfg.ENABLE_QUERY_REWRITE is True

    rev.apply_flags(rev.VERSION_FLAGS["V1"])
    assert cfg.ENABLE_BM25 is False
    assert cfg.ENABLE_RRF is False
    assert cfg.ENABLE_RERANKER is False
    assert cfg.ENABLE_QUERY_REWRITE is False


def test_extract_json_obj_pulls_object_from_noisy_text():
    out = rev._extract_json_obj('前缀垃圾文本 {"a": 1, "b": 2} 尾部噪声')
    assert json.loads(out) == {"a": 1, "b": 2}


def test_extract_json_obj_strips_markdown_fence():
    out = rev._extract_json_obj('```json\n{"intent": "x"}\n```')
    assert json.loads(out) == {"intent": "x"}


def test_score_response_returns_llm_scores_without_fallback(monkeypatch):
    payload = {"answer_relevancy": 0.9, "faithfulness": 0.8, "context_precision": 0.7}
    client = _fake_client(content=json.dumps(payload))

    embed_called = {"hit": False}
    monkeypatch.setattr(
        rev, "_score_by_embedding",
        lambda *a, **kw: embed_called.__setitem__("hit", True) or {"x": 1},
    )

    out = rev.score_response(client, "q?", "answer", ["ctx1", "ctx2"])
    assert out == payload
    assert embed_called["hit"] is False, "Embedding fallback must not run on LLM success"
    assert client._calls["n"] == 1


def test_score_response_falls_back_to_embedding_after_three_failures(monkeypatch):
    monkeypatch.setattr(rev.time, "sleep", lambda *_: None)  # don't actually sleep on retries
    client = _fake_client(exc=RuntimeError("LLM blew up"))

    fallback = {"answer_relevancy": 0.5, "faithfulness": 0.5, "context_precision": 0.5}
    monkeypatch.setattr(rev, "_score_by_embedding", lambda q, a, ctx: dict(fallback))

    out = rev.score_response(client, "q", "a", ["c1"])
    assert out == fallback
    assert client._calls["n"] == 3, "LLM scoring should retry exactly 3 times before degrading"


def test_score_response_returns_zeros_when_both_paths_fail(monkeypatch):
    monkeypatch.setattr(rev.time, "sleep", lambda *_: None)
    client = _fake_client(exc=RuntimeError("LLM blew up"))

    def boom(*a, **kw):
        raise RuntimeError("embedding service down too")

    monkeypatch.setattr(rev, "_score_by_embedding", boom)

    out = rev.score_response(client, "q", "a", ["c1"])
    assert out == {"answer_relevancy": 0.0, "faithfulness": 0.0, "context_precision": 0.0}


def test_score_response_uses_reasoning_content_when_message_content_empty(monkeypatch):
    payload = {"answer_relevancy": 0.42, "faithfulness": 0.42, "context_precision": 0.42}
    client = _fake_client(content="", reasoning=f"思考... {json.dumps(payload)} 结束")

    embed_called = {"hit": False}
    monkeypatch.setattr(
        rev, "_score_by_embedding",
        lambda *a, **kw: embed_called.__setitem__("hit", True) or {"x": 1},
    )

    out = rev.score_response(client, "q", "a", ["ctx"])
    assert out == payload
    assert embed_called["hit"] is False


def test_evaluate_version_resets_circuit_breaker_and_aggregates(monkeypatch):
    """evaluate_version() must reset the reranker circuit breaker once per
    version (technical_retrospective.md §4: cross-version pollution bug) and
    return a stable {version, flags, avg, records} structure."""
    reset_calls = {"n": 0}
    import core.reranker
    monkeypatch.setattr(
        core.reranker, "reset_circuit_breaker",
        lambda: reset_calls.__setitem__("n", reset_calls["n"] + 1),
    )

    monkeypatch.setattr(rev.time, "sleep", lambda *_: None)
    monkeypatch.setattr(
        rev, "run_single",
        lambda client, q, gt: {"answer": f"A:{q}", "contexts": [f"ctx-for-{q}"]},
    )
    monkeypatch.setattr(
        rev, "score_response",
        lambda client, q, a, ctx: {
            "answer_relevancy": 0.8, "faithfulness": 0.6, "context_precision": 0.4,
        },
    )

    testset = [
        {"question": "q1", "ground_truth": "g1"},
        {"question": "q2", "ground_truth": "g2"},
    ]
    out = rev.evaluate_version("V2", testset, limit=2, client=object())

    assert reset_calls["n"] == 1, "reset_circuit_breaker must be called exactly once per version"
    assert set(out.keys()) == {"version", "flags", "avg", "records"}
    assert out["version"] == "V2"
    assert out["flags"] == rev.VERSION_FLAGS["V2"]
    assert set(out["avg"].keys()) == {"answer_relevancy", "faithfulness", "context_precision"}
    assert out["avg"]["answer_relevancy"] == pytest.approx(0.8)
    assert out["avg"]["faithfulness"] == pytest.approx(0.6)
    assert out["avg"]["context_precision"] == pytest.approx(0.4)

    assert len(out["records"]) == 2
    for rec, item in zip(out["records"], testset):
        assert rec["question"] == item["question"]
        assert rec["ground_truth"] == item["ground_truth"]
        assert rec["answer"] == f"A:{item['question']}"
        assert rec["contexts"] == [f"ctx-for-{item['question']}"]
        assert rec["answer_relevancy"] == 0.8


def test_evaluate_version_isolates_run_single_failures(monkeypatch):
    """One failing question must not poison the rest of the version's records."""
    monkeypatch.setattr(rev.time, "sleep", lambda *_: None)
    import core.reranker
    monkeypatch.setattr(core.reranker, "reset_circuit_breaker", lambda: None)

    def flaky_run_single(client, q, gt):
        if q == "bad":
            raise RuntimeError("retrieval blew up for this question")
        return {"answer": f"A:{q}", "contexts": ["ctx"]}

    monkeypatch.setattr(rev, "run_single", flaky_run_single)
    monkeypatch.setattr(
        rev, "score_response",
        lambda *a, **kw: {"answer_relevancy": 1.0, "faithfulness": 1.0, "context_precision": 1.0},
    )

    testset = [
        {"question": "good1", "ground_truth": "g1"},
        {"question": "bad",   "ground_truth": "g2"},
        {"question": "good2", "ground_truth": "g3"},
    ]
    out = rev.evaluate_version("V2", testset, limit=3, client=object())

    assert len(out["records"]) == 3
    bad = next(r for r in out["records"] if r["question"] == "bad")
    assert "error" in bad
    assert bad["answer_relevancy"] == 0.0
    assert bad["faithfulness"] == 0.0
    assert bad["context_precision"] == 0.0

    for q in ("good1", "good2"):
        rec = next(r for r in out["records"] if r["question"] == q)
        assert rec["answer_relevancy"] == 1.0
