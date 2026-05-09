"""Tests for agents/coordinator.py — intent classification + GENERAL route fallback."""
from types import SimpleNamespace

import pytest

import agents.coordinator as coord_mod
from agents.coordinator import classify_intent, route
from models.schemas import AgentRequest, AgentResponse, IntentType


def _fake_client(response):
    """Build a fake OpenAI client whose .chat.completions.create returns `response`,
    or raises it if `response` is an Exception."""
    class _Completions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if isinstance(response, Exception):
                raise response
            return response

    completions = _Completions()
    chat = SimpleNamespace(completions=completions)
    return SimpleNamespace(chat=chat, _completions=completions)


def _patch_client(monkeypatch, client):
    monkeypatch.setattr(coord_mod, "_get_client", lambda: client)


def test_classify_intent_parses_normal_response(monkeypatch, fake_openai_response):
    resp = fake_openai_response(content='{"intent": "retrieve", "reason": "查找邮件"}')
    _patch_client(monkeypatch, _fake_client(resp))
    assert classify_intent("找一下关于X的邮件") == IntentType.RETRIEVE


@pytest.mark.parametrize("intent_str,expected", [
    ("retrieve", IntentType.RETRIEVE),
    ("summarize", IntentType.SUMMARIZE),
    ("write_reply", IntentType.WRITE_REPLY),
    ("analyze", IntentType.ANALYZE),
    ("general", IntentType.GENERAL),
])
def test_classify_intent_supports_all_intent_values(
    monkeypatch, fake_openai_response, intent_str, expected
):
    resp = fake_openai_response(content=f'{{"intent": "{intent_str}", "reason": "..."}}')
    _patch_client(monkeypatch, _fake_client(resp))
    assert classify_intent("query") == expected


def test_classify_intent_falls_back_to_reasoning_content(monkeypatch, fake_openai_response):
    resp = fake_openai_response(
        content="",
        reasoning='思考过程... 最终判定 {"intent": "summarize", "reason": "总结"}',
        finish_reason="length",
    )
    _patch_client(monkeypatch, _fake_client(resp))
    assert classify_intent("总结一下") == IntentType.SUMMARIZE


def test_classify_intent_strips_markdown_json_fence(monkeypatch, fake_openai_response):
    resp = fake_openai_response(content='```json\n{"intent": "analyze", "reason": "stats"}\n```')
    _patch_client(monkeypatch, _fake_client(resp))
    assert classify_intent("谁发邮件最多") == IntentType.ANALYZE


def test_classify_intent_falls_back_to_general_on_invalid_json(monkeypatch, fake_openai_response):
    resp = fake_openai_response(content="not json at all", reasoning=None)
    _patch_client(monkeypatch, _fake_client(resp))
    assert classify_intent("???") == IntentType.GENERAL


def test_classify_intent_falls_back_to_general_when_llm_raises(monkeypatch):
    client = _fake_client(RuntimeError("network down"))
    _patch_client(monkeypatch, client)
    assert classify_intent("anything") == IntentType.GENERAL


def test_classify_intent_falls_back_to_general_for_unknown_intent_label(
    monkeypatch, fake_openai_response
):
    resp = fake_openai_response(content='{"intent": "delete_everything", "reason": "..."}')
    _patch_client(monkeypatch, _fake_client(resp))
    assert classify_intent("???") == IntentType.GENERAL


def test_route_general_fallback_invokes_retriever_agent(monkeypatch):
    """GENERAL intent must land on RetrieverAgent (per agent_map)."""
    monkeypatch.setattr(coord_mod, "classify_intent", lambda q: IntentType.GENERAL)

    calls = {"init": 0, "run": 0, "request": None}

    class FakeRetrieverAgent:
        def __init__(self):
            calls["init"] += 1

        def run(self, request, memory=None):
            calls["run"] += 1
            calls["request"] = request
            return AgentResponse(answer="fallback answer", sources=[])

    import agents.retriever_agent as ra_mod
    monkeypatch.setattr(ra_mod, "RetrieverAgent", FakeRetrieverAgent)

    req = AgentRequest(query="hello world")
    response = route(req)

    assert calls["init"] == 1
    assert calls["run"] == 1
    assert calls["request"] is req
    assert response.intent == IntentType.GENERAL
    assert response.answer == "fallback answer"
