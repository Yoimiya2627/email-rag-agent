"""Tests for agents/agent_loop.py — the function-calling loop and its max-steps guardrail.

All tests use a scripted fake OpenAI client and a stubbed call_tool, so they
never hit the network or run real tools.
"""
import json
from types import SimpleNamespace

import agents.agent_loop as loop_mod
from agents.agent_loop import run_agent_loop
from models.schemas import AgentRequest


def _tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _response(content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])


class _ScriptedClient:
    """Fake OpenAI client that returns pre-scripted responses in order."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._responses:
            return self._responses.pop(0)
        return _response(content="(fallback)")


def _use_client(monkeypatch, client):
    monkeypatch.setattr(loop_mod, "_get_client", lambda: client)


def test_loop_returns_immediately_when_no_tool_calls(monkeypatch):
    client = _ScriptedClient([_response(content="直接回答")])
    _use_client(monkeypatch, client)

    out = run_agent_loop(AgentRequest(query="你好"))
    assert out.answer == "直接回答"
    assert out.metadata["steps"] == []
    assert len(client.calls) == 1


def test_loop_executes_tool_then_answers(monkeypatch):
    client = _ScriptedClient([
        _response(tool_calls=[_tool_call("c1", "search_emails", {"query": "预算"})]),
        _response(content="根据检索结果，预算邮件由 Alice 发出。"),
    ])
    _use_client(monkeypatch, client)

    tool_args = {}
    monkeypatch.setattr(
        loop_mod, "call_tool",
        lambda name, args: tool_args.update({"name": name, "args": args}) or [{"email_id": "e1"}],
    )

    out = run_agent_loop(AgentRequest(query="谁发了预算邮件"))
    assert "Alice" in out.answer
    assert tool_args == {"name": "search_emails", "args": {"query": "预算"}}
    assert out.metadata["steps"] == [{"tool": "search_emails", "arguments": {"query": "预算"}}]
    assert len(client.calls) == 2
    # The second LLM call must carry the tool result back as a tool-role message.
    second_msgs = client.calls[1]["messages"]
    assert any(m.get("role") == "tool" for m in second_msgs)
    assert any(m.get("role") == "assistant" and m.get("tool_calls") for m in second_msgs)


def test_loop_passes_tool_schemas(monkeypatch):
    client = _ScriptedClient([_response(content="ok")])
    _use_client(monkeypatch, client)

    run_agent_loop(AgentRequest(query="hi"))
    assert client.calls[0]["tools"] is loop_mod.TOOL_SCHEMAS


def test_loop_stops_at_max_steps(monkeypatch):
    """A model that always asks for a tool must not loop forever."""
    always_tool = [
        _response(tool_calls=[_tool_call(f"c{i}", "search_emails", {"query": "q"})])
        for i in range(50)
    ]
    client = _ScriptedClient(always_tool)
    _use_client(monkeypatch, client)
    monkeypatch.setattr(loop_mod, "call_tool", lambda name, args: {"ok": True})
    monkeypatch.setattr(loop_mod.cfg, "AGENT_MAX_STEPS", 4)

    out = run_agent_loop(AgentRequest(query="loopy"))
    assert out.metadata.get("max_steps_reached") is True
    assert len(out.metadata["steps"]) == 4   # exactly AGENT_MAX_STEPS tool rounds
    assert len(client.calls) == 5            # 4 loop calls + 1 forced-final call


def test_loop_degrades_malformed_tool_arguments(monkeypatch):
    """Malformed JSON arguments must degrade to an empty call, not crash."""
    bad = _response(tool_calls=[
        SimpleNamespace(
            id="c1", type="function",
            function=SimpleNamespace(name="email_stats", arguments="{not json"),
        )
    ])
    client = _ScriptedClient([bad, _response(content="done")])
    _use_client(monkeypatch, client)

    seen = {}
    monkeypatch.setattr(loop_mod, "call_tool", lambda name, args: seen.update({"args": args}) or {})

    out = run_agent_loop(AgentRequest(query="统计"))
    assert out.answer == "done"
    assert seen["args"] == {}
