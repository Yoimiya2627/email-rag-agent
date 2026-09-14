"""Tests for agents/agent_loop.py — the function-calling loop and its max-steps guardrail.

All tests use a scripted fake OpenAI client and a stubbed call_tool, so they
never hit the network or run real tools.
"""
import json
import pytest
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

    out = run_agent_loop(AgentRequest(query="帮我核对邮件内容"))
    assert out.answer == "直接回答"
    assert out.metadata["steps"] == []
    assert len(client.calls) == 1


@pytest.mark.parametrize(('name', 'offset', 'expected'), [
    ('', 8, 'UTC+08:00'), ('', -5, 'UTC-05:00'),
    ('America/New_York', 8, 'America/New_York'),
])
def test_agent_receives_actual_retrieval_calendar(monkeypatch, name, offset, expected):
    monkeypatch.setattr(loop_mod.cfg, 'RETRIEVAL_TIMEZONE', name)
    monkeypatch.setattr(loop_mod.cfg, 'RETRIEVAL_TIMEZONE_OFFSET_HOURS', offset)
    client = _ScriptedClient([_response(content='日期按检索时区核对')])
    _use_client(monkeypatch, client)
    run_agent_loop(AgentRequest(query='请核对邮件发件日期'))
    system = client.calls[0]['messages'][0]['content']
    assert f'日历时区为 {expected}' in system
    assert '先换算到此检索时区' in system
    assert '按邮件当日的夏令时规则' in system
    assert '不猜测为 UTC' in system


def _citation_context():
    from agents.runtime import RunContext
    context=RunContext()
    source={'email_id':'mail','chunk_id':'visible-chunk','content':'Read evidence','score':1.0,
            'metadata':{'index_generation':'v1'}}
    context.evidence[('mail','visible-chunk')]=source
    context.visible_evidence[('mail','visible-chunk')]=source
    return context


def test_bare_visible_chunk_citation_becomes_verified_full_reference():
    context=_citation_context()
    result=loop_mod._response('Evidence [visible-chunk]',context,[], 'trace','success')
    assert result.answer=='Evidence [mail#visible-chunk]'
    assert len(result.metadata['cited_evidence'])==1
    assert result.metadata['invalid_citation_count']==0


def test_bare_candidate_ambiguous_id_and_markdown_links_are_not_promoted():
    context=_citation_context()
    context.evidence[('mail','candidate')]={**context.evidence[('mail','visible-chunk')],'chunk_id':'candidate'}
    context.visible_evidence[('other','visible-chunk')]={**context.visible_evidence[('mail','visible-chunk')],'email_id':'other'}
    answer='[candidate] [unknown] [visible-chunk] [visible-chunk](https://example.invalid) ![visible-chunk](image)'
    result=loop_mod._response(answer,context,[], 'trace','success')
    assert result.answer==answer and result.metadata['cited_evidence']==[]


def test_repaired_bare_reference_still_checks_version_conflicts():
    context=_citation_context()
    context.visible_evidence[('mail','visible-chunk')]['visible_ranges']=[{'source_version':'v1'},{'source_version':'v2'}]
    result=loop_mod._response('[visible-chunk]',context,[],'trace','success')
    assert result.metadata['status']=='needs_review'
    assert result.metadata['citation_version_conflict_count']==1
    assert result.metadata['cited_evidence']==[]


def test_unique_bare_id_normalization_preserves_markdown_links_and_definitions():
    context=_citation_context()
    links='[visible-chunk](https://example.invalid) ![visible-chunk](image) [visible-chunk][ref]\n[visible-chunk] [ref]\n[visible-chunk]: https://example.invalid'
    result=loop_mod._response(links+'\nA citation: [visible-chunk]',context,[],'trace','success')
    assert result.answer==links+'\nA citation: [mail#visible-chunk]'
    assert len(result.metadata['cited_evidence'])==1


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
    assert out.metadata["steps"][0]["tool"] == "search_emails"
    assert out.metadata["steps"][0]["status"] == "success"
    assert "arguments" not in out.metadata["steps"][0]
    assert len(client.calls) == 2
    # The second LLM call must carry the tool result back as a tool-role message.
    second_msgs = client.calls[1]["messages"]
    assert any(m.get("role") == "tool" for m in second_msgs)
    assert any(m.get("role") == "assistant" and m.get("tool_calls") for m in second_msgs)


def test_loop_records_trace_metadata_and_tool_event(monkeypatch, tmp_path):
    client = _ScriptedClient([
        _response(tool_calls=[_tool_call("c1", "search_emails", {"query": "预算"})]),
        _response(content="done"),
    ])
    _use_client(monkeypatch, client)
    monkeypatch.setattr(loop_mod, "call_tool", lambda name, args: [{"email_id": "e1"}])
    monkeypatch.setattr(loop_mod.cfg, "ENABLE_AGENT_TRACE", True, raising=False)
    monkeypatch.setattr(loop_mod.cfg, "AGENT_TRACE_LOG_PATH", str(tmp_path / "traces.jsonl"), raising=False)

    out = run_agent_loop(AgentRequest(query="谁发了预算邮件"))

    assert out.metadata["trace_id"]
    trace_text = (tmp_path / "traces.jsonl").read_text(encoding="utf-8")
    assert '"event": "agent_start"' in trace_text
    assert '"event": "tool_call"' in trace_text
    assert '"tool": "search_emails"' in trace_text
    assert '"event": "agent_end"' in trace_text


def test_loop_passes_tool_schemas(monkeypatch):
    client = _ScriptedClient([_response(content="ok")])
    _use_client(monkeypatch, client)

    run_agent_loop(AgentRequest(query="search the project emails"))
    assert client.calls[0]["tools"] is loop_mod.TOOL_SCHEMAS


def test_loop_can_use_configured_mcp_tool_backend(monkeypatch):
    client = _ScriptedClient([
        _response(tool_calls=[_tool_call("c1", "email_stats", {})]),
        _response(content="共有 3 封邮件。"),
    ])
    _use_client(monkeypatch, client)

    class FakeBackend:
        def tool_schemas(self):
            return [{
                "type": "function",
                "function": {
                    "name": "email_stats",
                    "description": "Stats",
                    "parameters": {"type": "object", "properties": {}},
                },
            }]

        def call_tool(self, name, args):
            return {"total_emails": 3}

    monkeypatch.setattr(loop_mod.cfg, "AGENT_TOOL_BACKEND", "mcp")
    monkeypatch.setattr(loop_mod, "create_mcp_backend_from_settings", lambda: FakeBackend())

    out = run_agent_loop(AgentRequest(query="邮件总数"))

    assert out.answer == "共有 3 封邮件。"
    assert client.calls[0]["tools"][0]["function"]["name"] == "email_stats"
    assert out.metadata["steps"][0]["tool"] == "email_stats"
    assert out.metadata["steps"][0]["status"] == "success"


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
    """Malformed JSON must be rejected without executing even a zero-arg tool."""
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
    assert seen == {}
    assert out.metadata["steps"][0]["error_code"] == "validation_error"


def test_loop_handles_multi_step_task(monkeypatch):
    """A multi-step task: search once, then draft a reply per found email."""
    client = _ScriptedClient([
        _response(tool_calls=[_tool_call("c1", "search_emails", {"query": "询价"})]),
        _response(tool_calls=[
            _tool_call("c2", "draft_reply", {"email_id": "e1", "instruction": "报价"}),
            _tool_call("c3", "draft_reply", {"email_id": "e2", "instruction": "报价"}),
        ]),
        _response(content="已为 2 封询价邮件起草回复。"),
    ])
    _use_client(monkeypatch, client)
    monkeypatch.setattr(loop_mod, "call_tool", lambda name, args: {"tool": name, "args": args})

    out = run_agent_loop(AgentRequest(query="找出询价邮件并逐封起草回复"))
    tools_used = [s["tool"] for s in out.metadata["steps"]]
    assert tools_used == ["search_emails", "draft_reply", "draft_reply"]
    assert "2 封" in out.answer
    assert len(client.calls) == 3


def test_loop_blocks_repeated_identical_tool_calls(monkeypatch):
    """Same tool + identical args beyond AGENT_MAX_REPEAT is blocked, not re-run."""
    same_call = [
        _response(tool_calls=[_tool_call(f"c{i}", "search_emails", {"query": "q"})])
        for i in range(10)
    ]
    client = _ScriptedClient(same_call)
    _use_client(monkeypatch, client)
    exec_count = {"n": 0}
    monkeypatch.setattr(
        loop_mod, "call_tool",
        lambda name, args: exec_count.__setitem__("n", exec_count["n"] + 1) or {"ok": 1},
    )
    monkeypatch.setattr(loop_mod.cfg, "AGENT_MAX_REPEAT", 2)
    monkeypatch.setattr(loop_mod.cfg, "AGENT_MAX_STEPS", 6)

    out = run_agent_loop(AgentRequest(query="loopy"))
    assert exec_count["n"] == 2  # the tool actually runs only AGENT_MAX_REPEAT times
    assert any(s.get("blocked") == "repeat" for s in out.metadata["steps"])


def test_loop_truncates_oversized_tool_output(monkeypatch):
    """A huge tool result is truncated before being fed back into the context."""
    huge = "x" * 50000
    client = _ScriptedClient([
        _response(tool_calls=[_tool_call("c1", "get_email", {"email_id": "e1"})]),
        _response(content="done"),
    ])
    _use_client(monkeypatch, client)
    monkeypatch.setattr(loop_mod, "call_tool", lambda name, args: {"body": huge})
    monkeypatch.setattr(loop_mod.cfg, "AGENT_TOOL_OUTPUT_LIMIT", 1000)

    run_agent_loop(AgentRequest(query="读邮件"))
    tool_msg = next(m for m in client.calls[1]["messages"] if m.get("role") == "tool")
    assert len(tool_msg["content"]) <= 1000 + 40
    assert "truncated" in tool_msg["content"]
    assert json.loads(tool_msg["content"])["truncated"] is True
