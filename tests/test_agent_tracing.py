"""Tests for agent trace recording and summarization."""

import json

from agents.tracing import AgentTraceRecorder, summarize_events


def test_trace_recorder_writes_jsonl_events(tmp_path):
    path = tmp_path / "agent_traces.jsonl"
    recorder = AgentTraceRecorder(path=path, enabled=True, trace_id="trace-1")

    recorder.record("agent_start", query="hello")
    recorder.record("agent_end", status="success", answer_chars=12)

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["event"] for row in rows] == ["agent_start", "agent_end"]
    assert all(row["trace_id"] == "trace-1" for row in rows)
    assert "ts" in rows[0]


def test_summarize_events_computes_tool_and_error_metrics():
    events = [
        {"event": "agent_start", "trace_id": "t1"},
        {"event": "tool_call", "trace_id": "t1", "tool": "search_emails", "status": "success", "latency_ms": 10},
        {"event": "tool_call", "trace_id": "t1", "tool": "send_email", "status": "approval_required", "latency_ms": 5},
        {"event": "agent_end", "trace_id": "t1", "status": "success"},
        {"event": "agent_start", "trace_id": "t2"},
        {"event": "tool_call", "trace_id": "t2", "tool": "email_stats", "status": "error", "latency_ms": 20},
        {"event": "agent_end", "trace_id": "t2", "status": "error"},
    ]

    summary = summarize_events(events)

    assert summary["runs"] == 2
    assert summary["tool_calls"] == 3
    assert summary["tool_errors"] == 1
    assert summary["approval_required"] == 1
    assert summary["avg_tool_latency_ms"] == 11.67
