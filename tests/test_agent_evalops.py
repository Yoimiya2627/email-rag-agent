"""Tests for Agent EvalOps reporting and trace-linked failure attribution."""

import json
from pathlib import Path

from agents.evalops import build_eval_report, classify_eval_record, events_by_trace_id


def test_events_by_trace_id_groups_trace_rows():
    events = [
        {"trace_id": "t1", "event": "agent_start"},
        {"trace_id": "t2", "event": "agent_start"},
        {"trace_id": "t1", "event": "tool_call", "tool": "search_emails"},
        {"event": "tool_call", "tool": "email_stats"},
    ]

    grouped = events_by_trace_id(events)

    assert [row["event"] for row in grouped["t1"]] == ["agent_start", "tool_call"]
    assert grouped["t2"][0]["event"] == "agent_start"
    assert "" not in grouped


def test_classify_eval_record_prefers_policy_violations():
    record = {
        "success": 0,
        "tool_accuracy": True,
        "forbidden_tool_violation": True,
        "max_steps_reached": False,
    }

    assert classify_eval_record(record, []) == "forbidden_tool"


def test_classify_eval_record_uses_trace_events_for_tool_error_and_approval():
    failed_tool = [
        {"event": "tool_call", "tool": "email_stats", "status": "error"},
    ]
    approval = [
        {"event": "tool_call", "tool": "send_email", "status": "approval_required"},
    ]

    assert classify_eval_record({"success": 0, "tool_accuracy": True}, failed_tool) == "tool_error"
    assert classify_eval_record({"success": 0, "tool_accuracy": True}, approval) == "approval_required"


def test_build_eval_report_contains_summary_records_and_trace_details():
    payload = {
        "summary": {
            "n_tasks": 1,
            "task_success_rate": 0,
            "tool_accuracy": 0,
            "avg_steps": 2,
            "max_steps_reached_rate": 0,
            "forbidden_tool_violation_rate": 1,
        },
        "records": [
            {
                "id": "send-001",
                "task": "发送预算确认邮件",
                "task_type": "send_approval",
                "risk_level": "high",
                "expected_tools": ["send_email"],
                "forbidden_tools": [],
                "actual_tools": ["send_email"],
                "trace_id": "trace-1",
                "failure_category": "approval_required",
                "success": 0,
                "reason": "requires human approval",
            }
        ],
    }
    trace_events = [
        {"trace_id": "trace-1", "event": "agent_start"},
        {"trace_id": "trace-1", "event": "tool_call", "tool": "send_email", "status": "approval_required", "latency_ms": 7},
    ]

    report = build_eval_report(payload, trace_events)

    assert "# Agent EvalOps Report" in report
    assert "send-001" in report
    assert "approval_required" in report
    assert "trace-1" in report
    assert "send_email" in report


def test_agent_testset_has_evalops_metadata():
    path = Path("data/agent_testset.json")
    cases = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "id",
        "task",
        "task_type",
        "risk_level",
        "expected_tools",
        "forbidden_tools",
        "success_criteria",
    }

    assert len(cases) >= 50
    assert len({case["id"] for case in cases}) == len(cases)
    for case in cases:
        assert required.issubset(case)
        assert isinstance(case["expected_tools"], list)
        assert isinstance(case["forbidden_tools"], list)
        assert case["risk_level"] in {"low", "medium", "high"}
