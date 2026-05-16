"""Tests for scripts/run_agent_eval.py — tool-accuracy, aggregation, task scoring."""
import pytest

import scripts.run_agent_eval as ae


def test_tool_accuracy_is_subset_check():
    assert ae.tool_accuracy(["search_emails"], ["search_emails", "get_email"]) is True
    assert ae.tool_accuracy(["search_emails", "draft_reply"], ["search_emails"]) is False
    assert ae.tool_accuracy([], ["anything"]) is True          # no expectation → pass
    assert ae.tool_accuracy(["email_stats"], []) is False


def test_aggregate_computes_rates():
    records = [
        {"success": 1, "tool_accuracy": True,  "n_steps": 2, "max_steps_reached": False},
        {"success": 0, "tool_accuracy": True,  "n_steps": 4, "max_steps_reached": True},
        {"success": 1, "tool_accuracy": False, "n_steps": 3, "max_steps_reached": False},
    ]
    out = ae.aggregate(records)
    assert out["n_tasks"] == 3
    assert out["task_success_rate"] == pytest.approx(2 / 3, abs=1e-3)
    assert out["tool_accuracy"] == pytest.approx(2 / 3, abs=1e-3)
    assert out["avg_steps"] == pytest.approx(3.0)
    assert out["max_steps_reached_rate"] == pytest.approx(1 / 3, abs=1e-3)


def test_aggregate_handles_empty_records():
    out = ae.aggregate([])
    assert out["n_tasks"] == 0
    assert out["task_success_rate"] == 0


def test_evaluate_task_builds_record(monkeypatch):
    import agents.agent_loop as loop_mod
    from models.schemas import AgentResponse

    monkeypatch.setattr(
        loop_mod, "run_agent_loop",
        lambda req, memory=None: AgentResponse(
            answer="发件人是 Alice。",
            sources=[],
            metadata={
                "steps": [{"tool": "search_emails", "arguments": {}}],
                "max_steps_reached": False,
            },
        ),
    )
    monkeypatch.setattr(ae, "judge_success", lambda client, task, answer: {"success": 1, "reason": "ok"})

    rec = ae.evaluate_task({"task": "谁发的", "expected_tools": ["search_emails"]}, client=object())
    assert rec["actual_tools"] == ["search_emails"]
    assert rec["tool_accuracy"] is True
    assert rec["success"] == 1
    assert rec["n_steps"] == 1
    assert rec["max_steps_reached"] is False


def test_evaluate_task_flags_missing_expected_tool(monkeypatch):
    import agents.agent_loop as loop_mod
    from models.schemas import AgentResponse

    monkeypatch.setattr(
        loop_mod, "run_agent_loop",
        lambda req, memory=None: AgentResponse(
            answer="...", sources=[],
            metadata={"steps": [{"tool": "search_emails", "arguments": {}}], "max_steps_reached": False},
        ),
    )
    monkeypatch.setattr(ae, "judge_success", lambda client, task, answer: {"success": 1, "reason": ""})

    # Task expected a draft_reply too — the agent never called it.
    rec = ae.evaluate_task(
        {"task": "起草回复", "expected_tools": ["search_emails", "draft_reply"]}, client=object()
    )
    assert rec["tool_accuracy"] is False
