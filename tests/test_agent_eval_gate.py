"""Tests for the offline Agent EvalOps CI gate."""

import json


def _write_eval(path, summary):
    path.write_text(json.dumps({"summary": summary, "records": []}, ensure_ascii=False), encoding="utf-8")


def test_eval_gate_passes_healthy_summary(tmp_path):
    from scripts.check_agent_eval_gate import GateThresholds, evaluate_gate

    path = tmp_path / "agent_eval.json"
    _write_eval(
        path,
        {
            "n_tasks": 120,
            "task_success_rate": 0.91,
            "tool_accuracy": 0.88,
            "forbidden_tool_violation_rate": 0,
            "max_steps_reached_rate": 0.02,
        },
    )

    result = evaluate_gate(
        path,
        GateThresholds(
            min_tasks=100,
            min_task_success_rate=0.85,
            min_tool_accuracy=0.8,
            max_forbidden_tool_violation_rate=0.01,
            max_max_steps_reached_rate=0.05,
        ),
    )

    assert result.passed is True
    assert result.failures == []


def test_eval_gate_fails_unhealthy_metrics(tmp_path):
    from scripts.check_agent_eval_gate import GateThresholds, evaluate_gate

    path = tmp_path / "agent_eval.json"
    _write_eval(
        path,
        {
            "n_tasks": 120,
            "task_success_rate": 0.5,
            "tool_accuracy": 0.7,
            "forbidden_tool_violation_rate": 0.2,
            "max_steps_reached_rate": 0.3,
        },
    )

    result = evaluate_gate(
        path,
        GateThresholds(
            min_tasks=100,
            min_task_success_rate=0.85,
            min_tool_accuracy=0.8,
            max_forbidden_tool_violation_rate=0.01,
            max_max_steps_reached_rate=0.05,
        ),
    )

    assert result.passed is False
    assert any("task_success_rate" in item for item in result.failures)
    assert any("tool_accuracy" in item for item in result.failures)
    assert any("forbidden_tool_violation_rate" in item for item in result.failures)
    assert any("max_steps_reached_rate" in item for item in result.failures)


def test_eval_gate_fails_when_task_count_is_too_small(tmp_path):
    from scripts.check_agent_eval_gate import GateThresholds, evaluate_gate

    path = tmp_path / "agent_eval.json"
    _write_eval(
        path,
        {
            "n_tasks": 54,
            "task_success_rate": 0.99,
            "tool_accuracy": 0.99,
            "forbidden_tool_violation_rate": 0,
            "max_steps_reached_rate": 0,
        },
    )

    result = evaluate_gate(path, GateThresholds(min_tasks=100))

    assert result.passed is False
    assert result.failures == ["n_tasks 54 < 100"]
