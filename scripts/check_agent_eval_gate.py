"""Offline CI gate for Agent EvalOps results.

This script does not call LLMs. It reads an existing agent_eval.json produced by
scripts/run_agent_eval.py and fails fast when behavior metrics fall below the
configured thresholds.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_EVAL_PATH = Path(__file__).parent.parent / "data" / "eval_results" / "agent_eval.json"


@dataclass(frozen=True)
class GateThresholds:
    min_tasks: int = 100
    min_task_success_rate: float = 0.0
    min_tool_accuracy: float = 0.0
    max_forbidden_tool_violation_rate: float = 0.0
    max_max_steps_reached_rate: float = 1.0


@dataclass(frozen=True)
class GateResult:
    passed: bool
    failures: list[str]
    summary: dict[str, Any]


def load_summary(path: str | Path) -> dict[str, Any]:
    eval_path = Path(path)
    with eval_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return dict(payload.get("summary") or {})


def evaluate_gate(path: str | Path, thresholds: GateThresholds) -> GateResult:
    summary = load_summary(path)
    failures: list[str] = []

    n_tasks = int(summary.get("n_tasks", 0))
    task_success_rate = float(summary.get("task_success_rate", 0))
    tool_accuracy = float(summary.get("tool_accuracy", 0))
    forbidden_rate = float(summary.get("forbidden_tool_violation_rate", 0))
    max_steps_rate = float(summary.get("max_steps_reached_rate", 0))

    if n_tasks < thresholds.min_tasks:
        failures.append(f"n_tasks {n_tasks} < {thresholds.min_tasks}")
    if task_success_rate < thresholds.min_task_success_rate:
        failures.append(
            f"task_success_rate {task_success_rate:.4f} < {thresholds.min_task_success_rate:.4f}"
        )
    if tool_accuracy < thresholds.min_tool_accuracy:
        failures.append(f"tool_accuracy {tool_accuracy:.4f} < {thresholds.min_tool_accuracy:.4f}")
    if forbidden_rate > thresholds.max_forbidden_tool_violation_rate:
        failures.append(
            "forbidden_tool_violation_rate "
            f"{forbidden_rate:.4f} > {thresholds.max_forbidden_tool_violation_rate:.4f}"
        )
    if max_steps_rate > thresholds.max_max_steps_reached_rate:
        failures.append(
            f"max_steps_reached_rate {max_steps_rate:.4f} > "
            f"{thresholds.max_max_steps_reached_rate:.4f}"
        )

    return GateResult(passed=not failures, failures=failures, summary=summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check Agent EvalOps metrics against CI thresholds")
    parser.add_argument("--input", default=str(DEFAULT_EVAL_PATH), help="Path to agent_eval.json")
    parser.add_argument("--min-tasks", type=int, default=100)
    parser.add_argument("--min-task-success-rate", type=float, default=0.0)
    parser.add_argument("--min-tool-accuracy", type=float, default=0.0)
    parser.add_argument("--max-forbidden-tool-violation-rate", type=float, default=0.0)
    parser.add_argument("--max-max-steps-reached-rate", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate_gate(
        args.input,
        GateThresholds(
            min_tasks=args.min_tasks,
            min_task_success_rate=args.min_task_success_rate,
            min_tool_accuracy=args.min_tool_accuracy,
            max_forbidden_tool_violation_rate=args.max_forbidden_tool_violation_rate,
            max_max_steps_reached_rate=args.max_max_steps_reached_rate,
        ),
    )

    print("Agent EvalOps gate")
    print(f"  input                 : {args.input}")
    print(f"  tasks                 : {result.summary.get('n_tasks', 0)}")
    print(f"  task_success_rate     : {float(result.summary.get('task_success_rate', 0)):.4f}")
    print(f"  tool_accuracy         : {float(result.summary.get('tool_accuracy', 0)):.4f}")
    print(
        "  forbidden_tool_rate   : "
        f"{float(result.summary.get('forbidden_tool_violation_rate', 0)):.4f}"
    )
    print(f"  max_steps_rate        : {float(result.summary.get('max_steps_reached_rate', 0)):.4f}")
    if result.passed:
        print("  result                : PASS")
        return 0
    print("  result                : FAIL")
    for failure in result.failures:
        print(f"  - {failure}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
