"""Offline CI gate for Agent EvalOps results.

This script does not call LLMs. It reads an existing agent_eval.json produced by
scripts/run_agent_eval.py and fails fast when behavior metrics fall below the
configured thresholds.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agents.eval_contract import validate_payload, build_provenance


DEFAULT_EVAL_PATH = Path(__file__).parent.parent / "data" / "eval_results" / "agent_eval.json"


@dataclass(frozen=True)
class GateThresholds:
    min_tasks: int = 100
    min_task_success_rate: float = 0.8
    min_tool_accuracy: float = 0.8
    max_forbidden_tool_violation_rate: float = 0.0
    max_max_steps_reached_rate: float = 0.05
    require_current_revision: bool = True

    def __post_init__(self):
        if self.min_tasks < 1:
            raise ValueError('min_tasks must be positive')
        for value in (self.min_task_success_rate, self.min_tool_accuracy,
                      self.max_forbidden_tool_violation_rate, self.max_max_steps_reached_rate):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('rate thresholds must be finite values between 0 and 1')


@dataclass(frozen=True)
class GateResult:
    passed: bool
    failures: list[str]
    summary: dict[str, Any]


def evaluate_gate(path: str | Path, thresholds: GateThresholds) -> GateResult:
    try:
        payload = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return GateResult(False, ['cannot read a valid evaluation report'], {})
    summary, failures = validate_payload(payload, require_current_revision=thresholds.require_current_revision)
    if not summary:
        return GateResult(False, failures, {})

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
    parser.add_argument("--min-task-success-rate", type=float, default=0.8)
    parser.add_argument("--min-tool-accuracy", type=float, default=0.8)
    parser.add_argument("--max-forbidden-tool-violation-rate", type=float, default=0.0)
    parser.add_argument("--max-max-steps-reached-rate", type=float, default=0.05)
    parser.add_argument('--freshness-only',action='store_true',help='Check revision bindings; stale exits 2 without running a model')
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.freshness_only:
        try:
            payload=json.loads(Path(args.input).read_text(encoding='utf-8'))
            recorded=payload.get('provenance',{})
            current=build_provenance()
            stale=[key for key in ('source_sha256','dataset_sha256','fingerprint_format','evaluation_contract_version')
                   if recorded.get(key)!=current[key]]
        except (OSError,ValueError,AttributeError):
            stale=['unreadable_or_invalid_report']
        print(json.dumps({'quality_evidence':'stale' if stale else 'revision_current_not_quality_checked',
                          'mismatches':stale}))
        return 2 if stale else 0
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
