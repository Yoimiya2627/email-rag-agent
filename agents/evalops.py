"""Agent EvalOps helpers for trace-linked reports and failure attribution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


def events_by_trace_id(events: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group trace rows by trace_id, ignoring rows that cannot be linked."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in events:
        trace_id = str(row.get("trace_id") or "")
        if not trace_id:
            continue
        grouped.setdefault(trace_id, []).append(row)
    return grouped


def classify_eval_record(
    record: dict[str, Any],
    trace_events: Iterable[dict[str, Any]] | None = None,
) -> str:
    """Return a stable failure/success category for one agent eval record."""
    events = list(trace_events or [])
    if record.get("forbidden_tool_violation"):
        return "forbidden_tool"
    if record.get("max_steps_reached"):
        return "max_steps"
    if record.get("tool_accuracy") is False:
        return "missing_expected_tool"
    if any(step.get('status') in {'error', 'unknown'} for step in record.get('steps', [])):
        return 'tool_error'
    if record.get('tool_assertion_checks', {}).get('passed') is False:
        return 'tool_assertion_failed'
    if record.get('execution_contract_passed') is False:
        return 'execution_contract_failed'
    if any(row.get("event") == "tool_call" and row.get("status") == "error" for row in events):
        return "tool_error"
    if any(
        row.get("event") == "tool_call" and row.get("status") == "approval_required"
        for row in events
    ):
        return "approval_required"
    if record.get('scoring_status') == 'error' or str(record.get("reason", "")).lower().startswith("judge error"):
        return "judge_failed"
    if int(record.get("success", 0)) == 1:
        return "success"
    return "task_failed"


def build_eval_report(
    payload: dict[str, Any],
    trace_events: Iterable[dict[str, Any]] | None = None,
) -> str:
    """Build a Markdown EvalOps report from an agent-eval payload."""
    summary = payload.get("summary", {})
    records = payload.get("records", [])
    traces = events_by_trace_id(trace_events or [])
    lines = [
        "# Agent EvalOps Report",
        "",
        "## Summary",
        "",
        f"- Tasks: {summary.get('n_tasks', 0)}",
        f"- Task success rate: {_pct(summary.get('task_success_rate', 0))}",
        f"- Tool accuracy: {_pct(summary.get('tool_accuracy', 0))}",
        f"- Avg steps: {summary.get('avg_steps', 0)}",
        f"- Max steps reached rate: {_pct(summary.get('max_steps_reached_rate', 0))}",
        f"- Forbidden tool violation rate: {_pct(summary.get('forbidden_tool_violation_rate', 0))}",
        "",
        "## Records",
        "",
    ]

    for record in records:
        trace_id = str(record.get("trace_id") or "")
        trace_rows = traces.get(trace_id, [])
        category = record.get("failure_category") or classify_eval_record(record, trace_rows)
        lines.extend([
            f"### {record.get('id', record.get('task', 'task'))}",
            "",
            f"- Task: {record.get('task', '')}",
            f"- Type: {record.get('task_type', '')}",
            f"- Risk: {record.get('risk_level', '')}",
            f"- Success: {record.get('success', 0)}",
            f"- Failure category: {category}",
            f"- Scoring method/status: {record.get('scoring_method', 'unavailable')} / {record.get('scoring_status', 'unavailable')}",
            f"- Tool assertion checks: {record.get('tool_assertion_checks', 'unavailable')}",
            f"- Trace id: {trace_id}",
            f"- Expected tools: {', '.join(record.get('expected_tools', []) or [])}",
            f"- Actual tools: {', '.join(record.get('actual_tools', []) or [])}",
            f"- Forbidden tools: {', '.join(record.get('forbidden_tools', []) or [])}",
            f"- Reason: {record.get('reason', '')}",
            "",
        ])
        tool_events = [row for row in trace_rows if row.get("event") == "tool_call"]
        if tool_events:
            lines.append("| Tool | Status | Latency ms |")
            lines.append("|---|---|---:|")
            for row in tool_events:
                lines.append(
                    f"| {row.get('tool', '')} | {row.get('status', '')} | {row.get('latency_ms', 0)} |"
                )
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_eval_report(
    payload: dict[str, Any],
    output_path: str | Path,
    trace_events: Iterable[dict[str, Any]] | None = None,
) -> None:
    """Write a Markdown EvalOps report to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_eval_report(payload, trace_events), encoding="utf-8")


def _pct(value: Any) -> str:
    try:
        return f"{float(value):.0%}"
    except (TypeError, ValueError):
        return "0%"
