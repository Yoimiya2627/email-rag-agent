"""JSONL tracing helpers for agent runs and tool calls."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import config.settings as cfg


class AgentTraceRecorder:
    """Append-only trace recorder for one agent run."""

    def __init__(
        self,
        path: str | Path | None = None,
        enabled: bool | None = None,
        trace_id: str | None = None,
    ):
        self.path = Path(path or cfg.AGENT_TRACE_LOG_PATH)
        self.enabled = cfg.ENABLE_AGENT_TRACE if enabled is None else enabled
        self.trace_id = trace_id or str(uuid.uuid4())

    @classmethod
    def from_settings(cls) -> "AgentTraceRecorder":
        return cls()

    def record(self, event: str, **payload: Any) -> None:
        if not self.enabled:
            return
        row = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "trace_id": self.trace_id,
            "event": event,
            **payload,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def load_events(path: str | Path | None = None) -> list[dict[str, Any]]:
    trace_path = Path(path or cfg.AGENT_TRACE_LOG_PATH)
    if not trace_path.exists():
        return []
    rows = []
    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_events(events: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(events)
    runs = {row.get("trace_id") for row in rows if row.get("event") == "agent_start"}
    tool_events = [row for row in rows if row.get("event") == "tool_call"]
    latencies = [
        float(row["latency_ms"])
        for row in tool_events
        if isinstance(row.get("latency_ms"), (int, float))
    ]
    return {
        "runs": len(runs),
        "tool_calls": len(tool_events),
        "tool_errors": sum(1 for row in tool_events if row.get("status") == "error"),
        "approval_required": sum(
            1 for row in tool_events if row.get("status") == "approval_required"
        ),
        "avg_tool_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
    }
