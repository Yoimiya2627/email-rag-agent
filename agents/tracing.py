"""JSONL tracing helpers for agent runs and tool calls."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import config.settings as cfg
from agents.log_storage import append_jsonl, read_jsonl_tail

_WRITE_LOCK = threading.Lock()
_logger = logging.getLogger(__name__)
_ALLOWED = {"run_id", "tool", "status", "tool_backend", "step", "steps", "tool_calls",
            "query_chars", "answer_chars", "latency_ms", "argument_hash", "argument_chars",
            "request_id", "error_code", "error_type", "cache_hit"}


def safe_event(payload: dict[str, Any]) -> dict[str, Any]:
    """Only operational scalars are stored, never free-form bodies/errors/arguments."""
    output = {}
    for key in _ALLOWED & payload.keys():
        value = payload[key]
        if isinstance(value, (int, float, bool)) or value is None:
            output[key] = value
        elif isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", value):
            output[key] = value
    return output


class AgentTraceRecorder:
    """Append-only trace recorder for one agent run."""

    def __init__(
        self,
        path: str | Path | None = None,
        enabled: bool | None = None,
        trace_id: str | None = None,
    ):
        from agents.execution_scope import current_execution_scope
        scope = current_execution_scope()
        evaluating = scope is not None and scope.evaluation
        default_path = Path(scope.run_dir) / 'trace.jsonl' if evaluating else cfg.AGENT_TRACE_LOG_PATH
        self.path = Path(path or default_path)
        self.enabled = (True if evaluating else cfg.ENABLE_AGENT_TRACE) if enabled is None else enabled
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
            **safe_event(payload),
        }
        try:
            append_jsonl(self.path,row,max_bytes=getattr(cfg,'LOG_MAX_BYTES',5_000_000),
                         backups=getattr(cfg,'LOG_BACKUP_COUNT',3))
        except (OSError, ValueError) as exc:
            _logger.warning("trace write failed error_type=%s", type(exc).__name__)


def load_events(path: str | Path | None = None, *, limit: int = 1000,
                max_bytes: int = 1_000_000) -> list[dict[str, Any]]:
    trace_path = Path(path or cfg.AGENT_TRACE_LOG_PATH)
    return read_jsonl_tail(trace_path,limit=limit,max_bytes=max_bytes)


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
