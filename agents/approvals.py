"""Human-in-the-loop approval store for high-risk agent actions."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import config.settings as cfg


class ApprovalStore:
    """Approval store with JSON and SQLite backends.

    The public API stays stable for the tool/API layer. JSON remains useful for
    local tests and demos; SQLite gives a durable shared store for production-like
    multi-worker runs.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        backend: str | None = None,
        tenant_id: str | None = None,
    ):
        self.backend = (backend or cfg.APPROVAL_STORE_BACKEND).lower()
        default_path = cfg.APP_SQLITE_PATH if self.backend == "sqlite" else cfg.APPROVAL_STORE_PATH
        self.path = Path(path or default_path)
        self.tenant_id = tenant_id or cfg.DEFAULT_TENANT_ID
        self._lock = threading.Lock()
        if self.backend not in {"json", "sqlite"}:
            raise ValueError(f"unknown approval store backend: {self.backend}")
        if self.backend == "sqlite":
            self._ensure_sqlite()

    def _ensure_sqlite(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS approvals (
                    approval_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    risk_level TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    requested_by TEXT NOT NULL,
                    reviewer TEXT,
                    review_note TEXT,
                    result_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_approvals_tenant_status ON approvals(tenant_id, status)"
            )

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, items: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

    def _row_to_item(self, row: sqlite3.Row) -> dict[str, Any]:
        item = {
            "approval_id": row["approval_id"],
            "tenant_id": row["tenant_id"],
            "action_type": row["action_type"],
            "risk_level": row["risk_level"],
            "status": row["status"],
            "payload": json.loads(row["payload_json"]),
            "requested_by": row["requested_by"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
        if row["reviewer"] is not None:
            item["reviewer"] = row["reviewer"]
        if row["review_note"] is not None:
            item["review_note"] = row["review_note"]
        if row["result_json"] is not None:
            item["result"] = json.loads(row["result_json"])
        return item

    def create(
        self,
        action_type: str,
        payload: dict[str, Any],
        requested_by: str = "agent",
        risk_level: str = "high",
    ) -> dict[str, Any]:
        now = _utc_now()
        item = {
            "approval_id": str(uuid.uuid4()),
            "tenant_id": self.tenant_id,
            "action_type": action_type,
            "risk_level": risk_level,
            "status": "pending",
            "payload": payload,
            "requested_by": requested_by,
            "created_at": now,
            "updated_at": now,
        }
        if self.backend == "sqlite":
            with self._lock, sqlite3.connect(self.path) as conn:
                conn.execute(
                    """
                    INSERT INTO approvals (
                        approval_id, tenant_id, action_type, risk_level, status,
                        payload_json, requested_by, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item["approval_id"],
                        self.tenant_id,
                        action_type,
                        risk_level,
                        "pending",
                        json.dumps(payload, ensure_ascii=False),
                        requested_by,
                        now,
                        now,
                    ),
                )
            return item
        with self._lock:
            items = self._load()
            items.append(item)
            self._save(items)
        return item

    def list(self, status: str | None = None) -> list[dict[str, Any]]:
        if self.backend == "sqlite":
            query = "SELECT * FROM approvals WHERE tenant_id = ?"
            params: list[Any] = [self.tenant_id]
            if status:
                query += " AND status = ?"
                params.append(status)
            query += " ORDER BY created_at ASC"
            with self._lock, sqlite3.connect(self.path) as conn:
                conn.row_factory = sqlite3.Row
                return [self._row_to_item(row) for row in conn.execute(query, params)]
        with self._lock:
            items = self._load()
        items = [item for item in items if item.get("tenant_id", self.tenant_id) == self.tenant_id]
        if status:
            return [item for item in items if item.get("status") == status]
        return items

    def get(self, approval_id: str) -> dict[str, Any]:
        for item in self.list():
            if item.get("approval_id") == approval_id:
                return item
        raise KeyError(f"approval_id {approval_id!r} not found")

    def approve(
        self,
        approval_id: str,
        reviewer: str = "human",
        note: str = "",
        executor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return self._review(
            approval_id=approval_id,
            status="approved",
            reviewer=reviewer,
            note=note,
            result={"mode": "simulated_send", "provider": "simulated", "sent": True},
            executor=executor,
        )

    def reject(
        self,
        approval_id: str,
        reviewer: str = "human",
        note: str = "",
    ) -> dict[str, Any]:
        return self._review(
            approval_id=approval_id,
            status="rejected",
            reviewer=reviewer,
            note=note,
            result={"mode": "blocked_by_human", "sent": False},
        )

    def _review(
        self,
        approval_id: str,
        status: str,
        reviewer: str,
        note: str,
        result: dict[str, Any],
        executor: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if self.backend == "sqlite":
            with self._lock, sqlite3.connect(self.path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT * FROM approvals WHERE approval_id = ? AND tenant_id = ?",
                    (approval_id, self.tenant_id),
                ).fetchone()
                if row is None:
                    raise KeyError(f"approval_id {approval_id!r} not found")
                item = self._row_to_item(row)
                if item.get("status") != "pending":
                    raise ValueError(f"approval {approval_id!r} is already {item.get('status')}")
                resolved_result = executor(dict(item)) if executor else result
                now = _utc_now()
                conn.execute(
                    """
                    UPDATE approvals
                    SET status = ?, reviewer = ?, review_note = ?, result_json = ?, updated_at = ?
                    WHERE approval_id = ? AND tenant_id = ?
                    """,
                    (
                        status,
                        reviewer,
                        note,
                        json.dumps(resolved_result, ensure_ascii=False),
                        now,
                        approval_id,
                        self.tenant_id,
                    ),
                )
                item.update({
                    "status": status,
                    "reviewer": reviewer,
                    "review_note": note,
                    "result": resolved_result,
                    "updated_at": now,
                })
                return item
        with self._lock:
            items = self._load()
            for item in items:
                if (
                    item.get("approval_id") == approval_id
                    and item.get("tenant_id", self.tenant_id) == self.tenant_id
                ):
                    if item.get("status") != "pending":
                        raise ValueError(f"approval {approval_id!r} is already {item.get('status')}")
                    resolved_result = executor(dict(item)) if executor else result
                    item.update({
                        "status": status,
                        "reviewer": reviewer,
                        "review_note": note,
                        "result": resolved_result,
                        "updated_at": _utc_now(),
                    })
                    self._save(items)
                    return item
        raise KeyError(f"approval_id {approval_id!r} not found")


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
