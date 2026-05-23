"""Human-in-the-loop approval store for high-risk agent actions."""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import config.settings as cfg


class ApprovalStore:
    """Small JSON-file backed approval store.

    This is intentionally simple and local: enough to demonstrate the safety
    boundary.  A production deployment can swap the storage behind this class
    for Redis or a database without changing the tool/API contract.
    """

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or cfg.APPROVAL_STORE_PATH)
        self._lock = threading.Lock()

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, items: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

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
            "action_type": action_type,
            "risk_level": risk_level,
            "status": "pending",
            "payload": payload,
            "requested_by": requested_by,
            "created_at": now,
            "updated_at": now,
        }
        with self._lock:
            items = self._load()
            items.append(item)
            self._save(items)
        return item

    def list(self, status: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            items = self._load()
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
    ) -> dict[str, Any]:
        return self._review(
            approval_id=approval_id,
            status="approved",
            reviewer=reviewer,
            note=note,
            result={"mode": "simulated_send", "sent": True},
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
    ) -> dict[str, Any]:
        with self._lock:
            items = self._load()
            for item in items:
                if item.get("approval_id") == approval_id:
                    if item.get("status") != "pending":
                        raise ValueError(f"approval {approval_id!r} is already {item.get('status')}")
                    item.update({
                        "status": status,
                        "reviewer": reviewer,
                        "review_note": note,
                        "result": result,
                        "updated_at": _utc_now(),
                    })
                    self._save(items)
                    return item
        raise KeyError(f"approval_id {approval_id!r} not found")


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
