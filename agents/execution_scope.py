"""Trusted request identity and local state overrides, separate from model tools."""
from __future__ import annotations

import hashlib
import re
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Any


@dataclass
class ExecutionScope:
    owner_id: str
    session_id: str | None = None
    session_repository: Any = None
    context_epoch: int | None = None
    tool_result_store: Any = None
    operation_key: str | None = None
    approval_store_path: str | Path | None = None
    run_dir: str | Path | None = None
    evaluation: bool = False
    tool_observer: Callable[[str, dict, dict, str], dict[str, Any]] | None = None
    _slots: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.owner_id, str) or not self.owner_id.strip() or len(self.owner_id) > 256:
            raise ValueError('execution scope owner must be a nonempty identity')
        self.owner_id = self.owner_id.strip()
        if self.operation_key is not None:
            if (not isinstance(self.operation_key, str) or not self.operation_key.strip()
                    or len(self.operation_key) > 128 or '\n' in self.operation_key or '\r' in self.operation_key):
                raise ValueError('operation_key must contain 1 to 128 characters without newlines')
            self.operation_key = self.operation_key.strip()
        if self.evaluation and (not self.approval_store_path or not self.run_dir or not self.owner_id.startswith('eval:')):
            raise ValueError('evaluation requires an isolated owner, approval store and run directory')

    def approval_request_id(self, payload_digest: str) -> str | None:
        if self.operation_key is None:
            return None
        # Each logical request receives stable slots in action order. Replaying
        # a slot with changed content is rejected by ApprovalStore's hash check.
        # Identical calls in one run reuse the slot. Reordered writes fail closed.
        with self._lock:
            slot = self._slots.setdefault(payload_digest, len(self._slots) + 1)
        key = hashlib.sha256(self.operation_key.encode('utf-8')).hexdigest()
        return f'operation:{key}:send:{slot}'

    def snapshot_slots(self) -> dict[str, int]:
        with self._lock:
            return dict(self._slots)

    def restore_slots(self, slots: dict[str, int]) -> None:
        if (not isinstance(slots, dict) or len(slots) > 512
                or any(not isinstance(key, str) or not re.fullmatch(r'[0-9a-f]{64}', key)
                       or type(value) is not int or value < 1 for key, value in slots.items())
                or sorted(slots.values()) != list(range(1, len(slots) + 1))):
            raise ValueError('invalid persisted operation slots')
        with self._lock:
            if self._slots and self._slots != slots:
                raise ValueError('cannot replace active operation slots')
            self._slots = dict(slots)


_SCOPE: ContextVar[ExecutionScope | None] = ContextVar('email_execution_scope', default=None)


def current_execution_scope() -> ExecutionScope | None:
    return _SCOPE.get()


@contextmanager
def use_execution_scope(scope: ExecutionScope):
    token = _SCOPE.set(scope)
    try:
        yield scope
    finally:
        _SCOPE.reset(token)
