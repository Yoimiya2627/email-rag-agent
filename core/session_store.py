"""Tenant-aware session memory stores."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Protocol

import config.settings as cfg
from core.memory import ConversationMemory


class SessionMemoryLike(Protocol):
    def add(self, role: str, content: str) -> None: ...
    def to_messages(self) -> list[dict]: ...
    def clear(self) -> None: ...


class InMemorySessionMemoryStore:
    """Process-local session store kept for tests and simple local demos."""

    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self._sessions: dict[tuple[str, str], ConversationMemory] = {}
        self._lock = threading.Lock()

    def get(self, tenant_id: str, session_id: str) -> ConversationMemory:
        key = (tenant_id, session_id)
        with self._lock:
            memory = self._sessions.get(key)
            if memory is None:
                memory = ConversationMemory(max_turns=self.max_turns)
                self._sessions[key] = memory
            return memory

    def clear(self, tenant_id: str, session_id: str) -> None:
        self.get(tenant_id, session_id).clear()


class SQLiteSessionMemory:
    """ConversationMemory-compatible handle backed by SQLite rows."""

    def __init__(self, db_path: str | Path, tenant_id: str, session_id: str, max_turns: int):
        self.db_path = Path(db_path)
        self.tenant_id = tenant_id
        self.session_id = session_id
        self.max_turns = max_turns
        self._lock = threading.Lock()

    def add(self, role: str, content: str) -> None:
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO session_messages (tenant_id, session_id, role, content, created_order)
                VALUES (?, ?, ?, ?, (
                    SELECT COALESCE(MAX(created_order), 0) + 1
                    FROM session_messages
                    WHERE tenant_id = ? AND session_id = ?
                ))
                """,
                (self.tenant_id, self.session_id, role, content, self.tenant_id, self.session_id),
            )
            self._trim(conn)

    def to_messages(self) -> list[dict]:
        with self._lock, sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT role, content
                FROM session_messages
                WHERE tenant_id = ? AND session_id = ?
                ORDER BY created_order ASC, id ASC
                """,
                (self.tenant_id, self.session_id),
            ).fetchall()
        return [{"role": role, "content": content} for role, content in rows]

    def clear(self) -> None:
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM session_messages WHERE tenant_id = ? AND session_id = ?",
                (self.tenant_id, self.session_id),
            )

    def _trim(self, conn: sqlite3.Connection) -> None:
        max_messages = self.max_turns * 2
        conn.execute(
            """
            DELETE FROM session_messages
            WHERE id IN (
                SELECT id
                FROM session_messages
                WHERE tenant_id = ? AND session_id = ?
                ORDER BY created_order DESC, id DESC
                LIMIT -1 OFFSET ?
            )
            """,
            (self.tenant_id, self.session_id, max_messages),
        )


class SQLiteSessionMemoryStore:
    """SQLite-backed session store shared across app instances/workers."""

    def __init__(self, path: str | Path, max_turns: int = 5):
        self.path = Path(path)
        self.max_turns = max_turns
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tenant_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_order INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_session_messages_lookup
                ON session_messages(tenant_id, session_id, created_order)
                """
            )

    def get(self, tenant_id: str, session_id: str) -> SQLiteSessionMemory:
        return SQLiteSessionMemory(self.path, tenant_id, session_id, self.max_turns)

    def clear(self, tenant_id: str, session_id: str) -> None:
        self.get(tenant_id, session_id).clear()


def create_session_store_from_settings():
    backend = cfg.SESSION_STORE_BACKEND
    if backend == "sqlite":
        return SQLiteSessionMemoryStore(cfg.APP_SQLITE_PATH, max_turns=cfg.SESSION_MAX_TURNS)
    if backend == "memory":
        return InMemorySessionMemoryStore(max_turns=cfg.SESSION_MAX_TURNS)
    raise ValueError(f"unknown session store backend: {backend}")
