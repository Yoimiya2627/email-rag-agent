"""Transactional approvals. A legacy .json path uses an adjacent .sqlite3 store.

Legacy JSON is imported once and never rewritten. Executing/unknown actions are
never automatically replayed: SQLite cannot atomically commit a Gmail request.
"""
from __future__ import annotations

import hashlib
import base64
import json
import math
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

import config.settings as cfg


class ApprovalPreconditionError(ValueError):
    """A provider rejected the action before any external mutation was submitted."""


class ApprovalStore:
    """Claim each logical request once, across instances and processes.

    Success preserves status=approved and adds execution_state=succeeded.
    request_id is unique per owner and must be reused for logical-action retries.
    """

    def __init__(self, path: str | Path | None = None, *, ttl_seconds: float | None = None):
        self.path = Path(path or cfg.APPROVAL_STORE_PATH)
        self.db_path = self.path.with_suffix(".sqlite3") if self.path.suffix.lower() == ".json" else self.path
        self.ttl_seconds = _valid_ttl(getattr(cfg, "APPROVAL_TTL_SECONDS", 86400) if ttl_seconds is None else ttl_seconds)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._transaction() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS approvals (
                    approval_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL,
                    session_id TEXT, request_id TEXT NOT NULL,
                    action_type TEXT NOT NULL, risk_level TEXT NOT NULL,
                    payload_json TEXT NOT NULL, payload_hash TEXT NOT NULL,
                    status TEXT NOT NULL, execution_state TEXT NOT NULL,
                    requested_by TEXT NOT NULL, created_at REAL NOT NULL,
                    updated_at REAL NOT NULL, expires_at REAL NOT NULL,
                    reviewer TEXT, review_note TEXT, result_json TEXT,
                    error_code TEXT, claim_token TEXT,
                    UNIQUE(owner_id, request_id)
                )
            """)
            connection.execute("CREATE INDEX IF NOT EXISTS approval_owner_status ON approvals(owner_id,status,created_at)")
            connection.execute("CREATE INDEX IF NOT EXISTS approval_owner_created ON approvals(owner_id,created_at,approval_id)")
            if 'redacted_at' not in {row[1] for row in connection.execute('PRAGMA table_info(approvals)')}:
                connection.execute('ALTER TABLE approvals ADD COLUMN redacted_at REAL')
            connection.execute("CREATE TABLE IF NOT EXISTS approval_metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL)")
            connection.execute("""CREATE TABLE IF NOT EXISTS approval_reconciliations (
                id INTEGER PRIMARY KEY AUTOINCREMENT, approval_id TEXT NOT NULL, owner_id TEXT NOT NULL,
                outcome TEXT NOT NULL, evidence TEXT NOT NULL, reviewer TEXT NOT NULL,
                execution_stopped INTEGER NOT NULL, created_at REAL NOT NULL, result_json TEXT
            )""")
            self._import_legacy(connection)

    def _import_legacy(self, connection: sqlite3.Connection) -> None:
        if self.path.suffix.lower() != ".json" or not self.path.exists():
            return
        key = "legacy_json_v1:" + str(self.path.resolve())
        if connection.execute("SELECT 1 FROM approval_metadata WHERE key=?", (key,)).fetchone():
            return
        with self.path.open("r", encoding="utf-8-sig") as source:
            items = json.load(source)
        if not isinstance(items, list):
            raise ValueError("legacy approval store must contain a JSON list")
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("legacy approval record must be an object")
            approval_id = _identity(item.get("approval_id"), "approval_id")
            owner_id = _identity(item.get("owner_id", "local"), "owner_id")
            payload_json, payload_hash = _payload(item.get("action_type", ""), item.get("payload"))
            created = _parse_timestamp(item.get("created_at"), time.time())
            updated = _parse_timestamp(item.get("updated_at"), created)
            status = item.get("status", "pending")
            states = {"pending": "not_started", "approved": "succeeded", "rejected": "not_started",
                      "expired": "not_started", "failed": "not_started", "executing": "executing", "unknown": "unknown"}
            if status not in states or status == "executing":
                status = "unknown"
            result = item.get("result")
            if isinstance(result, dict) and (result.get("provider") == "simulated" or result.get("mode") == "simulated_send"):
                result = {**result, "sent": False}
            values = {
                "approval_id": approval_id, "owner_id": owner_id, "session_id": _session(item.get("session_id")),
                "request_id": _identity(item.get("request_id") or "legacy:" + approval_id, "request_id"),
                "action_type": item.get("action_type", ""), "risk_level": item.get("risk_level", "high"),
                "payload_json": payload_json, "payload_hash": payload_hash, "status": status, "execution_state": states[status],
                "requested_by": item.get("requested_by", "agent"), "created_at": created, "updated_at": updated,
                "expires_at": _parse_timestamp(item.get("expires_at"), created + self.ttl_seconds),
                "reviewer": item.get("reviewer"), "review_note": item.get("review_note"),
                "result_json": _json(result) if result is not None else None,
                "error_code": "reconciliation_required" if status == "unknown" else None,
            }
            existing = connection.execute("SELECT * FROM approvals WHERE approval_id=?", (approval_id,)).fetchone()
            if existing is not None:
                if any(existing[field] != values[field] for field in ("owner_id", "payload_hash", "request_id", "session_id")):
                    raise ValueError("legacy approval identity conflicts with SQLite record")
                continue
            self._insert(connection, values)
        connection.execute("INSERT INTO approval_metadata(key,value) VALUES (?,?)", (key, _utc_now()))

    @staticmethod
    def _insert(connection: sqlite3.Connection, values: dict[str, Any]) -> None:
        # Column names come only from the fixed internal dictionaries above/below.
        columns = ",".join(values)
        connection.execute(f"INSERT INTO approvals ({columns}) VALUES ({','.join('?' for _ in values)})", tuple(values.values()))

    @staticmethod
    def _find(connection: sqlite3.Connection, approval_id: str, owner_id: str) -> sqlite3.Row:
        row = connection.execute("SELECT * FROM approvals WHERE approval_id=?", (approval_id,)).fetchone()
        if row is None:
            raise KeyError(f"approval_id {approval_id!r} not found")
        if row["owner_id"] != owner_id:
            raise PermissionError("approval does not belong to the current owner")
        return row

    @staticmethod
    def _expire(connection: sqlite3.Connection, owner_id: str) -> None:
        now = time.time()
        connection.execute("UPDATE approvals SET status='expired',execution_state='not_started',updated_at=? "
                           "WHERE owner_id=? AND status='pending' AND expires_at<=?", (now, owner_id, now))

    def create(self, action_type: str, payload: dict[str, Any], requested_by: str = "agent", risk_level: str = "high", *,
               owner_id: str = "local", session_id: str | None = None, request_id: str | None = None,
               ttl_seconds: float | None = None) -> dict[str, Any]:
        owner_id = _identity(owner_id, "owner_id")
        request_id = _identity(str(uuid.uuid4()) if request_id is None else request_id, "request_id")
        session_id = _session(session_id)
        payload_json, payload_hash = _payload(action_type, payload)
        now = time.time()
        with self._transaction() as connection:
            self._expire(connection, owner_id)
            existing = connection.execute("SELECT * FROM approvals WHERE owner_id=? AND request_id=?", (owner_id, request_id)).fetchone()
            if existing is not None:
                if existing["payload_hash"] != payload_hash or existing["session_id"] != session_id:
                    raise ValueError("request_id already belongs to different approval content or session")
                return _item(existing)
            values = {
                "approval_id": str(uuid.uuid4()), "owner_id": owner_id, "session_id": session_id,
                "request_id": request_id, "action_type": action_type, "risk_level": risk_level,
                "payload_json": payload_json, "payload_hash": payload_hash, "status": "pending",
                "execution_state": "not_started", "requested_by": requested_by, "created_at": now, "updated_at": now,
                "expires_at": now + (self.ttl_seconds if ttl_seconds is None else _valid_ttl(ttl_seconds)),
            }
            self._insert(connection, values)
            return _item(self._find(connection, values["approval_id"], owner_id))

    def list(self, status: str | None = None, *, owner_id: str = "local", session_id: str | None = None) -> list[dict[str, Any]]:
        owner_id = _identity(owner_id, "owner_id")
        with self._transaction() as connection:
            self._expire(connection, owner_id)
            query, params = "SELECT * FROM approvals WHERE owner_id=?", [owner_id]
            if status:
                query += " AND status=?"
                params.append(status)
            if session_id is not None:
                query += " AND session_id IS ?"
                params.append(_session(session_id))
            return [_item(row) for row in connection.execute(query + " ORDER BY created_at,approval_id", params)]

    def get(self, approval_id: str, *, owner_id: str = "local") -> dict[str, Any]:
        owner_id = _identity(owner_id, "owner_id")
        with self._transaction() as connection:
            self._expire(connection, owner_id)
            return _item(self._find(connection, approval_id, owner_id))

    def list_page(self, status: str | None = None, *, owner_id: str = 'local',
                  session_id: str | None = None, limit: int = 50, cursor: str | None = None,
                  include_payload: bool = False) -> dict[str, Any]:
        """Bounded keyset pages. Bodies require explicit detail opt-in."""
        owner_id = _identity(owner_id, 'owner_id')
        if type(limit) is not int or not 1 <= limit <= 200:
            raise ValueError('limit must be an integer between 1 and 200')
        if type(include_payload) is not bool:
            raise ValueError('include_payload must be a boolean')
        binding = hashlib.sha256(_json([owner_id,status,session_id]).encode()).hexdigest()
        query, params = ' FROM approvals WHERE owner_id=?', [owner_id]
        if status:
            query += ' AND status=?'
            params.append(status)
        if session_id is not None:
            query += ' AND session_id IS ?'
            params.append(_session(session_id))
        if cursor:
            try:
                if len(cursor)>1024: raise ValueError()
                stamp, identifier, saved_binding = json.loads(base64.urlsafe_b64decode(cursor.encode()))
                if saved_binding != binding or type(stamp) not in (int,float) or not math.isfinite(stamp):
                    raise ValueError()
                identifier = _identity(identifier,'cursor id')
            except Exception as exc:
                raise ValueError('invalid approval page cursor') from exc
            query += ' AND (created_at>? OR (created_at=? AND approval_id>?))'
            params.extend([stamp,stamp,identifier])
        columns = '*' if include_payload else ('approval_id,owner_id,session_id,request_id,action_type,risk_level,'
                 'payload_hash,status,execution_state,created_at,updated_at,expires_at,error_code,redacted_at')
        with self._transaction() as connection:
            self._expire(connection,owner_id)
            rows = connection.execute('SELECT '+columns+query+' ORDER BY created_at,approval_id LIMIT ?',
                                      [*params,limit+1]).fetchall()
        next_cursor = None
        if len(rows)>limit:
            last = rows[limit-1]
            next_cursor = base64.urlsafe_b64encode(_json([last['created_at'],last['approval_id'],binding]).encode()).decode()
        return {'items':[_item(row) if include_payload else _summary(row) for row in rows[:limit]],
                'next_cursor':next_cursor,'limit':limit}

    def retain(self, *, owner_id: str = 'local', before: float, apply: bool = False) -> dict[str, Any]:
        """Redact terminal private content, preserving permanent dedup tombstones.

        Executing/unknown records retain evidence until manual reconciliation.
        Dry-run performs no writes. Backups/legacy JSON have separate retention.
        """
        owner_id = _identity(owner_id,'owner_id')
        if type(before) not in (int,float) or not math.isfinite(before) or before<0 or before>time.time():
            raise ValueError('before must be a finite past Unix timestamp')
        if type(apply) is not bool:
            raise ValueError('apply must be boolean')
        with self._transaction() as connection:
            predicate = "owner_id=? AND status IN ('approved','rejected','failed','expired') AND updated_at<? AND redacted_at IS NULL"
            params = (owner_id,before)
            count = connection.execute('SELECT COUNT(*) FROM approvals WHERE '+predicate,params).fetchone()[0]
            excluded = connection.execute("SELECT COUNT(*) FROM approvals WHERE owner_id=? AND status IN ('executing','unknown') AND updated_at<?",params).fetchone()[0]
            if apply and count:
                connection.execute("UPDATE approval_reconciliations SET evidence='[redacted]',reviewer='[redacted]',result_json=NULL WHERE approval_id IN (SELECT approval_id FROM approvals WHERE "+predicate+')',params)
                connection.execute("UPDATE approvals SET payload_json='{}',result_json=NULL,review_note=NULL,reviewer=NULL,requested_by='[redacted]',redacted_at=? WHERE "+predicate,(time.time(),*params))
        return {'dry_run':not apply,'eligible':count,'redacted':count if apply else 0,
                'uncertain_preserved':excluded,'tombstones_deleted':0}

    def approve(self, approval_id: str, reviewer: str = "human", note: str = "",
                executor: Callable[[dict[str, Any]], dict[str, Any]] | None = None, *, owner_id: str = "local") -> dict[str, Any]:
        owner_id = _identity(owner_id, "owner_id")
        claim_token, conflict = str(uuid.uuid4()), None
        with self._transaction() as connection:
            self._expire(connection, owner_id)
            row = self._find(connection, approval_id, owner_id)
            if row["status"] == "approved":
                return _item(row)
            if row["status"] != "pending":
                # Commit expiry before raising. In-flight/unknown records never replay.
                conflict = f"approval is {row['status']}; no action was executed"
            else:
                _, actual_hash = _payload(row["action_type"], json.loads(row["payload_json"]))
                if actual_hash != row["payload_hash"]:
                    connection.execute("UPDATE approvals SET status='failed',error_code='payload_integrity',updated_at=? WHERE approval_id=?", (time.time(), approval_id))
                    conflict = "approval payload integrity check failed"
                else:
                    connection.execute("UPDATE approvals SET status='executing',execution_state='executing',claim_token=?, "
                                       "reviewer=?,review_note=?,updated_at=? WHERE approval_id=? AND status='pending'",
                                       (claim_token, reviewer, note, time.time(), approval_id))
                    claimed = _item(self._find(connection, approval_id, owner_id))
        if conflict is not None:
            raise ValueError(conflict)
        # Do not hold a database transaction while calling an external provider.
        try:
            result = executor(claimed) if executor else {"mode": "simulated_send", "provider": "simulated", "sent": False}
            if not isinstance(result, dict):
                raise TypeError("approval executor must return a result object")
            _json(result)
        except BaseException as error:
            self._record_failure(approval_id, claim_token, error)
            raise
        try:
            return self._complete(approval_id, owner_id, claim_token, result)
        except BaseException as error:
            self._record_failure(approval_id, claim_token, error, observed_result=result)
            raise

    def _complete(self, approval_id: str, owner_id: str, claim_token: str, result: dict[str, Any]) -> dict[str, Any]:
        with self._transaction() as connection:
            updated = connection.execute("UPDATE approvals SET status='approved',execution_state='succeeded',result_json=?, "
                                         "updated_at=?,claim_token=NULL WHERE approval_id=? AND status='executing' AND claim_token=?",
                                         (_json(result), time.time(), approval_id, claim_token)).rowcount
            if updated != 1:
                raise ValueError("approval execution claim is no longer active")
            return _item(self._find(connection, approval_id, owner_id))

    def _record_failure(self, approval_id: str, claim_token: str, error: BaseException,
                        observed_result: dict[str, Any] | None = None) -> None:
        precondition = isinstance(error, ApprovalPreconditionError)
        try:
            with self._transaction() as connection:
                connection.execute("UPDATE approvals SET status=?,execution_state=?,error_code=?,updated_at=?,claim_token=NULL, "
                                   "result_json=COALESCE(?,result_json) "
                                   "WHERE approval_id=? AND status='executing' AND claim_token=?",
                                   ("failed" if precondition else "unknown", "not_started" if precondition else "unknown",
                                    "precondition_failed" if precondition else "reconciliation_required", time.time(),
                                    _json(observed_result) if observed_result is not None else None, approval_id, claim_token))
        except Exception:
            # A storage outage may prevent even unknown from persisting. The prior
            # executing claim remains durable and still prevents automatic replay.
            pass

    def reject(self, approval_id: str, reviewer: str = "human", note: str = "", *, owner_id: str = "local") -> dict[str, Any]:
        owner_id = _identity(owner_id, "owner_id")
        conflict = None
        with self._transaction() as connection:
            self._expire(connection, owner_id)
            row = self._find(connection, approval_id, owner_id)
            if row["status"] == "rejected":
                return _item(row)
            if row["status"] != "pending":
                conflict = f"approval is {row['status']}; cannot reject"
            else:
                connection.execute("UPDATE approvals SET status='rejected',execution_state='not_started',reviewer=?,review_note=?, "
                                   "result_json=?,updated_at=? WHERE approval_id=? AND status='pending'",
                                   (reviewer, note, _json({"mode": "blocked_by_human", "sent": False}), time.time(), approval_id))
                result = _item(self._find(connection, approval_id, owner_id))
        if conflict is not None:
            raise ValueError(conflict)
        return result

    def reconcile(self, approval_id: str, outcome: str, evidence: str, reviewer: str = 'human', *,
                  owner_id: str = 'local', expected_payload_hash: str,
                  result: dict[str, Any] | None = None, execution_stopped: bool = False) -> dict[str, Any]:
        """Persist an owner's manual conclusion; never execute or reopen an action.

        not_executed requires explicit confirmation that no request is still in
        flight. An absent draft alone is not proof of that. An unresolved action
        stays unknown. The payload hash binds review to the displayed content.
        """
        owner_id, reviewer = _identity(owner_id, 'owner_id'), _identity(reviewer, 'reviewer')
        if outcome not in {'succeeded', 'not_executed', 'unresolved'}:
            raise ValueError('outcome must be succeeded, not_executed or unresolved')
        if not isinstance(evidence, str) or not evidence.strip() or len(evidence) > 4000:
            raise ValueError('reconciliation requires 1 to 4000 characters of evidence')
        if type(execution_stopped) is not bool:
            raise ValueError('execution_stopped must be a boolean')
        if outcome == 'not_executed' and not execution_stopped:
            raise ValueError('confirm that execution has stopped before recording not_executed')
        if result is not None and not isinstance(result, dict):
            raise ValueError('reconciliation result must be an object')
        if result is not None and result.get('sent') is True:
            raise ValueError('current providers only create drafts; a sent result is invalid')
        if outcome == 'succeeded' and (not result or not result.get('draft_id')):
            raise ValueError('confirmed success requires the observed remote draft_id')
        if outcome == 'succeeded':
            result = {**result, 'sent': False, 'reconciled': True}
        result_json = _json(result) if result is not None else None
        with self._transaction() as connection:
            row = self._find(connection, approval_id, owner_id)
            if not expected_payload_hash or row['payload_hash'] != expected_payload_hash:
                raise ValueError('approval payload does not match the reviewed content')
            if row['status'] not in {'executing', 'unknown'}:
                raise ValueError(f"approval is {row['status']}; only uncertain outcomes can be reconciled")
            observed = json.loads(row['result_json']) if row['result_json'] else {}
            if outcome == 'not_executed' and isinstance(observed, dict) and observed.get('draft_id'):
                raise ValueError('a remote draft was already observed; reconcile its success instead')
            status, execution_state, error_code = {
                'succeeded': ('approved', 'succeeded', None),
                'not_executed': ('failed', 'not_started', 'manually_confirmed_not_executed'),
                'unresolved': ('unknown', 'unknown', 'reconciliation_required'),
            }[outcome]
            now = time.time()
            connection.execute("""INSERT INTO approval_reconciliations
                (approval_id,owner_id,outcome,evidence,reviewer,execution_stopped,created_at,result_json)
                VALUES (?,?,?,?,?,?,?,?)""", (approval_id,owner_id,outcome,evidence.strip(),reviewer,
                                              int(execution_stopped),now,result_json))
            connection.execute("""UPDATE approvals SET status=?,execution_state=?,error_code=?,updated_at=?,
                claim_token=NULL,reviewer=?,review_note=?,result_json=COALESCE(?,result_json) WHERE approval_id=?""",
                (status,execution_state,error_code,now,reviewer,evidence.strip(),result_json,approval_id))
            return _item(self._find(connection, approval_id, owner_id))

    def reconciliation_history(self, approval_id: str, *, owner_id: str = 'local') -> list[dict[str, Any]]:
        owner_id = _identity(owner_id, 'owner_id')
        with self._transaction() as connection:
            self._find(connection, approval_id, owner_id)
            rows = connection.execute('SELECT * FROM approval_reconciliations WHERE approval_id=? AND owner_id=? ORDER BY id',
                                      (approval_id, owner_id)).fetchall()
            return [{**{key: row[key] for key in row.keys() if key != 'result_json'},
                     'created_at': _utc_now(row['created_at']),
                     'result': json.loads(row['result_json']) if row['result_json'] else None} for row in rows]


def _identity(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{name} must be a nonempty string of at most 256 characters")
    return value.strip()


def _session(value: Any) -> str | None:
    return None if value is None or value == "" else _identity(value, "session_id")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _payload(action_type: str, payload: Any) -> tuple[str, str]:
    _identity(action_type, "action_type")
    if not isinstance(payload, dict):
        raise ValueError("approval payload must be an object")
    text = _json(payload)
    return text, hashlib.sha256(_json({"action_type": action_type, "payload": payload}).encode("utf-8")).hexdigest()


def _valid_ttl(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
        raise ValueError("approval TTL must be a finite nonnegative number")
    return float(value)


def _parse_timestamp(value: Any, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()


def _item(row: sqlite3.Row) -> dict[str, Any]:
    item = {key: row[key] for key in row.keys() if key not in ("payload_json", "result_json", "claim_token") and row[key] is not None}
    item["payload"] = json.loads(row["payload_json"])
    if row["result_json"] is not None:
        item["result"] = json.loads(row["result_json"])
    for name in ("created_at", "updated_at", "expires_at"):
        item[name] = _utc_now(row[name])
    return item


def _summary(row: sqlite3.Row) -> dict[str, Any]:
    item = {key:row[key] for key in row.keys() if row[key] is not None}
    for key in ('created_at','updated_at','expires_at','redacted_at'):
        if key in item: item[key] = _utc_now(item[key])
    return item


def _utc_now(timestamp: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() if timestamp is None else timestamp))
