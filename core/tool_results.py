"""Immutable, scope-bound tool snapshots. No model-selected paths or identities."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


class ToolResultUnavailable(ValueError):
    pass


def sanitize_tool_result(value):
    """Drop transport/debug/credential fields before both display and persistence.

    Mail body and source text are authorized task data. We do not claim to infer
    secrets from arbitrary prose or rewrite source content/hashes.
    """
    denied = {'token', 'access_token', 'refresh_token', 'api_key', 'apikey', 'authorization',
              'cookie', 'set_cookie', 'request_headers', 'response_headers', 'debug',
              'debug_info', 'raw_response', 'raw_request', 'client_secret', 'password',
              'credentials', 'credential', 'secret', 'secrets', 'private_key'}
    if isinstance(value, dict):
        return {key: sanitize_tool_result(child) for key, child in value.items()
                if isinstance(key, str) and key.lower().replace('-', '_') not in denied
                and not any(marker in key.lower() for marker in ('password', 'credential', 'api_key', 'client_secret'))}
    if isinstance(value, list):
        return [sanitize_tool_result(child) for child in value]
    return value


class ToolResultStore:
    """Snapshots are immutable within retention; expired reference IDs never revive.

    Expired payloads are reclaimed on new writes. Permanent operation replay
    protection belongs to backend run IDs, checkpoint references and operation
    records, not an unbounded ledger of historical snapshot call IDs.
    """
    def __init__(self, path, *, max_object_bytes=512_000, max_run_bytes=4_000_000,
                 max_total_bytes=64_000_000, max_objects=4096, retention_seconds=604800):
        self.path = Path(path)
        self.max_object_bytes = max_object_bytes
        self.max_run_bytes = max_run_bytes
        self.max_total_bytes = max_total_bytes
        self.max_objects = max_objects
        self.retention_seconds = retention_seconds
        if any(type(v) is not int or v < 1 for v in (max_object_bytes, max_run_bytes, max_total_bytes, max_objects, retention_seconds)):
            raise ValueError('Invalid tool result limits')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS tool_results (
                id TEXT PRIMARY KEY, owner TEXT NOT NULL, session TEXT NOT NULL,
                run TEXT NOT NULL, call_id TEXT NOT NULL, epoch INTEGER NOT NULL,
                tool TEXT NOT NULL, argument_hash TEXT NOT NULL, content_hash TEXT NOT NULL,
                payload TEXT NOT NULL, bytes INTEGER NOT NULL, expires REAL NOT NULL,
                UNIQUE(owner, session, run, call_id, epoch))''')
            db.execute('CREATE INDEX IF NOT EXISTS tool_results_scope ON tool_results(owner,session,run)')

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _scope(owner, session, run, epoch):
        if any(not isinstance(v, str) or not v or len(v) > 256 for v in (owner, session, run)) or type(epoch) is not int or epoch < 0:
            raise ToolResultUnavailable('trusted_scope_unavailable')

    def put(self, *, owner, session, run, call_id, epoch, tool, argument_hash, value):
        self._scope(owner, session, run, epoch)
        if not isinstance(call_id, str) or not 1 <= len(call_id) <= 256:
            raise ToolResultUnavailable('invalid_call_id')
        if not isinstance(tool, str) or not 1 <= len(tool) <= 128 or not isinstance(argument_hash, str) or not re.fullmatch('[a-f0-9]{64}', argument_hash):
            raise ToolResultUnavailable('invalid_tool_metadata')
        payload = json.dumps(sanitize_tool_result(value), ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
        size = len(payload.encode('utf-8'))
        if size > self.max_object_bytes:
            raise ToolResultUnavailable('object_byte_limit')
        digest = hashlib.sha256(payload.encode('utf-8')).hexdigest()
        with self._db() as db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute('SELECT * FROM tool_results WHERE owner=? AND session=? AND run=? AND call_id=? AND epoch=?', (owner, session, run, call_id, epoch)).fetchone()
            if old:
                if old['content_hash'] != digest or old['argument_hash'] != argument_hash or old['tool'] != tool or old['expires'] <= time.time():
                    raise ToolResultUnavailable('immutable_call_conflict')
                return self._ref(old)
            # Check the caller's existing immutable slot before reclaiming any
            # expired objects. A same-call retry cannot extend its retention.
            # Purged IDs stay missing; checkpoint validation therefore rejects
            # recovery even if a trusted caller later writes a fresh snapshot.
            db.execute('DELETE FROM tool_results WHERE expires<=?', (time.time(),))
            total = db.execute('SELECT COALESCE(SUM(bytes),0), COUNT(*) FROM tool_results').fetchone()
            run_size = db.execute('SELECT COALESCE(SUM(bytes),0) FROM tool_results WHERE owner=? AND session=? AND run=?', (owner, session, run)).fetchone()[0]
            if total[0] + size > self.max_total_bytes or total[1] >= self.max_objects or run_size + size > self.max_run_bytes:
                raise ToolResultUnavailable('store_capacity')
            result_id = uuid.uuid4().hex
            db.execute('INSERT INTO tool_results VALUES(?,?,?,?,?,?,?,?,?,?,?,?)', (result_id, owner, session, run, call_id, epoch, tool, argument_hash, digest, payload, size, time.time() + self.retention_seconds))
            return self._ref(db.execute('SELECT * FROM tool_results WHERE id=?', (result_id,)).fetchone())

    @staticmethod
    def _ref(row):
        return {'result_id': row['id'], 'content_hash': row['content_hash'], 'call_id': row['call_id'],
                'tool': row['tool'], 'argument_hash': row['argument_hash'], 'bytes': row['bytes'],
                'epoch': row['epoch'], 'material_type': 'historical_tool_result'}

    def load(self, result_id, *, owner, session, run, epoch, content_hash=None):
        self._scope(owner, session, run, epoch)
        if not isinstance(result_id, str) or not re.fullmatch('[a-f0-9]{32}', result_id):
            raise ToolResultUnavailable('invalid_result_id')
        with self._db() as db:
            row = db.execute('SELECT * FROM tool_results WHERE id=? AND owner=? AND session=? AND run=? AND epoch=?', (result_id, owner, session, run, epoch)).fetchone()
        if row is None or row['expires'] <= time.time():
            raise ToolResultUnavailable('result_missing_or_expired')
        digest = hashlib.sha256(row['payload'].encode('utf-8')).hexdigest()
        if digest != row['content_hash'] or (content_hash is not None and content_hash != digest):
            raise ToolResultUnavailable('result_hash_mismatch')
        size = len(row['payload'].encode('utf-8'))
        if size != row['bytes'] or size > self.max_object_bytes:
            raise ToolResultUnavailable('result_size_mismatch')
        return row['payload'], self._ref(row)

    def page(self, result_id, *, start=0, limit=1200, **scope):
        if type(start) is not int or start < 0 or type(limit) is not int or not 1 <= limit <= 4000:
            raise ToolResultUnavailable('invalid_page')
        payload, ref = self.load(result_id, **scope)
        if start > len(payload):
            raise ToolResultUnavailable('invalid_page')
        end = min(len(payload), start + limit)
        text = payload[start:end]
        return {'result_ref': ref, 'material_type': 'historical_tool_result', 'text': text,
                'page_start': start, 'page_end': end, 'total_chars': len(payload),
                'page_hash': hashlib.sha256(text.encode('utf-8')).hexdigest(),
                'next_start': end if end < len(payload) else None, 'has_more': end < len(payload),
                'coverage': 'stored_tool_snapshot_only; not current email evidence; reread get_email to cite'}

    def delete_session(self, owner, session):
        with self._db() as db:
            return db.execute('DELETE FROM tool_results WHERE owner=? AND session=?', (owner, session)).rowcount

    def prune(self, *, now=None):
        with self._db() as db:
            return db.execute('DELETE FROM tool_results WHERE expires<=?', (time.time() if now is None else now,)).rowcount


def trusted_result_scope(context):
    if context is None or context.session_repository is None:
        raise ToolResultUnavailable('trusted_scope_unavailable')
    context.session_repository.validate_context_epoch(context.owner_id, context.session_id, context.context_epoch)
    return dict(owner=context.owner_id, session=context.session_id, run=context.run_id, epoch=context.context_epoch)
