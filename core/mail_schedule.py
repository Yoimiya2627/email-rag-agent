"""Durable scheduling of local read-only IMAP jobs; never creates mail workers."""
from contextlib import contextmanager
import json
import re
import sqlite3
import threading
import time
from pathlib import Path
import uuid

from core.jobs import CapacityExceeded


_TERMINAL = {'succeeded', 'failed', 'incomplete', 'cancelled', 'interrupted'}
_MAX_FAILURES = 8
_SAFE_ERRORS = {'authentication_failed', 'imap_authentication_failed', 'credential_binding_changed',
                'account_sync_busy', 'secure_storage_failed', 'unsupported_secure_storage',
                'ImapSyncError', 'MailProviderError', 'TimeoutError', 'ConnectionError',
                'RuntimeError', 'ValueError', 'KeyError', 'worker_start_failed'}


def _identity(owner, account_id):
    if not isinstance(owner, str) or not owner.strip() or len(owner) > 256:
        raise ValueError('invalid_owner')
    if not isinstance(account_id, str) or not re.fullmatch('[0-9a-f]{32}', account_id):
        raise ValueError('invalid_account_id')


def _scope(enabled, folders, interval, maximum):
    if type(enabled) is not bool:
        raise ValueError('invalid_schedule_enabled')
    if (not isinstance(folders, list) or not 1 <= len(folders) <= 30
            or any(not isinstance(folder, str) or not folder or len(folder) > 1024
                   or any(ord(c) < 32 or ord(c) == 127 for c in folder) for folder in folders)
            or len(set(folders)) != len(folders)):
        raise ValueError('invalid_schedule_folders')
    if type(interval) is not int or not 60 <= interval <= 86400:
        raise ValueError('invalid_schedule_interval')
    if type(maximum) is not int or not 1 <= maximum <= 2000:
        raise ValueError('invalid_schedule_batch')


class MailScheduleStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS mail_schedules (
                owner TEXT NOT NULL, account_id TEXT NOT NULL, enabled INTEGER NOT NULL,
                folders TEXT NOT NULL, interval_seconds INTEGER NOT NULL, max_messages INTEGER NOT NULL,
                revision INTEGER NOT NULL, next_due REAL, failure_count INTEGER NOT NULL DEFAULT 0,
                last_error TEXT, last_job_id TEXT, operation_key TEXT, attempt_revision INTEGER,
                pending_request TEXT, updated REAL NOT NULL, PRIMARY KEY(owner,account_id))''')

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                db.execute('BEGIN IMMEDIATE')
                yield db
        finally:
            db.close()

    @staticmethod
    def _public(row):
        result = {key:row[key] for key in ('account_id', 'interval_seconds', 'max_messages',
                   'revision', 'next_due', 'failure_count', 'last_error', 'last_job_id', 'updated')}
        result['enabled'] = bool(row['enabled'])
        result['folders'] = json.loads(row['folders'])
        result['state'] = ('disabled' if not row['enabled'] else 'blocked' if row['failure_count'] >= _MAX_FAILURES
                           else 'active' if row['operation_key'] else 'waiting')
        return result

    def get(self, owner, account_id):
        _identity(owner, account_id)
        with self._db() as db:
            row = db.execute('SELECT * FROM mail_schedules WHERE owner=? AND account_id=?',
                             (owner, account_id)).fetchone()
        if row is None:
            return {'account_id':account_id, 'enabled':False, 'folders':['INBOX'], 'interval_seconds':300,
                    'max_messages':100, 'revision':0, 'next_due':None, 'failure_count':0,
                    'last_error':None, 'last_job_id':None, 'state':'disabled', 'updated':None}
        return self._public(row)

    def upsert(self, owner, account_id, *, enabled, folders, interval_seconds=300, max_messages=100):
        _identity(owner, account_id)
        _scope(enabled, folders, interval_seconds, max_messages)
        with self._db() as db:
            db.execute('''INSERT INTO mail_schedules
                (owner,account_id,enabled,folders,interval_seconds,max_messages,revision,next_due,updated)
                VALUES(?,?,?,?,?,?,1,?,?) ON CONFLICT(owner,account_id) DO UPDATE SET
                enabled=excluded.enabled,folders=excluded.folders,interval_seconds=excluded.interval_seconds,
                max_messages=excluded.max_messages,revision=mail_schedules.revision+1,
                next_due=excluded.next_due,failure_count=0,last_error=NULL,updated=excluded.updated''',
                (owner, account_id, int(enabled), json.dumps(folders), interval_seconds,
                 max_messages, 0 if enabled else None, time.time()))
            return self._public(db.execute('SELECT * FROM mail_schedules WHERE owner=? AND account_id=?',
                                           (owner, account_id)).fetchone())

    def all_due(self, now):
        """Internal owner/account identifiers, including old attempts to reconcile."""
        with self._db() as db:
            return [(row['owner'], row['account_id']) for row in db.execute('''SELECT owner,account_id
                FROM mail_schedules WHERE operation_key IS NOT NULL OR
                (enabled=1 AND next_due IS NOT NULL AND next_due<=?) ORDER BY owner,account_id''', (now,))]


class MailScheduler:
    def __init__(self, store, manager_factory, request_factory, *, clock=time.time, poll_seconds=2):
        if poll_seconds <= 0:
            raise ValueError('invalid_schedule_poll')
        self.store, self.manager_factory, self.request_factory = store, manager_factory, request_factory
        self.clock, self.poll_seconds = clock, poll_seconds
        self._stop = threading.Event()
        self._tick_lock = threading.Lock()
        self._lifecycle = threading.Lock()
        self._thread = None

    def start(self):
        with self._lifecycle:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True, name='local-mail-scheduler')
            self._thread.start()

    def stop(self, timeout=5):
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
        return thread is None or not thread.is_alive()

    def _loop(self):
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:
                # A transient database/manager failure must not kill the timer.
                pass
            self._stop.wait(self.poll_seconds)

    @staticmethod
    def _row(db, owner, account_id):
        return db.execute('SELECT * FROM mail_schedules WHERE owner=? AND account_id=?',
                          (owner, account_id)).fetchone()

    def _clear(self, db, row):
        db.execute('''UPDATE mail_schedules SET operation_key=NULL,attempt_revision=NULL,pending_request=NULL
                      WHERE owner=? AND account_id=?''', (row['owner'], row['account_id']))

    def _fail(self, db, row, code, *, clear=True):
        count = row['failure_count'] + 1
        next_due = None if count >= _MAX_FAILURES else self.clock() + min(3600, row['interval_seconds'] * 2 ** (count - 1))
        safe = code if code in _SAFE_ERRORS or code in {'partial_sync', 'job_missing', 'invalid_job_binding'} else 'schedule_operation_failed'
        db.execute('''UPDATE mail_schedules SET failure_count=?,last_error=?,next_due=?,updated=?
                      WHERE owner=? AND account_id=?''',
                   (count, safe, next_due, self.clock(), row['owner'], row['account_id']))
        if clear:
            self._clear(db, row)

    @staticmethod
    def _request(raw, row):
        # A request factory cannot accidentally persist credentials or mail text.
        if not isinstance(raw, dict) or set(raw) - {'account_id', 'credential_version', 'folders', 'max_messages', 'retry_failed'}:
            raise ValueError('invalid_schedule_request')
        if (raw.get('account_id') != row['account_id'] or raw.get('folders') != json.loads(row['folders'])
                or raw.get('max_messages') != row['max_messages'] or type(raw.get('credential_version')) is not int
                or raw['credential_version'] < 1 or raw.get('retry_failed', False) is not False):
            raise ValueError('invalid_schedule_request')
        return dict(raw, retry_failed=False)

    def tick(self):
        """One bounded reconciliation pass; usable without a background thread."""
        with self._tick_lock:
            if self._stop.is_set():
                return
            for owner, account_id in self.store.all_due(self.clock()):
                if self._stop.is_set():
                    break
                try:
                    self._process(owner, account_id)
                except Exception:
                    # Do not expose arbitrary provider messages or email content.
                    # Failed persistence remains recoverable through operation_key.
                    continue

    def _process(self, owner, account_id):
        # Persist an immutable intent before any submission. A process crash at
        # either side of submit is recovered by the JobStore operation key.
        with self.store._db() as db:
            row = self._row(db, owner, account_id)
            if not row:
                return
            if not row['operation_key']:
                if not row['enabled'] or row['next_due'] is None or row['next_due'] > self.clock():
                    return
                try:
                    request = self._request(self.request_factory(owner, account_id,
                                            json.loads(row['folders']), row['max_messages']), row)
                except Exception as exc:
                    self._fail(db, row, type(exc).__name__)
                    return
                db.execute('''UPDATE mail_schedules SET operation_key=?,attempt_revision=revision,pending_request=?
                              WHERE owner=? AND account_id=?''',
                           ('imap-schedule-'+uuid.uuid4().hex, json.dumps(request), owner, account_id))
        if self._stop.is_set():
            return
        with self.store._db() as db:
            row = self._row(db, owner, account_id)
            if not row or not row['operation_key']:
                return
            stale = not row['enabled'] or row['attempt_revision'] != row['revision']
            if not stale and row['failure_count'] and (row['next_due'] is None or row['next_due'] > self.clock()):
                return
            try:
                manager = self.manager_factory()
                try:
                    job = manager.store.find_operation(owner, row['operation_key'])
                except KeyError:
                    job = None
                if job is None:
                    if stale:
                        self._clear(db, row)
                        return
                    if row['next_due'] is None or row['next_due'] > self.clock() or self._stop.is_set():
                        return
                    job = manager.submit(owner, 'imap_sync', json.loads(row['pending_request']), row['operation_key'])
                if job.get('kind') != 'imap_sync':
                    self._fail(db, row, 'invalid_job_binding')
                    return
                db.execute('UPDATE mail_schedules SET last_job_id=? WHERE owner=? AND account_id=?',
                           (job['id'], owner, account_id))
                if job['status'] not in _TERMINAL:
                    return
                if stale:
                    self._clear(db, row)
                    return
                result = job.get('result') or {}
                failed = (result.get('last_run') or {}).get('failed', 0)
                if job['status'] != 'succeeded' or failed:
                    self._fail(db, row, 'partial_sync' if failed else job.get('error_code', 'schedule_operation_failed'))
                    return
                delay = 5 if result.get('remaining_eligible', 0) > 0 else row['interval_seconds']
                db.execute('''UPDATE mail_schedules SET next_due=?,failure_count=0,last_error=NULL,updated=?
                              WHERE owner=? AND account_id=?''', (self.clock()+delay, self.clock(), owner, account_id))
                self._clear(db, row)
            except CapacityExceeded:
                if not stale:
                    db.execute('UPDATE mail_schedules SET next_due=? WHERE owner=? AND account_id=?',
                               (self.clock()+max(5, self.poll_seconds), owner, account_id))
            except Exception as exc:
                if not stale:
                    # Keep the intent: submit may have succeeded before raising.
                    self._fail(db, row, type(exc).__name__, clear=False)
