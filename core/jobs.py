"""Durable local jobs with explicit recovery and bounded, in-process workers.

The database stores private requests/checkpoints; progress contains counters only.
An interrupted job is never automatically replayed. This is a single-instance
service, not a distributed queue.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


class CapacityExceeded(RuntimeError):
    pass


class JobConflict(ValueError):
    pass


class AdmissionGate:
    def __init__(self, limit=4):
        self.limit = max(1, int(limit))
        self._slots = threading.BoundedSemaphore(self.limit)
        self._lock = threading.Lock()
        self.active = 0

    def acquire(self):
        if not self._slots.acquire(blocking=False):
            raise CapacityExceeded('All execution slots are busy; retry after an active task finishes')
        with self._lock:
            self.active += 1

    def release(self):
        with self._lock:
            self.active -= 1
        self._slots.release()

    @contextmanager
    def slot(self):
        self.acquire()
        try:
            yield
        finally:
            self.release()


def _encode(value, limit=2_000_000):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    if len(text.encode('utf-8')) > limit:
        raise ValueError('Persisted job data exceeds its byte budget')
    return text


class JobStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, owner TEXT NOT NULL, operation_key TEXT NOT NULL,
                kind TEXT NOT NULL, request TEXT NOT NULL, request_hash TEXT NOT NULL,
                status TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL,
                progress TEXT NOT NULL DEFAULT '{}', checkpoint TEXT,
                result TEXT, error_code TEXT, cancel_requested INTEGER NOT NULL DEFAULT 0,
                attempt INTEGER NOT NULL DEFAULT 1, UNIQUE(owner, operation_key))''')
            db.execute('CREATE INDEX IF NOT EXISTS jobs_owner_updated ON jobs(owner, updated DESC, id)')

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA busy_timeout=5000')
        try:
            with db:
                yield db
        finally:
            db.close()

    def create(self, owner, kind, request, operation_key=None):
        if kind not in {'agent', 'index', 'imap_sync'} or not owner or len(owner) > 256:
            raise ValueError('Invalid job owner or kind')
        encoded = _encode(request, 100_000)
        digest = hashlib.sha256((kind + ':' + encoded).encode()).hexdigest()
        key = operation_key or str(uuid.uuid4())
        if not isinstance(key, str) or not 1 <= len(key) <= 128:
            raise ValueError('Invalid operation key')
        now, job_id = time.time(), str(uuid.uuid4())
        with self._db() as db:
            db.execute('BEGIN IMMEDIATE')
            old = db.execute('SELECT id, request_hash FROM jobs WHERE owner=? AND operation_key=?', (owner, key)).fetchone()
            if old:
                if old['request_hash'] != digest:
                    raise JobConflict('The operation key is already bound to a different request')
                return self.get(owner, old['id']), False
            db.execute('INSERT INTO jobs(id,owner,operation_key,kind,request,request_hash,status,created,updated) VALUES(?,?,?,?,?,?,?,?,?)',
                       (job_id, owner, key, kind, encoded, digest, 'queued', now, now))
        return self.get(owner, job_id), True

    def get(self, owner, job_id, *, private=False):
        with self._db() as db:
            row = db.execute('SELECT * FROM jobs WHERE owner=? AND id=?', (owner, job_id)).fetchone()
        if row is None:
            raise KeyError('Job not found')
        result = dict(row)
        for key in ('request', 'progress', 'checkpoint', 'result'):
            result[key] = json.loads(result[key]) if result[key] is not None else None
        result['cancel_requested'] = bool(result['cancel_requested'])
        recoverable = result['status'] in {'interrupted','cancelled','failed','incomplete'}
        reason = ('session_invalidated' if result.get('error_code') == 'session_invalidated' else
                  'recovery_budget_exhausted' if result['attempt'] >= 10 else
                  'unsafe_checkpoint' if not (result['checkpoint'] or {}).get('safe') else None)
        result['resumable'] = recoverable and reason is None
        result['resume_block_reason'] = reason if recoverable else None
        if not private:
            for key in ('request','checkpoint','request_hash','owner'):
                result.pop(key, None)
        return result

    def find_operation(self,owner,operation_key):
        with self._db() as db:
            row = db.execute('SELECT id FROM jobs WHERE owner=? AND operation_key=?',(owner,operation_key)).fetchone()
        if row is None:
            raise KeyError('Operation not found')
        return self.get(owner,row['id'])

    def list(self, owner, *, limit=25, offset=0):
        if not 1 <= limit <= 100 or offset < 0:
            raise ValueError('Invalid page')
        with self._db() as db:
            rows = db.execute('SELECT id FROM jobs WHERE owner=? ORDER BY updated DESC,id LIMIT ? OFFSET ?', (owner, limit + 1, offset)).fetchall()
        return {'jobs':[self.get(owner, row['id']) for row in rows[:limit]],
                'next_offset':offset + limit if len(rows) > limit else None}

    def session_jobs(self, owner, session_id, *, limit=20):
        if type(limit) is not int or not 1 <= limit <= 100 or not isinstance(session_id, str) or not session_id:
            raise ValueError('Invalid session jobs page')
        with self._db() as db:
            rows = db.execute("SELECT id FROM jobs WHERE owner=? AND json_extract(request,'$.session_id')=? ORDER BY updated DESC,id LIMIT ?",
                              (owner, session_id, limit)).fetchall()
        return [self.get(owner, row['id']) for row in rows]

    def claim(self, owner, job_id):
        with self._db() as db:
            changed = db.execute("UPDATE jobs SET status='running',updated=? WHERE id=? AND owner=? AND status='queued' AND cancel_requested=0", (time.time(), job_id, owner)).rowcount
        return bool(changed)

    def progress(self, owner, job_id, stage, **metrics):
        # Metrics are bounded primitive counters, never arbitrary documents.
        if not isinstance(stage, str) or len(stage) > 60 or len(metrics) > 20:
            raise ValueError('Invalid progress')
        if any(not isinstance(k,str) or len(k)>50 or not isinstance(v,(int,float,bool,type(None))) for k,v in metrics.items()):
            raise ValueError('Progress metrics must be numeric')
        with self._db() as db:
            db.execute("UPDATE jobs SET progress=?,updated=? WHERE id=? AND owner=? AND status='running'", (_encode({'stage':stage, **metrics}, 4000), time.time(), job_id, owner))

    def checkpoint(self, owner, job_id, value):
        if not isinstance(value,dict) or type(value.get('safe')) is not bool:
            raise ValueError('Checkpoint needs an explicit safety state')
        with self._db() as db:
            if db.execute("UPDATE jobs SET checkpoint=?,updated=? WHERE id=? AND owner=? AND status='running'", (_encode(value), time.time(), job_id, owner)).rowcount != 1:
                raise JobConflict('Job no longer owns its checkpoint')

    def finish(self, owner, job_id, status, *, result=None, error_code=None):
        if status not in {'succeeded','incomplete','cancelled','failed'}:
            raise ValueError('Invalid terminal status')
        encoded = _encode(result) if result is not None else None
        completed_checkpoint = (_encode({'safe': True, 'kind': 'completed_result', 'result': result})
                                if status == 'succeeded' else None)
        with self._db() as db:
            # Publish the completed result and its recovery boundary together.
            # A separate checkpoint write could fail after the runner committed
            # its transcript and leave an older planning checkpoint replayable.
            db.execute("UPDATE jobs SET status=?,result=?,checkpoint=COALESCE(?,checkpoint),error_code=?,updated=? WHERE id=? AND owner=? AND status='running'", (status, encoded, completed_checkpoint, error_code, time.time(), job_id, owner))

    def cancel(self, owner, job_id):
        self.get(owner, job_id)
        with self._db() as db:
            db.execute("UPDATE jobs SET cancel_requested=1,status=CASE WHEN status='queued' THEN 'cancelled' ELSE status END,updated=? WHERE id=? AND owner=? AND status IN ('queued','running')", (time.time(), job_id, owner))
        return self.get(owner, job_id)

    def resume(self, owner, job_id):
        with self._db() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT status,checkpoint,attempt FROM jobs WHERE id=? AND owner=?', (job_id,owner)).fetchone()
            if row is None:
                raise KeyError('Job not found')
            checkpoint = json.loads(row['checkpoint'] or '{}')
            if row['status'] not in {'interrupted','cancelled','failed','incomplete'} or not checkpoint.get('safe') or row['attempt'] >= 10:
                raise JobConflict('No safe checkpoint is available, or the recovery budget is exhausted')
            db.execute("UPDATE jobs SET status='queued',cancel_requested=0,error_code=NULL,result=NULL,attempt=attempt+1,updated=? WHERE id=? AND owner=?", (time.time(),job_id,owner))
        return self.get(owner,job_id)

    def recover_interrupted(self):
        with self._db() as db:
            # Only call once while assembling a single application instance.
            db.execute("UPDATE jobs SET status='interrupted',error_code='process_interrupted',updated=? WHERE status IN ('running','queued')", (time.time(),))

    def invalidate_session(self, owner, session_id):
        """Redact derived private data, retaining only identity/operation tombstones.

        Session repository invalidation must happen first. A running worker's
        late checkpoint/finish cannot match status='running' after this commits.
        """
        if not isinstance(session_id, str) or not session_id:
            raise ValueError('Invalid session identity')
        with self._db() as db:
            db.execute('BEGIN IMMEDIATE')
            rows = db.execute("SELECT id FROM jobs WHERE owner=? AND json_extract(request,'$.session_id')=?", (owner, session_id)).fetchall()
            ids = [row['id'] for row in rows]
            db.execute("""UPDATE jobs SET status='cancelled',cancel_requested=1,
                request=?,checkpoint=NULL,result=NULL,progress='{}',error_code='session_invalidated',updated=?
                WHERE owner=? AND json_extract(request,'$.session_id')=?""",
                (_encode({'session_id': session_id}), time.time(), owner, session_id))
        return ids


class JobManager:
    def __init__(self, store, runner, *, gate=None, max_workers=2):
        self.store, self.runner = store, runner
        self.gate = gate or AdmissionGate()
        self.workers = AdmissionGate(max_workers)
        self._lock = threading.Lock()
        self._events = {}
        self.store.recover_interrupted()

    def submit(self, owner, kind, request, operation_key=None):
        with self._lock:
            if operation_key is not None:
                try:
                    previous = self.store.find_operation(owner,operation_key)
                except KeyError:
                    previous = None
                if previous is not None:
                    # create checks the immutable request hash without a new row.
                    return self.store.create(owner,kind,request,operation_key)[0]
            with self._reserve():
                job, created = self.store.create(owner,kind,request,operation_key)
                if not created:
                    return job
                self._start(owner,job['id'])
        return self.store.get(owner,job['id'])

    @contextmanager
    def _reserve(self):
        self.workers.acquire()
        try:
            self.gate.acquire()
        except BaseException:
            self.workers.release()
            raise
        self._reservation_transferred = False
        try:
            yield
        finally:
            if not self._reservation_transferred:
                self.gate.release()
                self.workers.release()

    def _start(self,owner,job_id):
        event = threading.Event()
        self._events[job_id] = event
        worker = threading.Thread(target=self._run,args=(owner,job_id,event),daemon=True,name='email-agent-job')
        try:
            worker.start()
        except BaseException:
            self._events.pop(job_id,None)
            with self.store._db() as db:
                db.execute("UPDATE jobs SET status='failed',error_code='worker_start_failed',updated=? WHERE id=? AND owner=? AND status='queued'",(time.time(),job_id,owner))
            raise
        self._reservation_transferred = True

    def _run(self,owner,job_id,event):
        try:
            if not self.store.claim(owner,job_id):
                return
            job = self.store.get(owner,job_id,private=True)
            if (job['checkpoint'] or {}).get('kind')=='completed_result':
                result = job['checkpoint']['result']
            else:
                result = self.runner(job,event,
                    lambda stage, **metrics:self.store.progress(owner,job_id,stage,**metrics),
                    lambda data:self.store.checkpoint(owner,job_id,data))
            metadata = (result or {}).get('metadata') or {}
            # Once a complete result is returned, its transcript/side effects
            # have already been committed by the runner. A late cancellation
            # cannot move that completed work back to a replayable checkpoint.
            complete = metadata.get('status') in {'success','approval_required'} and metadata.get('completion_status','complete')=='complete'
            status = ('succeeded' if complete else 'cancelled' if event.is_set() else
                      'succeeded' if metadata.get('status','success') in {'success','approval_required'} else 'incomplete')
            self.store.finish(owner,job_id,status,result=result)
        except Exception as exc:
            # Store only a type code. The API maps it to a safe recovery message.
            self.store.finish(owner,job_id,'cancelled' if event.is_set() else 'failed',error_code=type(exc).__name__)
        finally:
            with self._lock:
                self._events.pop(job_id,None)
            self.gate.release()
            self.workers.release()

    def cancel(self,owner,job_id):
        with self._lock:
            result = self.store.cancel(owner,job_id)
            event = self._events.get(job_id)
            if event:
                event.set()
        return result

    def invalidate_session(self, owner, session_id):
        with self._lock:
            ids = self.store.invalidate_session(owner, session_id)
            for job_id in ids:
                event = self._events.get(job_id)
                if event is not None:
                    event.set()
        return ids

    def resume(self,owner,job_id):
        with self._lock:
            with self._reserve():
                self.store.resume(owner,job_id)
                self._start(owner,job_id)
        return self.store.get(owner,job_id)

    def stop(self):
        with self._lock:
            for event in self._events.values():
                event.set()

    def wait_idle(self,timeout=5):
        deadline = time.monotonic()+timeout
        while time.monotonic()<deadline:
            with self._lock:
                if not self._events:
                    return True
            time.sleep(.02)
        return False
