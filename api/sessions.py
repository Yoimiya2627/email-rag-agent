"""Bounded process-local sessions; one active turn per owner/session pair."""
from contextlib import contextmanager
from dataclasses import dataclass, field
import threading
import time

from core.memory import ConversationMemory
from core.session_repository import SessionRepository, excerpt_summary


class SessionBusyError(RuntimeError):
    pass


@dataclass
class _Session:
    memory: ConversationMemory = field(default_factory=ConversationMemory)
    touched: float = field(default_factory=time.monotonic)
    active: bool = False
    revision: int = 0
    invalidated: bool = False


class SessionStore:
    def __init__(self, max_sessions=1000, ttl_seconds=3600, *, path=None, max_history_turns=5,
                 max_cached_bytes=16*1024*1024):
        self.max_sessions = max(1, int(max_sessions))
        self.ttl_seconds = max(1, int(ttl_seconds))
        self._entries = {}
        self._lock = threading.Lock()
        self.max_history_turns = max_history_turns
        ConversationMemory(max_history_turns)  # Validate before opening storage.
        if type(max_cached_bytes) is not int or max_cached_bytes < 1:
            raise ValueError("session cache byte budget must be positive")
        self.max_cached_bytes = max_cached_bytes
        self.repository = SessionRepository(path) if path is not None else None

    def _cache_bytes(self):
        return sum(entry.memory.cache_size()+256 for entry in self._entries.values())

    @property
    def cached_bytes(self):
        with self._lock:
            return self._cache_bytes()

    def _evict_idle(self, exclude=()):
        while self._cache_bytes() > self.max_cached_bytes:
            idle = [(entry.touched, key) for key, entry in self._entries.items()
                    if not entry.active and key not in exclude]
            if not idle:
                return False
            del self._entries[min(idle)[1]]
        return True

    def _load(self, owner_id, session_id):
        entry = _Session(memory=ConversationMemory(self.max_history_turns))
        entry.memory.cache_byte_limit = self.max_cached_bytes
        if self.repository:
            entry.revision = self.repository.revision(owner_id, session_id)
            entry.memory.load_context(self.repository.recent_context(owner_id, session_id, self.max_history_turns))
        if entry.memory.cache_size()+256 > self.max_cached_bytes:
            raise SessionBusyError("Session history exceeds cache byte budget; reduce the history window or increase its budget")
        return entry

    @contextmanager
    def turn(self, owner_id, session_id):
        key = (owner_id, session_id)
        with self._lock:
            now = time.monotonic()
            expired = [k for k, v in self._entries.items() if not v.active and now-v.touched >= self.ttl_seconds]
            for old in expired:
                del self._entries[old]
            entry = self._entries.get(key)
            if entry is None:
                if len(self._entries) >= self.max_sessions:
                    idle = [(v.touched, k) for k, v in self._entries.items() if not v.active]
                    if not idle:
                        raise SessionBusyError('Session capacity reached')
                    del self._entries[min(idle)[1]]
                entry = self._entries[key] = self._load(owner_id, session_id)
            if entry.active:
                raise SessionBusyError('A turn is already running for this session')
            if self.repository and self.repository.revision(owner_id, session_id) != entry.revision:
                entry = self._entries[key] = self._load(owner_id, session_id)
            if not self._evict_idle(exclude={key}):
                self._entries.pop(key, None)
                raise SessionBusyError('Active session histories exhaust the cache byte budget; retry when a turn finishes')
            entry.active = True
            snapshot = entry.memory.snapshot()
        try:
            yield entry.memory
        except BaseException:
            entry.memory.restore(snapshot)
            raise
        else:
            try:
                with self._lock:
                    if entry.invalidated:
                        from core.session_repository import SessionConflictError
                        raise SessionConflictError('session deleted during active turn')
                    if self.repository:
                        entry.revision = self.repository.append_turns(owner_id, session_id,
                            entry.memory.pending_turns(), expected_revision=entry.revision)
                        entry.memory.committed(persistent=True, base_snapshot=snapshot)
                    else:
                        entry.memory.committed()
            except BaseException:
                entry.memory.restore(snapshot)
                entry.revision = -1  # Force reload after any uncertain persistence result.
                raise
        finally:
            with self._lock:
                entry.active = False
                if entry.invalidated and self._entries.get(key) is entry:
                    self._entries.pop(key,None)
                entry.touched = time.monotonic()
                self._evict_idle()

    def clear(self, owner_id, session_id, *, invalidate_active=False):
        with self._lock:
            entry = self._entries.get((owner_id, session_id))
            if entry and entry.active and not invalidate_active:
                raise SessionBusyError('Cannot clear a running session')
            if self.repository:
                self.repository.delete(owner_id, session_id)
            if entry and entry.active:
                entry.invalidated=True
                entry.revision=-1
            else:
                self._entries.pop((owner_id, session_id), None)
            return self.repository.context_epoch(owner_id,session_id) if self.repository else 0

    delete = clear

    def record_result(self, owner_id, session_id, query, answer, metadata=None, **kwargs):
        """Record a completed/failed result after its execution lock is released."""
        with self.turn(owner_id, session_id) as memory:
            return memory.append_turn(query, answer, metadata, **kwargs)

    def list_sessions(self, owner_id, *, limit=100):
        return self.list_sessions_page(owner_id, limit=limit)["sessions"]

    def list_sessions_page(self, owner_id, *, limit=50, offset=0):
        if self.repository:
            return self.repository.list_sessions_page(owner_id, limit=limit, offset=offset)
        if (type(limit) is not int or not 1 <= limit <= 200
                or type(offset) is not int or not 0 <= offset <= 1000000):
            raise ValueError("invalid session listing pagination")
        with self._lock:
            rows = [{"session_id": sid, "turn_count": len(entry.memory.transcript()),
                     "updated_at": entry.touched, "revision": 0}
                    for (owner, sid), entry in self._entries.items() if owner == owner_id and not entry.invalidated]
        rows = sorted(rows, key=lambda row: (-row["updated_at"], row["session_id"]))
        return {"sessions": rows[offset:offset + limit],
                "next_offset": offset + limit if len(rows) > offset + limit else None}

    def history(self, owner_id, session_id, *, after=0, limit=100):
        if self.repository:
            return self.repository.history(owner_id, session_id, after=after, limit=limit)
        with self._lock:
            entry = self._entries.get((owner_id, session_id))
            rows = entry.memory.transcript() if entry and not entry.invalidated else []
        rows = [row for row in rows if row["seq"] > after]
        page = rows[:limit]
        return {"session_id": session_id, "turns": page, "has_more": len(rows) > limit,
                "next_after": page[-1]["seq"] if page else after}

    def search_history(self, owner_id, session_id, query, *, limit=30, **kwargs):
        if self.repository:
            return self.repository.search_history(owner_id, session_id, query, limit=limit, **kwargs)
        with self._lock:
            entry = self._entries.get((owner_id, session_id))
            rows = entry.memory.transcript() if entry and not entry.invalidated else []
        return [row for row in reversed(rows) if query.casefold() in (row["query"]+"\n"+row["answer"]).casefold()][:limit]

    def get_turn(self, owner_id, session_id, turn_id):
        if self.repository:
            return self.repository.get_turn(owner_id, session_id, turn_id)
        with self._lock:
            entry = self._entries.get((owner_id, session_id))
            rows = entry.memory.transcript() if entry and not entry.invalidated else []
        for row in rows:
            if row["turn_id"] == turn_id:
                return row
        raise KeyError("transcript turn not found")

    def summary(self, owner_id, session_id, *, after=0, limit=20, max_chars=300):
        page = self.history(owner_id, session_id, after=after, limit=limit)
        return {**excerpt_summary(page["turns"], max_chars=max_chars),
                "has_more": page["has_more"], "next_after": page["next_after"]}

    def _persistent(self):
        if self.repository is None:
            raise ValueError("task facts and versioned evidence require a persistent session store")
        return self.repository

    def set_task_fact(self, owner_id, session_id, key, value, **kwargs):
        return self._persistent().set_task_fact(owner_id, session_id, key, value, **kwargs)

    def task_facts(self, owner_id, session_id, *, include_history=False, **kwargs):
        return self._persistent().task_facts(owner_id, session_id, include_history=include_history, **kwargs)

    def evidence_refs(self, owner_id, session_id, *, turn_id=None, limit=100):
        return self._persistent().evidence_refs(owner_id, session_id, turn_id=turn_id, limit=limit)

    def context_epoch(self, owner_id, session_id):
        return self._persistent().context_epoch(owner_id,session_id)

    deletion_epoch = context_epoch

    def validate_context_epoch(self, owner_id, session_id, expected_epoch):
        return self._persistent().validate_context_epoch(owner_id,session_id,expected_epoch)

    def context_state(self, owner_id, session_id):
        return self._persistent().context_state(owner_id,session_id)

    def record_current_request(self, owner_id, session_id, **kwargs):
        # Called inside turn() before model execution. Only advance the cache
        # revision if it still matches the exact pre-event revision; concurrent
        # transcript writes must continue to fail the append CAS.
        from core.session_repository import _key, _now
        _key(owner_id,session_id)
        with self._lock:
            repo=self._persistent()
            before=repo.revision(owner_id,session_id)
            entry=self._entries.get((owner_id,session_id))
            if entry is not None and entry.invalidated:
                from core.session_repository import SessionConflictError
                raise SessionConflictError('session deleted during active turn')
            if entry is not None and entry.active and entry.revision!=before:
                from core.session_repository import SessionConflictError
                raise SessionConflictError('session changed before current request')
            # A first request has no persisted turn yet; create a session at the
            # current tombstone revision, preserving epoch. This writes no turn.
            with repo._connect() as db:
                db.execute('INSERT OR IGNORE INTO sessions VALUES (?,?,?,?)',(owner_id,session_id,before,_now()))
                db.execute('INSERT OR IGNORE INTO session_epochs VALUES (?,?,0,?)',(owner_id,session_id,before))
            kwargs.setdefault('expected_revision',before)
            event=repo.record_current_request(owner_id,session_id,**kwargs)
            if entry is not None and entry.revision==before:
                entry.revision=event['revision']
            return event

    def generate_summary(self, owner_id, session_id, **kwargs):
        return self._persistent().generate_summary(owner_id,session_id,**kwargs)

    def get_semantic_summary(self, owner_id, session_id):
        return self._persistent().get_semantic_summary(owner_id,session_id)

    def summary_attempts(self, owner_id, session_id, **kwargs):
        return self._persistent().summary_attempts(owner_id,session_id,**kwargs)

    def update_task(self, owner_id, session_id, **kwargs):
        return self._persistent().update_task(owner_id,session_id,**kwargs)

    def list_tasks(self, owner_id, session_id):
        return self._persistent().list_tasks(owner_id,session_id)

    def select_task(self, owner_id, session_id, **kwargs):
        return self._persistent().select_task(owner_id,session_id,**kwargs)

    def revoke_task_fact(self, owner_id, session_id, key, **kwargs):
        return self._persistent().revoke_task_fact(owner_id,session_id,key,**kwargs)

    delete_task_fact = revoke_task_fact

    def confirm_task_fact(self, owner_id, session_id, key, **kwargs):
        return self._persistent().confirm_task_fact(owner_id,session_id,key,**kwargs)

    def get_turn_page(self, owner_id, session_id, turn_id, **kwargs):
        return self._persistent().get_turn_page(owner_id,session_id,turn_id,**kwargs)

    def extract_candidates(self, owner_id, session_id, **kwargs):
        # Candidate publication is a session mutation. This hook executes within
        # the same active turn as request capture, so advance only its known CAS.
        repo=self._persistent()
        before=repo.revision(owner_id,session_id)
        result=repo.extract_candidates(owner_id,session_id,**kwargs)
        if result.get('status')=='generated' and result.get('candidates'):
            with self._lock:
                entry=self._entries.get((owner_id,session_id))
                if (entry is not None and not entry.invalidated and entry.revision==before==result.get('source_revision')
                        and repo.revision(owner_id,session_id)==result.get('published_revision')
                        and repo.context_epoch(owner_id,session_id)==result.get('epoch')):
                    entry.revision=result['published_revision']
        return result
