"""Single-owner-friendly SQLite transcripts with explicit provenance.

All keys include owner and session. Stored assistant text/evidence references
are historical records, never proof of current mail contents or permission.
"""
from __future__ import annotations

from contextlib import closing, contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from core.session_state import ContextRepositoryMixin


class SessionConflictError(ValueError):
    pass


def _key(owner_id, session_id):
    if any(not isinstance(value, str) or not value.strip() or len(value) > 128
           for value in (owner_id, session_id)):
        raise ValueError("owner and session must be nonempty identifiers")
    return owner_id, session_id


def _limit(value, maximum=200):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError("invalid page limit")
    return value


def _now():
    return datetime.now(timezone.utc).isoformat()


def _decode(row):
    return {"seq": row["seq"], "turn_id": row["turn_id"], "query": row["query"],
            "answer": row["answer"], "metadata": json.loads(row["metadata"]),
            "include_in_context": bool(row["include_context"]), "created_at": row["created_at"]}


def excerpt_summary(turns, *, max_chars=300):
    """Bounded quoted excerpts with source IDs; no inferred facts or summaries."""
    if type(max_chars) is not int or not 40 <= max_chars <= 2000:
        raise ValueError("invalid summary excerpt limit")
    return {"method": "deterministic_excerpts_v1", "not_evidence": True,
            "turns": [{"turn_id": row["turn_id"], "seq": row["seq"],
                       "user_excerpt": row["query"][:max_chars],
                       "assistant_excerpt": row["answer"][:max_chars],
                       "status": row["metadata"].get("status", "success"),
                       "include_in_context": row["include_in_context"],
                       "truncated": len(row["query"]) > max_chars or len(row["answer"]) > max_chars}
                      for row in turns]}


class SessionRepository(ContextRepositoryMixin):
    def __init__(self, path):
        import threading
        self._diagnostics_local=threading.local()
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Back up pre-context databases with SQLite backup (includes WAL).
        if self.path.exists():
            with closing(sqlite3.connect(self.path)) as original:
                migrated = original.execute("SELECT 1 FROM sqlite_master WHERE name='context_schema'").fetchone()
                if migrated:
                    schema=original.execute('SELECT version FROM context_schema').fetchone()
                    if not schema or schema[0]!=1:
                        raise ValueError('unsupported context schema version')
                has_sessions = original.execute("SELECT 1 FROM sqlite_master WHERE name='sessions'").fetchone()
                if has_sessions and not migrated:
                    backup = self.path.with_suffix(self.path.suffix + ".pre-context.bak")
                    if not backup.exists():
                        with closing(sqlite3.connect(backup)) as target:
                            original.backup(target)
                        import hashlib
                        digest=hashlib.sha256()
                        with backup.open('rb') as stream:
                            for block in iter(lambda:stream.read(1024*1024),b''):
                                digest.update(block)
                        backup.with_suffix(backup.suffix+'.sha256').write_text(digest.hexdigest()+'\n',encoding='ascii')
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    owner_id TEXT NOT NULL, session_id TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL,
                    PRIMARY KEY(owner_id, session_id));
                CREATE TABLE IF NOT EXISTS session_turns (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, owner_id TEXT NOT NULL,
                    session_id TEXT NOT NULL, turn_id TEXT NOT NULL, query TEXT NOT NULL,
                    answer TEXT NOT NULL, metadata TEXT NOT NULL, include_context INTEGER NOT NULL,
                    created_at TEXT NOT NULL, UNIQUE(owner_id, session_id, turn_id),
                    FOREIGN KEY(owner_id, session_id) REFERENCES sessions ON DELETE CASCADE);
                CREATE INDEX IF NOT EXISTS turn_lookup ON session_turns(owner_id, session_id, seq);
                CREATE TABLE IF NOT EXISTS task_facts (
                    owner_id TEXT NOT NULL, session_id TEXT NOT NULL, fact_key TEXT NOT NULL,
                    version INTEGER NOT NULL, kind TEXT NOT NULL, value TEXT NOT NULL,
                    source_turn_id TEXT NOT NULL, updated_at TEXT NOT NULL,
                    PRIMARY KEY(owner_id, session_id, fact_key, version),
                    FOREIGN KEY(owner_id, session_id, source_turn_id)
                    REFERENCES session_turns(owner_id, session_id, turn_id) ON DELETE CASCADE);
                CREATE TABLE IF NOT EXISTS turn_evidence (
                    owner_id TEXT NOT NULL, session_id TEXT NOT NULL, turn_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL, reference TEXT NOT NULL,
                    PRIMARY KEY(owner_id, session_id, turn_id, ordinal),
                    FOREIGN KEY(owner_id, session_id, turn_id)
                    REFERENCES session_turns(owner_id, session_id, turn_id) ON DELETE CASCADE);
            """)

        self._init_context()

    @property
    def last_history_diagnostics(self):
        return getattr(self._diagnostics_local,'history',{})

    @last_history_diagnostics.setter
    def last_history_diagnostics(self,value):
        self._diagnostics_local.history=value

    @property
    def last_search_diagnostics(self):
        return self.last_history_diagnostics

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def revision(self, owner_id, session_id):
        with self._connect() as db:
            row = db.execute("SELECT revision FROM sessions WHERE owner_id=? AND session_id=?",
                             _key(owner_id, session_id)).fetchone()
            if row:
                return row[0]
            tombstone = db.execute("SELECT revision FROM session_epochs WHERE owner_id=? AND session_id=?", _key(owner_id,session_id)).fetchone()
            return tombstone[0] if tombstone else 0

    def append_turns(self, owner_id, session_id, turns, *, expected_revision):
        key = _key(owner_id, session_id)
        if not turns:
            return expected_revision
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute("SELECT revision FROM sessions WHERE owner_id=? AND session_id=?", key).fetchone()
            tombstone = db.execute("SELECT revision FROM session_epochs WHERE owner_id=? AND session_id=?", key).fetchone()
            revision = current[0] if current else (tombstone[0] if tombstone else 0)
            if revision != expected_revision:
                raise SessionConflictError("session changed; reload before retrying")
            db.execute("INSERT OR IGNORE INTO sessions VALUES (?,?,?,?)", (*key, revision, _now()))
            db.execute("INSERT OR IGNORE INTO session_epochs VALUES (?,?,0,?)", (*key,revision))
            for turn in turns:
                if not isinstance(turn.get("query"), str) or not isinstance(turn.get("answer"), str):
                    raise ValueError("a transcript turn requires both query and answer strings")
                metadata = turn.get("metadata") or {}
                status = metadata.get("status", "success")
                eligible = (status == "success" and metadata.get("completion_status", "complete") == "complete"
                            and bool(turn["answer"].strip()))
                include = bool(turn.get("include_in_context")) and eligible
                db.execute("""INSERT INTO session_turns(owner_id,session_id,turn_id,query,answer,metadata,
                              include_context,created_at) VALUES (?,?,?,?,?,?,?,?)""",
                           (*key, turn["turn_id"], turn["query"], turn["answer"],
                            json.dumps(metadata, ensure_ascii=False), int(include), turn.get("created_at") or _now()))
                db.execute("INSERT INTO context_turn_tasks VALUES (?,?,?,?)",(*key,turn["turn_id"],self._active_task(db,key)))
                if self.fts_available:
                    from core.history_index import index_turn
                    indexed = db.execute("SELECT * FROM session_turns WHERE owner_id=? AND session_id=? AND turn_id=?", (*key,turn["turn_id"])).fetchone()
                    db.execute('SAVEPOINT history_index_write')
                    try:
                        index_turn(db,indexed)
                        db.execute('RELEASE history_index_write')
                    except (sqlite3.OperationalError,ValueError) as exc:
                        db.execute('ROLLBACK TO history_index_write')
                        db.execute('RELEASE history_index_write')
                        if any(word in str(exc).lower() for word in ('disk','readonly','locked','i/o','malformed')):
                            raise
                        self.history_index_degraded=type(exc).__name__
                        db.execute('INSERT OR REPLACE INTO history_index_failures VALUES (?,?,?)',(indexed['seq'],type(exc).__name__,_now()))
                refs = turn.get("evidence_refs") or []
                if not isinstance(refs, list) or len(refs) > 200:
                    raise ValueError("too many evidence references")
                for ordinal, ref in enumerate(refs):
                    if (not isinstance(ref, dict) or any(not isinstance(ref.get(name), str) or
                        not ref[name] or len(ref[name]) > 512 for name in ("email_id", "chunk_id"))):
                        raise ValueError("invalid evidence reference")
                    value = {name: ref.get(name) for name in ("email_id", "chunk_id", "source_version",
                             "source_sha256", "visible_start", "visible_end", "visible_sha256",
                             "visible_hash", "chunk_sha256", "offset_basis") if name in ref}
                    value["validation_status"] = "unverified_requires_source_read"
                    db.execute("INSERT INTO turn_evidence VALUES (?,?,?,?,?)",
                               (*key, turn["turn_id"], ordinal, json.dumps(value, ensure_ascii=False)))
            revision += 1
            db.execute("UPDATE sessions SET revision=?,updated_at=? WHERE owner_id=? AND session_id=?",
                       (revision, _now(), *key))
            db.execute("UPDATE session_epochs SET revision=? WHERE owner_id=? AND session_id=?", (revision,*key))
            return revision

    def recent_context(self, owner_id, session_id, max_turns=30):
        # Even an unlimited requested window is bounded during cache loading;
        # old transcript remains available via paging/search rather than RAM.
        if max_turns is not None and (type(max_turns) is not int or max_turns < 1):
            raise ValueError("invalid history window")
        count = 1000 if max_turns is None else min(max_turns, 1000)
        with self._connect() as db:
            rows = db.execute("""SELECT * FROM session_turns WHERE owner_id=? AND session_id=?
                               AND include_context=1 AND turn_id IN (SELECT turn_id FROM context_turn_tasks
                               WHERE owner_id=? AND session_id=? AND task_id=?) ORDER BY seq DESC LIMIT ?""",
                              (*_key(owner_id, session_id), owner_id,session_id,self._active_task(db,(owner_id,session_id)),count)).fetchall()
        return [_decode(row) for row in reversed(rows)]

    def history(self, owner_id, session_id, *, after=0, limit=100):
        if type(after) is not int or after < 0:
            raise ValueError("invalid history cursor")
        with self._connect() as db:
            rows = db.execute("""SELECT * FROM session_turns WHERE owner_id=? AND session_id=?
                               AND seq>? ORDER BY seq LIMIT ?""",
                              (*_key(owner_id, session_id), after, _limit(limit)+1)).fetchall()
        page = [_decode(row) for row in rows[:limit]]
        return {"session_id": session_id, "turns": page, "has_more": len(rows) > limit,
                "next_after": page[-1]["seq"] if page else after}

    def list_sessions(self, owner_id, *, limit=100):
        return self.list_sessions_page(owner_id, limit=limit)["sessions"]

    def list_sessions_page(self, owner_id, *, limit=50, offset=0):
        _key(owner_id, "list")
        limit = _limit(limit)
        if type(offset) is not int or not 0 <= offset <= 1000000:
            raise ValueError("invalid session listing offset")
        with self._connect() as db:
            rows = db.execute("""SELECT session_id,updated_at,revision,
                              (SELECT count(*) FROM session_turns t WHERE t.owner_id=s.owner_id
                               AND t.session_id=s.session_id) AS turn_count
                              FROM sessions s WHERE owner_id=? ORDER BY updated_at DESC,session_id ASC LIMIT ? OFFSET ?""",
                              (owner_id, limit + 1, offset)).fetchall()
        return {"sessions": [dict(row) for row in rows[:limit]],
                "next_offset": offset + limit if len(rows) > limit else None}

    def get_turn(self, owner_id, session_id, turn_id):
        with self._connect() as db:
            row = db.execute("SELECT * FROM session_turns WHERE owner_id=? AND session_id=? AND turn_id=?",
                             (*_key(owner_id, session_id), turn_id)).fetchone()
        if row is None:
            raise KeyError("transcript turn not found")
        return _decode(row)

    def delete(self, owner_id, session_id):
        key=_key(owner_id,session_id)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            current=db.execute("SELECT revision FROM sessions WHERE owner_id=? AND session_id=?",key).fetchone()
            db.execute("INSERT OR IGNORE INTO session_epochs VALUES (?,?,0,?)",(*key,current[0] if current else 0))
            db.execute("UPDATE session_epochs SET epoch=epoch+1,revision=max(revision,?)+1 WHERE owner_id=? AND session_id=?",(current[0] if current else 0,*key))
            db.execute("DELETE FROM sessions WHERE owner_id=? AND session_id=?",key)

    def search_history(self, owner_id, session_id, query, *, limit=30, task_id=None):
        from core.history_index import search,hit_windows
        if not isinstance(query,str) or not query.strip() or len(query)>500:
            raise ValueError("invalid history search")
        with self._connect() as db:
            self.last_history_diagnostics={}
            results=search(db,_key(owner_id,session_id),query,_limit(limit),self.fts_available,task_id=task_id,diagnostics=self.last_history_diagnostics)
            incomplete=db.execute('SELECT 1 FROM session_turns t LEFT JOIN history_documents d ON d.seq=t.seq WHERE t.owner_id=? AND t.session_id=? AND d.seq IS NULL LIMIT 1',(owner_id,session_id)).fetchone() is not None
            self.last_history_diagnostics['index_incomplete']=incomplete
            for row,metadata in results:metadata['index_incomplete']=incomplete
            task_ids={}
            if results:
                placeholders=','.join('?' for _ in results)
                task_ids={row['turn_id']:row['task_id'] for row in db.execute('SELECT turn_id,task_id FROM context_turn_tasks WHERE owner_id=? AND session_id=? AND turn_id IN ('+placeholders+')',(owner_id,session_id,*(row['turn_id'] for row,metadata in results)))}
        return [{**_decode(row),"task_id":task_ids.get(row['turn_id'],'default'),"hits":hit_windows(row,query),"retrieval":metadata} for row,metadata in results]

    def get_turn_page(self, owner_id, session_id, turn_id, *, field='answer', offset=0, limit=2000, expected_epoch=None):
        import hashlib
        if field not in {'query','answer'} or type(offset) is not int or offset<0 or type(limit) is not int or not 1<=limit<=8000:
            raise ValueError('invalid transcript range')
        with self._connect() as db:
            db.execute('BEGIN')
            identity=_key(owner_id,session_id)
            state=db.execute('SELECT s.revision,e.epoch FROM sessions s JOIN session_epochs e ON e.owner_id=s.owner_id AND e.session_id=s.session_id WHERE s.owner_id=? AND s.session_id=?',identity).fetchone()
            if state is None: raise KeyError('session not found')
            if expected_epoch is not None and (type(expected_epoch) is not int or expected_epoch!=state['epoch']):
                raise SessionConflictError('session deletion epoch changed')
            original=db.execute('SELECT * FROM session_turns WHERE owner_id=? AND session_id=? AND turn_id=?',(*identity,turn_id)).fetchone()
            if original is None: raise KeyError('transcript turn not found')
            row=_decode(original)
            revision,epoch=state['revision'],state['epoch']
        source=row[field]
        if offset>len(source): raise ValueError('offset exceeds source')
        text=source[offset:offset+limit];end=offset+len(text)
        return {'turn_id':turn_id,'seq':row['seq'],'field':field,'text':text,'start':offset,'end':end,
                'sha256':hashlib.sha256(text.encode()).hexdigest(),'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
                'has_more':end<len(source),'next_offset':end if end<len(source) else None,
                'metadata':row['metadata'],'include_in_context':row['include_in_context'],'offset_basis':'unicode_codepoints',
                'revision':revision,'deletion_epoch':epoch}

    def summary(self, owner_id, session_id, *, after=0, limit=20, max_chars=300):
        page = self.history(owner_id, session_id, after=after, limit=limit)
        return {**excerpt_summary(page["turns"], max_chars=max_chars),
                "has_more": page["has_more"], "next_after": page["next_after"]}

    def evidence_refs(self, owner_id, session_id, *, turn_id=None, limit=100):
        query = "SELECT turn_id,reference FROM turn_evidence WHERE owner_id=? AND session_id=?"
        params = [*_key(owner_id, session_id)]
        if turn_id is not None:
            query += " AND turn_id=?"
            params.append(turn_id)
        query += " ORDER BY rowid DESC LIMIT ?"
        params.append(_limit(limit))
        with self._connect() as db:
            rows = db.execute(query, params).fetchall()
        return [{"turn_id": row["turn_id"], **json.loads(row["reference"])} for row in rows]
