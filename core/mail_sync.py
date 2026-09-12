"""Account-scoped, read-only IMAP capture and local parsing.

Raw content is durably saved before a per-message SQLite commit. UID lists are
snapshots, not a high-water cursor: failed or interrupted work cannot skip mail.
No corpus/index/model API is imported or called by this module.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
from itertools import zip_longest
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from agents.runtime import RunCancelled, current_run, remaining_timeout


class ImapSyncError(RuntimeError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, prefix='.pending-') as out:
            pending = Path(out.name)
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        os.replace(pending, path)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def message_key(account_id, folder, validity, uid):
    return hashlib.sha256(json.dumps([account_id, folder, str(validity), int(uid)],
                                    ensure_ascii=False).encode()).hexdigest()


def normalized_flags(values):
    system = {value.casefold():value for value in ('\\Seen','\\Flagged','\\Deleted','\\Answered','\\Draft','\\Recent')}
    return sorted({system.get(value.casefold(),value) for value in values})


def isolated_parse(raw_path, locator, timeout=30):
    """A fresh, bounded child process keeps malformed parsers out of the API."""
    from core.parse_process import parser_process, ParserProcessError
    raw_path = Path(raw_path)
    with tempfile.TemporaryDirectory(dir=raw_path.parent, prefix='.parse-') as temp:
        folder = Path(temp)
        target, location = folder/'result.json', folder/'locator.json'
        location.write_text(json.dumps(locator), encoding='utf-8')
        try:
            with parser_process([sys.executable, '-m', 'core.mail_parse_worker', str(raw_path),
                                 str(target), str(location)], cwd=Path(__file__).resolve().parent.parent) as child:
                end = time.monotonic() + timeout
                while child.poll() is None:
                    remaining_timeout(1)
                    if time.monotonic() >= end:
                        raise ImapSyncError('parse_timeout')
                    time.sleep(.05)
                if child.returncode or not target.is_file():
                    raise ImapSyncError('parser_process_failed')
                if target.stat().st_size > 12_000_000:
                    raise ImapSyncError('parse_output_limit')
                result = json.loads(target.read_text(encoding='utf-8'))
                if result.get('error_code'):
                    raise ImapSyncError(result['error_code'])
                from models.schemas import Email
                return Email.model_validate(result['email'])
        except ParserProcessError:
            raise ImapSyncError('parser_process_failed') from None


class MailSyncStore:
    def __init__(self, root, account_id):
        self.root, self.account_id = Path(root), account_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root/'mail.sqlite3'
        with self.db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS binding (account_id TEXT PRIMARY KEY)')
            old = db.execute('SELECT account_id FROM binding').fetchall()
            if old and [r[0] for r in old] != [account_id]:
                raise ImapSyncError('account_binding_mismatch')
            db.execute('INSERT OR IGNORE INTO binding VALUES (?)', (account_id,))
            db.execute('''CREATE TABLE IF NOT EXISTS folders (
                name TEXT PRIMARY KEY, uidvalidity TEXT NOT NULL, remote_count INTEGER NOT NULL,
                scanned REAL NOT NULL)''')
            db.execute('''CREATE TABLE IF NOT EXISTS messages (
                key TEXT PRIMARY KEY, folder TEXT NOT NULL, uidvalidity TEXT NOT NULL, uid INTEGER NOT NULL,
                present INTEGER NOT NULL DEFAULT 1, status TEXT NOT NULL, raw_sha256 TEXT,
                email TEXT, error_code TEXT, parser_version TEXT, attempts INTEGER NOT NULL DEFAULT 1,
                updated REAL NOT NULL)''')
            db.execute('CREATE INDEX IF NOT EXISTS message_folder ON messages(folder,uidvalidity,present,uid)')
            db.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
            if 'flags' not in {row[1] for row in db.execute('PRAGMA table_info(messages)')}:
                db.execute("ALTER TABLE messages ADD COLUMN flags TEXT NOT NULL DEFAULT '[]'")
                db.execute("UPDATE messages SET flags=COALESCE(json_extract(email,'$.labels'),'[]')")
            from core.mail_search import ensure
            ensure(db)

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @contextmanager
    def sync_lock(self):
        # Separate database: readers can browse mail while the sync lock is held.
        db = sqlite3.connect(self.root/'sync-lock.sqlite3', timeout=.1)
        try:
            try:
                db.execute('BEGIN IMMEDIATE')
            except sqlite3.OperationalError:
                raise ImapSyncError('account_sync_busy') from None
            yield
        finally:
            db.close()

    def snapshot(self, folder, validity, uids):
        from core.mail_search import update
        with self.db() as db:
            previously_absent = {row['key']:json.loads(row['email']) for row in db.execute(
                "SELECT key,email FROM messages WHERE folder=? AND present=0 AND status='parsed'", (folder,))}
            db.execute('UPDATE messages SET present=0 WHERE folder=?', (folder,))
            db.executemany('UPDATE messages SET present=1 WHERE folder=? AND uidvalidity=? AND uid=?',
                           ((folder, validity, uid) for uid in uids))
            db.execute('INSERT OR REPLACE INTO folders VALUES (?,?,?,?)', (folder, validity, len(uids), time.time()))
            for row in db.execute('SELECT key FROM messages WHERE folder=? AND present=0', (folder,)):
                update(db, row['key'], None)
            for row in db.execute("SELECT key FROM messages WHERE folder=? AND present=1 AND status='parsed'", (folder,)):
                if row['key'] in previously_absent:
                    update(db, row['key'], previously_absent[row['key']])
            return {row['uid']: dict(row) for row in db.execute(
                'SELECT key,uid,status,parser_version,attempts,raw_sha256 FROM messages WHERE folder=? AND uidvalidity=? AND present=1',
                (folder, validity))}

    def reconcile_folders(self, available):
        """A successful LIST can retire a folder which no longer exists."""
        from core.mail_search import update
        with self.db() as db:
            missing = [row[0] for row in db.execute('SELECT name FROM folders') if row[0] not in available]
            for folder in missing:
                for row in db.execute('SELECT key FROM messages WHERE folder=?', (folder,)):
                    update(db, row['key'], None)
                db.execute('UPDATE messages SET present=0 WHERE folder=?', (folder,))
                db.execute('UPDATE folders SET remote_count=0,scanned=? WHERE name=?', (time.time(),folder))

    def update_flags(self, folder, validity, flags):
        """Refresh local status without downloading or reparsing known bodies."""
        from core.mail_search import update
        changed = 0
        with self.db() as db:
            rows = db.execute('SELECT key,uid,flags,email FROM messages WHERE folder=? AND uidvalidity=?',
                              (folder,validity)).fetchall()
            for row in rows:
                if row['uid'] not in flags:
                    continue
                current = normalized_flags(flags[row['uid']])
                if current == sorted(set(json.loads(row['flags']))):
                    continue
                email = json.loads(row['email']) if row['email'] else None
                if email is not None:
                    email['labels'] = email['label_names'] = current
                db.execute('UPDATE messages SET flags=?,email=? WHERE key=?',
                           (json.dumps(current),json.dumps(email,ensure_ascii=False) if email else None,row['key']))
                if email is not None and ('\\Deleted' in current) != ('\\Deleted' in json.loads(row['flags'])):
                    update(db, row['key'], None if '\\Deleted' in current else email)
                changed += 1
        return changed

    def remaining_eligible(self, folders, version, retry_failed=False):
        """Pending bodies within this run's scope, including limited retries."""
        total = 0
        with self.db() as db:
            for folder in folders:
                snapshot = db.execute('SELECT remote_count FROM folders WHERE name=?',(folder,)).fetchone()
                if snapshot is None:
                    continue
                known = db.execute('SELECT COUNT(*) FROM messages WHERE folder=? AND present=1',(folder,)).fetchone()[0]
                retry = db.execute("SELECT COUNT(*) FROM messages WHERE folder=? AND present=1 AND "
                                   "(parser_version!=? OR (status!='parsed' AND (attempts<3 OR ?)))",
                                   (folder,version,retry_failed)).fetchone()[0]
                total += max(0,snapshot['remote_count']-known) + retry
        return total

    def save(self, folder, validity, uid, *, raw=None, email=None, error_code=None, version='', flags=None):
        from core.mail_search import update
        digest = hashlib.sha256(raw).hexdigest() if raw is not None else None
        if raw is not None:
            location = self.raw_path(digest)
            if not location.exists() or hashlib.sha256(location.read_bytes()).hexdigest() != digest:
                atomic_write(location, raw)
        key = message_key(self.account_id, folder, validity, uid)
        encoded = json.dumps(email.model_dump(), ensure_ascii=False) if email is not None else None
        encoded_flags = json.dumps(sorted(set(email.labels if email is not None else (flags or []))))
        with self.db() as db:
            db.execute('''INSERT INTO messages (key,folder,uidvalidity,uid,present,status,raw_sha256,email,error_code,parser_version,updated)
                VALUES (?,?,?,?,1,?,?,?,?,?,?) ON CONFLICT(key) DO UPDATE SET present=1,status=excluded.status,
                raw_sha256=COALESCE(excluded.raw_sha256,messages.raw_sha256),email=excluded.email,
                error_code=excluded.error_code,parser_version=excluded.parser_version,
                attempts=messages.attempts+1,updated=excluded.updated''',
                       (key, folder, validity, uid, 'parsed' if email is not None else 'failed', digest,
                        encoded, error_code, version, time.time()))
            db.execute('UPDATE messages SET flags=? WHERE key=?', (encoded_flags,key))
            update(db, key, email)
        return key

    def raw_path(self, digest):
        import re
        if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
            raise KeyError('Raw source not found')
        return self.root/'raw'/f'{digest}.eml'

    def report(self):
        counts, issues, attachments = Counter(), Counter(), Counter()
        with self.db() as db:
            folders = [dict(row) for row in db.execute('SELECT * FROM folders ORDER BY name')]
            archived = db.execute('SELECT COUNT(*) FROM messages WHERE present=0').fetchone()[0]
            for row in db.execute('SELECT status,email,error_code FROM messages WHERE present=1'):
                counts[row['status']] += 1
                if row['error_code']:
                    issues[row['error_code']] += 1
                if row['email']:
                    email = json.loads(row['email'])
                    counts['body_nonempty' if email.get('body','').strip() else 'body_empty'] += 1
                    quality = email.get('decode_quality') or {}
                    if quality.get('status') == 'suspect' or any(isinstance(v,dict) and v.get('status')=='suspect' for v in quality.values()):
                        counts['decode_suspect'] += 1
                    for warning in (email.get('source') or {}).get('warnings',[]):
                        issues['parse:'+str(warning)] += 1
                    for part in email.get('attachments', []):
                        attachments[part.get('status','not_read')] += 1
                        if part.get('status') != 'complete' and part.get('reason'):
                            issues['attachment:'+part['reason']] += 1
            row = db.execute("SELECT value FROM meta WHERE key='last_run'").fetchone()
            from core.mail_search import status
            search_status = status(db)
        remote = sum(row['remote_count'] for row in folders)
        return {'account_id':self.account_id, 'folders':folders, 'remote_snapshot_count':remote,
                'parsed':counts['parsed'], 'failed':counts['failed'],
                'not_downloaded':max(0,remote-counts['parsed']-counts['failed']),
                'body_nonempty':counts['body_nonempty'], 'body_empty':counts['body_empty'],
                'decode_suspect':counts['decode_suspect'], 'attachments':dict(attachments),
                'issues':dict(issues), 'historical_records':archived,
                'last_run':json.loads(row['value']) if row else None,
                'scope':'selected_folder_snapshots', 'local_only':True, 'model_calls':0,
                'local_search':search_status}

    def messages(self, *, offset=0, limit=25, failures_only=False):
        condition = "present=1" + (" AND status!='parsed'" if failures_only else '')
        with self.db() as db:
            total = db.execute(f'SELECT COUNT(*) FROM messages WHERE {condition}').fetchone()[0]
            rows = db.execute(f"SELECT * FROM messages WHERE {condition} ORDER BY julianday(json_extract(email,'$.date')) DESC,uid DESC,key LIMIT ? OFFSET ?",
                              (limit,offset)).fetchall()
        items = []
        for row in rows:
            value = dict(row)
            value['flags'] = json.loads(value.get('flags') or '[]')
            value['is_read'] = '\\Seen' in value['flags']
            value['is_starred'] = '\\Flagged' in value['flags']
            email = json.loads(value.pop('email') or '{}')
            value.update({k:email.get(k) for k in ('id','subject','sender','date')})
            value['attachment_count'] = len(email.get('attachments',[]))
            items.append(value)
        return {'items':items, 'total':total, 'next_offset':offset+limit if offset+limit<total else None}

    def message(self, key):
        with self.db() as db:
            row = db.execute('SELECT * FROM messages WHERE key=?', (key,)).fetchone()
        if row is None:
            raise KeyError('Message not found')
        result = dict(row)
        result['flags'] = json.loads(result.get('flags') or '[]')
        result['email'] = json.loads(result['email']) if result['email'] else None
        return result

    def search(self, query, **options):
        from core.mail_search import search
        with self.db() as db:
            return search(db,query,**options)


def sync_mailbox(provider, store, folders, *, max_messages=100, parser=None, parse_timeout=30, check_binding=None, retry_failed=False):
    from agents.imap_readonly import ImapReadError, ImapMessageMissing, ImapMessageTooLarge
    from core.imap_mime import MailParseError, parser_version
    if not folders or len(folders)>30 or len(set(folders)) != len(folders) or not 1 <= max_messages <= 2000:
        raise ValueError('invalid_sync_scope')
    version = parser_version()
    run = current_run()
    started, attempted, fetched, parsed, failures = time.time(), 0, 0, 0, 0
    flags_updated, excluded_deleted = 0, 0
    with store.sync_lock():
        if check_binding:
            check_binding()
        listed = provider.list_folders()
        available = {r['name'] for r in listed if r['selectable']}
        store.reconcile_folders({r['name'] for r in listed})
        if any(folder not in available for folder in folders):
            raise ImapSyncError('folder_not_available')
        with store.db() as db:
            cursor = db.execute("SELECT value FROM meta WHERE key='next_folder'").fetchone()
        start = folders.index(cursor['value']) if cursor and cursor['value'] in folders else 0
        ordered = folders[start:] + folders[:start]
        plans = {}
        for folder in ordered:
            remaining_timeout(30)
            selected = provider.select_folder(folder)
            validity = str(selected['uidvalidity'])
            uids = sorted(set(provider.list_uids()), reverse=True)
            flags = {uid:normalized_flags(value) for uid,value in provider.fetch_flags(uids).items()}
            missing = set(uids)-flags.keys()
            # Missing FETCH rows are valid only for UIDs actually expunged
            # during the scan; a partial server response cannot hide live mail.
            if missing and missing.intersection(provider.list_uids()):
                raise ImapReadError('incomplete_flags_snapshot')
            remaining_timeout(30)
            uids = [uid for uid in uids if uid in flags and '\\Deleted' not in flags[uid]]
            excluded_deleted += sum('\\Deleted' in value for value in flags.values())
            existing = store.snapshot(folder, validity, uids)
            flags_updated += store.update_flags(folder, validity, flags)
            candidates = [uid for uid in uids if uid not in existing]
            candidates += [uid for uid in uids if uid in existing and existing[uid]['parser_version']!=version]
            candidates += [uid for uid in uids if uid in existing and existing[uid]['parser_version']==version
                           and existing[uid]['status']!='parsed' and (retry_failed or existing[uid]['attempts']<3)]
            plans[folder] = (validity,candidates,flags)
        # Select the batch fairly, then group network reads to avoid selecting
        # another folder for every single message. Persist rotation across jobs.
        selected_work = {}
        work_count = 0
        next_folder = ordered[0]
        for row in zip_longest(*(plans[folder][1] for folder in ordered)):
            for index,uid in enumerate(row):
                if uid is None:
                    continue
                folder = ordered[index]
                selected_work.setdefault(folder,[]).append(uid)
                next_folder = ordered[(index+1)%len(ordered)]
                work_count += 1
                if work_count >= max_messages:
                    break
            if work_count >= max_messages:
                break
        for folder, candidates in selected_work.items():
            validity, _, flags = plans[folder]
            if str(provider.select_folder(folder)['uidvalidity']) != validity:
                raise ImapSyncError('mailbox_changed_during_sync')
            for uid in candidates:
                remaining_timeout(30)
                if check_binding:
                    check_binding()
                if run:
                    run.progress('imap_sync', attempted=attempted, parsed=parsed, failed=failures)
                attempted += 1
                raw = None
                try:
                    # Fetch again when retrying: metadata/raw version is bound to
                    # this selected UIDVALIDITY, never just a recycled UID.
                    value = provider.fetch_message(uid)
                    raw = value['raw']
                    digest = hashlib.sha256(raw).hexdigest()
                    location = store.raw_path(digest)
                    atomic_write(location, raw)
                    fetched += 1
                    locator = dict(account_id=store.account_id, folder=folder, uidvalidity=validity,
                                   uid=uid, internal_date=value.get('internal_date',''), flags=normalized_flags(value.get('flags',[])))
                    email = parser(raw, **locator) if parser else isolated_parse(location, locator, parse_timeout)
                    email.source['transport'] = {
                        'reported_size':value.get('reported_size',value.get('size',len(raw))),
                        'received_size':len(raw),
                        'size_mismatch_verified':bool(value.get('size_mismatch',False)),
                    }
                    store.save(folder, validity, uid, raw=raw, email=email, version=version)
                    parsed += 1
                except (ImapMessageMissing, ImapMessageTooLarge, MailParseError, ImapSyncError) as exc:
                    store.save(folder, validity, uid, raw=raw, error_code=exc.code, version=version,flags=flags[uid])
                    failures += 1
                except ImapReadError as exc:
                    # A fully received but invalid message response is isolated.
                    # Socket/auth/session failures still stop this connection.
                    if exc.code not in {'unexpected_literal','invalid_fetch_response','unexpected_uid',
                                        'incomplete_fetch_metadata','invalid_message_size','invalid_fetch_metadata',
                                        'incomplete_message','message_changed_during_fetch'}:
                        raise
                    store.save(folder, validity, uid, error_code=exc.code, version=version,flags=flags[uid])
                    failures += 1
                if run:
                    run.checkpoint({'safe':True, 'kind':'imap_sync', 'account_id':store.account_id})
                    if run.progress_callback:
                        # Publish committed work even if cancellation arrived
                        # during it; the next operation still checks the context.
                        run.progress_callback('imap_sync', attempted=attempted, parsed=parsed, failed=failures)
        summary = {'started':started, 'finished':time.time(), 'folders':folders,
                   'attempted':attempted, 'fetched':fetched, 'parsed':parsed, 'failed':failures,
                   'flags_updated':flags_updated,'excluded_deleted':excluded_deleted}
        with store.db() as db:
            db.execute("INSERT OR REPLACE INTO meta VALUES ('last_run',?)", (json.dumps(summary),))
            db.execute("INSERT OR REPLACE INTO meta VALUES ('next_folder',?)", (next_folder,))
        remaining = store.remaining_eligible(folders,version,retry_failed)
    report = store.report()
    if run and run.progress_callback:
        run.progress_callback('imap_complete', attempted=attempted, parsed=parsed, failed=failures)
    report['metadata'] = {'status':'success', 'completion_status':'complete', 'model_calls':0}
    report['remaining_eligible'] = remaining
    return report
