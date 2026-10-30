"""Local mailbox-only FTS5 retrieval. Uses only the caller's SQLite connection.

No model, network, embedding store, HTML rendering or fallback corpus scan.
All writes remain in the caller's transaction. Call ensure after adding the
messages.flags column, update within message-save transactions, and status for
index coverage. A connection used for search must not own another progress hook.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import re
import sqlite3
import time

from core.history_index import ANALYZER_VERSION, tokens

INDEX_VERSION = 'mail-fts-v1:' + ANALYZER_VERSION
MAX_QUERY_CHARS = 500
MAX_QUERY_TERMS = 32
MAX_RESULTS = 50
# MIME output: body <=800k, attachment text <=800k, headers <=128k,
# plus separators. The token count cannot exceed the indexed character count.
MAX_DOCUMENT_CHARS = 2_000_000
MAX_INDEX_TERMS = MAX_DOCUMENT_CHARS
MAX_STORED_JSON_CHARS = 12_000_000
MAX_RESULT_SCAN_CHARS = 4_000_000
SEARCH_SECONDS = 1.0
SNIPPET_CHARS = 320
_FIELDS = ('subject', 'sender', 'recipients', 'body', 'attachments')


class MailSearchError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


@contextmanager
def _atomic(db):
    # RELEASE of an outermost savepoint would otherwise commit unexpectedly.
    started = not db.in_transaction
    if started:
        db.execute('BEGIN')
    db.execute('SAVEPOINT mail_search_write')
    try:
        yield
        db.execute('RELEASE SAVEPOINT mail_search_write')
    except BaseException:
        db.execute('ROLLBACK TO SAVEPOINT mail_search_write')
        db.execute('RELEASE SAVEPOINT mail_search_write')
        if started:
            db.rollback()
        raise


def _key(key):
    if not isinstance(key, str) or not key or len(key) > 512:
        raise MailSearchError('invalid_message_key')


def _text(value):
    if value is None:
        return ''
    if not isinstance(value, str):
        raise MailSearchError('invalid_document_text')
    return value


def _document(email):
    if hasattr(email, 'model_dump'):
        email = email.model_dump()
    if not isinstance(email, dict):
        raise MailSearchError('invalid_document')
    if email.get('body_format', 'plain') != 'plain':
        raise MailSearchError('body_not_normalized_text')
    recipients = []
    for field in ('recipients', 'cc'):
        values = email.get(field, [])
        if not isinstance(values, list) or len(values) > MAX_DOCUMENT_CHARS:
            raise MailSearchError('invalid_document_recipients')
        recipients.extend(_text(value) for value in values)
    attached = email.get('attachments', [])
    if not isinstance(attached, list) or len(attached) > 256:
        raise MailSearchError('invalid_document_attachments')
    texts = []
    for attachment in attached:
        if not isinstance(attachment, dict):
            raise MailSearchError('invalid_document_attachments')
        if attachment.get('status') in {'complete', 'partial'}:
            texts.append(_text(attachment.get('text')))
    values = {'subject': _text(email.get('subject')), 'sender': _text(email.get('sender')),
              'recipients': '\n'.join(recipients), 'body': _text(email.get('body')),
              'attachments': '\n\n'.join(texts)}
    if sum(map(len, values.values())) > MAX_DOCUMENT_CHARS:
        raise MailSearchError('document_text_limit')
    date = _text(email.get('date'))
    if len(date) > 256:
        raise MailSearchError('invalid_document_date')
    terms = [tokens(values[field]) for field in _FIELDS]
    if sum(map(len, terms)) > MAX_INDEX_TERMS:
        raise MailSearchError('document_term_limit')
    digest = hashlib.sha256(json.dumps([values, date], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return values, date, [' '.join(items) for items in terms], digest


def _update(db, key, email):
    _key(key)
    old = db.execute('SELECT rowid FROM mail_search_documents WHERE message_key=?', (key,)).fetchone()
    if old:
        db.execute('DELETE FROM mail_search_fts WHERE rowid=?', (old[0],))
        db.execute('DELETE FROM mail_search_documents WHERE rowid=?', (old[0],))
    if email is None:
        return
    values, date, terms, digest = _document(email)
    cursor = db.execute('''INSERT INTO mail_search_documents
        (message_key,subject,sender,recipients,body,attachments,date,content_sha256)
        VALUES (?,?,?,?,?,?,?,?)''', (key, *(values[field] for field in _FIELDS), date, digest))
    db.execute('INSERT INTO mail_search_fts(rowid,subject,sender,recipients,body,attachments) VALUES (?,?,?,?,?,?)',
               (cursor.lastrowid, *terms))


def ensure(db):
    """Create/rebuild the index transactionally; never set version before success."""
    try:
        exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='mail_search_meta'").fetchone()
        if exists:
            version = db.execute("SELECT value FROM mail_search_meta WHERE key='version'").fetchone()
            if version and version[0] == INDEX_VERSION:
                for name in ('mail_search_documents', 'mail_search_fts'):
                    if not db.execute('SELECT 1 FROM sqlite_master WHERE name=?', (name,)).fetchone():
                        raise MailSearchError('index_schema_incomplete')
                return {'version': INDEX_VERSION, 'rebuilt': False}
        with _atomic(db):
            columns = {row[1] for row in db.execute('PRAGMA table_info(messages)')}
            if not {'key', 'email', 'present', 'status', 'folder', 'flags'}.issubset(columns):
                raise MailSearchError('messages_schema_required')
            db.execute('DROP TABLE IF EXISTS mail_search_fts')
            db.execute('DROP TABLE IF EXISTS mail_search_documents')
            db.execute('CREATE TABLE IF NOT EXISTS mail_search_meta (key TEXT PRIMARY KEY,value TEXT NOT NULL)')
            db.execute('''CREATE TABLE mail_search_documents (
                rowid INTEGER PRIMARY KEY, message_key TEXT NOT NULL UNIQUE,
                subject TEXT NOT NULL,sender TEXT NOT NULL,recipients TEXT NOT NULL,
                body TEXT NOT NULL,attachments TEXT NOT NULL,date TEXT NOT NULL,content_sha256 TEXT NOT NULL)''')
            try:
                db.execute("CREATE VIRTUAL TABLE mail_search_fts USING fts5(subject,sender,recipients,body,attachments,tokenize='unicode61')")
            except sqlite3.OperationalError as exc:
                if 'no such module' in str(exc).lower():
                    raise MailSearchError('fts_unavailable') from None
                raise
            cursor = db.execute("SELECT key,length(email) FROM messages WHERE present=1 AND status='parsed'")
            indexed = 0
            for key, length in cursor:
                if length is None or length > MAX_STORED_JSON_CHARS:
                    raise MailSearchError('stored_document_limit')
                encoded = db.execute('SELECT email FROM messages WHERE key=?', (key,)).fetchone()[0]
                try:
                    email = json.loads(encoded)
                except (TypeError, ValueError):
                    raise MailSearchError('invalid_stored_document') from None
                _update(db, key, email)
                indexed += 1
            db.execute("INSERT OR REPLACE INTO mail_search_meta VALUES ('version',?)", (INDEX_VERSION,))
        return {'version': INDEX_VERSION, 'rebuilt': True, 'indexed_count': indexed}
    except sqlite3.Error as exc:
        raise MailSearchError('index_write_failed') from exc


def update(db, key, email_or_None):
    """Replace one document atomically; None removes all of its indexed terms."""
    try:
        with _atomic(db):
            _update(db, key, email_or_None)
    except sqlite3.Error as exc:
        raise MailSearchError('index_write_failed') from exc


def _ready(db):
    try:
        row = db.execute("SELECT value FROM mail_search_meta WHERE key='version'").fetchone()
    except sqlite3.OperationalError:
        raise MailSearchError('index_not_ready') from None
    if row is None or row[0] != INDEX_VERSION:
        raise MailSearchError('index_not_ready')


def _visibility():
    # Invalid flag JSON fails closed rather than accidentally showing deleted mail.
    safe = "CASE WHEN json_valid(m.flags) THEN m.flags ELSE 'null' END"
    sql = ("m.present=1 AND m.status='parsed' AND json_type(" + safe + ")='array' "
           "AND NOT EXISTS (SELECT 1 FROM json_each(" + safe + ") WHERE value=? COLLATE NOCASE)")
    return sql, ['\\Deleted'], safe


def status(db):
    """Report present parsed coverage without initializing or rebuilding anything."""
    try:
        _ready(db)
    except MailSearchError:
        return {'status': 'not_ready', 'version': INDEX_VERSION, 'indexed_count': 0}
    visible, parameters, _ = _visibility()
    try:
        total = db.execute('SELECT count(*) FROM messages m WHERE ' + visible, parameters).fetchone()[0]
        indexed = db.execute('SELECT count(*) FROM mail_search_documents d JOIN messages m ON m.key=d.message_key WHERE ' + visible,
                             parameters).fetchone()[0]
    except sqlite3.Error:
        raise MailSearchError('index_status_failed') from None
    return {'status': 'ready' if indexed == total else 'incomplete', 'version': INDEX_VERSION,
            'indexed_count': indexed, 'eligible_count': total, 'missing_count': total - indexed,
            'scope': 'current_database_present_parsed_messages'}


def _snippet(values, pattern):
    # Prefer body/attachment evidence for the displayed excerpt over a header hit.
    for field in ('body', 'attachments', 'subject', 'sender', 'recipients'):
        original = values[field]
        match = pattern.search(original)
        if match is not None:
            start = max(0, match.start() - SNIPPET_CHARS // 3)
            end = min(len(original), start + SNIPPET_CHARS)
            return original[start:end], field, start, end
    text = values['body'] or values['attachments'] or values['subject']
    return text[:SNIPPET_CHARS], 'preview', 0, min(len(text), SNIPPET_CHARS)


def search(db, query, *, folders=None, unread_only=False, starred_only=False, limit=20):
    """Literal terms, FTS5 BM25, and current mailbox state; no full-scan fallback."""
    if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY_CHARS:
        raise MailSearchError('invalid_query')
    if type(limit) is not int or not 1 <= limit <= MAX_RESULTS:
        raise MailSearchError('invalid_result_limit')
    if type(unread_only) is not bool or type(starred_only) is not bool:
        raise MailSearchError('invalid_flag_filter')
    if folders is not None and (not isinstance(folders, (list, tuple)) or not 1 <= len(folders) <= 100
            or any(not isinstance(folder, str) or not folder or len(folder) > 1024 for folder in folders)):
        raise MailSearchError('invalid_folder_filter')
    _ready(db)
    started = time.monotonic()
    deadline = started + SEARCH_SECONDS
    terms = tokens(query)
    if len(terms) > MAX_QUERY_TERMS:
        raise MailSearchError('query_term_limit')
    diagnostics = {'index_version': INDEX_VERSION, 'query_mode': 'literal_all_terms', 'query_term_count': len(terms),
                   'result_limit': limit, 'time_limit_ms': int(SEARCH_SECONDS * 1000),
                   'local_only': True, 'fallback_scan': False, 'truncated': False}
    if not terms:
        return {'items': [], 'diagnostics': {**diagnostics, 'reason': 'no_searchable_terms', 'elapsed_ms': 0}}
    expression = ' AND '.join('"' + term.replace('"', '""') + '"' for term in terms)
    visible, parameters, safe_flags = _visibility()
    if folders is not None:
        visible += ' AND m.folder IN (' + ','.join('?' for _ in folders) + ')'
        parameters.extend('INBOX' if folder.upper() == 'INBOX' else folder for folder in folders)
    if unread_only:
        visible += ' AND NOT EXISTS (SELECT 1 FROM json_each(' + safe_flags + ') WHERE value=? COLLATE NOCASE)'
        parameters.append('\\Seen')
    if starred_only:
        visible += ' AND EXISTS (SELECT 1 FROM json_each(' + safe_flags + ') WHERE value=? COLLATE NOCASE)'
        parameters.append('\\Flagged')
    sql = '''SELECT m.key,m.folder,d.subject,d.sender,d.recipients,d.body,d.attachments,d.date,
             bm25(mail_search_fts,5.0,2.0,1.0,1.0,1.0) AS rank
             FROM mail_search_fts JOIN mail_search_documents d ON d.rowid=mail_search_fts.rowid
             JOIN messages m ON m.key=d.message_key WHERE mail_search_fts MATCH ? AND ''' + visible + ' ORDER BY rank,m.key LIMIT ?'
    pattern = re.compile('|'.join(re.escape(term) for term in sorted(terms, key=len, reverse=True)), re.IGNORECASE)
    items, scanned_chars = [], 0
    db.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
    try:
        cursor = db.execute(sql, (expression, *parameters, limit + 1))
        for row in cursor:
            if len(items) >= limit:
                diagnostics['truncated'] = True
                break
            if time.monotonic() > deadline:
                diagnostics.update(truncated=True, reason='search_time_limit')
                break
            key, folder, subject, sender, recipients, body, attachments, date, rank = tuple(row)
            values = dict(zip(_FIELDS, (subject, sender, recipients, body, attachments)))
            cost = sum(map(len, values.values()))
            if scanned_chars + cost > MAX_RESULT_SCAN_CHARS:
                diagnostics.update(truncated=True, reason='result_text_budget')
                break
            scanned_chars += cost
            snippet, field, start, end = _snippet(values, pattern)
            items.append({'message_key': key, 'folder': folder, 'subject': subject[:1000], 'sender': sender[:500], 'date': date,
                          'display_headers_truncated': len(subject) > 1000 or len(sender) > 500,
                          'snippet': snippet, 'snippet_field': field, 'snippet_start': start, 'snippet_end': end,
                          'snippet_format': 'plain_text', 'score': -rank})
    except sqlite3.OperationalError as exc:
        if 'interrupt' in str(exc).lower():
            diagnostics.update(truncated=True, reason='search_time_limit')
        elif 'no such module' in str(exc).lower():
            raise MailSearchError('fts_unavailable') from None
        else:
            raise MailSearchError('fts_query_failed') from None
    finally:
        db.set_progress_handler(None, 0)
    diagnostics.update(elapsed_ms=round((time.monotonic() - started) * 1000, 3),
                       returned_count=len(items), scanned_result_chars=scanned_chars)
    return {'items': items, 'diagnostics': diagnostics}
