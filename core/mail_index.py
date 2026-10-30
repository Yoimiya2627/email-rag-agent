"""Publish a mailbox snapshot into the single-owner AI corpus, using local embeddings."""
import json
import sqlite3
import tempfile
import time
from pathlib import Path

import config.settings as cfg
from agents.runtime import current_run, remaining_timeout
from core import index_manifest
from core.ingestion import prepare_email_chunks
from core.index_generation import build_index
from core.index_metrics import collect_index_metrics
from models.schemas import EmailChunk


def _save(store, value):
    with store.db() as db:
        db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)', ('ai_index', json.dumps(value)))


def index_status(store):
    with store.db() as db:
        row = db.execute("SELECT value FROM meta WHERE key='ai_index'").fetchone()
    return status_from_record(json.loads(row[0]) if row else {'state': 'not_started'})


def status_from_record(value):
    value = {**value, 'enabled': cfg.MAIL_AI_INDEX_ENABLED, 'llm_calls': 0}
    if not cfg.MAIL_AI_INDEX_ENABLED:
        return {**value, 'state': 'disabled'}
    if value.get('state') == 'ready':
        try:
            active = index_manifest.read_active_manifest()
            if value.get('generation') != (active or {}).get('generation'):
                value['state'] = 'needs_sync'
        except (OSError, ValueError):
            value['state'] = 'unavailable'
    return value


def _retained_chunks(owner, account_id):
    # Iterated by build_index while its cross-process writer lock is held.
    # This prevents losing another writer's completed import between read/publish.
    from core.embedder import _get_collection
    collection = _get_collection()
    count = collection.count()
    for offset in range(0, count, 256):
        remaining_timeout(60)
        page = collection.get(include=['documents', 'metadatas'], limit=min(256, count-offset), offset=offset)
        if not page['ids'] or not len(page['ids']) == len(page['documents']) == len(page['metadatas']):
            raise ValueError('Cannot enumerate retained index records')
        for cid, text, meta in zip(page['ids'], page['documents'], page['metadatas']):
            if meta.get('mail_index_owner') not in (None, owner):
                raise PermissionError('Index belongs to another mailbox owner')
            if meta.get('mail_index_owner') == owner and meta.get('mail_index_account_id') == account_id:
                continue
            yield EmailChunk(chunk_id=cid, email_id=meta['email_id'], chunk_index=meta['chunk_index'],
                             content=text, metadata={k:v for k,v in meta.items() if k!='index_generation'})


def index_mailbox(owner, store):
    if owner != cfg.API_OWNER_ID:
        raise PermissionError('Mailbox indexing requires the configured API owner')
    from core.mail_accounts import MailAccountStore
    MailAccountStore(cfg.MAIL_ACCOUNTS_PATH).get(owner, store.account_id)
    if not cfg.MAIL_AI_INDEX_ENABLED:
        return {'enabled': False, 'state': 'disabled', 'llm_calls': 0}
    with store.sync_lock():
        _save(store, {'state': 'running', 'updated': time.time()})
        emails = None
        try:
            # Keep the mailbox snapshot stable until publication, including a
            # concurrent manual sync. The private spool is bounded by ingestion.
            with tempfile.TemporaryDirectory(prefix='ai-index-', dir=store.root) as temporary:
                path = Path(temporary)/'emails.json'
                count = 0
                with sqlite3.connect(store.path.resolve().as_uri()+'?mode=ro', uri=True) as db, path.open('w', encoding='utf-8') as target:
                    if [r[0] for r in db.execute('SELECT account_id FROM binding')] != [store.account_id]:
                        raise ValueError('Mailbox binding mismatch')
                    target.write('[')
                    for raw, flags in db.execute("SELECT email,flags FROM messages WHERE present=1 AND status='parsed' ORDER BY key"):
                        remaining_timeout(60)
                        if '\\Deleted' in json.loads(flags):
                            continue
                        record = json.loads(raw)
                        source = record.get('source') or {}
                        if source.get('provider') != 'imap' or source.get('account_id') != store.account_id:
                            raise ValueError('Mailbox source binding mismatch')
                        if count:
                            target.write(',')
                        json.dump(record, target, ensure_ascii=False)
                        count += 1
                        if count > cfg.MAX_INDEX_INPUT_EMAILS or target.tell() > cfg.MAX_INDEX_INPUT_BYTES:
                            raise ValueError('Mailbox snapshot exceeds input budget')
                    target.write(']')
                run = current_run()
                if run:
                    run.progress('imap_indexing', email_count=count)
                with collect_index_metrics() as metrics:
                    if count:
                        emails, incoming = prepare_email_chunks(path)
                    else:
                        incoming = []
                    class Snapshot:
                        config_fingerprint = getattr(incoming, 'config_fingerprint', None)
                        options = getattr(incoming, 'options', None)

                        def __iter__(self):
                            yield from _retained_chunks(owner, store.account_id)
                            for chunk in incoming:
                                chunk.metadata.update(mail_index_owner=owner, mail_index_account_id=store.account_id)
                                yield chunk
                    published = {}
                    build_index(Snapshot(), replace=True, allow_empty=True, on_complete=published.update)
                value = {'state': 'ready', 'enabled': True, 'email_count': count,
                         'chunk_count': len(incoming), 'generation': published['generation'],
                         'updated': time.time(), 'llm_calls': 0, 'index_metrics': metrics.to_dict(),
                         'scope': 'present_parsed_local_mailbox_snapshot',
                         'attachments': 'metadata_and_extraction_coverage; body_index_only'}
                _save(store, value)
                if metrics.outcome != 'unchanged':
                    from core.retriever import invalidate_bm25_cache
                    invalidate_bm25_cache()
                return status_from_record(value)
        except BaseException as exc:
            _save(store, {'state': 'error', 'updated': time.time(), 'error_type': type(exc).__name__,
                          'message': '本地邮件已保留，AI 索引更新未完成；下次同步会重试。'})
            raise
        finally:
            if emails is not None:
                emails.close()
