"""Real local index generations, synthetic MIME, deterministic vectors; no model/network."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock
import threading

import pytest
import config.settings as cfg
from core import embedder, index_manifest, mail_index
from core.mail_accounts import MailAccountStore
from core.mail_sync import MailSyncStore
from core.imap_mime import parse_imap_message
from tests.test_index_generations import store, chunk
from tests.test_mailbox_routes import FakeProtector


@pytest.fixture
def mailbox(store, tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, 'MAIL_AI_INDEX_ENABLED', True)
    monkeypatch.setattr(cfg, 'MAIL_ACCOUNTS_PATH', str(tmp_path/'accounts.sqlite3'))
    monkeypatch.setattr('core.mail_accounts.WindowsDPAPIProtector', FakeProtector)
    accounts = MailAccountStore(cfg.MAIL_ACCOUNTS_PATH)
    account = accounts.upsert(cfg.API_OWNER_ID, 'fixture@163.com', 'synthetic-code')
    monkeypatch.setattr(MailAccountStore, 'credentials', Mock(side_effect=AssertionError('No credentials needed')))
    return MailSyncStore(tmp_path/'mail', account['id'])


def save(box, uid=1, body='Realistic synthetic evidence', flags=None):
    raw = ('From: Sender <sender@example.invalid>\r\nTo: reader@example.invalid\r\n'
           'Subject: Synthetic mailbox sample\r\nDate: Mon, 14 Sep 2026 00:00:00 +0000\r\n'
           'Content-Type: text/plain; charset=utf-8\r\n\r\n'+body).encode()
    email = parse_imap_message(raw, account_id=box.account_id, folder='INBOX', uidvalidity='1', uid=uid, flags=flags or [])
    box.save('INBOX','1',uid,raw=raw,email=email)
    return email.id


def ids():
    return {row['metadata']['email_id'] for row in embedder.get_all_chunks()}


def test_preserves_imports_and_repeated_sync_reuses_generation(mailbox, monkeypatch):
    embedder.index_chunks([chunk('imported')])
    first, second = save(mailbox), save(mailbox,2,'Another message')
    result = mail_index.index_mailbox(cfg.API_OWNER_ID, mailbox)
    assert result['email_count']==2 and result['llm_calls']==0
    assert ids()=={'imported',first,second}
    assert mail_index.index_status(mailbox)['state']=='ready'
    generation=result['generation']
    monkeypatch.setattr(embedder,'embed_texts',Mock(side_effect=AssertionError('Unchanged needs no embeddings')))
    repeat=mail_index.index_mailbox(cfg.API_OWNER_ID, mailbox)
    assert repeat['generation']==generation and repeat['index_metrics']['outcome']=='unchanged'


def test_updates_content_removes_deleted_and_keeps_other_sources(mailbox):
    embedder.index_chunks([chunk('imported')])
    first, second = save(mailbox), save(mailbox,2,'Old second body')
    mail_index.index_mailbox(cfg.API_OWNER_ID,mailbox)
    mailbox.snapshot('INBOX','1',[2])
    save(mailbox,2,'Updated second body')
    mail_index.index_mailbox(cfg.API_OWNER_ID,mailbox)
    assert ids()=={'imported',second}
    assert 'Updated second body' in embedder.get_indexed_email(second)['body']
    save(mailbox,2,'Updated second body',flags=['\\Deleted'])
    mail_index.index_mailbox(cfg.API_OWNER_ID,mailbox)
    assert ids()=={'imported'}


def test_empty_mailbox_publishes_empty_then_is_idempotent(mailbox):
    save(mailbox)
    mail_index.index_mailbox(cfg.API_OWNER_ID,mailbox)
    mailbox.snapshot('INBOX','1',[])
    empty=mail_index.index_mailbox(cfg.API_OWNER_ID,mailbox)
    assert ids()==set() and empty['email_count']==0
    assert mail_index.index_mailbox(cfg.API_OWNER_ID,mailbox)['generation']==empty['generation']


def test_failure_retains_active_index_and_reports_no_private_error(mailbox,monkeypatch):
    identifier=save(mailbox)
    before=mail_index.index_mailbox(cfg.API_OWNER_ID,mailbox)['generation']
    save(mailbox,body='Modified evidence')
    monkeypatch.setattr(embedder,'embed_texts',Mock(side_effect=RuntimeError('PRIVATE ERROR BODY')))
    with pytest.raises(RuntimeError):mail_index.index_mailbox(cfg.API_OWNER_ID,mailbox)
    assert index_manifest.read_active_manifest()['generation']==before
    assert 'Realistic synthetic evidence' in embedder.get_indexed_email(identifier)['body']
    status=mail_index.index_status(mailbox)
    assert status['state']=='error' and 'PRIVATE' not in str(status)


def test_wrong_owner_never_writes_and_clear_invalidates_status(mailbox):
    save(mailbox)
    with pytest.raises(PermissionError):mail_index.index_mailbox('another-owner',mailbox)
    mail_index.index_mailbox(cfg.API_OWNER_ID,mailbox)
    embedder.clear_collection()
    assert mail_index.index_status(mailbox)['state']=='needs_sync'


def test_two_concurrent_mailboxes_preserve_both_and_imports(mailbox,tmp_path):
    account=MailAccountStore(cfg.MAIL_ACCOUNTS_PATH).upsert(cfg.API_OWNER_ID,'second@163.com','synthetic')
    second=MailSyncStore(tmp_path/'second',account['id'])
    first_id,second_id=save(mailbox),save(second)
    embedder.index_chunks([chunk('imported')])
    barrier=threading.Barrier(2)
    def run(box):
        barrier.wait(timeout=10)
        return mail_index.index_mailbox(cfg.API_OWNER_ID,box)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(run,[mailbox,second]))
    assert all(r['state']=='ready' for r in results)
    assert ids()=={'imported',first_id,second_id}


@pytest.mark.parametrize('scheduled',[False,True])
def test_manual_and_scheduled_sync_share_auto_index_hook(mailbox,monkeypatch,scheduled):
    import api.mailbox_routes as routes
    account=MailAccountStore(cfg.MAIL_ACCOUNTS_PATH).get(cfg.API_OWNER_ID,mailbox.account_id)
    provider=nullcontext()
    monkeypatch.setattr(routes,'provider_for',lambda *args:(account,provider))
    monkeypatch.setattr(routes,'mailbox',lambda *args:mailbox)
    monkeypatch.setattr(routes,'sync_mailbox',lambda *a,**k:{'last_run':{'fetched':0,'failed':0},'metadata':{'status':'success'}})
    save(mailbox)
    payload=routes.scheduled_request(cfg.API_OWNER_ID,mailbox.account_id,['INBOX'],10) if scheduled else {
        'account_id':mailbox.account_id,'credential_version':account['credential_version'],'folders':['INBOX'],'max_messages':10}
    result=routes.run_mail_sync({'owner':cfg.API_OWNER_ID,'request':payload})
    assert result['last_run']['fetched']==0 and result['ai_index']['email_count']==1


def test_failed_index_makes_sync_incomplete_and_can_retry(mailbox,monkeypatch):
    import api.mailbox_routes as routes
    account=MailAccountStore(cfg.MAIL_ACCOUNTS_PATH).get(cfg.API_OWNER_ID,mailbox.account_id)
    monkeypatch.setattr(routes,'provider_for',lambda *a:(account,nullcontext()))
    monkeypatch.setattr(routes,'mailbox',lambda *a:mailbox)
    monkeypatch.setattr(routes,'sync_mailbox',lambda *a,**k:{'last_run':{'fetched':0}})
    save(mailbox)
    monkeypatch.setattr(embedder,'embed_texts',Mock(side_effect=RuntimeError('private')))
    payload=routes.scheduled_request(cfg.API_OWNER_ID,mailbox.account_id,['INBOX'],10)
    result=routes.run_mail_sync({'owner':cfg.API_OWNER_ID,'request':payload})
    assert result['metadata']['completion_status']=='incomplete'
    assert result['ai_index']['state']=='error'


def test_concurrent_replacement_cannot_be_misreported_as_our_generation(mailbox,monkeypatch):
    save(mailbox)
    build=mail_index.build_index
    def replaced(*args,**kwargs):
        result=build(*args,**kwargs)
        embedder.clear_collection()
        return result
    monkeypatch.setattr(mail_index,'build_index',replaced)
    result=mail_index.index_mailbox(cfg.API_OWNER_ID,mailbox)
    assert result['generation']!=index_manifest.read_active_manifest()['generation']
    assert result['state']=='needs_sync' and mail_index.index_status(mailbox)['state']=='needs_sync'


def test_cancellation_retains_previous_generation_and_can_retry(mailbox,monkeypatch):
    from agents.runtime import RunContext,RunCancelled,use_run_context
    save(mailbox)
    before=mail_index.index_mailbox(cfg.API_OWNER_ID,mailbox)['generation']
    save(mailbox,body='Updated after cancellation')
    event=threading.Event()
    def encode(texts):
        event.set()
        return [[1.,0.] for text in texts]
    with monkeypatch.context() as patch:
        patch.setattr(embedder,'embed_texts',encode)
        with use_run_context(RunContext(cancel_event=event)),pytest.raises(RunCancelled):
            mail_index.index_mailbox(cfg.API_OWNER_ID,mailbox)
    assert index_manifest.read_active_manifest()['generation']==before
    assert mail_index.index_status(mailbox)['state']=='error'
    assert mail_index.index_mailbox(cfg.API_OWNER_ID,mailbox)['state']=='ready'


def test_prepared_chunk_configuration_must_match_publication(mailbox,monkeypatch):
    save(mailbox)
    prepare=mail_index.prepare_email_chunks
    def changed(*args):
        result=prepare(*args)
        monkeypatch.setattr(cfg,'CHUNK_SIZE',cfg.CHUNK_SIZE+1)
        return result
    monkeypatch.setattr(mail_index,'prepare_email_chunks',changed)
    with pytest.raises(index_manifest.IndexCompatibilityError):
        mail_index.index_mailbox(cfg.API_OWNER_ID,mailbox)
    assert index_manifest.read_active_manifest() is None
