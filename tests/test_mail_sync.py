"""Sync coverage uses actual local commits and controlled IMAP snapshots."""
import hashlib
from unittest.mock import patch

import pytest

from agents.imap_readonly import ImapReadError, ImapMessageTooLarge
from agents.runtime import RunContext, RunCancelled, use_run_context
from core.imap_mime import parse_imap_message, parser_version
from core.mail_sync import MailSyncStore, ImapSyncError, sync_mailbox


RAW = b'Subject: Synthetic\r\nFrom: a@example.test\r\nDate: Sat, 12 Sep 2026 08:00:00 +0000\r\nContent-Type: text/plain; charset=utf-8\r\n\r\nSynthetic local-only body'


class Provider:
    def __init__(self, folders=None):
        self.folders=folders or {'INBOX':('1',[1,2,3])}
        self.selected=None
        self.calls=[]
        self.errors={}

    def list_folders(self):
        return [{'name':name,'selectable':True} for name in self.folders]

    def select_folder(self,name):
        self.selected=name
        validity,uids=self.folders[name]
        return {'uidvalidity':validity,'exists':len(uids)}

    def list_uids(self):
        return self.folders[self.selected][1]

    def fetch_message(self,uid):
        self.calls.append((self.selected,uid))
        if uid in self.errors:
            raise self.errors[uid]
        return {'uid':uid,'raw':RAW,'size':len(RAW),'flags':[], 'internal_date':'12-Sep-2026 08:00:00 +0000'}


def perform(provider, store, **kwargs):
    return sync_mailbox(provider,store,kwargs.pop('folders',['INBOX']),parser=parse_imap_message,**kwargs)


def test_bounded_sync_continues_older_mail_without_duplicate_records(tmp_path):
    store=MailSyncStore(tmp_path,'account-a')
    provider=Provider()
    first=perform(provider,store,max_messages=2)
    assert first['parsed']==2 and first['not_downloaded']==1
    assert provider.calls==[('INBOX',3),('INBOX',2)]
    second=perform(provider,store,max_messages=2)
    assert second['parsed']==3 and second['not_downloaded']==0
    assert provider.calls==[('INBOX',3),('INBOX',2),('INBOX',1)]
    third=perform(provider,store)
    assert third['last_run']['attempted']==0
    assert store.messages()['total']==3
    assert third['model_calls']==0
    assert len(list((tmp_path/'raw').glob('*.eml')))==1


def test_uidvalidity_reset_and_folder_collision_preserve_raw_and_history(tmp_path):
    store=MailSyncStore(tmp_path,'account-a')
    provider=Provider({'INBOX':('1',[1]),'Archive':('1',[1])})
    perform(provider,store,folders=['INBOX','Archive'])
    original=store.messages()['items']
    assert len({row['id'] for row in original})==2
    provider.folders['INBOX']=('2',[1])
    result=perform(provider,store,folders=['INBOX','Archive'])
    assert result['parsed']==2 and result['historical_records']==1
    assert len({row['id'] for row in store.messages()['items']})==2
    old=next(row for row in original if row['folder']=='INBOX')
    assert store.message(old['key'])['present']==0
    assert store.raw_path(store.message(old['key'])['raw_sha256']).read_bytes()==RAW


@pytest.mark.parametrize('error',[ImapReadError('incomplete_message'),ImapReadError('unexpected_uid'),ImapMessageTooLarge()])
def test_bad_message_does_not_starve_older_mail_and_failure_is_visible(tmp_path,error):
    provider=Provider()
    provider.errors[3]=error
    store=MailSyncStore(tmp_path,'account-a')
    first=perform(provider,store,max_messages=1)
    assert first['failed']==1 and first['not_downloaded']==2
    second=perform(provider,store,max_messages=2)
    assert second['parsed']==2 and second['failed']==1 and second['not_downloaded']==0
    assert provider.calls==[('INBOX',3),('INBOX',2),('INBOX',1)]
    assert store.messages(failures_only=True)['items'][0]['error_code']==error.code


def test_connection_failure_stops_and_does_not_advance_missing_mail(tmp_path):
    provider=Provider()
    provider.errors[2]=ImapReadError('connection_failed')
    store=MailSyncStore(tmp_path,'account-a')
    with pytest.raises(ImapReadError,match='connection_failed'):
        perform(provider,store)
    assert store.report()['parsed']==1
    provider.errors.clear()
    perform(provider,store)
    assert provider.calls==[('INBOX',3),('INBOX',2),('INBOX',2),('INBOX',1)]
    assert store.report()['parsed']==3


def test_cancellation_keeps_committed_record_then_can_continue(tmp_path):
    import threading
    event=threading.Event()
    checkpoints=[]
    def checkpoint(value):
        checkpoints.append(value)
        event.set()
    store=MailSyncStore(tmp_path,'account-a')
    provider=Provider()
    with use_run_context(RunContext(cancel_event=event,checkpoint_callback=checkpoint)):
        with pytest.raises(RunCancelled):
            perform(provider,store)
    assert store.report()['parsed']==1 and checkpoints[0]['safe']
    perform(provider,store)
    assert store.report()['parsed']==3 and len(provider.calls)==3


def test_parse_failure_archives_exact_source_and_is_retryable(tmp_path):
    store=MailSyncStore(tmp_path,'account-a')
    provider=Provider({'INBOX':('1',[1])})
    from core.imap_mime import MailParseError
    with patch('core.mail_sync.isolated_parse',side_effect=MailParseError('synthetic_invalid_mime')):
        result=sync_mailbox(provider,store,['INBOX'])
    assert result['failed']==1
    row=store.messages()['items'][0]
    assert row['raw_sha256']==hashlib.sha256(RAW).hexdigest()
    assert store.raw_path(row['raw_sha256']).read_bytes()==RAW
    result=perform(provider,store)
    assert result['parsed']==1 and result['failed']==0


def test_parser_upgrade_reprocesses_and_version_is_from_current_parser(tmp_path):
    provider=Provider({'INBOX':('1',[1])})
    store=MailSyncStore(tmp_path,'account-a')
    perform(provider,store)
    with store.db() as db:
        db.execute("UPDATE messages SET parser_version='old-parser'")
    perform(provider,store)
    assert len(provider.calls)==2
    assert store.messages()['items'][0]['parser_version']==parser_version()


def test_database_binding_and_concurrent_sync_lock(tmp_path):
    store=MailSyncStore(tmp_path,'account-a')
    with pytest.raises(ImapSyncError,match='account_binding_mismatch'):
        MailSyncStore(tmp_path,'account-b')
    with store.sync_lock():
        with pytest.raises(ImapSyncError,match='account_sync_busy'):
            perform(Provider(),store)


def test_failure_attempt_limit_requires_explicit_retry_and_does_not_hide_failure(tmp_path):
    store=MailSyncStore(tmp_path,'account-a')
    provider=Provider({'INBOX':('1',[1])})
    provider.errors[1]=ImapMessageTooLarge()
    for _ in range(4):
        result=perform(provider,store)
    assert len(provider.calls)==3 and result['failed']==1
    perform(provider,store,retry_failed=True)
    assert len(provider.calls)==4


def test_capture_commit_failure_can_recover_without_false_success(tmp_path):
    store=MailSyncStore(tmp_path,'account-a')
    with store.db() as db:
        db.execute("CREATE TRIGGER injected BEFORE INSERT ON messages BEGIN SELECT RAISE(ABORT,'test'); END")
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        perform(Provider(),store)
    assert store.report()['parsed']==0 and store.report()['not_downloaded']==3
    assert list((tmp_path/'raw').glob('*.eml'))
    with store.db() as db:
        db.execute('DROP TRIGGER injected')
    perform(Provider(),store)
    assert store.report()['parsed']==3


def test_scoped_report_marks_removed_uid_historical_without_remote_delete(tmp_path):
    store=MailSyncStore(tmp_path,'account-a')
    provider=Provider()
    perform(provider,store)
    provider.folders['INBOX']=('1',[2,3])
    result=perform(provider,store)
    assert result['parsed']==2 and result['historical_records']==1


def test_binding_check_stops_before_new_message(tmp_path):
    store=MailSyncStore(tmp_path,'account-a')
    provider=Provider()
    def binding():
        if provider.calls:
            raise ImapSyncError('credential_binding_changed')
    with pytest.raises(ImapSyncError,match='credential_binding_changed'):
        perform(provider,store,check_binding=binding)
    assert provider.calls==[('INBOX',3)] and store.report()['parsed']==1


def test_progress_matches_committed_counts_and_resets_on_noop_sync(tmp_path):
    store=MailSyncStore(tmp_path,'account-a')
    provider=Provider()
    provider.errors[2]=ImapReadError('incomplete_message')
    progress=[]
    with use_run_context(RunContext(progress_callback=lambda stage, **value: progress.append((stage,value)))):
        result=perform(provider,store)
        assert progress[-1]==('imap_complete',{'attempted':3,'parsed':2,'failed':1})
        assert result['last_run']['attempted']==3
        # A finished job with no eligible work must not retain old counters.
        with store.db() as db:
            db.execute('UPDATE messages SET attempts=3')
        perform(provider,store)
        assert progress[-1]==('imap_complete',{'attempted':0,'parsed':0,'failed':0})


def test_verified_server_size_mismatch_is_retained_as_transport_metadata(tmp_path):
    store=MailSyncStore(tmp_path,'account-a')
    provider=Provider({'INBOX':('1',[1])})
    original=provider.fetch_message
    provider.fetch_message=lambda uid: {**original(uid),'reported_size':len(RAW)+4,'size_mismatch':True}
    assert perform(provider,store)['parsed']==1
    row=store.messages()['items'][0]
    source=store.message(row['key'])['email']['source']
    assert source['transport']=={'received_size':len(RAW),'reported_size':len(RAW)+4,
                                 'size_mismatch_verified':True}


def test_report_only_requests_review_for_attachments_with_incomplete_coverage(tmp_path):
    store=MailSyncStore(tmp_path,'account-a')
    email=parse_imap_message(RAW,account_id='account-a',folder='INBOX',uidvalidity='1',uid=1)
    email.attachments=[{'status':'complete','reason':'text_extracted'},
                       {'status':'partial','reason':'decode_or_extraction_warning'}]
    store.save('INBOX','1',1,raw=RAW,email=email)
    report=store.report()
    assert report['attachments']=={'complete':1,'partial':1}
    assert report['issues']=={'attachment:decode_or_extraction_warning':1}
