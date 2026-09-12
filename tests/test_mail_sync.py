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
        self.flags={}

    def list_folders(self):
        return [{'name':name,'selectable':True} for name in self.folders]

    def select_folder(self,name):
        self.selected=name
        validity,uids=self.folders[name]
        return {'uidvalidity':validity,'exists':len(uids)}

    def list_uids(self):
        return self.folders[self.selected][1]

    def fetch_flags(self,uids):
        return {uid:self.flags.get((self.selected,uid),[]) for uid in uids}

    def fetch_message(self,uid):
        self.calls.append((self.selected,uid))
        if uid in self.errors:
            raise self.errors[uid]
        return {'uid':uid,'raw':RAW,'size':len(RAW),'flags':self.flags.get((self.selected,uid),[]), 'internal_date':'12-Sep-2026 08:00:00 +0000'}


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


def test_cancel_during_flags_scan_preserves_previous_snapshot(tmp_path):
    import threading
    store = MailSyncStore(tmp_path, 'account-a')
    provider = Provider()
    perform(provider, store)
    before = store.report()
    event = threading.Event()
    provider.folders['INBOX'] = ('1', [1])
    def cancelled_flags(uids):
        event.set()
        return {uid:['\\Seen'] for uid in uids}
    provider.fetch_flags = cancelled_flags
    with use_run_context(RunContext(cancel_event=event)):
        with pytest.raises(RunCancelled):
            perform(provider, store)
    assert store.report() == before
    assert store.messages()['total'] == 3
    assert all(not item['is_read'] for item in store.messages()['items'])


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


def test_status_refresh_updates_read_star_filters_without_body_download(tmp_path):
    store=MailSyncStore(tmp_path,'account-a')
    provider=Provider()
    perform(provider,store)
    calls=list(provider.calls)
    provider.flags[('INBOX',3)]=['\\seen','\\flagged']
    result=perform(provider,store)
    assert provider.calls==calls and result['last_run']['flags_updated']==1
    assert result['last_run']['fetched']==0 and result['remaining_eligible']==0
    assert len(store.search('Synthetic',unread_only=True)['items'])==2
    starred=store.search('Synthetic',starred_only=True)['items']
    assert len(starred)==1
    assert store.message(starred[0]['message_key'])['flags']==['\\Flagged','\\Seen']


def test_deleted_flag_then_undelete_restores_search_without_redownload(tmp_path):
    store=MailSyncStore(tmp_path,'account-a')
    provider=Provider({'INBOX':('1',[1])})
    perform(provider,store)
    provider.flags[('INBOX',1)]=['\\deleted']
    result=perform(provider,store)
    assert result['parsed']==0 and result['historical_records']==1
    assert result['last_run']['excluded_deleted']==1 and not store.search('Synthetic')['items']
    provider.flags.clear()
    result=perform(provider,store)
    assert result['parsed']==1 and len(store.search('Synthetic')['items'])==1
    assert len(provider.calls)==1


def test_remote_move_retires_old_location_and_indexes_new_folder(tmp_path):
    store=MailSyncStore(tmp_path,'account-a')
    provider=Provider({'INBOX':('1',[1]),'Archive':('1',[])})
    perform(provider,store,folders=['INBOX','Archive'])
    provider.folders={'INBOX':('1',[]),'Archive':('1',[7])}
    result=perform(provider,store,folders=['INBOX','Archive'])
    assert result['parsed']==1 and result['historical_records']==1
    items=store.search('Synthetic')['items']
    assert len(items)==1 and items[0]['folder']=='Archive'


def test_folder_removed_from_successful_list_retires_its_old_search_rows(tmp_path):
    store=MailSyncStore(tmp_path,'account-a')
    provider=Provider({'INBOX':('1',[1]),'Archive':('1',[7])})
    perform(provider,store,folders=['INBOX','Archive'])
    del provider.folders['Archive']
    result=perform(provider,store)
    assert result['parsed']==1 and result['historical_records']==1
    assert not store.search('Synthetic',folders=['Archive'])['items']


def test_partial_flag_response_cannot_hide_live_mail(tmp_path):
    store=MailSyncStore(tmp_path,'account-a')
    provider=Provider()
    perform(provider,store)
    provider.fetch_flags=lambda uids:{1:[],2:[]}
    with pytest.raises(ImapReadError,match='incomplete_flags_snapshot'):
        perform(provider,store)
    assert store.report()['parsed']==3 and len(store.search('Synthetic')['items'])==3


def test_expunged_during_flag_scan_is_confirmed_and_removed_locally(tmp_path):
    store=MailSyncStore(tmp_path,'account-a')
    provider=Provider()
    perform(provider,store)
    snapshots=iter([[1,2,3],[1,2]])
    provider.list_uids=lambda:next(snapshots)
    provider.fetch_flags=lambda uids:{1:[],2:[]}
    result=perform(provider,store)
    assert result['parsed']==2 and result['historical_records']==1


def test_single_message_batch_rotates_folders_and_exposes_backfill_remaining(tmp_path):
    store=MailSyncStore(tmp_path,'account-a')
    provider=Provider({'INBOX':('1',list(range(1,10))),'Archive':('1',[1,2])})
    for count in range(4):
        report=perform(provider,store,folders=['INBOX','Archive'],max_messages=1)
        assert report['remaining_eligible']==10-count
    assert provider.calls==[('INBOX',9),('Archive',2),('INBOX',8),('Archive',1)]


def test_uidvalidity_change_after_planning_cannot_bind_recycled_uid(tmp_path):
    store=MailSyncStore(tmp_path,'account-a')
    provider=Provider({'INBOX':('1',[1])})
    old=provider.select_folder
    calls=[]
    def select(folder):
        result=old(folder)
        calls.append(folder)
        return result if len(calls)==1 else {'uidvalidity':'2','exists':1}
    provider.select_folder=select
    with pytest.raises(ImapSyncError,match='mailbox_changed_during_sync'):
        perform(provider,store)
    assert not provider.calls and store.report()['parsed']==0


def test_backfilled_older_mail_does_not_displace_newer_mail_in_browse_order(tmp_path):
    store=MailSyncStore(tmp_path,'account-a')
    for uid,date in [(2,'2026-09-12T12:00:00+08:00'),(1,'2026-08-01T12:00:00+08:00')]:
        email=parse_imap_message(RAW,account_id='account-a',folder='INBOX',uidvalidity='1',uid=uid)
        email.date=date
        store.save('INBOX','1',uid,raw=RAW,email=email)
    assert [row['uid'] for row in store.messages()['items']]==[2,1]
