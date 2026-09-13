"""Mailbox status reads only scoped metadata and never invents a live connection."""
import json
import sqlite3
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest

import api.main as api
import config.settings as cfg
import core.mailbox_status as status
from api.sessions import SessionStore
from core.jobs import JobStore
from core.mail_accounts import MailAccountStore, get_data_dir
from core.mail_schedule import MailScheduleStore
from core.mail_sync import MailSyncStore
from tests.test_general_chat import streamed
from tests.test_mailbox_routes import FakeProtector


@pytest.fixture
def env(tmp_path_factory, monkeypatch):
    root = tmp_path_factory.mktemp('ms')
    for name, filename in [('MAIL_ACCOUNTS_PATH','accounts.db'), ('MAIL_SCHEDULE_PATH','schedule.db'),
                           ('IMAP_DATA_ROOT','imap'), ('JOB_STORE_PATH','jobs.db')]:
        monkeypatch.setattr(cfg, name, str(root/filename))
    monkeypatch.setattr(cfg, 'API_OWNER_ID', 'owner-a')
    monkeypatch.setattr(cfg, 'MAIL_SCHEDULER_ENABLED', True)
    accounts = MailAccountStore(cfg.MAIL_ACCOUNTS_PATH, protector=FakeProtector())
    first = accounts.upsert('owner-a', 'first@163.com', 'synthetic-code')
    second = accounts.upsert('owner-a', 'second@163.com', 'synthetic-code')
    foreign = accounts.upsert('owner-b', 'foreign@163.com', 'synthetic-code')
    boxes = {}
    for account, owner, count in [(first,'owner-a',5), (second,'owner-a',12), (foreign,'owner-b',77)]:
        store = MailSyncStore(get_data_dir(cfg.IMAP_DATA_ROOT, owner, account['id']), account['id'])
        # Use the real schema, but no parser, IMAP, model or real mail fixture.
        with store.db() as db:
            db.execute('INSERT INTO folders VALUES (?,?,?,?)', ('INBOX','1',count+3,1000))
            for uid in range(count+1):
                db.execute('''INSERT INTO messages (key,folder,uidvalidity,uid,status,email,updated)
                              VALUES (?,?,?,?,?,?,?)''', (str(uid),'INBOX','1',uid,
                              'parsed' if uid<count else 'failed', '{"body":"PRIVATE-BODY"}',1000))
            db.execute('INSERT INTO meta VALUES (?,?)', ('last_run',json.dumps({'finished':1000,'failed':1})))
        boxes[account['id']] = store
    schedule = MailScheduleStore(cfg.MAIL_SCHEDULE_PATH)
    schedule.upsert('owner-a',first['id'],enabled=True,folders=['INBOX'])
    jobs = JobStore(cfg.JOB_STORE_PATH)
    sessions = SessionStore(path=root/'sessions.db')
    monkeypatch.setattr(api, 'sessions', sessions)
    return SimpleNamespace(root=root,first=first,second=second,foreign=foreign,accounts=accounts,
                           boxes=boxes,schedule=schedule,jobs=jobs,sessions=sessions,client=TestClient(api.app))


def read(env, account=None, owner='owner-a', provider=None):
    return status.read_mailbox_status(owner, account or env.first['id'], provider)


def add_job(env, state, *, account=None, owner='owner-a', version=1, updated=2000):
    job,_ = env.jobs.create(owner,'imap_sync',{'account_id':account or env.first['id'], 'credential_version':version})
    with env.jobs._db() as db:
        db.execute('UPDATE jobs SET status=?,updated=? WHERE id=?',(state,updated,job['id']))
    return job


def test_status_does_not_read_credentials_or_email_columns_or_write(env, monkeypatch):
    add_job(env,'succeeded')
    original = status._read_db
    denied = []
    @contextmanager
    def guarded(path):
        with original(path) as db:
            def authorize(action, table, column, database, trigger):
                secret_read = action == sqlite3.SQLITE_READ and (
                    (table == 'mail_accounts' and column in {'address','display_name','encrypted_code'})
                    or (table == 'messages' and column == 'email'))
                mutation = action in {sqlite3.SQLITE_INSERT,sqlite3.SQLITE_UPDATE,sqlite3.SQLITE_DELETE,
                                      sqlite3.SQLITE_CREATE_TABLE,sqlite3.SQLITE_ALTER_TABLE}
                if secret_read or mutation:
                    denied.append((action,table,column))
                    return sqlite3.SQLITE_DENY
                return sqlite3.SQLITE_OK
            db.set_authorizer(authorize)
            yield db
    monkeypatch.setattr(status,'_read_db',guarded)
    monkeypatch.setattr(MailAccountStore,'credentials',Mock(side_effect=AssertionError('No credentials')))
    monkeypatch.setattr(MailSyncStore,'__init__',Mock(side_effect=AssertionError('No migrating store')))
    value = read(env)
    assert denied == []
    assert value['local_sync']['parsed'] == 5 and value['local_sync']['failed'] == 1
    assert value['local_sync']['not_downloaded'] == 2
    assert value['schedule']['enabled'] and value['latest_job']['state'] == 'succeeded'
    text = json.dumps(value)+status.format_mailbox_status(value)
    assert not any(word in text for word in ['PRIVATE-BODY','synthetic-code','first@163.com'])
    assert value['live_connection_verified'] is False and value['ai_read_enabled'] is False


def test_account_selection_owner_and_provider_never_fall_back(env):
    assert read(env,env.second['id'])['local_sync']['parsed'] == 12
    assert read(env,env.foreign['id'])['account_state'] == 'not_accessible'
    assert read(env,'f'*32)['account_state'] == 'not_accessible'
    assert status.read_mailbox_status('owner-a')['account_state'] == 'selection_required'
    assert status.read_mailbox_status('owner-b')['local_sync']['parsed'] == 77
    assert status.read_mailbox_status('nobody')['account_state'] == 'not_configured'
    for provider, state in [('qq','provider_mismatch'),('multiple','provider_selection_required')]:
        value = read(env,provider=provider)
        assert value['account_state'] == state and 'local_sync' not in value


def test_missing_storage_does_not_create_files_or_invent_counts(env, monkeypatch):
    monkeypatch.setattr(cfg,'MAIL_ACCOUNTS_PATH',str(env.root/'absent'/'accounts.db'))
    value = status.read_mailbox_status('owner-a')
    assert value['account_state'] == 'not_configured'
    assert not (env.root/'absent').exists()
    assert '0 封' not in status.format_mailbox_status(value)


def test_failed_recent_attempt_does_not_hide_behind_old_completed_sync(env):
    add_job(env,'succeeded',updated=1100)
    add_job(env,'failed',updated=1200)
    # Another owner/account's more recent job cannot override this account.
    add_job(env,'running',account=env.foreign['id'],owner='owner-b',updated=1300)
    value = read(env)
    answer = status.format_mailbox_status(value)
    assert value['latest_job']['state'] == 'failed'
    assert value['local_sync']['last_completed'] == {'finished':1000,'failed':1}
    assert '任务失败' in answer and '其中 1 封处理失败' in answer
    assert '未进行实时连接测试' in answer and '已连接' not in answer


def test_changed_credentials_do_not_reuse_old_task_as_proof(env):
    add_job(env,'succeeded')
    env.accounts.upsert('owner-a','first@163.com','new-synthetic-code')
    value = read(env)
    assert value['latest_job']['current_authorization'] is False
    assert '不能证明当前授权有效' in status.format_mailbox_status(value)


@pytest.mark.parametrize('worker,failures,expected', [
    (True,0,'5 分钟'),(False,0,'后台同步服务未启用'),
    (True,2,'等待重试'),(True,8,'连续失败已暂停'),
])
def test_schedule_settings_are_not_connection_proof(env,monkeypatch,worker,failures,expected):
    monkeypatch.setattr(cfg,'MAIL_SCHEDULER_ENABLED',worker)
    with env.schedule._db() as db:
        db.execute('UPDATE mail_schedules SET failure_count=?',(failures,))
    assert expected in status.format_mailbox_status(read(env))


def test_valid_empty_sync_and_unavailable_metadata_are_distinct(env):
    with env.boxes[env.first['id']].db() as db:
        db.execute('DELETE FROM messages')
        db.execute('UPDATE folders SET remote_count=0')
        db.execute("UPDATE meta SET value=? WHERE key='last_run'",(json.dumps({'finished':1000,'failed':0,'parsed':0}),))
    value = read(env)
    assert value['local_sync']['state'] == 'available' and value['local_sync']['parsed'] == 0
    assert '本地已解析 0 封' in status.format_mailbox_status(value)
    with env.boxes[env.first['id']].db() as db:
        db.execute("UPDATE meta SET value='corrupt' WHERE key='last_run'")
        db.execute('UPDATE binding SET account_id=?',('e'*32,))
    value = read(env)
    assert value['local_sync']['state'] == 'unavailable'
    assert '不能确认邮件数量' in status.format_mailbox_status(value)


def test_status_endpoint_rejects_other_owners_and_refetches_changes(env):
    path = '/mailboxes/'+env.first['id']+'/status'
    assert env.client.get(path).json()['local_sync']['parsed'] == 5
    with env.boxes[env.first['id']].db() as db:
        db.execute("UPDATE messages SET status='parsed' WHERE status='failed'")
    assert env.client.get(path).json()['local_sync']['parsed'] == 6
    assert env.client.get('/mailboxes/'+env.foreign['id']+'/status').status_code == 404


@pytest.mark.parametrize('endpoint',['/chat','/chat/stream','/chat/graph','/chat/agent','/jobs/agent'])
@pytest.mark.parametrize('query',['你现在连接上了163了吗？','你现在可以看到我163的邮件吗？','我的邮箱有多少封邮件？'])
def test_all_chat_modes_use_scoped_local_status_without_model_or_derived_context(env,monkeypatch,endpoint,query):
    with env.sessions.turn('owner-a','status-session') as memory:
        memory.append_turn('旧邮件历史','SENSITIVE-HISTORY')
    forbidden = Mock(side_effect=AssertionError('Status must not read history, models or mail bodies'))
    monkeypatch.setattr(api,'_prepare_session_context',forbidden)
    monkeypatch.setattr('core.pipeline.retrieve',forbidden)
    monkeypatch.setattr('core.model_clients.create_completion',forbidden)
    monkeypatch.setattr(MailAccountStore,'credentials',forbidden)
    monkeypatch.setattr(MailSyncStore,'report',forbidden)
    monkeypatch.setattr(api,'_jobs',SimpleNamespace(submit=lambda *args:env.jobs.create(*args)[0]))
    response = env.client.post(endpoint,json={'query':query,'session_id':'status-session',
        'mailbox_account_id':env.second['id'], 'context':{'owner_id':'owner-b','account_id':env.foreign['id']}})
    assert response.status_code == (202 if endpoint == '/jobs/agent' else 200)
    if endpoint == '/jobs/agent':
        from threading import Event
        job = env.jobs.get('owner-a',response.json()['id'],private=True)
        assert job['request']['mailbox_account_id'] == env.second['id']
        result = api._run_background_job(job,Event(),lambda *a,**kw:None,lambda *a,**kw:None)
    else:
        result = streamed(response) if endpoint.endswith('/stream') else response.json()
        if endpoint.endswith('/stream'):
            events = [json.loads(line[6:]) for line in response.text.splitlines()
                      if line.startswith('data: ') and line != 'data: [DONE]']
            result['intent'] = next(event['intent'] for event in events if 'intent' in event)
    assert result['intent'] == 'mailbox_status' and result['sources'] == []
    assert '12 封' in result['answer'] and 'SENSITIVE-HISTORY' not in result['answer']
    assert result['metadata']['mailbox_status']['local_sync']['parsed'] == 12
    assert result['metadata']['model_usage']['call_count'] == 0
    assert result['metadata']['model_visible_evidence'] == []
    assert result['metadata']['exclude_from_model_context'] is True
    assert env.sessions.history('owner-a','status-session')['turns'][-1]['include_in_context'] is False
    forbidden.assert_not_called()
