"""Exercise registered mailbox routes without network, models, or corpus writes."""
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest

from core.mail_accounts import MailAccountStore
from core.jobs import JobStore


class FakeProtector:
    def protect(self, data, entropy):
        return entropy + bytes(value ^ 0xA5 for value in data)

    def unprotect(self, data, entropy):
        assert data[:32] == entropy
        return bytes(value ^ 0xA5 for value in data[32:])


@pytest.fixture
def api_setup(tmp_path_factory, monkeypatch):
    import api.main as api
    import api.mailbox_routes as routes
    import config.settings as cfg
    import core.model_clients as clients
    import core.embedder as embedder

    # Leave room for owner/account/content hashes below Windows MAX_PATH.
    tmp_path = tmp_path_factory.mktemp('mb')
    accounts = MailAccountStore(tmp_path / 'accounts.sqlite3', protector=FakeProtector())
    jobs = JobStore(tmp_path / 'jobs.sqlite3')
    monkeypatch.setattr(routes, 'accounts', lambda: accounts)
    monkeypatch.setattr(cfg, 'IMAP_DATA_ROOT', str(tmp_path / 'mailboxes'))
    monkeypatch.setattr(cfg, 'API_OWNER_ID', 'owner-a')
    monkeypatch.setattr(api, '_jobs', SimpleNamespace(submit=lambda *args: jobs.create(*args)[0]))
    forbidden = []
    for module, name in [(clients, 'get_model_client'), (embedder, 'index_chunks'),
                         (embedder, 'clear_collection'), (api, 'index_chunks'),
                         (api, 'clear_collection'), (api, 'route'), (api, 'direct_general_response')]:
        guard = Mock(side_effect=AssertionError('local mailbox must not call models or index'))
        monkeypatch.setattr(module, name, guard)
        forbidden.append(guard)
    result = SimpleNamespace(client=TestClient(api.app), accounts=accounts, jobs=jobs,
                             routes=routes, cfg=cfg, guards=forbidden)
    yield result
    for guard in forbidden:
        guard.assert_not_called()


def configure(env, secret='synthetic-only-authorization-code'):
    response = env.client.post('/mailboxes', json={'address':'Fixture@163.COM',
                               'authorization_code':secret, 'display_name':'测试邮箱'})
    assert response.status_code == 200
    assert secret not in response.text
    return response.json()


def test_registered_routes_keep_accounts_isolated(api_setup, monkeypatch):
    env = api_setup
    account = configure(env)
    assert env.client.get('/mailboxes').json() == {'accounts':[account], 'local_only':True}
    monkeypatch.setattr(env.cfg, 'API_OWNER_ID', 'owner-b')
    assert env.client.get('/mailboxes').json()['accounts'] == []
    for suffix in ('/report', '/messages', '/messages/missing', '/messages/missing/raw'):
        assert env.client.get('/mailboxes/'+account['id']+suffix).status_code == 404
    for suffix in ('/connect', '/sync'):
        assert env.client.post('/mailboxes/'+account['id']+suffix, json={}).status_code == 404
    other = configure(env)
    assert other['id'] != account['id']


def test_job_pins_server_account_and_version_without_secret(api_setup):
    env = api_setup
    secret = 'private-authorization-never-in-a-job'
    account = configure(env, secret)
    response = env.client.post('/mailboxes/'+account['id']+'/sync', json={
        'folders':['INBOX'], 'max_messages':7, 'operation_key':'synthetic-run',
        'account_id':'forged', 'credential_version':999, 'authorization_code':secret})
    assert response.status_code == 202
    assert secret not in response.text
    job = env.jobs.get('owner-a', response.json()['id'], private=True)
    assert job['kind'] == 'imap_sync'
    assert job['request'] == {'folders':['INBOX'], 'max_messages':7, 'retry_failed':False,
                              'account_id':account['id'], 'credential_version':1}
    assert secret.encode() not in env.jobs.path.read_bytes()


@pytest.mark.parametrize('payload', [
    {'address':'fixture@evil.test', 'authorization_code':'sensitive-shape-token'},
    {'address':'fixture@163.com', 'authorization_code':{'value':'sensitive-shape-token'}},
    {'address':'fixture@163.com', 'authorization_code':'sensitive-shape-token', 'host':'evil.test'},
    ['sensitive-shape-token'], 'sensitive-shape-token',
])
def test_invalid_configuration_never_echoes_secret(api_setup, payload):
    response = api_setup.client.post('/mailboxes', json=payload)
    assert response.status_code == 422
    assert 'sensitive-shape-token' not in response.text


def test_connection_error_never_echoes_provider_exception(api_setup, monkeypatch):
    env = api_setup
    account = configure(env)
    def failure(*args):
        raise RuntimeError('server reflected synthetic-only-authorization-code')
    monkeypatch.setattr(env.routes, 'provider_for', failure)
    response = env.client.post('/mailboxes/'+account['id']+'/connect')
    assert response.status_code == 502
    assert 'synthetic-only-authorization-code' not in response.text


class FakeProvider:
    def __init__(self):
        self.entered = False
        self.fetched = []

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *args):
        return False

    def describe_account(self):
        return {'provider':'163', 'read_only':True}

    def list_folders(self):
        return [{'name':'INBOX', 'display_name':'收件箱', 'selectable':True}]

    def select_folder(self, folder):
        return {'uidvalidity':'42'}

    def list_uids(self):
        return [1, 2]

    def fetch_message(self, uid):
        self.fetched.append(uid)
        return {'raw':b'Subject: Offline local fixture\r\nFrom: sender@example.test\r\n\r\nLocal mail only.',
                'flags':[], 'internal_date':''}


def test_connect_reports_read_only_folders(api_setup, monkeypatch):
    env = api_setup
    account = configure(env)
    provider = FakeProvider()
    monkeypatch.setattr(env.routes, 'provider_for', lambda owner, aid:(env.accounts.get(owner, aid), provider))
    response = env.client.post('/mailboxes/'+account['id']+'/connect')
    assert response.status_code == 200
    assert response.json()['read_only'] is True
    assert response.json()['local_only'] is True
    assert response.json()['folders'][0]['name'] == 'INBOX'


def submit_job(env, account):
    response = env.client.post('/mailboxes/'+account['id']+'/sync', json={'max_messages':2})
    assert response.status_code == 202
    return env.jobs.get('owner-a', response.json()['id'], private=True)


def test_rebound_credentials_stop_old_job_before_connection(api_setup, monkeypatch):
    env = api_setup
    account = configure(env)
    job = submit_job(env, account)
    env.accounts.upsert('owner-a', account['address'], 'rebound-code')
    provider = FakeProvider()
    monkeypatch.setattr(env.routes, 'provider_for', lambda owner, aid:(env.accounts.get(owner, aid), provider))
    with pytest.raises(env.routes.ImapSyncError, match='credential_binding_changed'):
        env.routes.run_mail_sync(job)
    assert provider.entered is False
    assert provider.fetched == []


def test_local_sync_report_message_and_raw_never_touch_models(api_setup, monkeypatch):
    import core.mail_sync as sync
    from core.imap_mime import parse_imap_message
    env = api_setup
    account = configure(env)
    provider = FakeProvider()
    monkeypatch.setattr(env.routes, 'provider_for', lambda owner, aid:(env.accounts.get(owner, aid), provider))
    monkeypatch.setattr(sync, 'isolated_parse', lambda path, locator, timeout:parse_imap_message(path.read_bytes(), **locator))
    result = env.routes.run_mail_sync(submit_job(env, account))
    assert result['parsed'] == 2
    assert result['model_calls'] == 0
    prefix = '/mailboxes/'+account['id']
    assert env.client.get(prefix+'/report').json()['parsed'] == 2
    page = env.client.get(prefix+'/messages').json()
    assert page['total'] == 2
    key = page['items'][0]['key']
    message = env.client.get(prefix+'/messages/'+key).json()
    assert message['email']['body'].strip() == 'Local mail only.'
    raw = env.client.get(prefix+'/messages/'+key+'/raw')
    assert raw.status_code == 200
    assert raw.headers['x-content-type-options'] == 'nosniff'
    assert raw.headers['content-type'] == 'application/octet-stream'
    assert 'attachment' in raw.headers['content-disposition']


def test_mid_sync_rebind_prevents_next_message_fetch(api_setup, monkeypatch):
    import core.mail_sync as sync
    from core.imap_mime import parse_imap_message
    env = api_setup
    account = configure(env)
    provider = FakeProvider()
    monkeypatch.setattr(env.routes, 'provider_for', lambda owner, aid:(env.accounts.get(owner, aid), provider))
    def parse_and_rebind(path, locator, timeout):
        parsed = parse_imap_message(path.read_bytes(), **locator)
        env.accounts.upsert('owner-a', account['address'], 'new-binding-code')
        return parsed
    monkeypatch.setattr(sync, 'isolated_parse', parse_and_rebind)
    with pytest.raises(env.routes.ImapSyncError, match='credential_binding_changed'):
        env.routes.run_mail_sync(submit_job(env, account))
    assert provider.fetched == [2]
    report = env.client.get('/mailboxes/'+account['id']+'/report').json()
    assert report['parsed'] == 1
    assert report['not_downloaded'] == 1
