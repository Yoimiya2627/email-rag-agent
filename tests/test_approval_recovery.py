"""Draft-only approval phases and persistent manual recovery, with fake Gmail."""
import threading
import json
import time
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

from agents.approvals import ApprovalStore
from agents.mail_providers import GmailDraftProvider, MailProviderError, MailProviderPreconditionError
import agents.mail_providers as providers
from tests.mail_binding_helpers import write_gmail_token, bound_payload


PAYLOAD = {'to': ['test@example.com'], 'subject': 'test', 'body': 'test'}


@pytest.fixture
def ready_gmail(monkeypatch, tmp_path):
    token = write_gmail_token(tmp_path / 'token.json')
    monkeypatch.setattr(providers.cfg, 'GMAIL_TOKEN_PATH', str(token))
    return bound_payload(GmailDraftProvider(), PAYLOAD)


class FakeGmail:
    def __init__(self, execute_fn=None):
        self.calls = 0
        self.execute_fn = execute_fn

    def users(self): return self
    def getProfile(self, **kwargs):
        return SimpleNamespace(execute=lambda: {'emailAddress': 'owner@example.test'})
    def drafts(self): return self
    def create(self, **kwargs): return self
    def execute(self):
        self.calls += 1
        return self.execute_fn() if self.execute_fn else {'id': 'draft-1'}


def test_construction_failure_is_not_started(tmp_path, ready_gmail):
    store = ApprovalStore(tmp_path / 'approval.sqlite3')
    item = store.create('send_email', ready_gmail)
    provider = GmailDraftProvider()
    with patch.object(provider, '_build_service', side_effect=MailProviderError('setup failure')):
        with pytest.raises(MailProviderPreconditionError):
            store.approve(item['approval_id'], executor=provider.execute_approval)
    final = store.get(item['approval_id'])
    assert (final['status'], final['execution_state']) == ('failed', 'not_started')


def test_worker_start_failure_is_not_started(tmp_path, ready_gmail):
    service=FakeGmail()
    provider=GmailDraftProvider(service=service)
    with patch('agents.mail_providers.threading.Thread.start',side_effect=RuntimeError('no thread')):
        with pytest.raises(MailProviderPreconditionError,match='could not start'):
            provider.execute_approval({'payload':ready_gmail})
    assert service.calls == 0


def test_real_style_wrapped_transport_receives_remaining_budget(ready_gmail):
    service=FakeGmail()
    service.http=SimpleNamespace(http=SimpleNamespace(timeout=999))
    result=GmailDraftProvider(service=service,timeout_seconds=1).execute_approval({'payload':ready_gmail})
    assert result['sent'] is False and 0 < service.http.http.timeout <= 1


def test_remaining_budget_never_exceeds_limit_after_clock_rounding(ready_gmail):
    service = FakeGmail()
    service.http = SimpleNamespace(http=SimpleNamespace(timeout=999))
    # Binary float rounding makes (now + 1) - now slightly greater than 1.
    with patch('agents.mail_providers.time.monotonic', return_value=16383.9):
        result = GmailDraftProvider(service=service, timeout_seconds=1).execute_approval({'payload': ready_gmail})
    assert result['sent'] is False and 0 < service.http.http.timeout <= 1


def test_timeout_before_submission_prevents_late_execute(tmp_path, ready_gmail):
    release, setup_done = threading.Event(), threading.Event()
    service = FakeGmail()
    def slow_setup(**kwargs):
        release.wait(1)
        setup_done.set()
        return service
    provider = GmailDraftProvider(timeout_seconds=0.02)
    with patch.object(provider, '_build_service', side_effect=slow_setup):
        try:
            with pytest.raises(MailProviderPreconditionError, match='before submission'):
                provider.execute_approval({'payload': ready_gmail})
        finally:
            release.set()
            assert setup_done.wait(1)
    # Give the worker a chance to reach the guarded submission point.
    time.sleep(0.03)
    assert service.calls == 0


def test_timeout_after_submission_is_unknown_and_never_replayed(tmp_path, ready_gmail):
    release, completed = threading.Event(), threading.Event()
    def slow_execute():
        release.wait(1)
        completed.set()
        return {'id': 'late-draft'}
    service = FakeGmail(slow_execute)
    provider = GmailDraftProvider(service=service, timeout_seconds=0.02)
    store = ApprovalStore(tmp_path / 'approval.sqlite3')
    item = store.create('send_email', ready_gmail)
    try:
        with pytest.raises(MailProviderError, match='after submission'):
            store.approve(item['approval_id'], executor=provider.execute_approval)
        assert store.get(item['approval_id'])['status'] == 'unknown'
        with pytest.raises(ValueError, match='unknown'):
            store.approve(item['approval_id'], executor=provider.execute_approval)
    finally:
        release.set()
        assert completed.wait(1)
    assert service.calls == 1


def uncertain(store):
    item = store.create('send_email', PAYLOAD)
    with pytest.raises(TimeoutError):
        store.approve(item['approval_id'], executor=lambda _: (_ for _ in ()).throw(TimeoutError()))
    return item


def test_manual_reconcile_requires_owner_content_and_evidence(tmp_path):
    store = ApprovalStore(tmp_path / 'approval.sqlite3')
    item = uncertain(store)
    kwargs = dict(outcome='not_executed', evidence='owner checked no in-flight request',
                  expected_payload_hash=item['payload_hash'], execution_stopped=True)
    with pytest.raises(PermissionError):
        store.reconcile(item['approval_id'], owner_id='other', **kwargs)
    with pytest.raises(ValueError, match='reviewed content'):
        store.reconcile(item['approval_id'], **{**kwargs, 'expected_payload_hash': 'changed'})
    with pytest.raises(ValueError, match='execution has stopped'):
        store.reconcile(item['approval_id'], **{**kwargs, 'execution_stopped': False})
    with pytest.raises(ValueError, match='evidence'):
        store.reconcile(item['approval_id'], **{**kwargs, 'evidence': ''})
    final = store.reconcile(item['approval_id'], reviewer='local', **kwargs)
    assert final['status'] == 'failed'
    reloaded = ApprovalStore(store.path)
    with pytest.raises(ValueError, match='failed'):
        reloaded.approve(item['approval_id'], executor=lambda _: pytest.fail('must not replay'))
    audit = reloaded.reconciliation_history(item['approval_id'])
    assert len(audit) == 1 and audit[0]['outcome'] == 'not_executed' and audit[0]['reviewer'] == 'local'


def test_unresolved_can_be_followed_by_verified_draft_success(tmp_path):
    store = ApprovalStore(tmp_path / 'approval.sqlite3')
    item = uncertain(store)
    kwargs = {'expected_payload_hash': item['payload_hash']}
    still = store.reconcile(item['approval_id'], 'unresolved', 'not enough evidence', **kwargs)
    assert still['status'] == 'unknown'
    with pytest.raises(ValueError, match='draft_id'):
        store.reconcile(item['approval_id'], 'succeeded', 'checked', **kwargs)
    final = store.reconcile(item['approval_id'], 'succeeded', 'owner found matching remote draft',
                            result={'draft_id': 'verified-1', 'sent': False}, **kwargs)
    assert final['status'] == 'approved' and final['result']['draft_id'] == 'verified-1'
    replay = store.approve(item['approval_id'], executor=lambda _: pytest.fail('must not replay'))
    assert replay['result']['draft_id'] == 'verified-1'
    assert len(store.reconciliation_history(item['approval_id'])) == 2


@pytest.mark.parametrize('timeout', [0, -1, float('nan'), float('inf'), True])
def test_provider_budget_must_be_positive_finite(timeout):
    with pytest.raises(MailProviderPreconditionError):
        GmailDraftProvider(timeout_seconds=timeout)


def test_ready_client_uses_static_discovery_and_nonrefreshing_token(monkeypatch, tmp_path):
    observed = {}
    class Credentials:
        def __init__(self, **kwargs): observed['token_copy'] = kwargs
        @classmethod
        def from_authorized_user_info(cls, info, scopes):
            return SimpleNamespace(valid=True, token='synthetic-access-token')
    def fake_module(name, **attrs):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    fake_module('httplib2', Http=lambda **kw:observed.setdefault('http', kw))
    fake_module('google.oauth2.credentials', Credentials=Credentials)
    fake_module('google_auth_httplib2', AuthorizedHttp=lambda *a,**kw:observed.setdefault('authorized',kw))
    fake_module('googleapiclient.discovery', build=lambda *a,**kw:observed.setdefault('build',kw))
    token = tmp_path / 'synthetic.json'
    token.write_text('{}', encoding='utf-8')
    provider = GmailDraftProvider(token_path=token)
    provider._build_service(timeout_seconds=1.25)
    assert observed['http']['timeout'] == 1.25
    assert observed['authorized']['max_refresh_attempts'] == 0
    assert set(observed['token_copy']) == {'token','scopes'}
    assert observed['build']['static_discovery'] is True and observed['build']['cache_discovery'] is False


def test_invalid_token_fails_without_interactive_authorization(monkeypatch, tmp_path):
    class Credentials:
        @classmethod
        def from_authorized_user_info(cls, info, scopes):
            return SimpleNamespace(valid=False)
    for name, attrs in [('httplib2',{}), ('google.oauth2.credentials',{'Credentials':Credentials}),
                        ('google_auth_httplib2',{'AuthorizedHttp':lambda *a,**kw:pytest.fail('no http')}),
                        ('googleapiclient.discovery',{'build':lambda *a,**kw:pytest.fail('no discovery')})]:
        module=ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules,name,module)
    token=tmp_path/'synthetic.json'
    token.write_text('{}',encoding='utf-8')
    provider=GmailDraftProvider(token_path=token)
    with pytest.raises(MailProviderPreconditionError,match='refresh locally'):
        provider._build_service(timeout_seconds=1)


def test_not_executed_cannot_contradict_observed_draft(tmp_path):
    store=ApprovalStore(tmp_path/'approval.sqlite3')
    item=store.create('send_email',PAYLOAD)
    with patch.object(store,'_complete',side_effect=OSError('synthetic storage failure')):
        with pytest.raises(OSError):
            store.approve(item['approval_id'],executor=lambda _:{'draft_id':'observed','sent':False})
    with pytest.raises(ValueError,match='already observed'):
        store.reconcile(item['approval_id'],'not_executed','checked',expected_payload_hash=item['payload_hash'],
                        execution_stopped=True)


def test_explicit_local_setup_refreshes_and_atomically_persists_only_synthetic_token(monkeypatch, tmp_path):
    token=tmp_path/'synthetic-token.json'
    token.write_text('{}',encoding='utf-8')
    events=[]
    creds=SimpleNamespace(valid=False,expired=True,refresh_token='synthetic',
                          refresh=lambda request:events.append('refresh'),
                          to_json=lambda:'{"synthetic":true}')
    for name, attrs in [('google.auth.transport.requests',{'Request':lambda:object()}),
                        ('google.oauth2.credentials',{'Credentials':SimpleNamespace(from_authorized_user_file=lambda *a:creds)}),
                        ('google_auth_oauthlib.flow',{'InstalledAppFlow':SimpleNamespace(
                            from_client_secrets_file=lambda *a:pytest.fail('valid refresh path must not open browser'))})]:
        module=ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules,name,module)
    monkeypatch.setattr(providers.cfg,'GMAIL_TOKEN_PATH',str(token))
    monkeypatch.setattr(providers.cfg,'GMAIL_CREDENTIALS_PATH',str(tmp_path/'unused-client.json'))
    monkeypatch.setattr(GmailDraftProvider, '_build_service', lambda *a, **kw: FakeGmail())
    providers.authorize_gmail_drafts_locally()
    saved = json.loads(token.read_text(encoding='utf-8'))
    assert events == ['refresh'] and saved['synthetic'] is True
    assert saved['_email_agent_binding']['account_id'] == 'owner@example.test'
    assert len(saved['_email_agent_binding']['authorization_id']) == 32
    assert list(tmp_path.glob('.gmail-token-*')) == []
