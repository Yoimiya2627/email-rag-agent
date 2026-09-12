import json
import sys
from types import ModuleType
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

from agents.approvals import ApprovalStore
from agents.mail_providers import GmailDraftProvider, MailProviderPreconditionError, SimulatedMailProvider
from tests.mail_binding_helpers import write_gmail_token, bound_payload


class Service:
    def __init__(self, account='owner@example.test', profile=None):
        self.account, self.profile_callback = account, profile
        self.profile_calls, self.draft_calls = 0, 0
        self.http = NS(http=NS(timeout=999))

    def users(self): return self
    def drafts(self): return self
    def getProfile(self, **kwargs):
        assert kwargs == {'userId': 'me'}
        def execute():
            self.profile_calls += 1
            if self.profile_callback:
                self.profile_callback()
            return {'emailAddress': self.account}
        return NS(execute=execute, http=self.http)

    def create(self, **kwargs):
        def execute():
            self.draft_calls += 1
            return {'id': 'draft-bound', 'message': {'id': 'message-bound'}}
        return NS(execute=execute, http=self.http)


@pytest.fixture
def setup(monkeypatch, tmp_path):
    import config.settings as cfg
    token = write_gmail_token(tmp_path / 'token.json')
    monkeypatch.setattr(cfg, 'MAIL_PROVIDER', 'gmail')
    monkeypatch.setattr(cfg, 'GMAIL_USER_ID', 'me')
    monkeypatch.setattr(cfg, 'GMAIL_TOKEN_PATH', str(token))
    monkeypatch.setattr(cfg, 'APPROVAL_STORE_PATH', str(tmp_path / 'approval.sqlite3'))
    service = Service()
    monkeypatch.setattr(GmailDraftProvider, '_build_service', lambda self, **kwargs: service)
    return cfg, token, service


def pending():
    from agents.tools import send_email
    return send_email(['recipient@example.test'], 'Quote', 'Please check.', 'User requested a draft')['approval_id']


def approve(identifier):
    import api.main as api
    return TestClient(api.app).post('/agent/approvals/' + identifier + '/approve', json={})


@pytest.mark.parametrize('change', ['account', 'token', 'grant', 'selector', 'scope', 'provider', 'missing'])
def test_pending_approval_refuses_account_or_authorization_change(setup, monkeypatch, change):
    cfg, token, service = setup
    identifier = pending()
    if change == 'account':
        write_gmail_token(token, account='other@example.test')
    elif change == 'token':
        write_gmail_token(token, token='another-synthetic-token')
    elif change == 'grant':
        write_gmail_token(token, grant='b' * 32)
    elif change == 'selector':
        monkeypatch.setattr(cfg, 'GMAIL_USER_ID', 'owner@example.test')
    elif change == 'scope':
        monkeypatch.setattr(cfg, 'GMAIL_SCOPES', ['changed-scope'])
    elif change == 'provider':
        monkeypatch.setattr(cfg, 'MAIL_PROVIDER', 'simulated')
    else:
        token.unlink()
    assert approve(identifier).status_code == 409
    result = ApprovalStore().get(identifier)
    assert (result['status'], result['execution_state']) == ('failed', 'not_started')
    assert service.draft_calls == 0 and service.profile_calls == 0


def test_verified_account_binding_persists_and_completed_action_is_not_replayed(setup):
    cfg, token, service = setup
    identifier = pending()
    item = ApprovalStore().get(identifier)
    binding = item['payload']['execution_binding']
    assert binding['account_id'] == 'owner@example.test'
    assert binding['provider'] == 'gmail'
    assert 'synthetic-access-token' not in json.dumps(item)
    assert approve(identifier).status_code == 200
    assert (service.profile_calls, service.draft_calls) == (1, 1)
    write_gmail_token(token, account='other@example.test')
    assert approve(identifier).status_code == 200
    assert service.draft_calls == 1


@pytest.mark.parametrize('account', ['wrong@example.test', '', None])
def test_profile_mismatch_or_invalid_identity_stops_before_mutation(setup, account):
    cfg, token, service = setup
    identifier = pending()
    service.account = account
    assert approve(identifier).status_code == 409
    assert service.profile_calls == 1 and service.draft_calls == 0


def test_authorization_changed_during_profile_read_stops_before_mutation(setup):
    cfg, token, service = setup
    identifier = pending()
    service.profile_callback = lambda: write_gmail_token(token, grant='b' * 32)
    assert approve(identifier).status_code == 409
    assert service.draft_calls == 0


def test_unbound_legacy_approval_cannot_create_a_gmail_draft(setup):
    cfg, token, service = setup
    item = ApprovalStore().create('send_email', {'to': ['x@example.test'], 'subject': 'Hi', 'body': 'Hi'})
    assert approve(item['approval_id']).status_code == 409
    assert service.profile_calls == service.draft_calls == 0


def test_old_token_without_verified_binding_requires_explicit_setup(setup):
    cfg, token, service = setup
    info = json.loads(token.read_text())
    del info['_email_agent_binding']
    token.write_text(json.dumps(info))
    with pytest.raises(MailProviderPreconditionError, match='binding is missing'):
        pending()
    assert ApprovalStore().list() == []
    assert service.profile_calls == 0


def test_profile_failure_is_not_an_unknown_mutation(setup):
    cfg, token, service = setup
    identifier = pending()
    def fail(): raise TimeoutError('synthetic read timeout')
    service.profile_callback = fail
    assert approve(identifier).status_code == 409
    assert ApprovalStore().get(identifier)['execution_state'] == 'not_started'
    assert service.draft_calls == 0


@pytest.mark.parametrize('payload', [None, [], 'invalid'])
def test_simulated_invalid_payload_remains_a_precondition(payload):
    with pytest.raises(MailProviderPreconditionError):
        SimulatedMailProvider().execute_approval({'payload': payload})


def local_setup_modules(monkeypatch, info):
    creds = NS(valid=True, to_json=lambda: json.dumps(info))
    modules = {
        'google.auth.transport.requests': {'Request': lambda: object()},
        'google.oauth2.credentials': {'Credentials': NS(from_authorized_user_file=lambda *a: creds)},
        'google_auth_oauthlib.flow': {'InstalledAppFlow': NS(
            from_client_secrets_file=lambda *a: pytest.fail('ready token must not open a browser'))},
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)


def test_explicit_setup_upgrades_legacy_token_and_rotates_grant(setup, monkeypatch):
    from agents.mail_providers import authorize_gmail_drafts_locally
    cfg, token, service = setup
    info = json.loads(token.read_text())
    del info['_email_agent_binding']
    token.write_text(json.dumps(info))
    local_setup_modules(monkeypatch, info)
    authorize_gmail_drafts_locally()
    first = json.loads(token.read_text())['_email_agent_binding']
    assert first['account_id'] == service.account
    identifier = pending()
    authorize_gmail_drafts_locally()
    second = json.loads(token.read_text())['_email_agent_binding']
    assert first['authorization_id'] != second['authorization_id']
    assert approve(identifier).status_code == 409
    assert service.profile_calls == 2 and service.draft_calls == 0
    assert list(token.parent.glob('.gmail-token-*')) == []


@pytest.mark.parametrize('failure', ['profile_error', 'invalid_account', 'wrong_selector'])
def test_local_setup_failure_never_replaces_existing_grant(setup, monkeypatch, failure):
    from agents.mail_providers import authorize_gmail_drafts_locally
    cfg, token, service = setup
    original = token.read_bytes()
    local_setup_modules(monkeypatch, json.loads(original))
    if failure == 'profile_error':
        def fail(): raise TimeoutError('synthetic')
        service.profile_callback = fail
    elif failure == 'invalid_account':
        service.account = ''
    else:
        monkeypatch.setattr(cfg, 'GMAIL_USER_ID', 'wrong@example.test')
    with pytest.raises((TimeoutError, MailProviderPreconditionError)):
        authorize_gmail_drafts_locally()
    assert token.read_bytes() == original
    assert service.draft_calls == 0
    assert list(token.parent.glob('.gmail-token-*')) == []


@pytest.mark.parametrize('data', [b'[]', b'{broken', b'x' * 65_537],
                         ids=['non_object', 'bad_json', 'oversized'])
def test_invalid_or_oversized_authorization_stops_before_provider(setup, data):
    cfg, token, service = setup
    token.write_bytes(data)
    with pytest.raises(MailProviderPreconditionError):
        pending()
    assert service.profile_calls == service.draft_calls == 0
