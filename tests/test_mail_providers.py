"""Tests for real mail-provider execution behind human approval."""

import base64
from types import SimpleNamespace
from tests.mail_binding_helpers import write_gmail_token, bound_payload

import pytest


class FakeCreateCall:
    def __init__(self, sink):
        self.sink = sink

    def execute(self):
        return {"id": "draft-123", "message": {"id": "msg-456"}}


class FakeDrafts:
    def __init__(self, sink):
        self.sink = sink

    def create(self, userId, body):
        self.sink["userId"] = userId
        self.sink["body"] = body
        return FakeCreateCall(self.sink)


class FakeUsers:
    def __init__(self, sink):
        self.sink = sink

    def drafts(self):
        return FakeDrafts(self.sink)

    def getProfile(self, **kwargs):
        return SimpleNamespace(execute=lambda: {'emailAddress': 'owner@example.test'})


class FakeGmailService:
    def __init__(self):
        self.sink = {}

    def users(self):
        return FakeUsers(self.sink)


def test_build_gmail_raw_message_contains_headers_and_body():
    from agents.mail_providers import build_gmail_raw_message

    raw = build_gmail_raw_message(
        {
            "to": ["alice@example.com", "bob@example.com"],
            "subject": "Budget",
            "body": "确认收到",
        }
    )

    decoded = base64.urlsafe_b64decode(raw.encode("utf-8")).decode("utf-8")
    assert "To: alice@example.com, bob@example.com" in decoded
    assert "Subject: Budget" in decoded
    assert "确认收到" in decoded


def test_gmail_provider_creates_draft_with_fake_service(tmp_path):
    from agents.mail_providers import GmailDraftProvider

    service = FakeGmailService()
    provider = GmailDraftProvider(service=service, user_id="me", token_path=write_gmail_token(tmp_path / 'token.json'))
    out = provider.execute_approval(
        {
            "approval_id": "approval-1",
            "payload": bound_payload(provider, {
                "to": ["alice@example.com"],
                "subject": "Budget",
                "body": "确认收到",
            }),
        }
    )

    assert out["mode"] == "gmail_draft"
    assert out["provider"] == "gmail"
    assert out["draft_id"] == "draft-123"
    assert out["message_id"] == "msg-456"
    assert out["sent"] is False
    assert "raw" in service.sink["body"]["message"]


def test_gmail_provider_requires_recipient():
    from agents.mail_providers import GmailDraftProvider, MailProviderError

    provider = GmailDraftProvider(service=FakeGmailService())

    with pytest.raises(MailProviderError, match="recipient"):
        provider.execute_approval({"payload": {"to": [], "subject": "x", "body": "y"}})


def test_simulated_provider_never_claims_real_send():
    from agents.mail_providers import SimulatedMailProvider

    result = SimulatedMailProvider().execute_approval({
        "request_id": "stable-request", "payload": {"to": ["alice@example.com"], "subject": "Hi", "body": "Hello"}
    })
    assert result["sent"] is False
    assert result["simulated"] is True
    assert result["request_id"] == "stable-request"


@pytest.mark.parametrize("payload", [
    {"to": "alice@example.com", "subject": "Hi", "body": "Hello"},
    {"to": [12], "subject": "Hi", "body": "Hello"},
    {"to": ["alice@example.com\nBcc: hidden@example.com"], "subject": "Hi", "body": "Hello"},
    {"to": ["alice@example.com"], "subject": "Hi\r\nBcc: hidden@example.com", "body": "Hello"},
])
def test_invalid_payload_fails_before_provider_call(payload):
    from agents.mail_providers import GmailDraftProvider, MailProviderPreconditionError

    service = FakeGmailService()
    with pytest.raises(MailProviderPreconditionError):
        GmailDraftProvider(service=service).execute_approval({"payload": payload})
    assert not service.sink
