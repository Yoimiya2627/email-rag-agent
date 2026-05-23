"""Tests for real mail-provider execution behind human approval."""

import base64

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


def test_gmail_provider_creates_draft_with_fake_service():
    from agents.mail_providers import GmailDraftProvider

    service = FakeGmailService()
    provider = GmailDraftProvider(service=service, user_id="me")
    out = provider.execute_approval(
        {
            "approval_id": "approval-1",
            "payload": {
                "to": ["alice@example.com"],
                "subject": "Budget",
                "body": "确认收到",
            },
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
