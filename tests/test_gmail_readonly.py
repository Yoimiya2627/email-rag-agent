from __future__ import annotations

import base64
import json
from pathlib import Path


def _b64url(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("utf-8").rstrip("=")


def _b64url_bytes(payload: bytes) -> str:
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


class FakeExecute:
    def __init__(self, payload):
        self.payload = payload

    def execute(self):
        return self.payload


class FakeMessages:
    def __init__(self, pages, messages, calls):
        self.pages = pages
        self.messages = messages
        self.calls = calls

    def list(self, **kwargs):
        self.calls.append(("list", kwargs))
        token = kwargs.get("pageToken") or "first"
        return FakeExecute(self.pages[token])

    def get(self, **kwargs):
        self.calls.append(("get", kwargs))
        return FakeExecute(self.messages[kwargs["id"]])


class FakeUsers:
    def __init__(self, pages, messages, calls):
        self.pages = pages
        self._messages = messages
        self.calls = calls

    def messages(self):
        return FakeMessages(self.pages, self._messages, self.calls)


class FakeGmailReadService:
    def __init__(self, pages, messages):
        self.pages = pages
        self.messages = messages
        self.calls = []

    def users(self):
        return FakeUsers(self.pages, self.messages, self.calls)


class FakeHttpResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeAuthorizedSession:
    def __init__(self, pages, messages):
        self.pages = pages
        self.messages = messages
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params, timeout))
        params = params or {}
        if url.endswith("/messages"):
            token = params.get("pageToken") or "first"
            return FakeHttpResponse(self.pages[token])
        message_id = url.rsplit("/", 1)[-1]
        return FakeHttpResponse(self.messages[message_id])


def _message(message_id: str, body: str = "plain body") -> dict:
    return {
        "id": message_id,
        "threadId": "thread-1",
        "labelIds": ["INBOX", "IMPORTANT"],
        "internalDate": "1710000000000",
        "payload": {
            "mimeType": "multipart/alternative",
            "headers": [
                {"name": "Subject", "value": "Budget update"},
                {"name": "From", "value": "Alice <alice@example.com>"},
                {"name": "To", "value": "Bob <bob@example.com>, carol@example.com"},
                {"name": "Date", "value": "Mon, 01 Jan 2024 10:00:00 +0000"},
            ],
            "parts": [
                {"mimeType": "text/html", "body": {"data": _b64url("<b>html</b>")}},
                {"mimeType": "text/plain", "body": {"data": _b64url(body)}},
            ],
        },
    }


def test_gmail_readonly_provider_lists_pages_and_fetches_messages():
    from agents.gmail_readonly import GmailReadOnlyProvider

    service = FakeGmailReadService(
        pages={
            "first": {"messages": [{"id": "m1"}], "nextPageToken": "next"},
            "next": {"messages": [{"id": "m2"}]},
        },
        messages={"m1": _message("m1"), "m2": _message("m2", body="second")},
    )
    provider = GmailReadOnlyProvider(service=service, user_id="me")

    message_ids = provider.list_message_ids(query="newer_than:7d", max_results=10)
    emails = [provider.get_email(message_id) for message_id in message_ids]

    assert message_ids == ["m1", "m2"]
    assert emails[0].id == "gmail_m1"
    assert emails[0].subject == "Budget update"
    assert emails[0].sender == "alice@example.com"
    assert emails[0].recipients == ["bob@example.com", "carol@example.com"]
    assert emails[0].date == "2024-01-01T10:00:00+00:00"
    assert emails[0].body == "plain body"
    assert emails[0].labels == ["INBOX", "IMPORTANT"]
    assert emails[0].thread_id == "thread-1"
    assert service.calls[0] == (
        "list",
        {
            "userId": "me",
            "q": "newer_than:7d",
            "maxResults": 10,
            "includeSpamTrash": False,
        },
    )


def test_gmail_readonly_provider_uses_authorized_session_for_real_api_calls():
    from agents.gmail_readonly import GmailReadOnlyProvider

    session = FakeAuthorizedSession(
        pages={
            "first": {"messages": [{"id": "m1"}], "nextPageToken": "next"},
            "next": {"messages": [{"id": "m2"}]},
        },
        messages={"m1": _message("m1"), "m2": _message("m2", body="second")},
    )
    provider = GmailReadOnlyProvider(session=session, user_id="me", request_timeout=12)

    message_ids = provider.list_message_ids(query="newer_than:7d", max_results=10)
    email = provider.get_email(message_ids[0])

    assert message_ids == ["m1", "m2"]
    assert email.id == "gmail_m1"
    assert session.calls[0] == (
        "https://gmail.googleapis.com/gmail/v1/users/me/messages",
        {"q": "newer_than:7d", "maxResults": 10, "includeSpamTrash": False},
        12,
    )
    assert session.calls[1] == (
        "https://gmail.googleapis.com/gmail/v1/users/me/messages",
        {
            "q": "newer_than:7d",
            "maxResults": 10,
            "includeSpamTrash": False,
            "pageToken": "next",
        },
        12,
    )
    assert session.calls[2] == (
        "https://gmail.googleapis.com/gmail/v1/users/me/messages/m1",
        {"format": "full"},
        12,
    )


def test_gmail_readonly_provider_falls_back_to_html_body():
    from agents.gmail_readonly import gmail_message_to_email

    raw = _message("m1")
    raw["payload"]["parts"] = [
        {"mimeType": "text/html", "body": {"data": _b64url("<p>Hello&nbsp;<b>team</b></p>")}},
    ]

    email = gmail_message_to_email(raw)

    assert email.body == "Hello team"


def test_gmail_message_to_email_decodes_part_charset_from_content_type():
    from agents.gmail_readonly import gmail_message_to_email

    text = "中文通知：预算已经批准"
    raw = _message("m1")
    raw["payload"]["parts"] = [
        {
            "mimeType": "text/plain",
            "headers": [{"name": "Content-Type", "value": "text/plain; charset=gb18030"}],
            "body": {"data": _b64url_bytes(text.encode("gb18030"))},
        }
    ]

    email = gmail_message_to_email(raw)

    assert email.body == text


def test_sync_gmail_to_json_skips_seen_messages_and_persists_state(tmp_path: Path):
    from agents.gmail_readonly import GmailReadOnlyProvider
    from scripts.sync_gmail_readonly import sync_gmail_to_json

    output_path = tmp_path / "emails.json"
    state_path = tmp_path / "state.json"
    output_path.write_text(
        json.dumps(
            [
                {
                    "id": "gmail_m1",
                    "subject": "Existing",
                    "sender": "old@example.com",
                    "recipients": ["bob@example.com"],
                    "date": "2024-01-01T00:00:00+00:00",
                    "body": "old",
                    "labels": [],
                    "thread_id": "t0",
                }
            ]
        ),
        encoding="utf-8",
    )
    state_path.write_text(
        json.dumps({"seen_message_ids": ["m1"], "last_internal_date_ms": 1}),
        encoding="utf-8",
    )
    service = FakeGmailReadService(
        pages={"first": {"messages": [{"id": "m1"}, {"id": "m2"}]}},
        messages={"m1": _message("m1"), "m2": _message("m2", body="new body")},
    )
    provider = GmailReadOnlyProvider(service=service)

    result = sync_gmail_to_json(
        provider=provider,
        output_path=output_path,
        state_path=state_path,
        query="newer_than:30d",
        max_results=10,
    )

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert result == {"fetched": 2, "added": 1, "skipped": 1, "total": 2}
    assert [item["id"] for item in saved] == ["gmail_m1", "gmail_m2"]
    assert state["seen_message_ids"] == ["m1", "m2"]
    assert state["last_internal_date_ms"] == 1710000000000
