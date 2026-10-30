import base64
import hashlib
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import config.settings as cfg
from agents.gmail_readonly import GmailReadOnlyProvider, MailContentError, gmail_message_to_email
from agents.runtime import RunContext, RunCancelled, RunDeadlineExceeded, use_run_context
from core.filters import FilterSpec
from core.chunker import chunk_email
from core.pipeline import _retrieval_timezone
from scripts.sync_gmail_readonly import sync_gmail_to_json
from scripts.replay_gmail_captures import replay_captures


def part(text="plain", mime="text/plain", **extra):
    return {"mimeType": mime, "body": {"data": base64.urlsafe_b64encode(text.encode()).decode()}, **extra}


def mail(identifier="a", payload=None):
    payload = payload or part()
    payload.setdefault("headers", [{"name": "From", "value": "Alice <alice@example.com>"},
        {"name": "Subject", "value": "test"}, {"name": "Message-ID", "value": "<a@example.com>"},
        {"name": "In-Reply-To", "value": "<prior@example.com>"},
        {"name": "References", "value": "<older@example.com> <prior@example.com>"},
        {"name": "Cc", "value": "Carol <carol@example.com>"}])
    return {"id": identifier, "internalDate": "1770000000000", "threadId": "thread", "labelIds": ["Label_1"], "payload": payload}


@pytest.mark.parametrize("setting,value,payload,code", [
    ("MIME_DECODED_BYTE_LIMIT", 4, part("12345"), "mime_decoded_limit"),
    ("MIME_ENCODED_BYTE_LIMIT", 4, part("12345"), "mime_encoded_limit"),
    ("MIME_PART_LIMIT", 2, {"mimeType": "multipart/mixed", "parts": [part(), part()]}, "mime_part_limit"),
    ("MIME_DEPTH_LIMIT", 1, {"mimeType": "multipart/mixed", "parts": [{"mimeType": "multipart/mixed", "parts": [part()]}]}, "mime_depth_limit"),
    ("MIME_HEADER_CHAR_LIMIT", 3, part(), "mime_header_limit"),
    ("MIME_OUTPUT_CHAR_LIMIT", 3, part("12345"), "mime_output_limit"),
])
def test_message_wide_budgets_fail_explicitly_before_normalizing(monkeypatch, setting, value, payload, code):
    monkeypatch.setattr(cfg, setting, value, raising=False)
    with pytest.raises(MailContentError) as failure:
        gmail_message_to_email(mail(payload=payload))
    assert failure.value.code == code


def test_cumulative_body_budget_and_external_size_checked_before_fetch(monkeypatch):
    monkeypatch.setattr(cfg, "MIME_DECODED_BYTE_LIMIT", 8, raising=False)
    with pytest.raises(MailContentError, match="mime_decoded_limit"):
        gmail_message_to_email(mail(payload={"mimeType": "multipart/mixed", "parts": [part("12345"), part("67890")]}))
    fetch = Mock(side_effect=AssertionError("must reject declared size before downloading"))
    with pytest.raises(MailContentError, match="mime_decoded_limit"):
        gmail_message_to_email(mail(payload={"mimeType": "text/plain", "body": {"attachmentId": "external", "size": 999}}), fetch)
    fetch.assert_not_called()


def test_attachment_inventory_and_decoding_signals_keep_original_facts():
    text = "cafÃ© total 123.45 not approved \ufffd"
    payload = {"mimeType": "multipart/mixed", "parts": [part(text),
        {"mimeType": "application/pdf", "filename": "quote.pdf", "body": {"attachmentId": "pdf", "size": 123456}},
        {"mimeType": "message/rfc822", "body": {"attachmentId": "forward", "size": 500}}]}
    email = gmail_message_to_email(mail(payload=payload), Mock(side_effect=AssertionError("attachments not read")))
    assert email.body == text
    assert email.decode_quality["body"]["status"] == "suspect"
    assert set(email.decode_quality["body"]["signals"]) == {"possible_utf8_mojibake", "unicode_replacement_character"}
    assert len(email.attachments) == 2 and all(item["status"] == "not_read" for item in email.attachments)
    assert email.sender_name == "Alice" and email.cc == ["carol@example.com"]
    assert email.references == ["<older@example.com>", "<prior@example.com>"]
    metadata = chunk_email(email)[0].metadata
    assert json.loads(metadata["attachments"])[0]["filename"] == "quote.pdf"
    assert json.loads(metadata["source"])["parser_version"].startswith("gmail-mime-v2:")


def test_sender_display_name_label_alias_all_and_dst_boundaries(monkeypatch):
    monkeypatch.setattr(cfg, "RETRIEVAL_TIMEZONE", "America/New_York", raising=False)
    now = datetime(2026, 3, 8, 12, tzinfo=_retrieval_timezone())
    scope = FilterSpec.from_mapping({"sender": "ALICE", "labels": ["Projects", "INBOX"], "date_hint": "today"}, now=now)
    assert (scope.end.astimezone(timezone.utc) - scope.start.astimezone(timezone.utc)).total_seconds() == 23 * 3600
    metadata = {"sender": "a@example.com", "sender_name": "Alice", "labels": '["Label_1","INBOX"]',
                "label_names": '["Projects","INBOX"]', "date": "2026-03-09T03:59:59+00:00"}
    assert scope.matches(metadata)
    assert not scope.matches({**metadata, "date": "2026-03-09T04:00:00+00:00"})
    assert not scope.matches({**metadata, "label_names": "[]"})
    autumn = FilterSpec.from_mapping({"date_hint": "2026-11-01"}, now=now)
    from core.filters import FilterCoverageError
    with pytest.raises(FilterCoverageError, match="ambiguous"):
        autumn.matches({"date": "2026-11-01T01:30:00"})


class ReconcileProvider:
    account = "personal@example.com"
    label_map = {"Label_1": "Projects", "INBOX": "Inbox"}
    def __init__(self):
        self.ids, self.remote = ["a", "b"], {"a": mail("a"), "b": mail("b")}
        self.metadata_calls = []
    def describe_account(self):
        return {"account_id": self.account, "provider": "gmail", "authorization": {"scope_status": "synthetic"}}
    def list_message_page(self, *, page_token, page_size, **kwargs):
        start = int(page_token or 0)
        ids = self.ids[start:start + page_size]
        end = start + len(ids)
        return {"message_ids": ids, "next_page_token": str(end) if end < len(self.ids) else None}
    def get_message(self, mid):
        return self.remote[mid]
    def get_message_metadata(self, mid):
        self.metadata_calls.append(mid)
        return self.remote[mid]


def test_bound_account_metadata_and_scope_changes_wait_for_complete_traversal(tmp_path):
    provider = ReconcileProvider()
    output, state = tmp_path / "mail.json", tmp_path / "state.json"
    sync_gmail_to_json(provider, output, state, "label:Projects", 10, reconcile=True)
    assert len(json.loads(output.read_text())) == 2
    provider.remote["a"]["labelIds"] = ["INBOX"]
    provider.ids = ["a"]
    sync_gmail_to_json(provider, output, state, "label:Projects", 1, reconcile=True)
    rows = json.loads(output.read_text())
    assert [row["id"] for row in rows] == ["gmail_a"]
    assert rows[0]["labels"] == ["INBOX"] and rows[0]["label_names"] == ["Inbox"]
    assert json.loads(state.read_text())["metadata_complete"] is True
    before = output.read_bytes()
    provider.account = "other@example.com"
    with pytest.raises(ValueError, match="account differs"):
        sync_gmail_to_json(provider, output, state, "label:Projects", 1, reconcile=True)
    assert output.read_bytes() == before


def test_incomplete_scope_never_removes_existing_mail(tmp_path):
    provider = ReconcileProvider()
    output, state = tmp_path / "mail.json", tmp_path / "state.json"
    sync_gmail_to_json(provider, output, state, "", 10, reconcile=True)
    provider.ids = ["a", "new"]
    provider.remote["new"] = mail("new")
    sync_gmail_to_json(provider, output, state, "", 1, reconcile=True)
    assert {row["id"] for row in json.loads(output.read_text())} == {"gmail_a", "gmail_b"}
    assert json.loads(state.read_text())["metadata_complete"] is False
    sync_gmail_to_json(provider, output, state, "", 1, reconcile=True)
    assert {row["id"] for row in json.loads(output.read_text())} == {"gmail_a", "gmail_new"}


def test_offline_replay_preserves_captures_uses_checkpoint_and_rejects_tamper(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    provider = GmailReadOnlyProvider()
    capture = provider.capture_message(mail())
    file = raw / "a.json"
    file.write_text(json.dumps(capture), encoding="utf-8")
    original = file.read_bytes()
    output = tmp_path / "new.json"
    summary = replay_captures(raw, output)
    assert summary["parsed_count"] == 1 and not output.exists()
    replay_captures(raw, output, dry_run=False)
    assert file.read_bytes() == original and len(json.loads(output.read_text())) == 1
    with pytest.raises(ValueError, match="new file"):
        replay_captures(raw, output, dry_run=False)
    capture["message"]["payload"]["body"]["data"] = base64.urlsafe_b64encode(b"tampered").decode()
    file.write_text(json.dumps(capture), encoding="utf-8")
    bad_output = tmp_path / "tampered.json"
    with pytest.raises(ValueError, match="failed captures"):
        replay_captures(raw, bad_output, dry_run=False)
    assert not bad_output.exists()


def test_retry_after_cannot_exceed_overall_deadline(monkeypatch):
    class Failure(Exception):
        resp = type("Response", (dict,), {"status": 429})({"retry-after": "50"})
    provider = GmailReadOnlyProvider()
    request = Mock()
    request.execute.side_effect = Failure()
    monkeypatch.setattr(cfg, "GMAIL_READ_TIMEOUT", 0.1, raising=False)
    with patch("agents.gmail_readonly.time.sleep") as sleep, pytest.raises(RunDeadlineExceeded):
        provider._execute_read(lambda: request)
    assert request.execute.call_count == 1
    sleep.assert_not_called()


def test_table_explicit_headers_and_unsupported_nested_layout_are_visible():
    from core.cleaner import html_to_structured_text
    text, rows = html_to_structured_text('<table><tr><th id="amount">金额</th><th id="tax">税</th></tr>'
        '<tr><td headers="tax">7</td><td headers="amount">123.45</td></tr></table>')
    assert rows[-1]["cells"][0]["headers"] == ["税"]
    assert rows[-1]["cells"][1]["headers"] == ["金额"]
    assert rows[-1]["status"] == "complete"
    text, rows = html_to_structured_text('<table><tr><td headers="missing">not approved 123.45'
        '<table><tr><td>nested</td></tr></table></td></tr></table>')
    assert "unresolved_header_reference" in text and "nested_table_flattened" in text
    assert "not approved 123.45" in text


def test_sync_cli_reconcile_replaces_index_and_can_publish_empty(tmp_path):
    from scripts import sync_gmail_readonly as cli
    result = {"failed": 0, "total": 1}
    with patch("sys.argv", ["sync_gmail_readonly.py", "--output", str(tmp_path / "corpus"), "--index"]), \
         patch.object(cli, "GmailReadOnlyProvider"), patch.object(cli, "sync_gmail_to_json", return_value=result), \
         patch.object(cli, "index_email_json", return_value={}) as index:
        cli.main()
        assert index.call_args.kwargs["clear"] is True
    with patch("sys.argv", ["sync_gmail_readonly.py", "--index"]), \
         patch.object(cli, "GmailReadOnlyProvider"), patch.object(cli, "sync_gmail_to_json", return_value={"failed": 0, "total": 0}), \
         patch("core.embedder.clear_collection") as clear:
        cli.main()
        clear.assert_called_once()
