"""Offline provider contracts: pagination, bounded retries and raw-body replay."""
import base64
import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agents.gmail_readonly import GmailReadOnlyProvider, MailContentError, gmail_message_to_email
from agents.mail_providers import MailProviderError


def message(identifier="m", data=None):
    return {"id": identifier, "internalDate": "1710000000000", "payload": {
        "mimeType": "text/plain", "body": data or {"attachmentId": "body-1"}}}


class Service:
    def __init__(self, pages=None, bodies=None):
        self.pages = pages or {}
        self.bodies = bodies or {}
        self.calls = []

    def users(self): return self
    def messages(self): return self
    def attachments(self): return self

    def list(self, **kwargs):
        self.calls.append(("list", kwargs))
        return SimpleNamespace(execute=lambda: self.pages[kwargs.get("pageToken")])

    def get(self, **kwargs):
        self.calls.append(("get", kwargs))
        return SimpleNamespace(execute=lambda: {"data": self.bodies[kwargs["id"]]})


class GmailCaptureTests(unittest.TestCase):
    def test_total_over_page_maximum_and_remaining_request_limit(self):
        pages = {None: {"messages": [{"id": str(i)} for i in range(500)], "nextPageToken": "p2"},
                 "p2": {"messages": [{"id": str(i)} for i in range(500, 1000)], "nextPageToken": "p3"},
                 "p3": {"messages": [{"id": "1000"}]}}
        service = Service(pages)
        self.assertEqual(len(GmailReadOnlyProvider(service=service).list_message_ids(max_results=1001)), 1001)
        self.assertEqual([call[1]["maxResults"] for call in service.calls], [500, 500, 1])

    def test_page_duplicates_do_not_consume_total_and_repeat_tokens_fail(self):
        service = Service({None: {"messages": [{"id": "a"}], "nextPageToken": "p2"},
                           "p2": {"messages": [{"id": "a"}, {"id": "b"}]}})
        self.assertEqual(GmailReadOnlyProvider(service=service).list_message_ids(max_results=2), ["a", "b"])
        service.pages["p2"] = {"messages": [], "nextPageToken": "p2"}
        with self.assertRaises(MailProviderError):
            GmailReadOnlyProvider(service=service).list_message_ids(max_results=3)

    def test_zero_and_invalid_limits_do_not_construct_client(self):
        provider = GmailReadOnlyProvider()
        with patch.object(provider, "_build_service", side_effect=AssertionError("unexpected client")):
            self.assertEqual(provider.list_message_ids(max_results=0), [])
            for value in [-1, True, 1.2, "100"]:
                with self.subTest(value=value), self.assertRaises(ValueError):
                    provider.list_message_ids(max_results=value)
            for value in [0, 501, True, 1.2]:
                with self.subTest(value=value), self.assertRaises(ValueError):
                    provider.list_message_ids(page_size=value)

    def test_client_reused_by_serial_reader(self):
        service = Service({None: {"messages": []}})
        provider = GmailReadOnlyProvider()
        with patch.object(provider, "_build_service", return_value=service) as build:
            provider.list_message_ids()
            provider.list_message_ids()
        build.assert_called_once()

    def test_captured_external_body_replays_without_network_or_mutation(self):
        raw = message()
        original = copy.deepcopy(raw)
        data = base64.urlsafe_b64encode("报价 700".encode()).decode()
        service = Service(bodies={"body-1": data})
        provider = GmailReadOnlyProvider(service=service)
        capture = provider.capture_message(raw)
        self.assertEqual(raw, original)
        self.assertEqual(capture["message"], raw)
        self.assertEqual(capture["body_data"], {"body-1": data})
        with patch.object(provider, "_get_service", side_effect=AssertionError("network")):
            converted = provider.email_from_capture(capture)
        self.assertEqual(converted.body, "报价 700")
        self.assertEqual(len(service.calls), 1)

    def test_invalid_encoded_body_is_preserved_with_safe_error(self):
        provider = GmailReadOnlyProvider(service=Service(bodies={"body-1": "%%%"}))
        capture = provider.capture_message(message())
        self.assertEqual(capture["body_data"], {"body-1": "%%%"})
        self.assertEqual(capture["body_errors"], {"content": "invalid_body_encoding"})
        with self.assertRaises(MailContentError):
            provider.email_from_capture(capture)

    def test_missing_capture_data_never_refetches(self):
        provider = GmailReadOnlyProvider()
        with patch.object(provider, "_get_service", side_effect=AssertionError("network")):
            with self.assertRaises(MailContentError):
                provider.email_from_capture({"format": "gmail-full-v1", "message": message(), "body_data": {}})

    def test_generic_attachment_is_not_downloaded_or_used_as_body(self):
        raw = {"id": "m", "payload": {"mimeType": "multipart/mixed", "parts": [
            {"mimeType": "text/plain", "filename": "private.txt", "body": {"attachmentId": "secret"}},
            {"mimeType": "text/plain", "body": {"data": "b2s="}}]}}
        provider = GmailReadOnlyProvider()
        with patch.object(provider, "_get_service", side_effect=AssertionError("network")):
            capture = provider.capture_message(raw)
            self.assertEqual(provider.email_from_capture(capture).body, "ok")
        self.assertEqual(capture["body_data"], {})

    def test_malformed_message_is_a_content_error_without_body_in_message(self):
        for raw in [None, {"id": "m", "payload": {"headers": ["private body"]}},
                    {"id": "m", "payload": {"body": {"data": 123}, "mimeType": "text/plain"}}]:
            with self.subTest(raw=raw), self.assertRaises(MailContentError) as caught:
                gmail_message_to_email(raw)
            self.assertNotIn("private", str(caught.exception))

    def test_declared_nonempty_body_cannot_silently_become_empty(self):
        with self.assertRaises(MailContentError) as error:
            gmail_message_to_email(message(data={"size": 100}))
        self.assertEqual(error.exception.code, "missing_body_data")
        self.assertEqual(gmail_message_to_email(message(data={"size": 0})).body, "")

    def test_transient_reads_retry_but_auth_does_not(self):
        provider = GmailReadOnlyProvider(max_attempts=3)
        class HttpFailure(Exception):
            def __init__(self, status): self.resp = SimpleNamespace(status=status)
        for status, attempts in [(429, 3), (503, 3), (401, 1), (403, 1)]:
            with self.subTest(status=status):
                request = SimpleNamespace(execute=lambda: (_ for _ in ()).throw(HttpFailure(status)))
                with patch("agents.gmail_readonly.time.sleep") as sleep:
                    with self.assertRaises(HttpFailure):
                        provider._execute_read(lambda: request)
                self.assertEqual(sleep.call_count, attempts - 1)
        with patch("agents.gmail_readonly.time.sleep") as sleep:
            count = [0]
            def execute():
                count[0] += 1
                if count[0] == 1: raise HttpFailure(500)
                return {"ok": True}
            self.assertEqual(provider._execute_read(lambda: SimpleNamespace(execute=execute)), {"ok": True})
            sleep.assert_called_once()


if __name__ == "__main__":
    unittest.main()
