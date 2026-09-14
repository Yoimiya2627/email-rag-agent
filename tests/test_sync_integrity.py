"""Offline sync durability/recovery tests using real files and process locks.

Run with python -B -m unittest tests.test_sync_integrity -v.
Only the Gmail provider is synthetic; no mailbox or model is contacted.
"""
import base64
import json
import multiprocessing
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import sync_gmail_readonly as sync


class FakeProvider:
    def __init__(self, ids, entered=None, release=None):
        self.ids = ids
        self.reads = []
        self.entered = entered
        self.release = release

    def list_message_ids(self, **kwargs):
        if self.entered is not None:
            self.entered.set()
        if self.release is not None and not self.release.wait(60):
            raise TimeoutError("test did not release the first sync")
        return self.ids

    def get_message(self, message_id):
        self.reads.append(message_id)
        return {
            "id": message_id, "internalDate": "1710000000000",
            "payload": {"mimeType": "text/plain", "body": {
                "data": base64.urlsafe_b64encode(f"body {message_id}".encode()).decode(),
            }},
        }


def _sync_process(output, state, message_id, started, go, attempting, entered, release):
    started.set()
    if not go.wait(60):
        raise TimeoutError("test did not start the sync operation")
    attempting.set()
    sync.sync_gmail_to_json(FakeProvider([message_id], entered, release), output, state, "", 10)


class SyncIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="sync_integrity_")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.output = self.directory / "emails.json"
        self.state = self.directory / "state.json"

    def run_sync(self, provider=None, output=None, state=None):
        return sync.sync_gmail_to_json(provider or FakeProvider(["one"]),
                                      output or self.output, state or self.state, "", 10)

    def test_failed_serialization_flush_or_replace_preserves_previous_document(self):
        original = b'[{"id":"original","body":"do not lose this"}]'
        for target in ("json.dump", "os.fsync", "os.replace"):
            with self.subTest(failure=target):
                self.output.write_bytes(original)
                with patch("scripts.sync_gmail_readonly." + target, side_effect=OSError("injected failure")):
                    with self.assertRaises(OSError):
                        sync._dump_json(self.output, [{"id": "new"}])
                self.assertEqual(self.output.read_bytes(), original)
                self.assertEqual(list(self.directory.glob(".*.tmp")), [])

    def test_corpus_write_failure_preserves_both_files_and_allows_retry(self):
        self.run_sync()
        corpus_before, state_before = self.output.read_bytes(), self.state.read_bytes()
        with patch("scripts.sync_gmail_readonly.os.replace", side_effect=OSError("injected failure")):
            with self.assertRaises(OSError):
                self.run_sync(FakeProvider(["one", "two"]))
        self.assertEqual(self.output.read_bytes(), corpus_before)
        self.assertEqual(self.state.read_bytes(), state_before)
        result = self.run_sync(FakeProvider(["one", "two"]))
        self.assertEqual(result, {"fetched": 2, "added": 1, "skipped": 1, "total": 2,
                                  "failed": 0, "failed_ids": []})

    def test_state_write_failure_keeps_committed_corpus_and_retry_deduplicates(self):
        self.run_sync()
        replace = sync.os.replace
        state_writes = []

        def fail_state(source, destination):
            if Path(destination) == sync._canonical_path(self.state):
                if state_writes:
                    raise OSError("injected state failure")
                state_writes.append(Path(source).read_bytes())
            return replace(source, destination)

        with patch("scripts.sync_gmail_readonly.os.replace", side_effect=fail_state):
            with self.assertRaises(OSError):
                self.run_sync(FakeProvider(["one", "two"]))
        self.assertEqual(self.state.read_bytes(), state_writes[0])
        self.assertEqual([row["id"] for row in json.loads(self.output.read_text())], ["gmail_one", "gmail_two"])
        provider = FakeProvider(["one", "two"])
        self.assertEqual(self.run_sync(provider)["added"], 0)
        self.assertEqual(provider.reads, [])
        self.assertEqual(json.loads(self.state.read_text())["seen_message_ids"], ["one", "two"])

    def test_missing_corpus_is_rebuilt_despite_retained_seen_state(self):
        self.run_sync()
        self.output.unlink()
        provider = FakeProvider(["one"])
        self.assertEqual(self.run_sync(provider)["added"], 1)
        self.assertEqual(provider.reads, ["one"])
        self.assertEqual(len(json.loads(self.output.read_text())), 1)

    def test_changed_output_does_not_inherit_another_corpus_seen_ids(self):
        self.run_sync()
        new_output = self.directory / "another.json"
        provider = FakeProvider(["one"])
        self.assertEqual(self.run_sync(provider, output=new_output)["added"], 1)
        self.assertEqual(provider.reads, ["one"])
        self.assertEqual(json.loads(self.state.read_text())["corpus_path"], str(sync._canonical_path(new_output)))

    def test_missing_state_is_rebuilt_from_corpus_without_refetching(self):
        self.run_sync()
        self.state.unlink()
        provider = FakeProvider(["one", "two"])
        self.assertEqual(self.run_sync(provider)["added"], 1)
        self.assertEqual(provider.reads, ["two"])
        self.assertEqual(json.loads(self.state.read_text())["seen_message_ids"], ["one", "two"])

    def test_restored_subset_and_legacy_state_refetch_missing_messages(self):
        self.run_sync()
        self.state.write_text(json.dumps({"seen_message_ids": ["one", "two"], "last_internal_date_ms": 999}), encoding="utf-8")
        provider = FakeProvider(["one", "two"])
        self.assertEqual(self.run_sync(provider)["added"], 1)
        self.assertEqual(provider.reads, ["two"])

    def test_same_output_and_state_path_is_rejected_before_writing(self):
        self.output.write_text("[]", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.run_sync(state=self.output)
        self.assertEqual(self.output.read_text(), "[]")

    def test_sync_loads_separated_body_through_real_provider_conversion(self):
        class FakeService:
            def __init__(self):
                self.attachment_reads = []

            def users(self):
                return self

            def messages(self):
                return self

            def attachments(self):
                return self

            def list(self, **kwargs):
                return SimpleNamespace(execute=lambda: {"messages": [{"id": "large"}]})

            def get(self, **kwargs):
                if "messageId" in kwargs:
                    self.attachment_reads.append(kwargs)
                    return SimpleNamespace(execute=lambda: {
                        "data": base64.urlsafe_b64encode(b"actual full body").decode(),
                    })
                return SimpleNamespace(execute=lambda: {
                    "id": "large", "internalDate": "1710000000000",
                    "payload": {"mimeType": "text/plain", "body": {"attachmentId": "body-data"}},
                })

        service = FakeService()
        self.run_sync(sync.GmailReadOnlyProvider(service=service, user_id="me"))
        saved = json.loads(self.output.read_text())
        self.assertEqual(saved[0]["body"], "actual full body")
        self.assertEqual(saved[0]["body_format"], "plain")
        self.assertEqual(service.attachment_reads, [{"userId": "me", "messageId": "large", "id": "body-data"}])

    def test_processes_sharing_output_keep_both_updates_even_with_distinct_state(self):
        context = multiprocessing.get_context("spawn")
        for distinct_state in (False, True):
            with self.subTest(distinct_state=distinct_state):
                folder = self.directory / str(distinct_state)
                output, state = folder / "emails.json", folder / "state.json"
                second_state = folder / "other-state.json" if distinct_state else state
                first_started, second_started = context.Event(), context.Event()
                first_go, second_go = context.Event(), context.Event()
                first_attempting, second_attempting = context.Event(), context.Event()
                first_entered, second_entered, release = context.Event(), context.Event(), context.Event()
                first = context.Process(target=_sync_process, args=(output, state, "one", first_started, first_go, first_attempting, first_entered, release))
                second = context.Process(target=_sync_process, args=(output, second_state, "two", second_started, second_go, second_attempting, second_entered, None))
                try:
                    # Spawn/import time must not consume the first writer's
                    # lock-holding deadline. Both workers are ready first.
                    first.start()
                    second.start()
                    self.assertTrue(first_started.wait(60), "first worker did not start")
                    self.assertTrue(second_started.wait(60), "second worker did not start")
                    first_go.set()
                    self.assertTrue(first_entered.wait(10), "first writer did not acquire the corpus")
                    second_go.set()
                    self.assertTrue(second_attempting.wait(10), "second writer did not attempt sync")
                    self.assertFalse(second_entered.wait(.2), "second writer read while first owned the corpus")
                    self.assertTrue(first.is_alive(), "first writer exited before release")
                finally:
                    release.set()
                    first_go.set()
                    second_go.set()
                    for process in (first, second):
                        if process.pid is not None:
                            process.join(15)
                            if process.is_alive():
                                process.terminate()
                                process.join(5)
                self.assertEqual(first.exitcode, 0)
                self.assertEqual(second.exitcode, 0)
                self.assertEqual({row["id"] for row in json.loads(output.read_text())}, {"gmail_one", "gmail_two"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
