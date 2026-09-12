"""Offline checkpoints, quarantine, raw evidence, and safe failure behavior."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.mail_providers import MailProviderError
from scripts import sync_gmail_readonly as sync
from tests.test_sync_integrity import FakeProvider


class RecoveryProvider(FakeProvider):
    def __init__(self, ids, corrupt=(), unavailable=()):
        super().__init__(ids)
        self.corrupt = set(corrupt)
        self.unavailable = set(unavailable)

    def get_message(self, message_id):
        if message_id in self.unavailable:
            raise MailProviderError("private authentication diagnostic")
        result = super().get_message(message_id)
        if message_id in self.corrupt:
            result["payload"]["body"]["data"] = "***not-base64***"
        return result


class SyncRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="sync_recovery_")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.output = self.directory / "emails.json"
        self.state = self.directory / "state.json"
        self.archive = Path(str(self.output) + ".raw")

    def run_sync(self, provider, query="", **kwargs):
        return sync.sync_gmail_to_json(provider, self.output, self.state, query, 10, **kwargs)

    def load_state(self):
        return json.loads(self.state.read_text(encoding="utf-8"))

    def raw_path(self, message_id):
        return self.archive / (hashlib.sha256(message_id.encode()).hexdigest() + ".json")

    def test_corrupt_message_is_archived_skipped_and_retried_outside_latest_window(self):
        result = self.run_sync(RecoveryProvider(["good", "bad", "last"], corrupt=["bad"]), checkpoint_size=1)
        self.assertEqual(result["added"], 2)
        self.assertEqual(result["failed_ids"], ["bad"])
        status = self.load_state()
        self.assertEqual(status["seen_message_ids"], ["good", "last"])
        self.assertEqual(status["status"], "partial")
        self.assertEqual(status["failed_messages"]["bad"]["code"], "invalid_body_encoding")
        self.assertNotIn("base64", self.state.read_text())
        self.assertEqual(json.loads(self.raw_path("bad").read_text())["message"]["payload"]["body"]["data"], "***not-base64***")
        provider = RecoveryProvider(["latest"])
        result = self.run_sync(provider)
        self.assertEqual(provider.reads, ["bad", "latest"])
        self.assertEqual(result["failed"], 0)
        self.assertEqual(result["total"], 4)
        self.assertEqual(self.load_state()["status"], "complete")

    def test_retry_scope_changes_with_query_or_output(self):
        self.run_sync(RecoveryProvider(["bad"], corrupt=["bad"]), query="first")
        provider = RecoveryProvider(["new"])
        self.run_sync(provider, query="second")
        self.assertEqual(provider.reads, ["new"])
        self.run_sync(RecoveryProvider(["bad"], corrupt=["bad"]), query="second")
        self.output = self.directory / "another.json"
        provider = RecoveryProvider(["other"])
        self.run_sync(provider, query="second")
        self.assertEqual(provider.reads, ["other"])

    def test_terminal_read_error_commits_progress_and_preserves_pending_not_quarantine(self):
        with self.assertRaises(MailProviderError):
            self.run_sync(RecoveryProvider(["good", "offline", "later"], unavailable=["offline"]))
        self.assertEqual([row["id"] for row in json.loads(self.output.read_text())], ["gmail_good"])
        status = self.load_state()
        self.assertEqual(status["status"], "aborted")
        self.assertEqual(status["failed_messages"], {})
        self.assertEqual(status["pending_message_ids"], ["offline", "later"])
        self.assertNotIn("authentication", self.state.read_text())
        provider = RecoveryProvider([])
        self.run_sync(provider)
        self.assertEqual(provider.reads, ["offline", "later"])

    def test_terminal_detached_body_error_still_archives_original_message(self):
        provider = RecoveryProvider(["one", "two"])
        def capture(message):
            if message["id"] == "two":
                raise MailProviderError("network unavailable")
            return {"format": "gmail-full-v1", "message": message,
                    "body_data": {}, "body_errors": {}}
        provider.capture_message = capture
        with self.assertRaises(MailProviderError):
            self.run_sync(provider)
        self.assertEqual(json.loads(self.raw_path("two").read_text())["message"]["id"], "two")
        self.assertEqual(self.load_state()["seen_message_ids"], ["one"])
        self.assertEqual(self.load_state()["failed"], 0)

    def test_systemic_conversion_error_is_not_quarantined(self):
        provider = RecoveryProvider(["one"])
        def convert(message):
            raise MailProviderError("systemic failure")
        provider.message_to_email = convert
        with self.assertRaises(MailProviderError):
            self.run_sync(provider)
        self.assertEqual(self.load_state()["status"], "aborted")
        self.assertEqual(self.load_state()["failed"], 0)

    def test_enriched_capture_is_archived_before_offline_conversion(self):
        provider = RecoveryProvider(["one"])
        envelope = {"format": "gmail-full-v1", "message": provider.get_message("one"),
                    "body_data": {"detached": "saved-data"}, "body_errors": {}}
        provider.capture_message = lambda message: envelope
        version_hash = hashlib.sha256(json.dumps(envelope, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        version_path = self.archive / "versions" / (version_hash + ".json")
        conversions = []
        def convert(capture):
            # This assertion runs *inside* conversion: raw + enriched immutable
            # capture must already be durable before the parser sees the data.
            self.assertEqual(capture, envelope)
            self.assertEqual(json.loads(version_path.read_text(encoding="utf-8")), capture)
            raw = json.loads(self.raw_path("one").read_text(encoding="utf-8"))
            self.assertEqual(raw["message"], capture["message"])
            self.assertEqual(raw["body_data"], {})
            selection_path = self.archive / 'selections' / self.raw_path('one').name
            self.assertEqual(set(self.archive.rglob("*.json")), {self.raw_path("one"), version_path, selection_path})
            self.assertEqual(json.loads(selection_path.read_text())['capture_file'], version_path.relative_to(self.archive).as_posix())
            conversions.append(capture)
            return sync.gmail_message_to_email(capture["message"])
        provider.email_from_capture = convert
        self.run_sync(provider)
        self.assertEqual(conversions, [envelope])
        self.assertEqual(self.load_state()["seen_message_ids"], ["one"])

    def test_enriched_version_archive_failure_prevents_conversion_and_seen_commit(self):
        provider = RecoveryProvider(["one"])
        provider.capture_message = lambda message: {"format": "gmail-full-v1", "message": message,
                                                    "body_data": {"detached": "saved-data"}, "body_errors": {}}
        original_dump = sync._dump_json
        def fail_version(path, payload):
            if path.parent == self.archive / "versions":
                raise OSError("synthetic version volume failure")
            return original_dump(path, payload)
        with patch.object(sync, "_dump_json", side_effect=fail_version), \
             patch.object(provider, "email_from_capture", create=True) as convert:
            with self.assertRaises(OSError):
                self.run_sync(provider)
        convert.assert_not_called()
        self.assertTrue(self.raw_path("one").is_file())
        self.assertEqual(json.loads(self.output.read_text(encoding="utf-8")), [])
        self.assertEqual(self.load_state()["seen_message_ids"], [])
        self.assertEqual(self.load_state()["pending_message_ids"], ["one"])

    def test_batch_checkpoint_survives_abrupt_interruption_and_retry_deduplicates(self):
        provider = RecoveryProvider(["one", "two", "three", "four"])
        original_get = provider.get_message
        def interrupt(message_id):
            if message_id == "four":
                raise KeyboardInterrupt()
            return original_get(message_id)
        provider.get_message = interrupt
        with self.assertRaises(KeyboardInterrupt):
            self.run_sync(provider, checkpoint_size=2)
        self.assertEqual(self.load_state()["seen_message_ids"], ["one", "two"])
        self.assertTrue(self.raw_path("three").exists())
        retry = RecoveryProvider([])
        self.run_sync(retry, checkpoint_size=2)
        self.assertEqual(retry.reads, ["three", "four"])
        self.assertEqual(len(json.loads(self.output.read_text())), 4)

    def test_first_batch_interruption_retains_off_window_queue(self):
        provider = RecoveryProvider(["one", "two"])
        original_get = provider.get_message
        def interrupt(message_id):
            if message_id == "two":
                raise KeyboardInterrupt()
            return original_get(message_id)
        provider.get_message = interrupt
        with self.assertRaises(KeyboardInterrupt):
            self.run_sync(provider, checkpoint_size=25)
        self.assertEqual(self.load_state()["pending_message_ids"], ["one", "two"])
        self.assertEqual(json.loads(self.output.read_text()), [])
        retry = RecoveryProvider([])
        self.run_sync(retry)
        self.assertEqual(retry.reads, ["one", "two"])

    def test_initial_state_write_failure_stops_before_fetch(self):
        original_dump = sync._dump_json
        def fail_state(path, payload):
            if path == sync._canonical_path(self.state):
                raise OSError("state volume unavailable")
            return original_dump(path, payload)
        provider = RecoveryProvider(["one"])
        with patch.object(sync, "_dump_json", side_effect=fail_state):
            with self.assertRaises(OSError):
                self.run_sync(provider)
        self.assertEqual(provider.reads, [])
        self.assertFalse(self.state.exists())
        self.assertEqual(json.loads(self.output.read_text()), [])

    def test_invalid_batch_and_limit_options_rejected_before_writes(self):
        for option in ("checkpoint_size", "max_results"):
            for value in (True, 0, -1, 1.5, "2"):
                with self.subTest(option=option, value=value):
                    arguments = {"checkpoint_size": 25, "max_results": 10, option: value}
                    with self.assertRaises(ValueError):
                        sync.sync_gmail_to_json(RecoveryProvider(["one"]), self.output,
                                               self.state, "", **arguments)
                    self.assertEqual(list(self.directory.iterdir()), [])

    def test_duplicate_ids_are_fetched_once_and_mismatched_response_is_quarantined(self):
        provider = RecoveryProvider(["one", "one"])
        self.assertEqual(self.run_sync(provider)["added"], 1)
        self.assertEqual(provider.reads, ["one"])
        provider = RecoveryProvider(["two"])
        provider.get_message = lambda mid: FakeProvider([]).get_message("wrong")
        result = self.run_sync(provider)
        self.assertEqual(result["failed_ids"], ["two"])
        self.assertEqual(result["total"], 1)

    def test_archive_write_failure_never_commits_seen(self):
        original_dump = sync._dump_json
        def fail_archive(path, payload):
            if path.parent == sync._canonical_path(self.archive):
                raise OSError("disk full")
            return original_dump(path, payload)
        with patch.object(sync, "_dump_json", side_effect=fail_archive):
            with self.assertRaises(OSError):
                self.run_sync(RecoveryProvider(["one"]))
        self.assertEqual(json.loads(self.output.read_text()), [])
        self.assertEqual(self.load_state()["seen_message_ids"], [])
        self.assertEqual(self.load_state()["status"], "aborted")
        self.assertEqual(self.run_sync(RecoveryProvider(["one"]))["added"], 1)

    def test_archive_failure_commits_successful_prefix(self):
        original_dump = sync._dump_json
        def fail_second_archive(path, payload):
            if path == self.raw_path("two"):
                raise OSError("archive volume unavailable")
            return original_dump(path, payload)
        with patch.object(sync, "_dump_json", side_effect=fail_second_archive):
            with self.assertRaises(OSError):
                self.run_sync(RecoveryProvider(["one", "two"]))
        self.assertEqual(self.load_state()["seen_message_ids"], ["one"])
        self.assertEqual(self.load_state()["pending_message_ids"], ["two"])

    def test_raw_archive_identifier_cannot_escape_directory(self):
        identifier = "../../outside"
        self.run_sync(RecoveryProvider([identifier]))
        self.assertTrue(self.raw_path(identifier).is_file())
        raw = json.loads(self.raw_path(identifier).read_text(encoding="utf-8"))
        digest = hashlib.sha256(json.dumps(raw, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        version_path = self.archive / "versions" / (digest + ".json")
        self.assertEqual(json.loads(version_path.read_text(encoding="utf-8")), raw)
        self.assertEqual(set(self.archive.rglob("*")),
                         {self.raw_path(identifier), self.archive / "versions", version_path,
                          self.archive / 'selections', self.archive / 'selections' / self.raw_path(identifier).name})
        self.assertTrue(all(path.resolve().is_relative_to(self.archive.resolve()) for path in self.archive.rglob("*")))
        self.assertFalse((self.directory / "outside").exists())
        self.assertFalse((self.directory.parent / "outside").exists())

    def test_corpus_repairs_stale_failure_metadata_without_refetching(self):
        self.run_sync(RecoveryProvider(["one"]))
        state = self.load_state()
        state["failed_messages"] = {"one": {"code": "conversion_failed"}}
        state["pending_message_ids"] = ["one"]
        self.state.write_text(json.dumps(state), encoding="utf-8")
        provider = RecoveryProvider([])
        result = self.run_sync(provider)
        self.assertEqual(result["failed"], 0)
        self.assertEqual(provider.reads, [])

    def test_lock_and_archive_collisions_rejected_before_writes(self):
        cases = [
            (Path(str(self.output) + ".sync-lock.sqlite3"), None),
            (self.state, self.output),
            (self.state, Path(str(self.state) + ".sync-lock.sqlite3")),
            (self.state, self.directory),
        ]
        for state, raw_dir in cases:
            with self.subTest(state=state, raw_dir=raw_dir):
                with self.assertRaises(ValueError):
                    sync.sync_gmail_to_json(RecoveryProvider(["one"]), self.output,
                                            state, "", 10, raw_dir=raw_dir)
                self.assertEqual(list(self.directory.iterdir()), [])

    def test_partial_cli_run_never_indexes_or_clears_and_reports_status_path(self):
        argv = ["sync", "--output", str(self.output), "--state-path", str(self.state),
                "--index", "--clear-index"]
        with patch.object(sync.sys, "argv", argv), patch.object(sync, "GmailReadOnlyProvider", return_value=RecoveryProvider(["bad"], corrupt=["bad"])), patch.object(sync, "index_email_json") as index, patch("builtins.print") as output:
            with self.assertRaises(SystemExit) as caught:
                sync.main()
        self.assertEqual(caught.exception.code, 1)
        index.assert_not_called()
        report = json.loads(output.call_args.args[0])
        self.assertEqual(report["status_path"], str(sync._canonical_path(self.state)))

    def test_fatal_cli_never_indexes_and_suppresses_private_diagnostics(self):
        argv = ["sync", "--output", str(self.output), "--state-path", str(self.state),
                "--index", "--clear-index"]
        with patch.object(sync.sys, "argv", argv), patch.object(sync, "GmailReadOnlyProvider", return_value=RecoveryProvider(["offline"], unavailable=["offline"])), patch.object(sync, "index_email_json") as index, patch("builtins.print") as output:
            with self.assertRaises(SystemExit) as caught:
                sync.main()
        self.assertEqual(caught.exception.code, 1)
        index.assert_not_called()
        report = output.call_args.args[0]
        self.assertNotIn("authentication", report)
        self.assertEqual(json.loads(report)["sync"]["status"], "aborted")


if __name__ == "__main__":
    unittest.main()
