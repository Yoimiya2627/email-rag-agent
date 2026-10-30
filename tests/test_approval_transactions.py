"""Real SQLite/concurrency tests runnable with only Python's standard library.

Run: python -B tests/test_approval_transactions.py
Only the settings module is stubbed; store/provider code and SQLite are real.
No real Gmail client, credentials, environment file, or model is loaded.
"""

import base64
import importlib.util
import json
import sqlite3
import sys
import tempfile
import threading
import time
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from unittest.mock import patch


def _load_modules():
    root = Path(__file__).resolve().parents[1]
    settings = types.ModuleType("config.settings")
    settings.APPROVAL_STORE_PATH = "unused.sqlite3"
    settings.APPROVAL_TTL_SECONDS = 86400
    settings.GMAIL_USER_ID = "me"
    settings.GMAIL_CREDENTIALS_PATH = "unused-credentials.json"
    settings.GMAIL_TOKEN_PATH = "unused-token.json"
    settings.GMAIL_SCOPES = []
    config = types.ModuleType("config")
    config.settings = settings
    agents = types.ModuleType("agents")
    with patch.dict(sys.modules, {"config": config, "config.settings": settings, "agents": agents}):
        spec = importlib.util.spec_from_file_location("agents.approvals", root / "agents/approvals.py")
        approvals = importlib.util.module_from_spec(spec)
        sys.modules["agents.approvals"] = approvals
        agents.approvals = approvals
        spec.loader.exec_module(approvals)
        spec = importlib.util.spec_from_file_location("agents.mail_providers", root / "agents/mail_providers.py")
        providers = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(providers)
    return approvals, providers


approvals, providers = _load_modules()
ApprovalStore = approvals.ApprovalStore


class ApprovalTransactionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "approvals.json"
        self.payload = {"to": ["alice@example.com"], "subject": "Hi", "body": "Hello"}

    def create(self, **kwargs):
        return ApprovalStore(self.path).create("send_email", self.payload, **kwargs)

    def test_concurrent_create_reuses_one_logical_request(self):
        barrier = threading.Barrier(8)

        def create(_):
            store = ApprovalStore(self.path)
            barrier.wait()
            return store.create("send_email", self.payload, request_id="request-1")["approval_id"]

        with ThreadPoolExecutor(max_workers=8) as pool:
            identifiers = list(pool.map(create, range(8)))
        self.assertEqual(len(set(identifiers)), 1)
        self.assertEqual(len(ApprovalStore(self.path).list()), 1)

    def test_concurrent_approve_calls_executor_once_across_instances(self):
        item = self.create()
        barrier, lock = threading.Barrier(8), threading.Lock()
        calls = []

        def execute(approval):
            with lock:
                calls.append(approval["request_id"])
            time.sleep(0.08)
            return {"draft_id": "one-draft", "sent": False}

        def approve(_):
            store = ApprovalStore(self.path)
            barrier.wait()
            try:
                return store.approve(item["approval_id"], executor=execute)["status"]
            except ValueError:
                return "in_flight"

        with ThreadPoolExecutor(max_workers=8) as pool:
            outcomes = list(pool.map(approve, range(8)))
        self.assertEqual(calls, [item["request_id"]])
        self.assertIn("approved", outcomes)
        replay = ApprovalStore(self.path).approve(item["approval_id"], executor=execute)
        self.assertEqual(replay["result"]["draft_id"], "one-draft")
        self.assertEqual(replay["execution_state"], "succeeded")
        self.assertEqual(len(calls), 1)

    def test_success_then_completion_write_failure_cannot_repeat_side_effect(self):
        store = ApprovalStore(self.path)
        item = self.create()
        calls = []

        def execute(_):
            calls.append("draft-created")
            return {"draft_id": "remote-created", "sent": False}

        with patch.object(store, "_complete", side_effect=sqlite3.OperationalError("synthetic disk failure")):
            with self.assertRaises(sqlite3.OperationalError):
                store.approve(item["approval_id"], executor=execute)
        reloaded = ApprovalStore(self.path)
        self.assertEqual(reloaded.get(item["approval_id"])["status"], "unknown")
        self.assertEqual(reloaded.get(item["approval_id"])["result"]["draft_id"], "remote-created")
        with self.assertRaisesRegex(ValueError, "unknown"):
            reloaded.approve(item["approval_id"], executor=execute)
        self.assertEqual(len(calls), 1)

    def test_storage_failure_to_mark_unknown_keeps_claim_and_blocks_restart_replay(self):
        store = ApprovalStore(self.path)
        item = self.create()

        def execute(_):
            raise TimeoutError("provider outcome unknown")

        with patch.object(store, "_record_failure", return_value=None):
            with self.assertRaises(TimeoutError):
                store.approve(item["approval_id"], executor=execute)
        reloaded = ApprovalStore(self.path)
        self.assertEqual(reloaded.get(item["approval_id"])["status"], "executing")
        with self.assertRaisesRegex(ValueError, "executing"):
            reloaded.approve(item["approval_id"], executor=lambda _: self.fail("must not replay"))

    def test_provider_exception_is_unknown_not_pending(self):
        item = self.create()
        store = ApprovalStore(self.path)
        with self.assertRaises(TimeoutError):
            store.approve(item["approval_id"], executor=lambda _: (_ for _ in ()).throw(TimeoutError()))
        self.assertEqual(store.get(item["approval_id"])["execution_state"], "unknown")

    def test_explicit_precondition_failure_is_failed_before_side_effect(self):
        item = self.create()
        store = ApprovalStore(self.path)
        with self.assertRaises(approvals.ApprovalPreconditionError):
            store.approve(item["approval_id"], executor=lambda _: (_ for _ in ()).throw(approvals.ApprovalPreconditionError("invalid")))
        self.assertEqual(store.get(item["approval_id"])["status"], "failed")
        self.assertEqual(store.get(item["approval_id"])["execution_state"], "not_started")

    def test_owner_filter_and_cross_owner_actions(self):
        item = self.create(owner_id="alice", session_id="session-1")
        store = ApprovalStore(self.path)
        self.assertEqual(store.list(owner_id="bob"), [])
        for action in (store.get, store.approve, store.reject):
            with self.assertRaises(PermissionError):
                action(item["approval_id"], owner_id="bob")
        self.assertEqual(len(store.list(owner_id="alice", session_id="session-1")), 1)

    def test_request_identity_rejects_changed_payload_or_session(self):
        item = self.create(request_id="stable", session_id="session-1")
        store = ApprovalStore(self.path)
        duplicate = store.create("send_email", dict(reversed(list(self.payload.items()))), request_id="stable", session_id="session-1")
        self.assertEqual(item["approval_id"], duplicate["approval_id"])
        with self.assertRaises(ValueError):
            store.create("send_email", {**self.payload, "body": "changed"}, request_id="stable", session_id="session-1")
        with self.assertRaises(ValueError):
            store.create("send_email", self.payload, request_id="stable", session_id="session-2")

    def test_expired_pending_is_committed_and_not_executed(self):
        item = self.create(ttl_seconds=0)
        store = ApprovalStore(self.path)
        with self.assertRaisesRegex(ValueError, "expired"):
            store.approve(item["approval_id"], executor=lambda _: self.fail("must not execute"))
        self.assertEqual(ApprovalStore(self.path).get(item["approval_id"])["status"], "expired")

    def test_payload_copy_is_immutable_and_storage_hash_is_checked(self):
        store = ApprovalStore(self.path)
        item = self.create()
        item["payload"]["body"] = "mutated returned dictionary"
        self.assertEqual(store.get(item["approval_id"])["payload"]["body"], "Hello")
        with closing(sqlite3.connect(store.db_path)) as connection, connection:
            connection.execute("UPDATE approvals SET payload_json=? WHERE approval_id=?", (json.dumps({**self.payload, "body": "tampered"}), item["approval_id"]))
        with self.assertRaisesRegex(ValueError, "integrity"):
            store.approve(item["approval_id"], executor=lambda _: self.fail("must not execute"))
        self.assertEqual(store.get(item["approval_id"])["status"], "failed")

    def test_legacy_json_import_once_never_overwrites_original_or_resurrects(self):
        legacy = [{"approval_id": "legacy-1", "action_type": "send_email", "payload": self.payload, "status": "pending"}]
        self.path.write_text(json.dumps(legacy), encoding="utf-8")
        original = self.path.read_bytes()
        store = ApprovalStore(self.path)
        store.reject("legacy-1")
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(ApprovalStore(self.path).get("legacy-1")["status"], "rejected")
        self.assertEqual(len(ApprovalStore(self.path).list()), 1)
        self.assertTrue(store.db_path.exists())

    def test_import_rolls_back_entire_batch_on_invalid_record(self):
        self.path.write_text(json.dumps([
            {"approval_id": "legacy-1", "action_type": "send_email", "payload": self.payload},
            {"approval_id": "broken", "action_type": "send_email", "payload": []},
        ]), encoding="utf-8")
        with self.assertRaises(ValueError):
            ApprovalStore(self.path)
        self.path.write_text("[]", encoding="utf-8")
        self.assertEqual(ApprovalStore(self.path).list(), [])

    def test_simulation_never_claims_sent_and_gmail_marker_is_stable(self):
        item = self.create(request_id="stable-action")
        result = ApprovalStore(self.path).approve(item["approval_id"], executor=providers.SimulatedMailProvider().execute_approval)
        self.assertFalse(result["result"]["sent"])
        raw = providers.build_gmail_raw_message(self.payload, request_id=item["request_id"])
        self.assertEqual(raw, providers.build_gmail_raw_message(self.payload, request_id=item["request_id"]))
        self.assertIn("Message-ID:", base64.urlsafe_b64decode(raw).decode())

    def test_provider_rejects_header_injection_and_string_recipients_before_service(self):
        provider = providers.GmailDraftProvider(service=object())
        for payload in ({**self.payload, "to": "alice@example.com"}, {**self.payload, "subject": "Hi\nBcc: other@example.com"}):
            with self.assertRaises(providers.MailProviderPreconditionError):
                provider.execute_approval({"payload": payload})

    def test_gmail_provider_only_creates_draft_even_with_legacy_send_flag(self):
        class FakeGmail:
            def __init__(self):
                self.calls = []

            def users(self):
                return self

            def getProfile(self, **kwargs):
                return types.SimpleNamespace(execute=lambda: {'emailAddress': 'owner@example.test'})

            def drafts(self):
                self.calls.append("drafts")
                return self

            def create(self, **kwargs):
                self.calls.append(kwargs)
                return self

            def execute(self):
                return {"id": "draft-123", "message": {"id": "message-456"}}

            def messages(self):
                raise AssertionError("sending messages is forbidden")

        service = FakeGmail()
        token = self.path.parent / 'synthetic-token.json'
        token.write_text(json.dumps({'_email_agent_binding': {'version': 1,
            'account_id': 'owner@example.test', 'authorization_id': 'a' * 32}}), encoding='utf-8')
        provider = providers.GmailDraftProvider(service=service, token_path=token)
        item = ApprovalStore(self.path).create('send_email',
            {**self.payload, 'execution_binding': provider.approval_binding()}, request_id='gmail-request')
        with patch.object(providers.cfg, "ENABLE_REAL_EMAIL_SEND", True, create=True):
            result = ApprovalStore(self.path).approve(
                item["approval_id"], executor=provider.execute_approval
            )
        self.assertEqual(result["status"], "approved")
        self.assertEqual(result["result"]["draft_id"], "draft-123")
        self.assertFalse(result["result"]["sent"])
        self.assertEqual(service.calls[0], "drafts")
        mime = base64.urlsafe_b64decode(service.calls[1]["body"]["message"]["raw"]).decode()
        self.assertIn("Message-ID:", mime)

    def test_approve_and_reject_race_has_one_final_decision(self):
        item = self.create()
        barrier, calls = threading.Barrier(2), []

        def review(approve):
            store = ApprovalStore(self.path)
            barrier.wait()
            try:
                if approve:
                    return store.approve(item["approval_id"], executor=lambda _: calls.append("created") or {"sent": False})["status"]
                return store.reject(item["approval_id"])["status"]
            except ValueError:
                return "conflict"

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(review, [True, False]))
        final = ApprovalStore(self.path).get(item["approval_id"])["status"]
        self.assertIn(final, ("approved", "rejected"))
        self.assertEqual(len(calls), 1 if final == "approved" else 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
