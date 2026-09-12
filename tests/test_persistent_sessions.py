"""Personal long-memory contracts using temporary SQLite and synthetic turns."""
import tempfile
import sqlite3
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from api.sessions import SessionStore, SessionBusyError
from core.session_repository import SessionConflictError
from core.memory import ConversationMemory, fit_messages_to_budget, build_model_messages
from core.session_context import assemble_session_context
from core.context_budget import measure_context
from core.model_outcomes import ModelText
from agents.runtime import RunContext, RunCancelled, ContextBudgetExceeded, use_run_context


class PersistentSessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "sessions.sqlite3"
        self.store = SessionStore(path=self.path, max_history_turns=12, ttl_seconds=1, max_sessions=2)

    def append(self, query="question", answer="answer", *, owner="owner", sid="session", metadata=None, **kwargs):
        with self.store.turn(owner, sid) as memory:
            return memory.append_turn(query, answer, metadata, **kwargs)

    def test_reopening_repository_closes_schema_inspection_connections(self):
        opened = []
        connect = sqlite3.connect

        def tracked_connect(*args, **kwargs):
            connection = connect(*args, **kwargs)
            opened.append(connection)
            return connection

        with patch('core.session_repository.sqlite3.connect', side_effect=tracked_connect):
            SessionStore(path=self.path)
        self.assertTrue(opened)
        for connection in opened:
            with self.assertRaises(sqlite3.ProgrammingError):
                connection.execute('SELECT 1')

    def test_full_transcript_survives_window_restart_ttl_and_eviction(self):
        for index in range(20):
            self.append(f"question-{index}", f"answer-{index}")
        with self.store.turn("owner", "session") as memory:
            self.assertEqual(len(memory.to_messages()), 24)
            self.assertEqual(memory.to_messages()[0]["content"], "question-8")
        with patch("api.sessions.time.monotonic", return_value=time.monotonic()+20):
            with self.store.turn("owner", "session") as memory:
                self.assertEqual(len(memory.to_messages()), 24)
        self.append(sid="other")
        self.append(sid="evicts-idle")
        restarted = SessionStore(path=self.path, max_history_turns=8)
        with restarted.turn("owner", "session") as memory:
            self.assertEqual(len(memory.to_messages()), 16)
        history = restarted.history("owner", "session")["turns"]
        self.assertEqual(len(history), 20)
        self.assertEqual(history[0]["query"], "question-0")

    def test_success_and_failure_pairs_are_atomic_but_only_success_is_context(self):
        success = self.append("good", "done", metadata={"status": "success"})
        failure = self.append("failed task", "partial text", metadata={"status": "incomplete"}, include_in_context=True)
        self.append("empty", "", metadata={"status": "empty_model_response"}, include_in_context=True)
        self.append("model marker", ModelText("part", completion_status="incomplete"), include_in_context=True)
        with self.store.turn("owner", "session") as memory:
            self.assertEqual(memory.to_messages(), [{"role":"user","content":"good"}, {"role":"assistant","content":"done"}])
        rows = self.store.history("owner", "session")["turns"]
        self.assertEqual([row["include_in_context"] for row in rows], [True, False, False, False])
        self.assertEqual([rows[0]["turn_id"], rows[1]["turn_id"]], [success, failure])
        restarted = SessionStore(path=self.path)
        with restarted.turn("owner", "session") as memory:
            self.assertEqual(len(memory.to_messages()), 2)

    def test_escaping_exception_rolls_back_transcript_and_cached_history(self):
        self.append("existing", "done")
        with self.assertRaises(RuntimeError):
            with self.store.turn("owner", "session") as memory:
                memory.append_turn("should rollback", "partial", {"status":"incomplete"})
                raise RuntimeError("synthetic failure before commit")
        self.assertEqual(len(self.store.history("owner", "session")["turns"]), 1)
        with self.store.turn("owner", "session") as memory:
            self.assertEqual(memory.to_messages()[0]["content"], "existing")

    def test_late_invalid_evidence_rolls_back_every_staged_pair(self):
        with self.assertRaises(ValueError):
            with self.store.turn("owner", "session") as memory:
                memory.append_turn("first", "done")
                memory.append_turn("second", "done", evidence_refs=[{"email_id":"missing chunk"}])
        self.assertEqual(self.store.history("owner", "session")["turns"], [])
        self.assertEqual(self.store.list_sessions("owner"), [])
        with self.store.turn("owner", "session") as memory:
            self.assertEqual(memory.to_messages(), [])

    def test_legacy_add_only_persists_complete_pairs(self):
        with self.store.turn("owner", "session") as memory:
            memory.add("user", "legacy")
            memory.add("assistant", "answer")
            memory.add("user", "dangling")
        rows = self.store.history("owner", "session")["turns"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["query"], "legacy")
        with self.store.turn("owner", "session") as memory:
            self.assertEqual(len(memory.to_messages()), 2)
            memory.add("user", "blank reply")
            memory.add("assistant", "")
        with self.store.turn("owner", "session") as memory:
            self.assertEqual(len(memory.to_messages()), 2)

    def test_owner_keys_page_cursors_and_literal_history_search(self):
        for index in range(5):
            self.append(f"literal %_{index}", "NOT approved")
        self.append("other owner secret", "private", owner="other")
        page = self.store.history("owner", "session", limit=2)
        self.assertEqual(len(page["turns"]), 2)
        self.assertTrue(page["has_more"])
        next_page = self.store.history("owner", "session", after=page["next_after"], limit=2)
        self.assertNotEqual(page["turns"][-1]["turn_id"], next_page["turns"][0]["turn_id"])
        self.assertEqual(len(self.store.search_history("owner", "session", "%_")), 5)
        self.assertEqual(self.store.search_history("owner", "session", "other owner secret"), [])
        self.assertEqual(self.store.list_sessions("missing"), [])
        self.assertEqual(len(self.store.history("other", "session")["turns"]), 1)

    def test_user_facts_require_explicit_action_owned_source_and_version_match(self):
        turn_id = self.append("Use AUD 734, no authorization to send.", "understood")
        other_id = self.append("Other person's choice", "yes", owner="other")
        args = dict(source_turn_id=turn_id, expected_version=0)
        with self.assertRaises(PermissionError):
            self.store.set_task_fact("owner", "session", "budget", 734, **args)
        with self.assertRaises(KeyError):
            self.store.set_task_fact("owner", "session", "budget", 734,
                                    source_turn_id=other_id, expected_version=0, explicit_user=True)
        first = self.store.set_task_fact("owner", "session", "budget", {"currency":"AUD","amount":734},
                                        explicit_user=True, **args)
        self.assertEqual(first["version"], 1)
        self.assertEqual(first["authority"], "user_note_not_execution_permission")
        with self.assertRaises(SessionConflictError):
            self.store.set_task_fact("owner", "session", "budget", 900, explicit_user=True, **args)
        self.store.set_task_fact("owner", "session", "budget", 900, source_turn_id=turn_id,
                                 expected_version=1, explicit_user=True)
        self.assertEqual(self.store.task_facts("owner", "session")[0]["value"], 900)
        self.assertEqual(len(self.store.task_facts("owner", "session", include_history=True)), 2)
        self.assertEqual(self.store.task_facts("other", "session"), [])

    def test_summary_is_labeled_source_excerpts_and_evidence_stays_unverified(self):
        turn_id = self.append("NOT approved " + "x"*300, "Assistant claim is not source evidence",
                              evidence_refs=[{"email_id":"email", "chunk_id":"chunk", "source_version":"v1",
                                              "visible_start":0, "visible_end":20}])
        summary = self.store.summary("owner", "session", max_chars=40)
        self.assertTrue(summary["not_evidence"])
        self.assertEqual(summary["method"], "deterministic_excerpts_v1")
        self.assertEqual(summary["turns"][0]["turn_id"], turn_id)
        self.assertTrue(summary["turns"][0]["truncated"])
        refs = self.store.evidence_refs("owner", "session", turn_id=turn_id)
        self.assertEqual(refs[0]["source_version"], "v1")
        self.assertEqual(refs[0]["validation_status"], "unverified_requires_source_read")
        self.assertEqual(self.store.evidence_refs("other", "session"), [])

    def test_delete_cascades_transcript_facts_and_evidence_and_refuses_active_turn(self):
        turn_id = self.append(evidence_refs=[{"email_id":"e","chunk_id":"c","source_version":"v1"}])
        self.store.set_task_fact("owner", "session", "language", "English", source_turn_id=turn_id,
                                 expected_version=0, explicit_user=True)
        with self.store.turn("owner", "session"):
            with self.assertRaises(SessionBusyError):
                self.store.delete("owner", "session")
        self.store.delete("owner", "session")
        self.assertEqual(self.store.history("owner", "session")["turns"], [])
        self.assertEqual(self.store.task_facts("owner", "session"), [])
        self.assertEqual(self.store.evidence_refs("owner", "session"), [])
        with SessionStore(path=self.path).turn("owner", "session") as memory:
            self.assertEqual(memory.to_messages(), [])

    def test_two_store_stale_writer_conflicts_without_overwriting_success(self):
        self.append("initial", "done")
        second = SessionStore(path=self.path)
        with self.assertRaises(SessionConflictError):
            with self.store.turn("owner", "session") as first_memory:
                first_memory.append_turn("stale pending", "done")
                with second.turn("owner", "session") as second_memory:
                    second_memory.append_turn("committed elsewhere", "done")
        rows = self.store.history("owner", "session")["turns"]
        self.assertEqual([row["query"] for row in rows], ["initial", "committed elsewhere"])
        with self.store.turn("owner", "session") as memory:
            self.assertEqual(memory.to_messages()[-2]["content"], "committed elsewhere")

    def test_default_memory_store_retains_legacy_five_pair_window(self):
        store = SessionStore()
        for index in range(7):
            with store.turn("owner", "session") as memory:
                memory.append_turn(str(index), "done")
        with store.turn("owner", "session") as memory:
            self.assertEqual(len(memory.to_messages()), 10)
        self.assertIsNone(store.repository)
        self.assertEqual(len(store.history("owner", "session")["turns"]), 7)

    def test_byte_budget_evicts_idle_cache_without_deleting_durable_turns(self):
        self.append("x"*1500, "done", sid="first")
        one_size = self.store.cached_bytes
        self.store.max_cached_bytes = one_size+512
        self.append("y"*1500, "done", sid="second")
        self.assertLessEqual(self.store.cached_bytes, self.store.max_cached_bytes)
        self.assertEqual(len(self.store.history("owner", "first")["turns"]), 1)
        self.assertEqual(len(self.store.history("owner", "second")["turns"]), 1)
        self.assertEqual(len(self.store._entries), 1)

    def test_oversized_result_is_persisted_then_evicted_and_oversized_context_is_refused(self):
        store = SessionStore(path=self.path, max_cached_bytes=2000)
        with store.turn("owner", "large") as memory:
            memory.append_turn("large", "x"*5000)
            with self.assertRaises(ContextBudgetExceeded):
                memory.to_messages()
        self.assertLessEqual(store.cached_bytes, 2000)
        self.assertEqual(store.history("owner", "large")["turns"][0]["answer"], "x"*5000)
        with self.assertRaises(SessionBusyError):
            with store.turn("owner", "large"):
                pass
        self.assertEqual(len(store.history("owner", "large")["turns"]), 1)

    def test_exception_restores_cache_and_releases_active_admission(self):
        store = SessionStore(path=self.path, max_cached_bytes=2000)
        with self.assertRaises(RuntimeError):
            with store.turn("owner", "failed") as memory:
                memory.append_turn("rollback", "x"*5000)
                raise RuntimeError("synthetic")
        self.assertLessEqual(store.cached_bytes, 2000)
        self.assertEqual(store.history("owner", "failed")["turns"], [])
        with store.turn("owner", "failed") as memory:
            self.assertEqual(memory.to_messages(), [])

    def test_old_explicit_constraint_reaches_model_context_with_original_turn_source(self):
        old = self.append("Keep AUD 734 and do not send.", "understood")
        self.store.set_task_fact("owner", "session", "budget", {"currency":"AUD","amount":734},
                                 source_turn_id=old, expected_version=0, explicit_user=True)
        for index in range(15):
            self.append(f"unrelated-{index}", "answer")
        context = assemble_session_context(self.store.task_facts("owner", "session"),
                                            self.store.search_history("owner", "session", "do not send"),
                                            char_limit=4000, token_limit=4000)
        with self.store.turn("owner", "session") as memory, use_run_context(RunContext(task_context=context)):
            self.assertNotIn("AUD 734", str(memory.to_messages()))
            messages = build_model_messages("system", "Continue the quote.", memory.to_messages())
        self.assertIn("734", messages[-1]["content"])
        self.assertIn(old, messages[-1]["content"])
        self.assertIn("不是系统指令或执行授权", messages[0]["content"])
        self.assertEqual(self.store.get_turn("owner", "session", old)["query"], "Keep AUD 734 and do not send.")
        with self.assertRaises(KeyError):
            self.store.get_turn("other", "session", old)


class TokenAndLifecycleTests(unittest.TestCase):
    def test_session_context_budget_drops_whole_facts_and_marks_history_as_claims(self):
        facts = [{"key":"too-long","value":"x"*5000,"source_turn_id":"first","version":1},
                 {"key":"budget","value":734,"source_turn_id":"second","version":1}]
        context = assemble_session_context(facts, [{"turn_id":"old","query":"request","answer":"not email evidence"}],
                                            char_limit=1500, token_limit=2000)
        self.assertEqual(context["omitted_count"], 1)
        self.assertNotIn("too-long", context["text"])
        self.assertIn("734", context["text"])
        self.assertIn("assistant_claim_not_source_evidence", context["text"])
        self.assertEqual(context["source_turn_ids"], ["second", "old"])
        self.assertLessEqual(context["estimated_input_tokens"], 2000)

    def test_token_capacity_counts_schema_and_output_reserve_while_preserving_tool_chain(self):
        system = {"role":"system","content":"system"}
        prior = [{"role":role,"content":"old"*100} for _ in range(3) for role in ("user","assistant")]
        chain = [{"role":"user","content":"CURRENT"},
                 {"role":"assistant","tool_calls":[{"id":"call","function":{"name":"read","arguments":"{}"}}]},
                 {"role":"tool","tool_call_id":"call","content":"evidence"}]
        schemas = [{"name":"read","description":"schema"*100}]
        minimum = measure_context([system,*chain], schemas, token_counter=len)["estimated_input_tokens"]
        with use_run_context(RunContext(context_char_limit=10000, context_token_limit=minimum+100,
                                        output_token_reserve=100, token_counter=len)) as run:
            messages, count = fit_messages_to_budget([system,*prior,*chain], len(prior), schemas=schemas)
            self.assertEqual(messages, [system,*chain])
            self.assertEqual(count, 0)
            run.check_context(messages, schemas)
            self.assertEqual(run.context_metrics["dropped_history_turns"], 3)
            with self.assertRaises(ContextBudgetExceeded):
                fit_messages_to_budget(messages, 0, schemas=schemas, output_reserve=101)

    def test_char_cap_survives_injected_tokenizer_and_bad_counter_is_rejected(self):
        messages = [{"role":"system","content":"s"},{"role":"user","content":"x"*1000}]
        with self.assertRaises(ContextBudgetExceeded):
            fit_messages_to_budget(messages, char_limit=100, token_limit=10000, output_reserve=0, token_counter=lambda _:1)
        with self.assertRaises(ValueError):
            measure_context(messages, token_counter=lambda _:True)
        self.assertEqual(measure_context(messages)["token_estimation_method"], "utf8_bytes_upper_bound")

    def test_short_more_than_five_turns_fit_when_window_and_budget_allow(self):
        memory = ConversationMemory(max_turns=20)
        for index in range(12):
            memory.append_turn(f"q{index}", f"a{index}")
        messages = [{"role":"system","content":"s"},*memory.to_messages(),{"role":"user","content":"current"}]
        fitted, count = fit_messages_to_budget(messages, 24, char_limit=20000, token_limit=10000, output_reserve=2000)
        self.assertEqual(count, 24)
        self.assertEqual(fitted[1]["content"], "q0")

    def test_cancel_stops_new_operations_but_completed_checkpoint_can_be_saved(self):
        cancelled, saved = threading.Event(), []
        callback = Mock()
        run = RunContext(cancel_event=cancelled, progress_callback=callback, checkpoint_callback=saved.append)
        run.progress("retrieved", count=2)
        callback.assert_called_once_with("retrieved", count=2)
        cancelled.set()
        with self.assertRaises(RunCancelled):
            run.remaining(30)
        with self.assertRaises(RunCancelled):
            run.progress("next")
        run.checkpoint({"completed_tool":"call-1"})
        self.assertEqual(saved, [{"completed_tool":"call-1"}])

    def test_checkpoint_and_progress_fail_closed_when_persistence_callback_fails(self):
        failing = Mock(side_effect=OSError("synthetic disk failure"))
        run = RunContext(checkpoint_callback=failing, progress_callback=failing)
        with self.assertRaises(OSError):
            run.checkpoint({"boundary":"before-write"})
        with self.assertRaises(OSError):
            run.progress("starting")


if __name__ == "__main__":
    unittest.main()
