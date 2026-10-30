"""Deterministic runtime failures; all model/tool operations are synthetic."""
import asyncio
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import agents.agent_loop as loop
import agents.tools as tools
import config.settings as cfg
from agents.mcp_adapter import MCPToolBackend, StreamableHttpMCPClient, normalize_mcp_result
from agents.runtime import (RunContext, ContextBudgetExceeded, RunDeadlineExceeded,
                            bounded_json, current_run, remaining_timeout, use_run_context)
from agents.tracing import AgentTraceRecorder
from models.schemas import AgentRequest


def call(identifier, name, arguments):
    return SimpleNamespace(id=identifier, function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))


def answer(text="done", calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=calls))])


class Client:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self.replies.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class RuntimeSafetyTests(unittest.TestCase):
    def test_mcp_same_run_approval_reuses_result_but_new_run_does_not(self):
        remote_calls, events = [], []
        def remote(name, arguments):
            remote_calls.append(name)
            return {"status": "pending_approval", "approval_id": "approval-1"}
        backend = MCPToolBackend(SimpleNamespace(call_tool=remote),
                                 audit_logger=SimpleNamespace(record=lambda **event: events.append(event)))
        payload = {"to": ["x@example.invalid"], "subject": "s", "body": "b", "rationale": "r"}
        with patch.object(cfg, "MCP_OWNER_ID", "local", create=True):
            with use_run_context(RunContext()):
                first = backend.call_tool("send_email", payload)
                second = backend.call_tool("send_email", dict(reversed(list(payload.items()))))
                self.assertEqual(first, second)
                self.assertEqual(remote_calls, ["send_email"])
                self.assertTrue(events[-1]["cache_hit"])
            with use_run_context(RunContext()):
                backend.call_tool("send_email", payload)
                self.assertEqual(len(remote_calls), 2)

    def test_mcp_unknown_write_outcome_is_cached_inside_run(self):
        calls = []
        def remote(*args):
            calls.append(1)
            raise TimeoutError("synthetic timeout")
        backend = MCPToolBackend(SimpleNamespace(call_tool=remote),
                                 audit_logger=SimpleNamespace(record=lambda **event: None))
        payload = {"to": ["x@example.invalid"], "subject": "s", "body": "b", "rationale": "r"}
        with patch.object(cfg, "MCP_OWNER_ID", "local", create=True), use_run_context(RunContext()):
            first = backend.call_tool("send_email", payload)
            second = backend.call_tool("send_email", payload)
        self.assertEqual(first["status"], "unknown")
        self.assertEqual(first, second)
        self.assertEqual(len(calls), 1)

    def test_analyzer_uses_remaining_budget_and_disables_sdk_retries(self):
        import agents.analyzer_agent as analyzer
        client = Client([answer()])
        with patch.object(analyzer, "OpenAI", return_value=client) as factory, patch.object(analyzer, "compute_email_stats", return_value={"total": 1}), use_run_context(RunContext(deadline=time.monotonic() + 10)):
            analyzer.AnalyzerAgent().run(AgentRequest(query="synthetic"))
        self.assertEqual(factory.call_args.kwargs["max_retries"], 0)
        self.assertGreater(client.calls[0]["timeout"], 0)
        self.assertLessEqual(client.calls[0]["timeout"], 10)

    def test_analyzer_context_budget_prevents_model_call(self):
        import agents.analyzer_agent as analyzer
        client = Client([])
        with patch.object(analyzer, "OpenAI", return_value=client), patch.object(analyzer, "compute_email_stats", return_value={"total": 1}), use_run_context(RunContext(context_char_limit=10)), self.assertRaises(ContextBudgetExceeded):
            analyzer.AnalyzerAgent().run(AgentRequest(query="synthetic"))
        self.assertEqual(client.calls, [])

    def test_empty_answer_is_not_success(self):
        client = Client([answer("")])
        with patch.object(loop, "_get_client", return_value=client):
            response = loop.run_agent_loop(AgentRequest(query="synthetic"))
        self.assertEqual(response.metadata["status"], "empty_model_response")

    def test_pending_approval_remains_visible_when_budget_is_exhausted(self):
        payload = {"to": ["x@example.invalid"], "subject": "s", "body": "b", "rationale": "r"}
        client = Client([answer(calls=[call("1", "send_email", payload)]), answer()])
        with patch.object(loop, "_get_client", return_value=client), patch.object(loop, "call_tool", return_value={"status": "pending_approval", "approval_id": "approval-1"}), patch.object(cfg, "AGENT_MAX_TOOL_CALLS", 1, create=True):
            response = loop.run_agent_loop(AgentRequest(query="synthetic"))
        self.assertEqual(response.metadata["status"], "tool_budget_exceeded")
        self.assertEqual(response.metadata["pending_approval_ids"], ["approval-1"])

    def test_reranker_budget_exceptions_propagate_without_opening_circuit(self):
        import core.reranker as reranker
        hits = [SimpleNamespace(content="x"), SimpleNamespace(content="y")]
        for error in (RunDeadlineExceeded("deadline"), ContextBudgetExceeded("context")):
            with patch.object(cfg, "ENABLE_RERANKER", True), patch.object(cfg, "RERANKER_BACKEND", "llm", create=True), patch.object(reranker, "_consecutive_failures", 0), patch.object(reranker, "_rerank_with_llm", side_effect=error):
                with self.assertRaises(type(error)):
                    reranker.rerank("x", hits)
                self.assertEqual(reranker._consecutive_failures, 0)

    def test_invalid_arguments_never_invoke_tool(self):
        with patch.dict(tools.TOOL_DISPATCH, {"send_email": lambda **kwargs: self.fail("side effect")}):
            payload = {"to": ["test@example.invalid"], "subject": "synthetic", "body": "body", "rationale": "requested"}
            for invalid in (None, 2, [], {**payload, "owner_id": "other"}, {**payload, "to": "x"}, {**payload, "to": []}):
                self.assertEqual(tools.call_tool("send_email", invalid)["error_code"], "validation_error")
        for invalid in (True, 0, -1, 51, 1.5):
            self.assertEqual(tools.call_tool("search_emails", {"query": "x", "limit": invalid})["error_code"], "validation_error")

    def test_tool_budget_counts_actions_inside_one_round(self):
        client = Client([answer(calls=[call(str(i), "email_stats", {}) for i in range(5)]), answer()])
        with patch.object(loop, "_get_client", return_value=client), patch.object(loop, "call_tool", return_value={"total": 1}) as execute, patch.object(cfg, "AGENT_MAX_TOOL_CALLS", 2, create=True), patch.object(cfg, "AGENT_MAX_REPEAT", 10):
            response = loop.run_agent_loop(AgentRequest(query="synthetic"))
        self.assertEqual(execute.call_count, 2)
        self.assertEqual(response.metadata["actual_tool_calls"], 2)
        self.assertEqual(response.metadata["status"], "tool_budget_exceeded")

    def test_sources_are_candidates_and_citations_are_checked(self):
        hit = {"email_id": "e1", "chunk_id": "c1", "snippet": "synthetic evidence", "score": 1.0}
        client = Client([answer(calls=[call("1", "search_emails", {"query": "x"})]), answer("事实[e1#c1] 错误[e2#c2]")])
        with patch.object(loop, "_get_client", return_value=client), patch.object(loop, "call_tool", return_value=[hit]):
            response = loop.run_agent_loop(AgentRequest(query="x"))
        self.assertEqual(response.sources[0].chunk_id, "c1")
        self.assertEqual(response.metadata["sources_kind"], "retrieved_candidates")
        ref = response.metadata["cited_evidence"][0]
        self.assertEqual((ref["email_id"], ref["chunk_id"]), ("e1", "c1"))
        self.assertEqual((ref["visible_start"], ref["visible_end"]), (0, len(hit["snippet"])))
        self.assertEqual(ref["visible_hash"], __import__("hashlib").sha256(hit["snippet"].encode()).hexdigest())
        self.assertEqual(response.metadata["invalid_citation_count"], 1)
        self.assertEqual(response.metadata["status"], "needs_review")
        self.assertNotIn("[e2#c2]", response.answer)

    def test_mcp_error_variants_preserve_failure(self):
        for response in (SimpleNamespace(isError=True, content=[SimpleNamespace(text="private exception")]),
                         {"isError": True, "structuredContent": {"secret": "synthetic"}},
                         SimpleNamespace(isError=True, structuredContent={"result": 1})):
            normalized = normalize_mcp_result(response)
            self.assertEqual(normalized["status"], "error")
            self.assertEqual(normalized["error_code"], "mcp_tool_error")
            self.assertNotIn("private exception", json.dumps(normalized))

    def test_mcp_timeout_cancels_cooperative_local_coroutine(self):
        cancelled, completed = threading.Event(), threading.Event()
        async def slow():
            try:
                await asyncio.sleep(1)
                completed.set()
            finally:
                cancelled.set()
        client = StreamableHttpMCPClient("unused", timeout=.01)
        with self.assertRaises(TimeoutError):
            client._run(slow)
        self.assertTrue(cancelled.wait(.2))
        self.assertFalse(completed.is_set())

    def test_remote_unknown_stops_loop_without_retry(self):
        client = Client([answer(calls=[call("1", "send_email", {"to": ["x@example.invalid"], "subject": "s", "body": "b", "rationale": "r"}), call("2", "email_stats", {})])])
        from agents.runtime import tool_error
        with patch.object(loop, "_get_client", return_value=client), patch.object(loop, "call_tool", return_value=tool_error("mcp_transport_error", "unknown", unknown=True)) as execute:
            response = loop.run_agent_loop(AgentRequest(query="x"))
        self.assertEqual(response.metadata["status"], "unknown")
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(len(client.calls), 1)

    def test_mcp_owner_mismatch_does_not_call_remote(self):
        client = SimpleNamespace(call_tool=lambda *args: self.fail("remote invoked"))
        audit = SimpleNamespace(record=lambda **event: None)
        with patch.object(cfg, "MCP_OWNER_ID", "server-owner", create=True), use_run_context(RunContext(owner_id="other")):
            output = MCPToolBackend(client, audit_logger=audit).call_tool("email_stats", {})
        self.assertEqual(output["error_code"], "owner_mismatch")

    def test_local_owner_and_stable_request_id_are_not_model_inputs(self):
        records = []
        class Store:
            def create(self, **kwargs):
                records.append(kwargs)
                return {"approval_id": "synthetic"}
        payload = {"to": ["x@example.invalid"], "subject": "s", "body": "b", "rationale": "r"}
        with patch.object(tools, "ApprovalStore", Store), use_run_context(RunContext(owner_id="trusted", session_id="session")):
            tools.send_email(**payload)
            tools.send_email(**payload)
        self.assertEqual(records[0]["owner_id"], "trusted")
        self.assertEqual(records[0]["session_id"], "session")
        self.assertEqual(records[0]["request_id"], records[1]["request_id"])

    def test_context_deadline_propagates_and_resets(self):
        context = RunContext(deadline=time.monotonic() - 1)
        with use_run_context(context), self.assertRaises(RunDeadlineExceeded):
            remaining_timeout(60)
        self.assertIsNone(current_run())

    def test_context_budget_blocks_model_call(self):
        client = Client([])
        with patch.object(loop, "_get_client", return_value=client), patch.object(cfg, "AGENT_CONTEXT_CHAR_LIMIT", 20, create=True):
            response = loop.run_agent_loop(AgentRequest(query="x"))
        self.assertEqual(response.metadata["status"], "context_budget_exceeded")
        self.assertEqual(client.calls, [])

    def test_nested_context_exhaustion_is_not_swallowed(self):
        def fail():
            raise ContextBudgetExceeded("synthetic")
        with patch.dict(tools.TOOL_DISPATCH, {"email_stats": fail}), self.assertRaises(ContextBudgetExceeded):
            tools.call_tool("email_stats", {})

    def test_valid_json_truncation(self):
        result = {"status": "success", "data": {"body": "x" * 10000}}
        for limit in (2, 18, 100, 1000):
            encoded = bounded_json(result, limit)
            self.assertLessEqual(len(encoded), limit)
            self.assertIsInstance(json.loads(encoded), dict)

    def test_error_always_has_terminal_trace_without_body(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trace.jsonl"
            client = Client([RuntimeError("CANARY_PRIVATE_CONTENT")])
            with patch.object(loop, "_get_client", return_value=client), patch.object(cfg, "ENABLE_AGENT_TRACE", True), patch.object(cfg, "AGENT_TRACE_LOG_PATH", str(path)), self.assertRaises(RuntimeError):
                loop.run_agent_loop(AgentRequest(query="CANARY_PRIVATE_CONTENT"))
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[-1]["event"], "agent_end")
            self.assertEqual(rows[-1]["status"], "error")
            self.assertNotIn("CANARY_PRIVATE_CONTENT", path.read_text(encoding="utf-8"))

    def test_trace_rejects_body_arguments_and_raw_errors(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trace.jsonl"
            trace = AgentTraceRecorder(path, enabled=True)
            trace.record("tool_call", tool="send_email", status="error", arguments={"body": "CANARY"}, query="CANARY", error="CANARY")
            self.assertNotIn("CANARY", path.read_text(encoding="utf-8"))

    def test_search_deduplicates_email_ids(self):
        hits = [SimpleNamespace(email_id=email, chunk_id=chunk, content="synthetic", metadata={}, score=1)
                for email, chunk in (("e1", "c1"), ("e1", "c2"), ("e2", "c3"))]
        with patch.object(tools, "retrieve", return_value=hits):
            output = tools.search_emails("x", limit=2)
        self.assertEqual([hit["email_id"] for hit in output["items"]], ["e1", "e2"])

    def test_query_and_draft_instruction_stay_separate(self):
        captured = {}
        class Writer:
            def run(self, request, **kwargs):
                captured.update(query=request.query, instruction=kwargs["instruction"])
                return SimpleNamespace(answer="draft", sources=[])
        with patch.object(tools, "WriterAgent", Writer):
            tools.draft_reply(query="find invoice", instruction="politely decline")
        self.assertEqual(captured, {"query": "find invoice", "instruction": "politely decline"})


if __name__ == "__main__":
    unittest.main()
