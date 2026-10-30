"""Real in-process MCP SDK and model-visible payload regressions, no transport."""
import asyncio
import copy
import importlib.metadata
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from mcp.server.fastmcp.exceptions import ToolError

import agents.agent_loop as loop
import agents.tools as tools
import config.settings as cfg
import mcp_server
from agents.runtime import bounded_json, normalize_tool_result, result_evidence
from agents.tool_registry import TOOL_REGISTRY
from models.schemas import AgentRequest, SearchResult
from core.evidence import text_hash


def hit(index=1, content="approved budget is 500"):
    return SearchResult(email_id=f"e{index}", chunk_id=f"c{index}", content=content,
                        metadata={"subject": f"Budget {index}", "sender": "a@example.invalid", "date": "2026-09-01"}, score=1)


class SDKContracts(unittest.TestCase):
    def setUp(self):
        self.settings = patch.multiple(cfg, MCP_HOST="127.0.0.1", MCP_AUTH_TOKEN="")
        self.settings.start()
        self.addCleanup(self.settings.stop)
        self.server = mcp_server.build_server()

    def invoke(self, name, arguments):
        # FastMCP.call_tool returns (content, structuredContent) in SDK 1.28.1.
        blocks, structured = asyncio.run(self.server.call_tool(name, arguments))
        self.assertEqual(json.loads(blocks[0].text), structured)
        return structured

    def test_actual_sdk_and_object_output_schema_for_every_registered_tool(self):
        self.assertTrue(importlib.metadata.version("mcp"))
        listed = asyncio.run(self.server.list_tools())
        self.assertEqual({tool.name for tool in listed}, set(TOOL_REGISTRY))
        self.assertTrue(all(tool.outputSchema["type"] == "object" for tool in listed))

    def test_search_required_fields_only_uses_defaults(self):
        with patch.object(tools, "retrieve", return_value=[hit()]) as retrieve:
            result = self.invoke("search_emails", {"query": "budget"})
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["data"]["items"][0]["snippet"], "approved budget is 500")
        self.assertEqual(result["data"]["coverage"]["scope"], "selected_ranked_candidates")
        self.assertEqual(retrieve.call_args.kwargs["filters"]["labels"], [])

    def test_search_explicit_nulls_and_explicit_values_match_local_contract(self):
        with patch.object(tools, "retrieve", return_value=[hit()]):
            for optional in ({"labels": None, "limit": None}, {"labels": [], "limit": 3}):
                arguments = {"query": "budget", **optional}
                remote = self.invoke("search_emails", arguments)
                local = tools.call_tool("search_emails", arguments)
                self.assertEqual(remote["status"], "success")
                # Each independent search deliberately issues an opaque random
                # snapshot capability. Compare its semantics, then prove each
                # capability resumes its own exact page without another search.
                def semantic_page(page):
                    result = copy.deepcopy(page)
                    for key in ("resume_cursor", "next_cursor"):
                        result[key] = bool(result[key])
                    for item in result["items"]:
                        item["continuation_cursor"] = bool(item["continuation_cursor"])
                    return result
                self.assertEqual(semantic_page(remote["data"]), semantic_page(local["data"]))
                self.assertNotEqual(remote["data"]["resume_cursor"], local["data"]["resume_cursor"])
                with patch.object(tools, "retrieve", side_effect=AssertionError("snapshot must be reused")):
                    resumed = self.invoke("search_emails", {**arguments, "cursor": remote["data"]["resume_cursor"]})
                    self.assertEqual(resumed["data"], remote["data"])
                    resumed = tools.call_tool("search_emails", {**arguments, "cursor": local["data"]["resume_cursor"]})
                    self.assertEqual(resumed["data"], local["data"])

    def test_search_invalid_limit_is_rejected_before_business_execution(self):
        with patch.object(tools, "retrieve", side_effect=AssertionError("must not execute")):
            for limit in (True, 0, 51, 1.5, "three"):
                with self.assertRaises(ToolError):
                    self.invoke("search_emails", {"query": "budget", "limit": limit})

    def test_summary_string_business_output_returns_envelope(self):
        with patch.object(tools, "SummarizerAgent", return_value=SimpleNamespace(
                run=lambda request: SimpleNamespace(answer="Budget approved", sources=[hit()]))):
            result = self.invoke("summarize_emails", {"query": "budget"})
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["data"], "Budget approved")
        self.assertEqual(result["evidence_refs"], [{"email_id": "e1", "chunk_id": "c1"}])

    def test_get_email_object_business_output_returns_envelope(self):
        with patch.object(tools, "get_indexed_email", return_value={"email_id": "e1", "body": "Budget approved", "chunks": []}):
            result = self.invoke("get_email", {"email_id": "e1"})
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["data"]["body"], "Budget approved")

    def test_domain_error_remains_error_through_sdk_output_validation(self):
        with patch.object(tools, "retrieve", side_effect=RuntimeError("synthetic failure")):
            result = self.invoke("search_emails", {"query": "budget"})
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_code"], "tool_failed")

    def test_draft_reply_string_output_and_required_instruction(self):
        with patch.object(tools, "WriterAgent", return_value=SimpleNamespace(
                run=lambda request, **kwargs: SimpleNamespace(answer="Draft text", sources=[]))):
            result = self.invoke("draft_reply", {"instruction": "Confirm receipt"})
            with self.assertRaises(ToolError):
                self.invoke("draft_reply", {})
        self.assertEqual(result["data"], "Draft text")

    def test_send_email_preserves_pending_approval_envelope(self):
        store = SimpleNamespace(create=lambda **kwargs: {"approval_id": "synthetic-approval", "status": "pending"})
        with patch.object(tools, "ApprovalStore", return_value=store):
            result = self.invoke("send_email", {"to": ["x@example.invalid"], "subject": "Budget",
                                                "body": "Approved", "rationale": "Requested"})
        self.assertEqual(result["status"], "approval_required")
        self.assertEqual(result["data"]["approval_id"], "synthetic-approval")

    def test_email_stats_returns_object_envelope(self):
        rows = [{"chunk_id": "c1", "metadata": {"email_id": "e1"}}]
        with patch.object(tools, "get_all_metadata", return_value=rows), \
             patch.object(tools, "compute_scoped_stats", return_value={"total": 3}) as stats:
            result = self.invoke("email_stats", {"sender": "a@example.invalid", "email_ids": ["e1"]})
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["data"], {"total": 3})
        self.assertEqual(stats.call_args.args[0], rows)
        self.assertEqual(stats.call_args.kwargs["filters"], {"sender": "a@example.invalid", "date_hint": "", "labels": []})
        self.assertEqual(stats.call_args.kwargs["email_ids"], ["e1"])


class ModelPayloadContracts(unittest.TestCase):
    def run_with_calls(self, calls, final="done", limit=4000):
        requests = []
        messages = [SimpleNamespace(content="", tool_calls=[SimpleNamespace(id=str(index),
                    function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))])
                    for index, (name, arguments) in enumerate(calls)]
        messages.append(SimpleNamespace(content=final, tool_calls=None))
        def create(**kwargs):
            requests.append(copy.deepcopy(kwargs))
            return SimpleNamespace(choices=[SimpleNamespace(message=messages.pop(0))])
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch.object(loop, "_get_client", return_value=client), patch.multiple(cfg,
                AGENT_TOOL_BACKEND="local", AGENT_TOOL_OUTPUT_LIMIT=limit,
                AGENT_MAX_STEPS=6, ENABLE_AGENT_TRACE=False):
            response = loop.run_agent_loop(AgentRequest(query="inspect budget"))
        outputs = [json.loads(row["content"]) for row in requests[-1]["messages"] if row["role"] == "tool"]
        return response, outputs

    def test_twenty_search_results_keep_readable_real_evidence_in_model_message(self):
        hits = [hit(index, f"evidence-{index}:" + "文" * 290) for index in range(20)]
        with patch.object(tools, "retrieve", return_value=hits):
            response, outputs = self.run_with_calls([("search_emails", {"query": "budget", "limit": 20})])
        output = outputs[0]
        self.assertTrue(output["truncated"])
        self.assertGreater(len(output["data"]["items"]), 0)
        for row in output["data"]["items"]:
            original = next(source for source in hits if source.email_id == row["email_id"])
            self.assertTrue(original.content.startswith(row["snippet"]))
            self.assertTrue(row["snippet"])
            ref = next(ref for ref in output["evidence_refs"] if ref["email_id"] == row["email_id"])
            self.assertEqual(ref["visible_hash"], text_hash(row["snippet"]))
            self.assertEqual(ref["visible_end"] - ref["visible_start"], len(row["snippet"]))
        self.assertEqual(len(response.sources), 20)
        self.assertLessEqual(len(json.dumps(output, ensure_ascii=False, separators=(",", ":"))), 4000)
        self.assertEqual(output["evidence_refs"], response.metadata["model_visible_evidence"])

    def test_2100_character_email_body_is_available_without_duplication(self):
        body = "Approved amount 500. " + "文" * 2080
        source = hit(content=body).model_dump()
        with patch.object(tools, "get_indexed_email", return_value={"email_id": "e1", "body": body, "chunks": [source]}):
            first = tools.get_email("e1")
            self.assertEqual(first["next_start"], 1200)
            _, outputs = self.run_with_calls([
                ("get_email", {"email_id": "e1"}),
                ("get_email", {"email_id": "e1", "start": first["next_start"],
                               "source_version": first["source_version"], "source_sha256": first["source_sha256"]})])
        retained = []
        for output in outputs:
            self.assertEqual(output["status"], "success")
            self.assertNotIn("body", output["data"])
            self.assertNotIn("candidate_sources", output)
            self.assertNotIn("truncated", output)
            self.assertEqual(len(output["data"]["chunks"]), 1)
            text = output["data"]["chunks"][0]["content"]
            ref = output["evidence_refs"][0]
            self.assertEqual(text, body[ref["visible_start"]:ref["visible_end"]])
            self.assertEqual(ref["visible_hash"], text_hash(text))
            retained.append(text)
        self.assertEqual("".join(retained), body)
        self.assertEqual(outputs[0]["evidence_refs"][0]["visible_end"], outputs[1]["evidence_refs"][0]["visible_start"])
        self.assertTrue(outputs[0]["data"]["has_more"])
        self.assertFalse(outputs[1]["data"]["has_more"])
        self.assertIsNone(outputs[1]["data"]["next_start"])

    def test_each_message_has_its_own_references_and_no_earlier_sources(self):
        with patch.object(tools, "retrieve", side_effect=[[hit(1)], [hit(2)]]):
            response, outputs = self.run_with_calls([("search_emails", {"query": "one"}), ("search_emails", {"query": "two"})])
        self.assertEqual([(ref["email_id"], ref["chunk_id"]) for ref in outputs[1]["evidence_refs"]], [("e2", "c2")])
        ref = outputs[1]["evidence_refs"][0]
        self.assertEqual(ref["visible_hash"], text_hash(hit(2).content))
        self.assertEqual(ref["chunk_sha256"], text_hash(hit(2).content))
        self.assertEqual(ref["visible_start"], 0)
        self.assertEqual(ref["visible_end"], len(hit(2).content))
        self.assertEqual({row["email_id"] for row in result_evidence(outputs[1])}, {"e2"})
        self.assertEqual(len(response.sources), 2)

    def test_omitted_candidate_cannot_be_claimed_as_visible_citation(self):
        with patch.object(tools, "retrieve", return_value=[hit(index, "x" * 300) for index in range(20)]):
            response, outputs = self.run_with_calls([("search_emails", {"query": "budget", "limit": 20})], final="claim [e19#c19]", limit=3000)
        self.assertEqual(outputs[0]["status"], "success")
        self.assertTrue(outputs[0]["evidence_refs"])
        self.assertTrue(outputs[0]["data"]["items"][0]["snippet"])
        self.assertNotIn(("e19", "c19"), [(ref["email_id"], ref["chunk_id"]) for ref in outputs[0]["evidence_refs"]])
        self.assertEqual(response.metadata["invalid_citation_count"], 1)
        self.assertEqual(response.metadata["status"], "needs_review")

    def test_tiny_budget_is_not_content_free_success(self):
        with patch.object(tools, "retrieve", return_value=[hit()]):
            response, outputs = self.run_with_calls([("search_emails", {"query": "budget"})], limit=100)
        self.assertNotEqual(outputs[0].get("status"), "success")
        self.assertEqual(response.metadata["status"], "partial")
        self.assertEqual(response.metadata["steps"][0]["output_error_code"], "output_budget_exceeded")
        self.assertEqual(response.metadata["model_visible_evidence"], [])

    def test_versioned_page_that_cannot_fit_never_claims_missing_evidence_success(self):
        with patch.object(tools, "retrieve", return_value=[hit(index, "x" * 300) for index in range(20)]):
            response, outputs = self.run_with_calls(
                [("search_emails", {"query": "budget", "limit": 20})],
                final="claim [e19#c19]", limit=1000)
        # The full version/hash/coverage envelope cannot fit this budget.
        # Missing refs are valid only for this explicit failure, never success.
        self.assertEqual(outputs[0]["status"], "error")
        self.assertEqual(outputs[0]["error_code"], "output_budget_exceeded")
        self.assertEqual(response.metadata["steps"][0]["output_error_code"], "output_budget_exceeded")
        self.assertEqual(response.metadata["model_visible_evidence"], [])
        self.assertEqual(response.metadata["invalid_citation_count"], 1)
        self.assertNotEqual(response.metadata["status"], "success")

    def test_json_is_bounded_and_original_data_is_unchanged(self):
        original = normalize_tool_result([{"email_id": "e1", "chunk_id": "c1", "snippet": '"\\\n中文' * 3000}])
        original["evidence_refs"] = [{"email_id": "old", "chunk_id": "old"}] * 1000
        before = copy.deepcopy(original)
        for limit in (2, 18, 100, 300, 1000, 4000):
            output = bounded_json(original, limit)
            self.assertLessEqual(len(output), limit)
            parsed = json.loads(output)
            self.assertIsInstance(parsed, dict)
            self.assertFalse(any(row["email_id"] == "old" for row in parsed.get("evidence_refs", [])))
        self.assertEqual(original, before)


if __name__ == "__main__":
    unittest.main()
