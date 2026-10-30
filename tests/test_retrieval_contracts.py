"""Offline contract tests: real domain modules, synthetic storage/model doubles.

Run without pytest: python -B -m unittest discover -s tests -p test_retrieval_contracts.py
No model, Gmail, Chroma service or network is contacted.
"""
from __future__ import annotations

import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

import config.settings as cfg
import core.embedder as embedder
import core.generator as generator
import core.pipeline as pipeline
import core.retriever as retriever
import agents.graph_workflow as graph
import agents.writer_agent as writer
import agents.summarizer_agent as summary
import agents.coordinator as coordinator
import agents.retriever_agent as retriever_agent
from agents.runtime import RunContext, RunDeadlineExceeded, ContextBudgetExceeded, use_run_context
from models.schemas import AgentRequest, AgentResponse, EmailChunk, SearchResult, IntentType


def hit(cid="e1_chunk_0", email_id="e1", content="synthetic content", **metadata):
    return SearchResult(chunk_id=cid, email_id=email_id, content=content, score=1.0, metadata=metadata)


def fake_client(content="done"):
    create = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content), finish_reason="stop")]))
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


class FakeCollection:
    def __init__(self, chunks=()):
        self.rows = {row["chunk_id"]: dict(row) for row in chunks}
        self.get_calls = []

    def get(self, *, where=None, include=None):
        self.get_calls.append(where)
        rows = [row for row in self.rows.values()
                if not where or row["metadata"].get("email_id") == where["email_id"]]
        return {"ids": [row["chunk_id"] for row in rows],
                "documents": [row["content"] for row in rows],
                "metadatas": [row["metadata"] for row in rows]}

    def upsert(self, *, ids, embeddings, documents, metadatas):
        for cid, doc, meta in zip(ids, documents, metadatas):
            self.rows[cid] = {"chunk_id": cid, "content": doc, "metadata": meta}

    def delete(self, *, ids):
        for cid in ids:
            self.rows.pop(cid, None)


class RetrievalContracts(unittest.TestCase):
    def test_provider_error_logs_exclude_exception_text(self):
        cases = [
            (pipeline, lambda: pipeline.rewrite_query("synthetic query")),
            (generator, lambda: generator.generate_answer("synthetic query", [hit()])),
            (coordinator, lambda: coordinator.classify_intent("synthetic query")),
            (graph, lambda: graph.node_rewrite({"query": "synthetic query"})),
        ]
        for module, invoke in cases:
            client = fake_client()
            client.chat.completions.create.side_effect = RuntimeError("SYNTHETIC_PRIVATE_CANARY")
            with self.subTest(module=module.__name__), \
                 patch.object(cfg, "ENABLE_QUERY_REWRITE", True), \
                 patch.object(module, "_get_client", return_value=client), \
                 self.assertLogs(module.logger, level="WARNING") as logs:
                invoke()
            self.assertIn("error_type=RuntimeError", " ".join(logs.output))
            self.assertNotIn("SYNTHETIC_PRIVATE_CANARY", " ".join(logs.output))

    def test_preclassified_route_does_not_call_classifier_again(self):
        with patch.object(coordinator, "classify_intent") as classify, \
             patch.object(retriever_agent.RetrieverAgent, "run", return_value=AgentResponse(answer="done")):
            response = coordinator.route(AgentRequest(query="q"), intent=IntentType.GENERAL)
        classify.assert_not_called()
        self.assertEqual(response.intent, IntentType.GENERAL)

    def test_filters_preserve_empty_and_require_all_labels(self):
        rows = [hit(sender="a@example.test", labels='["work"]')]
        self.assertEqual(pipeline.apply_post_filters(rows, {"sender": "missing"}), [])
        self.assertEqual(pipeline.apply_post_filters(rows, {"labels": ["work", "urgent"]}), [])
        self.assertEqual(pipeline.apply_post_filters(rows, {"labels": ["work"]}), rows)

    def test_yesterday_uses_calendar_day_and_converts_utc(self):
        now = datetime(2026, 9, 9, 12, tzinfo=timezone(timedelta(hours=8)))
        rows = [hit("before", date="2026-09-07T15:59:59Z"),
                hit("start", date="2026-09-07T16:00:00Z"),
                hit("end", date="2026-09-08T16:00:00Z")]
        with patch.object(pipeline, "_now", return_value=now):
            out = pipeline.apply_post_filters(rows, {"date_hint": "昨天"})
        self.assertEqual([row.chunk_id for row in out], ["start"])

    def test_previous_month_handles_year_boundary(self):
        now = datetime(2026, 1, 3, tzinfo=timezone.utc)
        with patch.object(pipeline, "_now", return_value=now):
            start, end = pipeline._date_window("上月")
        self.assertEqual(start, datetime(2025, 12, 1, tzinfo=timezone.utc))
        self.assertEqual(end, datetime(2026, 1, 1, tzinfo=timezone.utc))

    def test_unknown_date_and_bad_filter_types_fail_closed(self):
        for filters in ({"date_hint": "some day"}, {"labels": "work"},
                        {"sender": 42}, {"date_hint": "2026-09-10至2026-09-01"}):
            with self.subTest(filters=filters), self.assertRaises(ValueError):
                pipeline.apply_post_filters([], filters)

    def test_explicit_filters_skip_llm_and_never_relax(self):
        rows = [hit(sender="other@example.test")]
        with patch.object(pipeline, "rewrite_query") as rewrite, \
             patch.object(pipeline, "extract_filters") as extract, \
             patch.object(pipeline, "hybrid_search", return_value=rows), \
             patch.object(pipeline, "rerank", side_effect=lambda q, rows, top_n: rows):
            self.assertEqual(pipeline.retrieve("query", filters={"sender": "missing"}), [])
        rewrite.assert_not_called()
        extract.assert_not_called()

    def test_invalid_extraction_does_not_fall_back_to_unfiltered(self):
        with patch.object(pipeline, "_get_client", return_value=fake_client("[]")):
            with self.assertRaises(ValueError):
                pipeline.extract_filters("synthetic request")

    def test_history_reaches_retrieval_resolution(self):
        history = [{"role": "user", "content": "synthetic prior target"}]
        with patch.object(pipeline, "rewrite_query", return_value="resolved") as rewrite, \
             patch.object(pipeline, "extract_filters", return_value={}), \
             patch.object(pipeline, "hybrid_search", return_value=[]), \
             patch.object(pipeline, "rerank", return_value=[]):
            pipeline.retrieve("that email", history=history)
        rewrite.assert_called_once_with("that email", history=history)

    def test_writer_keeps_search_and_instruction_separate(self):
        source = hit()
        email = {"body": "complete indexed text", "chunks": [source.model_dump()]}
        with patch.object(writer, "retrieve", return_value=[source]) as retrieve, \
             patch.object(writer, "get_indexed_email", return_value=email) as load, \
             patch.object(writer, "draft_reply_for_email", return_value="draft") as draft:
            result = writer.WriterAgent().run(AgentRequest(query="locate target"),
                                              instruction="decline politely", filters={"sender": "a"})
        self.assertEqual(retrieve.call_args.args[0], "locate target")
        self.assertEqual(retrieve.call_args.kwargs["filters"], {"sender": "a"})
        load.assert_called_once_with("e1")
        draft.assert_called_once_with(email, "decline politely")
        self.assertEqual(result.sources, [source])

    def test_summary_uses_shared_pipeline_and_explicit_filters(self):
        agent = summary.SummarizerAgent.__new__(summary.SummarizerAgent)
        agent._client = fake_client()
        with patch.object(summary, "retrieve", return_value=[]) as retrieve:
            result = agent.run(AgentRequest(query="summary target"), filters={"labels": ["work"]})
        self.assertEqual(retrieve.call_args.kwargs["filters"], {"labels": ["work"]})
        self.assertEqual(result.sources, [])
        agent._client.chat.completions.create.assert_not_called()

    def test_graph_never_generates_from_rejected_candidates(self):
        state = {"query": "q", "results": [hit()], "relevant_results": [], "retry_count": 2}
        with patch.object(graph, "generate_answer", return_value="no evidence") as generate:
            graph.node_generate(state)
        self.assertEqual(generate.call_args.args[1], [])
        with patch.object(graph, "get_graph", return_value=SimpleNamespace(invoke=lambda _: state)), \
             patch.object(coordinator, 'classify_intent', return_value='retrieve'):
            response = graph.run_graph(AgentRequest(query="q"))
        self.assertEqual(response.sources, [])
        self.assertFalse(response.metadata["grounded"])

    def test_invalid_grader_output_is_rejected(self):
        state = {"query": "q", "results": [hit()]}
        with patch.object(graph, "_get_client", return_value=fake_client("[true]")):
            graph.node_grade_contexts(state)
        self.assertEqual(state["relevant_results"], [])
        self.assertEqual(state["grading_error"], "context_grading_failed")

    def test_graph_does_not_repeat_same_query_when_rewrite_disabled(self):
        with patch.object(cfg, "ENABLE_QUERY_REWRITE", False):
            self.assertEqual(graph._should_retry({"relevant_results": [], "retry_count": 0}), "generate")

    def test_details_query_only_one_email_and_preserve_evidence_ids(self):
        import hashlib
        rows = [{"chunk_id": "e1_1", "content": "defghi", "metadata": {"email_id": "e1", "chunk_index": 1}},
                {"chunk_id": "e1_0", "content": "abcdef", "metadata": {"email_id": "e1", "chunk_index": 0}},
                {"chunk_id": "e2_0", "content": "other", "metadata": {"email_id": "e2", "chunk_index": 0}}]
        for row, start in zip(rows[:2], [3, 0]):
            row["metadata"].update(source_start=start, source_end=start + 6, source_length=9,
                                   source_sha256=hashlib.sha256(b"abcdefghi").hexdigest())
        collection = FakeCollection(rows)
        with patch.object(embedder, "_get_collection", return_value=collection), patch.object(cfg, "CHUNK_OVERLAP", 3):
            email = embedder.get_indexed_email("e1")
        self.assertEqual(collection.get_calls, [{"email_id": "e1"}])
        self.assertEqual(email["body"], "abcdefghi")
        self.assertEqual([c["chunk_id"] for c in email["chunks"]], ["e1_0", "e1_1"])
        self.assertEqual(email["body_source"], "indexed_chunks")
        self.assertTrue(email["reconstruction_exact"])

    def test_index_generations_revision_and_batch_alignment(self):
        from tests.index_store_helpers import MemoryClient
        client = MemoryClient()
        chunks = [EmailChunk(chunk_id=f"e{i}_0", email_id=f"e{i}", content=str(i), chunk_index=0)
                  for i in range(6)]
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(cfg, "CHROMA_PERSIST_DIR", directory), \
             patch.object(cfg, "EMBEDDING_MODEL_REVISION", "offline-v1", create=True), \
             patch.object(embedder, "_get_client", return_value=client), \
             patch.object(embedder, "embed_texts", side_effect=lambda texts: [[float(t)] for t in texts]) as encode:
            embedder._write_marker_path().write_text("interrupted", encoding="ascii")
            self.assertNotEqual(embedder.get_corpus_revision(), embedder.get_corpus_revision())
            self.assertEqual(embedder.index_chunks(chunks, batch_size=4), 6)
            self.assertEqual([call.args[0] for call in encode.call_args_list], [["0", "1", "2", "3"], ["4", "5"]])
            collection = embedder._get_collection()
            for i in range(6):
                self.assertEqual(collection.rows[f"e{i}_0"]["embedding"], [float(i)])
            version = embedder.get_corpus_revision()
            self.assertEqual(version, embedder.get_corpus_revision())
            with patch.object(embedder, "embed_texts", side_effect=RuntimeError("synthetic failure")):
                with self.assertRaises(RuntimeError):
                    # A deliberate forced calculation exercises failure here;
                    # an unchanged ordinary update now correctly skips encode.
                    embedder.index_chunks([chunks[0]], force_reembed=True)
            self.assertEqual(version, embedder.get_corpus_revision())
            self.assertEqual(collection, embedder._get_collection())
            embedder.index_chunks([chunks[0].model_copy(update={"content": "9"})])
            self.assertNotEqual(version, embedder.get_corpus_revision())
            self.assertEqual(embedder._get_collection().rows["e0_0"]["content"], "9")
            self.assertEqual(collection.rows["e0_0"]["content"], "0")

    def test_details_do_not_remove_incidental_short_prefix(self):
        rows = [{"chunk_id": "a", "content": "partx", "metadata": {"email_id": "e1", "chunk_index": 0}},
                {"chunk_id": "b", "content": "xnext", "metadata": {"email_id": "e1", "chunk_index": 1}}]
        with patch.object(embedder, "_get_collection", return_value=FakeCollection(rows)), patch.object(cfg, "CHUNK_OVERLAP", 3):
            self.assertEqual(embedder.get_indexed_email("e1")["body"], "partx\n\nxnext")

    def test_bm25_rebuilds_when_revision_changes_without_count_change(self):
        built = []
        class FakeBM25:
            def __init__(self, corpus):
                built.append(corpus)
        rows = [{"chunk_id": "c", "content": "synthetic", "metadata": {"email_id": "e"}}]
        revision = ["v1"]
        retriever.invalidate_bm25_cache()
        with patch.dict(sys.modules, {"rank_bm25": SimpleNamespace(BM25Okapi=FakeBM25)}), \
             patch.object(retriever, "get_collection_count", return_value=1), \
             patch.object(retriever, "get_all_chunks", return_value=rows), \
             patch.object(retriever, "get_corpus_revision", side_effect=lambda: revision[0]):
            retriever._get_bm25_index()
            retriever._get_bm25_index()
            revision[0] = "v2"
            retriever._get_bm25_index()
        retriever.invalidate_bm25_cache()
        self.assertEqual(len(built), 2)

    def test_generation_deadline_is_not_swallowed_by_fallback(self):
        client = fake_client()
        with use_run_context(RunContext(deadline=time.monotonic() - 1)), \
             patch.object(generator, "_get_client", return_value=client):
            with self.assertRaises(RunDeadlineExceeded):
                generator.generate_answer("q", [hit()])
        client.chat.completions.create.assert_not_called()

    def test_nested_context_budget_blocks_request(self):
        client = fake_client()
        with use_run_context(RunContext(context_char_limit=10)), \
             patch.object(writer, "_get_client", return_value=client):
            with self.assertRaises(ContextBudgetExceeded):
                writer.draft_reply_for_email({"body": "synthetic"}, "draft")
        client.chat.completions.create.assert_not_called()

    def test_stream_yields_only_answer_and_closes(self):
        class FakeStream:
            closed = False
            def __iter__(self):
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(reasoning_content="private", content=""))])
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(reasoning_content=None, content="answer"))])
                yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None), finish_reason="stop")])
            def close(self):
                self.closed = True
        stream = FakeStream()
        client = fake_client()
        client.chat.completions.create.return_value = stream
        with patch.object(generator, "_get_client", return_value=client):
            self.assertEqual(list(generator.stream_generate("q", [hit()])), ["answer"])
        self.assertTrue(stream.closed)


if __name__ == "__main__":
    unittest.main()
