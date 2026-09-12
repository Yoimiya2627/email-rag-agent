"""Exact final-prompt source ranges for generator, stream, summary and writer."""
import json
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

import config.settings as cfg
from agents.runtime import RunContext, ContextBudgetExceeded, use_run_context
from core.evidence import text_hash
from models.schemas import AgentRequest, SearchResult


def source(content="A" * 1000 + "UNREAD_TAIL", *, table=False, chunk="c1"):
    metadata = {"email_id": "e1", "subject": "synthetic", "index_generation": "g1",
                "source_sha256": text_hash(content)}
    if table:
        metadata["table_context"] = json.dumps([{"table_id": "t1", "row_id": "r1", "status": "parsed",
            "cells": [{"source_id": "cell1", "column": 0, "headers": ["budget"]}]}])
    return SearchResult(email_id="e1", chunk_id=chunk, content=content, metadata=metadata, score=1)


def client(response=None):
    response = response or NS(choices=[NS(finish_reason="stop", message=NS(content="answer"))])
    return NS(chat=NS(completions=NS(create=Mock(return_value=response))))


def assert_prefix_ref(ref, original, expected_length=None):
    assert ref["source_version"] == "g1"
    assert ref["visible_start"] == 0
    assert ref["visible_hash"] == text_hash(original[:ref["visible_end"]])
    if expected_length is not None:
        assert ref["visible_end"] == expected_length


def test_generator_records_only_prefix_actually_sent_and_has_citable_ids(monkeypatch):
    from core import generator
    row, model, run = source(), client(), RunContext()
    monkeypatch.setattr(generator, "_get_client", lambda: model)
    monkeypatch.setattr(cfg, "GENERATION_CONTEXT_CHAR_LIMIT", 60)
    with use_run_context(run):
        generator.generate_answer("q", [row])
    assert len(run.generation_evidence_refs) == 1
    assert_prefix_ref(run.generation_evidence_refs[0], row.content, 60)
    prompt = model.chat.completions.create.call_args.kwargs["messages"][-1]["content"]
    assert "UNREAD_TAIL" not in prompt and "[e1#c1]" in prompt
    assert "evidence_refs" not in model.chat.completions.create.call_args.kwargs


def test_table_note_consumes_budget_but_never_counts_as_visible_original_text(monkeypatch):
    from core import generator
    row, model, run = source(table=True), client(), RunContext()
    monkeypatch.setattr(generator, "_get_client", lambda: model)
    monkeypatch.setattr(cfg, "GENERATION_CONTEXT_CHAR_LIMIT", 500)
    with use_run_context(run):
        generator.generate_answer("q", [row])
    ref = run.generation_evidence_refs[0]
    assert 0 < ref["visible_end"] < 500
    assert_prefix_ref(ref, row.content)
    prompt = model.chat.completions.create.call_args.kwargs["messages"][-1]["content"]
    assert "budget" in prompt and "UNREAD_TAIL" not in prompt


def test_zero_space_for_complete_table_note_records_no_false_visible_source(monkeypatch):
    from core import generator
    row, model, run = source(table=True), client(), RunContext()
    monkeypatch.setattr(generator, "_get_client", lambda: model)
    monkeypatch.setattr(cfg, "GENERATION_CONTEXT_CHAR_LIMIT", 20)
    with use_run_context(run):
        generator.generate_answer("q", [row])
    assert run.generation_evidence_refs == []


def test_failed_provider_or_rejected_context_does_not_claim_model_visibility(monkeypatch):
    from core import generator
    row, model, run = source(), client(), RunContext()
    model.chat.completions.create.side_effect = RuntimeError("offline failure")
    monkeypatch.setattr(generator, "_get_client", lambda: model)
    with use_run_context(run):
        generator.generate_answer("q", [row])
    assert run.generation_evidence_refs == []
    tiny = RunContext(context_char_limit=10)
    with use_run_context(tiny), pytest.raises(ContextBudgetExceeded):
        generator.generate_answer("q", [row])
    assert tiny.generation_evidence_refs == []


def test_stream_records_exact_prompt_prefix_after_provider_accepts(monkeypatch):
    from core import generator
    row, run = source(), RunContext()
    chunks = [NS(choices=[NS(finish_reason="stop", delta=NS(content="answer"))])]
    model = client(iter(chunks))
    monkeypatch.setattr(generator, "_get_client", lambda: model)
    monkeypatch.setattr(cfg, "GENERATION_CONTEXT_CHAR_LIMIT", 80)
    with use_run_context(run):
        stream = generator.stream_generate("q", [row])
        assert run.generation_evidence_refs == []
        assert list(stream) == ["answer"]
    assert_prefix_ref(run.generation_evidence_refs[0], row.content, 80)


def test_writer_records_only_chunks_actually_in_final_body(monkeypatch):
    from agents import writer_agent
    row, absent, model, run = source("real visible body"), source("unseen other chunk", chunk="c2"), client(), RunContext()
    email = {"email_id": "e1", "body": row.content, "chunks": [row.model_dump(), absent.model_dump()]}
    monkeypatch.setattr(writer_agent, "_get_client", lambda: model)
    with use_run_context(run):
        writer_agent.draft_reply_for_email(email, "reply")
    assert len(run.generation_evidence_refs) == 1
    assert_prefix_ref(run.generation_evidence_refs[0], row.content, len(row.content))


def test_summary_records_exact_rendered_source_and_keeps_candidate_scope(monkeypatch):
    from agents import summarizer_agent
    row, model, run = source("table row content", table=True), client(), RunContext()
    monkeypatch.setattr(summarizer_agent, "OpenAI", Mock(return_value=model))
    monkeypatch.setattr(summarizer_agent, "retrieve", lambda *args, **kwargs: [row])
    with use_run_context(run):
        result = summarizer_agent.SummarizerAgent().run(AgentRequest(query="q"))
    assert_prefix_ref(run.generation_evidence_refs[0], row.content, len(row.content))
    assert result.metadata["model_visible_evidence"] == run.generation_evidence_refs
    assert result.metadata["sources_kind"] == "retrieved_candidates"
    assert result.metadata["coverage"]["mailbox_complete"] is False


def test_exact_refs_shared_with_parent_and_preserved_in_agent_checkpoint(monkeypatch):
    import agents.agent_loop as loop
    from core.model_clients import create_completion
    row, model, parent = source("visible text"), client(), RunContext(owner_id="local", session_id="s")
    from core.evidence import evidence_reference
    ref = evidence_reference(row.model_dump())
    def fake_tool(name, args):
        create_completion(model, stage="summarize", model="offline", messages=[], max_tokens=10, evidence_refs=[ref])
        return "summary"
    responses = iter([NS(choices=[NS(finish_reason="tool_calls", message=NS(content=None, tool_calls=[
        NS(id="call1", type="function", function=NS(name="summarize_emails", arguments='{"query":"q"}'))]))]),
        NS(choices=[NS(finish_reason="stop", message=NS(content="answer", tool_calls=None))])])
    monkeypatch.setattr(loop, "_get_client", lambda: client())
    scripted = NS(chat=NS(completions=NS(create=lambda **kwargs: next(responses))))
    monkeypatch.setattr(loop, "_get_client", lambda: scripted)
    monkeypatch.setattr(loop, "call_tool", fake_tool)
    saved = []
    parent.checkpoint_callback = lambda payload: saved.append(json.loads(json.dumps(payload)))
    with use_run_context(parent):
        loop.run_agent_loop(AgentRequest(query="q", session_id="s"))
    assert parent.generation_evidence_refs == [ref]
    assert saved[-1]["generation_evidence_refs"] == [ref]
