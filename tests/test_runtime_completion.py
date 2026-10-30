"""Completion and second-turn context contracts; no model, mailbox or index I/O."""
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from agents import agent_loop, analyzer_agent, summarizer_agent, writer_agent, retriever_agent
from agents.runtime import RunContext, RunDeadlineExceeded, normalize_tool_result, use_run_context
from core import generator
from core.memory import ConversationMemory
from core.model_outcomes import (ModelText, ModelOutputError, outcome_metadata,
                                 text_from_choice, text_from_response)
from models.schemas import AgentRequest, AgentResponse, SearchResult
import config.settings as cfg


def source():
    return SearchResult(email_id="synthetic", chunk_id="synthetic_0", content="Original email",
                        score=1, metadata={})


def reply(content="answer", finish="stop", calls=None):
    return NS(choices=[NS(message=NS(content=content, tool_calls=calls,
                                    reasoning_content="private reasoning"), finish_reason=finish)])


def client(response):
    return NS(chat=NS(completions=NS(create=Mock(return_value=response))))


def delta(content=None, finish=None):
    return NS(choices=[NS(delta=NS(content=content, reasoning_content="private reasoning"),
                         finish_reason=finish)])


class Stream:
    def __init__(self, items, fail=None):
        self.items, self.fail, self.closed = items, fail, False

    def __iter__(self):
        yield from self.items
        if self.fail:
            raise self.fail

    def close(self):
        self.closed = True


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.trace = patch.object(cfg, "ENABLE_AGENT_TRACE", False)
        self.trace.start()
        self.addCleanup(self.trace.stop)

    def test_choice_length_is_incomplete_and_reasoning_is_never_answer(self):
        text = text_from_choice(reply("partial", "length").choices[0])
        self.assertEqual(text, "partial")
        self.assertEqual(outcome_metadata(text)["status"], "incomplete")
        self.assertEqual(text.finish_reason, "length")
        for finish in ("stop", "length", "content_filter"):
            with self.subTest(finish=finish):
                empty = text_from_choice(reply(None, finish).choices[0])
                self.assertEqual(empty, "")
                self.assertEqual(outcome_metadata(empty)["status"], "empty_model_response")

    def test_bare_empty_string_and_serialized_outcome_do_not_become_success(self):
        self.assertEqual(outcome_metadata("  ")["status"], "empty_model_response")
        self.assertEqual(outcome_metadata(text_from_response(NS(answer="legacy answer")))["status"], "success")
        self.assertEqual(outcome_metadata(text_from_response(NS(answer="  ")))["status"], "empty_model_response")
        original = ModelText("partial", completion_status="incomplete", finish_reason="length",
                             error_code="model_output_incomplete")
        response = AgentResponse(answer=original, metadata=outcome_metadata(original))
        restored = text_from_response(response)
        result = normalize_tool_result(restored)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["data"], {"partial_text": "partial"})
        self.assertEqual(result["finish_reason"], "length")

    def test_agent_truncated_planning_call_does_not_execute_a_tool(self):
        call = NS(id="call-one", function=NS(name="inspect", arguments="{}"))
        backend = NS(tool_schemas=lambda: [], call_tool=Mock())
        model = client(reply("partial plan", "length", [call]))
        with patch.object(agent_loop, "_get_client", return_value=model), \
             patch.object(agent_loop, "_get_tool_backend", return_value=backend):
            response = agent_loop.run_agent_loop(AgentRequest(query="synthetic"))
        self.assertEqual(response.metadata["status"], "incomplete")
        self.assertEqual(response.metadata["finish_reason"], "length")
        self.assertEqual(response.metadata["actual_tool_calls"], 0)
        backend.call_tool.assert_not_called()
        self.assertEqual(model.chat.completions.create.call_count, 1)

    def test_agent_final_length_and_empty_have_reliable_terminal_states(self):
        for content, finish, status in [("done", "stop", "success"),
                                         ("partial", "length", "incomplete"),
                                         (None, "length", "empty_model_response")]:
            with self.subTest(content=content, finish=finish), \
                 patch.object(agent_loop, "_get_client", return_value=client(reply(content, finish))), \
                 patch.object(agent_loop, "_get_tool_backend", return_value=NS(tool_schemas=lambda: [])):
                response = agent_loop.run_agent_loop(AgentRequest(query="synthetic"))
                self.assertEqual(response.metadata["status"], status)
                self.assertNotIn("private reasoning", response.answer)

    def test_budget_finalization_retains_length_without_replaying_tools(self):
        backend = NS(tool_schemas=lambda: [], call_tool=Mock())
        model = client(reply("partial summary", "length"))
        with patch.object(cfg, "AGENT_MAX_STEPS", 0), \
             patch.object(agent_loop, "_get_client", return_value=model), \
             patch.object(agent_loop, "_get_tool_backend", return_value=backend):
            response = agent_loop.run_agent_loop(AgentRequest(query="synthetic"))
        self.assertEqual(response.metadata["status"], "max_steps_reached")
        self.assertEqual(response.metadata["completion_status"], "incomplete")
        self.assertEqual(response.metadata["finish_reason"], "length")
        backend.call_tool.assert_not_called()

    def test_generator_nonstream_outcomes_and_late_deadline(self):
        with patch.object(generator, "_get_client", return_value=client(reply("part", "length"))):
            answer = generator.generate_answer("synthetic", [source()])
        self.assertEqual(outcome_metadata(answer)["status"], "incomplete")
        with patch.object(generator, "_get_client", return_value=client(reply(None))):
            answer = generator.generate_answer("synthetic", [source()])
        self.assertEqual(outcome_metadata(answer)["status"], "empty_model_response")
        with patch.object(generator, "_get_client", return_value=client(reply())), \
             patch.object(generator, "remaining_timeout", side_effect=[30, RunDeadlineExceeded("expired")]):
            with self.assertRaises(RunDeadlineExceeded):
                generator.generate_answer("synthetic", [source()])

    def test_generation_fallback_is_marked_error_instead_of_success(self):
        model = client(reply())
        model.chat.completions.create.side_effect = RuntimeError("private provider details")
        with patch.object(generator, "_get_client", return_value=model):
            answer = generator.generate_answer("synthetic", [source()])
        self.assertIn("Original email", answer)
        self.assertEqual(outcome_metadata(answer)["status"], "error")
        self.assertNotIn("private provider details", str(outcome_metadata(answer)))

    def test_stream_stop_stays_token_compatible_and_closes(self):
        stream = Stream([delta("one"), delta(" two"), delta(finish="stop")])
        with patch.object(generator, "_get_client", return_value=client(stream)):
            self.assertEqual(list(generator.stream_generate("synthetic", [source()])), ["one", " two"])
        self.assertTrue(stream.closed)

    def test_stream_incomplete_preserves_partial_and_safe_error(self):
        for finish in ("length", "content_filter", None):
            with self.subTest(finish=finish):
                stream = Stream([delta("private partial answer"), delta(finish=finish)])
                seen = []
                with patch.object(generator, "_get_client", return_value=client(stream)), \
                     self.assertRaises(ModelOutputError) as caught:
                    for token in generator.stream_generate("synthetic", [source()]):
                        seen.append(token)
                self.assertEqual(caught.exception.partial_text, "".join(seen))
                self.assertEqual(caught.exception.metadata["status"], "incomplete")
                self.assertNotIn("private partial", str(caught.exception))
                self.assertTrue(stream.closed)

    def test_stream_empty_and_reasoning_only_are_errors(self):
        for chunks in ([], [delta(), delta(finish="stop")], [delta("  "), delta(finish="length")]):
            with self.subTest(chunks=chunks):
                stream = Stream(chunks)
                with patch.object(generator, "_get_client", return_value=client(stream)), \
                     self.assertRaises(ModelOutputError) as caught:
                    list(generator.stream_generate("synthetic", [source()]))
                self.assertEqual(caught.exception.metadata["status"], "empty_model_response")
                self.assertNotIn("private reasoning", caught.exception.partial_text)
                self.assertTrue(stream.closed)

    def test_stream_mid_failure_preserves_text_and_deadline_is_not_disguised(self):
        for error in (RuntimeError("private failure"), RunDeadlineExceeded("expired")):
            with self.subTest(error=error):
                stream = Stream([delta("partial")], fail=error)
                wanted = RunDeadlineExceeded if isinstance(error, RunDeadlineExceeded) else ModelOutputError
                with patch.object(generator, "_get_client", return_value=client(stream)), \
                     self.assertRaises(wanted) as caught:
                    list(generator.stream_generate("synthetic", [source()]))
                if wanted is ModelOutputError:
                    self.assertEqual(caught.exception.partial_text, "partial")
                    self.assertEqual(caught.exception.completion_status, "error")
                self.assertTrue(stream.closed)

    def test_stream_cancellation_closes_provider(self):
        stream = Stream([delta("one"), delta("two"), delta(finish="stop")])
        with patch.object(generator, "_get_client", return_value=client(stream)):
            tokens = generator.stream_generate("synthetic", [source()])
            self.assertEqual(next(tokens), "one")
            tokens.close()
        self.assertTrue(stream.closed)


class SpecialistHistoryTests(unittest.TestCase):
    def memory(self):
        memory = ConversationMemory()
        memory.add("user", "Keep the price AUD 734 and the second paragraph.")
        memory.add("assistant", "Draft version one: the agreed second paragraph.")
        return memory

    def test_writer_keeps_previous_draft_and_constraints_in_generation(self):
        memory = self.memory()
        model = client(reply("partial draft", "length"))
        email = {"body": "Original email", "chunks": [source().model_dump()]}
        with patch.object(writer_agent, "retrieve", return_value=[source()]), \
             patch.object(writer_agent, "get_indexed_email", return_value=email), \
             patch.object(writer_agent, "_get_client", return_value=model):
            response = writer_agent.WriterAgent().run(AgentRequest(query="Make it more polite."), memory)
        sent = model.chat.completions.create.call_args.kwargs["messages"]
        self.assertEqual(sent[1:-1], memory.to_messages())
        self.assertIn("Make it more polite.", sent[-1]["content"])
        self.assertEqual(response.metadata["status"], "incomplete")

    def test_analyzer_resolves_second_turn_from_history_and_rejects_empty(self):
        memory, model = self.memory(), client(reply(None, "length"))
        with patch.object(analyzer_agent, "OpenAI", return_value=model), \
             patch.object(analyzer_agent, "compute_email_stats", return_value={"total_emails": 2}):
            response = analyzer_agent.AnalyzerAgent().run(AgentRequest(query="What about that sender?"), memory)
        sent = model.chat.completions.create.call_args.kwargs["messages"]
        self.assertEqual(sent[1:-1], memory.to_messages())
        self.assertEqual(response.metadata["total_emails"], 2)
        self.assertEqual(response.metadata["status"], "empty_model_response")
        self.assertTrue(response.answer)

    def test_writer_and_analyzer_trim_old_history_but_keep_current_question(self):
        memory = ConversationMemory()
        for n in range(4):
            memory.add("user", f"old-{n}" + "x" * 1000)
            memory.add("assistant", "y" * 1000)
        email = {"body": "Original email", "chunks": [source().model_dump()]}
        writer, analyzer = client(reply()), client(reply())
        with use_run_context(RunContext(context_char_limit=3000)), \
             patch.object(writer_agent, "retrieve", return_value=[source()]), \
             patch.object(writer_agent, "get_indexed_email", return_value=email), \
             patch.object(writer_agent, "_get_client", return_value=writer), \
             patch.object(analyzer_agent, "OpenAI", return_value=analyzer), \
             patch.object(analyzer_agent, "compute_email_stats", return_value={"total_emails": 2}):
            writer_agent.WriterAgent().run(AgentRequest(query="CURRENT"), memory)
            analyzer_agent.AnalyzerAgent().run(AgentRequest(query="CURRENT"), memory)
        for model in (writer, analyzer):
            sent = model.chat.completions.create.call_args.kwargs["messages"]
            self.assertEqual(sent[1:-1], memory.to_messages()[-2:])
            self.assertIn("CURRENT", sent[-1]["content"])

    def test_summary_and_retriever_preserve_completion_metadata(self):
        with patch.object(summarizer_agent, "OpenAI", return_value=client(reply("part", "length"))), \
             patch.object(summarizer_agent, "retrieve", return_value=[source()]):
            summary = summarizer_agent.SummarizerAgent().run(AgentRequest(query="summarize"))
        self.assertEqual(summary.metadata["status"], "incomplete")
        with patch.object(retriever_agent, "retrieve", return_value=[source()]), \
             patch.object(retriever_agent, "generate_answer", return_value=ModelText("")):
            response = retriever_agent.RetrieverAgent().run(AgentRequest(query="find"))
        self.assertEqual(response.metadata["status"], "empty_model_response")
        self.assertEqual(normalize_tool_result(text_from_response(summary))["status"], "error")


if __name__ == "__main__":
    unittest.main()
