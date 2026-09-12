"""Long conversations must retain complete turns and leave room for the current run."""
import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agents.runtime import RunContext, ContextBudgetExceeded, use_run_context
from core.memory import ConversationMemory, conversation_pairs, build_model_messages, fit_messages_to_budget
from models.schemas import AgentRequest, SearchResult
import config.settings as cfg


def client(answer):
    reply = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=answer), finish_reason='stop')])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=Mock(return_value=reply))))


def history():
    return [{'role':role, 'content':f'{index}-{role}-' + 'x' * 1200}
            for index in range(5) for role in ('user', 'assistant')]


def source():
    return SearchResult(email_id='e', chunk_id='e_0', content='Approved budget 500', score=1, metadata={})


class ContextHistoryTests(unittest.TestCase):
    def test_only_complete_conversation_turns_are_used(self):
        rows = [{'role':'assistant', 'content':'orphan'}, {'role':'user', 'content':'old'},
                {'role':'user','content':'keep'}, {'role':'assistant','content':'answer'},
                {'role':'system','content':'ignore'}, {'role':'tool','content':'ignore'},
                {'role':'user','content':'incomplete'}]
        self.assertEqual(conversation_pairs(rows), rows[2:4])

    def test_trim_oldest_pairs_without_mutating_history(self):
        rows = history()
        before = copy.deepcopy(rows)
        with use_run_context(RunContext(context_char_limit=3000)):
            messages = build_model_messages('system', 'current', rows)
        self.assertEqual(messages[1:-1], rows[-2:])
        self.assertEqual(rows, before)
        self.assertEqual(messages[-1]['content'], 'current')

    def test_tool_schemas_and_current_tool_chain_are_preserved(self):
        chain = [{'role':'user','content':'current'},
                 {'role':'assistant','tool_calls':[{'id':'tc','type':'function','function':{'name':'tool','arguments':'{}'}}]},
                 {'role':'tool','tool_call_id':'tc','content':'evidence' * 30}]
        system = {'role':'system','content':'system'}
        schemas = [{'schema':'x' * 500}]
        minimum = len(json.dumps([[system, *chain], schemas], ensure_ascii=False))
        messages = [system, *history(), *chain]
        fitted, count = fit_messages_to_budget(messages, 10, schemas=schemas, char_limit=minimum)
        self.assertEqual(fitted, [system, *chain])
        self.assertEqual(count, 0)
        self.assertEqual(len(messages), 14)
        with self.assertRaises(ContextBudgetExceeded):
            fit_messages_to_budget(fitted, count, schemas=schemas, char_limit=minimum-1)

    def test_single_oversized_current_request_is_rejected(self):
        with use_run_context(RunContext(context_char_limit=100)), self.assertRaises(ContextBudgetExceeded):
            build_model_messages('system', 'current' * 100, history())

    def test_classifier_and_rewriters_fit_history_before_model_calls(self):
        from agents import coordinator, graph_workflow
        from core import pipeline
        cases = [(coordinator, '{"intent":"retrieve"}', lambda: coordinator.classify_intent('current', history())),
                 (pipeline, 'resolved', lambda: pipeline.rewrite_query('current', history())),
                 (graph_workflow, 'resolved', lambda: graph_workflow.node_rewrite({'query':'current','history':history()}))]
        for module, answer, invoke in cases:
            model = client(answer)
            with self.subTest(module=module.__name__), use_run_context(RunContext(context_char_limit=4000)), \
                 patch.object(module, '_get_client', return_value=model):
                invoke()
            sent = model.chat.completions.create.call_args.kwargs['messages']
            self.assertEqual(sent[-1]['content'], 'current')
            self.assertEqual(sent[1:-1], history()[-2:])

    def test_summary_and_both_generators_fit_history(self):
        from core import generator
        from agents import summarizer_agent
        memory = ConversationMemory()
        for row in history():
            memory.add(row['role'], row['content'])
        model = client('answer')
        with use_run_context(RunContext(context_char_limit=4000)), \
             patch.object(generator, '_get_client', return_value=model):
            self.assertEqual(generator.generate_answer('current', [source()], history=memory.to_messages()), 'answer')
        self.assertEqual(model.chat.completions.create.call_args.kwargs['messages'][1:-1], history()[-2:])
        stream = Mock()
        stream.__iter__ = lambda _: iter([
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content='token'), finish_reason=None)]),
            SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None), finish_reason='stop')]),
        ])
        model.chat.completions.create.return_value = stream
        with use_run_context(RunContext(context_char_limit=4000)), patch.object(generator, '_get_client', return_value=model):
            self.assertEqual(list(generator.stream_generate('current', [source()], history=memory.to_messages())), ['token'])
        stream.close.assert_called_once()
        self.assertEqual(model.chat.completions.create.call_args.kwargs['messages'][1:-1], history()[-2:])
        model = client('summary')
        with use_run_context(RunContext(context_char_limit=4000)), patch.object(summarizer_agent, 'OpenAI', return_value=model), \
             patch.object(summarizer_agent, 'retrieve', return_value=[source()]):
            response = summarizer_agent.SummarizerAgent().run(AgentRequest(query='current'), memory)
        self.assertEqual(response.answer, 'summary')
        self.assertEqual(model.chat.completions.create.call_args.kwargs['messages'][1:-1], history()[-2:])

    def test_actual_api_recovers_both_original_long_dialogue_sequences(self):
        import api.main as api
        from api.sessions import SessionStore
        from api.security import Identity
        for lengths, answer_size in [([18000,18000,20000,2,2], 1500), ([4000]*5+[20000,2], 4000)]:
            intent, answer = client('{"intent":"retrieve"}'), client('a' * answer_size)
            with self.subTest(lengths=lengths), patch.object(api, 'sessions', SessionStore()), \
                 patch.object(cfg, 'AGENT_CONTEXT_CHAR_LIMIT', 60000), \
                 patch('agents.coordinator._get_client', return_value=intent), \
                 patch('agents.retriever_agent.retrieve', return_value=[source()]), \
                 patch('core.generator._get_client', return_value=answer):
                for length in lengths:
                    response = api.chat(AgentRequest(query='Q'*length, session_id='long'), Identity('local'))
                    self.assertEqual(len(response.answer), answer_size)
            self.assertEqual(intent.chat.completions.create.call_count, len(lengths))
            self.assertEqual(answer.chat.completions.create.call_count, len(lengths))

    def test_actual_api_oversize_current_input_returns_413_without_memory_commit(self):
        import api.main as api
        from api.sessions import SessionStore
        from api.security import Identity
        from fastapi import HTTPException
        with patch.object(api, 'sessions', SessionStore()), patch.object(cfg, 'AGENT_CONTEXT_CHAR_LIMIT', 100):
            with self.assertRaises(HTTPException) as caught:
                api.chat(AgentRequest(query='Q'*1000, session_id='too-large'), Identity('local'))
            with api.sessions.turn('local', 'too-large') as memory:
                self.assertEqual(memory.to_messages(), [])
        self.assertEqual(caught.exception.status_code, 413)

    def test_agent_loop_drops_old_turns_as_tool_messages_grow(self):
        from agents import agent_loop as loop
        memory = ConversationMemory()
        for index in range(2):
            memory.add('user', f'old-{index}-' + 'x'*500)
            memory.add('assistant', f'answer-{index}-' + 'y'*500)
        schemas = [{'type':'function','function':{'name':'inspect','description':'read',
                   'parameters':{'type':'object','properties':{},'additionalProperties':False}}}]
        first_messages = [{'role':'system','content':loop._SYSTEM}, *memory.to_messages(),
                          {'role':'user','content':'current'}]
        budget = len(json.dumps([first_messages, schemas], ensure_ascii=False)) + 30
        requests = []
        tool_call = SimpleNamespace(id='call-1', function=SimpleNamespace(name='inspect', arguments='{}'))
        replies = [SimpleNamespace(content='',tool_calls=[tool_call]), SimpleNamespace(content='done',tool_calls=None)]
        def create(**kwargs):
            requests.append(copy.deepcopy(kwargs['messages']))
            return SimpleNamespace(choices=[SimpleNamespace(message=replies.pop(0))])
        model = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        backend = SimpleNamespace(tool_schemas=lambda:schemas,
                                  call_tool=lambda *_:{'detail':'evidence-' + 'z'*400})
        with patch.object(loop, '_get_client', return_value=model), patch.object(loop, '_get_tool_backend', return_value=backend), \
             patch.multiple(cfg, AGENT_CONTEXT_CHAR_LIMIT=budget, ENABLE_AGENT_TRACE=False, AGENT_MAX_STEPS=3):
            response = loop.run_agent_loop(AgentRequest(query='current'), memory)
        self.assertEqual(response.metadata['status'], 'success')
        self.assertEqual(response.metadata['actual_tool_calls'], 1)
        self.assertEqual(requests[0], first_messages)
        self.assertEqual(requests[1][1:3], memory.to_messages()[-2:])
        self.assertEqual(requests[1][-2]['tool_calls'][0]['id'], 'call-1')
        self.assertEqual(requests[1][-1]['tool_call_id'], 'call-1')
        self.assertIn('evidence-', requests[1][-1]['content'])
        self.assertEqual(len(memory.to_messages()), 4)


if __name__ == '__main__':
    unittest.main()
