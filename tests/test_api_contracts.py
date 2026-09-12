"""Offline integration checks: real ASGI app, sessions and SQLite; no network/model."""
import concurrent.futures
import json
import tempfile
import threading
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
import api.main as api
import config.settings as cfg
from agents.approvals import ApprovalStore
from agents.mail_providers import MailProviderError
from api.sessions import SessionBusyError, SessionStore
from models.schemas import AgentResponse, IntentType, SearchResult


class ApiContracts(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.temp = self.stack.enter_context(tempfile.TemporaryDirectory())
        for name, value in {'API_AUTH_TOKEN':'offline-test-token', 'API_OWNER_ID':'owner-a',
                            'MAIL_PROVIDER':'simulated', 'APPROVAL_STORE_PATH':str(Path(self.temp)/'actions.json')}.items():
            self.stack.enter_context(patch.object(cfg, name, value))
        self.stack.enter_context(patch.object(api, 'sessions', SessionStore()))
        self.client = self.stack.enter_context(TestClient(api.app))
        self.headers = {'Authorization':'Bearer offline-test-token'}

    def test_authentication_covers_private_and_mutating_endpoints(self):
        requests = [('get','/index/status',None), ('get','/agent/approvals',None),
                    ('get','/agent/mcp-audit',None), ('post','/index/clear',{}),
                    ('post','/chat',{'query':'hello'}), ('post','/query',{'query':'hello'}),
                    ('post','/chat/stream',{'query':'hello'}), ('delete','/chat/history?session_id=x',None)]
        for method, url, body in requests:
            with self.subTest(url=url):
                kwargs = {'json':body} if body is not None else {}
                self.assertEqual(self.client.request(method,url,**kwargs).status_code,401)
        self.assertEqual(self.client.get('/health').json(), {'status':'ok'})

    def test_no_token_allows_only_loopback_and_allowed_origin(self):
        from api.security import require_identity
        from starlette.requests import Request
        with patch.object(cfg,'API_AUTH_TOKEN',''):
            local = Request({'type':'http','path':'/index/status','client':('127.0.0.1',80),'headers':[(b'host',b'localhost')]})
            self.assertEqual(require_identity(local).owner_id,'owner-a')
            for host, headers in [('192.0.2.1',[(b'host',b'localhost')]),
                                  ('127.0.0.1',[(b'host',b'localhost'),(b'origin',b'https://evil.invalid')]),
                                  ('127.0.0.1',[(b'host',b'attacker.invalid:8000')])]:
                with self.assertRaises(Exception) as result:
                    require_identity(Request({'type':'http','path':'/index/status','client':(host,80),'headers':headers}))
                self.assertEqual(result.exception.status_code,403)

    def test_new_sessions_are_distinct_and_committed_once(self):
        seen=[]
        def runner(request,memory):
            seen.append(memory.to_messages())
            return AgentResponse(answer='answer')
        with patch.object(api,'route',runner):
            first=self.client.post('/chat',json={'query':'one'},headers=self.headers).json()
            second=self.client.post('/chat',json={'query':'two'},headers=self.headers).json()
            sid=first['metadata']['session_id']
            self.assertNotEqual(sid,second['metadata']['session_id'])
            self.client.post('/chat',json={'query':'three','session_id':sid},headers=self.headers)
        self.assertEqual(seen[:2],[[],[]])
        self.assertEqual(seen[2],[{'role':'user','content':'one'},{'role':'assistant','content':'answer'}])

    def test_running_session_returns_conflict_and_other_requests_progress(self):
        entered, release=threading.Event(),threading.Event()
        def slow(request,memory):
            entered.set()
            release.wait(5)
            return AgentResponse(answer='done')
        with patch.object(api,'route',slow), concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(self.client.post,'/chat',json={'query':'first','session_id':'same'},headers=self.headers)
            try:
                self.assertTrue(entered.wait(2))
                self.assertEqual(self.client.get('/health').status_code,200)
                conflict=self.client.post('/chat',json={'query':'second','session_id':'same'},headers=self.headers)
                self.assertEqual(conflict.status_code,409)
                # Context deletion now invalidates an active turn. The same
                # session still rejects concurrent chat, while late commit is
                # refused after the explicit delete rather than resurrecting it.
                self.assertEqual(self.client.delete('/chat/history?session_id=same',headers=self.headers).status_code,200)
            finally:
                release.set()
            self.assertEqual(future.result(timeout=3).status_code,409)
            self.assertEqual(self.client.get('/chat/history?session_id=same',headers=self.headers).json()['turns'],[])

    def test_approval_owner_reviewer_and_idempotency(self):
        store=ApprovalStore()
        mine=store.create('send_email',{'to':['a@example.com'],'subject':'s','body':'b'},owner_id='owner-a')
        other=store.create('send_email',{'to':['b@example.com'],'subject':'s','body':'b'},owner_id='owner-b')
        listed=self.client.get('/agent/approvals',headers=self.headers).json()['approvals']
        self.assertEqual([x['approval_id'] for x in listed],[mine['approval_id']])
        self.assertEqual(self.client.post(f"/agent/approvals/{other['approval_id']}/approve",json={},headers=self.headers).status_code,403)
        path=f"/agent/approvals/{mine['approval_id']}/approve"
        first=self.client.post(path,json={'reviewer':'forged-user'},headers=self.headers)
        self.assertEqual(first.status_code,200,first.text)
        self.assertEqual(first.json()['reviewer'],'owner-a')
        self.assertIs(first.json()['result']['sent'],False)
        self.assertEqual(self.client.post(path,json={},headers=self.headers).json()['result'],first.json()['result'])

    def test_provider_failure_is_unknown_and_retry_conflicts(self):
        pending=ApprovalStore().create('send_email',{'to':['a@example.com'],'subject':'s','body':'b'},owner_id='owner-a')
        class Broken:
            def execute_approval(self,item):
                raise MailProviderError('private diagnostic must not escape')
        path=f"/agent/approvals/{pending['approval_id']}/approve"
        with patch.object(api,'create_mail_provider_from_settings',return_value=Broken()):
            response=self.client.post(path,json={},headers=self.headers)
            self.assertEqual(response.status_code,502)
            self.assertNotIn('private diagnostic',response.text)
            self.assertEqual(self.client.post(path,json={},headers=self.headers).status_code,409)
        self.assertEqual(ApprovalStore().get(pending['approval_id'],owner_id='owner-a')['status'],'unknown')

    def test_stream_preserves_history_sources_and_answer_only(self):
        import core.pipeline as pipeline
        import core.generator as generator
        import agents.coordinator as coordinator
        source=SearchResult(email_id='e',chunk_id='e_0',content='evidence',score=1,metadata={})
        with api.sessions.turn('owner-a','stream') as memory:
            memory.add('user','previous')
            memory.add('assistant','old answer')
        with patch.object(coordinator,'classify_intent',return_value=IntentType.RETRIEVE), \
             patch.object(pipeline,'retrieve',return_value=[source]) as retrieve, \
             patch.object(generator,'stream_generate',return_value=iter(['hello',' world'])):
            response=self.client.post('/chat/stream',json={'query':'next','session_id':'stream'},headers=self.headers)
        rows=[json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: {')]
        self.assertEqual(retrieve.call_args.kwargs['history'][0]['content'],'previous')
        self.assertEqual([r for r in rows if 'sources' in r][0]['sources'][0]['email_id'],'e')
        self.assertIn('data: [DONE]',response.text)
        with api.sessions.turn('owner-a','stream') as memory:
            self.assertEqual(memory.to_messages()[-1],{'role':'assistant','content':'hello world'})

    def test_query_uses_unified_pipeline_and_bounds(self):
        with patch('core.pipeline.retrieve',return_value=[]) as retrieve, patch('core.generator.generate_answer',return_value='none'):
            response=self.client.post('/query',json={'query':'find','top_k':9},headers=self.headers)
            self.assertEqual(response.status_code,200)
            retrieve.assert_called_once_with('find',top_n=9)
        self.assertEqual(self.client.post('/query',json={'query':'find','top_k':0},headers=self.headers).status_code,422)
        self.assertEqual(self.client.delete('/chat/history',headers=self.headers).status_code,422)

    def test_stream_does_not_start_generation_after_expired_retrieval(self):
        import time
        finished=threading.Event()
        def slow(*args,**kwargs):
            time.sleep(.08)
            finished.set()
            return []
        with patch.object(cfg,'AGENT_RUN_TIMEOUT',.02), \
             patch('agents.coordinator.classify_intent',return_value=IntentType.RETRIEVE), \
             patch('core.pipeline.retrieve',side_effect=slow), \
             patch('core.generator.stream_generate') as generate:
            response=self.client.post('/chat/stream',json={'query':'next'},headers=self.headers)
            self.assertTrue(finished.wait(1))
            self.assertIn('error',response.text)
            generate.assert_not_called()


class SessionContracts(unittest.TestCase):
    def test_owner_isolation_and_busy_capacity(self):
        store=SessionStore(max_sessions=2)
        with store.turn('a','same') as a:
            a.add('user','private')
            with store.turn('b','same') as b:
                self.assertEqual(b.to_messages(),[])
                with self.assertRaises(SessionBusyError):
                    with store.turn('c','third'): pass
        with store.turn('a','same') as a:
            self.assertEqual(a.to_messages()[0]['content'],'private')

    def test_ttl_expiration_and_idle_eviction(self):
        store=SessionStore(max_sessions=1,ttl_seconds=10)
        with patch('api.sessions.time.monotonic',return_value=100):
            with store.turn('a','one') as memory: memory.add('user','old')
        with patch('api.sessions.time.monotonic',return_value=111):
            with store.turn('a','one') as memory: self.assertEqual(memory.to_messages(),[])
        with store.turn('a','two') as memory: memory.add('user','new')
        with store.turn('a','one') as memory: self.assertEqual(memory.to_messages(),[])


if __name__=='__main__':
    unittest.main()
