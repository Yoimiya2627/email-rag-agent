"""Deterministic SSE terminal-event races with real session persistence."""
import json
import queue
import threading
from types import SimpleNamespace

from fastapi.testclient import TestClient

import api.main as api
import config.settings as cfg
from api.sessions import SessionStore
from models.schemas import IntentType, SearchResult


def test_stream_drains_completion_queued_after_empty_before_worker_exit(tmp_path, monkeypatch):
    store = SessionStore(path=tmp_path / 'sessions.sqlite3')
    monkeypatch.setattr(api, 'sessions', store)
    monkeypatch.setattr(cfg, 'ENABLE_CONTEXT_SUMMARY', False)
    monkeypatch.setattr(cfg, 'ENABLE_CONTEXT_CANDIDATES', False)
    monkeypatch.setattr(cfg, 'AGENT_RUN_TIMEOUT', 120)
    with store.turn('local', 'race') as memory:
        memory.append_turn('Remember the budget.', 'Recorded.')

    classifier_waiting = threading.Event()
    allow_prefix = threading.Event()
    producer_waiting = threading.Event()
    allow_completion = threading.Event()
    workers = []
    observed = {}

    def worker_factory(**kwargs):
        worker = threading.Thread(**kwargs)
        workers.append(worker)
        return worker

    class DelayedEmptyQueue(queue.Queue):
        prefix_injected = False
        armed = False
        injected = False

        def get(self, block=True, timeout=None):
            if block and not self.prefix_injected:
                self.prefix_injected = True
                assert classifier_waiting.wait(10), 'producer did not reach startup barrier'
                assert self.empty(), 'prefix must not exist before startup is released'
                observed['empty_before_prefix'] = True
                allow_prefix.set()
                raise queue.Empty
            if block and self.armed and not self.injected:
                self.injected = True
                # Intent and token have been consumed. The queue read expires
                # while generation is pending, but the event loop observes its
                # Empty only after the producer commits and exits.
                assert producer_waiting.wait(10), 'producer did not reach completion barrier'
                assert self.empty(), 'queue must be empty at the simulated timeout'
                allow_completion.set()
                workers[0].join(10)
                assert not workers[0].is_alive(), 'producer did not finish'
                observed['pending_terminal_events'] = self.qsize()
                observed['committed_turns_before_empty'] = len(store.history('local', 'race')['turns'])
                raise queue.Empty
            item = super().get(block, timeout)
            if isinstance(item, dict) and 'token' in item:
                self.armed = True
            return item

    events = DelayedEmptyQueue(maxsize=128)
    # Replace only api.main's bindings, never the shared threading/queue modules.
    monkeypatch.setattr(api, 'threading', SimpleNamespace(Event=threading.Event, Thread=worker_factory))
    monkeypatch.setattr(api, 'queue', SimpleNamespace(Queue=lambda **kwargs: events, Empty=queue.Empty, Full=queue.Full))
    def classify(*args, **kwargs):
        classifier_waiting.set()
        assert allow_prefix.wait(10), 'consumer did not release startup barrier'
        return IntentType.RETRIEVE

    monkeypatch.setattr('agents.coordinator.classify_intent', classify)
    hit = SearchResult(email_id='e', chunk_id='c', content='source', score=1, metadata={})
    monkeypatch.setattr('core.pipeline.retrieve', lambda *a, **kw: [hit])

    def stream(*args, **kwargs):
        yield 'complete answer'
        producer_waiting.set()
        assert allow_completion.wait(10), 'consumer did not release completion barrier'

    monkeypatch.setattr('core.generator.stream_generate', stream)
    try:
        response = TestClient(api.app).post('/chat/stream', json={'session_id': 'race', 'query': 'Continue.'})
    finally:
        allow_prefix.set()
        allow_completion.set()
        for worker in workers:
            worker.join(10)
            assert not worker.is_alive(), 'test leaked its producer'

    assert observed == {'empty_before_prefix': True, 'pending_terminal_events': 2,
                        'committed_turns_before_empty': 2}
    assert response.status_code == 200
    assert 'data: [DONE]' in response.text, response.text
    rows = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: {')]
    assert [row['intent'] for row in rows if 'intent' in row] == ['retrieve']
    assert [row['token'] for row in rows if 'token' in row] == ['complete answer']
    terminal = next(row for row in rows if 'sources' in row)
    assert terminal['sources'][0]['chunk_id'] == 'c'
    assert terminal['metadata']['completion_status'] == 'complete'
    assert not any('error' in row for row in rows)
    turns = store.history('local', 'race')['turns']
    assert len(turns) == 2 and turns[-1]['answer'] == 'complete answer'
    assert terminal['metadata']['turn_id'] == turns[-1]['turn_id']


def test_stream_reports_error_when_worker_exits_without_terminal_events(monkeypatch):
    class FinishedWorker(threading.Thread):
        def __init__(self, **kwargs):
            # Model an unexpectedly terminated worker without producing events.
            super().__init__(target=lambda: None, daemon=True)

        def start(self):
            super().start()
            self.join(10)
            assert not self.is_alive()

    class EmptyQueue(queue.Queue):
        def get(self, block=True, timeout=None):
            raise queue.Empty

    monkeypatch.setattr(api, 'threading', SimpleNamespace(Event=threading.Event, Thread=FinishedWorker))
    monkeypatch.setattr(api, 'queue', SimpleNamespace(Queue=EmptyQueue, Empty=queue.Empty, Full=queue.Full))
    monkeypatch.setattr(cfg, 'AGENT_RUN_TIMEOUT', 120)
    response = TestClient(api.app).post('/chat/stream', json={'session_id': 'empty', 'query': 'Continue.'})
    assert response.status_code == 200
    assert 'Stream stopped or timed out' in response.text
    assert 'data: [DONE]' not in response.text
