"""Synthetic Streamlit tests for conversation-scoped task presentation."""
import sys
from types import ModuleType
from unittest.mock import patch

import pytest

st = pytest.importorskip('streamlit')
AppTest = pytest.importorskip('streamlit.testing.v1').AppTest

SCRIPT = '''
import streamlit as st
from frontend.agent_activity import render_agent_activity
from synthetic_agent_activity_service import get, post
busy = render_agent_activity(st, get, post)
st.session_state['probe_busy'] = busy
st.chat_input('输入', disabled=busy)
'''


@pytest.fixture
def harness(monkeypatch):
    service = ModuleType('synthetic_agent_activity_service')
    service.job = {'id': 'job-one', 'kind': 'agent', 'status': 'queued'}
    service.gets, service.posts = [], []
    def get(path):
        service.gets.append(path)
        return service.job
    def post(path):
        service.posts.append(path)
        return {'cancel_requested': True}
    service.get, service.post = get, post
    monkeypatch.setitem(sys.modules, service.__name__, service)
    app = AppTest.from_string(SCRIPT, default_timeout=10)
    app.session_state['session_id'] = 'session-one'
    app.session_state['messages'] = []
    app.session_state['agent_job_by_session'] = {'session-one': 'job-one'}
    return app, service


def test_no_binding_never_lists_global_jobs(harness):
    app, service = harness
    app.session_state['agent_job_by_session'] = {}
    app.run()
    assert not app.exception and not app.session_state['probe_busy']
    assert service.gets == service.posts == []


@pytest.mark.parametrize('status', ['queued', 'running'])
def test_active_task_only_explicit_cancel_posts(harness, status):
    app, service = harness
    service.job.update(status=status, progress={'private-field': 'not-for-display'})
    with patch('streamlit.fragment', wraps=st.fragment) as fragment:
        app.run()
        assert fragment.call_args.kwargs['run_every'] == 2
    assert not app.exception and app.session_state['probe_busy']
    assert service.gets == ['/jobs/job-one'] and not service.posts
    text = ' '.join(str(x.value) for x in [*app.caption, *app.info, *app.markdown])
    assert 'job-one' not in text and 'private-field' not in text
    app.button[0].click().run()
    assert service.posts == ['/jobs/job-one/cancel']


def test_success_delivered_once_and_terminal_timer_removed(harness):
    app, service = harness
    app.run()
    service.job.update(status='succeeded', result={'answer': 'synthetic answer', 'intent': 'general',
        'sources': [{'email_id': 'synthetic'}], 'metadata': {'session_id': 'session-one'}})
    with patch('streamlit.fragment', wraps=st.fragment) as fragment:
        app.run()
        assert fragment.call_args.kwargs['run_every'] is None
    app.run()
    assert not app.exception and not app.session_state['probe_busy']
    assert app.session_state['messages'] == [{'role': 'assistant', 'content': 'synthetic answer',
        'intent': 'general', 'sources': [{'email_id': 'synthetic'}],
        'extra_metadata': {'session_id': 'session-one'}, 'agent_job_id': 'job-one'}]
    assert not service.posts


@pytest.mark.parametrize('status', ['failed', 'cancelled', 'interrupted', 'incomplete', 'succeeded'])
def test_terminal_without_answer_warns_and_stops_polling(harness, status):
    app, service = harness
    service.job['status'] = status
    with patch('streamlit.fragment', wraps=st.fragment) as fragment:
        app.run()
        assert fragment.call_args.kwargs['run_every'] is None
    assert not app.exception and not app.session_state['probe_busy']
    assert any('高级工具' in item.value for item in app.warning)
    assert app.session_state['messages'] == [] and not service.posts


def test_unavailable_preserves_binding_and_busy(harness):
    app, service = harness
    app.run()
    service.job = None
    app.run()
    assert not app.exception and app.session_state['probe_busy']
    assert app.session_state['agent_job_by_session'] == {'session-one': 'job-one'}
    assert app.warning and not service.posts


@pytest.mark.parametrize('wrong', [{'kind': 'imap_sync'}, {'id': 'other'},
    {'result': {'answer': 'must not show', 'metadata': {'session_id': 'other'}}}])
def test_mismatched_job_never_delivers(harness, wrong):
    app, service = harness
    service.job.update(status='succeeded', result={'answer': 'must not show'})
    service.job.update(wrong)
    app.run()
    assert not app.exception and not app.session_state['probe_busy']
    assert app.session_state['messages'] == [] and app.warning


def test_session_switch_does_not_fetch_or_deliver_old_job(harness):
    app, service = harness
    app.run()
    service.gets.clear()
    service.job.update(status='succeeded', result={'answer': 'old session answer'})
    app.session_state['session_id'] = 'session-two'
    app.session_state['messages'] = []
    app.run()
    assert not app.exception and not service.gets
    assert app.session_state['messages'] == [] and not app.session_state['probe_busy']


def test_explicitly_resumed_job_restarts_polling(harness):
    app, service = harness
    service.job['status'] = 'interrupted'
    app.run()
    service.job['status'] = 'running'
    with patch('streamlit.fragment', wraps=st.fragment) as fragment:
        app.run()
        assert fragment.call_args.kwargs['run_every'] == 2
    assert not app.exception and app.session_state['probe_busy']
    assert not service.posts


def test_session_changes_during_fetch_cannot_receive_old_answer(harness):
    app, service = harness
    original = service.get
    def switched_get(path):
        result = original(path)
        st.session_state['session_id'] = 'session-two'
        st.session_state['messages'] = []
        return result
    service.job.update(status='succeeded', result={'answer': 'old answer'})
    service.get = switched_get
    app.run()
    assert not app.exception and app.session_state['messages'] == []
    assert 'agent_activity_delivered' not in app.session_state
    assert not app.session_state['probe_busy']


def test_cancel_requested_has_no_second_cancel_button(harness):
    app, service = harness
    service.job.update(status='running', cancel_requested=True)
    app.run()
    assert not app.exception and not app.button and app.session_state['probe_busy']
    assert app.info and not service.posts
