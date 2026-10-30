from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

import config.settings as cfg
from agents.execution_scope import ExecutionScope, current_execution_scope
from agents.agent_loop import _system_prompt
from core.pipeline import _now, _date_window
from core.evidence_pages import compute_scoped_stats
from scripts import run_agent_eval as evaluation


def test_reference_date_controls_prompt_filters_and_statistics_without_changing_wall_clock(monkeypatch,tmp_path):
    monkeypatch.setattr(cfg,'RETRIEVAL_TIMEZONE','')
    monkeypatch.setattr(cfg,'RETRIEVAL_TIMEZONE_OFFSET_HOURS',8)
    wall_before = datetime.now(timezone.utc)
    rows = [{'metadata':{'email_id':'spring','date':'2026-04-28'}},
            {'metadata':{'email_id':'autumn','date':'2026-10-28'}}]
    with evaluation.evaluation_scope(tmp_path/'eval',reference_date=date(2026,4,28)):
        assert _now() == datetime(2026,4,28,tzinfo=timezone(timedelta(hours=8)))
        assert [x.date() for x in _date_window('本周')] == [date(2026,4,27),date(2026,5,4)]
        assert '2026-04-28' in _system_prompt()
        stats=compute_scoped_stats(rows,filters={'date_hint':'本周'})
        assert stats['total_emails'] == 1
        assert stats['daily_counts'] == [{'date':'2026-04-28','count':1}]
        with pytest.raises(ValueError,match='same reference date'):
            with evaluation.evaluation_scope(reference_date=date(2026,4,29)):
                pytest.fail('nested evaluation must not change the clock')
    assert wall_before <= _now().astimezone(timezone.utc) <= datetime.now(timezone.utc)
    assert current_execution_scope() is None


def test_reference_date_is_inherited_by_each_task_and_its_judge(monkeypatch,tmp_path):
    captured=[]
    def loop(request,**kwargs):
        captured.append(current_execution_scope().evaluation_reference_date)
        return NS(answer='fixture',sources=[],metadata={'steps':[],'status':'success','trace_id':'fixture'})
    def judge(*args,**kwargs):
        captured.append(current_execution_scope().evaluation_reference_date)
        return {'success':1,'reason':'fixture'}
    monkeypatch.setattr(evaluation,'judge_success',judge)
    with evaluation.evaluation_scope(tmp_path/'eval',reference_date=date(2026,4,28)), \
            patch('agents.agent_loop.run_agent_loop',side_effect=loop):
        evaluation.evaluate_task({'id':'fixture','task':'today','expected_tools':[]},object())
    assert captured == [date(2026,4,28),date(2026,4,28)]


def test_regular_requests_cannot_override_reference_date():
    with pytest.raises(ValueError,match='trusted evaluation'):
        ExecutionScope(owner_id='ordinary',evaluation_reference_date=date(2026,4,28))
