from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace as NS
from unittest.mock import patch
from unittest.mock import Mock
from contextlib import nullcontext
import json
import sys

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


def test_complete_reference_instant_reaches_planner_task_and_judge(monkeypatch,tmp_path):
    monkeypatch.setattr(cfg,'RETRIEVAL_TIMEZONE','')
    monkeypatch.setattr(cfg,'RETRIEVAL_TIMEZONE_OFFSET_HOURS',8)
    instant=datetime.fromisoformat('2026-04-28T12:30:00+08:00')
    captured=[]
    def loop(request,**kwargs):
        captured.append(current_execution_scope().evaluation_reference_time)
        assert instant.isoformat() in _system_prompt()
        return NS(answer='fixture',sources=[],metadata={'steps':[],'status':'success','trace_id':'fixture'})
    create=Mock(return_value=NS(choices=[NS(finish_reason='stop',message=NS(content='{"success":1}'))]))
    client=NS(chat=NS(completions=NS(create=create)))
    with evaluation.evaluation_scope(tmp_path/'eval',reference_time=instant), \
            patch('agents.agent_loop.run_agent_loop',side_effect=loop):
        result=evaluation.evaluate_task({'id':'fixture','task':'recent','expected_tools':[]},client)
        assert _now() == instant
    assert result['success'] == 1 and captured == [instant]
    assert instant.isoformat() in create.call_args.kwargs['messages'][0]['content']
    assert current_execution_scope() is None


def test_nested_evaluation_compares_instants_not_dates(monkeypatch,tmp_path):
    monkeypatch.setattr(cfg,'RETRIEVAL_TIMEZONE','')
    monkeypatch.setattr(cfg,'RETRIEVAL_TIMEZONE_OFFSET_HOURS',8)
    instant=datetime.fromisoformat('2026-04-28T12:00:00+08:00')
    with evaluation.evaluation_scope(tmp_path/'eval',reference_time=instant) as original:
        with evaluation.evaluation_scope(reference_time=instant.astimezone(timezone.utc)) as inherited:
            assert inherited is original
        with pytest.raises(ValueError,match='same reference'):
            with evaluation.evaluation_scope(reference_time=instant.replace(hour=0)):
                pytest.fail('same date with a different time must not be accepted')
        with pytest.raises(ValueError,match='same reference'):
            with evaluation.evaluation_scope(reference_date=instant.date()):
                pytest.fail('date mode means midnight, not any instant on that day')
    with evaluation.evaluation_scope(tmp_path/'live'):
        with pytest.raises(ValueError,match='same reference'):
            with evaluation.evaluation_scope(reference_time=instant):
                pytest.fail('nested run cannot add a clock absent from its parent')


@pytest.mark.parametrize('argv',[
    ['--reference-time','2026-04-28T12:00:00'],
    ['--reference-date','2026-04-28','--reference-time','2026-04-28T00:00:00+08:00'],
])
def test_cli_rejects_ambiguous_or_conflicting_reference_before_creating_state(monkeypatch,tmp_path,argv):
    run=tmp_path/'not-created'
    monkeypatch.setattr(sys,'argv',['eval','--run-dir',str(run),*argv])
    with pytest.raises(SystemExit) as error:
        evaluation.main()
    assert error.value.code == 2 and not run.exists()


def test_reference_time_provenance_records_effective_timezone_and_mode(monkeypatch,tmp_path):
    monkeypatch.setattr(cfg,'RETRIEVAL_TIMEZONE','')
    monkeypatch.setattr(cfg,'RETRIEVAL_TIMEZONE_OFFSET_HOURS',8)
    tasks=tmp_path/'tasks.json'
    tasks.write_text(json.dumps([{'id':'fixture','task':'fixture','expected_tools':[]}]),encoding='utf-8')
    output=tmp_path/'report.json'
    monkeypatch.setattr(evaluation,'TESTSET_PATH',tasks)
    monkeypatch.setattr(evaluation,'build_provenance',lambda **kwargs:{'artifacts':{}})
    monkeypatch.setattr(evaluation,'OpenAI',lambda **kwargs:object())
    monkeypatch.setattr(evaluation,'evaluate_task',lambda item,client:evaluation.failure_record(item,ValueError()))
    monkeypatch.setattr(evaluation.time,'sleep',lambda *a:None)
    monkeypatch.setattr('core.embedder.index_snapshot',lambda:nullcontext())
    monkeypatch.setattr('core.embedder.current_index_manifest',lambda:None)
    monkeypatch.setattr(sys,'argv',['eval','--run-dir',str(tmp_path/'run'),'--output',str(output),
                                  '--reference-time','2026-04-28T04:00:00+00:00'])
    assert evaluation.main() == 0
    provenance=json.loads(output.read_text(encoding='utf-8'))['provenance']
    assert provenance['evaluation_reference_date'] is None
    assert provenance['evaluation_time_mode'] == 'time'
    assert provenance['evaluation_reference_time'] == '2026-04-28T12:00:00+08:00'
