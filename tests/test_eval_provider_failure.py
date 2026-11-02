from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest.mock import Mock
import json
import sys

import pytest
import httpx
from openai import APIStatusError

from scripts import run_agent_eval as evaluation
from core.model_errors import PermanentModelProviderError


def sdk_failure(status=402):
    return APIStatusError('private provider response', response=httpx.Response(status,
        request=httpx.Request('POST', 'https://model.example.invalid/chat/completions')),
        body={'private':'private provider response'})


class ProviderFailure(Exception):
    status_code = 402


def test_judge_records_only_safe_provider_failure_metadata():
    create=Mock(side_effect=sdk_failure())
    result=evaluation.judge_success(NS(chat=NS(completions=NS(create=create))),'task','answer')
    assert result['provider_http_status'] == 402
    assert result['success'] == 0 and result['scoring_status'] == 'error'
    assert 'private provider response' not in json.dumps(result)


@pytest.mark.parametrize('failure_stage',['agent','judge'])
def test_batch_stops_on_permanent_provider_error_and_preserves_attempted_case(monkeypatch,tmp_path,failure_stage):
    tasks=tmp_path/'tasks.json'
    tasks.write_text(json.dumps([{'id':str(i),'task':'fixture','expected_tools':[]} for i in range(3)]),encoding='utf-8')
    output=tmp_path/'report.json'
    monkeypatch.setattr(evaluation,'TESTSET_PATH',tasks)
    monkeypatch.setattr(evaluation,'build_provenance',lambda **kwargs:{'artifacts':{}})
    monkeypatch.setattr(evaluation,'OpenAI',lambda **kwargs:object())
    monkeypatch.setattr('core.embedder.index_snapshot',lambda:nullcontext())
    monkeypatch.setattr('core.embedder.current_index_manifest',lambda:None)
    attempts=[]
    def evaluate(item,client):
        attempts.append(item['id'])
        if failure_stage == 'agent':
            raise PermanentModelProviderError(402, 'agent_plan')
        row=evaluation.failure_record(item,PermanentModelProviderError(402, 'judge'))
        row.update(scoring_status='error',scoring_method='llm_binary_v2',judge_success=0)
        return row
    monkeypatch.setattr(evaluation,'evaluate_task',evaluate)
    monkeypatch.setattr(sys,'argv',['eval','--run-dir',str(tmp_path/'run'),'--output',str(output),
                                    '--reference-date','2026-04-28'])
    assert evaluation.main() == 2
    payload=json.loads(output.read_text(encoding='utf-8'))
    assert attempts == ['0']
    assert len(payload['records']) == 1 and payload['summary']['n_tasks'] == 1
    assert payload['provenance']['batch'] == {'status':'stopped_provider_error','requested_tasks':3,
                                            'attempted_tasks':1,'provider_http_status':402}
    assert payload['provenance']['evaluation_reference_date'] == '2026-04-28'
    assert len((tmp_path/'run/records.jsonl').read_text(encoding='utf-8').splitlines()) == 1
    assert 'private provider response' not in json.dumps(payload)


def test_arbitrary_http_attribute_is_not_a_model_provider_failure():
    item={'id':'business','task':'fixture','expected_tools':[]}
    result=evaluation.failure_record(item,ProviderFailure('private business response'))
    assert result['provider_http_status'] is None
    assert result['provider_failure'] is None
    create=Mock(side_effect=ProviderFailure('private business response'))
    verdict=evaluation.judge_success(NS(chat=NS(completions=NS(create=create))),'task','answer')
    assert verdict['provider_http_status'] is None
    assert verdict['scoring_status'] == 'error'


@pytest.mark.parametrize('status',[401,402,403,404])
def test_real_tool_model_failure_stops_before_sibling_planner_judge_and_next_case(monkeypatch,tmp_path,status,caplog):
    """Exercise SDK boundary, tool normalization, Agent response and evaluator together."""
    import agents.agent_loop as loop
    import agents.tools as domain
    from core.model_clients import create_completion
    from scripts.check_agent_eval_gate import GateThresholds,evaluate_gate
    tasks=tmp_path/'tasks.json'
    tasks.write_text(json.dumps([{'id':str(i),'task':'Summarize fixture mail',
                                 'expected_tools':['summarize_emails']} for i in range(3)]),encoding='utf-8')
    output=tmp_path/'report.json'
    calls=[NS(id='summary',type='function',function=NS(name='summarize_emails',arguments='{"query":"fixture"}')),
           NS(id='later',type='function',function=NS(name='email_stats',arguments='{}'))]
    planner_create=Mock(return_value=NS(choices=[NS(finish_reason='tool_calls',
                                                   message=NS(content=None,tool_calls=calls))]))
    tool_create=Mock(side_effect=sdk_failure(status))
    judge_create=Mock(side_effect=AssertionError('judge must not run after permanent model failure'))
    planner=NS(chat=NS(completions=NS(create=planner_create)))
    tool_client=NS(chat=NS(completions=NS(create=tool_create)))
    judge=NS(chat=NS(completions=NS(create=judge_create)))
    sibling=Mock(side_effect=AssertionError('later tool must not run'))
    def summarize(**kwargs):
        return create_completion(tool_client,stage='summarize',model='fixture',
            messages=[{'role':'user','content':'fixture'}],max_tokens=100)
    monkeypatch.setitem(domain.TOOL_DISPATCH,'summarize_emails',summarize)
    monkeypatch.setitem(domain.TOOL_DISPATCH,'email_stats',sibling)
    monkeypatch.setattr(loop,'_get_client',lambda:planner)
    monkeypatch.setattr(evaluation,'TESTSET_PATH',tasks)
    monkeypatch.setattr(evaluation,'build_provenance',lambda **kwargs:{'artifacts':{}})
    monkeypatch.setattr(evaluation,'OpenAI',lambda **kwargs:judge)
    monkeypatch.setattr('core.embedder.index_snapshot',lambda:nullcontext())
    monkeypatch.setattr('core.embedder.current_index_manifest',lambda:None)
    monkeypatch.setattr(sys,'argv',['eval','--run-dir',str(tmp_path/'run'),'--output',str(output),
                                  '--reference-time','2026-04-28T12:00:00+08:00'])
    assert evaluation.main() == 2
    payload=json.loads(output.read_text(encoding='utf-8'))
    assert payload['provenance']['batch']['status'] == 'stopped_provider_error'
    assert payload['provenance']['batch']['attempted_tasks'] == 1
    row=payload['records'][0]
    assert row['actual_tools'] == ['summarize_emails'] and row['n_steps'] == 1
    assert row['steps'][0]['status'] == 'error'
    assert row['run_status'] == 'error' and row['execution_status'] == 'error'
    assert row['scoring_status'] == 'not_run' and row['scoring_method'] == 'unavailable'
    assert row['provider_http_status'] == status
    assert row['provider_failure']['stage'] == 'summarize'
    assert row['success'] == 0 and row['judge_success'] == 0
    assert planner_create.call_count == 1 and tool_create.call_count == 1
    judge_create.assert_not_called()
    sibling.assert_not_called()
    journal=(tmp_path/'run/records.jsonl').read_text(encoding='utf-8')
    assert len(journal.splitlines()) == 1
    assert 'private provider response' not in json.dumps(payload)+journal+caplog.text
    gate=evaluate_gate(output,GateThresholds(min_tasks=1,min_task_success_rate=0,
                                            min_tool_accuracy=0,require_current_revision=False))
    assert not gate.passed
    assert 'evaluation batch did not complete all requested tasks' in gate.failures


def test_permanent_failure_on_last_task_still_marks_batch_stopped(monkeypatch,tmp_path):
    from scripts.check_agent_eval_gate import GateThresholds,evaluate_gate
    tasks=tmp_path/'tasks.json'
    items=[{'id':str(i),'task':'fixture','expected_tools':[]} for i in range(2)]
    tasks.write_text(json.dumps(items),encoding='utf-8')
    output=tmp_path/'report.json'
    monkeypatch.setattr(evaluation,'TESTSET_PATH',tasks)
    monkeypatch.setattr(evaluation,'build_provenance',lambda **kwargs:{'artifacts':{}})
    monkeypatch.setattr(evaluation,'OpenAI',lambda **kwargs:object())
    monkeypatch.setattr('core.embedder.index_snapshot',lambda:nullcontext())
    monkeypatch.setattr('core.embedder.current_index_manifest',lambda:None)
    monkeypatch.setattr(evaluation.time,'sleep',lambda *a:None)
    def evaluate(item,client):
        failure=(ValueError('ordinary case error') if item['id']=='0'
                 else PermanentModelProviderError(404,'agent_plan'))
        return evaluation.failure_record(item,failure)
    monkeypatch.setattr(evaluation,'evaluate_task',evaluate)
    monkeypatch.setattr(sys,'argv',['eval','--run-dir',str(tmp_path/'run'),'--output',str(output)])
    assert evaluation.main() == 2
    payload=json.loads(output.read_text(encoding='utf-8'))
    assert payload['provenance']['batch'] == {'status':'stopped_provider_error','requested_tasks':2,
                                            'attempted_tasks':2,'provider_http_status':404}
    gate=evaluate_gate(output,GateThresholds(min_tasks=1,min_task_success_rate=0,
                                            min_tool_accuracy=0,require_current_revision=False))
    assert not gate.passed and 'evaluation batch did not complete all requested tasks' in gate.failures
