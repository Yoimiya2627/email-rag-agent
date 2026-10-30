from contextlib import nullcontext
from types import SimpleNamespace as NS
from unittest.mock import Mock
import json
import sys

import pytest

from scripts import run_agent_eval as evaluation


class ProviderFailure(Exception):
    status_code = 402


def test_judge_records_only_safe_provider_failure_metadata():
    create=Mock(side_effect=ProviderFailure('private provider response'))
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
            raise ProviderFailure('private provider response')
        row=evaluation.failure_record(item,ProviderFailure('private provider response'))
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
