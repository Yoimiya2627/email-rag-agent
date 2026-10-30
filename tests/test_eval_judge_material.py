"""Evaluation uses complete task evidence and reports grading failures honestly."""
from types import SimpleNamespace as NS
from unittest.mock import Mock

import config.settings as cfg
from models.schemas import AgentResponse, SearchResult
from scripts import run_agent_eval as evaluation


def test_judge_projection_preserves_content_headers_and_uncertainty():
    metadata = {'subject':'fixture','sender':'sender@example.invalid','date':'2026-01-01',
                'coverage':{'body':'partial','attachments':'not_read'},
                'custom_uncertainty':'must remain visible','source_start':10,'source_end':28,
                'source_length':400,'source_sha256':'a'*64,'chunk_sha256':'b'*64}
    source = SearchResult(email_id='e',chunk_id='c',content='Complete evidence.',score=1,metadata=metadata)
    steps = [{'tool':'get_email','status':'success','output_truncated':True,
              'evidence_refs':[{'source_sha256':'a'*64}], 'tool_call_id':'fixture-call'}]
    execution, sources = evaluation.judge_material(AgentResponse(answer='fixture',sources=[source]),steps)
    assert sources[0]['content'] == source.content
    assert sources[0]['metadata'] == {key:value for key,value in metadata.items()
                                     if key not in {'source_sha256','chunk_sha256'}}
    assert execution == [{'tool':'get_email','status':'success','output_truncated':True}]
    assert steps[0]['evidence_refs'] == [{'source_sha256':'a'*64}]
    assert source.metadata == metadata


def test_judge_provider_mode_matches_application_and_reports_input_size(monkeypatch):
    monkeypatch.setattr(cfg,'AGENT_PLANNER_MODEL','deepseek-flash')
    monkeypatch.setattr(cfg,'DEEPSEEK_THINKING_MODE','disabled')
    create = Mock(return_value=NS(choices=[NS(finish_reason='stop',message=NS(content='{"success":1,"reason":"ok"}'))]))
    client = NS(base_url='https://api.deepseek.com/v1/',chat=NS(completions=NS(create=create)))
    client.with_options = Mock(return_value=client)
    result = evaluation.judge_success(client,'task','answer',sources=[{'content':'evidence'}])
    assert result['success'] == 1 and result['scoring_status'] == 'ok'
    assert result['judge_input_chars'] == len(create.call_args.kwargs['messages'][0]['content'])
    assert create.call_args.kwargs['extra_body'] == {'thinking':{'type':'disabled'}}
    assert client.with_options.call_args.kwargs['max_retries'] == 0


def test_judge_excludes_unseen_candidates_but_keeps_every_visible_source():
    sources = [SearchResult(email_id='e',chunk_id=str(i),content='full evidence '+str(i),score=1,metadata={})
               for i in range(3)]
    refs = [{'email_id':'e','chunk_id':str(i),'visible_start':0,'visible_end':4,'offset_basis':'chunk_text'}
            for i in (0,1)]
    response = AgentResponse(answer='fixture',sources=sources,metadata={'model_visible_evidence':refs})
    _, projected = evaluation.judge_material(response, [])
    assert [row['chunk_id'] for row in projected] == ['0','1']
    assert [row['content'] for row in projected] == [row.content for row in sources[:2]]
    assert all(row['model_visible_ranges'] == [{'visible_start':0,'visible_end':4,'offset_basis':'chunk_text'}]
               for row in projected)
    response.metadata['model_visible_evidence'] = []
    assert evaluation.judge_material(response, [])[1] == []


def test_oversized_evidence_still_fails_without_clipping_or_provider_call(monkeypatch):
    monkeypatch.setattr(cfg,'EVAL_JUDGE_CONTEXT_CHAR_LIMIT',1000)
    create = Mock()
    client = NS(chat=NS(completions=NS(create=create)))
    result = evaluation.judge_success(client,'task','answer',sources=[{'content':'原始证据'*1000}])
    assert result['scoring_status'] == 'error' and result['success'] == 0
    assert result['scoring_error_code'] == 'judge_input_budget_exceeded'
    assert result['judge_input_chars'] > 1000
    create.assert_not_called()


def test_incomplete_judge_response_remains_a_failed_score():
    create = Mock(return_value=NS(choices=[NS(finish_reason='length',message=NS(content='{"success":1}'))]))
    client = NS(chat=NS(completions=NS(create=create)))
    result = evaluation.judge_success(client,'task','answer')
    assert result['scoring_status'] == 'error' and result['success'] == 0
    assert result['scoring_error_code'] == 'judge_incomplete_response'
