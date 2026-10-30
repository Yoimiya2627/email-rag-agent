from types import SimpleNamespace
from unittest.mock import patch
import copy

from agents.eval_contract import make_tool_observer, check_tool_assertions, effective_config, artifact_fingerprint
from scripts import run_agent_eval as evaluation


def case():
    return {'tool_assertions':[
        {'tool':'search_emails','occurrence':1,'source':'arguments','path':'sender','equals':'alice@example.com'},
        {'tool':'search_emails','occurrence':1,'source':'result','path':'data.0.email_id','equals':'expected-email'},
    ]}


def test_tool_name_alone_cannot_pass_wrong_filter_or_result():
    observer=make_tool_observer(case())
    checks=observer('search_emails',{'sender':'wrong@example.com'},
                    {'data':[{'email_id':'expected-email','body':'PRIVATE BODY'}]},'call-1')
    result=check_tool_assertions(case(),[{'tool':'search_emails','evaluation_checks':checks}])
    assert result=={'passed':False,'required':2,'matched':1,'failed_indices':[0]}
    assert 'PRIVATE' not in str(checks) and 'example.com' not in str(checks)


def test_expected_parameters_and_results_pass_and_missing_steps_fail():
    observer=make_tool_observer(case())
    checks=observer('search_emails',{'sender':'alice@example.com'}, {'data':[{'email_id':'expected-email'}]},'call')
    assert check_tool_assertions(case(),[{'tool':'search_emails','evaluation_checks':checks}])['passed']
    assert not check_tool_assertions(case(),[])['passed']
    assert not check_tool_assertions(case(),[{'tool':'email_stats','evaluation_checks':checks}])['passed']


def test_provenance_configuration_never_contains_secrets_and_artifact_changes_are_bound(tmp_path):
    settings=SimpleNamespace(DEEPSEEK_API_KEY='PRIVATE',API_AUTH_TOKEN='PRIVATE',
                             GMAIL_TOKEN_PATH='PRIVATE',AGENT_MAX_TOKENS=4000,
                             DEEPSEEK_BASE_URL='https://example.com/?secret=PRIVATE')
    data=effective_config(settings)
    assert data['AGENT_MAX_TOKENS']==4000 and 'PRIVATE' not in str(data)
    path=tmp_path/'synthetic.json'
    path.write_text('one',encoding='utf-8')
    first=artifact_fingerprint(path)
    path.write_text('two',encoding='utf-8')
    assert first['sha256']!=artifact_fingerprint(path)['sha256']
    assert artifact_fingerprint(None)['status']=='unavailable'


def test_judge_rejects_bool_score_and_passes_budget_without_retries(monkeypatch):
    calls=[]
    class Client:
        def with_options(self,**kwargs):
            calls.append(kwargs)
            return self
        @property
        def chat(self): return SimpleNamespace(completions=SimpleNamespace(create=self.create))
        def create(self,**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content='{"success":true}'))])
    result=evaluation.judge_success(Client(),'task','answer')
    assert result['success']==0 and result['scoring_status']=='error'
    assert calls[0]['max_retries']==0 and calls[1]['timeout']<=30
    assert result['scoring_method']=='llm_binary_v2'


def test_judge_context_overflow_never_calls_provider(monkeypatch):
    monkeypatch.setattr(evaluation.cfg,'EVAL_JUDGE_CONTEXT_CHAR_LIMIT',1000,raising=False)
    result=evaluation.judge_success(object(),'t','a'*2000)
    assert result['success']==0 and result['scoring_status']=='error'
