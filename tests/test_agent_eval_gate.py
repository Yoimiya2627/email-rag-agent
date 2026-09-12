"""Offline gates reject stale, inconsistent, empty and deceptively successful reports."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agents.eval_contract import aggregate, build_provenance, validate_payload
from scripts.check_agent_eval_gate import GateThresholds, evaluate_gate
import scripts.run_agent_eval as evaluation
from models.schemas import AgentResponse, SearchResult


def record(identity='test-1'):
    return {'id':identity,'trace_id':'trace-test','success':1,'judge_success':1,
            'scoring_method':'llm_binary_v2','scoring_status':'ok','tool_accuracy':True,
            'max_steps_reached':False,'forbidden_tool_violation':False,'n_steps':1,
            'actual_tools':['search_emails'],'expected_tools':['search_emails'],'forbidden_tools':['send_email'],
            'steps':[{'tool':'search_emails','status':'success'}],
            'run_status':'success','execution_contract_passed':True}


def report(records=None):
    records=[record()] if records is None else records
    return {'schema_version':2,'records':records,'summary':aggregate(records),
            'provenance':build_provenance(model='offline-fixture')}


class EvalGateContracts(unittest.TestCase):
    def test_judge_failure_missing_fields_or_wrong_method_cannot_claim_success(self):
        for update in ({'judge_success':0},{'judge_success':True},{'scoring_status':'error'},
                       {'scoring_method':'embedding_proxy'},{'scoring_method':None},{'judge_success':None}):
            payload=report()
            payload['records'][0].update(update)
            self.assertFalse(self.gate(payload).passed,update)

    def gate(self,payload,**kwargs):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'report.json'
            path.write_text(json.dumps(payload),encoding='utf-8')
            return evaluate_gate(path,GateThresholds(min_tasks=1,require_current_revision=False,**kwargs))

    def test_valid_records_pass_and_summary_is_recomputed(self):
        result=self.gate(report())
        self.assertTrue(result.passed,result.failures)
        payload=report()
        payload['summary']['n_tasks']=999
        self.assertFalse(self.gate(payload).passed)

    def test_empty_summary_only_and_duplicate_ids_fail(self):
        for payload in ({'summary':{'n_tasks':100,'task_success_rate':1}}, report([]), report([record(),record()])):
            self.assertFalse(self.gate(payload).passed)

    def test_missing_trace_and_bad_numeric_types_fail(self):
        for key,value in [('trace_id',''),('success',True),('n_steps',-1),('tool_accuracy',1)]:
            payload=report()
            payload['records'][0][key]=value
            self.assertFalse(self.gate(payload).passed,key)

    def test_error_tool_cannot_claim_success_even_with_recomputed_summary(self):
        payload=report()
        payload['records'][0]['steps'][0]['status']='error'
        payload['records'][0]['execution_contract_passed']=False
        self.assertFalse(self.gate(payload).passed)

    def test_forged_coverage_cannot_pass(self):
        payload=report()
        payload['records'][0]['actual_tools']=[]
        payload['records'][0]['steps']=[]
        payload['records'][0]['n_steps']=0
        payload['summary']=aggregate(payload['records'])
        self.assertFalse(self.gate(payload).passed)

    def test_stale_source_and_dataset_are_rejected(self):
        from agents.eval_contract import ROOT
        case=json.loads((ROOT/'data/agent_testset.json').read_text(encoding='utf-8'))[0]
        canonical=record(case['id'])
        canonical.update({key:case.get(key, default) for key, default in
                          [('task',''),('success_criteria',''),('task_type','general'),('risk_level','low'),
                           ('expected_tools',[]),('forbidden_tools',[])]})
        canonical['actual_tools']=list(canonical['expected_tools'])
        canonical['steps']=[{'tool':tool,'status':'success'} for tool in canonical['actual_tools']]
        canonical['n_steps']=len(canonical['steps'])
        payload=report([canonical])
        self.assertEqual(validate_payload(payload)[1],[])
        for key in ('source_sha256','dataset_sha256'):
            bad=copy.deepcopy(payload)
            bad['provenance'][key]='outdated'
            self.assertTrue(any(key in item for item in validate_payload(bad)[1]))

    def test_report_cannot_erase_canonical_task_constraints(self):
        from agents.eval_contract import ROOT
        cases=json.loads((ROOT/'data/agent_testset.json').read_text(encoding='utf-8'))
        records=[]
        for case in cases:
            item=record(case['id'])
            item.update({'actual_tools':[],'expected_tools':[],'forbidden_tools':[],
                         'steps':[],'n_steps':0,'task':case['task'],
                         'success_criteria':case.get('success_criteria',''),
                         'task_type':case.get('task_type','general'),'risk_level':case.get('risk_level','low')})
            records.append(item)
        failures=validate_payload(report(records))[1]
        self.assertTrue(any('expected_tools does not match dataset' in failure for failure in failures))

    def test_canonical_task_and_success_criteria_are_bound(self):
        from agents.eval_contract import ROOT
        case=json.loads((ROOT/'data/agent_testset.json').read_text(encoding='utf-8'))[0]
        item=record(case['id'])
        item.update({'task':'changed task', 'success_criteria':'easy replacement'})
        failures=validate_payload(report([item]))[1]
        self.assertTrue(any('task does not match dataset' in failure for failure in failures))
        self.assertTrue(any('success_criteria does not match dataset' in failure for failure in failures))

    def test_failed_run_is_valid_report_but_fails_quality_threshold(self):
        failure=evaluation.failure_record({'id':'test-1','task':'test','expected_tools':[]},RuntimeError('private'))
        payload=report([failure])
        self.assertEqual(validate_payload(payload,require_current_revision=False)[1],[])
        self.assertFalse(self.gate(payload).passed)
        self.assertEqual(failure['trace_scope'],'evaluation')
        self.assertNotIn('private',json.dumps(failure))

    def test_failed_run_preserves_real_trace_and_prior_tools(self):
        error=RuntimeError('failure')
        error.agent_trace_id='real-agent-trace'
        error.agent_steps=[{'tool':'send_email','status':'approval_required'}]
        item=evaluation.failure_record({'id':'test-1','task':'test','forbidden_tools':['send_email']},error)
        self.assertEqual(item['trace_id'],'real-agent-trace')
        self.assertTrue(item['forbidden_tool_violation'])
        self.assertEqual(item['n_steps'],1)
        self.assertEqual(validate_payload(report([item]),require_current_revision=False)[1],[])

    def test_nonfinite_or_out_of_range_thresholds_rejected(self):
        for value in (float('nan'),float('inf'),-1,2):
            with self.assertRaises(ValueError): GateThresholds(min_task_success_rate=value)

    def test_judge_receives_success_criteria_execution_and_sources(self):
        source=SearchResult(email_id='e',chunk_id='e_0',content='evidence',score=1,metadata={})
        response=AgentResponse(answer='answer',sources=[source],metadata={
            'trace_id':'run','steps':[{'tool':'search_emails','status':'success'}],'status':'success'})
        with patch('agents.agent_loop.run_agent_loop',return_value=response), \
             patch.object(evaluation,'judge_success',return_value={'success':1,'reason':'ok'}) as judge:
            item=evaluation.evaluate_task({'id':'test-1','task':'find','expected_tools':['search_emails'],
                                          'success_criteria':'must cite evidence'},object())
        self.assertEqual(item['success'],1)
        self.assertEqual(judge.call_args.kwargs['success_criteria'],'must cite evidence')
        self.assertEqual(judge.call_args.kwargs['sources'][0]['content'],'evidence')

    def test_judge_cannot_override_runtime_failure(self):
        for status in ('timeout','context_budget_exceeded','unknown','partial','needs_review','max_steps_reached'):
            response=AgentResponse(answer='claims success',metadata={'steps':[],'status':status,'trace_id':'run'})
            with patch('agents.agent_loop.run_agent_loop',return_value=response), \
                 patch.object(evaluation,'judge_success',return_value={'success':1,'reason':'ok'}):
                item=evaluation.evaluate_task({'id':'test-1','task':'test','expected_tools':[]},object())
            self.assertEqual(item['success'],0,status)


if __name__=='__main__':
    unittest.main()
