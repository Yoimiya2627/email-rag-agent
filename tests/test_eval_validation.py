"""Provider JSON and checkout differences must not corrupt evaluation results."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import config.settings as cfg
from scripts import run_ragas_eval as evaluation
from agents import eval_contract


def scores(value=0.5):
    return dict.fromkeys(evaluation.METRICS, value)


def client(payloads):
    responses = [SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(value)), finish_reason='stop')])
                 for value in payloads]
    create = Mock(side_effect=responses)
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


class EvalValidationTests(unittest.TestCase):
    def test_invalid_scores_rejected_without_coercion(self):
        invalid = [None, [], {}, {**scores(), 'faithfulness':None}]
        invalid += [scores(value) for value in ('high', '0.5', True, False, -1, 2, float('nan'), float('inf'), -float('inf'))]
        for item in invalid:
            with self.subTest(item=item), self.assertRaises(ValueError):
                evaluation.validate_scores(item)
        self.assertEqual(evaluation.validate_scores({**scores(1), 'question':'cannot override'}), scores(1.0))

    def test_bad_judge_result_retries_and_recovers(self):
        model = client([scores('high'), scores(0.8)])
        with patch.object(evaluation.time, 'sleep'), patch.object(evaluation, '_score_by_embedding') as fallback:
            result = evaluation.score_response(model, 'q', 'a', ['ctx'])
        self.assertEqual(model.chat.completions.create.call_count, 2)
        self.assertEqual(result['scoring_method'], 'llm')
        self.assertEqual(evaluation.validate_scores(result), scores(0.8))
        fallback.assert_not_called()

    def test_invalid_judge_falls_back_to_marked_embedding_proxy(self):
        model = client([scores(2)]*3)
        with patch.object(evaluation.time, 'sleep'), patch.object(evaluation, '_score_by_embedding', return_value=scores(0.4)):
            result = evaluation.score_response(model, 'q', 'a', ['ctx'])
        self.assertEqual(result['scoring_method'], 'embedding_proxy')
        self.assertEqual(result['scoring_status'], 'degraded')
        self.assertEqual(evaluation.validate_scores(result), scores(0.4))

    def test_bad_embedding_scores_are_recorded_as_failed_not_nan(self):
        model = client([{}]*3)
        with patch.object(evaluation.time, 'sleep'), patch.object(evaluation, '_score_by_embedding', return_value=scores(float('nan'))):
            result = evaluation.score_response(model, 'q', 'a', ['ctx'])
        self.assertEqual(result['scoring_status'], 'error')
        self.assertEqual(result['scoring_method'], 'unavailable')
        self.assertEqual(evaluation.validate_scores(result), scores(0))
        json.dumps(result, allow_nan=False)

    def test_bad_record_cannot_crash_aggregation_or_replace_question(self):
        records = [{**scores('high')}, {**scores(1), 'question':'forged', 'scoring_method':'llm', 'scoring_status':'success'}]
        cases = [{'question':'bad', 'ground_truth':'g'}, {'question':'good', 'ground_truth':'g'}]
        with patch.multiple(cfg, **{name:getattr(cfg,name) for name in evaluation.VERSION_FLAGS['V2']}), \
             patch.object(evaluation.time, 'sleep'), \
             patch.object(evaluation, 'run_single', return_value={'answer':'a','contexts':['ctx']}), \
             patch.object(evaluation, 'score_response', side_effect=records):
            result = evaluation.evaluate_version('V2', cases, 2, object())
        self.assertEqual([r['question'] for r in result['records']], ['bad','good'])
        self.assertEqual(result['avg'], scores(0.5))
        self.assertEqual(result['scoring']['failed_count'], 1)
        self.assertTrue(result['scoring']['mixed_methods'])
        self.assertEqual(result['scoring']['averages_by_method']['llm'], scores(1))
        json.dumps(result, allow_nan=False)

    def test_fingerprints_match_across_lf_crlf_and_still_detect_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = [Path(directory)/name for name in ('lf','crlf')]
            for root, newline in zip(roots, (b'\n',b'\r\n')):
                (root/'core').mkdir(parents=True)
                (root/'data').mkdir()
                (root/'core/a.py').write_bytes(b'def answer():\n    return 1\n'.replace(b'\n',newline))
                (root/'data/agent_testset.json').write_bytes(b'[\n {"id":"case"}\n]\n'.replace(b'\n',newline))
            left, right = [eval_contract.build_provenance(root=root) for root in roots]
            for key in ('source_sha256','dataset_sha256','fingerprint_format'):
                self.assertEqual(left[key], right[key], key)
            (roots[1]/'core/a.py').write_text('def answer():\n    return 2\n', encoding='utf-8')
            self.assertNotEqual(left['source_sha256'], eval_contract.source_fingerprint(roots[1]))
            (roots[1]/'data/agent_testset.json').write_text('[{"id":"changed"}]', encoding='utf-8')
            self.assertNotEqual(left['dataset_sha256'], eval_contract.build_provenance(root=roots[1])['dataset_sha256'])

    def test_current_revision_requires_fingerprint_format(self):
        from tests.test_agent_eval_gate import report, record
        case = json.loads((eval_contract.ROOT/'data/agent_testset.json').read_text(encoding='utf-8'))[0]
        item = record(case['id'])
        item.update({key:case.get(key,default) for key,default in [('task',''),('success_criteria',''),
                    ('task_type','general'),('risk_level','low'),('expected_tools',[]),('forbidden_tools',[])]})
        item['actual_tools']=list(item['expected_tools'])
        item['steps']=[{'tool':tool,'status':'success'} for tool in item['actual_tools']]
        item['n_steps']=len(item['steps'])
        payload = report([item])
        self.assertEqual(eval_contract.validate_payload(payload)[1], [])
        payload['provenance'].pop('fingerprint_format')
        self.assertIn('stale or missing provenance.fingerprint_format', eval_contract.validate_payload(payload)[1])


if __name__ == '__main__':
    unittest.main()
