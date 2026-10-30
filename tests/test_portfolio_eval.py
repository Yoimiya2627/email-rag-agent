import json

import pytest

from scripts.run_portfolio_eval import FIXTURES, ROOT, aggregate, score_case, validate_run_dir


CASE = {'gold_email_ids': ['budget'], 'required_patterns': ['12800'], 'forbidden_patterns': ['99999']}


def response(answer='12800 元 [budget#budget_chunk_0]'):
    return {'answer': answer, 'sources': [{'email_id': 'budget', 'chunk_id': 'budget_chunk_0'}],
            'metadata': {'completion_status': 'complete',
                         'model_usage': {'actual_total_tokens': 42, 'unknown_usage_calls': 0}}}


def test_answer_citation_must_resolve_to_an_actual_returned_chunk():
    result = score_case(CASE, response('12800 元 [budget#invented]'))
    assert result['checks']['gold_retrieved']
    assert result['checks']['gold_cited']
    assert not result['checks']['citation_ids_valid']
    assert not result['passed']


@pytest.mark.parametrize('answer', ['99999 元 [budget#budget_chunk_0]', '12800 元', '12800 元 [other#other_chunk_0]'])
def test_wrong_facts_or_missing_gold_citation_cannot_pass(answer):
    assert not score_case(CASE, response(answer))['passed']


def test_incomplete_generation_cannot_pass_even_with_correct_text():
    value = response()
    value['metadata']['completion_status'] = 'incomplete'
    assert not score_case(CASE, value)['passed']


def test_failed_request_stays_in_denominator_and_missing_usage_is_not_zero():
    good = response()
    rows = [{'seconds': 1, 'response': good, 'score': score_case(CASE, good)},
            {'seconds': 9, 'response': {}, 'score': score_case(CASE, {}, 503)}]
    summary = aggregate(rows)
    assert summary['case_count'] == 2
    assert summary['contract_pass_rate'] == .5
    assert summary['mean_gold_email_recall_at_5'] == .5
    assert summary['p95_seconds_nearest_rank'] == 9
    assert summary['known_provider_tokens'] == 42
    assert summary['total_provider_tokens'] is None
    assert summary['monetary_cost'] is None


def test_empty_eval_is_not_a_success():
    with pytest.raises(ValueError):
        aggregate([])


def test_runtime_does_not_reuse_a_directory_or_write_into_source(tmp_path):
    sentinel = tmp_path / 'keep.txt'
    sentinel.write_text('keep', encoding='utf-8')
    with pytest.raises(FileExistsError):
        validate_run_dir(tmp_path)
    with pytest.raises(ValueError):
        validate_run_dir(ROOT / 'portfolio-state')
    assert sentinel.read_text(encoding='utf-8') == 'keep'


def test_fixed_cases_refer_to_existing_synthetic_mail_and_have_unique_ids():
    emails = json.loads((FIXTURES / 'emails.json').read_text(encoding='utf-8'))
    cases = json.loads((FIXTURES / 'cases.json').read_text(encoding='utf-8'))
    assert len(emails) == len({mail['id'] for mail in emails}) == 12
    assert len(cases) == len({case['id'] for case in cases}) == 12
    assert all(set(case['gold_email_ids']) <= {mail['id'] for mail in emails} for case in cases)
    assert all(mail['sender'].endswith('@example.invalid') for mail in emails)
