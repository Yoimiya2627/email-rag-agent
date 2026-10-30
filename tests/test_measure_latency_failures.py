from unittest.mock import patch

from scripts import measure_latency as bench


def test_failed_request_remains_in_denominator_and_has_elapsed_time():
    with patch.object(bench, 'time_one', side_effect=[2.0, TimeoutError('private')]), \
         patch.object(bench.time, 'perf_counter', side_effect=[0.0, 2.0, 5.0]), \
         patch.object(bench.time, 'sleep'), patch('core.reranker.reset_circuit_breaker'):
        result = bench.measure_version('V2', ['first', 'second'])
    assert result['attempted'] == 2 and result['succeeded'] == 1 and result['failed'] == 1
    assert result['success_rate'] == 0.5 and result['n'] == 1
    assert result['attempts'][1] == {'index':1, 'status':'error', 'seconds':3.0, 'error_type':'TimeoutError'}
    assert result['median'] == 2.0 and result['latency_population'] == 'successful_requests_only'
    assert 'private' not in str(result)


def test_all_failures_report_counts_and_unavailable_success_quantiles():
    with patch.object(bench, 'time_one', side_effect=RuntimeError()), patch.object(bench.time, 'sleep'), \
         patch('core.reranker.reset_circuit_breaker'):
        result = bench.measure_version('V2', ['one', 'two'])
    assert result['attempted'] == result['failed'] == 2
    assert result['success_rate'] == 0 and result['median'] is None and result['p95'] is None


def test_empty_input_has_no_invented_success_rate():
    with patch('core.reranker.reset_circuit_breaker'):
        result = bench.measure_version('V2', [])
    assert result['attempted'] == 0 and result['success_rate'] is None


def test_incomplete_model_answer_is_not_counted_as_success():
    from core.model_outcomes import ModelText
    with patch('core.pipeline.retrieve',return_value=[]), \
         patch('core.generator.generate_answer',return_value=ModelText('partial',completion_status='incomplete')), \
         patch.object(bench.time,'sleep'), patch('core.reranker.reset_circuit_breaker'):
        result=bench.measure_version('V2',['one'])
    assert result['failed'] == 1 and result['succeeded'] == 0
