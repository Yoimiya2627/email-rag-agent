import json
import threading

import httpx

from scripts.benchmark_service import measure_request,run_benchmark
from scripts.ablation_matrix import prepare_matrix


def test_http_attempts_include_structured_errors_and_missing_completion(tmp_path):
    count=0
    lock=threading.Lock()
    def handler(request):
        nonlocal count
        with lock: count+=1; current=count
        if current%3==0: return httpx.Response(429,json={'detail':'PRIVATE ERROR'})
        if current%3==1: return httpx.Response(200,json={'answer':'PRIVATE','metadata':{'status':'success','completion_status':'complete'}})
        return httpx.Response(200,json={'answer':'truncated','metadata':{'status':'incomplete'}})
    factory=lambda:httpx.Client(base_url='http://synthetic.test',transport=httpx.MockTransport(handler))
    run,result=run_benchmark(url='http://synthetic.test',run_root=tmp_path,attempts=9,concurrency=3,client_factory=factory)
    assert result['attempted']==9 and result['succeeded']==3 and result['failed']==6
    assert result['success_rate']==1/3 and result['first_request']['phase']=='first_observed'
    assert 'PRIVATE' not in (run/'attempts.jsonl').read_text()
    other,_=run_benchmark(url='http://synthetic.test',run_root=tmp_path,attempts=1,client_factory=factory)
    assert other!=run


def test_sse_ttft_ignores_metadata_and_counts_failed_partial_stream():
    tick=[0.0]
    class Body(httpx.SyncByteStream):
        def __iter__(self):
            tick[0]=1
            yield b'data: {"intent":"retrieve"}\n\n'
            tick[0]=3
            yield b'data: {"token":"first"}\n\n'
            tick[0]=5
            yield b'data: {"metadata":{"status":"success","completion_status":"complete"}}\n\ndata: [DONE]\n\n'
    client=httpx.Client(base_url='http://synthetic.test',transport=httpx.MockTransport(lambda _:httpx.Response(200,stream=Body())))
    result=measure_request(client,endpoint='stream',query='fixture',timeout=10,clock=lambda:tick[0])
    assert result['ttft_seconds']==3 and result['elapsed_seconds']==5 and result['status']=='success'
    client.close()
    client=httpx.Client(base_url='http://synthetic.test',transport=httpx.MockTransport(lambda _:httpx.Response(200,content=b'data: {"token":"partial"}\n\ndata: [DONE]\n\n')))
    result=measure_request(client,endpoint='stream',query='fixture',timeout=10)
    assert result['status']=='error' and result['error_code']=='incomplete_stream'
    client.close()


def test_duration_stops_new_attempts_with_fake_clock(tmp_path):
    ticks=[0.0]
    def handler(request):
        ticks[0]+=1
        return httpx.Response(200,json={'answer':'fixture','metadata':{'status':'success','completion_status':'complete'}})
    factory=lambda:httpx.Client(base_url='http://synthetic.test',transport=httpx.MockTransport(handler))
    _,result=run_benchmark(url='http://synthetic.test',run_root=tmp_path,attempts=0,duration_seconds=3,
                            concurrency=1,client_factory=factory,clock=lambda:ticks[0])
    assert result['attempted']==3 and result['succeeded']==3


def test_ablation_plan_contains_canonical_flags_without_model_execution(tmp_path):
    run,plan=prepare_matrix(tmp_path,versions=['V1','V7'],limit=2)
    assert plan['cells'][0]['flags']['ENABLE_BM25'] is False
    assert plan['cells'][1]['flags']['RERANKER_BACKEND']=='cross_encoder'
    assert all(cell['execution_status']=='not_executed' and cell['quality_scores']['status']=='unavailable' for cell in plan['cells'])
    assert (run/'V7/config.json').exists()
