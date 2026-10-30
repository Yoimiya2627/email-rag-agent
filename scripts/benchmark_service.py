"""Explicit HTTP RAG service measurement; never invokes Agent approval/job endpoints."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import random
import threading
import time
from urllib.parse import urlsplit
import uuid

import httpx


def _complete(metadata):
    return (isinstance(metadata,dict) and metadata.get('status')=='success'
            and metadata.get('completion_status')=='complete')


def measure_request(client,*,endpoint,query,timeout,clock=time.perf_counter):
    started=clock()
    row={'status':'error','error_code':None,'ttft_seconds':None,'response_chars':0,'http_status':None}
    try:
        if endpoint=='query':
            response=client.post('/query',json={'query':query},timeout=timeout)
            row['http_status']=response.status_code
            if response.status_code!=200:
                row['error_code']='http_'+str(response.status_code)
            else:
                data=response.json()
                if not isinstance(data,dict) or not isinstance(data.get('answer'),str):
                    row['error_code']='invalid_response_shape'
                elif not data['answer'].strip() or not _complete(data.get('metadata')):
                    row['error_code']='incomplete_response'
                else:
                    row.update(status='success',response_chars=len(data['answer']))
        elif endpoint=='stream':
            done,complete=False,False
            # Unique session prevents cross-attempt context growth; use an isolated service store.
            with client.stream('POST','/chat/stream',json={'query':query,'session_id':'benchmark-'+uuid.uuid4().hex},timeout=timeout) as response:
                row['http_status']=response.status_code
                if response.status_code!=200:
                    row['error_code']='http_'+str(response.status_code)
                else:
                    for line in response.iter_lines():
                        if clock()-started>timeout: raise TimeoutError('request deadline')
                        if not line.startswith('data:'): continue
                        raw=line[5:].strip()
                        if raw=='[DONE]': done=True; break
                        event=json.loads(raw)
                        if not isinstance(event,dict): raise ValueError('invalid SSE event')
                        if event.get('error'): row['error_code']='stream_error'
                        if 'token' in event:
                            if not isinstance(event['token'],str): raise ValueError('invalid SSE token')
                            if event['token']:
                                if row['ttft_seconds'] is None: row['ttft_seconds']=clock()-started
                                row['response_chars']+=len(event['token'])
                        if 'metadata' in event: complete=_complete(event['metadata'])
                    if done and complete and row['response_chars'] and not row['error_code']:
                        row['status']='success'
                    elif not row['error_code']: row['error_code']='incomplete_stream'
        else: raise ValueError('unsupported endpoint')
    except Exception as exc:
        row['error_code']=type(exc).__name__
    row['elapsed_seconds']=max(0,clock()-started)
    if row['elapsed_seconds']>timeout:
        row.update(status='error',error_code='request_budget_exceeded')
    return row


def run_benchmark(*,url,run_root,query='Summarize the synthetic fixture project.',endpoint='query',
                  attempts=100,concurrency=1,duration_seconds=None,timeout=60,cold_process_first=False,
                  token=None,rss_pid=None,client_factory=None,clock=time.perf_counter):
    parts=urlsplit(url)
    if parts.scheme not in {'http','https'} or not parts.hostname or parts.username or parts.password or parts.query or parts.fragment or parts.path not in {'','/'}:
        raise ValueError('url must be an explicit HTTP(S) origin without credentials/path/query')
    if endpoint not in {'query','stream'} or type(concurrency) is not int or not 1<=concurrency<=64:
        raise ValueError('invalid endpoint/concurrency')
    if type(attempts) is not int or not 0<=attempts<=10_000_000:
        raise ValueError('attempts must be between 0 and 10000000')
    if duration_seconds is not None and (type(duration_seconds) not in (int,float) or not math.isfinite(duration_seconds) or not 0<duration_seconds<=604800):
        raise ValueError('duration must be positive and at most seven days')
    if not attempts and duration_seconds is None: raise ValueError('unlimited attempts require a duration')
    if type(timeout) not in (int,float) or not math.isfinite(timeout) or timeout<=0: raise ValueError('invalid timeout')
    if not isinstance(query,str) or not 1<=len(query)<=20000: raise ValueError('invalid query length')
    run=Path(run_root)/('service-'+uuid.uuid4().hex)
    run.mkdir(parents=True,exist_ok=False)
    config={'url':url,'endpoint':endpoint,'attempt_limit':attempts,'concurrency':concurrency,
            'duration_seconds':duration_seconds,'request_timeout':timeout,'query_chars':len(query),
            'query_sha256':hashlib.sha256(query.encode()).hexdigest(),'cold_process_first_attested':bool(cold_process_first),
            'rss_pid':rss_pid,'authentication_configured':bool(token)}
    (run/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    started=clock()
    deadline=started+duration_seconds if duration_seconds is not None else math.inf
    lock=threading.Lock()
    counters={'attempted':0,'succeeded':0,'failed':0,'ttft_observed':0}
    samples=[]
    sampler=random.Random(0)
    max_rss=None
    errors={}
    if rss_pid is not None:
        import psutil
        process=psutil.Process(rss_pid)
    else: process=None
    factory=client_factory or (lambda:httpx.Client(base_url=url,headers={'Authorization':'Bearer '+token} if token else {},follow_redirects=False))
    def claim():
        with lock:
            if clock()>=deadline or (attempts and counters['attempted']>=attempts): return None
            index=counters['attempted']
            counters['attempted']+=1
            return index
    def one(client,index,output):
        nonlocal max_rss
        remaining=min(timeout,deadline-clock())
        row=measure_request(client,endpoint=endpoint,query=query,timeout=max(.001,remaining),clock=clock)
        row.update(attempt=index+1,phase=('cold_process_first_attested' if cold_process_first else 'first_observed') if index==0 else 'subsequent_request')
        rss=None
        if process:
            try: rss=process.memory_info().rss
            except Exception: pass
        with lock:
            counters['succeeded' if row['status']=='success' else 'failed']+=1
            counters['ttft_observed']+=row['ttft_seconds'] is not None
            if row['error_code']: errors[row['error_code']]=errors.get(row['error_code'],0)+1
            if rss is not None: max_rss=max(max_rss or 0,rss)
            if row['status']=='success' and index>0:
                observation={'elapsed_seconds':row['elapsed_seconds'],'ttft_seconds':row['ttft_seconds']}
                successful_subsequent=counters['succeeded']-(1 if first and first['status']=='success' else 0)
                if len(samples)<10000: samples.append(observation)
                else:
                    slot=sampler.randrange(max(1,successful_subsequent))
                    if slot<10000: samples[slot]=observation
            output.write(json.dumps(row)+'\n'); output.flush()
        return row
    first=None
    with (run/'attempts.jsonl').open('x',encoding='utf-8') as output:
        with factory() as client:
            index=claim()
            if index is not None: first=one(client,index,output)
        def worker():
            with factory() as client:
                while True:
                    index=claim()
                    if index is None: return
                    one(client,index,output)
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures=[executor.submit(worker) for _ in range(concurrency)]
            for future in futures: future.result()
    def quantiles(key):
        values=sorted(row[key] for row in samples if row[key] is not None)
        return {label:values[min(len(values)-1,math.ceil(len(values)*fraction)-1)] if values else None
                for label,fraction in [('p50',.5),('p95',.95),('p99',.99)]}
    result={**counters,'success_rate':counters['succeeded']/counters['attempted'] if counters['attempted'] else None,
            'elapsed_seconds':clock()-started,'first_request':first,'errors':errors,'rss_peak_observed_bytes':max_rss,
            'subsequent_success_elapsed':quantiles('elapsed_seconds'),'subsequent_success_ttft':quantiles('ttft_seconds'),
            'quantile_sample_count':len(samples),'quantile_method':'successful subsequent requests; reservoir <=10000',
            'limits':['Cold state is operator-attested; cache warmth is not verified.',
                      'TTFT begins at client request start and requires the first nonempty SSE token.',
                      'RSS is optional sampled process RSS, not whole-machine memory or exact peak.',
                      'HTTP read timeout bounds idle reads; underlying SDK/socket scheduling can exceed the budget.',
                      'SSE writes session history: point the service at isolated benchmark state.']}
    (run/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return run,result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',required=True)
    parser.add_argument('--run-root',required=True)
    parser.add_argument('--endpoint',choices=['query','stream'],default='query')
    parser.add_argument('--query',default='Summarize the synthetic fixture project.')
    parser.add_argument('--attempts',type=int,default=100)
    parser.add_argument('--concurrency',type=int,default=1)
    parser.add_argument('--duration-seconds',type=float)
    parser.add_argument('--timeout',type=float,default=60)
    parser.add_argument('--cold-process-first',action='store_true')
    parser.add_argument('--token-env')
    parser.add_argument('--rss-pid',type=int)
    args=parser.parse_args(argv)
    token=os.environ.get(args.token_env) if args.token_env else None
    if args.token_env and not token: parser.error('requested token environment variable is empty')
    run,result=run_benchmark(url=args.url,run_root=args.run_root,query=args.query,endpoint=args.endpoint,
                            attempts=args.attempts,concurrency=args.concurrency,duration_seconds=args.duration_seconds,
                            timeout=args.timeout,cold_process_first=args.cold_process_first,token=token,rss_pid=args.rss_pid)
    print(json.dumps({'run_directory':str(run),**result},indent=2))
    return 1 if result['failed'] else 0


if __name__=='__main__': raise SystemExit(main())
