"""Synthetic approval-store microbenchmarks in fresh processes; never use real mail/models.

These measure SQLite approval persistence only, not RAG quality, embedding startup,
model TTFT or full-service capacity. Each case has an isolated temporary store.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
import types

ROOT=Path(__file__).resolve().parent.parent
sys.dont_write_bytecode=True


def _load_store():
    settings=types.ModuleType('config.settings')
    settings.APPROVAL_TTL_SECONDS=86400
    settings.APPROVAL_STORE_PATH='not-used.sqlite3'
    package=types.ModuleType('config')
    package.settings=settings
    previous={name:sys.modules.get(name) for name in ('config','config.settings')}
    sys.modules.update({'config':package,'config.settings':settings})
    try:
        spec=importlib.util.spec_from_file_location('benchmark_approvals',ROOT/'agents/approvals.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.ApprovalStore
    finally:
        for name,value in previous.items():
            if value is None: sys.modules.pop(name,None)
            else: sys.modules[name]=value


def _case(case):
    count,length,workers=[int(case[name]) for name in ('count','body_chars','workers')]
    if not (1<=count<=10000 and 1<=length<=100000 and 1<=workers<=16):
        raise ValueError('case exceeds explicit benchmark limits')
    work=Path(case['work_dir']).resolve()
    work.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    Store=_load_store()
    imported=time.perf_counter()-started
    tracemalloc.start()
    rows=[]
    with tempfile.TemporaryDirectory(prefix='approval-bench-',dir=work) as directory:
        initialized=time.perf_counter()
        store=Store(Path(directory)/'synthetic.sqlite3')
        initialize_seconds=time.perf_counter()-initialized
        payload={'to':['synthetic@example.invalid'],'subject':'synthetic benchmark','body':'x'*length}
        def create(index):
            tick=time.perf_counter()
            try:
                store.create('send_email',payload,owner_id='benchmark',request_id=str(index))
                return {'status':'success','seconds':time.perf_counter()-tick}
            except Exception as exc:
                return {'status':'error','seconds':time.perf_counter()-tick,'error_type':type(exc).__name__}
        first=create(0)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows=[first]+list(pool.map(create,range(1,count)))
        listed=time.perf_counter()
        stored=len(store.list(owner_id='benchmark'))
        list_seconds=time.perf_counter()-listed
        _,peak=tracemalloc.get_traced_memory()
    tracemalloc.stop()
    successful=[row['seconds'] for row in rows if row['status']=='success']
    warm=[row['seconds'] for row in rows[1:] if row['status']=='success']
    sorted_seconds=sorted(successful)
    return {'status':'success' if len(successful)==count else 'partial','case':{k:v for k,v in case.items() if k!='work_dir'},
            'attempted':count,'succeeded':len(successful),'failed':count-len(successful),'stored':stored,
            'fresh_process_import_seconds':imported,'store_initialize_seconds':initialize_seconds,'first_create_seconds':first['seconds'],
            'warm_create_median_seconds':statistics.median(warm) if warm else None,
            'create_p95_seconds':sorted_seconds[max(0,math.ceil(len(sorted_seconds)*.95)-1)] if sorted_seconds else None,
            'list_seconds':list_seconds,'wall_seconds':time.perf_counter()-started,
            'python_tracemalloc_peak_bytes':peak,'attempts':rows}


def run_matrix(*,sizes,lengths,workers,work_dir,timeout=60):
    results=[]
    for count in sizes:
        for length in lengths:
            for concurrency in workers:
                case={'count':count,'body_chars':length,'workers':concurrency,'work_dir':str(Path(work_dir).resolve())}
                started=time.perf_counter()
                try:
                    process=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--worker',json.dumps(case)],
                                           capture_output=True,text=True,timeout=timeout,check=True)
                    results.append({**json.loads(process.stdout),'fresh_process_wall_seconds':time.perf_counter()-started})
                except (subprocess.SubprocessError,ValueError) as exc:
                    results.append({'status':'error','case':{k:v for k,v in case.items() if k!='work_dir'},
                                    'error_type':type(exc).__name__,'wall_seconds':time.perf_counter()-started})
    return {'schema_version':1,'benchmark':'synthetic_approval_store','results':results,
            'source_sha256':hashlib.sha256((ROOT/'agents/approvals.py').read_bytes()).hexdigest(),
            'limitations':['No real provider, model, RAG retrieval or API startup was measured.',
                           'tracemalloc measures Python allocations, not total RSS/native SQLite memory.',
                           'Synthetic results do not establish real mailbox capacity.']}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes',default='10,100')
    parser.add_argument('--lengths',default='256,2048')
    parser.add_argument('--workers',default='1,2')
    parser.add_argument('--work-dir')
    parser.add_argument('--output')
    parser.add_argument('--worker',help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.worker:
        print(json.dumps(_case(json.loads(args.worker))))
        return
    if not args.work_dir or not args.output:
        parser.error('--work-dir and --output are required')
    parse=lambda value:[int(part) for part in value.split(',')]
    result=run_matrix(sizes=parse(args.sizes),lengths=parse(args.lengths),workers=parse(args.workers),work_dir=args.work_dir)
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(output),'cases':len(result['results']),
                      'failed_cases':sum(row['status']!='success' for row in result['results'])}))
    return 1 if any(row['status']!='success' for row in result['results']) else 0


if __name__=='__main__':
    raise SystemExit(main())
