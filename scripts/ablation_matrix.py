"""Plan isolated RAG ablations; model execution requires an explicit --execute."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parent.parent


def version_flags():
    # Read the existing evaluator's literal configurations without importing models/settings.
    tree=ast.parse((ROOT/'scripts/run_ragas_eval.py').read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='VERSION_FLAGS' for t in node.targets):
            return {ast.literal_eval(key):{kw.arg:ast.literal_eval(kw.value) for kw in value.keywords}
                    for key,value in zip(node.value.keys,node.value.values)}
    raise ValueError('canonical version flags unavailable')


def prepare_matrix(run_root,*,versions=None,limit=10,version_timeout=3600):
    flags=version_flags()
    versions=versions or list(flags)
    if not versions or len(versions)!=len(set(versions)) or any(version not in flags for version in versions):
        raise ValueError('invalid ablation versions')
    if type(limit) is not int or limit<1 or type(version_timeout) not in (int,float) or not 0<version_timeout<=86400:
        raise ValueError('invalid matrix budget')
    run=Path(run_root)/('rag-ablation-'+uuid.uuid4().hex)
    run.mkdir(parents=True,exist_ok=False)
    source=ROOT/'scripts/run_ragas_eval.py'
    dataset=ROOT/'data/ragas_testset.json'
    cells=[]
    for version in versions:
        directory=run/version
        directory.mkdir()
        cell={'version':version,'flags':flags[version],'limit':limit,'version_timeout_seconds':version_timeout,
              'execution_status':'not_executed','quality_scores':{'status':'unavailable'},
              'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'dataset_sha256':hashlib.sha256(dataset.read_bytes()).hexdigest() if dataset.is_file() else None,
              'run_directory':str(directory.resolve())}
        (directory/'config.json').write_text(json.dumps(cell,indent=2),encoding='utf-8')
        cells.append(cell)
    plan={'schema_version':1,'cells':cells,'limits':['One fresh subprocess per version; no claim of equal model/corpus snapshots without captured live provenance.',
        'No scores are available until the explicit model evaluation completes. Embedding proxy scores must remain distinct from LLM scores.',
        'Existing evaluator reads configured corpus/index and model credentials only during explicit execution.']}
    (run/'matrix.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    return run,plan


def execute_cell(cell):
    directory=Path(cell['run_directory'])
    env=os.environ.copy()
    for name,file in {'APPROVAL_STORE_PATH':'approvals.sqlite3','SESSION_STORE_PATH':'sessions.sqlite3',
                      'JOB_STORE_PATH':'jobs.sqlite3','AGENT_TRACE_LOG_PATH':'trace.jsonl',
                      'MCP_AUDIT_LOG_PATH':'audit.jsonl','MCP_SERVER_AUDIT_LOG_PATH':'server.jsonl'}.items():
        env[name]=str(directory/file)
    env['PYTHONDONTWRITEBYTECODE']='1'
    started=time.monotonic()
    try:
        with (directory/'execution.log').open('x',encoding='utf-8') as output:
            result=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--execute','--worker',str(directory/'config.json')],
                                  env=env,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,timeout=cell['version_timeout_seconds'],check=False)
        status='completed' if result.returncode==0 else 'failed'
        code=result.returncode
    except subprocess.TimeoutExpired:
        status,code='timeout',None
    result={'version':cell['version'],'execution_status':status,'exit_code':code,'elapsed_seconds':time.monotonic()-started,
            'quality_scores':{'status':'inspect_scoring_methods_in_evaluator_report' if status=='completed' else 'unavailable'}}
    (directory/'execution.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root')
    parser.add_argument('--versions',default='V1,V2,V3,V4,V5,V6,V7')
    parser.add_argument('--limit',type=int,default=10)
    parser.add_argument('--version-timeout',type=float,default=3600)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--worker',help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.worker:
        if not args.execute: parser.error('worker requires explicit execution')
        cell=json.loads(Path(args.worker).read_text(encoding='utf-8'))
        sys.path.insert(0,str(ROOT))
        from scripts import run_ragas_eval as evaluator
        if evaluator.VERSION_FLAGS[cell['version']]!=cell['flags']: raise ValueError('version flags changed since planning')
        evaluator.RESULTS_DIR=Path(cell['run_directory'])
        sys.argv=[str(ROOT/'scripts/run_ragas_eval.py'),'--versions',cell['version'],'--limit',str(cell['limit']),
                  '--output',str(evaluator.RESULTS_DIR/'comparison.json')]
        evaluator.main()
        return 0
    if not args.run_root: parser.error('--run-root is required')
    run,plan=prepare_matrix(args.run_root,versions=args.versions.split(','),limit=args.limit,version_timeout=args.version_timeout)
    results=[execute_cell(cell) for cell in plan['cells']] if args.execute else []
    print(json.dumps({'run_directory':str(run),'executed':bool(args.execute),'results':results},indent=2))
    return int(any(row['execution_status']!='completed' for row in results))


if __name__=='__main__': raise SystemExit(main())
