"""Read-only environment diagnosis. Never load models, authorize OAuth or install packages."""
from __future__ import annotations

import argparse
from importlib import metadata
import ipaddress
import json
import math
import os
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parent.parent
PROFILES = {
    'test':['pytest','fastapi','pydantic','openai','httpx','python-dotenv','rank-bm25','mcp'],
    'core':['fastapi','uvicorn','pydantic','openai','chromadb','sentence-transformers','rank-bm25','mcp'],
    'gmail':['google-api-python-client','google-auth-oauthlib','google-auth-httplib2'],
    'ui':['streamlit','requests','sseclient-py'],
    'graph':['langgraph'],
}


def diagnose(settings, *, profile='core', root=ROOT, probe_filesystem=False):
    checks=[]
    def add(name,status,detail): checks.append({'check':name,'status':status,'detail':detail})
    add('python','ok' if sys.version_info>=(3,10) else 'error',platform.python_version())
    for package in PROFILES[profile]:
        try:
            add('dependency:'+package,'ok',metadata.version(package))
        except metadata.PackageNotFoundError:
            add('dependency:'+package,'error','not installed; no automatic installation attempted')
    for name, valid, default in [('AGENT_TOOL_BACKEND',{'local','mcp'},'local'),
                                 ('RERANKER_BACKEND',{'cross_encoder','llm'},'cross_encoder')]:
        add('config:'+name,'ok' if getattr(settings,name,default) in valid else 'error','enum validation')
    for name,default in [('AGENT_RUN_TIMEOUT',120),('LLM_TIMEOUT',60),('AGENT_MAX_STEPS',6),
                         ('AGENT_MAX_TOOL_CALLS',12),('AGENT_CONTEXT_CHAR_LIMIT',60000),
                         ('AGENT_TOOL_OUTPUT_LIMIT',4000),('MAIL_APPROVAL_TIMEOUT_SECONDS',30)]:
        value=getattr(settings,name,default)
        valid=type(value) in (int,float) and math.isfinite(value) and value>0
        add('config:'+name,'ok' if valid else 'error','must be a finite positive number')
    for prefix,default in [('API','127.0.0.1'),('MCP','127.0.0.1')]:
        host=getattr(settings,prefix+'_HOST',default)
        try:
            local=ipaddress.ip_address(host).is_loopback
        except ValueError:
            local=host=='localhost'
        token_present=bool(getattr(settings,prefix+'_AUTH_TOKEN',''))
        add('boundary:'+prefix,'ok' if local or token_present else 'error',
            'loopback or configured token required; no token value is displayed')
    corpus=Path(getattr(settings,'EMAIL_DATA_PATH',Path(root)/'data/emails.json'))
    add('corpus','ok' if corpus.is_file() else 'warning','file present' if corpus.is_file() else 'not present')
    from core.storage_paths import validate_chroma_path
    try:
        validate_chroma_path(getattr(settings,'CHROMA_PERSIST_DIR',Path(root)/'chroma_db'))
        add('filesystem:CHROMA_ASCII_PATH','ok','native index path is supported')
    except ValueError as exc:
        add('filesystem:CHROMA_ASCII_PATH','error',str(exc))
    for name,default in [('CHROMA_PERSIST_DIR','chroma_db'),('APPROVAL_STORE_PATH','data/approvals/pending_actions.json')]:
        path=Path(getattr(settings,name,Path(root)/default))
        ancestor=path if path.is_dir() else path.parent
        while not ancestor.exists() and ancestor!=ancestor.parent:
            ancestor=ancestor.parent
        add('filesystem:'+name,'ok' if os.access(ancestor,os.W_OK) else 'error','best-effort parent write access check')
    if profile=='gmail':
        token=Path(getattr(settings,'GMAIL_TOKEN_PATH',Path(root)/'credentials/gmail_token.json'))
        add('gmail_authorization','warning' if token.is_file() else 'error',
            'token file present; contents, grant and account not inspected' if token.is_file()
            else 'prepared token missing; run explicit local authorization when ready')
    from scripts.context_paths import check_context_path
    import sqlite3
    try:
        with sqlite3.connect(':memory:') as db:
            db.execute('CREATE VIRTUAL TABLE ctx_probe USING fts5(tokens)')
        add('context:fts5', 'ok', 'FTS5 available; multilingual behavior is verified separately')
    except sqlite3.Error:
        add('context:fts5', 'warning', 'FTS5 unavailable; bounded history fallback will be used')
    for name, default in [('SESSION_STORE_PATH','data/sessions/sessions.sqlite3'),
                           ('TOOL_RESULT_STORE_PATH','data/sessions/tool_results.sqlite3'),
                           ('JOB_STORE_PATH','data/jobs/jobs.sqlite3')]:
        report = check_context_path(getattr(settings,name,Path(root)/default), probe=probe_filesystem)
        add('context_path:'+name, report['status'], report)
    add('context:semantic_summary', 'warning' if getattr(settings,'ENABLE_CONTEXT_SUMMARY',False) else 'ok',
        'experimental; live quality requires separate evaluation' if getattr(settings,'ENABLE_CONTEXT_SUMMARY',False)
        else 'disabled by default; deterministic context remains available')
    add('readiness','unverified','models, live provider, index compatibility and network were not exercised')
    return {'schema_version':1,'profile':profile,'ready':False,
            'configuration_valid':not any(row['status']=='error' for row in checks),'checks':checks}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=PROFILES,default='core')
    parser.add_argument('--probe-filesystem',action='store_true',help='Create and remove a small temporary write probe in each state parent')
    args=parser.parse_args(argv)
    sys.path.insert(0,str(ROOT))
    try:
        import config.settings as settings
        result=diagnose(settings,profile=args.profile,probe_filesystem=args.probe_filesystem)
    except (ValueError,TypeError) as exc:
        result={'configuration_valid':False,'ready':False,'error':'invalid configuration','error_type':type(exc).__name__}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result['configuration_valid'] else 1


if __name__=='__main__':
    raise SystemExit(main())
