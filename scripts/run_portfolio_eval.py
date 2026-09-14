"""Small live RAG evaluation / isolated UI demo. Uses synthetic fixtures only.

Requires an existing embedding cache and a configured model API key. Every run
uses a NEW directory outside the repository; no mailbox provider is contacted.
Gold answers are scored locally and are never included in model requests.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'data' / 'portfolio_demo'


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def score_case(case, response, http_status=200):
    """Contract checks, not an LLM judge or a general faithfulness metric."""
    metadata = response.get('metadata') or {}
    answer = response.get('answer', '').replace(',', '').replace('，', '')
    sources = response.get('sources') or []
    source_pairs = {(s.get('email_id'), s.get('chunk_id')) for s in sources}
    returned = {s.get('email_id') for s in sources}
    gold = set(case['gold_email_ids'])
    # Parse the answer itself, so a server omitting an invalid citation does not
    # turn that invalid citation into an apparent success.
    citations = re.findall(r'\[([^\[\]#\s]+)#([^\[\]\s]+)\]', response.get('answer', ''))
    cited = {email for email, _ in citations}
    checks = {
        'http_ok': http_status == 200,
        'complete': metadata.get('completion_status') == 'complete',
        'required_facts': all(re.search(p, answer, re.I) for p in case['required_patterns']),
        'forbidden_facts_absent': not any(re.search(p, answer, re.I) for p in case['forbidden_patterns']),
        'gold_retrieved': gold <= returned,
        'gold_cited': gold <= cited,
        'citation_ids_valid': bool(citations) and all(pair in source_pairs for pair in citations),
    }
    return {'passed': all(checks.values()), 'checks': checks,
            'gold_email_recall': len(gold & returned) / len(gold),
            'citation_count': len(citations),
            'valid_citation_count': sum(pair in source_pairs for pair in citations)}


def aggregate(rows):
    if not rows:
        raise ValueError('Cannot report an empty evaluation')
    times = sorted(row['seconds'] for row in rows)
    usages = [(row.get('response', {}).get('metadata') or {}).get('model_usage') for row in rows]
    usage_complete = all(u is not None and u.get('unknown_usage_calls') == 0 for u in usages)
    citations = sum(row['score']['citation_count'] for row in rows)
    return {'case_count': len(rows), 'passed': sum(row['score']['passed'] for row in rows),
            'contract_pass_rate': sum(row['score']['passed'] for row in rows) / len(rows),
            'mean_gold_email_recall_at_5': sum(row['score']['gold_email_recall'] for row in rows) / len(rows),
            'citation_id_validity': (sum(row['score']['valid_citation_count'] for row in rows) / citations
                                     if citations else None),
            'citation_count': citations, 'mean_seconds': sum(times) / len(times),
            'p95_seconds_nearest_rank': times[math.ceil(.95 * len(times)) - 1],
            'provider_usage_complete': usage_complete,
            'known_provider_tokens': sum(u.get('actual_total_tokens', 0) for u in usages if u),
            'total_provider_tokens': (sum(u['actual_total_tokens'] for u in usages) if usage_complete else None),
            'monetary_cost': None}


def source_manifest():
    paths = set()
    for directory in ('agents', 'api', 'config', 'core', 'frontend', 'models'):
        paths.update((ROOT / directory).rglob('*.py'))
    paths.update(ROOT.glob('requirements*.txt'))
    paths.update([ROOT / 'scripts' / 'run_portfolio_eval.py', *FIXTURES.glob('*.json')])
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(
        p.read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in sorted(paths)}
    return {'sha256': hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
            'files': hashes, 'normalization': 'CRLF to LF; current working-tree contents, including uncommitted edits'}


def isolated_environment(run_dir, api_port):
    from dotenv import dotenv_values
    runtime = dotenv_values(ROOT / '.env')
    # Ignore all application settings inherited from the daily shell or .env.
    settings_text = (ROOT / 'config' / 'settings.py').read_text(encoding='utf-8')
    setting_keys = set(re.findall(r'os\.getenv\([\s]*[\'\"]([^\'\"]+)', settings_text))
    env = {key: value for key, value in os.environ.items() if key not in setting_keys and key not in runtime}
    allowed = ('DEEPSEEK_API_KEY', 'DEEPSEEK_BASE_URL', 'DEEPSEEK_MODEL', 'DEEPSEEK_THINKING_MODE',
               'AGENT_PLANNER_MODEL', 'EMBEDDING_MODEL', 'EMBEDDING_MODEL_REVISION',
               'HF_HOME', 'HF_HUB_CACHE', 'HF_ENDPOINT')
    for key in allowed:
        value = os.environ.get(key) or runtime.get(key)
        if value:
            env[key] = value
    if not env.get('DEEPSEEK_API_KEY'):
        raise ValueError('DEEPSEEK_API_KEY is required for this live evaluation/demo')
    state = run_dir / 'state'
    state.mkdir()
    paths = {'EMAIL_DATA_PATH': 'emails.json', 'CHROMA_PERSIST_DIR': 'chroma',
             'APPROVAL_STORE_PATH': 'approvals.sqlite3', 'SESSION_STORE_PATH': 'sessions.sqlite3',
             'TOOL_RESULT_STORE_PATH': 'tool-results.sqlite3', 'JOB_STORE_PATH': 'jobs.sqlite3',
             'MAIL_ACCOUNTS_PATH': 'accounts.sqlite3', 'IMAP_DATA_ROOT': 'imap',
             'MAIL_SCHEDULE_PATH': 'schedule.sqlite3', 'AGENT_TRACE_LOG_PATH': 'trace.jsonl',
             'MCP_AUDIT_LOG_PATH': 'audit.jsonl', 'MCP_SERVER_AUDIT_LOG_PATH': 'server-audit.jsonl',
             'GMAIL_CREDENTIALS_PATH': 'absent-client.json', 'GMAIL_TOKEN_PATH': 'absent-token.json',
             'GMAIL_READONLY_TOKEN_PATH': 'absent-readonly.json', 'GMAIL_SYNC_OUTPUT_PATH': 'gmail.json',
             'GMAIL_SYNC_STATE_PATH': 'gmail-state.json'}
    env.update({key: str(state / value) for key, value in paths.items()})
    (state / 'emails.json').write_bytes((FIXTURES / 'emails.json').read_bytes())
    env.update(API_AUTH_TOKEN=secrets.token_urlsafe(32), API_OWNER_ID='portfolio-demo',
               API_HOST='127.0.0.1', API_PORT=str(api_port), API_URL=f'http://127.0.0.1:{api_port}',
               MAIL_PROVIDER='simulated', ENABLE_REAL_EMAIL_SEND='false', MAIL_SCHEDULER_ENABLED='false',
               AGENT_TOOL_BACKEND='local', ENABLE_AGENT_TRACE='false', ENABLE_MCP_AUDIT='false',
               ENABLE_CONTEXT_SUMMARY='false', ENABLE_CONTEXT_CANDIDATES='false',
               ENABLE_BM25='true', ENABLE_RRF='true', ENABLE_RERANKER='false', ENABLE_QUERY_REWRITE='false',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1',
               EMBEDDING_DEVICE='cpu', EMBEDDING_CPU_THREADS='2', EMBEDDING_BATCH_SIZE='4',
               LLM_TIMEOUT='45', AGENT_RUN_TIMEOUT='120', AGENT_MAX_STEPS='6', AGENT_MAX_TOOL_CALLS='8',
               MODEL_RUN_TOKEN_LIMIT='60000', MODEL_CONTEXT_TOKENS='32000', MODEL_RUN_COST_LIMIT='0',
               WARMUP_ON_START='false', PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8')
    return env


def validate_run_dir(value):
    path = Path(value).expanduser().resolve()
    if path == ROOT or ROOT in path.parents:
        raise ValueError('Use a NEW runtime directory outside the source repository')
    if os.name == 'nt' and not str(path).isascii():
        raise ValueError('Windows Chroma runtime needs a full ASCII path')
    path.mkdir(parents=True, exist_ok=False)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, help='New directory outside repository; never reused or cleared')
    parser.add_argument('--api-port', type=int, default=18801)
    parser.add_argument('--ui-port', type=int, default=18802)
    parser.add_argument('--demo', action='store_true', help='Start empty-index API + UI; wait for Ctrl+C or run-dir/STOP')
    args = parser.parse_args()
    if not (1024 <= args.api_port <= 65535 and 1024 <= args.ui_port <= 65535) or args.api_port == args.ui_port:
        parser.error('Choose two distinct ports between 1024 and 65535')
    import requests
    run_dir = validate_run_dir(args.run_dir)
    env = isolated_environment(run_dir, args.api_port)
    manifest = source_manifest()
    write_json(run_dir / 'source-manifest.json', manifest)
    packages = {}
    for name in ('openai', 'chromadb', 'sentence-transformers', 'fastapi', 'streamlit', 'pydantic'):
        packages[name] = importlib.metadata.version(name)
    # Only an explicit public configuration allowlist is written to evidence.
    public_keys = ('DEEPSEEK_MODEL', 'AGENT_PLANNER_MODEL', 'EMBEDDING_MODEL', 'EMBEDDING_MODEL_REVISION',
                   'ENABLE_BM25', 'ENABLE_RRF', 'ENABLE_RERANKER', 'ENABLE_QUERY_REWRITE',
                   'MODEL_RUN_TOKEN_LIMIT', 'MODEL_CONTEXT_TOKENS', 'AGENT_MAX_STEPS', 'MAIL_PROVIDER')
    configuration = {key: env.get(key) for key in public_keys}
    configuration.update(DEEPSEEK_MODEL=env.get('DEEPSEEK_MODEL', 'deepseek-v4-flash'),
                         AGENT_PLANNER_MODEL=env.get('AGENT_PLANNER_MODEL', 'deepseek-chat'),
                         EMBEDDING_MODEL=env.get('EMBEDDING_MODEL', 'BAAI/bge-m3'))
    write_json(run_dir / 'environment.json', {'python': sys.version.split()[0], 'packages': packages,
                                             'configuration': configuration})
    processes, logs = [], []

    def start(name, code):
        handle = (run_dir / (name + '.log')).open('w', encoding='utf-8')
        logs.append(handle)
        process = subprocess.Popen([sys.executable, '-B', '-c', code], cwd=ROOT, env=env,
                                   stdout=handle, stderr=handle,
                                   creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        processes.append(process)
        return process

    session = requests.Session()
    session.headers['Authorization'] = 'Bearer ' + env['API_AUTH_TOKEN']

    def request(method, path, **kwargs):
        return session.request(method, env['API_URL'] + path, timeout=180, **kwargs)

    result = {'recorded_at': datetime.now(timezone.utc).isoformat(), 'source_sha256': manifest['sha256'],
              'configuration': configuration, 'scope': '12 synthetic emails; 12 fixed /query cases; one run per case',
              'scoring': 'Frozen regex facts + gold email retrieval + citation ID checks; manual semantic review is separate',
              'rows': []}
    exit_code = 1
    try:
        api = start('api', "import dotenv; dotenv.load_dotenv=lambda *a,**k:False; import uvicorn; "
                    f"uvicorn.run('api.main:app',host='127.0.0.1',port={args.api_port},workers=1,log_level='warning')")
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if api.poll() is not None:
                raise RuntimeError('API exited during startup; inspect api.log')
            try:
                if request('GET', '/health').status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(.25)
        else:
            raise TimeoutError('API startup timeout')
        # Prove the process is ours: the unique bearer token must work, and the
        # fresh state must be empty before either indexing or browser recording.
        status = request('GET', '/index/status')
        status.raise_for_status()
        if status.json().get('chunk_count') != 0:
            raise RuntimeError('Expected an empty isolated index')
        if args.demo:
            ui = start('frontend', "import dotenv; dotenv.load_dotenv=lambda *a,**k:False; import sys; "
                       "from streamlit.web.cli import main; "
                       f"sys.argv=['streamlit','run','frontend/app.py','--server.address','127.0.0.1','--server.port','{args.ui_port}',"
                       "'--server.headless','true','--browser.gatherUsageStats','false']; main()")
            print(json.dumps({'demo_url': f'http://127.0.0.1:{args.ui_port}',
                              'data_path': str(FIXTURES / 'emails.json'), 'provider': 'simulated'}), flush=True)
            while not (run_dir / 'STOP').exists():
                if api.poll() is not None or ui.poll() is not None:
                    raise RuntimeError('Demo process exited; inspect local logs')
                time.sleep(.5)
            exit_code = 0
        else:
            started = time.perf_counter()
            indexed = request('POST', '/index', json={})
            indexed.raise_for_status()
            result['index'] = {'seconds': time.perf_counter() - started, 'response': indexed.json()}
            if not indexed.json().get('success'):
                raise RuntimeError('Indexing failed')
            warm = request('POST', '/warmup')
            warm.raise_for_status()
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                if request('GET', '/ready').status_code == 200:
                    break
                time.sleep(.5)
            else:
                raise TimeoutError('Warmup did not reach ready')
            cases = json.loads((FIXTURES / 'cases.json').read_text(encoding='utf-8'))
            for case in cases:
                started = time.perf_counter()
                try:
                    response = request('POST', '/query', json={'query': case['query'], 'top_k': 5})
                    body = response.json()
                    http_status = response.status_code
                except (requests.RequestException, ValueError) as exc:
                    body, http_status = {'error_type': type(exc).__name__}, 0
                row = {'id': case['id'], 'query': case['query'], 'gold_email_ids': case['gold_email_ids'],
                       'seconds': time.perf_counter() - started, 'http_status': http_status,
                       'response': body, 'score': score_case(case, body, http_status)}
                result['rows'].append(row)
                result['summary'] = aggregate(result['rows'])
                write_json(run_dir / 'results.json', result)
                print(json.dumps({'id': case['id'], 'passed': row['score']['passed'],
                                  'seconds': round(row['seconds'], 2)}, ensure_ascii=False), flush=True)
            result['source_unchanged_during_run'] = source_manifest()['sha256'] == manifest['sha256']
            write_json(run_dir / 'results.json', result)
            exit_code = 0 if result['source_unchanged_during_run'] and all(r['score']['passed'] for r in result['rows']) else 1
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        session.close()
        for handle in logs:
            handle.close()
        write_json(run_dir / 'cleanup.json', {'all_child_processes_stopped': all(p.poll() is not None for p in processes)})
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
