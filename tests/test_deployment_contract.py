import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from agents.eval_contract import effective_config

ROOT=Path(__file__).resolve().parents[1]


def test_frontend_docker_source_allowlist_can_run_app_and_shared_helpers(tmp_path):
    pytest.importorskip('streamlit.testing.v1')
    import shlex
    image=tmp_path/'frontend-image';image.mkdir()
    for line in (ROOT/'Dockerfile.frontend').read_text().splitlines():
        if not line.startswith('COPY '):continue
        _,source,target=shlex.split(line)
        if (ROOT/source).is_dir():
            shutil.copytree(ROOT/source,image/target,ignore=shutil.ignore_patterns('__pycache__'))
        else:
            shutil.copy2(ROOT/source,image/Path(source).name)
    probe=image/'probe.py'
    probe.write_text('''import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import dotenv
dotenv.load_dotenv=lambda *a,**k:False
import requests
from types import SimpleNamespace as NS
requests.get=lambda *a,**k:NS(ok=True,status_code=200,raise_for_status=lambda:None,json=lambda:{'jobs':[],'sessions':[],'accounts':[]})
from streamlit.testing.v1 import AppTest
app=AppTest.from_file('frontend/app.py',default_timeout=15).run()
assert not app.exception, str(app.exception)
from frontend.mailbox_view import folder_label
from core.cleaner import _normalize_whitespace
from core.evidence import source_coverage
assert folder_label('Archive')=='Archive'
assert _normalize_whitespace('a  b')=='a b'
assert source_coverage({})
''',encoding='utf-8')
    result=subprocess.run([sys.executable,'-I','-B',str(probe)],cwd=image,capture_output=True,text=True,timeout=40)
    assert result.returncode==0,result.stdout+result.stderr


def test_compose_isolates_host_paths_and_frontend_secrets():
    services=yaml.safe_load((ROOT/'docker-compose.yml').read_text())['services']
    api=services['api']
    env=api['environment']
    assert env['API_HOST']=='0.0.0.0'
    for name in ('CHROMA_PERSIST_DIR','EMAIL_DATA_PATH','APPROVAL_STORE_PATH','SESSION_STORE_PATH','TOOL_RESULT_STORE_PATH','JOB_STORE_PATH',
                 'AGENT_TRACE_LOG_PATH','MCP_AUDIT_LOG_PATH','MCP_SERVER_AUDIT_LOG_PATH','GMAIL_SYNC_OUTPUT_PATH','GMAIL_SYNC_STATE_PATH',
                 'MAIL_ACCOUNTS_PATH','IMAP_DATA_ROOT','MAIL_SCHEDULE_PATH'):
        assert env[name].startswith('/app/')
    for name in ('GMAIL_CREDENTIALS_PATH','GMAIL_TOKEN_PATH','GMAIL_READONLY_TOKEN_PATH'):
        assert env[name].startswith('/run/gmail/')
    assert 'env_file' not in services['frontend']
    assert set(services['frontend']['environment'])=={'API_URL','API_AUTH_TOKEN','AGENT_RUN_TIMEOUT'}
    assert all(str(port).startswith('127.0.0.1:') for service in services.values() for port in service['ports'])
    docker=(ROOT/'Dockerfile').read_text()
    assert 'ARG PRELOAD_EMBEDDING_MODEL=false' in docker and 'COPY data/' not in docker
    cmd=json.loads(next(line[4:] for line in docker.splitlines() if line.startswith('CMD ')))
    assert cmd[cmd.index('--workers')+1]=='1'


def test_provenance_records_new_budgets_without_local_model_paths():
    settings=SimpleNamespace(EMBEDDING_MODEL=r'C:\PRIVATE\models\embedding',MODEL_CONTEXT_TOKENS=12000,
                MODEL_RUN_TOKEN_LIMIT=20000,MODEL_RUN_COST_LIMIT=1.5,FILTER_METADATA_SCAN_LIMIT=100,
                MODEL_TOKEN_PRICES={'fixture':{'input':1,'output':2}},DEEPSEEK_API_KEY='PRIVATE',
                EMAIL_DATA_PATH='PRIVATE',MODEL_OUTPUT_RESERVE_TOKENS=2000,RETRIEVAL_TIMEZONE='Asia/Shanghai')
    result=effective_config(settings)
    assert result['MODEL_RUN_TOKEN_LIMIT']==20000 and result['MODEL_RUN_COST_LIMIT']==1.5
    assert result['FILTER_METADATA_SCAN_LIMIT']==100 and result['MODEL_CONTEXT_TOKENS']==12000
    assert result['MODEL_TOKEN_PRICES']==settings.MODEL_TOKEN_PRICES
    assert result['EMBEDDING_MODEL']['kind']=='local_path' and 'PRIVATE' not in json.dumps(result)


def test_local_frontend_launchers_bind_loopback_and_fixed_port():
    for filename in ('Makefile', 'tasks.ps1'):
        lines = [line for line in (ROOT/filename).read_text(encoding='utf-8').splitlines()
                 if 'frontend/app.py' in line and ('streamlit' in line)]
        assert len(lines) == 2  # Standalone UI and combined API/UI launchers.
        for line in lines:
            command = line.replace('"', '').replace(',', '')
            assert '--server.address 127.0.0.1' in command
            assert '--server.port 8501' in command
            assert '--server.headless true' in command


def test_explicit_preload_passes_model_revision_without_real_loader(monkeypatch):
    from scripts import preload_model
    calls=[]
    fake=SimpleNamespace(SentenceTransformer=lambda name,**kwargs:(calls.append((name,kwargs)) or
                            SimpleNamespace(encode=lambda *a,**k:None)))
    monkeypatch.setitem(sys.modules,'sentence_transformers',fake)
    monkeypatch.setattr(preload_model.cfg,'EMBEDDING_MODEL','fixture/model')
    monkeypatch.setattr(preload_model.cfg,'EMBEDDING_MODEL_REVISION','fixture-revision')
    monkeypatch.setattr(preload_model.cfg,'EMBEDDING_DEVICE','cpu')
    preload_model.main()
    assert calls==[('fixture/model',{'device':'cpu','revision':'fixture-revision'})]


@pytest.mark.skipif(os.name!='nt',reason='PowerShell task integration is Windows-specific')
def test_tasks_install_is_separate_and_preserves_exit_code(tmp_path):
    shell=shutil.which('pwsh') or shutil.which('powershell')
    if not shell: pytest.skip('PowerShell unavailable')
    (tmp_path/'tasks.ps1').write_text((ROOT/'tasks.ps1').read_text(),encoding='utf-8')
    stub=tmp_path/'python-stub.cmd'
    stub.write_text('@echo off\necho %*>> "%TASK_TEST_LOG%"\nexit /b %TASK_TEST_EXIT%\n',encoding='utf-8')
    env={**os.environ,'PYTHON':str(stub),'TASK_TEST_LOG':str(tmp_path/'calls.txt'),'TASK_TEST_EXIT':'0'}
    def run(command,*args):
        return subprocess.run([shell,'-NoProfile','-ExecutionPolicy','Bypass','-File',str(tmp_path/'tasks.ps1'),command,*args],
                              env=env,cwd=tmp_path,check=False,capture_output=True,text=True)
    assert run('install').returncode==0
    assert (tmp_path/'calls.txt').read_text().strip()=='-m pip install -r requirements.txt'
    assert run('ui').returncode==0
    assert (tmp_path/'calls.txt').read_text().splitlines()[-1] == (
        '-m streamlit run frontend/app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true')
    env['TASK_TEST_EXIT']='7'
    assert run('index').returncode==7
    index=tmp_path/'chroma_db';index.mkdir();(index/'fixture.txt').write_text('synthetic')
    assert run('clean','-IncludeIndex').returncode==0 and index.exists()
    assert run('clean','-IncludeIndex','-Apply').returncode==0 and not index.exists()


@pytest.mark.skipif(os.name!='nt',reason='Process tree cleanup is Windows-specific')
def test_tasks_run_stops_spawned_server_descendants(tmp_path):
    shell=shutil.which('pwsh') or shutil.which('powershell')
    if not shell: pytest.skip('PowerShell unavailable')
    (tmp_path/'tasks.ps1').write_text((ROOT/'tasks.ps1').read_text(),encoding='utf-8')
    (tmp_path/'api').mkdir()
    (tmp_path/'api'/'__init__.py').write_text('')
    (tmp_path/'api'/'main.py').write_text(
        "import subprocess, sys, time\nfrom pathlib import Path\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "Path('child.pid').write_text(str(child.pid))\ntime.sleep(60)\n",encoding='utf-8')
    (tmp_path/'streamlit.py').write_text(
        "import time\nfrom pathlib import Path\n"
        "deadline = time.monotonic() + 10\n"
        "while not Path('child.pid').exists() and time.monotonic() < deadline: time.sleep(.05)\n"
        "raise SystemExit(7)\n",encoding='utf-8')
    env={**os.environ,'PYTHON':sys.executable,'PYTHONDONTWRITEBYTECODE':'1'}
    taskkill=str(Path(os.environ['SystemRoot'])/'System32'/'taskkill.exe')
    child_id=None
    try:
        result=subprocess.run([shell,'-NoProfile','-ExecutionPolicy','Bypass','-File',str(tmp_path/'tasks.ps1'),'run'],
                              env=env,cwd=tmp_path,check=False,capture_output=True,text=True,timeout=25)
        assert result.returncode==7, result.stderr
        child_id=int((tmp_path/'child.pid').read_text())
        # Poll only the synthetic descendant from this test, never deployment processes.
        for _ in range(10):
            alive=subprocess.run([shell,'-NoProfile','-Command',
                f'if (Get-Process -Id {child_id} -ErrorAction SilentlyContinue) {{ exit 1 }}'],
                capture_output=True,check=False).returncode
            if alive==0: break
            time.sleep(.1)
        assert alive==0, 'Combined launcher left its server descendant running'
    finally:
        if child_id is None and (tmp_path/'child.pid').exists():
            child_id=int((tmp_path/'child.pid').read_text())
        if child_id is not None:
            subprocess.run([taskkill,'/PID',str(child_id),'/T','/F'],capture_output=True,check=False)
