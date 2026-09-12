from types import SimpleNamespace
import json

from scripts import doctor, offline_benchmark


def test_doctor_never_displays_tokens_or_claims_model_readiness(monkeypatch,tmp_path):
    monkeypatch.setattr(doctor.metadata,'version',lambda name:'synthetic-version')
    settings=SimpleNamespace(API_HOST='0.0.0.0',API_AUTH_TOKEN='PRIVATE TOKEN',
                             MCP_HOST='127.0.0.1',MCP_AUTH_TOKEN='PRIVATE TOKEN',
                             EMAIL_DATA_PATH=tmp_path/'missing.json',CHROMA_PERSIST_DIR=tmp_path/'chroma',
                             APPROVAL_STORE_PATH=tmp_path/'approval.sqlite3',AGENT_TOOL_BACKEND='local')
    result=doctor.diagnose(settings,profile='test',root=tmp_path)
    assert result['configuration_valid'] and result['ready'] is False
    assert 'PRIVATE' not in json.dumps(result)
    assert not (tmp_path/'approval.sqlite3').exists()


def test_doctor_rejects_invalid_budget_and_remote_no_token(monkeypatch,tmp_path):
    monkeypatch.setattr(doctor.metadata,'version',lambda name:'synthetic-version')
    settings=SimpleNamespace(API_HOST='0.0.0.0',API_AUTH_TOKEN='',AGENT_RUN_TIMEOUT=-1,
                             EMAIL_DATA_PATH=tmp_path/'missing.json')
    result=doctor.diagnose(settings,profile='test',root=tmp_path)
    assert result['configuration_valid'] is False
    errors={row['check'] for row in result['checks'] if row['status']=='error'}
    assert {'boundary:API','config:AGENT_RUN_TIMEOUT'}.issubset(errors)


def test_synthetic_benchmark_uses_real_isolated_store_without_provider(tmp_path):
    result=offline_benchmark._case({'count':4,'body_chars':128,'workers':2,'work_dir':str(tmp_path)})
    assert result['attempted']==result['succeeded']==result['stored']==4 and result['failed']==0
    assert result['python_tracemalloc_peak_bytes']>0
    assert list(tmp_path.iterdir())==[]
