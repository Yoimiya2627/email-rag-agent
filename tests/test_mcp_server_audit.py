import json
import time

import pytest

import config.settings as cfg
import mcp_server


def audit_file(monkeypatch, tmp_path):
    path=tmp_path/'server.jsonl'
    monkeypatch.setattr(cfg,'ENABLE_MCP_AUDIT',True)
    monkeypatch.setattr(cfg,'MCP_SERVER_AUDIT_LOG_PATH',str(path),raising=False)
    return path


def test_direct_resource_call_is_audited_without_body_or_email_id(monkeypatch, tmp_path):
    path=audit_file(monkeypatch,tmp_path)
    monkeypatch.setattr(mcp_server,'get_email',lambda _: {'body':'PRIVATE BODY'})
    assert 'PRIVATE BODY' in mcp_server.read_email_resource('PRIVATE-ID')
    row=json.loads(path.read_text(encoding='utf-8'))
    assert row['event']=='resource_read' and row['status']=='success' and row['boundary']=='server'
    assert 'PRIVATE' not in path.read_text(encoding='utf-8')


def test_failed_resource_keeps_exception_and_records_safe_error(monkeypatch,tmp_path):
    path=audit_file(monkeypatch,tmp_path)
    monkeypatch.setattr(mcp_server,'get_email',lambda _: (_ for _ in ()).throw(ValueError('PRIVATE')))
    with pytest.raises(ValueError):
        mcp_server.read_email_resource('PRIVATE')
    row=json.loads(path.read_text(encoding='utf-8'))
    assert row['status']=='error' and 'PRIVATE' not in str(row)


def test_audit_write_failure_cannot_replace_completed_resource(monkeypatch,tmp_path):
    path=audit_file(monkeypatch,tmp_path)
    path.mkdir()
    monkeypatch.setattr(mcp_server,'email_stats',lambda:{'count':3})
    assert json.loads(mcp_server.read_email_corpus_stats_resource())=={'count':3}


def test_direct_tool_result_contains_server_correlation(monkeypatch,tmp_path):
    from agents import tools
    path=audit_file(monkeypatch,tmp_path)
    monkeypatch.setattr(tools,'call_tool',lambda *a:{'status':'success','data':{'count':3},'_tool_result':1})
    wrapper=mcp_server._trusted_tool('email_stats',tools.email_stats)
    result=wrapper()
    row=json.loads(path.read_text(encoding='utf-8'))
    assert result['server_request_id']==row['request_id']
    assert row['target']=='email_stats' and row['run_id']
