"""Tests for querying MCP audit events."""

import json

from fastapi.testclient import TestClient

from agents.mcp_adapter import MCPAuditLogger


def test_mcp_audit_logger_loads_filtered_events(tmp_path):
    path = tmp_path / "mcp_audit.jsonl"
    logger = MCPAuditLogger(path=path, enabled=True)
    logger.record(event="tool_call", tool="email_stats", status="success", request_id="r1")
    logger.record(event="tool_call", tool="send_email", status="error", request_id="r2")
    path.write_text(
        path.read_text(encoding="utf-8") + json.dumps({"bad": "row"}) + "\n",
        encoding="utf-8",
    )

    rows = MCPAuditLogger.load_events(path=path, tool="email_stats", status="success", limit=10)

    assert len(rows) == 1
    assert rows[0]["tool"] == "email_stats"
    assert rows[0]["status"] == "success"


def test_mcp_audit_api_filters_by_tool_status_and_limit(monkeypatch, tmp_path):
    import api.main as api_mod
    import config.settings as cfg

    path = tmp_path / "mcp_audit.jsonl"
    logger = MCPAuditLogger(path=path, enabled=True)
    logger.record(event="tool_call", tool="email_stats", status="success", request_id="r1")
    logger.record(event="tool_call", tool="email_stats", status="error", request_id="r2")
    logger.record(event="tool_call", tool="send_email", status="success", request_id="r3")
    monkeypatch.setattr(cfg, "MCP_AUDIT_LOG_PATH", str(path), raising=False)

    client = TestClient(api_mod.app)
    resp = client.get(
        "/agent/mcp-audit",
        params={"tool": "email_stats", "status": "success", "limit": 1},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert body["events"][0]["request_id"] == "r1"
