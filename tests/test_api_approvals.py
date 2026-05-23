"""FastAPI tests for human-in-the-loop approval endpoints."""

from fastapi.testclient import TestClient


def test_approval_endpoints_list_and_approve_pending_action(monkeypatch, tmp_path):
    import api.main as api_mod
    import config.settings as cfg
    from agents.approvals import ApprovalStore

    monkeypatch.setattr(cfg, "APPROVAL_STORE_PATH", str(tmp_path / "approvals.json"), raising=False)
    pending = ApprovalStore().create(
        action_type="send_email",
        payload={"to": ["alice@example.com"], "subject": "Hi", "body": "Hello"},
    )

    client = TestClient(api_mod.app)
    listed = client.get("/agent/approvals", params={"status": "pending"})
    approved = client.post(
        f"/agent/approvals/{pending['approval_id']}/approve",
        json={"reviewer": "tester", "note": "ok"},
    )

    assert listed.status_code == 200
    assert listed.json()["approvals"][0]["approval_id"] == pending["approval_id"]
    assert approved.status_code == 200
    assert approved.json()["status"] == "approved"
    assert approved.json()["reviewer"] == "tester"
