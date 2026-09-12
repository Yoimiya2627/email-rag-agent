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
    assert approved.json()["reviewer"] == "local"


def test_approval_api_uses_configured_mail_provider(monkeypatch, tmp_path):
    import api.main as api_mod
    import config.settings as cfg
    from agents.approvals import ApprovalStore

    class FakeProvider:
        def execute_approval(self, approval):
            return {
                "mode": "gmail_draft",
                "provider": "gmail",
                "draft_id": "draft-123",
                "sent": False,
                "subject": approval["payload"]["subject"],
            }

    monkeypatch.setattr(cfg, "APPROVAL_STORE_PATH", str(tmp_path / "approvals.json"), raising=False)
    monkeypatch.setattr(api_mod, "create_mail_provider_from_settings", lambda: FakeProvider(), raising=False)
    pending = ApprovalStore().create(
        action_type="send_email",
        payload={"to": ["alice@example.com"], "subject": "Hi", "body": "Hello"},
    )

    client = TestClient(api_mod.app)
    approved = client.post(
        f"/agent/approvals/{pending['approval_id']}/approve",
        json={"reviewer": "tester", "note": "ok"},
    )

    assert approved.status_code == 200
    assert approved.json()["result"]["mode"] == "gmail_draft"
    assert approved.json()["result"]["draft_id"] == "draft-123"


def test_approval_api_provider_error_marks_item_unknown(monkeypatch, tmp_path):
    import api.main as api_mod
    import config.settings as cfg
    from agents.approvals import ApprovalStore
    from agents.mail_providers import MailProviderError

    class FailingProvider:
        def execute_approval(self, approval):
            raise MailProviderError("gmail unavailable")

    monkeypatch.setattr(cfg, "APPROVAL_STORE_PATH", str(tmp_path / "approvals.json"), raising=False)
    monkeypatch.setattr(api_mod, "create_mail_provider_from_settings", lambda: FailingProvider(), raising=False)
    pending = ApprovalStore().create(
        action_type="send_email",
        payload={"to": ["alice@example.com"], "subject": "Hi", "body": "Hello"},
    )

    client = TestClient(api_mod.app)
    approved = client.post(
        f"/agent/approvals/{pending['approval_id']}/approve",
        json={"reviewer": "tester", "note": "ok"},
    )

    assert approved.status_code == 502
    assert ApprovalStore().get(pending["approval_id"])["status"] == "unknown"
