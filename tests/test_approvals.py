"""Tests for human-in-the-loop approval workflows."""

import pytest

from agents.approvals import ApprovalStore


def test_approval_store_creates_and_approves_pending_action(tmp_path):
    store = ApprovalStore(tmp_path / "approvals.json")

    item = store.create(
        action_type="send_email",
        payload={"to": ["alice@example.com"], "subject": "Budget", "body": "确认收到"},
        requested_by="agent",
        risk_level="high",
    )

    assert item["status"] == "pending"
    assert item["approval_id"]
    assert store.get(item["approval_id"])["payload"]["subject"] == "Budget"

    approved = store.approve(item["approval_id"], reviewer="human", note="looks good")

    assert approved["status"] == "approved"
    assert approved["reviewer"] == "human"
    assert approved["result"]["mode"] == "simulated_send"


def test_approval_store_persists_executor_result(tmp_path):
    store = ApprovalStore(tmp_path / "approvals.json")
    item = store.create(
        action_type="send_email",
        payload={"to": ["alice@example.com"], "subject": "Budget", "body": "确认收到"},
    )

    approved = store.approve(
        item["approval_id"],
        reviewer="human",
        note="ok",
        executor=lambda approval: {
            "mode": "gmail_draft",
            "provider": "gmail",
            "draft_id": "draft-123",
            "sent": False,
            "subject": approval["payload"]["subject"],
        },
    )

    assert approved["status"] == "approved"
    assert approved["result"]["mode"] == "gmail_draft"
    assert approved["result"]["draft_id"] == "draft-123"
    assert approved["result"]["subject"] == "Budget"


def test_approval_executor_failure_marks_unknown_and_blocks_replay(tmp_path):
    store = ApprovalStore(tmp_path / "approvals.json")
    item = store.create(action_type="send_email", payload={"to": ["alice@example.com"]})

    def fail(_approval):
        raise RuntimeError("gmail unavailable")

    with pytest.raises(RuntimeError, match="gmail unavailable"):
        store.approve(item["approval_id"], executor=fail)

    assert store.get(item["approval_id"])["status"] == "unknown"
    with pytest.raises(ValueError, match="unknown"):
        ApprovalStore(store.path).approve(item["approval_id"], executor=fail)


def test_send_email_tool_creates_pending_approval(monkeypatch, tmp_path):
    import agents.tools as tools_mod
    import config.settings as cfg

    monkeypatch.setattr(cfg, "APPROVAL_STORE_PATH", str(tmp_path / "approvals.json"), raising=False)

    out = tools_mod.send_email(
        to=["alice@example.com"],
        subject="Budget",
        body="确认参会",
        rationale="用户要求发送确认邮件",
    )

    assert out["status"] == "pending_approval"
    assert out["approval_id"]
    pending = ApprovalStore(cfg.APPROVAL_STORE_PATH).get(out["approval_id"])
    assert pending["action_type"] == "send_email"
    assert pending["payload"]["subject"] == "Budget"
