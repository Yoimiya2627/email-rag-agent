"""API auth, tenant context, and rate-limit behavior."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_stays_public_when_api_auth_is_enabled(monkeypatch):
    import api.main as api_mod
    import config.settings as cfg

    monkeypatch.setattr(cfg, "API_AUTH_TOKEN", "secret-token", raising=False)

    resp = TestClient(api_mod.app).get("/health")

    assert resp.status_code == 200


def test_protected_endpoint_requires_bearer_token(monkeypatch, tmp_path):
    import api.main as api_mod
    import config.settings as cfg

    monkeypatch.setattr(cfg, "API_AUTH_TOKEN", "secret-token", raising=False)
    monkeypatch.setattr(cfg, "APPROVAL_STORE_BACKEND", "json", raising=False)
    monkeypatch.setattr(cfg, "APPROVAL_STORE_PATH", str(tmp_path / "approvals.json"), raising=False)

    client = TestClient(api_mod.app)

    assert client.get("/agent/approvals").status_code == 401
    assert client.get(
        "/agent/approvals",
        headers={"Authorization": "Bearer secret-token"},
    ).status_code == 200


def test_approval_api_is_scoped_by_tenant_header(monkeypatch, tmp_path):
    import api.main as api_mod
    import config.settings as cfg
    from agents.approvals import ApprovalStore

    db_path = tmp_path / "app_state.sqlite3"
    monkeypatch.setattr(cfg, "API_AUTH_TOKEN", "", raising=False)
    monkeypatch.setattr(cfg, "APPROVAL_STORE_BACKEND", "sqlite", raising=False)
    monkeypatch.setattr(cfg, "APP_SQLITE_PATH", str(db_path), raising=False)

    item_a = ApprovalStore(db_path, backend="sqlite", tenant_id="tenant-a").create(
        action_type="send_email",
        payload={"subject": "A"},
    )
    item_b = ApprovalStore(db_path, backend="sqlite", tenant_id="tenant-b").create(
        action_type="send_email",
        payload={"subject": "B"},
    )

    client = TestClient(api_mod.app)
    listed_a = client.get("/agent/approvals", headers={"X-Tenant-ID": "tenant-a"})
    listed_b = client.get("/agent/approvals", headers={"X-Tenant-ID": "tenant-b"})

    assert [item["approval_id"] for item in listed_a.json()["approvals"]] == [item_a["approval_id"]]
    assert [item["approval_id"] for item in listed_b.json()["approvals"]] == [item_b["approval_id"]]


def test_rate_limit_rejects_requests_after_threshold(monkeypatch, tmp_path):
    import api.main as api_mod
    import config.settings as cfg

    monkeypatch.setattr(cfg, "API_AUTH_TOKEN", "", raising=False)
    monkeypatch.setattr(cfg, "APPROVAL_STORE_BACKEND", "json", raising=False)
    monkeypatch.setattr(cfg, "APPROVAL_STORE_PATH", str(tmp_path / "approvals.json"), raising=False)
    monkeypatch.setattr(cfg, "RATE_LIMIT_ENABLED", True, raising=False)
    monkeypatch.setattr(cfg, "RATE_LIMIT_REQUESTS", 1, raising=False)
    monkeypatch.setattr(cfg, "RATE_LIMIT_WINDOW_SECONDS", 60, raising=False)
    api_mod._rate_limiter.clear()

    client = TestClient(api_mod.app)

    first = client.get("/agent/approvals", headers={"X-Tenant-ID": "rate-test"})
    second = client.get("/agent/approvals", headers={"X-Tenant-ID": "rate-test"})

    assert first.status_code == 200
    assert second.status_code == 429
