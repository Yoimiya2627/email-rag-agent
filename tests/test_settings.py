from __future__ import annotations

import importlib


def test_cors_origins_can_be_configured_from_env(monkeypatch):
    import config.settings as settings

    monkeypatch.setenv("CORS_ORIGINS", "https://mail.example.com, http://localhost:8501")
    reloaded = importlib.reload(settings)

    try:
        assert reloaded.CORS_ORIGINS == [
            "https://mail.example.com",
            "http://localhost:8501",
        ]
    finally:
        monkeypatch.delenv("CORS_ORIGINS", raising=False)
        importlib.reload(settings)
