from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from scripts.gmail_phase2a_preflight import (
    GmailPhase2AConfig,
    check_phase2a_readiness,
)


READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"


def _config(tmp_path: Path, scopes: list[str] | None = None) -> GmailPhase2AConfig:
    return GmailPhase2AConfig(
        credentials_path=tmp_path / "credentials" / "gmail_credentials.json",
        readonly_token_path=tmp_path / "credentials" / "gmail_readonly_token.json",
        readonly_scopes=scopes if scopes is not None else [READONLY_SCOPE],
        sync_output_path=tmp_path / "data" / "real_emails" / "gmail_emails.json",
        sync_state_path=tmp_path / "data" / "mail_sync" / "gmail_sync_state.json",
        gold_template_path=tmp_path / "data" / "real_emails" / "gold_chunks.real.json",
    )


def test_preflight_reports_missing_credentials_token_and_real_corpus(tmp_path: Path):
    report = check_phase2a_readiness(_config(tmp_path))

    assert report.ready_for_interactive_sync is False
    assert report.ready_for_noninteractive_sync is False
    assert report.ready_for_real_eval is False
    assert [item.code for item in report.blockers] == [
        "gmail_credentials_missing",
        "gmail_readonly_token_missing",
        "real_email_corpus_missing",
        "gmail_sync_state_missing",
        "real_gold_template_missing",
    ]


def test_preflight_accepts_ready_local_artifacts(tmp_path: Path):
    config = _config(tmp_path)
    for path in (
        config.credentials_path,
        config.readonly_token_path,
        config.sync_output_path,
        config.sync_state_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    config.gold_template_path.parent.mkdir(parents=True, exist_ok=True)
    config.gold_template_path.write_text(
        json.dumps(
            [
                {
                    "question": "What is the renewal budget status?",
                    "ground_truth": "The renewal budget is approved.",
                    "gold_chunk_ids": ["gmail_m1_chunk_0"],
                }
            ]
        ),
        encoding="utf-8",
    )

    report = check_phase2a_readiness(config)

    assert report.ready_for_interactive_sync is True
    assert report.ready_for_noninteractive_sync is True
    assert report.ready_for_real_eval is True
    assert report.blockers == []
    assert report.warnings == []


def test_preflight_blocks_unlabeled_real_gold_template(tmp_path: Path):
    config = _config(tmp_path)
    for path in (
        config.credentials_path,
        config.readonly_token_path,
        config.sync_output_path,
        config.sync_state_path,
        config.gold_template_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    config.gold_template_path.write_text(
        json.dumps(
            [
                {
                    "question": "",
                    "ground_truth": "",
                    "gold_chunk_ids": ["gmail_m1_chunk_0"],
                }
            ]
        ),
        encoding="utf-8",
    )

    report = check_phase2a_readiness(config)

    assert "real_gold_template_unlabeled" in [item.code for item in report.blockers]
    assert report.ready_for_real_eval is False


def test_preflight_blocks_non_readonly_gmail_scope(tmp_path: Path):
    config = _config(
        tmp_path,
        scopes=["https://www.googleapis.com/auth/gmail.compose"],
    )
    config.credentials_path.parent.mkdir(parents=True)
    config.credentials_path.write_text("{}", encoding="utf-8")

    report = check_phase2a_readiness(config)

    assert "gmail_readonly_scope_missing" in [item.code for item in report.blockers]
    assert "gmail_read_scope_too_broad" in [item.code for item in report.warnings]
    assert report.ready_for_interactive_sync is False


def test_preflight_cli_returns_nonzero_when_blocked(tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/gmail_phase2a_preflight.py",
            "--credentials",
            str(tmp_path / "credentials" / "gmail_credentials.json"),
            "--readonly-token",
            str(tmp_path / "credentials" / "gmail_readonly_token.json"),
            "--sync-output",
            str(tmp_path / "data" / "real_emails" / "gmail_emails.json"),
            "--sync-state",
            str(tmp_path / "data" / "mail_sync" / "gmail_sync_state.json"),
            "--gold-template",
            str(tmp_path / "data" / "real_emails" / "gold_chunks.real.json"),
            "--scope",
            READONLY_SCOPE,
            "--json",
        ],
        cwd=Path(__file__).resolve().parent.parent,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload["ready_for_noninteractive_sync"] is False
    assert "gmail_credentials_missing" in [
        item["code"] for item in payload["blockers"]
    ]
