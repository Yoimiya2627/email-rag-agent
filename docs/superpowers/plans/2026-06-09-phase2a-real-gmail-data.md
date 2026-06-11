# Phase 2A Real Gmail Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a repeatable real Gmail data readiness, sync, index, and gold-template workflow without requiring secrets in tests.

**Architecture:** Keep Gmail network access in the existing sync script. Add a separate local-only preflight module that inspects configured paths and scopes, then expose it through Makefile and `tasks.ps1`.

**Tech Stack:** Python 3.11, pytest, argparse, JSON files, Gmail OAuth local artifacts, Makefile, PowerShell.

---

## File Structure

- Create `scripts/gmail_phase2a_preflight.py`: local-only readiness model and CLI.
- Create `scripts/build_real_gmail_gold_template.py`: real-mail gold chunk annotation template builder.
- Create `tests/test_gmail_phase2a_preflight.py`: TDD coverage for missing and ready Gmail artifacts.
- Create `tests/test_real_gmail_gold_template.py`: TDD coverage for real-mail template chunk IDs.
- Modify `Makefile`: add `gmail-preflight`, `gmail-sync-index`, `gmail-gold-template`, and `context-recall-real`.
- Modify `tasks.ps1`: add Windows mirrors for the same commands.
- Modify `.env.example`: document real-mail output paths and template path.
- Modify `README.md`: document the honest Phase 2A workflow.

### Task 1: Preflight Test API

**Files:**
- Create: `tests/test_gmail_phase2a_preflight.py`
- Create later: `scripts/gmail_phase2a_preflight.py`

- [x] **Step 1: Write the failing tests**

```python
from pathlib import Path

from scripts.gmail_phase2a_preflight import (
    GmailPhase2AConfig,
    check_phase2a_readiness,
)


READONLY_SCOPE = "https://www.googleapis.com/auth/gmail.readonly"


def test_preflight_reports_missing_credentials_token_and_real_corpus(tmp_path: Path):
    config = GmailPhase2AConfig(
        credentials_path=tmp_path / "credentials" / "gmail_credentials.json",
        readonly_token_path=tmp_path / "credentials" / "gmail_readonly_token.json",
        readonly_scopes=[READONLY_SCOPE],
        sync_output_path=tmp_path / "data" / "real_emails" / "gmail_emails.json",
        sync_state_path=tmp_path / "data" / "mail_sync" / "gmail_sync_state.json",
        gold_template_path=tmp_path / "data" / "real_emails" / "gold_chunks.real.json",
    )

    report = check_phase2a_readiness(config)

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
```

- [x] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_gmail_phase2a_preflight.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.gmail_phase2a_preflight'`.

- [x] **Step 3: Add the minimal public API**

Create `scripts/gmail_phase2a_preflight.py` with dataclasses `GmailPhase2AConfig`, `ReadinessIssue`, `ReadinessReport`, and function `check_phase2a_readiness(config)`.

- [x] **Step 4: Run test to verify it passes**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_gmail_phase2a_preflight.py -q`

Expected: PASS.

### Task 2: Scope and Ready-State Coverage

**Files:**
- Modify: `tests/test_gmail_phase2a_preflight.py`
- Modify: `scripts/gmail_phase2a_preflight.py`

- [x] **Step 1: Write the failing tests**

```python
def test_preflight_accepts_ready_local_artifacts(tmp_path: Path):
    credentials = tmp_path / "credentials" / "gmail_credentials.json"
    token = tmp_path / "credentials" / "gmail_readonly_token.json"
    corpus = tmp_path / "data" / "real_emails" / "gmail_emails.json"
    state = tmp_path / "data" / "mail_sync" / "gmail_sync_state.json"
    gold = tmp_path / "data" / "real_emails" / "gold_chunks.real.json"
    for path in (credentials, token, corpus, state, gold):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")

    report = check_phase2a_readiness(
        GmailPhase2AConfig(
            credentials_path=credentials,
            readonly_token_path=token,
            readonly_scopes=[READONLY_SCOPE],
            sync_output_path=corpus,
            sync_state_path=state,
            gold_template_path=gold,
        )
    )

    assert report.ready_for_interactive_sync is True
    assert report.ready_for_noninteractive_sync is True
    assert report.ready_for_real_eval is True
    assert report.blockers == []


def test_preflight_blocks_non_readonly_gmail_scope(tmp_path: Path):
    credentials = tmp_path / "credentials" / "gmail_credentials.json"
    credentials.parent.mkdir(parents=True)
    credentials.write_text("{}", encoding="utf-8")

    report = check_phase2a_readiness(
        GmailPhase2AConfig(
            credentials_path=credentials,
            readonly_token_path=tmp_path / "credentials" / "gmail_readonly_token.json",
            readonly_scopes=["https://www.googleapis.com/auth/gmail.compose"],
            sync_output_path=tmp_path / "data" / "real_emails" / "gmail_emails.json",
            sync_state_path=tmp_path / "data" / "mail_sync" / "gmail_sync_state.json",
            gold_template_path=tmp_path / "data" / "real_emails" / "gold_chunks.real.json",
        )
    )

    assert "gmail_readonly_scope_missing" in [item.code for item in report.blockers]
    assert "gmail_read_scope_too_broad" in [item.code for item in report.warnings]
```

- [x] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_gmail_phase2a_preflight.py -q`

Expected: FAIL because ready state and scope checks are not implemented yet.

- [x] **Step 3: Implement ready-state and scope checks**

Add path existence checks, exact read-only scope detection, broad Gmail scope warnings, and `to_dict()` serialization.

- [x] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_gmail_phase2a_preflight.py -q`

Expected: PASS.

### Task 3: Real-Mail Gold Template

**Files:**
- Create: `tests/test_real_gmail_gold_template.py`
- Create: `scripts/build_real_gmail_gold_template.py`

- [x] **Step 1: Write failing tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_real_gmail_gold_template.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.build_real_gmail_gold_template'`.

- [x] **Step 2: Implement template builder**

Create `build_real_mail_gold_template(emails, limit, chunks_per_email, preview_chars)` that cleans and chunks real synced `Email` objects and emits `real_gold_###` cases with real `gmail_*_chunk_*` IDs.

- [x] **Step 3: Run tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_real_gmail_gold_template.py -q`

Expected: PASS.

### Task 4: CLI and Command Wrappers

**Files:**
- Modify: `scripts/gmail_phase2a_preflight.py`
- Modify: `Makefile`
- Modify: `tasks.ps1`

- [x] **Step 1: Write CLI test**

Add a test that runs the script with explicit missing paths and asserts the process returns nonzero and prints JSON containing `gmail_credentials_missing`.

- [x] **Step 2: Run CLI test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_gmail_phase2a_preflight.py::test_preflight_cli_returns_nonzero_when_blocked -q`

Expected: FAIL because CLI flags are not implemented.

- [x] **Step 3: Implement CLI**

Add argparse flags for paths, scopes, `--json`, and return code `0` only when non-interactive sync is ready or `--allow-interactive` has credentials for OAuth.

- [x] **Step 4: Add command wrappers**

Add these targets:

```makefile
gmail-preflight:
	$(PYTHON) scripts/gmail_phase2a_preflight.py

gmail-sync-index:
	$(PYTHON) scripts/sync_gmail_readonly.py --index --clear-index

gmail-gold-template:
	$(PYTHON) scripts/build_real_gmail_gold_template.py

context-recall-real:
	$(PYTHON) scripts/evaluate_context_recall.py --gold data/real_emails/gold_chunks.real.json --output data/eval_results/context_recall.real.json
```

Add matching PowerShell functions and switch cases in `tasks.ps1`.

- [x] **Step 5: Run tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_gmail_phase2a_preflight.py -q`

Expected: PASS.

### Task 5: Documentation

**Files:**
- Modify: `.env.example`
- Modify: `README.md`

- [x] **Step 1: Update `.env.example`**

Add `GMAIL_REAL_GOLD_PATH=./data/real_emails/gold_chunks.real.json` next to the Gmail read-only variables.

- [x] **Step 2: Update README**

Document this order:

```powershell
.\tasks.ps1 gmail-preflight
.\tasks.ps1 gmail-sync-index
.\tasks.ps1 gmail-gold-template
.\tasks.ps1 context-recall-real
```

Explain that `gmail-preflight` must be green before claiming real Gmail data has been run.

### Task 6: Verification

**Files:**
- No production files changed in this task.

- [x] **Step 1: Run focused tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_gmail_phase2a_preflight.py -q`

Expected: PASS.

- [x] **Step 2: Run full project verification**

Run: `.\tasks.ps1 verify`

Expected: compile succeeds, all tests pass, and smoke EvalOps gate passes.

- [x] **Step 3: Run real-data preflight against this machine**

Run: `.\tasks.ps1 gmail-preflight`

Expected without local OAuth files: nonzero exit plus blockers for missing Gmail credentials or token. Expected with OAuth files: zero exit and ready states showing the project can run real Gmail sync.
