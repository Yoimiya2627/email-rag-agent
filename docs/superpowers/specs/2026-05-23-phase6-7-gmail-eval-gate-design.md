# Phase 6-7 Gmail Draft And Eval Gate Design

## Goal

Upgrade the agent from simulated high-risk email approval to a real Gmail draft integration, then strengthen Agent EvalOps with a CI-friendly gate and a 100+ case metadata testset.

## Scope

Phase 6 adds a mail provider boundary behind the existing approval workflow. The agent still cannot send email directly: `send_email` creates a pending approval, and only `/agent/approvals/{id}/approve` may execute the approved action. The default provider remains simulated; `MAIL_PROVIDER=gmail` creates a Gmail draft through the Gmail API and returns `draft_id` without sending.

Phase 7 expands `data/agent_testset.json` beyond 100 cases and adds an offline gate script that checks already-produced `agent_eval.json` metrics. The gate must not call LLMs in CI; it only reads result JSON and fails with a non-zero exit code when thresholds are not met.

## Non-Goals

- No default real sending.
- No QQ/163 provider in this phase.
- No OAuth multi-user tenant model; that belongs to a later MCP enterprise security phase.
- No changes to tracked interview docs. `docs/面经/` is local-only and ignored, but it may be updated locally for the user's preparation.

## Phase 6 Design

Create `agents/mail_providers.py` with:

- `MailProviderError`
- `SimulatedMailProvider`
- `GmailDraftProvider`
- `create_mail_provider_from_settings()`
- `build_rfc822_message()` / `build_gmail_raw_message()`

`ApprovalStore.approve()` gets an optional executor callback. If no executor is provided, it preserves the current simulated result. If an executor is provided, it receives the pending approval item and returns the result to persist. Executor failures must leave the approval pending.

`api/main.py` uses `create_mail_provider_from_settings()` inside the approve endpoint, so configured deployments can create real Gmail drafts while unit tests continue using mocks.

Settings:

- `MAIL_PROVIDER=simulated|gmail`
- `GMAIL_CREDENTIALS_PATH`
- `GMAIL_TOKEN_PATH`
- `GMAIL_SCOPES=https://www.googleapis.com/auth/gmail.compose`
- `GMAIL_USER_ID=me`
- `ENABLE_REAL_EMAIL_SEND=false`

## Phase 7 Design

Create `scripts/check_agent_eval_gate.py`:

- Reads `data/eval_results/agent_eval.json` by default.
- Supports thresholds:
  - `--min-task-success-rate`
  - `--min-tool-accuracy`
  - `--max-forbidden-tool-violation-rate`
  - `--max-max-steps-reached-rate`
  - `--min-tasks`
- Prints a short pass/fail report.
- Exits `0` on pass and `1` on fail.

Expand `data/agent_testset.json` to at least 100 cases while preserving the existing schema:

- `id`
- `task`
- `task_type`
- `risk_level`
- `expected_tools`
- `forbidden_tools`
- `success_criteria`

## Testing

Phase 6 selected tests:

- mail provider RFC822/raw encoding
- Gmail provider fake service creates a draft and never sends
- approval executor result is persisted
- approval executor failure leaves item pending
- API approve endpoint uses configured provider

Phase 7 selected tests:

- gate passes with healthy metrics
- gate fails with unhealthy metrics
- gate fails when `n_tasks` is below threshold
- testset has at least 100 metadata-rich cases

Full verification:

- `.\.venv\Scripts\python.exe -m pytest tests/ -q --basetemp=.pytest_tmp_final`
- CLI smoke for `scripts/check_agent_eval_gate.py`
- README scan for stale counts
