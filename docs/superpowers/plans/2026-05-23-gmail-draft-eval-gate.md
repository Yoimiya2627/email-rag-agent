# Gmail Draft And Eval Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Phase 6 Gmail draft-only approval execution and Phase 7 Agent EvalOps CI gate with a 100+ case metadata testset.

**Architecture:** Keep `send_email` as a high-risk approval creator. Add a provider executor behind `ApprovalStore.approve()` so the API can create simulated results or Gmail drafts after human approval. Add an offline eval gate script that reads existing eval JSON, plus expand the testset without requiring LLM calls in tests.

**Tech Stack:** Python, FastAPI, Gmail API optional dependencies, pytest, JSON metadata testsets.

---

### Task 1: Phase 6 Mail Provider Tests

**Files:**
- Create: `tests/test_mail_providers.py`
- Modify: `tests/test_approvals.py`
- Modify: `tests/test_api_approvals.py`

- [ ] Write failing tests for Gmail raw message encoding, fake Gmail draft creation, approval executor persistence, executor failure leaving approval pending, and API approve using a fake provider.
- [ ] Run `.\.venv\Scripts\python.exe -m pytest tests/test_mail_providers.py tests/test_approvals.py tests/test_api_approvals.py -q --basetemp=.pytest_tmp_phase6_red` and confirm failures are for missing provider/executor behavior.

### Task 2: Phase 6 Implementation

**Files:**
- Create: `agents/mail_providers.py`
- Modify: `agents/approvals.py`
- Modify: `api/main.py`
- Modify: `config/settings.py`
- Modify: `.env.example`
- Modify: `requirements.txt`

- [ ] Implement `MailProviderError`, `SimulatedMailProvider`, `GmailDraftProvider`, raw MIME helpers, and `create_mail_provider_from_settings()`.
- [ ] Add optional executor support to `ApprovalStore.approve()` while preserving default simulated behavior.
- [ ] Update the approval API endpoint to execute the configured provider after human approval.
- [ ] Add Gmail/provider configuration and optional Gmail dependencies.
- [ ] Run `.\.venv\Scripts\python.exe -m pytest tests/test_mail_providers.py tests/test_approvals.py tests/test_api_approvals.py -q --basetemp=.pytest_tmp_phase6_green`.

### Task 3: Phase 7 Gate Tests

**Files:**
- Create: `tests/test_agent_eval_gate.py`
- Modify: `tests/test_agent_evalops.py`

- [ ] Write failing tests for gate pass/fail/min-task behavior and for testset size >= 100.
- [ ] Run `.\.venv\Scripts\python.exe -m pytest tests/test_agent_eval_gate.py tests/test_agent_evalops.py -q --basetemp=.pytest_tmp_phase7_red` and confirm failures are for missing gate script and insufficient testset size.

### Task 4: Phase 7 Implementation

**Files:**
- Create: `scripts/check_agent_eval_gate.py`
- Modify: `data/agent_testset.json`

- [ ] Implement offline gate loading summary metrics from eval JSON and returning exit code `0` or `1`.
- [ ] Expand `data/agent_testset.json` to at least 100 metadata-rich tasks.
- [ ] Run `.\.venv\Scripts\python.exe -m pytest tests/test_agent_eval_gate.py tests/test_agent_evalops.py tests/test_agent_eval.py -q --basetemp=.pytest_tmp_phase7_green`.

### Task 5: Docs, Local Interview Notes, Full Regression, Commit And Push

**Files:**
- Modify: `README.md`
- Local-only ignored updates: `docs/面经/*`

- [ ] Update README with Phase 6/7 configuration, Gmail draft-only boundary, eval gate usage, new test count, and limitations.
- [ ] Update local ignored interview notes with Phase 6/7 talking points, without staging them.
- [ ] Run selected Phase 6 + Phase 7 tests together.
- [ ] Run full regression: `.\.venv\Scripts\python.exe -m pytest tests/ -q --basetemp=.pytest_tmp_final`.
- [ ] Run CLI smoke for `scripts/check_agent_eval_gate.py`.
- [ ] Stage only tracked project files and new spec/plan, excluding `docs/面经/` and unrelated local files.
- [ ] Commit and push current `feature/agent-loop` branch.
