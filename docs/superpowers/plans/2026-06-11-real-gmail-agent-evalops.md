# Real Gmail Agent EvalOps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a real Gmail Agent EvalOps path from existing real-mail gold labels.

**Architecture:** Generate a private real-agent testset from `gold_chunks.real.json`, run the existing agent eval runner against a configurable testset path, and gate the ignored real eval result with lower-volume real-data thresholds.

**Tech Stack:** Python 3.11, existing agent loop, DeepSeek-compatible OpenAI client, PowerShell/Make task runners, pytest.

---

### Task 1: Real Agent Testset Generator

**Files:**
- Create: `scripts/build_real_agent_testset.py`
- Create: `tests/test_real_agent_testset.py`

- [x] Write failing tests for real gold to agent task conversion.
- [x] Implement labeled-case filtering and retrieval/detail task generation.
- [x] Add quality filtering for fragmentary/boilerplate real gold and one case per source email.
- [x] Add CLI defaults for `data/real_emails/gold_chunks.real.json` to `data/real_emails/agent_testset.real.json`.
- [x] Run `pytest tests/test_real_agent_testset.py -q --basetemp .pytest_tmp`.

### Task 2: Custom Testset Support In Agent Eval

**Files:**
- Modify: `scripts/run_agent_eval.py`
- Modify: `tests/test_agent_eval.py`

- [x] Write failing test for loading a custom testset path with a limit.
- [x] Add `--testset-path` and a `load_testset()` helper.
- [x] Ensure resume metadata sync uses the selected testset.
- [x] Run `pytest tests/test_agent_eval.py -q --basetemp .pytest_tmp`.

### Task 3: Tasks, Ignore Rules, And Docs

**Files:**
- Modify: `.gitignore`
- Modify: `.dockerignore`
- Modify: `Makefile`
- Modify: `tasks.ps1`
- Modify: `README.md`
- Modify: `docs/evaluation.md`
- Modify: `tests/test_project_integrity.py`

- [x] Add `gmail-agent-testset`, `agent-eval-real`, and `agent-eval-real-gate`.
- [x] Ignore real eval Markdown reports.
- [x] Document required index alignment and real gate thresholds.
- [x] Run project integrity tests.

### Task 4: Local Real Eval Attempt

**Files:**
- Ignored output: `data/real_emails/agent_testset.real.json`
- Ignored output: `data/eval_results/agent_eval.real.json`
- Ignored output: `data/eval_results/agent_eval.real_report.md`

- [x] Generate the real taskset from current gold labels.
- [x] If real Gmail JSON exists, rebuild Chroma against that JSON.
- [x] If API credentials are available, run a 30-case real agent eval.
- [x] Run `agent-eval-real-gate` and report PASS/FAIL with metrics.

### Task 5: Verification

- [x] Run `powershell.exe -ExecutionPolicy Bypass -File .\tasks.ps1 verify`.
- [x] Run `git diff --check`.
- [x] Commit the implementation.
