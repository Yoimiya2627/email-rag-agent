# Project Integrity Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the project easier to reproduce, verify, and package without changing the core Agent/RAG behavior.

**Architecture:** Add repository-level contracts for dependencies, CI gates, EvalOps gate modes, Docker build context, and compose runtime behavior. Keep existing application modules stable; cover the new repository contracts with a focused integrity test.

**Tech Stack:** Python 3.11, pytest, ruff configuration, GitHub Actions, Docker Compose, FastAPI, Streamlit.

---

### Task 1: Repository Integrity Test

**Files:**
- Create: `tests/test_project_integrity.py`

- [x] **Step 1: Write the failing test**

Add tests that assert the project has dependency metadata, CI workflow gates, smoke/full EvalOps shortcuts, local browser profile ignores, and a frontend compose service that does not install packages at runtime.

- [x] **Step 2: Run the test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_project_integrity.py -q`

Expected: FAIL because `pyproject.toml`, `.github/workflows/ci.yml`, task shortcuts, ignore patterns, and compose frontend build target are missing.

### Task 2: Dependency and Tooling Metadata

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.lock`
- Modify: `requirements.txt`

- [x] **Step 1: Add `pyproject.toml`**

Declare runtime dependencies, optional groups `dev`, `gmail`, and `eval`, pytest defaults, and ruff lint settings.

- [x] **Step 2: Add direct dependency lock**

Add `requirements.lock` with pinned direct dependencies from the current verified environment. Keep `requirements.txt` as the human-friendly compatibility file.

### Task 3: CI and EvalOps Gates

**Files:**
- Create: `.github/workflows/ci.yml`
- Modify: `Makefile`
- Modify: `tasks.ps1`

- [x] **Step 1: Add CI workflow**

Run install, `compileall`, full pytest, and a smoke EvalOps gate on every push/PR. Include a `workflow_dispatch` full gate job that uses the 100-task threshold.

- [x] **Step 2: Add task shortcuts**

Add `verify`, `agent-eval-smoke`, and `agent-eval-full` to both Unix and Windows task runners.

### Task 4: Packaging Hygiene

**Files:**
- Modify: `.gitignore`
- Modify: `.dockerignore`
- Modify: `Dockerfile`
- Modify: `docker-compose.yml`

- [x] **Step 1: Tighten ignore files**

Ignore browser profiles, netlogs, temporary ASCII/browser output, credentials, private workspace files, and logs in both Git and Docker build contexts.

- [x] **Step 2: Build frontend from the project image**

Add a Dockerfile `frontend` target and update compose so Streamlit runs from a built image instead of installing packages during container startup.

### Task 5: Verification

**Files:**
- No new files

- [x] **Step 1: Run focused integrity test**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_project_integrity.py -q`

Expected: PASS.

- [x] **Step 2: Run full regression**

Run: `.\.venv\Scripts\python.exe -m pytest tests/ -q`

Expected: all tests pass.

- [x] **Step 3: Run compile verification**

Run: `.\.venv\Scripts\python.exe -m compileall -q api agents core config frontend models scripts mcp_server.py`

Expected: exit code 0.

- [x] **Step 4: Run EvalOps smoke and full gate commands**

Run: `.\tasks.ps1 agent-eval-smoke`

Expected: PASS against the current 8-task smoke result.

Run: `.\tasks.ps1 agent-eval-full`

Expected: FAIL until a 100+ task `agent_eval.json` is produced; this validates the stricter gate remains strict.
