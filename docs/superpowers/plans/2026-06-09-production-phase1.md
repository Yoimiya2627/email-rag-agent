# Production Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SQLite-backed approvals/session state plus basic auth, tenant boundaries, and rate limiting.

**Architecture:** Keep existing public APIs stable and insert storage/middleware layers underneath. `ApprovalStore` becomes backend-selecting while preserving existing methods; `core.session_store` owns session persistence; `api.main` derives tenant/client context and applies auth/rate limiting before protected handlers run.

**Tech Stack:** Python 3.11 stdlib `sqlite3`, FastAPI middleware, pytest/TestClient, existing project settings.

---

### Task 1: SQLite Approval Store

**Files:**
- Modify: `agents/approvals.py`
- Modify: `config/settings.py`
- Test: `tests/test_approvals.py`

- [x] **Step 1: Write failing SQLite approval tests**

Add tests that create `ApprovalStore(path, backend="sqlite", tenant_id="tenant-a")`, create/list/get/approve an approval, then open a new store pointing at the same DB and confirm the result persists. Add tenant isolation coverage.

- [x] **Step 2: Run tests and confirm RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_approvals.py -q`

Expected: fail because `ApprovalStore` does not accept `backend` or `tenant_id`.

- [x] **Step 3: Implement SQLite backend**

Use `sqlite3`, create an `approvals` table, serialize payload/result JSON, and filter all reads by `tenant_id`.

- [x] **Step 4: Run tests and confirm GREEN**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_approvals.py -q`

Expected: pass.

### Task 2: SQLite Session Store

**Files:**
- Create: `core/session_store.py`
- Modify: `api/main.py`
- Modify: `config/settings.py`
- Test: `tests/test_memory.py`

- [x] **Step 1: Write failing session store tests**

Add tests for `SQLiteSessionMemoryStore`: messages persist across store instances, are isolated by tenant/session, and trim to `max_turns * 2`.

- [x] **Step 2: Run tests and confirm RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_memory.py -q`

Expected: fail because `core.session_store` does not exist.

- [x] **Step 3: Implement session store and wire API**

Create `InMemorySessionMemoryStore`, `SQLiteSessionMemoryStore`, and `create_session_store_from_settings()`. Replace `_sessions` dict usage in `api.main` with a store object.

- [x] **Step 4: Run tests and confirm GREEN**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_memory.py -q`

Expected: pass.

### Task 3: API Auth, Tenant Context, and Rate Limit

**Files:**
- Modify: `api/main.py`
- Modify: `config/settings.py`
- Test: `tests/test_api_security.py`
- Test: `tests/test_api_approvals.py`

- [x] **Step 1: Write failing API security tests**

Add tests that set `API_AUTH_TOKEN`, assert protected endpoints return 401 without bearer token, pass with token, approvals are scoped by `X-Tenant-ID`, and rate limit returns 429 after the configured threshold.

- [x] **Step 2: Run tests and confirm RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_api_security.py tests\test_api_approvals.py -q`

Expected: fail because middleware and tenant scoping are not implemented.

- [x] **Step 3: Implement middleware and tenant helpers**

Add public-path skip list, bearer-token check, request-scoped tenant id, fixed-window limiter, and tenant-aware `ApprovalStore`/session store calls.

- [x] **Step 4: Run tests and confirm GREEN**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_api_security.py tests\test_api_approvals.py -q`

Expected: pass.

### Task 4: Docs and Verification

**Files:**
- Modify: `.env.example`
- Modify: `README.md`
- Modify: `docs/architecture.md`

- [x] **Step 1: Document production Phase 1 settings**

Add configuration notes for SQLite state, auth, tenant id, and rate limiting.

- [x] **Step 2: Run focused tests**

Run approval, memory, and API security tests.

- [x] **Step 3: Run full verification**

Run: `.\tasks.ps1 verify`

Expected: all tests pass and smoke EvalOps gate passes.
