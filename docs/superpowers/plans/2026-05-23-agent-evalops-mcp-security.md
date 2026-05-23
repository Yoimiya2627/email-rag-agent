# Agent EvalOps And MCP Security Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the current trace/eval and MCP hardening into a small EvalOps + tool-security layer that is testable, documented, and safe for the existing `/chat/agent` path.

**Architecture:** Phase 4 adds pure helper modules for agent eval records, failure attribution, trace lookup, and report generation; `run_agent_eval.py` calls those helpers without changing the agent loop contract. Phase 5 adds a tool policy layer that filters MCP-visible tools and exposes audit-query helpers/API endpoints while keeping local function calling unchanged.

**Tech Stack:** Python, FastAPI, pytest, JSONL trace/audit logs, FastMCP, OpenAI-compatible function calling.

---

### Task 1: Phase 4 EvalOps Failure Attribution And Reports

**Files:**
- Create: `agents/evalops.py`
- Modify: `scripts/run_agent_eval.py`
- Test: `tests/test_agent_evalops.py`
- Test: `tests/test_agent_eval.py`

- [ ] **Step 1: Write failing tests**

Add tests that import `agents.evalops` and expect:

```python
def test_classify_eval_record_prefers_policy_violations():
    record = {
        "success": 0,
        "tool_accuracy": True,
        "forbidden_tool_violation": True,
        "max_steps_reached": False,
    }
    assert classify_eval_record(record, []) == "forbidden_tool"
```

Also test that Markdown report output includes summary metrics, trace ids, failure categories, and expected/actual tools.

- [ ] **Step 2: Verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agent_evalops.py -q
```

Expected: FAIL because `agents.evalops` does not exist.

- [ ] **Step 3: Implement minimal EvalOps helpers**

Create `agents/evalops.py` with:
- `events_by_trace_id(events)`
- `classify_eval_record(record, trace_events)`
- `build_eval_report(payload, trace_events=None)`
- `write_eval_report(payload, output_path, trace_events=None)`

Classification order: forbidden tool, max steps, missing expected tool, tool error, approval required, judge failed, success, task failed.

- [ ] **Step 4: Wire `run_agent_eval.py`**

Extend each eval record with:
- `id`
- `task_type`
- `risk_level`
- `success_criteria`
- `forbidden_tools`
- `forbidden_tool_violation`
- `failure_category`

Add CLI flags:
- `--report-output`
- `--trace-input`

- [ ] **Step 5: Verify Phase 4**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agent_evalops.py tests/test_agent_eval.py tests/test_agent_tracing.py -q
```

Expected: all selected tests pass.

### Task 2: Phase 4 Agent Testset Metadata

**Files:**
- Modify: `data/agent_testset.json`
- Test: `tests/test_agent_evalops.py`

- [ ] **Step 1: Write failing schema test**

Test that every case has `id`, `task`, `task_type`, `risk_level`, `expected_tools`, `forbidden_tools`, and `success_criteria`.

- [ ] **Step 2: Verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agent_evalops.py::test_agent_testset_has_evalops_metadata -q
```

Expected: FAIL on the existing 8-case schema.

- [ ] **Step 3: Expand the testset**

Update `data/agent_testset.json` to 50 metadata-rich tasks covering retrieval, detail lookup, summary, drafting, stats, high-risk send approval, forbidden-tool safety, ambiguous tasks, and boundary cases.

- [ ] **Step 4: Verify Phase 4 again**

Run the same Phase 4 test command from Task 1 Step 5.

### Task 3: Phase 5 MCP Tool Policy

**Files:**
- Create: `agents/tool_policy.py`
- Modify: `config/settings.py`
- Modify: `.env.example`
- Modify: `mcp_server.py`
- Test: `tests/test_mcp_policy.py`
- Test: `tests/test_mcp_server.py`

- [ ] **Step 1: Write failing policy tests**

Tests expect:
- read-only MCP mode hides `send_email` and medium/high-risk tools from registration
- explicit `MCP_ALLOWED_TOOLS=search_emails,email_stats` only exposes those tools
- unknown allowed tools are ignored, not fatal

- [ ] **Step 2: Verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_mcp_policy.py -q
```

Expected: FAIL because `agents.tool_policy` does not exist.

- [ ] **Step 3: Implement policy**

Create `ToolPolicy` with `visible_specs(registry)` and `from_settings()`. Add settings:
- `MCP_ALLOWED_TOOLS`
- `MCP_READ_ONLY_MODE`

Update `mcp_server.register_tools(server, policy=None)` and `build_server(..., policy=None)` to register only visible tools.

- [ ] **Step 4: Verify Phase 5 policy**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_mcp_policy.py tests/test_mcp_server.py tests/test_mcp_production.py -q
```

Expected: selected MCP tests pass.

### Task 4: Phase 5 Audit Query API

**Files:**
- Modify: `agents/mcp_adapter.py`
- Modify: `api/main.py`
- Test: `tests/test_mcp_audit_api.py`
- Test: `tests/test_mcp_production.py`

- [ ] **Step 1: Write failing audit query tests**

Tests expect a helper to load/filter audit JSONL by tool, status, and limit, and a FastAPI endpoint:

```text
GET /agent/mcp-audit?tool=email_stats&status=success&limit=20
```

- [ ] **Step 2: Verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_mcp_audit_api.py -q
```

Expected: FAIL because the endpoint/helper does not exist.

- [ ] **Step 3: Implement helper and endpoint**

Add `MCPAuditLogger.load_events(...)` and `GET /agent/mcp-audit`.

- [ ] **Step 4: Verify Phase 5**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_mcp_policy.py tests/test_mcp_audit_api.py tests/test_mcp_server.py tests/test_mcp_production.py -q
```

Expected: all selected Phase 5 tests pass.

### Task 5: Docs, 面经, Full Regression, Commit And Push

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture.md`
- Modify: `docs/面经/code_walkthrough_private.md`
- Modify: `docs/面经/interview_qa_walkthrough_v2_full.md`
- Modify: `docs/面经/resume_interview_question_bank.html`
- Regenerate: `docs/面经/code_walkthrough_private.html`
- Regenerate: `docs/面经/interview_qa_walkthrough_v2_full.html`

- [ ] **Step 1: Update docs**

Document Phase 4 EvalOps report/testset/failure attribution and Phase 5 MCP policy/audit API.

- [ ] **Step 2: Verify docs consistency**

Run:

```powershell
rg -n "107|8 条|首批 8|MCP 已有 token verifier" README.md docs\面经 docs\architecture.md
```

Expected: old counts or wording only appear where intentionally historic.

- [ ] **Step 3: Full regression**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q
```

Expected: all tests pass.

- [ ] **Step 4: Commit and push current branch**

Stage only task-related files, commit, and push `feature/agent-loop`.
