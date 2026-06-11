# Full Agent EvalOps Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 8-task smoke Agent EvalOps result with a 100+ task full LLM agent evaluation and a gateable report.

**Architecture:** Reuse the existing `data/agent_testset.json` 105-case task set and `scripts/run_agent_eval.py`. Run the full evaluation into the default `data/eval_results/agent_eval.json`, generate a Markdown report under `data/eval_results/agent_eval_report.md`, then run the strict `agent-eval-full` gate. Update public docs only after fresh verification.

**Tech Stack:** Python 3.11, DeepSeek-compatible OpenAI client, existing agent loop, EvalOps gate.

---

### Task 1: Full Eval Execution

- [ ] **Step 1: Confirm testset size**

Run a local JSON count and confirm `data/agent_testset.json` contains at least 100 cases.

- [ ] **Step 2: Run full agent eval**

Run:

```powershell
.\.venv\Scripts\python.exe scripts\run_agent_eval.py --output data/eval_results/agent_eval.json --report-output data/eval_results/agent_eval_report.md
```

Expected: the summary reports at least 100 tasks and writes JSON plus Markdown report.

- [ ] **Step 3: Run strict full gate**

Run:

```powershell
.\tasks.ps1 agent-eval-full
```

Expected: strict full gate passes or reports concrete failing metrics.

### Task 2: Verification and Docs

- [ ] **Step 1: Update README/docs with actual full eval metrics**

Only document metrics from the fresh full run.

- [ ] **Step 2: Run full verification**

Run:

```powershell
.\tasks.ps1 verify
git diff --check
```

Expected: compile, tests, smoke gate, and whitespace check pass.
