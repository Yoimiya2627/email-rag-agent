# Real Email Recall Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the real Gmail eval set and improve retrieval recall by preserving labels and making chunks header-aware.

**Architecture:** Keep sync and eval commands intact. Add merge semantics to the real gold template builder, update the shared email chunker to prefix every body chunk with email headers, then rebuild the real mailbox index and rerun real context recall.

**Tech Stack:** Python 3.11, pytest, Gmail read-only API, ChromaDB, deterministic context recall.

---

## File Structure

- Modify `scripts/build_real_gmail_gold_template.py`: add existing-label preservation and merge stats.
- Modify `tests/test_real_gmail_gold_template.py`: add TDD coverage for preserving labels and replacing stale chunk ids.
- Modify `core/chunker.py`: chunk email body first, then prefix each chunk with email headers.
- Modify `tests/test_chunker.py`: add TDD coverage that every chunk contains email headers and no subject-only chunk is emitted for long bodies.
- Real local outputs under ignored paths: `data/real_emails/`, `data/mail_sync/`, `data/eval_results/*.real*.json`.

### Task 1: Preserve Existing Gold Labels

**Files:**
- Modify: `tests/test_real_gmail_gold_template.py`
- Modify: `scripts/build_real_gmail_gold_template.py`

- [x] **Step 1: Write failing merge test**

```python
def test_merge_existing_labels_preserves_question_and_ground_truth_on_same_email():
    generated = [{"source_email_ids": ["gmail_1"], "gold_chunk_ids": ["gmail_1_chunk_0"], "question": "", "ground_truth": ""}]
    existing = [{"source_email_ids": ["gmail_1"], "gold_chunk_ids": ["gmail_1_chunk_7"], "question": "Q?", "ground_truth": "A"}]
    merged, stats = merge_existing_gold_labels(generated, existing)
    assert merged[0]["question"] == "Q?"
    assert merged[0]["ground_truth"] == "A"
    assert merged[0]["gold_chunk_ids"] == ["gmail_1_chunk_0"]
    assert stats == {"generated": 1, "preserved_labels": 1, "new_items": 0}
```

- [x] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_real_gmail_gold_template.py::test_merge_existing_labels_preserves_question_and_ground_truth_on_same_email -q`

Expected: FAIL because `merge_existing_gold_labels` does not exist.

- [x] **Step 3: Implement merge helper and CLI preservation**

Add `merge_existing_gold_labels(generated, existing)` and make the CLI preserve existing labels by default when the output file exists.

- [x] **Step 4: Run tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_real_gmail_gold_template.py -q`

Expected: PASS.

### Task 2: Header-Aware Email Chunks

**Files:**
- Modify: `tests/test_chunker.py`
- Modify: `core/chunker.py`

- [x] **Step 1: Write failing chunker test**

```python
def test_chunk_email_prefixes_headers_on_every_long_body_chunk(make_email, monkeypatch):
    import config.settings as cfg
    monkeypatch.setattr(cfg, "CHUNK_SIZE", 120)
    monkeypatch.setattr(cfg, "CHUNK_OVERLAP", 20)
    monkeypatch.setattr(cfg, "MIN_CHUNK_SIZE", 20)
    email = make_email(id="gmail_1", subject="Security notice", sender="noreply@example.com", body="A" * 350)
    chunks = chunk_email(email)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.content.startswith("Subject: Security notice\nFrom: noreply@example.com")
        assert "\n\nBody:\n" in chunk.content
        assert len(chunk.content.split("Body:\n", 1)[1].strip()) > 0
```

- [x] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_chunker.py::test_chunk_email_prefixes_headers_on_every_long_body_chunk -q`

Expected: FAIL because only the first chunk currently carries subject context.

- [x] **Step 3: Implement header-aware chunking**

Change `chunk_email()` to split `email.body`, then prefix each part with `Subject`, `From`, `To`, `Date`, `Labels`, `Thread-ID`, and `Body`.

- [x] **Step 4: Run chunker tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_chunker.py -q`

Expected: PASS.

### Task 3: Real Sync Expansion and Eval

**Files:**
- Local ignored data only.

- [x] **Step 1: Run wider Gmail sync and rebuild index**

Run: `.\.venv\Scripts\python.exe scripts\sync_gmail_readonly.py --query "newer_than:365d" --max-results 100 --index --clear-index`

Expected: Sync reports fetched/added/skipped/total and index reports chunk counts.

- [x] **Step 2: Generate merged 50-case template**

Run: `.\.venv\Scripts\python.exe scripts\build_real_gmail_gold_template.py --limit 50 --chunks-per-email 1 --min-content-chars 160`

Expected: Existing labels are preserved and new cases are appended.

- [x] **Step 3: Label new cases locally**

Use local script logic to fill `question` and `ground_truth` for new cases using only each case's `chunk_preview` and metadata. Do not send content to an external labeling API.

- [x] **Step 4: Run real context recall**

Run: `.\tasks.ps1 context-recall-real`

Expected: Fresh V2 top5 metrics are written to ignored `data/eval_results/context_recall.real.json`.

### Task 4: Verification

**Files:**
- Modify: README test count if test total changes.

- [x] **Step 1: Strict preflight**

Run: `.\.venv\Scripts\python.exe scripts\gmail_phase2a_preflight.py --require-real-eval`

Expected: `ready_for_real_eval: True`.

- [x] **Step 2: Full verify**

Run: `.\tasks.ps1 verify`

Expected: all tests pass and smoke EvalOps gate passes.

- [x] **Step 3: Privacy check**

Run: `git check-ignore -v credentials\gmail_readonly_token.json data\real_emails\gold_chunks.real.json data\eval_results\context_recall.real.json data\eval_results\context_recall.real.top5.json`

Expected: all paths are ignored.
