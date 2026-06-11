# Real Gmail 100 Gold Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the real Gmail deterministic context-recall gate from 50 labeled cases to 100 chunk-level cases without leaking real email content.

**Architecture:** Keep the existing Gmail sync/index and context recall scripts. Change the real gold template merge semantics from email-level preservation to chunk-level preservation so multiple chunks from the same email can be labeled independently. Generate a 100-case local gold file from ignored real Gmail data, label new cases locally, rerun the V2 recall@10 real gate, and document only aggregate metrics and case ids.

**Tech Stack:** Python 3.11, pytest, Gmail read-only local JSON, deterministic context recall.

---

## File Structure

- Modify `tests/test_real_gmail_gold_template.py`: cover chunk-id-based label preservation and prevent cross-chunk label reuse.
- Modify `scripts/build_real_gmail_gold_template.py`: preserve existing labels by `gold_chunk_ids` instead of `source_email_ids`.
- Modify ignored local data: `data/real_emails/gold_chunks.real.json` with 100 labeled cases.
- Modify ignored local report: `data/eval_results/context_recall.real.json`.
- Modify docs after verification: `README.md`, `docs/evaluation.md`, and this plan.

### Task 1: Chunk-Level Label Preservation

- [x] **Step 1: Write failing tests**

Add tests proving:

```python
def test_merge_existing_labels_preserves_by_chunk_id():
    generated = [{"gold_chunk_ids": ["gmail_m1_chunk_0"], "question": "", "ground_truth": ""}]
    existing = [{"gold_chunk_ids": ["gmail_m1_chunk_0"], "question": "Q?", "ground_truth": "A"}]
    merged, stats = merge_existing_gold_labels(generated, existing)
    assert merged[0]["question"] == "Q?"
    assert stats["preserved_labels"] == 1
```

and:

```python
def test_merge_existing_labels_does_not_reuse_labels_for_other_chunks_in_same_email():
    generated = [{"source_email_ids": ["gmail_m1"], "gold_chunk_ids": ["gmail_m1_chunk_1"], "question": "", "ground_truth": ""}]
    existing = [{"source_email_ids": ["gmail_m1"], "gold_chunk_ids": ["gmail_m1_chunk_0"], "question": "Q?", "ground_truth": "A"}]
    merged, stats = merge_existing_gold_labels(generated, existing)
    assert merged[0]["question"] == ""
    assert stats["new_items"] == 1
```

- [x] **Step 2: Run tests and confirm RED**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_real_gmail_gold_template.py -q`

Expected: the second test fails under email-level merge semantics.

- [x] **Step 3: Implement chunk-id preservation**

Add a `_chunk_key()` helper that returns a tuple of non-empty `gold_chunk_ids`, and make `merge_existing_gold_labels()` use that key for labeled existing items.

- [x] **Step 4: Run focused tests**

Run: `.\.venv\Scripts\python.exe -m pytest tests\test_real_gmail_gold_template.py -q`

Expected: all real gold template tests pass.

### Task 2: 100-Case Real Gold Data

- [x] **Step 1: Generate 100 chunk-level template**

Run: `.\.venv\Scripts\python.exe scripts\build_real_gmail_gold_template.py --limit 100 --chunks-per-email 3 --min-content-chars 160`

Expected: 100 generated cases if the existing real Gmail corpus has enough usable chunks; 50 existing labels are preserved by chunk id.

- [x] **Step 2: Label new cases locally**

Use local deterministic labeling from `chunk_preview` and metadata. Do not call an external labeling API and do not print email bodies.

- [x] **Step 3: Validate local counts**

Run a local JSON count script and confirm: `total=100`, `labeled=100`, and `unique_chunk_ids=100`.

### Task 3: Real Recall Gate and Docs

- [x] **Step 1: Run strict preflight**

Run: `.\.venv\Scripts\python.exe scripts\gmail_phase2a_preflight.py --require-real-eval`

Expected: `ready_for_real_eval: True`.

- [x] **Step 2: Run real recall gate**

Run: `.\tasks.ps1 context-recall-real`

Expected: V2 recall@10 metrics are written to ignored `data/eval_results/context_recall.real.json`.

- [x] **Step 3: Update docs**

Update public docs with only counts, metrics, and miss case ids. Do not include email content.

- [x] **Step 4: Full verification**

Run:

```powershell
.\tasks.ps1 verify
git check-ignore -v credentials\gmail_readonly_token.json data\real_emails\gold_chunks.real.json data\eval_results\context_recall.real.json
git diff --check
```

Expected: verification passes, private artifacts are ignored, and there are no whitespace errors beyond line-ending warnings.
