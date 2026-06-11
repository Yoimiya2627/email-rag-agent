# Real Email Recall Optimization Design

## Goal

Expand the real Gmail evaluation set from a 9-case proof to a 30-50 case validation set, and improve retrieval recall toward 0.9+ without exposing real mailbox content in Git or chat output.

## Current Evidence

- Gmail OAuth, read-only sync, indexing, real gold labels, and real context recall have run end to end.
- Current real eval has 9 cases.
- V2 and V7 both scored `mean_context_recall=0.7778`.
- Misses show realistic retrieval ambiguity across security notices, writing-help mail, and general ChatGPT onboarding mail.

## Requirements

1. Preserve already labeled real gold cases when regenerating templates.
2. Allow the real gold template builder to append new cases from newly synced mail.
3. Avoid selecting subject-only chunks for gold labels.
4. Ensure each indexed email chunk carries enough email header context to retrieve by subject, sender, date, and labels.
5. Expand Gmail sync to a larger read-only sample, targeting up to 100 synced messages.
6. Produce a 30-50 case real gold set in ignored local data.
7. Run `context-recall-real` and report only metrics and miss ids, not private email text.
8. Keep credentials, real emails, real gold labels, and real eval outputs ignored by Git and Docker.

## Non-Goals

- Do not publish real mailbox data, OAuth credentials, tokens, or real eval JSON.
- Do not send real email.
- Do not claim enterprise production readiness.
- Do not introduce an external labeling service that sends private email content to another model provider.

## Design

### Gold Template Merge

`scripts/build_real_gmail_gold_template.py` will load an existing output file when present, preserve labeled `question` and `ground_truth` values, and merge them onto regenerated cases. The primary merge key is `source_email_ids`; this matches the current real-eval workflow where we use one selected chunk per email. Generated chunk ids and previews remain fresh, so chunking improvements can update the evidence target while preserving the human-authored label.

### Header-Aware Chunking

`core.chunker.chunk_email()` will chunk the cleaned body first, then prefix every body chunk with a stable email header:

```text
Subject: ...
From: ...
To: ...
Date: ...
Labels: ...
Thread-ID: ...

Body:
...
```

This removes subject-only first chunks for long emails and gives later chunks enough retrieval context.

### Real Data Expansion

The sync command will run with a wider Gmail query, for example `newer_than:365d`, and `--max-results 100`. The project will then rebuild the index, regenerate/merge a 50-case gold template, label new cases locally, and run real context recall.

### Privacy Boundary

All real artifacts remain under ignored paths:

- `credentials/`
- `data/real_emails/`
- `data/mail_sync/`
- `data/eval_results/*.real*.json`

Final reports may include counts, metric values, and case ids, but not private email body text.

## Acceptance Criteria

- Unit tests cover label-preserving merge behavior.
- Unit tests cover header-aware chunk content on every chunk.
- `.\tasks.ps1 verify` passes.
- `git check-ignore` confirms real Gmail credentials, corpus, gold labels, sync state, and real eval output are ignored.
- Real Gmail sync/index runs with up to 100 messages.
- Real gold set contains at least 30 labeled cases if enough real mail is available.
- `context-recall-real` produces a fresh V2 top5 report.
