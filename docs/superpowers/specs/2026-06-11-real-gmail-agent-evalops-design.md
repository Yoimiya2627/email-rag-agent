# Real Gmail Agent EvalOps Design

## Goal

Add a reproducible, local-only Agent EvalOps path for synced real Gmail data, so the project can separately report synthetic full Agent EvalOps, real Gmail retrieval recall, and real Gmail agent behavior.

## Scope

The first real Gmail agent gate is read-only and draft-safe. It uses existing `data/real_emails/gold_chunks.real.json` labels to generate task cases, then runs the existing agent loop with a custom testset path. It does not send real email and treats `send_email` as forbidden.

## Components

- `scripts/build_real_agent_testset.py` builds `data/real_emails/agent_testset.real.json` from labeled real gold cases.
- The builder filters out fragmentary, boilerplate, hidden-filler, and duplicate-source gold cases before creating agent tasks.
- `scripts/run_agent_eval.py` accepts `--testset-path` so the same runner can evaluate synthetic or real tasksets.
- Makefile and `tasks.ps1` expose:
  - `gmail-agent-testset`
  - `agent-eval-real`
  - `agent-eval-real-gate`
- `.gitignore` keeps real eval JSON/Markdown reports out of Git.

## Task Shape

The generated testset contains two initial variants:

- `real_retrieval`: answer the gold question from Gmail evidence using `search_emails`.
- `real_detail_lookup`: locate the source Gmail message, read details, then answer the gold question using `search_emails -> get_email`.

Each case preserves source metadata (`source_gold_id`, `source_email_ids`, `gold_chunk_ids`, `source_subject`, `source_sender`) for auditability. The success criteria include the existing ground truth and require email evidence only.

To keep the first real agent gate meaningful, generated tasks default to one gold case per source email. This prevents one long newsletter with many low-context chunks from dominating a 30-case run.

## Gates

The real gate is intentionally lower volume than synthetic full eval at first:

- `min_tasks=30`
- `min_task_success_rate=0.75`
- `min_tool_accuracy=0.80`
- `max_forbidden_tool_violation_rate=0.00`
- `max_max_steps_reached_rate=0.05`

These thresholds are strict enough to detect broken behavior while acknowledging the first real Gmail taskset is small and based on local private data.

## Operational Notes

Before running `agent-eval-real`, the Chroma index must point at synced real Gmail data, normally via `gmail-sync-index` or `scripts/index_emails.py --data-path data/real_emails/gmail_emails.json --clear`. This is separate from the synthetic 105-case full eval, which requires the default `data/emails.json` index.
