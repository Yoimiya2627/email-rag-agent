# Phase 2A Real Gmail Data Design

## Goal

Turn the existing Gmail read-only ingestion into a repeatable real-data validation path. The project must be able to say, with evidence, whether real Gmail data can be synced, indexed, and evaluated, and if not, exactly which local credential or artifact is missing.

## Current State

- `agents/gmail_readonly.py` can read Gmail messages with `gmail.readonly` scope and convert them into the local `Email` schema.
- `scripts/sync_gmail_readonly.py` can append unseen Gmail messages to an ignored JSON corpus and optionally index it.
- `scripts/evaluate_context_recall.py` can generate gold chunk templates and run deterministic context recall over labeled `gold_chunk_ids`.
- Local secrets and real mailbox data are already ignored by `.gitignore` through `credentials/`, `data/real_emails/`, and `data/mail_sync/`.

## Problem

The project has real Gmail sync code, but the developer experience still blurs these states:

- no Gmail OAuth client file exists;
- an OAuth client exists but the read-only token has not been authorized;
- a token exists but the real-mail JSON corpus is still absent;
- real mail exists but has not been indexed;
- context recall is still running against synthetic labels.

That ambiguity makes the project easy to overclaim in interviews and hard to validate before demoing.

## Requirements

1. Add a preflight command that checks the real Gmail data path without contacting Gmail.
2. The preflight output must distinguish missing credentials, missing token, missing synced corpus, and missing sync state.
3. The preflight must verify the configured read scope is exactly read-only enough for ingestion and does not require compose/send scope.
4. The command must have JSON output so CI, local scripts, and future EvalOps gates can consume it.
5. Add a Windows `tasks.ps1` command and a Makefile target for the preflight.
6. Add a Windows `tasks.ps1` command and a Makefile target for sync plus indexing.
7. Add a Windows `tasks.ps1` command and a Makefile target for creating a real-mail gold chunk template.
8. Add a real-mail context recall command so synthetic and real gold files are not mixed.
9. Document the real-data workflow and the honest blocked state when OAuth artifacts are missing.

## Non-Goals

- Do not commit real Gmail credentials, OAuth tokens, mailbox data, or generated sync state.
- Do not require real Gmail credentials for unit tests or normal `verify`.
- Do not replace the existing synthetic evaluation dataset in this phase.
- Do not send email or request Gmail compose/send scopes in the ingestion path.

## Acceptance Criteria

- Unit tests cover the preflight readiness model.
- `.\tasks.ps1 verify` still passes without Gmail credentials.
- `.\tasks.ps1 gmail-preflight` exits nonzero with a useful report when credentials are missing.
- If `credentials/gmail_credentials.json` and `credentials/gmail_readonly_token.json` exist, the same preflight can return ready for non-interactive sync.
- `.\tasks.ps1 gmail-sync-index` is the one-command real mailbox sync and index path.
- `.\tasks.ps1 gmail-gold-template` creates a gold template under ignored real-mail data by default.
- `.\tasks.ps1 context-recall-real` evaluates against the real-mail gold template after manual labels are filled.
