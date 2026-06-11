# Production Phase 1 Design

## Goal

Reduce the remaining "local demo only" weak points by adding durable SQLite-backed state, basic API protection, tenant-aware boundaries, and rate limiting while preserving the existing local demo flow.

## Approved Scope

- Add SQLite-backed approval storage with the same public `ApprovalStore` API.
- Add SQLite-backed session memory so multi-worker deployments can share chat history through a file-backed store.
- Keep JSON approval storage and in-memory session storage as explicit fallback backends for tests and demos.
- Add API bearer-token authentication when `API_AUTH_TOKEN` is configured.
- Add fixed-window rate limiting per tenant/client identity.
- Add tenant identification through `X-Tenant-ID`, defaulting to `default` for local demos.

## Architecture

Approval storage remains behind `agents.approvals.ApprovalStore`. The class will select a backend from configuration or constructor arguments and preserve `create/list/get/approve/reject` behavior. SQLite rows store JSON payloads/results as text and include `tenant_id` so pending approvals are isolated by tenant.

Session memory moves out of `api.main`'s module-level dict into `core.session_store`. The API gets a `SessionMemoryHandle` for each `(tenant_id, session_id)` pair. In-memory handles keep the existing behavior; SQLite handles persist ordered messages and trim to the same sliding window size.

API protection is middleware-level. Public paths such as `/health` remain open. When `API_AUTH_TOKEN` is non-empty, protected endpoints require `Authorization: Bearer <token>`. Rate limiting uses an in-process fixed window keyed by tenant and client identity; it is intentionally basic but testable and replaceable later.

## Configuration

- `APP_SQLITE_PATH=./data/app/app_state.sqlite3`
- `APPROVAL_STORE_BACKEND=sqlite`
- `SESSION_STORE_BACKEND=sqlite`
- `SESSION_MAX_TURNS=5`
- `API_AUTH_TOKEN=`
- `DEFAULT_TENANT_ID=default`
- `RATE_LIMIT_ENABLED=false`
- `RATE_LIMIT_REQUESTS=60`
- `RATE_LIMIT_WINDOW_SECONDS=60`

## Testing

Tests cover SQLite approval persistence, tenant isolation, executor failure staying pending, SQLite session persistence and trimming, auth required/accepted behavior, tenant-scoped approvals, and rate-limit rejection. Existing JSON approval and memory tests remain as fallback coverage.

## Out of Scope

- Redis/Postgres production drivers.
- Full user identity, OAuth, RBAC, or enterprise tenant provisioning.
- Real Gmail sync and 100+ real LLM full eval generation.
