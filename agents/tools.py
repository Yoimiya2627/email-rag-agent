"""Email domain tools with one validated dispatch boundary."""
from __future__ import annotations

import logging
import sqlite3
from typing import Any, List

from core.pipeline import retrieve
from core.embedder import get_indexed_email, get_corpus_revision, get_all_metadata, index_snapshot
from core.evidence import table_context_note, evidence_reference, with_visible_reference, source_coverage
from core.evidence_pages import (SEARCH_PAGES, read_email_page, read_thread_evidence,
                                 compute_scoped_stats, EvidenceVersionMismatch, EvidenceCursorError)
from core.model_clients import ModelBudgetExceeded
from agents.runtime import RunCancelled
from core.model_outcomes import text_from_response
from agents.summarizer_agent import SummarizerAgent
from agents.writer_agent import WriterAgent, draft_reply_for_email
from agents.analyzer_agent import compute_email_stats
from agents.approvals import ApprovalStore
from agents.execution_scope import current_execution_scope
from agents.runtime import (add_evidence, content_digest, current_run, normalize_tool_result,
                            remaining_timeout, tool_error, validate_schema, RunDeadlineExceeded, ContextBudgetExceeded)
from models.schemas import AgentRequest, SearchResult
import config.settings as cfg

logger = logging.getLogger(__name__)


def _format_hit(result: SearchResult) -> dict:
    metadata = result.metadata
    original = {"email_id": result.email_id, "chunk_id": result.chunk_id,
                "content": result.content, "metadata": metadata}
    ref = evidence_reference(original)
    return with_visible_reference({"email_id": result.email_id, "chunk_id": result.chunk_id,
            "subject": str(metadata.get("subject", ""))[:300], "sender": str(metadata.get("sender", ""))[:200],
            "date": str(metadata.get("date", ""))[:80], "snippet": result.content[:300],
            "score": round(float(result.score), 4), "coverage": source_coverage(metadata), **ref,
            "snippet_truncated": len(result.content) > 300,
            **({"table_context": table_context_note(metadata, excerpt_truncated=len(result.content) > 300)}
               if metadata.get("table_context") else {})})


@index_snapshot()
def search_emails(query: str, sender: str = "", date_hint: str = "",
                  labels: List[str] | None = None, limit: int | None = None,
                  cursor: str | None = None) -> dict:
    """Page one fixed ranked candidate snapshot, explicitly not the whole mailbox."""
    limit = cfg.RERANK_TOP_N if limit is None else limit
    if type(limit) is not int or not 1 <= limit <= 50:
        raise ValueError("limit must be an integer between 1 and 50")
    filters = {"sender": sender, "date_hint": date_hint, "labels": labels or []}
    context = current_run()
    generation = get_corpus_revision()
    def fetch():
        horizon = int(getattr(cfg, "SEARCH_PAGE_CANDIDATES", 100))
        if not 1 <= horizon <= 500:
            raise ValueError("SEARCH_PAGE_CANDIDATES must be between 1 and 500")
        results = retrieve(query, filters=filters, top_n=horizon, fetch_k=horizon)
        selected, seen = [], set()
        for result in results:
            if result.email_id not in seen:
                seen.add(result.email_id)
                selected.append(_format_hit(result))
        return selected
    page = SEARCH_PAGES.page(query=query, filters=filters, generation=generation,
                             owner=context.owner_id if context else "local", limit=limit,
                             cursor=cursor, fetch=fetch)
    add_evidence(page["items"])
    return page


def get_email(email_id: str, chunk_id: str | None = None, start: int = 0, limit: int = 1200,
              source_version: str | None = None, source_sha256: str | None = None) -> dict:
    """Read a bounded indexed page; follow next_start with the same version."""
    with index_snapshot():
        email = read_email_page(email_id, chunk_id=chunk_id, start=start, limit=limit,
                                source_version=source_version, source_sha256=source_sha256,
                                loader=get_indexed_email)
    add_evidence(email.get("chunks", []))
    return email


def get_thread(thread_id: str, start: int = 0, limit: int = 10, source_version: str | None = None) -> dict:
    return read_thread_evidence(thread_id, start=start, limit=limit, source_version=source_version)


def summarize_emails(query: str) -> str:
    response = SummarizerAgent().run(AgentRequest(query=query))
    add_evidence(getattr(response, "sources", []))
    return text_from_response(response)


def draft_reply(instruction: str = "", email_id: str = "", query: str = "") -> Any:
    if email_id:
        email = get_email(email_id)
        if "error" in email:
            return email
        if email.get("has_more"):
            # Draft generation has its own context guard; never silently draft
            # against only the first page after the public reader became paged.
            email = get_indexed_email(email_id)
        return draft_reply_for_email(email, instruction)
    response = WriterAgent().run(AgentRequest(query=query or instruction), instruction=instruction or None)
    add_evidence(getattr(response, "sources", []))
    return text_from_response(response)


def send_email(to: List[str], subject: str, body: str, rationale: str) -> dict:
    """Create a versioned approval; never execute a provider from this tool."""
    payload = {"to": to, "subject": subject, "body": body, "rationale": rationale}
    # Direct Python callers receive the same checks as the model dispatcher.
    validate_schema(payload, TOOL_REGISTRY["send_email"].parameters)
    context = current_run()
    scope = current_execution_scope()
    owner = context.owner_id if context else "local"
    if scope and scope.owner_id != owner:
        raise PermissionError('approval owner does not match trusted execution scope')
    from agents.mail_providers import create_mail_provider_from_settings, SimulatedMailProvider
    # Evaluation must not inspect daily credentials even when the application
    # uses Gmail. The binding is server-owned, never supplied by model arguments.
    provider = SimulatedMailProvider() if scope and scope.evaluation else create_mail_provider_from_settings()
    payload['execution_binding'] = provider.approval_binding()
    request_id = scope.approval_request_id(content_digest(payload)) if scope else None
    if request_id is None:
        request_id = f"{context.run_id}:send:{content_digest(payload)}" if context else None
    store = ApprovalStore(path=scope.approval_store_path) if scope and scope.approval_store_path else ApprovalStore()
    item = store.create(
        action_type="send_email", payload=payload, requested_by="agent", risk_level="high",
        owner_id=owner,
        session_id=context.session_id if context else None, request_id=request_id,
    )
    state = item.get("status", "pending")
    if state in {"unknown", "executing"}:
        return tool_error("approval_outcome_unknown", "Existing approval requires outcome reconciliation.", unknown=True)
    if state in {"rejected", "expired", "failed"}:
        return tool_error("approval_not_pending", "The existing action is no longer pending; create a new request explicitly.")
    if state == "approved":
        return {"status": "approved", "approval_id": item["approval_id"],
                "result": item.get("result"), "message": "此请求已经执行，返回原有执行结果。"}
    return {"status": "pending_approval", "approval_id": item["approval_id"],
            "message": "已创建待审批的草稿请求，需要人工确认后才会执行；此工具不会发送邮件。"}


def email_stats(sender: str = "", date_hint: str = "", labels: List[str] | None = None,
                email_ids: List[str] | None = None) -> dict:
    with index_snapshot():
        return compute_scoped_stats(get_all_metadata(),
            filters={"sender": sender, "date_hint": date_hint, "labels": labels or []},
            email_ids=email_ids, generation=get_corpus_revision())


def _history_scope():
    from core.tool_results import ToolResultUnavailable, trusted_result_scope
    context = current_run()
    if context is None:
        scope = current_execution_scope()
        if scope is None or not scope.session_id or scope.session_repository is None:
            raise ToolResultUnavailable('trusted_scope_unavailable')
        from agents.runtime import RunContext
        context = RunContext(owner_id=scope.owner_id, session_id=scope.session_id,
                             session_repository=scope.session_repository, context_epoch=scope.context_epoch,
                             tool_result_store=scope.tool_result_store)
    trusted_result_scope(context)
    return context


def search_history(query: str, limit: int = 5) -> dict:
    """Read current-session excerpts; historical assistant claims stay unverified."""
    try:
        context = _history_scope()
        rows = context.session_repository.search_history(context.owner_id, context.session_id, query, limit=limit)
        _history_scope()  # recheck invalidation after read
        items = [{key: row[key] for key in ('turn_id', 'id', 'seq', 'hits', 'retrieval', 'metadata', 'include_in_context') if key in row} for row in rows]
        return {'items': items, 'material_type': 'historical_conversation',
                'coverage': 'bounded current-session excerpts; historical answers are not verified email evidence'}
    except (ValueError, KeyError, PermissionError, OSError, sqlite3.Error):
        return tool_error('history_unavailable', 'Trusted session history is unavailable or invalidated.')


def get_turn(turn_id: str, field: str = 'answer', offset: int = 0, limit: int = 1200) -> dict:
    try:
        context = _history_scope()
        page = {**context.session_repository.get_turn_page(context.owner_id, context.session_id, turn_id,
                field=field, offset=offset, limit=limit, expected_epoch=context.context_epoch),
                'material_type': 'historical_conversation', 'evidence_status': 'historical_claim_not_email_evidence'}
        _history_scope()
        return page
    except (ValueError, KeyError, PermissionError, OSError, sqlite3.Error):
        return tool_error('history_unavailable', 'Trusted session turn is unavailable or invalidated.')


def get_tool_result(result_id: str, start: int = 0, limit: int = 1200) -> dict:
    from core.tool_results import trusted_result_scope
    try:
        context = _history_scope()
        if context.tool_result_store is None:
            return tool_error('tool_result_unavailable', 'Tool result storage is unavailable.')
        # A run can page only refs issued by this trusted loop, never arbitrary IDs.
        ref = next((ref for ref in context.tool_result_refs.values() if ref['result_id'] == result_id), None)
        if ref is None:
            return tool_error('tool_result_unavailable', 'Tool result is unavailable in this run.')
        page = context.tool_result_store.page(result_id, start=start, limit=limit,
                 content_hash=ref['content_hash'], **trusted_result_scope(context))
        trusted_result_scope(context)
        return page
    except (ValueError, KeyError, PermissionError, OSError, sqlite3.Error):
        return tool_error('tool_result_unavailable', 'Tool result is missing, changed or invalidated.')


from agents.tool_registry import TOOL_REGISTRY, openai_tool_schemas, tool_dispatch

TOOL_SCHEMAS = openai_tool_schemas()
TOOL_DISPATCH = tool_dispatch()


def call_tool(name: str, arguments: Any) -> dict:
    """Validate before any invocation; return the shared local/MCP result envelope."""
    spec, function = TOOL_REGISTRY.get(name), TOOL_DISPATCH.get(name)
    if spec is None or function is None:
        return tool_error("unknown_tool", "Unknown tool.")
    try:
        validate_schema(arguments, spec.parameters)
    except (ValueError, TypeError, RecursionError):
        # Parameter values may contain mail bodies or malicious exception text.
        return tool_error("validation_error", "Tool arguments do not match the declared schema.")
    try:
        remaining_timeout(getattr(cfg, "LLM_TIMEOUT", 60))
        context = current_run()
        if context:
            context.tool_evidence.clear()
        result = normalize_tool_result(function(**arguments))
        if context and context.tool_evidence and result["status"] == "success":
            result["candidate_sources"] = list(context.tool_evidence.values())
        return result
    except (RunDeadlineExceeded, ContextBudgetExceeded, RunCancelled, ModelBudgetExceeded):
        raise
    except EvidenceVersionMismatch:
        return tool_error("source_version_changed", "Evidence changed; retrieve and inspect the current source explicitly.")
    except EvidenceCursorError:
        return tool_error("search_cursor_invalid", "Search cursor expired or scope changed; start a new search.")
    except TimeoutError:
        return tool_error("tool_timeout", "The tool timed out; check its outcome before retrying.",
                          unknown=spec.risk_level != "low")
    except Exception as exc:
        logger.warning("tool failed tool=%s error_type=%s", spec.name, type(exc).__name__)
        return tool_error("tool_failed", "The tool could not complete the request.",
                          unknown=spec.requires_approval)
