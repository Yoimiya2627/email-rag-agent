"""Sync Gmail read-only messages into the local email JSON corpus.

The sync archives fetched source messages, checkpoints the corpus, and retries
isolated content failures from a query-scoped state file. Use
``--index`` when you want to immediately run the normal cleaner/chunker/embedder
pipeline after syncing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sqlite3
import sys
import tempfile
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import config.settings as cfg
from agents.gmail_readonly import (
    GmailReadOnlyProvider,
    MailContentError,
    gmail_message_to_email,
    message_internal_date_ms,
    is_missing_message_error,
    GmailPageTokenError,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
CONTENT_ERROR_CODES = frozenset({
    "invalid_body_encoding", "missing_related_root", "missing_body_data",
    "missing_message_id", "invalid_message_structure", "invalid_capture_format",
    "missing_captured_body_data", "mime_time_limit", "mime_part_limit", "mime_depth_limit",
    "mime_encoded_limit", "mime_decoded_limit", "mime_output_limit", "mime_header_limit", "invalid_mime_part", "capture_integrity_failed",
})


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, payload: Any) -> None:
    """Replace a complete JSON document only after its bytes reach the disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as f:
            temporary = Path(f.name)
            json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
        temporary = None
        # POSIX also requires the renamed directory entry to be flushed. Windows
        # does not support opening directories with os.open for this operation.
        if os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _canonical_path(path: str | Path) -> Path:
    return Path(os.path.normcase(str(Path(path).resolve())))


@contextmanager
def _sync_locks(*paths: Path):
    """Serialize read/merge/write across processes sharing either output or state.

    SQLite supplies cross-platform OS locks, released even if a process exits.
    Keep the small sidecars: removing a lock file could split concurrent writers
    across different file identities. A busy writer fails after 30 seconds.
    """
    with ExitStack() as stack:
        for path in sorted(set(paths), key=str):
            path.parent.mkdir(parents=True, exist_ok=True)
            lock_path = path.with_name(path.name + ".sync-lock.sqlite3")
            connection = sqlite3.connect(lock_path, timeout=30, isolation_level=None)
            stack.callback(connection.close)
            connection.execute("BEGIN IMMEDIATE")
        yield


def _raw_message_id(email_id: str) -> str:
    return email_id.removeprefix("gmail_")


def sync_gmail_to_json(
    provider: GmailReadOnlyProvider,
    output_path: str | Path,
    state_path: str | Path,
    query: str,
    max_results: int,
    checkpoint_size: int = 25,
    raw_dir: str | Path | None = None,
    reconcile: bool = False,
) -> dict[str, Any]:
    output = _canonical_path(output_path)
    state_file = _canonical_path(state_path)
    archive = _canonical_path(raw_dir if raw_dir is not None else str(output) + ".raw")
    if isinstance(checkpoint_size, bool) or not isinstance(checkpoint_size, int) or checkpoint_size < 1:
        raise ValueError("checkpoint_size must be positive")
    if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results < 1:
        raise ValueError("max_results must be a positive integer")
    # Include lock sidecars in collision checks before opening SQLite or writing.
    files = [output, state_file]
    files += [path.with_name(path.name + ".sync-lock.sqlite3")
              for path in (output, state_file, archive)]
    if len(set(files)) != len(files) or any(
        path == archive or archive in path.parents or path in archive.parents
        for path in files
    ) or any(a in b.parents or b in a.parents
             for i, a in enumerate(files) for b in files[i + 1:]):
        raise ValueError("Gmail corpus, state, raw directory and lock paths must be separate")
    with _sync_locks(output, state_file, archive):
        return _sync_gmail_locked(provider, output, state_file, query, max_results,
                                  checkpoint_size, archive, reconcile)


def _sync_gmail_locked(provider, output: Path, state_file: Path, query: str,
                       max_results: int, checkpoint_size: int, archive: Path, reconcile=False) -> dict[str, Any]:
    existing = _load_json(output, [])
    state = _load_json(state_file, {})
    if not isinstance(existing, list) or any(not isinstance(row, dict) for row in existing):
        raise ValueError("Gmail corpus must contain a list of objects")
    if not isinstance(state, dict):
        raise ValueError("Gmail sync state must contain an object")

    identity = provider.describe_account() if reconcile else state.get("identity")
    if reconcile:
        previous_accounts = {row.get("source", {}).get("account_id") for row in existing
                             if isinstance(row.get("source"), dict)} - {None, ""}
        previous_identity = state.get("identity") or {}
        if previous_identity.get("account_id"):
            previous_accounts.add(previous_identity["account_id"])
        if previous_accounts and previous_accounts != {identity["account_id"]}:
            raise ValueError("Gmail account differs from the corpus binding; use a separate corpus and state")
        if existing and not previous_accounts:
            raise ValueError("Existing corpus has no account binding; rebuild into a new output before reconciliation")

    # The corpus is authoritative. A stale state, a restored corpus or a new
    # --output must never cause absent messages to be treated as persisted.
    seen_order = []
    seen = set()
    existing_ids = {item.get("id", "") for item in existing}
    for item in existing:
        if str(item.get("id", "")).startswith("gmail_"):
            raw_id = _raw_message_id(str(item["id"]))
            if raw_id not in seen:
                seen.add(raw_id)
                seen_order.append(raw_id)

    added = 0
    skipped = 0
    same_output = state.get("corpus_path") == str(output)
    same_scope = same_output and state.get("query") == query
    missing = set(state.get("deleted_message_ids") or []) if same_scope else set()
    failures = dict(state.get("failed_messages") or {}) if same_scope else {}
    failures = {key: value for key, value in failures.items() if key not in seen and key not in missing}
    retry_ids = list(failures)
    if same_scope:
        retry_ids.extend(state.get("pending_message_ids") or [])
    pending = list(dict.fromkeys(mid for mid in retry_ids if mid not in seen and mid not in missing))
    metadata_pending = list(state.get("metadata_pending_ids") or []) if reconcile and same_scope else []
    last_internal = int(state.get("last_internal_date_ms") or 0) if same_output and existing else 0
    page_reader = getattr(provider, "list_message_page", None)
    if isinstance(provider, GmailReadOnlyProvider):
        # An existing subclass may intentionally provide its own finite-list
        # transport. Do not bypass it via the newly inherited page method.
        custom_list = ("list_message_ids" in vars(provider) or
                       type(provider).list_message_ids is not GmailReadOnlyProvider.list_message_ids)
        custom_page = ("list_message_page" in vars(provider) or
                       type(provider).list_message_page is not GmailReadOnlyProvider.list_message_page)
        if custom_list and not custom_page:
            page_reader = None
    pagination = dict(state.get("pagination") or {}) if same_scope and callable(page_reader) else {}
    if pagination.get("completed") or state.get("seen_message_ids", []) != seen_order:
        # A completed traversal starts a new polling cycle from the newest page.
        pagination = {}
    pagination.setdefault("next_page_token", None)
    pagination.setdefault("completed", False)
    pagination.setdefault("seen_page_tokens", [])
    pagination.setdefault("cursor_resets", 0)
    pagination.setdefault("scope_message_ids", [])
    if (not isinstance(pagination["seen_page_tokens"], list)
            or any(not isinstance(token, str) for token in pagination["seen_page_tokens"])
            or type(pagination["cursor_resets"]) is not int or pagination["cursor_resets"] < 0):
        raise ValueError("Invalid Gmail pagination state")

    def checkpoint(status: str) -> None:
        # Corpus commits first: a state failure can always be repaired from it.
        _dump_json(output, existing)
        _dump_json(state_file, {
            "corpus_path": str(output),
            "identity": identity, "sync_mode": "reconcile" if reconcile else "append_only",
            "scope_consistency": "best_effort_paginated", "metadata_pending_ids": metadata_pending,
            "metadata_complete": bool(reconcile and pagination.get("completed") and not pending and not metadata_pending and not failures),
            "raw_dir": str(archive),
            "seen_message_ids": seen_order,
            "last_internal_date_ms": last_internal,
            "last_sync_at": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "status": status,
            "failed_messages": failures,
            "failed_message_ids": list(failures),
            "failed": len(failures),
            "pending_message_ids": pending,
            "deleted_message_ids": sorted(missing),
            "pagination": pagination if callable(page_reader) else None,
            "backfill_complete": bool(callable(page_reader) and pagination.get("completed")
                                      and not pending and not failures),
        })

    def archive_message(path: Path, envelope: dict[str, Any]) -> None:
        try:
            _dump_json(path, envelope)
        except Exception:
            # A failed archive volume need not prevent committing prior messages
            # to a separate corpus volume. The current ID remains pending.
            checkpoint("aborted")
            raise

    def select_capture(root_file, capture_file, message_id):
        from core.capture_selection import selection_record
        try:
            record = selection_record(archive, root_file, capture_file, message_id)
        except Exception:
            checkpoint('aborted')
            raise
        archive_message(archive / 'selections' / root_file.name, record)

    try:
        if not callable(page_reader):
            # Compatibility for older custom providers. A finite list does not
            # certify complete mailbox coverage; state records that explicitly.
            message_ids = provider.list_message_ids(query=query, max_results=max_results)
        else:
            message_ids = []
            listed = set()
            page_size = int(getattr(cfg, "GMAIL_SYNC_PAGE_SIZE", 100))
            if not 1 <= page_size <= 500:
                raise ValueError("GMAIL_SYNC_PAGE_SIZE must be between 1 and 500")
            page_budget = max(1, int(getattr(cfg, "GMAIL_SYNC_MAX_PAGES_PER_RUN", 1000)))
            reset_limit = max(0, int(getattr(cfg, "GMAIL_SYNC_CURSOR_RESET_LIMIT", 1)))
            for _ in range(page_budget):
                token = pagination["next_page_token"]
                try:
                    page = page_reader(query=query, page_token=token,
                                       page_size=min(page_size, max_results - len(message_ids)))
                except GmailPageTokenError:
                    if pagination["cursor_resets"] >= reset_limit:
                        raise GmailPageTokenError("Gmail cursor recovery budget exhausted; restart backfill with a new state file") from None
                    pagination["cursor_resets"] += 1
                    pagination["next_page_token"] = None
                    pagination["seen_page_tokens"] = []
                    checkpoint("backfilling")
                    continue
                ids = page.get("message_ids")
                next_token = page.get("next_page_token")
                if (not isinstance(ids, list) or len(ids) > min(page_size, max_results - len(message_ids))
                        or any(not isinstance(mid, str) or not mid for mid in ids)):
                    raise ValueError("Invalid Gmail continuation page")
                if next_token is not None and (not isinstance(next_token, str) or not next_token
                        or next_token == token or next_token in pagination["seen_page_tokens"]):
                    raise ValueError("Gmail pagination did not advance")
                for mid in ids:
                    if reconcile and mid in seen and mid not in metadata_pending:
                        metadata_pending.append(mid)
                    if reconcile and mid not in pagination["scope_message_ids"]:
                        pagination["scope_message_ids"].append(mid)
                    if mid not in listed:
                        listed.add(mid)
                        message_ids.append(mid)
                    if mid not in seen and mid not in missing and mid not in pending:
                        pending.append(mid)
                if token is not None:
                    pagination["seen_page_tokens"].append(token)
                pagination["next_page_token"] = next_token
                pagination["completed"] = next_token is None
                # The cursor and every newly listed pending ID commit together.
                # A crash must not advance past an unrecorded work queue.
                checkpoint("backfilling")
                if next_token is None or len(message_ids) >= max_results:
                    break
            else:
                # Keep the committed queue/cursor for a later bounded run.
                logger.info("Gmail page budget reached; continuation retained")
    except Exception:
        checkpoint("aborted")
        raise
    # Failed/interrupted IDs remain eligible even after falling outside the list
    # window. Retrying them first also prevents starvation in a busy mailbox.
    candidates = list(dict.fromkeys([*pending, *metadata_pending, *message_ids]))
    pending = [mid for mid in candidates if mid not in seen and mid not in missing]
    # Record the work queue before fetching: even a kill in the first batch can
    # be resumed when those IDs no longer appear in the latest list window.
    checkpoint("running")
    processed = 0
    for message_id in candidates:
        from agents.runtime import remaining_timeout
        remaining_timeout(60)
        local_id = f"gmail_{message_id}"
        if message_id in missing:
            skipped += 1
            continue
        if message_id in seen or local_id in existing_ids:
            if reconcile:
                try:
                    remote = provider.get_message_metadata(message_id)
                except Exception as exc:
                    if not is_missing_message_error(exc):
                        checkpoint("aborted")
                        raise
                    existing[:] = [row for row in existing if row.get("id") != local_id]
                    existing_ids.discard(local_id)
                    seen.discard(message_id)
                    seen_order[:] = [mid for mid in seen_order if mid != message_id]
                    missing.add(message_id)
                else:
                    for row in existing:
                        if row.get("id") == local_id:
                            row["labels"] = list(remote.get("labelIds") or [])
                            row["label_names"] = [getattr(provider, "label_map", {}).get(label, label) for label in row["labels"]]
                            row["thread_id"] = str(remote.get("threadId") or row.get("thread_id") or "")
                            row.setdefault("source", {})["metadata_refreshed_at"] = datetime.now(timezone.utc).isoformat()
                if message_id in metadata_pending:
                    metadata_pending.remove(message_id)
            skipped += 1
            continue

        try:
            raw_message = provider.get_message(message_id)
        except Exception as exc:
            if is_missing_message_error(exc):
                missing.add(message_id)
                failures.pop(message_id, None)
                pending.remove(message_id)
                processed += 1
                if processed % checkpoint_size == 0:
                    checkpoint("running")
                continue
            checkpoint("aborted")
            raise
        # The identifier cannot influence the archive path, even for fake or
        # malformed provider responses. Never put raw content in the state file.
        raw_path = archive / (hashlib.sha256(message_id.encode("utf-8")).hexdigest() + ".json")
        envelope = {"format": "gmail-full-v1", "message": raw_message,
                    "body_data": {}, "body_errors": {}}
        archive_message(raw_path, envelope)
        select_capture(raw_path, raw_path, message_id)
        capture = getattr(provider, "capture_message", None)
        if capture is not None:
            try:
                envelope = capture(raw_message)
            except Exception:
                checkpoint("aborted")
                raise
            # Content-addressed copies preserve each fetched original version; the
        # stable legacy filename remains a convenience pointer for old tooling.
        version_hash = hashlib.sha256(json.dumps(envelope, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        version_path = archive / "versions" / (version_hash + ".json")
        if not version_path.exists():
            archive_message(version_path, envelope)
        select_capture(raw_path, version_path, message_id)
        try:
            convert_capture = getattr(provider, "email_from_capture", None)
            convert = getattr(provider, "message_to_email", gmail_message_to_email)
            email = convert_capture(envelope) if convert_capture is not None else convert(raw_message)
            if email.id != local_id:
                raise ValueError("Gmail response id differs from requested id")
            if identity:
                email.source["account_id"] = identity["account_id"]
            normalized = email.model_dump()
            internal_date = message_internal_date_ms(raw_message)
        except (MailContentError, ValueError, TypeError, KeyError, UnicodeError, OverflowError) as exc:
            code = getattr(exc, "code", None) if isinstance(exc, MailContentError) else None
            failures[message_id] = {"code": code if code in CONTENT_ERROR_CODES else "conversion_failed",
                                    "raw_file": raw_path.name}
            pending.remove(message_id)
            processed += 1
            if processed % checkpoint_size == 0:
                checkpoint("running")
            continue
        except Exception:
            checkpoint("aborted")
            raise
        existing.append(normalized)
        existing_ids.add(email.id)
        seen.add(message_id)
        seen_order.append(message_id)
        last_internal = max(last_internal, internal_date)
        failures.pop(message_id, None)
        pending.remove(message_id)
        added += 1
        processed += 1
        if processed % checkpoint_size == 0:
            checkpoint("running")

    if reconcile and callable(page_reader) and pagination["completed"] and not pending and not metadata_pending and not failures:
        in_scope = set(pagination["scope_message_ids"])
        # Remove out-of-scope records only after a complete traversal. Interrupted
        # or bounded pages never prove absence. Raw captures remain retained.
        existing[:] = [row for row in existing if not str(row.get("id", "")).startswith("gmail_")
                       or _raw_message_id(row["id"]) in in_scope]
        seen_order[:] = [mid for mid in seen_order if mid in in_scope]
    checkpoint("partial" if failures else ("backfilling" if callable(page_reader)
               and not pagination["completed"] else "complete"))
    return {
        "fetched": len(message_ids),
        "added": added,
        "skipped": skipped,
        "total": len(existing),
        "failed": len(failures),
        "failed_ids": list(failures),
    }


def index_email_json(data_path: str | Path, clear: bool = False, *, force_reembed: bool = False) -> dict[str, Any]:
    from core.embedder import get_collection_stats, index_chunks
    from scripts.index_emails import prepare_email_chunks
    from core.index_metrics import collect_index_metrics

    with collect_index_metrics() as report:
        emails, chunks = prepare_email_chunks(data_path)
        try:
            email_count = len(emails)
            indexed = index_chunks(chunks, replace=clear,
                                   **({'force_reembed':True} if force_reembed else {}))
        finally:
            close = getattr(emails,'close',None)
            if close:
                close()
    if report.outcome != 'unchanged':
        from core.retriever import invalidate_bm25_cache
        invalidate_bm25_cache()
    stats = get_collection_stats()
    return {'emails':email_count,'indexed_chunks':indexed,
            'collection_chunks':int(stats.get('chunk_count',0)), 'index_metrics':report.to_dict()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync Gmail read-only messages into local JSON")
    parser.add_argument("--output", default=cfg.GMAIL_SYNC_OUTPUT_PATH)
    parser.add_argument("--state-path", default=cfg.GMAIL_SYNC_STATE_PATH)
    parser.add_argument("--query", default=cfg.GMAIL_SYNC_QUERY)
    parser.add_argument("--max-results", type=int, default=cfg.GMAIL_SYNC_MAX_RESULTS)
    parser.add_argument("--checkpoint-size", type=int, default=25)
    parser.add_argument("--raw-dir", default=None, help="Raw archive directory (default: OUTPUT.raw)")
    parser.add_argument("--append-only", action="store_true", help="Legacy unbound ingestion; does not reconcile known metadata or removals")
    parser.add_argument("--index", action="store_true", help="Index synced JSON after writing it")
    parser.add_argument("--clear-index", action="store_true", help="Replace the corpus after successful indexing")
    parser.add_argument('--force-reembed', action='store_true', help='With --index, explicitly re-encode supplied chunks')
    args = parser.parse_args()
    if args.force_reembed and not args.index:
        parser.error('--force-reembed requires --index')

    try:
        provider = GmailReadOnlyProvider()
        result = sync_gmail_to_json(
            provider=provider,
            output_path=args.output,
            state_path=args.state_path,
            query=args.query,
            max_results=args.max_results,
            checkpoint_size=args.checkpoint_size,
            raw_dir=args.raw_dir,
            reconcile=not args.append_only,
        )
    except Exception:
        # Provider diagnostics may contain message content or credential data.
        print(json.dumps({"sync": {"status": "aborted"},
                          "status_path": str(_canonical_path(args.state_path))},
                         ensure_ascii=False, indent=2))
        logger.error("Sync aborted; inspect the status file (if written) before retrying")
        raise SystemExit(1) from None
    print(json.dumps({"sync": result, "status_path": str(_canonical_path(args.state_path))},
                     ensure_ascii=False, indent=2))

    if result["failed"]:
        logger.error("Sync is partial; inspect the status file before indexing")
        raise SystemExit(1)
    if args.index:
        if not args.append_only and result["total"] == 0:
            from core.embedder import clear_collection
            from core.index_metrics import collect_index_metrics
            with collect_index_metrics() as report:
                clear_collection()
            if report.outcome != 'unchanged':
                from core.retriever import invalidate_bm25_cache
                invalidate_bm25_cache()
            index_result = {'emails':0,'indexed_chunks':0,'collection_chunks':0,'index_metrics':report.to_dict()}
        else:
            # Reconciliation removes rows from the authoritative JSON snapshot;
            # an upsert-only index would resurrect those removed emails.
            index_result = index_email_json(args.output, clear=args.clear_index or not args.append_only,
                **({'force_reembed':True} if args.force_reembed else {}))
        print(json.dumps({"index": index_result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
