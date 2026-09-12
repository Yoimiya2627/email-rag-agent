"""Versioned evidence reads and bounded search snapshots; no model inference."""
from __future__ import annotations

import copy
import hashlib
import hmac
import json
import secrets
import threading
import time
from collections import OrderedDict
from collections import Counter
from datetime import datetime, timezone

from core.evidence import evidence_reference, source_coverage, text_hash, with_visible_reference, table_context_note


class EvidenceVersionMismatch(ValueError):
    """The current source differs from the explicitly requested version."""


class EvidenceCursorError(ValueError):
    """A cursor expired or does not belong to this exact search snapshot."""


def _integer(value, name, minimum=0, maximum=100000000):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"invalid {name}")
    return value


def _identifier(value, name, *, optional=False):
    if optional and value is None:
        return
    if not isinstance(value, str) or not value or len(value) > 500:
        raise ValueError(f"invalid {name}")


def _metadata_view(meta):
    fields = ("subject", "sender", "sender_name", "date", "thread_id", "message_id", "in_reply_to")
    view = {name: str(meta.get(name, ""))[:300] for name in fields}
    for name in ("references", "cc"):
        value = meta.get(name, [])
        if isinstance(value, str):
            try:
                value = json.loads(value) if len(value) <= 10000 else []
            except ValueError:
                value = []
        view[name] = [item[:300] for item in value[:20] if isinstance(item, str)] if isinstance(value, list) else []
    return view


def _attachment_view(meta):
    items = meta.get("attachments", [])
    try:
        items = json.loads(items) if isinstance(items, str) and len(items) <= 20000 else items
    except ValueError:
        items = []
    if not isinstance(items, list):
        return []
    return [{key: str(item.get(key, ""))[:200] for key in ("filename", "mime_type", "status")}
            for item in items[:5] if isinstance(item, dict)]


def _read_email_page(email, *, chunk_id=None, start=0, limit=1200,
                     source_version=None, source_sha256=None):
    _integer(start, "start")
    _integer(limit, "limit", 1, 4000)
    _identifier(chunk_id, "chunk_id", optional=True)
    _identifier(source_version, "source_version", optional=True)
    _identifier(source_sha256, "source_sha256", optional=True)
    if "error" in email:
        return {"error": "email_id not found", "error_code": "evidence_not_found"}
    chunks = email.get("chunks") or []
    if chunk_id is not None:
        chunks = [row for row in chunks if row.get("chunk_id") == chunk_id]
        if not chunks:
            return {"error": "chunk_id not found for email_id", "error_code": "evidence_not_found"}
        text = chunks[0]["content"]
    else:
        text = email.get("body", "")
    if start > len(text):
        raise ValueError("start exceeds indexed text length")
    meta = chunks[0].get("metadata", {}) if chunks else email
    if chunk_id is not None:
        ref = evidence_reference(chunks[0])
        version, source_hash = ref["source_version"], ref["source_sha256"]
    else:
        source_hash = meta.get("source_sha256")
        version = meta.get("index_generation") or meta.get("source_version") or "indexed-body:" + text_hash(text)
    if (source_version is not None and source_version != version
            or source_sha256 is not None and source_sha256 != source_hash):
        raise EvidenceVersionMismatch("source version changed; retrieve and inspect the current source explicitly")
    end = min(start + limit, len(text))
    visible, approximate_offset = [], 0
    for row in chunks:
        original = row["content"]
        metadata = row.get("metadata") or {}
        if chunk_id is not None:
            row_start = 0
        elif email.get("reconstruction_exact") and type(metadata.get("source_start")) is int:
            row_start = metadata["source_start"]
        else:
            row_start = approximate_offset
            approximate_offset += len(original) + 2
        left, right = max(start, row_start), min(end, row_start + len(original))
        if right <= left:
            continue
        ref = evidence_reference(row)
        local_start = left - row_start
        piece = {"email_id": row["email_id"], "chunk_id": row["chunk_id"],
                 "content": original[local_start:right - row_start], "score": row.get("score", 0.0),
                 "metadata": _metadata_view(metadata), "coverage": source_coverage(metadata), **ref,
                 "visible_start": local_start, "page_start": left, "page_end": right}
        if metadata.get("table_context"):
            partial = local_start > 0 or right - row_start < len(original)
            piece["table_context"] = table_context_note(metadata, excerpt_truncated=partial)
            piece["table_excerpt_truncated"] = partial
        # Full body reads share the same document version; each chunk still
        # carries the actual chunk hash and local offsets needed for expansion.
        visible.append(with_visible_reference(piece))
    return {"page_kind": "email_text", "email_id": email["email_id"], "chunk_id": chunk_id,
            **_metadata_view(meta), "body": text[start:end], "body_source": "indexed_chunks",
            "body_format": email.get("body_format", "plain"),
            "source_version": version, "source_sha256": source_hash,
            "read_start": start, "read_end": end, "total_chars": len(text),
            "has_more": end < len(text), "next_start": end if end < len(text) else None,
            "chunks": visible, "coverage": source_coverage(meta),
            "attachment_inventory_preview": _attachment_view(meta),
            "reconstruction_exact": email.get("reconstruction_exact", False),
            "conclusion_status": "source_material_only_not_latest_valid_conclusion"}


def read_email_page(email_id, *, chunk_id=None, start=0, limit=1200,
                    source_version=None, source_sha256=None, loader=None):
    """Read exact indexed text; mismatched email/chunk/version is never replaced.

    ``loader`` is an explicit offline/test adapter. Default reads pin one index
    generation. Original MIME and unread attachments are outside this text view.
    """
    _identifier(email_id, "email_id")
    _integer(start, "start")
    _integer(limit, "limit", 1, 4000)
    _identifier(chunk_id, "chunk_id", optional=True)
    _identifier(source_version, "source_version", optional=True)
    _identifier(source_sha256, "source_sha256", optional=True)
    if loader is None:
        from core.embedder import get_indexed_email, index_snapshot
        with index_snapshot():
            email = get_indexed_email(email_id)
            return _read_email_page(email, chunk_id=chunk_id, start=start, limit=limit,
                                    source_version=source_version, source_sha256=source_sha256)
    email = loader(email_id)
    return _read_email_page(email, chunk_id=chunk_id, start=start, limit=limit,
                            source_version=source_version, source_sha256=source_sha256)


def reread_evidence(reference: dict, *, loader=None, start=None, limit=1200):
    """Verify a persisted visible range in bounded pieces; return a bounded preview.

    Verification and preview coverage are separate. At most the configured
    EVIDENCE_VERIFY_CHAR_LIMIT (default 60000) characters are hashed, using 4000
    character pages and one pinned source read. Larger refs fail explicitly.
    """
    if not isinstance(reference, dict):
        raise ValueError("evidence reference must be an object")
    visible_start, end = reference.get("visible_start"), reference.get("visible_end")
    _integer(visible_start, "visible_start")
    _integer(end, "visible_end", visible_start + 1)
    _integer(limit, "limit", 1, 4000)
    start = visible_start if start is None else _integer(start, "start", visible_start, end - 1)
    import config.settings as cfg
    maximum = int(getattr(cfg, "EVIDENCE_VERIFY_CHAR_LIMIT", 60000))
    _integer(maximum, "EVIDENCE_VERIFY_CHAR_LIMIT", 1, 1000000)
    if end - visible_start > maximum:
        raise ValueError("evidence verification range exceeds its explicit character budget")
    visible_hash = reference.get("visible_hash") or reference.get("visible_sha256")
    if not reference.get("source_version") or not visible_hash:
        raise EvidenceVersionMismatch("legacy reference has no verifiable version and visible hash")
    _identifier(reference.get("email_id"), "email_id")
    _identifier(reference.get("chunk_id"), "chunk_id")
    def verify(email):
        from agents.runtime import remaining_timeout
        expected_chunk_hash = reference.get("chunk_sha256")
        if expected_chunk_hash is not None:
            _identifier(expected_chunk_hash, "chunk_sha256")
            matching = [row for row in email.get("chunks", []) if row.get("chunk_id") == reference["chunk_id"]
                        and row.get("email_id") == reference["email_id"]]
            if len(matching) == 1 and len(matching[0]["content"]) > maximum:
                raise ValueError("full chunk hash verification exceeds its explicit character budget")
            if len(matching) != 1 or text_hash(matching[0]["content"]) != expected_chunk_hash:
                raise EvidenceVersionMismatch("the referenced chunk hash no longer matches its recorded version")
        hasher, offset = hashlib.sha256(), visible_start
        while offset < end:
            remaining_timeout(60)
            piece = _read_email_page(email, chunk_id=reference["chunk_id"], start=offset,
                limit=min(4000, end - offset), source_version=reference["source_version"],
                source_sha256=reference.get("source_sha256"))
            if "error" in piece or not piece["body"] or piece["read_end"] <= offset:
                raise EvidenceVersionMismatch("the referenced range no longer exists")
            hasher.update(piece["body"].encode("utf-8"))
            offset = piece["read_end"]
        if hasher.hexdigest() != visible_hash:
            raise EvidenceVersionMismatch("the referenced visible text no longer matches")
        page = _read_email_page(email, chunk_id=reference["chunk_id"], start=start,
            limit=min(limit, end - start), source_version=reference["source_version"],
            source_sha256=reference.get("source_sha256"))
        page.update(validation_status="version_and_visible_hash_match_not_entailment",
                    verification_complete=True, verified_start=visible_start, verified_end=end,
                    verified_hash=visible_hash, verification_char_limit=maximum,
                    chunk_hash_match=True if expected_chunk_hash is not None else None,
                    has_more=page["read_end"] < end,
                    next_start=page["read_end"] if page["read_end"] < end else None)
        return page
    if loader is None:
        from core.embedder import get_indexed_email, index_snapshot
        with index_snapshot():
            return verify(get_indexed_email(reference["email_id"]))
    return verify(loader(reference["email_id"]))


class SearchPages:
    """Bounded process cache of ranked candidates; cursors fail closed on restart.

    Exhausting this snapshot never claims complete semantic/mailbox coverage.
    Owner/query/filter/generation bindings are checked on every continuation.
    """
    def __init__(self, *, max_bytes=8 * 1024 * 1024, max_snapshots=64, ttl_seconds=600):
        _integer(max_bytes, "max_bytes", 1024)
        _integer(max_snapshots, "max_snapshots", 1, 1000)
        _integer(ttl_seconds, "ttl_seconds", 1, 86400)
        self.max_bytes, self.max_snapshots, self.ttl_seconds = max_bytes, max_snapshots, ttl_seconds
        self._secret = secrets.token_bytes(32)
        self._snapshots, self._lock = OrderedDict(), threading.RLock()

    def _cursor(self, token, offset):
        raw = f"{token}:{offset}"
        return raw + ":" + hmac.new(self._secret, raw.encode(), hashlib.sha256).hexdigest()

    def _decode(self, cursor):
        try:
            if not isinstance(cursor, str) or len(cursor) > 180:
                raise ValueError()
            token, offset, signature = cursor.split(":")
            if not hmac.compare_digest(self._cursor(token, int(offset)), cursor):
                raise ValueError()
            return token, _integer(int(offset), "cursor offset")
        except (ValueError, TypeError):
            raise EvidenceCursorError("invalid search cursor") from None

    def page(self, *, query, filters, generation, owner, limit, fetch, cursor=None):
        _integer(limit, "limit", 1, 50)
        binding = text_hash(json.dumps([owner, query, filters, generation], sort_keys=True, ensure_ascii=False))
        now = time.monotonic()
        if cursor is None:
            rows = fetch()
            token, offset = secrets.token_hex(16), 0
            size = len(json.dumps(rows, ensure_ascii=False).encode("utf-8"))
            if size > self.max_bytes:
                raise EvidenceCursorError("search snapshot exceeds its byte budget; narrow the search")
            item = {"binding": binding, "rows": rows, "created": now, "size": size}
        else:
            token, offset = self._decode(cursor)
            item = None
        with self._lock:
            for key, value in list(self._snapshots.items()):
                if now - value["created"] >= self.ttl_seconds:
                    self._snapshots.pop(key)
            if item is not None:
                self._snapshots[token] = item
                while (len(self._snapshots) > self.max_snapshots
                       or sum(value["size"] for value in self._snapshots.values()) > self.max_bytes):
                    self._snapshots.popitem(last=False)
            else:
                item = self._snapshots.get(token)
                if item is None or item["binding"] != binding:
                    raise EvidenceCursorError("search cursor expired or query/filter/owner/source version changed")
                self._snapshots.move_to_end(token)
            total = len(item["rows"])
            if offset > total:
                raise EvidenceCursorError("search cursor exceeds its snapshot")
            end = min(offset + limit, total)
            rows = copy.deepcopy(item["rows"][offset:end])
            for position, row in enumerate(rows, offset + 1):
                row["continuation_cursor"] = self._cursor(token, position) if position < total else None
        return {"page_kind": "search_candidates", "items": rows, "offset": offset,
                "resume_cursor": self._cursor(token, offset),
                "next_cursor": self._cursor(token, end) if end < total else None,
                "has_more": end < total, "remaining": total - end,
                "source_version": generation,
                "coverage": {"scope": "selected_ranked_candidates", "selected_total": total,
                             "returned": len(rows), "corpus_match_total": None,
                             "mailbox_complete": False, "semantic_coverage_complete": False,
                             "filters": filters}}


SEARCH_PAGES = SearchPages()


def read_thread_evidence(thread_id, *, start=0, limit=10, source_version=None, metadata_loader=None):
    """Chronological metadata for one indexed thread, without inferring validity."""
    _identifier(thread_id, "thread_id")
    _integer(start, "start")
    _integer(limit, "limit", 1, 50)
    if metadata_loader is None:
        from core.embedder import get_all_metadata, get_corpus_revision, index_snapshot
        with index_snapshot():
            rows, generation = get_all_metadata(), get_corpus_revision()
    else:
        rows, generation = metadata_loader()
    if source_version is not None and source_version != generation:
        raise EvidenceVersionMismatch("thread source version changed; inspect the current thread explicitly")
    selected = {}
    for row in rows:
        meta = row.get("metadata") or {}
        eid = meta.get("email_id")
        if meta.get("thread_id") == thread_id and isinstance(eid, str) and eid:
            selected.setdefault(eid, {"email_id": eid, "chunk_id": row["chunk_id"],
                                       **_metadata_view(meta), "source_version": generation,
                                       "source_sha256": meta.get("source_sha256"),
                                       "coverage": source_coverage(meta)})
    def date_key(row):
        try:
            value = datetime.fromisoformat(row["date"].replace("Z", "+00:00"))
            if value.tzinfo is None:
                return float("-inf")
            return value.timestamp()
        except (ValueError, TypeError):
            return float("-inf")
    ordered = sorted(selected.values(), key=lambda row: (date_key(row), row["email_id"]))
    if start > len(ordered):
        raise ValueError("start exceeds thread metadata length")
    message_ids = {row["message_id"] for row in ordered if row["message_id"]}
    for row in ordered:
        row["reply_parent_present_in_index"] = row["in_reply_to"] in message_ids if row["in_reply_to"] else None
        row["date_order_known"] = date_key(row) != float("-inf")
    end = min(start + limit, len(ordered))
    latest_date = date_key(ordered[-1]) if ordered else float("-inf")
    return {"page_kind": "thread_metadata", "thread_id": thread_id, "source_version": generation,
            "items": ordered[start:end], "read_start": start, "total_indexed_messages": len(ordered),
            "has_more": end < len(ordered), "next_start": end if end < len(ordered) else None,
            "latest_dated_indexed_ids": [row["email_id"] for row in ordered if date_key(row) == latest_date]
                                         if latest_date != float("-inf") else [],
            "coverage": "indexed_thread_only_mailbox_completeness_unknown",
            "conclusion_status": "latest_date_is_not_proof_of_current_validity_read_messages_and_conflicts"}


def compute_scoped_stats(rows, *, filters=None, email_ids=None, generation=None, now=None):
    """Complete deterministic statistics over the explicitly indexed metadata set.

    Natural-language semantic membership and business notions such as
    'unanswered' are not inferred. Display limits are separate from counted scope.
    """
    from core.filters import FilterSpec
    if now is None:
        from core.pipeline import _now
        now = _now()
    if not isinstance(now, datetime) or now.tzinfo is None:
        raise ValueError("statistics require a timezone-aware current time")
    stats_timezone = now.tzinfo
    filters = filters or {}
    if filters.get("query"):
        raise ValueError("semantic-query statistics require an explicit selected email ID set")
    spec = FilterSpec.from_mapping(filters, now=now)
    if email_ids is not None:
        if not isinstance(email_ids, list) or not 1 <= len(email_ids) <= 200:
            raise ValueError("email_ids must contain 1 to 200 identifiers")
        for email_id in email_ids:
            _identifier(email_id, "email_id")
    selected_ids = set(email_ids) if email_ids is not None else None
    selected, all_ids = {}, set()
    for row in rows:
        meta = row.get("metadata") or {}
        eid = meta.get("email_id")
        if not isinstance(eid, str) or not eid:
            continue
        all_ids.add(eid)
        if ((selected_ids is None or eid in selected_ids) and spec.matches(meta)):
            selected.setdefault(eid, meta)
    senders, labels, daily = Counter(), Counter(), Counter()
    unread, suspect, unknown_inventory, unknown_dates = 0, 0, 0, 0
    for meta in selected.values():
        senders[str(meta.get("sender", "unknown"))] += 1
        raw = meta.get("labels", [])
        try:
            raw = json.loads(raw) if isinstance(raw, str) else raw
        except (ValueError, TypeError):
            raw = []
        if isinstance(raw, list):
            labels.update(set(item for item in raw if isinstance(item, str)))
        date = meta.get("date", "")
        try:
            moment = datetime.fromisoformat(date.replace("Z", "+00:00"))
            if moment.tzinfo is None:
                first, second = moment.replace(tzinfo=stats_timezone, fold=0), moment.replace(tzinfo=stats_timezone, fold=1)
                if (first.utcoffset() != second.utcoffset() or
                        first.astimezone(timezone.utc).astimezone(stats_timezone).replace(tzinfo=None) != moment):
                    raise ValueError("ambiguous local date")
                moment = first
            daily[moment.astimezone(stats_timezone).date().isoformat()] += 1
        except (ValueError, TypeError, AttributeError, OverflowError):
            unknown_dates += 1
        coverage = source_coverage(meta)
        unread += coverage["unread_attachments"] or 0
        unknown_inventory += coverage["attachment_inventory_status"] == "unknown"
        suspect += coverage["decode_status"] == "suspect"
    missing = sorted(selected_ids - all_ids) if selected_ids is not None else []
    return {"total_emails": len(selected),
            "top5_senders": [{"sender": sender, "count": count} for sender, count in senders.most_common(5)],
            "label_distribution": dict(labels.most_common(10)),
            "daily_counts": [{"date": date, "count": count} for date, count in sorted(daily.items())[-30:]],
            "source_version": generation,
            "source_set_hash": text_hash(json.dumps(sorted((eid, meta.get("source_sha256")) for eid, meta in selected.items()))),
            "coverage": {"scope": "selected_email_ids" if selected_ids is not None else "indexed_filter_set",
                         "indexed_email_total": len(all_ids), "matched_total": len(selected),
                         "metadata_rows_scanned": len(rows), "complete_within_index_scope": not missing,
                         "mailbox_complete": False, "missing_selected_ids": missing,
                         "filters": {"sender": spec.sender, "labels": sorted(spec.labels),
                                     "start_inclusive": spec.start.isoformat() if spec.start else None,
                                     "end_exclusive": spec.end.isoformat() if spec.end else None},
                         "unread_attachments": unread, "attachment_inventory_unknown_emails": unknown_inventory,
                         "suspect_decode_emails": suspect},
            "date_bucketing": {"timezone": str(stats_timezone), "unknown_date_emails": unknown_dates,
                               "naive_date_policy": "configured_timezone_ambiguous_dates_unknown"},
            "display_limits": {"senders": 5, "labels": 10, "active_dates": 30,
                               "omitted_senders": max(0, len(senders) - 5),
                               "omitted_labels": max(0, len(labels) - 10),
                               "omitted_active_dates": max(0, len(daily) - 30)}}
