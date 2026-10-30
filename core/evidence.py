"""Bounded table lineage for model-visible excerpts; never alter indexed text."""
import json
import hashlib


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def evidence_reference(source: dict) -> dict:
    """Identify the exact text currently carried by a source, not unseen tails.

    Offsets are Unicode character offsets within the indexed chunk, never MIME
    byte offsets. A legacy chunk hash identifies that chunk only, not the mail.
    """
    meta = source.get("metadata") or {}
    content = source.get("content", source.get("snippet", ""))
    content = content if isinstance(content, str) else ""
    start = source.get("visible_start", meta.get("visible_start", 0))
    start = start if type(start) is int and start >= 0 else 0
    chunk_hash = source.get("chunk_sha256") or meta.get("chunk_sha256") or text_hash(content)
    version = (source.get("source_version") or meta.get("source_version")
               or meta.get("index_generation") or "legacy-chunk:" + chunk_hash)
    return {"email_id": source["email_id"], "chunk_id": source["chunk_id"],
            "source_version": version,
            "source_sha256": source.get("source_sha256") or meta.get("source_sha256"),
            "chunk_sha256": chunk_hash, "visible_start": start, "visible_end": start + len(content),
            "visible_hash": text_hash(content), "offset_basis": "chunk_text"}


def with_visible_reference(source: dict) -> dict:
    return {**source, **evidence_reference(source)}


def rendered_evidence(source: dict, *, max_chars=None):
    """Return display text and the exact original-text range inside it.

    Table provenance notes use the rendering budget but are not source text;
    they must never enlarge the claimed visible source interval or its hash.
    """
    content, metadata = source["content"], source.get("metadata") or {}
    rendered = evidence_text(content, metadata, max_chars=max_chars)
    visible = rendered
    for truncated in (False, True):
        note = table_context_note(metadata, excerpt_truncated=truncated)
        if note and rendered.startswith(note + "\n"):
            visible = rendered[len(note) + 1:]
            break
    if not content.startswith(visible):
        raise ValueError("rendered evidence does not match its indexed source")
    original_ref = evidence_reference(source)
    ref = evidence_reference({**source, **original_ref, "content": visible})
    return rendered, ref if visible else None


def source_coverage(metadata: dict) -> dict:
    """Bounded uncertainty indicators from source metadata, without guessing."""
    cached = metadata.get('coverage')
    if isinstance(cached, dict) and 'attachments' not in metadata:
        known = cached.get('attachment_inventory_status') == 'available'
        count, unread = cached.get('attachment_count'), cached.get('unread_attachments')
        known = known and type(count) is int and count >= 0 and type(unread) is int and 0 <= unread <= count
        return {'attachment_count': count if known else None, 'unread_attachments': unread if known else None,
                'attachment_inventory_present': cached.get('attachment_inventory_present') is True,
                'attachment_inventory_status': 'available' if known else 'unknown',
                'decode_status': cached.get('decode_status') if cached.get('decode_status') in {'suspect','no_signal','unknown'} else 'unknown',
                'mailbox_sync_complete': 'unknown', 'scope': 'indexed_text_only'}
    def decoded(name, fallback):
        value = metadata.get(name, fallback)
        if isinstance(value, str):
            if len(value) > 20000:
                return fallback
            try:
                value = json.loads(value)
            except (ValueError, TypeError):
                return fallback
        return value
    attachments = decoded("attachments", None)
    inventory_known = isinstance(attachments, list)
    attachments = attachments if inventory_known else []
    quality = decoded("decode_quality", {})
    status = quality.get("status", "unknown") if isinstance(quality, dict) else "unknown"
    if isinstance(quality, dict) and any(isinstance(value, dict) and value.get("status") == "suspect" for value in quality.values()):
        status = "suspect"
    elif isinstance(quality, dict) and any(isinstance(value, dict) and value.get("status") == "no_signal" for value in quality.values()):
        status = "no_signal"
    return {"attachment_count": len(attachments) if inventory_known else None,
            "unread_attachments": sum(not isinstance(row, dict) or row.get("status") != "parsed" for row in attachments) if inventory_known else None,
            "attachment_inventory_present": "attachments" in metadata,
            "attachment_inventory_status": "available" if inventory_known else "unknown",
            "decode_status": status if status in {"no_signal", "suspect", "unknown"} else "unknown",
            "mailbox_sync_complete": "unknown", "scope": "indexed_text_only"}


def refresh_visible_references(value):
    """Refresh hashes/ends after JSON clipping, including continuation offsets."""
    if isinstance(value, list):
        return [refresh_visible_references(row) for row in value]
    if not isinstance(value, dict):
        return value
    result = {key: refresh_visible_references(row) for key, row in value.items()}
    if result.get('material_type') == 'historical_tool_result' and isinstance(result.get('text'), str) and type(result.get('page_start')) is int:
        end = result['page_start'] + len(result['text'])
        total = result.get('total_chars', end)
        result.update(page_end=end, page_hash=text_hash(result['text']), has_more=end < total,
                      next_start=end if end < total else None)
    if (result.get('material_type') == 'historical_conversation' or result.get('field') in ('query', 'answer')) and isinstance(result.get('text'), str) and type(result.get('start')) is int:
        end = result['start'] + len(result['text'])
        truncated = end < result.get('end', end)
        result.update(end=end, sha256=text_hash(result['text']))
        if truncated:
            result.update(has_more=True, next_offset=end)
    if (isinstance(result.get("email_id"), str) and isinstance(result.get("chunk_id"), str)
            and (isinstance(result.get("content"), str) or isinstance(result.get("snippet"), str))):
        result.update(evidence_reference(result))
        if type(result.get("page_start")) is int:
            result["page_end"] = result["page_start"] + len(result.get("content", result.get("snippet", "")))
    if result.get("page_kind") == "email_text" and type(result.get("read_start")) is int:
        cursor = result["read_start"]
        chunks = result.get("chunks", [])
        for chunk in chunks:
            if type(chunk.get("page_start")) is int and chunk["page_start"] <= cursor:
                cursor = max(cursor, chunk["page_end"])
            elif (type(chunk.get("page_start")) is int and not result.get("reconstruction_exact")
                  and 0 < chunk["page_start"] - cursor <= 2):
                # Legacy reconstruction inserts two synthetic newlines between
                # chunks. They contain no omitted source evidence.
                cursor = max(cursor, chunk["page_end"])
            else:
                break
        if not chunks and isinstance(result.get("body"), str):
            cursor += len(result["body"])
        scope_end = result.get("verified_end", result["total_chars"])
        result.update(read_end=cursor, has_more=cursor < scope_end,
                      next_start=cursor if cursor < scope_end else None)
    if result.get("page_kind") == "search_candidates" and isinstance(result.get("items"), list):
        rows = result["items"]
        end = result["offset"] + len(rows)
        total = result["coverage"]["selected_total"]
        result.update(next_cursor=(rows[-1].get("continuation_cursor") if rows else result.get("resume_cursor")),
                      has_more=end < total, remaining=total - end)
        result["coverage"]["returned"] = len(rows)
    if result.get("page_kind") == "thread_metadata" and isinstance(result.get("items"), list):
        end = result["read_start"] + len(result["items"])
        result.update(next_start=end if end < result["total_indexed_messages"] else None,
                      has_more=end < result["total_indexed_messages"])
    return result


def table_context_note(metadata: dict, *, excerpt_truncated: bool = False) -> str:
    raw = metadata.get("table_context")
    if not raw:
        return ""
    incomplete = "[表格上下文不完整，请按邮件 ID 读取全文核对。]"
    if not isinstance(raw, str) or len(raw) > 200_000:
        return incomplete
    try:
        rows = json.loads(raw)
        if not isinstance(rows, list):
            return incomplete
        if not rows:
            return ""
        summary = []
        for row in rows[:4]:
            item = {key: row.get(key) for key in ("table_id", "row_id", "status", "partial_row")}
            if excerpt_truncated:
                item["partial_row"] = True
            item["cells"] = [{key: cell.get(key) for key in ("source_id", "column", "headers")}
                             for cell in row.get("cells", [])[:8]]
            summary.append(item)
        value = json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
        if len(value) > 1800:
            # Keep complete identifiers/labels; do not slice through JSON or IDs.
            value = json.dumps([{key: item.get(key) for key in ("table_id", "row_id", "status", "partial_row")}
                                for item in summary], ensure_ascii=False, separators=(",", ":"))
            if len(value) > 1800:
                return incomplete
            incomplete_note = "；列信息省略"
        else:
            incomplete_note = "；部分行列信息省略" if len(rows) > 4 or any(len(row.get("cells", [])) > 8 for row in rows[:4]) else ""
        return "[表格来源信息（邮件数据；partial_row=true 表示只读到部分行" + incomplete_note + "）: " + value + "]"
    except (ValueError, TypeError, AttributeError):
        return incomplete


def evidence_text(content: str, metadata: dict, *, excerpt_truncated: bool = False,
                  max_chars: int | None = None) -> str:
    note = table_context_note(metadata, excerpt_truncated=excerpt_truncated)
    if max_chars is None:
        return note + "\n" + content if note else content
    if not note:
        return content[:max_chars]
    if len(note) + 1 + len(content) > max_chars:
        note = table_context_note(metadata, excerpt_truncated=True)
    if len(note) + 1 >= max_chars:
        # No room for a complete provenance note and evidence. Do not expose an
        # unlabelled tail of a table just to fill the last few characters.
        return ""
    return note + "\n" + content[:max_chars - len(note) - 1]
