"""Sync Gmail read-only messages into the local email JSON corpus.

The sync is conservative: it stores raw Gmail message ids in a local state file
and appends only unseen messages to the configured JSON corpus.  Use
``--index`` when you want to immediately run the normal cleaner/chunker/embedder
pipeline after syncing.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import config.settings as cfg
from agents.gmail_readonly import (
    GmailReadOnlyProvider,
    gmail_message_to_email,
    message_internal_date_ms,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _raw_message_id(email_id: str) -> str:
    return email_id.removeprefix("gmail_")


def sync_gmail_to_json(
    provider: GmailReadOnlyProvider,
    output_path: str | Path,
    state_path: str | Path,
    query: str,
    max_results: int,
) -> dict[str, int]:
    output = Path(output_path)
    state_file = Path(state_path)
    existing = _load_json(output, [])
    state = _load_json(state_file, {})

    seen_order = list(dict.fromkeys(state.get("seen_message_ids") or []))
    seen = set(seen_order)
    existing_ids = {item.get("id", "") for item in existing}
    for item in existing:
        if str(item.get("id", "")).startswith("gmail_"):
            raw_id = _raw_message_id(str(item["id"]))
            if raw_id not in seen:
                seen.add(raw_id)
                seen_order.append(raw_id)

    message_ids = provider.list_message_ids(query=query, max_results=max_results)
    added = 0
    skipped = 0
    last_internal = int(state.get("last_internal_date_ms") or 0)

    for message_id in message_ids:
        local_id = f"gmail_{message_id}"
        if message_id in seen or local_id in existing_ids:
            skipped += 1
            continue

        raw_message = provider.get_message(message_id)
        email = gmail_message_to_email(raw_message)
        existing.append(email.model_dump())
        existing_ids.add(email.id)
        seen.add(message_id)
        seen_order.append(message_id)
        last_internal = max(last_internal, message_internal_date_ms(raw_message))
        added += 1

    _dump_json(output, existing)
    _dump_json(
        state_file,
        {
            "seen_message_ids": seen_order,
            "last_internal_date_ms": last_internal,
            "last_sync_at": datetime.now(timezone.utc).isoformat(),
            "query": query,
        },
    )
    return {
        "fetched": len(message_ids),
        "added": added,
        "skipped": skipped,
        "total": len(existing),
    }


def index_email_json(data_path: str | Path, clear: bool = False) -> dict[str, int]:
    from core.cleaner import clean_email
    from core.chunker import chunk_email
    from core.embedder import clear_collection, get_collection_stats, index_chunks
    from core.loader import load_emails

    if clear:
        clear_collection()
    emails = load_emails(str(data_path))
    chunks = []
    for email in emails:
        chunks.extend(chunk_email(clean_email(email)))
    indexed = index_chunks(chunks)
    from core.retriever import invalidate_bm25_cache

    invalidate_bm25_cache()
    stats = get_collection_stats()
    return {
        "emails": len(emails),
        "indexed_chunks": indexed,
        "collection_chunks": int(stats.get("chunk_count", 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync Gmail read-only messages into local JSON")
    parser.add_argument("--output", default=cfg.GMAIL_SYNC_OUTPUT_PATH)
    parser.add_argument("--state-path", default=cfg.GMAIL_SYNC_STATE_PATH)
    parser.add_argument("--query", default=cfg.GMAIL_SYNC_QUERY)
    parser.add_argument("--max-results", type=int, default=cfg.GMAIL_SYNC_MAX_RESULTS)
    parser.add_argument("--index", action="store_true", help="Index synced JSON after writing it")
    parser.add_argument("--clear-index", action="store_true", help="Clear Chroma collection before indexing")
    args = parser.parse_args()

    provider = GmailReadOnlyProvider()
    result = sync_gmail_to_json(
        provider=provider,
        output_path=args.output,
        state_path=args.state_path,
        query=args.query,
        max_results=args.max_results,
    )
    print(json.dumps({"sync": result}, ensure_ascii=False, indent=2))

    if args.index:
        index_result = index_email_json(args.output, clear=args.clear_index)
        print(json.dumps({"index": index_result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
