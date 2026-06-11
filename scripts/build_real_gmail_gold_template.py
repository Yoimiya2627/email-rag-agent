"""Build a manual context-recall gold template from synced real Gmail data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config.settings as cfg
from core.chunker import chunk_email
from core.cleaner import clean_email
from core.loader import load_emails
from models.schemas import Email


def _preview(text: str, max_chars: int) -> str:
    normalized = text.strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max(0, max_chars - 3)].rstrip() + "..."


def build_real_mail_gold_template(
    emails: list[Email],
    limit: int = 30,
    chunks_per_email: int = 1,
    preview_chars: int = 280,
    min_content_chars: int = 1,
) -> list[dict[str, object]]:
    template: list[dict[str, object]] = []
    if limit <= 0 or chunks_per_email <= 0:
        return template

    for email in emails:
        chunks = chunk_email(clean_email(email))
        usable_chunks = [chunk for chunk in chunks if len(chunk.content.strip()) >= min_content_chars]
        for chunk in usable_chunks[:chunks_per_email]:
            template.append(
                {
                    "id": f"real_gold_{len(template) + 1:03d}",
                    "question": "",
                    "ground_truth": "",
                    "source_email_ids": [email.id],
                    "gold_chunk_ids": [chunk.chunk_id],
                    "source_subject": email.subject,
                    "source_sender": email.sender,
                    "chunk_preview": _preview(chunk.content, preview_chars),
                    "notes": (
                        "Fill question and ground_truth for this real-mail chunk "
                        "before running context recall."
                    ),
                }
            )
            if len(template) >= limit:
                return template
    return template


def _email_key(item: dict[str, object]) -> tuple[str, ...]:
    return tuple(str(value) for value in item.get("source_email_ids") or [])


def _chunk_key(item: dict[str, object]) -> tuple[str, ...]:
    return tuple(str(value) for value in item.get("gold_chunk_ids") or [] if str(value))


def _is_labeled(item: dict[str, object]) -> bool:
    return bool(str(item.get("question") or "").strip()) and bool(
        str(item.get("ground_truth") or "").strip()
    )


def merge_existing_gold_labels(
    generated: list[dict[str, object]],
    existing: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, int]]:
    existing_by_chunk: dict[tuple[str, ...], dict[str, object]] = {}
    for item in existing:
        key = _chunk_key(item)
        if key and _is_labeled(item) and key not in existing_by_chunk:
            existing_by_chunk[key] = item

    preserved = 0
    merged: list[dict[str, object]] = []
    for item in generated:
        fresh = dict(item)
        existing_item = existing_by_chunk.get(_chunk_key(fresh))
        if existing_item:
            fresh["question"] = existing_item["question"]
            fresh["ground_truth"] = existing_item["ground_truth"]
            fresh["notes"] = "Labeled from a preserved real-mail annotation."
            preserved += 1
        merged.append(fresh)

    return merged, {
        "generated": len(generated),
        "preserved_labels": preserved,
        "new_items": len(generated) - preserved,
    }


def _default_output_path() -> str:
    return getattr(
        cfg,
        "GMAIL_REAL_GOLD_PATH",
        str(cfg.BASE_DIR / "data" / "real_emails" / "gold_chunks.real.json"),
    )


def dump_template(path: str | Path, template: list[dict[str, object]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a real-mail gold chunk annotation template."
    )
    parser.add_argument("--input", default=cfg.GMAIL_SYNC_OUTPUT_PATH)
    parser.add_argument("--output", default=_default_output_path())
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--chunks-per-email", type=int, default=1)
    parser.add_argument("--preview-chars", type=int, default=280)
    parser.add_argument("--min-content-chars", type=int, default=120)
    parser.add_argument(
        "--no-preserve-existing",
        action="store_true",
        help="Overwrite labels instead of preserving existing annotations for the same email.",
    )
    args = parser.parse_args()

    emails = load_emails(args.input)
    template = build_real_mail_gold_template(
        emails,
        limit=args.limit,
        chunks_per_email=args.chunks_per_email,
        preview_chars=args.preview_chars,
        min_content_chars=args.min_content_chars,
    )
    stats = {
        "generated": len(template),
        "preserved_labels": 0,
        "new_items": len(template),
    }
    output = Path(args.output)
    if output.exists() and not args.no_preserve_existing:
        existing = json.loads(output.read_text(encoding="utf-8"))
        if not isinstance(existing, list):
            raise ValueError(f"{output} must contain a JSON array")
        template, stats = merge_existing_gold_labels(template, existing)
    dump_template(args.output, template)
    print(
        json.dumps(
            {
                "input": args.input,
                "output": args.output,
                "emails": len(emails),
                "template_items": len(template),
                **stats,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
