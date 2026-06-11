"""Quality gate for local real-mail context-recall gold labels."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import config.settings as cfg


TEXT_FIELDS = ("question", "ground_truth", "chunk_preview")
CHUNK_POSITION_RE = re.compile(
    r"(\bchunk\s*[_#-]?\s*\d+\b|第\s*\d+\s*个片段|对应片段)",
    flags=re.IGNORECASE,
)
LOCAL_ANNOTATION_MARKERS = (
    "locally labeled",
    "local quality",
    "local annotation",
    "auto",
    "generated",
)


def _default_gold_path() -> str:
    return getattr(
        cfg,
        "GMAIL_REAL_GOLD_PATH",
        str(cfg.BASE_DIR / "data" / "real_emails" / "gold_chunks.real.json"),
    )


def _load_json(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON array")
    return payload


def _case_id(item: dict[str, Any], index: int) -> str:
    return str(item.get("id") or f"case_{index + 1}")


def _chunk_ids(item: dict[str, Any]) -> list[str]:
    return [str(value) for value in item.get("gold_chunk_ids") or [] if str(value)]


def _has_text_artifact(value: object) -> bool:
    text = str(value or "")
    return "\ufffd" in text or "????" in text or "Ã" in text or "Â" in text


def _is_labeled(item: dict[str, Any]) -> bool:
    return bool(str(item.get("question") or "").strip()) and bool(
        str(item.get("ground_truth") or "").strip()
    )


def _is_local_annotation(item: dict[str, Any]) -> bool:
    notes = str(item.get("notes") or "").lower()
    return any(marker in notes for marker in LOCAL_ANNOTATION_MARKERS)


def _issue(code: str, message: str, case_id: str | None = None) -> dict[str, str]:
    payload = {"code": code, "message": message}
    if case_id:
        payload["case_id"] = case_id
    return payload


def evaluate_gold_quality(
    items: list[dict[str, Any]],
    min_cases: int = 100,
    max_local_annotation_ratio: float = 1.0,
) -> dict[str, Any]:
    blockers: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    if len(items) < min_cases:
        blockers.append(
            _issue("too_few_cases", f"gold cases {len(items)} < required {min_cases}")
        )

    seen_chunks: dict[str, str] = {}
    unique_emails: set[tuple[str, ...]] = set()
    local_annotations = 0
    for index, item in enumerate(items):
        case_id = _case_id(item, index)
        chunk_ids = _chunk_ids(item)
        if not _is_labeled(item) or not chunk_ids:
            blockers.append(
                _issue("unlabeled_case", "case must have question, ground_truth, and gold_chunk_ids", case_id)
            )
        for chunk_id in chunk_ids:
            previous_case = seen_chunks.get(chunk_id)
            if previous_case:
                blockers.append(
                    _issue(
                        "duplicate_gold_chunk_id",
                        f"{chunk_id} is used by both {previous_case} and {case_id}",
                        case_id,
                    )
                )
            else:
                seen_chunks[chunk_id] = case_id

        email_key = tuple(str(value) for value in item.get("source_email_ids") or [])
        if email_key:
            unique_emails.add(email_key)

        if CHUNK_POSITION_RE.search(str(item.get("question") or "")):
            blockers.append(
                _issue(
                    "question_leaks_chunk_position",
                    "question must not expose chunk ids or chunk position hints",
                    case_id,
                )
            )

        for field in TEXT_FIELDS:
            if _has_text_artifact(item.get(field)):
                blockers.append(
                    _issue(
                        "suspicious_text_artifact",
                        f"{field} contains replacement/mojibake markers",
                        case_id,
                    )
                )
                break

        if _is_local_annotation(item):
            local_annotations += 1

    local_ratio = round(local_annotations / len(items), 4) if items else 0.0
    if local_ratio > max_local_annotation_ratio:
        blockers.append(
            _issue(
                "too_many_local_annotations",
                f"local annotation ratio {local_ratio:.4f} > {max_local_annotation_ratio:.4f}",
            )
        )
    elif local_annotations:
        warnings.append(
            _issue(
                "local_annotations_present",
                f"{local_annotations} cases are marked as local/generated annotations",
            )
        )

    summary = {
        "total_cases": len(items),
        "labeled_cases": sum(1 for item in items if _is_labeled(item) and _chunk_ids(item)),
        "unique_gold_chunks": len(seen_chunks),
        "unique_source_emails": len(unique_emails),
        "local_annotation_count": local_annotations,
        "local_annotation_ratio": local_ratio,
    }
    return {
        "ok": not blockers,
        "summary": summary,
        "blockers": blockers,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check real-mail gold label quality.")
    parser.add_argument("--gold", default=_default_gold_path())
    parser.add_argument("--min-cases", type=int, default=100)
    parser.add_argument("--max-local-annotation-ratio", type=float, default=1.0)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()

    report = evaluate_gold_quality(
        _load_json(args.gold),
        min_cases=args.min_cases,
        max_local_annotation_ratio=args.max_local_annotation_ratio,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("Real gold quality gate")
        for key, value in report["summary"].items():
            print(f"  {key:<24}: {value}")
        print(f"  result                  : {'PASS' if report['ok'] else 'FAIL'}")
        for issue in report["blockers"]:
            print(f"  blocker {issue['code']}: {issue['message']}")
        for issue in report["warnings"]:
            print(f"  warning {issue['code']}: {issue['message']}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
