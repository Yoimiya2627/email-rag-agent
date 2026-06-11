"""Build a private Agent EvalOps taskset from real Gmail gold labels."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import config.settings as cfg

DEFAULT_OUTPUT = Path("data/real_emails/agent_testset.real.json")
SUPPORTED_VARIANTS = {"retrieval", "detail"}
_BOILERPLATE_MARKERS = (
    "email not displaying correctly",
    "view it",
    "unsubscribe",
    "copyright",
    "google llc",
    "amphitheatre",
    "privacy",
    "在浏览器中查看",
    "取消订阅",
    "隐私权",
    "条款",
)
_FOCUS_RE = re.compile(r"\bsay regarding\s+(.*)\?\s*$", flags=re.IGNORECASE)


def _is_labeled(item: dict[str, Any]) -> bool:
    return bool(str(item.get("question") or "").strip()) and bool(
        str(item.get("ground_truth") or "").strip()
    )


def _question_focus(question: str) -> tuple[str, bool]:
    match = _FOCUS_RE.search(question)
    if match:
        return " ".join(match.group(1).split()), True
    return " ".join(question.split()), False


def _has_hidden_filler(text: str) -> bool:
    return "\u034f" in text or any(ord(char) < 32 for char in text)


def _is_agent_usable(item: dict[str, Any]) -> bool:
    focus, is_generated_focus = _question_focus(str(item.get("question") or ""))
    if len(focus) < 24 or _has_hidden_filler(focus):
        return False

    lower_focus = focus.lower()
    if any(marker in lower_focus for marker in _BOILERPLATE_MARKERS):
        return False

    cjk_chars = sum("\u4e00" <= char <= "\u9fff" for char in focus)
    ascii_letters = sum(char.isascii() and char.isalpha() for char in focus)
    english_words = re.findall(r"[A-Za-z][A-Za-z']+", focus)

    if is_generated_focus and ascii_letters > cjk_chars * 2:
        first_word = english_words[0] if english_words else ""
        if len(english_words) < 8:
            return False
        if first_word and first_word[0].islower():
            return False

    if cjk_chars and cjk_chars < 12 and ascii_letters < 20:
        return False

    return True


def _source_key(item: dict[str, Any]) -> tuple[str, ...]:
    source_ids = tuple(str(value) for value in item.get("source_email_ids") or [] if value)
    if source_ids:
        return source_ids
    return (
        str(item.get("source_sender") or ""),
        str(item.get("source_subject") or ""),
    )


def _base_metadata(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "risk_level": "low",
        "forbidden_tools": ["send_email"],
        "source_gold_id": item.get("id", ""),
        "source_email_ids": list(item.get("source_email_ids") or []),
        "gold_chunk_ids": list(item.get("gold_chunk_ids") or []),
        "source_subject": item.get("source_subject", ""),
        "source_sender": item.get("source_sender", ""),
    }


def _success_criteria(item: dict[str, Any]) -> str:
    source_ids = ", ".join(str(value) for value in item.get("source_email_ids") or [])
    return (
        "Answer from synced Gmail evidence only. The answer should be consistent with "
        f"this gold reference: {item.get('ground_truth', '')}. "
        f"Source email ids: {source_ids}."
    )


def _retrieval_task(item: dict[str, Any], index: int) -> dict[str, Any]:
    question = str(item.get("question") or "").strip()
    task = {
        "id": f"real_agent_retrieval_{index:03d}",
        "task": f"Using the synced Gmail mailbox, answer from email evidence only: {question}",
        "task_type": "real_retrieval",
        "expected_tools": ["search_emails"],
        "success_criteria": _success_criteria(item),
        **_base_metadata(item),
    }
    return task


def _detail_task(item: dict[str, Any], index: int) -> dict[str, Any]:
    question = str(item.get("question") or "").strip()
    subject = str(item.get("source_subject") or "").strip()
    sender = str(item.get("source_sender") or "").strip()
    task = {
        "id": f"real_agent_detail_{index:03d}",
        "task": (
            "Find the synced Gmail message"
            f" from {sender or 'the source sender'}"
            f" about \"{subject or 'the labeled source email'}\", read its full details, "
            f"then answer from evidence only: {question}"
        ),
        "task_type": "real_detail_lookup",
        "expected_tools": ["search_emails", "get_email"],
        "success_criteria": _success_criteria(item),
        **_base_metadata(item),
    }
    return task


def build_real_agent_testset(
    gold_cases: list[dict[str, Any]],
    limit: int = 30,
    variants: list[str] | tuple[str, ...] = ("retrieval", "detail"),
    max_cases_per_source_email: int = 1,
) -> list[dict[str, Any]]:
    """Convert labeled real gold cases into private Agent EvalOps tasks."""
    unknown = [variant for variant in variants if variant not in SUPPORTED_VARIANTS]
    if unknown:
        raise ValueError(f"Unsupported real agent task variant: {', '.join(unknown)}")
    if limit <= 0:
        return []

    tasks: list[dict[str, Any]] = []
    variant_counts = {variant: 0 for variant in variants}
    source_counts: dict[tuple[str, ...], int] = {}
    for item in gold_cases:
        if not _is_labeled(item) or not _is_agent_usable(item):
            continue
        source_key = _source_key(item)
        if max_cases_per_source_email > 0:
            if source_counts.get(source_key, 0) >= max_cases_per_source_email:
                continue
            source_counts[source_key] = source_counts.get(source_key, 0) + 1
        for variant in variants:
            variant_counts[variant] += 1
            if variant == "retrieval":
                tasks.append(_retrieval_task(item, variant_counts[variant]))
            elif variant == "detail":
                tasks.append(_detail_task(item, variant_counts[variant]))
            if len(tasks) >= limit:
                return tasks
    return tasks


def _default_gold_path() -> str:
    return getattr(
        cfg,
        "GMAIL_REAL_GOLD_PATH",
        str(cfg.BASE_DIR / "data" / "real_emails" / "gold_chunks.real.json"),
    )


def _write_json(path: str | Path, data: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build real Gmail Agent EvalOps taskset")
    parser.add_argument("--gold", default=_default_gold_path())
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--variants", default="retrieval,detail")
    args = parser.parse_args(argv)

    gold_path = Path(args.gold)
    gold_cases = json.loads(gold_path.read_text(encoding="utf-8"))
    if not isinstance(gold_cases, list):
        raise ValueError(f"{gold_path} must contain a JSON array")
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    tasks = build_real_agent_testset(gold_cases, limit=args.limit, variants=variants)
    _write_json(args.output, tasks)
    print(
        json.dumps(
            {
                "gold": str(gold_path),
                "output": args.output,
                "gold_cases": len(gold_cases),
                "tasks": len(tasks),
                "variants": variants,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
