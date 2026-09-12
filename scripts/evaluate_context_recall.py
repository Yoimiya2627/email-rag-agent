"""Evaluate retrieval context_recall against manually labeled gold chunks.

RAGAS-style metrics in ``run_ragas_eval.py`` use an LLM judge.  This script
adds a deterministic retrieval metric: for each question, compare retrieved
``chunk_id`` values with human-labeled ``gold_chunk_ids``.

Gold file shape:
[
  {
    "id": "gold_001",
    "question": "...",
    "ground_truth": "...",
    "source_email_ids": ["..."],
    "gold_chunk_ids": ["email_1_chunk_0"],
    "notes": "optional"
  }
]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Callable, Iterable

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.schemas import SearchResult
from scripts.run_ragas_eval import TESTSET_PATH, VERSION_FLAGS, apply_flags

GOLD_PATH = Path(__file__).parent.parent / "data" / "gold_chunks.json"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "eval_results" / "context_recall.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _load_json(path: str | Path) -> list:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON array")
    return payload


def _dump_json(path: str | Path, payload: object) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_gold_template(testset: list, limit: int | None = None) -> list[dict]:
    items = testset[:limit] if limit else testset
    template = []
    for idx, item in enumerate(items, start=1):
        template.append(
            {
                "id": f"gold_{idx:03d}",
                "question": item.get("question", ""),
                "ground_truth": item.get("ground_truth", ""),
                "source_email_ids": item.get("email_ids", []),
                "gold_chunk_ids": [],
                "notes": "Fill gold_chunk_ids after inspecting retrieved/source chunks.",
            }
        )
    return template


def score_context_recall_case(case: dict, retrieved: Iterable[SearchResult]) -> dict:
    gold_chunk_ids = [str(item) for item in case.get("gold_chunk_ids", []) if str(item)]
    retrieved_chunk_ids = [item.chunk_id for item in retrieved]
    retrieved_set = set(retrieved_chunk_ids)
    hit_chunk_ids = [chunk_id for chunk_id in gold_chunk_ids if chunk_id in retrieved_set]
    gold_count = len(gold_chunk_ids)
    hit_count = len(hit_chunk_ids)
    context_recall = round(hit_count / gold_count, 4) if gold_count else 0.0

    return {
        "id": case.get("id", ""),
        "question": case.get("question", ""),
        "gold_chunk_ids": gold_chunk_ids,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "hit_chunk_ids": hit_chunk_ids,
        "hit_count": hit_count,
        "gold_count": gold_count,
        "context_recall": context_recall,
        "has_any_hit": hit_count > 0,
        "has_perfect_recall": gold_count > 0 and hit_count == gold_count,
    }


def evaluate_context_recall_cases(
    cases: list[dict],
    retrieve_fn: Callable[[str], list[SearchResult]],
    limit: int | None = None,
) -> dict:
    selected = cases[:limit] if limit else cases
    records = []
    for case in selected:
        records.append(score_context_recall_case(case, retrieve_fn(case["question"])))

    n = len(records)
    mean_context_recall = (
        round(sum(record["context_recall"] for record in records) / n, 4) if n else 0.0
    )
    chunk_hit_rate = (
        round(sum(1 for record in records if record["has_any_hit"]) / n, 4) if n else 0.0
    )
    perfect_recall_rate = (
        round(sum(1 for record in records if record["has_perfect_recall"]) / n, 4)
        if n
        else 0.0
    )
    return {
        "summary": {
            "n": n,
            "mean_context_recall": mean_context_recall,
            "chunk_hit_rate": chunk_hit_rate,
            "perfect_recall_rate": perfect_recall_rate,
        },
        "records": records,
    }


def _parse_versions(raw: str) -> list[str]:
    versions = [item.strip().upper() for item in raw.split(",") if item.strip()]
    unknown = [version for version in versions if version not in VERSION_FLAGS]
    if unknown:
        raise ValueError(f"Unknown versions: {', '.join(unknown)}")
    return versions


def evaluate_version(version: str, cases: list[dict], limit: int | None = None) -> dict:
    apply_flags(VERSION_FLAGS[version])
    from core.pipeline import retrieve
    from core.reranker import reset_circuit_breaker

    reset_circuit_breaker()
    report = evaluate_context_recall_cases(cases, retrieve_fn=retrieve, limit=limit)
    return {
        "version": version,
        "flags": VERSION_FLAGS[version],
        **report,
    }


def _print_table(results: list[dict]) -> None:
    print(f"{'Version':<8} {'n':>4} {'context_recall':>16} {'hit_rate':>10} {'perfect':>10}")
    print("-" * 62)
    for result in results:
        summary = result["summary"]
        print(
            f"{result['version']:<8} {summary['n']:>4} "
            f"{summary['mean_context_recall']:>16.4f} "
            f"{summary['chunk_hit_rate']:>10.4f} "
            f"{summary['perfect_recall_rate']:>10.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default=str(GOLD_PATH), help="Gold chunk JSON path")
    parser.add_argument("--versions", default="V2,V7", help="Comma-separated versions to evaluate")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument(
        "--init-template",
        action="store_true",
        help="Create a gold chunk template from the RAGAS testset and exit",
    )
    parser.add_argument("--template-from", default=str(TESTSET_PATH))
    args = parser.parse_args()

    if args.init_template:
        template = build_gold_template(_load_json(args.template_from), limit=args.limit)
        _dump_json(args.gold, template)
        logger.info("wrote gold chunk template -> %s", args.gold)
        return

    cases = _load_json(args.gold)
    versions = _parse_versions(args.versions)
    results = [evaluate_version(version, cases, limit=args.limit) for version in versions]
    payload = {"gold_path": args.gold, "results": results}
    _dump_json(args.output, payload)
    _print_table(results)
    logger.info("saved context_recall report -> %s", args.output)


if __name__ == "__main__":
    main()
