"""Micro-benchmark the reranker step for V2/V3/V7 serving choices.

This script isolates rerank latency from generation latency. It builds small
candidate sets from the RAGAS testset and runs `core.reranker.rerank()` under
the same flags used by the ablation versions:

  V2: reranker disabled
  V3: LLM scorer reranker
  V7: Cross-Encoder reranker

Use `--mock-cross-encoder` for a no-model smoke test. Do not publish mock
numbers as real latency.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

sys.path.insert(0, str(Path(__file__).parent.parent))

import config.settings as cfg
from core.reranker_policy import choose_reranker_policy
from models.schemas import SearchResult

TESTSET_PATH = Path(__file__).parent.parent / "data" / "ragas_testset.json"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "eval_results" / "reranker_latency.json"


@dataclass(frozen=True)
class RerankerBenchmarkTarget:
    version: str
    enable_reranker: bool
    backend: str


@dataclass(frozen=True)
class RerankerBenchmarkCase:
    query: str
    candidates: List[SearchResult]


RERANKER_TARGETS = {
    "V2": RerankerBenchmarkTarget(version="V2", enable_reranker=False, backend="none"),
    "V3": RerankerBenchmarkTarget(version="V3", enable_reranker=True, backend="llm"),
    "V7": RerankerBenchmarkTarget(version="V7", enable_reranker=True, backend="cross_encoder"),
}


class _MockCrossEncoder:
    def predict(self, pairs):
        # Deterministic and cheap: longer candidate text gets a larger score.
        return [float(len(doc)) for _, doc in pairs]


def parse_versions(raw: str) -> List[RerankerBenchmarkTarget]:
    targets = []
    for item in raw.split(","):
        version = item.strip().upper()
        if not version:
            continue
        if version not in RERANKER_TARGETS:
            raise ValueError(f"Unknown reranker benchmark version: {version}")
        targets.append(RERANKER_TARGETS[version])
    return targets


def _candidate(
    query_index: int,
    candidate_index: int,
    content: str,
    source: str,
    email_id: str,
) -> SearchResult:
    return SearchResult(
        chunk_id=f"bench_{query_index}_{candidate_index}",
        email_id=email_id,
        content=content,
        score=0.0,
        metadata={"source": source, "benchmark": "reranker_latency"},
    )


def build_cases(testset: list, limit: int, candidate_count: int) -> List[RerankerBenchmarkCase]:
    items = testset[:limit]
    all_truths = [item.get("ground_truth", "") for item in testset if item.get("ground_truth")]
    cases = []

    for idx, item in enumerate(items):
        contents = [item.get("ground_truth", "")]
        distractor_pool = all_truths[idx + 1 :] + all_truths[:idx]
        contents.extend(distractor_pool[: max(0, candidate_count - 1)])
        while len(contents) < candidate_count:
            contents.append(f"Synthetic distractor {len(contents)} for {item.get('question', '')}")

        email_ids = item.get("email_ids") or [f"synthetic_{idx}"]
        candidates = []
        for cand_idx, content in enumerate(contents[:candidate_count]):
            source = "ground_truth" if cand_idx == 0 else "distractor"
            email_id = email_ids[0] if source == "ground_truth" else f"distractor_{idx}_{cand_idx}"
            candidates.append(_candidate(idx, cand_idx, content, source, email_id))

        cases.append(RerankerBenchmarkCase(query=item["question"], candidates=candidates))
    return cases


def summarize_timings(timings: Iterable[float]) -> dict:
    values = list(timings)
    if not values:
        return {"n": 0, "raw_ms": [], "mean_trimmed_ms": 0.0, "median_ms": 0.0, "p95_ms": 0.0}

    raw_ms = [round(value * 1000, 2) for value in values]
    sorted_ms = sorted(raw_ms)
    trimmed = sorted_ms[1:-1] if len(sorted_ms) >= 4 else sorted_ms
    p95_idx = min(len(sorted_ms) - 1, math.ceil(len(sorted_ms) * 0.95) - 1)
    return {
        "n": len(raw_ms),
        "raw_ms": raw_ms,
        "mean_trimmed_ms": round(statistics.mean(trimmed), 2),
        "median_ms": round(statistics.median(sorted_ms), 2),
        "p95_ms": round(sorted_ms[max(0, p95_idx)], 2),
    }


def _apply_target(target: RerankerBenchmarkTarget) -> None:
    cfg.ENABLE_RERANKER = target.enable_reranker
    if target.backend != "none":
        cfg.RERANKER_BACKEND = target.backend


def measure_target(
    target: RerankerBenchmarkTarget,
    cases: List[RerankerBenchmarkCase],
    runs: int,
    mock_cross_encoder: bool = False,
) -> dict:
    import core.reranker as reranker_mod

    _apply_target(target)
    reranker_mod.reset_circuit_breaker()
    if mock_cross_encoder and target.backend == "cross_encoder":
        reranker_mod._cross_encoder = _MockCrossEncoder()

    timings = []
    for _ in range(runs):
        for case in cases:
            t0 = time.perf_counter()
            reranker_mod.rerank(case.query, case.candidates, top_n=cfg.RERANK_TOP_N)
            timings.append(time.perf_counter() - t0)

    summary = summarize_timings(timings)
    return {
        "version": target.version,
        "enable_reranker": target.enable_reranker,
        "backend": target.backend,
        **summary,
    }


def load_testset(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--versions", default="V2,V7")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--candidate-count", type=int, default=20)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument("--mock-cross-encoder", action="store_true")
    args = parser.parse_args()

    targets = parse_versions(args.versions)
    testset = load_testset(TESTSET_PATH)
    cases = build_cases(testset, limit=args.limit, candidate_count=args.candidate_count)

    results = [
        measure_target(
            target,
            cases=cases,
            runs=args.runs,
            mock_cross_encoder=args.mock_cross_encoder,
        )
        for target in targets
    ]
    policy = {
        "conversation": asdict(choose_reranker_policy("conversation", latency_budget_ms=9000)),
        "quality": asdict(choose_reranker_policy("quality", latency_budget_ms=20000)),
        "precision_deterministic": asdict(choose_reranker_policy("precision", allow_llm_reranker=False)),
        "precision_llm_reference": asdict(choose_reranker_policy("precision", allow_llm_reranker=True)),
    }
    payload = {
        "benchmark": "reranker_latency",
        "mock_cross_encoder": args.mock_cross_encoder,
        "limit": args.limit,
        "candidate_count": args.candidate_count,
        "runs": args.runs,
        "results": results,
        "policy": policy,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"{'Ver':<4} {'backend':<14} {'n':>4} {'mean(trim)':>12} {'median':>10} {'p95':>10}")
    print("-" * 62)
    for result in results:
        print(
            f"{result['version']:<4} {result['backend']:<14} {result['n']:>4} "
            f"{result['mean_trimmed_ms']:>10.2f}ms "
            f"{result['median_ms']:>8.2f}ms "
            f"{result['p95_ms']:>8.2f}ms"
        )
    print(f"saved -> {output_path}")


if __name__ == "__main__":
    main()
