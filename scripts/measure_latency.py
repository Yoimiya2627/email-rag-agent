"""
端到端延迟基准：每版跑 N 道题，记录单题挂钟时间均值/中位数/p95。

设计取舍：
  - 复用 run_ragas_eval.py 的 VERSION_FLAGS，只测延迟不打 RAGAS 分数（省一次 LLM 调用）。
  - 默认 N=5，全 7 版在含 reranker 的版本上仍要较久，按需用 --limit 调整。
  - 只测 retriever + rerank + generate 这条主链路（不含意图分类/Self-RAG），与 evaluation.md 第 3 节延迟列对应。
  - 对每题记录 wall-clock 时间，剔除最高最低后取均值（小样本下抗 API 抖动）。

用法:
  python scripts/measure_latency.py                       # 默认 7 版 × 5 题
  python scripts/measure_latency.py --versions V2,V4 --limit 10
  python scripts/measure_latency.py --output data/eval_results/latency.json
"""
import argparse
import json
import logging
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
import config.settings as cfg
from scripts.run_ragas_eval import VERSION_FLAGS, apply_flags

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TESTSET_PATH = Path(__file__).parent.parent / "data" / "ragas_testset.json"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "eval_results" / "latency.json"


def time_one(question: str) -> float:
    from core.pipeline import retrieve
    from core.generator import generate_answer
    from core.model_outcomes import outcome_metadata, ModelOutputError

    t0 = time.perf_counter()
    reranked = retrieve(question)
    answer = generate_answer(question, reranked)
    metadata = outcome_metadata(answer)
    if metadata['status'] != 'success':
        raise ModelOutputError(str(answer), completion_status=metadata['completion_status'],
                               finish_reason=metadata['finish_reason'], error_code=metadata['error_code'])
    return time.perf_counter() - t0


def measure_version(version: str, questions: list) -> dict:
    flags = VERSION_FLAGS[version]
    apply_flags(flags)
    from core.reranker import reset_circuit_breaker
    reset_circuit_breaker()
    logger.info(f"=== {version} | flags={flags}")

    timings, attempts = [], []
    for i, q in enumerate(questions):
        started = time.perf_counter()
        try:
            dt = time_one(q)
            logger.info(f"  [{i+1}/{len(questions)}] {dt:6.2f}s  {q[:50]}")
            timings.append(dt)
            attempts.append({'index': i, 'status': 'success', 'seconds': round(dt, 4)})
        except Exception as exc:
            attempts.append({'index': i, 'status': 'error',
                             'seconds': round(time.perf_counter() - started, 4),
                             'error_type': type(exc).__name__})
            logger.warning('  [%s] failed error_type=%s', i + 1, type(exc).__name__)
        time.sleep(0.3)

    attempted, succeeded = len(attempts), len(timings)
    base = {'version': version, 'flags': flags, 'n': succeeded,
            'attempted': attempted, 'succeeded': succeeded, 'failed': attempted - succeeded,
            'success_rate': round(succeeded / attempted, 4) if attempted else None,
            'attempts': attempts, 'latency_population': 'successful_requests_only'}
    if not timings:
        return {**base, 'raw_seconds': [], 'mean_trimmed': None, 'median': None, 'p95': None}

    trimmed = sorted(timings)[1:-1] if len(timings) >= 4 else timings
    sorted_t = sorted(timings)
    p95_idx = min(len(sorted_t) - 1, math.ceil(len(sorted_t) * 0.95) - 1)
    return {
        **base,
        "raw_seconds": [round(t, 2) for t in timings],
        "mean_trimmed": round(statistics.mean(trimmed), 2),
        "median": round(statistics.median(timings), 2),
        "p95": round(sorted_t[max(0, p95_idx)], 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--versions", default="V1,V2,V3,V4,V5,V6,V7")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive')

    with open(TESTSET_PATH, encoding="utf-8") as f:
        testset = json.load(f)
    questions = [item["question"] for item in testset[: args.limit]]
    logger.info(f"Measuring {len(questions)} questions × {len(args.versions.split(','))} versions")

    OpenAI(api_key=cfg.DEEPSEEK_API_KEY, base_url=cfg.DEEPSEEK_BASE_URL)

    results = []
    for v in args.versions.split(","):
        v = v.strip()
        if v not in VERSION_FLAGS:
            logger.warning(f"unknown version {v}, skipping")
            continue
        results.append(measure_version(v, questions))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"{'Ver':<5} {'ok/attempts':<12} {'failed':<7} {'mean(trim)':>12} {'median':>10} {'p95':>8}")
    print("-" * 60)
    for r in results:
        if r.get("n"):
            print(f"{r['version']:<5} {str(r['succeeded'])+'/'+str(r['attempted']):<12} {r['failed']:<7} {r['mean_trimmed']:>10.2f}s "
                  f"{r['median']:>8.2f}s {r['p95']:>6.2f}s")
        else:
            print(f"{r['version']:<5} 0/{r['attempted']} failed={r['failed']} latency unavailable")
    print('Latency quantiles describe successful requests only; failures remain in the attempt denominator.')
    print("=" * 60)
    logger.info(f"saved → {args.output}")


if __name__ == "__main__":
    main()
