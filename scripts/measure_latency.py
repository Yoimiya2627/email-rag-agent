"""
端到端延迟基准：每版跑 N 道题，记录单题挂钟时间均值/中位数/p95。

设计取舍：
  - 复用 run_ragas_eval.py 的 VERSION_FLAGS，只测延迟不打 RAGAS 分数（省一次 LLM 调用）。
  - 默认 N=5，全 6 版在含 reranker 的版本上仍要 ~10 分钟，按需用 --limit 调整。
  - 只测 retriever + rerank + generate 这条主链路（不含意图分类/Self-RAG），与 evaluation.md 第 3 节延迟列对应。
  - 对每题记录 wall-clock 时间，剔除最高最低后取均值（小样本下抗 API 抖动）。

用法:
  python scripts/measure_latency.py                       # 默认 6 版 × 5 题
  python scripts/measure_latency.py --versions V2,V4 --limit 10
  python scripts/measure_latency.py --output data/eval_results/latency.json
"""
import argparse
import json
import logging
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
    from core.retriever import hybrid_search
    from core.reranker import rerank
    from core.generator import generate_answer
    from agents.retriever_agent import RetrieverAgent

    agent = RetrieverAgent()
    t0 = time.perf_counter()
    rewritten = agent._rewrite_query(question)
    filters = agent._extract_filters(rewritten)
    search_query = filters.get("query") or rewritten
    results = hybrid_search(search_query, top_k=cfg.TOP_K * 4)
    reranked = rerank(question, results, top_n=cfg.RERANK_TOP_N)
    _ = generate_answer(question, reranked)
    return time.perf_counter() - t0


def measure_version(version: str, questions: list) -> dict:
    flags = VERSION_FLAGS[version]
    apply_flags(flags)
    from core.reranker import reset_circuit_breaker
    reset_circuit_breaker()
    logger.info(f"=== {version} | flags={flags}")

    timings = []
    for i, q in enumerate(questions):
        try:
            dt = time_one(q)
            logger.info(f"  [{i+1}/{len(questions)}] {dt:6.2f}s  {q[:50]}")
            timings.append(dt)
        except Exception as exc:
            logger.warning(f"  [{i+1}] failed: {exc}")
        time.sleep(0.3)

    if not timings:
        return {"version": version, "flags": flags, "n": 0}

    trimmed = sorted(timings)[1:-1] if len(timings) >= 4 else timings
    return {
        "version": version,
        "flags": flags,
        "n": len(timings),
        "raw_seconds": [round(t, 2) for t in timings],
        "mean_trimmed": round(statistics.mean(trimmed), 2),
        "median": round(statistics.median(timings), 2),
        "p95": round(sorted(timings)[max(0, int(len(timings) * 0.95) - 1)], 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--versions", default="V1,V2,V3,V4,V5,V6")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    args = parser.parse_args()

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
    print(f"{'Ver':<5} {'n':<4} {'mean(trim)':>12} {'median':>10} {'p95':>8}")
    print("-" * 60)
    for r in results:
        if r.get("n"):
            print(f"{r['version']:<5} {r['n']:<4} {r['mean_trimmed']:>10.2f}s "
                  f"{r['median']:>8.2f}s {r['p95']:>6.2f}s")
    print("=" * 60)
    logger.info(f"saved → {args.output}")


if __name__ == "__main__":
    main()
