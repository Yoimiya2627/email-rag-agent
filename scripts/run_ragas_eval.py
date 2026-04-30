"""
RAGAS-style ablation evaluation across 6 retrieval versions.

Versions:
  V1: Vector only (no BM25, no RRF, no reranker, no query rewrite)
  V2: Vector + BM25 + RRF
  V3: V2 + Reranker
  V4: V3 + Query Rewrite  (full pipeline)
  V5: V4 without RRF      (BM25 simple merge)
  V6: V4 without Reranker (to ablate reranker contribution)

Metrics (implemented locally, no ragas package required):
  - Answer Relevancy: LLM scores how relevant the answer is to the question (0-1)
  - Faithfulness: LLM scores if answer is grounded in retrieved contexts (0-1)
  - Context Precision: fraction of retrieved chunks that are truly relevant (0-1)

Usage:
  python scripts/run_ragas_eval.py [--versions V1,V2,V3] [--limit N] [--output path]
"""
import argparse
import json
import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
import config.settings as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TESTSET_PATH = Path(__file__).parent.parent / "data" / "ragas_testset.json"
RESULTS_DIR = Path(__file__).parent.parent / "data" / "eval_results"

VERSION_FLAGS = {
    "V1": dict(ENABLE_BM25=False, ENABLE_RRF=False, ENABLE_RERANKER=False, ENABLE_QUERY_REWRITE=False),
    "V2": dict(ENABLE_BM25=True,  ENABLE_RRF=True,  ENABLE_RERANKER=False, ENABLE_QUERY_REWRITE=False),
    "V3": dict(ENABLE_BM25=True,  ENABLE_RRF=True,  ENABLE_RERANKER=True,  ENABLE_QUERY_REWRITE=False),
    "V4": dict(ENABLE_BM25=True,  ENABLE_RRF=True,  ENABLE_RERANKER=True,  ENABLE_QUERY_REWRITE=True),
    "V5": dict(ENABLE_BM25=True,  ENABLE_RRF=False, ENABLE_RERANKER=True,  ENABLE_QUERY_REWRITE=True),
    "V6": dict(ENABLE_BM25=True,  ENABLE_RRF=True,  ENABLE_RERANKER=False, ENABLE_QUERY_REWRITE=True),
}



def apply_flags(flags: dict):
    for k, v in flags.items():
        setattr(cfg, k, v)


def run_single(client: OpenAI, question: str, ground_truth: str) -> Dict[str, Any]:
    """Run one question through current RAG pipeline, return answer + contexts."""
    from core.retriever import hybrid_search
    from core.reranker import rerank
    from core.generator import generate_answer
    from agents.retriever_agent import RetrieverAgent

    agent = RetrieverAgent()
    rewritten = agent._rewrite_query(question)
    filters = agent._extract_filters(rewritten)
    search_query = filters.get("query") or rewritten

    results = hybrid_search(search_query, top_k=cfg.TOP_K * 4)
    reranked = rerank(question, results, top_n=cfg.RERANK_TOP_N)
    answer = generate_answer(question, reranked)
    contexts = [r.content for r in reranked]
    return {"answer": answer, "contexts": contexts}


_EVAL_USER_TMPL = """你是RAG系统评测专家。请给下面的检索问答打分。

【问题】{question}

【检索到的上下文片段】
{ctx_text}

【系统生成的答案】
{answer}

请根据以下三个维度打分（每个维度0.00到1.00之间的小数）：
1. answer_relevancy：答案与问题的相关程度（1.0=完全切题）
2. faithfulness：答案是否有上下文依据（1.0=完全有据可查）
3. context_precision：上下文中真正有用的片段比例（1.0=全部有用）

请直接输出JSON，例如：
{{"answer_relevancy": 0.85, "faithfulness": 0.90, "context_precision": 0.75}}"""


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb + 1e-9)


def _score_by_embedding(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """降级方案：用向量相似度打分，不依赖 LLM。"""
    from core.embedder import embed_texts
    texts = [question, answer] + contexts
    vecs = embed_texts(texts)
    q_vec, a_vec, ctx_vecs = vecs[0], vecs[1], vecs[2:]

    answer_relevancy = round(max(0.0, _cosine(q_vec, a_vec)), 4)
    faithfulness = round(max((max(0.0, _cosine(a_vec, c)) for c in ctx_vecs), default=0.0), 4)
    threshold = 0.5
    relevant = sum(1 for c in ctx_vecs if _cosine(q_vec, c) >= threshold)
    context_precision = round(relevant / max(len(ctx_vecs), 1), 4)
    return {"answer_relevancy": answer_relevancy, "faithfulness": faithfulness, "context_precision": context_precision}


def _extract_json_obj(text: str) -> str:
    """从一段文本里抽出最后一个 {...} JSON 对象。"""
    text = text.strip()
    if "```" in text:
        # 优先提取 ```json ... ``` 代码块内的内容
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1].lstrip("json").strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        return text[start:end]
    return text


def score_response(client: OpenAI, question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """先尝试 LLM 打分，失败则降级为向量相似度打分。

    DEEPSEEK_MODEL=deepseek-v4-flash 是推理模型，会先输出 reasoning_content 再输出
    content。max_tokens 太小会让推理过程吃光预算，content 为空（finish_reason='length'）。
    这里给到 1500 token 留足空间，并在 content 仍为空时从 reasoning_content 中兜底解析 JSON。
    """
    ctx_text = "\n\n".join(f"[{i+1}] {c[:300]}" for i, c in enumerate(contexts))
    prompt = _EVAL_USER_TMPL.format(question=question, ctx_text=ctx_text, answer=answer)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=cfg.DEEPSEEK_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1500,
            )
            choice = resp.choices[0]
            raw = choice.message.content or ""
            finish = getattr(choice, "finish_reason", None)

            # content 为空时尝试从 reasoning_content（推理模型独有字段）兜底
            if not raw.strip():
                rc = getattr(choice.message, "reasoning_content", None) or ""
                logger.warning(
                    f"LLM score attempt {attempt+1}: empty content "
                    f"(finish_reason={finish!r}, reasoning_len={len(rc)})"
                )
                if rc and "{" in rc and "}" in rc:
                    raw = rc
                else:
                    raise ValueError(f"Empty response from LLM (finish_reason={finish!r})")

            raw = _extract_json_obj(raw)
            return json.loads(raw)
        except Exception as exc:
            logger.warning(f"LLM score attempt {attempt+1} failed: {exc}")
            if attempt < 2:
                time.sleep(1)

    # 降级：LLM 打分全部失败，改用向量相似度
    logger.warning("LLM scoring failed 3 times, falling back to embedding-based scoring")
    try:
        return _score_by_embedding(question, answer, contexts)
    except Exception as exc:
        logger.warning(f"Embedding scoring also failed: {exc}")
        return {"answer_relevancy": 0.0, "faithfulness": 0.0, "context_precision": 0.0}


def evaluate_version(version: str, testset: list, limit: int, client: OpenAI) -> Dict[str, Any]:
    flags = VERSION_FLAGS[version]
    apply_flags(flags)
    logger.info(f"\n=== {version} | flags={flags} ===")

    samples = testset[:limit]
    records = []
    for i, item in enumerate(samples):
        q = item["question"]
        gt = item["ground_truth"]
        logger.info(f"  [{i+1}/{len(samples)}] {q[:60]}")

        try:
            result = run_single(client, q, gt)
            scores = score_response(client, q, result["answer"], result["contexts"])
            records.append({
                "question": q,
                "ground_truth": gt,
                "answer": result["answer"],
                "contexts": result["contexts"],
                **scores,
            })
        except Exception as exc:
            logger.warning(f"  Failed: {exc}")
            records.append({"question": q, "ground_truth": gt, "error": str(exc),
                            "answer_relevancy": 0.0, "faithfulness": 0.0, "context_precision": 0.0})
        time.sleep(0.3)

    metrics = ["answer_relevancy", "faithfulness", "context_precision"]
    avg = {m: round(sum(r.get(m, 0) for r in records) / max(len(records), 1), 4) for m in metrics}
    return {"version": version, "flags": flags, "avg": avg, "records": records}


def print_comparison_table(summaries: List[Dict]):
    print("\n" + "=" * 70)
    print(f"{'Version':<8} {'BM25':<6} {'RRF':<6} {'Rerank':<8} {'Rewrite':<8} "
          f"{'Relevancy':>10} {'Faithful':>10} {'Precision':>10}")
    print("-" * 70)
    for s in summaries:
        f = s["flags"]
        a = s["avg"]
        print(f"{s['version']:<8} {str(f.get('ENABLE_BM25','')):<6} {str(f.get('ENABLE_RRF','')):<6} "
              f"{str(f.get('ENABLE_RERANKER','')):<8} {str(f.get('ENABLE_QUERY_REWRITE','')):<8} "
              f"{a['answer_relevancy']:>10.4f} {a['faithfulness']:>10.4f} {a['context_precision']:>10.4f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--versions", default="V1,V2,V3,V4,V5,V6", help="Comma-separated versions to run")
    parser.add_argument("--limit", type=int, default=30, help="Max test questions per version")
    parser.add_argument("--output", default=str(RESULTS_DIR / "comparison.json"))
    args = parser.parse_args()

    versions = [v.strip() for v in args.versions.split(",")]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(TESTSET_PATH, encoding="utf-8") as f:
        testset = json.load(f)
    logger.info(f"Loaded {len(testset)} test questions, running {len(versions)} versions")

    client = OpenAI(api_key=cfg.DEEPSEEK_API_KEY, base_url=cfg.DEEPSEEK_BASE_URL)
    summaries = []

    for version in versions:
        if version not in VERSION_FLAGS:
            logger.warning(f"Unknown version {version}, skipping")
            continue
        result = evaluate_version(version, testset, args.limit, client)
        summaries.append(result)

        version_path = RESULTS_DIR / f"{version}.json"
        with open(version_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {version} results → {version_path}")

    # Save combined comparison
    comparison = {"summaries": [{"version": s["version"], "flags": s["flags"], "avg": s["avg"]} for s in summaries]}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    print_comparison_table(summaries)
    logger.info(f"Comparison table saved → {args.output}")


if __name__ == "__main__":
    main()
