"""
Agent-level evaluation — runs multi-step tasks through the agent loop and
measures task success rate, tool-call accuracy and average steps.

Complements scripts/run_ragas_eval.py: that scores the *retrieval pipeline*
(answer_relevancy / faithfulness / context_precision); this scores the
*agent* — does it pick the right tools and actually complete the task.

Metrics:
  - task_success_rate     : fraction the LLM judge marks as completed
  - tool_accuracy         : fraction where every expected tool was used
  - avg_steps             : mean tool calls per task
  - max_steps_reached_rate: fraction that hit AGENT_MAX_STEPS (health signal)

Usage:
  python scripts/run_agent_eval.py [--limit N] [--output path]
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
import config.settings as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TESTSET_PATH = Path(__file__).parent.parent / "data" / "agent_testset.json"
RESULTS_DIR = Path(__file__).parent.parent / "data" / "eval_results"


def tool_accuracy(expected: List[str], actual: List[str]) -> bool:
    """True if every expected tool was actually used by the agent."""
    return set(expected or []).issubset(set(actual or []))


_JUDGE_PROMPT = """你是 AI agent 评测专家。判断 agent 是否成功完成了用户任务。

【任务】{task}

【agent 的最终回答】
{answer}

评判标准：success=1 表示回答切题、确实完成了任务；success=0 表示答非所问、
跑偏、信息明显错误或未完成。只返回 JSON，例如：{{"success": 1, "reason": "简短理由"}}"""


def judge_success(client: OpenAI, task: str, answer: str) -> Dict[str, Any]:
    """LLM-as-judge: did the agent complete the task? Returns {success, reason}."""
    try:
        resp = client.chat.completions.create(
            model=cfg.AGENT_PLANNER_MODEL,
            messages=[{"role": "user", "content": _JUDGE_PROMPT.format(task=task, answer=answer)}],
            temperature=0,
            max_tokens=500,
            timeout=cfg.LLM_TIMEOUT,
        )
        raw = (resp.choices[0].message.content or "").strip()
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()
        s, e = raw.find("{"), raw.rfind("}") + 1
        if s >= 0 and e > s:
            raw = raw[s:e]
        data = json.loads(raw)
        return {"success": int(data.get("success", 0)), "reason": str(data.get("reason", ""))}
    except Exception as exc:
        logger.warning(f"judge failed: {exc}")
        return {"success": 0, "reason": f"judge error: {exc}"}


def evaluate_task(item: Dict[str, Any], client: OpenAI) -> Dict[str, Any]:
    """Run one task through the agent loop and score it."""
    from agents.agent_loop import run_agent_loop
    from models.schemas import AgentRequest

    task = item["task"]
    expected = item.get("expected_tools", [])
    resp = run_agent_loop(AgentRequest(query=task))
    actual = [s["tool"] for s in resp.metadata.get("steps", [])]
    verdict = judge_success(client, task, resp.answer)
    return {
        "task": task,
        "expected_tools": expected,
        "actual_tools": actual,
        "tool_accuracy": tool_accuracy(expected, actual),
        "n_steps": len(actual),
        "max_steps_reached": bool(resp.metadata.get("max_steps_reached", False)),
        "success": verdict["success"],
        "reason": verdict["reason"],
        "answer": resp.answer,
    }


def aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records) or 1
    return {
        "n_tasks": len(records),
        "task_success_rate": round(sum(r["success"] for r in records) / n, 4),
        "tool_accuracy": round(sum(1 for r in records if r["tool_accuracy"]) / n, 4),
        "avg_steps": round(sum(r["n_steps"] for r in records) / n, 2),
        "max_steps_reached_rate": round(sum(1 for r in records if r["max_steps_reached"]) / n, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Max tasks to run")
    parser.add_argument("--output", default=str(RESULTS_DIR / "agent_eval.json"))
    args = parser.parse_args()

    with open(TESTSET_PATH, encoding="utf-8") as f:
        testset = json.load(f)
    if args.limit:
        testset = testset[: args.limit]
    logger.info(f"Running {len(testset)} agent tasks")

    client = OpenAI(api_key=cfg.DEEPSEEK_API_KEY, base_url=cfg.DEEPSEEK_BASE_URL)
    records: List[Dict[str, Any]] = []
    for i, item in enumerate(testset):
        logger.info(f"[{i + 1}/{len(testset)}] {item['task'][:50]}")
        try:
            records.append(evaluate_task(item, client))
        except Exception as exc:
            logger.warning(f"  task failed: {exc}")
            records.append({
                "task": item["task"], "error": str(exc), "success": 0,
                "tool_accuracy": False, "n_steps": 0, "max_steps_reached": False,
                "actual_tools": [], "expected_tools": item.get("expected_tools", []),
            })
        time.sleep(0.3)

    summary = aggregate(records)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": records}, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("Agent evaluation")
    print("-" * 60)
    print(f"  tasks                 : {summary['n_tasks']}")
    print(f"  task success rate     : {summary['task_success_rate']:.0%}")
    print(f"  tool-call accuracy    : {summary['tool_accuracy']:.0%}")
    print(f"  avg steps / task      : {summary['avg_steps']}")
    print(f"  max-steps-reached rate: {summary['max_steps_reached_rate']:.0%}")
    print("=" * 60)
    logger.info(f"saved → {args.output}")


if __name__ == "__main__":
    main()
