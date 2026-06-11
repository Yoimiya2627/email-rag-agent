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

from agents.evalops import classify_eval_record, events_by_trace_id, write_eval_report
from agents.tracing import load_events
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
    forbidden = item.get("forbidden_tools", [])
    resp = run_agent_loop(AgentRequest(query=task))
    actual = [s["tool"] for s in resp.metadata.get("steps", [])]
    verdict = judge_success(client, task, resp.answer)
    record = {
        "id": item.get("id", ""),
        "task": task,
        "task_type": item.get("task_type", "general"),
        "risk_level": item.get("risk_level", "low"),
        "success_criteria": item.get("success_criteria", ""),
        "expected_tools": expected,
        "forbidden_tools": forbidden,
        "actual_tools": actual,
        "tool_accuracy": tool_accuracy(expected, actual),
        "forbidden_tool_violation": bool(set(forbidden or []) & set(actual or [])),
        "n_steps": len(actual),
        "max_steps_reached": bool(resp.metadata.get("max_steps_reached", False)),
        "trace_id": resp.metadata.get("trace_id", ""),
        "success": verdict["success"],
        "reason": verdict["reason"],
        "answer": resp.answer,
    }
    record["failure_category"] = classify_eval_record(record, [])
    return record


def aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records) or 1
    return {
        "n_tasks": len(records),
        "task_success_rate": round(sum(r["success"] for r in records) / n, 4),
        "tool_accuracy": round(sum(1 for r in records if r["tool_accuracy"]) / n, 4),
        "avg_steps": round(sum(r["n_steps"] for r in records) / n, 2),
        "max_steps_reached_rate": round(sum(1 for r in records if r["max_steps_reached"]) / n, 4),
        "forbidden_tool_violation_rate": round(
            sum(1 for r in records if r.get("forbidden_tool_violation")) / n,
            4,
        ),
    }


def load_completed_records(output_path: str | Path) -> List[Dict[str, Any]]:
    """Load completed eval records from a previous checkpoint."""
    path = Path(output_path)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    records = payload.get("records", [])
    return records if isinstance(records, list) else []


def pending_items(
    testset: List[Dict[str, Any]],
    completed_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return testset items whose ids are not already present in completed records."""
    completed_ids = {str(record.get("id") or "") for record in completed_records if record.get("id")}
    return [item for item in testset if str(item.get("id") or "") not in completed_ids]


def sync_record_metadata(
    records: List[Dict[str, Any]],
    testset: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Refresh eval-record metadata from the current testset and recompute tool flags."""
    cases_by_id = {str(item.get("id") or ""): item for item in testset if item.get("id")}
    synced: List[Dict[str, Any]] = []
    for record in records:
        item = dict(record)
        case = cases_by_id.get(str(item.get("id") or ""))
        if case:
            for key in ("task_type", "risk_level", "success_criteria"):
                item[key] = case.get(key, item.get(key))
            item["expected_tools"] = list(case.get("expected_tools", []) or [])
            item["forbidden_tools"] = list(case.get("forbidden_tools", []) or [])
        actual = list(item.get("actual_tools", []) or [])
        expected = list(item.get("expected_tools", []) or [])
        forbidden = list(item.get("forbidden_tools", []) or [])
        item["tool_accuracy"] = tool_accuracy(expected, actual)
        item["forbidden_tool_violation"] = bool(set(forbidden) & set(actual))
        synced.append(item)
    return synced


def _payload_with_trace_categories(
    records: List[Dict[str, Any]],
    trace_events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    traces = events_by_trace_id(trace_events)
    enriched: List[Dict[str, Any]] = []
    for record in records:
        item = dict(record)
        if item.get("failure_category") != "exception":
            item["failure_category"] = classify_eval_record(
                item,
                traces.get(item.get("trace_id", ""), []),
            )
        enriched.append(item)
    return {"summary": aggregate(enriched), "records": enriched}


def write_eval_outputs(
    records: List[Dict[str, Any]],
    output_path: str | Path,
    report_output: str | Path | None = None,
    trace_input: str | Path | None = None,
) -> Dict[str, Any]:
    """Write eval JSON and optional Markdown report for the current checkpoint."""
    trace_events = load_events(trace_input) if trace_input else []
    payload = _payload_with_trace_categories(records, trace_events)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    if report_output:
        write_eval_report(payload, report_output, trace_events)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Max tasks to run")
    parser.add_argument("--output", default=str(RESULTS_DIR / "agent_eval.json"))
    parser.add_argument("--report-output", default=None, help="Optional Markdown EvalOps report path")
    parser.add_argument("--trace-input", default=None, help="Optional trace JSONL path for failure attribution")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output records")
    args = parser.parse_args()

    with open(TESTSET_PATH, encoding="utf-8") as f:
        testset = json.load(f)
    if args.limit:
        testset = testset[: args.limit]

    records: List[Dict[str, Any]] = load_completed_records(args.output) if args.resume else []
    if args.resume and records:
        records = sync_record_metadata(records, testset)
        testset = pending_items(testset, records)
        logger.info(f"Resuming from {len(records)} completed records")
    logger.info(f"Running {len(testset)} agent tasks")

    client = OpenAI(api_key=cfg.DEEPSEEK_API_KEY, base_url=cfg.DEEPSEEK_BASE_URL)
    for i, item in enumerate(testset):
        logger.info(f"[{i + 1}/{len(testset)}] {item['task'][:50]}")
        try:
            records.append(evaluate_task(item, client))
        except Exception as exc:
            logger.warning(f"  task failed: {exc}")
            records.append({
                "id": item.get("id", ""),
                "task": item["task"],
                "task_type": item.get("task_type", "general"),
                "risk_level": item.get("risk_level", "low"),
                "success_criteria": item.get("success_criteria", ""),
                "error": str(exc),
                "success": 0,
                "tool_accuracy": False, "n_steps": 0, "max_steps_reached": False,
                "actual_tools": [], "expected_tools": item.get("expected_tools", []),
                "forbidden_tools": item.get("forbidden_tools", []),
                "forbidden_tool_violation": False,
                "trace_id": "",
                "failure_category": "exception",
            })
        write_eval_outputs(records, args.output, args.report_output, args.trace_input)
        time.sleep(0.3)

    payload = write_eval_outputs(records, args.output, args.report_output, args.trace_input)
    summary = payload["summary"]

    print("\n" + "=" * 60)
    print("Agent evaluation")
    print("-" * 60)
    print(f"  tasks                 : {summary['n_tasks']}")
    print(f"  task success rate     : {summary['task_success_rate']:.0%}")
    print(f"  tool-call accuracy    : {summary['tool_accuracy']:.0%}")
    print(f"  avg steps / task      : {summary['avg_steps']}")
    print(f"  max-steps-reached rate: {summary['max_steps_reached_rate']:.0%}")
    print(f"  forbidden-tool rate   : {summary['forbidden_tool_violation_rate']:.0%}")
    print("=" * 60)
    logger.info(f"saved → {args.output}")


if __name__ == "__main__":
    main()
