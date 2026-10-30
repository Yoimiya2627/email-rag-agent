"""
Step 0 pre-flight probe for the agent-loop upgrade.

Answers two questions before any agent code is written:

  1. Does DeepSeek function calling (the OpenAI `tools` param) reliably return
     `tool_calls`?  The agent loop is built entirely on this — if the
     configured reasoning model does not support it, the loop must use a
     different model for planning.  Tested on both the configured model and
     `deepseek-chat`.

  2. How long does one real LLM call actually take?  `LLM_TIMEOUT` is 15s in
     config but `core/generator.py` claims the reasoning model spends 30-70s
     on `reasoning_content` alone.  An agent loop makes many calls, so a wrong
     timeout means every step degrades.  This probe measures the truth.

Run:  .venv\\Scripts\\python.exe scripts\\probe_function_calling.py
This spends a small amount of API tokens (~6 short calls).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
import config.settings as cfg

# Generous timeout so we MEASURE true latency instead of being cut off by the
# production LLM_TIMEOUT (currently 15s, the very value under question).
PROBE_TIMEOUT = 180

_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_emails",
        "description": "在邮件库中检索与查询相关的邮件，返回匹配的邮件片段。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "检索关键词或自然语言问题",
                },
            },
            "required": ["query"],
        },
    },
}


def _client() -> OpenAI:
    return OpenAI(api_key=cfg.DEEPSEEK_API_KEY, base_url=cfg.DEEPSEEK_BASE_URL)


def probe_latency(model: str) -> float | None:
    """Time one plain chat call with a realistic-size RAG prompt."""
    prompt = (
        "请根据以下邮件内容回答问题。\n\n"
        + ("【邮件正文】这是一封关于 Q3 预算评审会议安排的邮件，涉及多个部门的预算分配。" * 20)
        + "\n\n问题：这封邮件的核心议题是什么？"
    )
    t0 = time.time()
    try:
        resp = _client().chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1500,
            timeout=PROBE_TIMEOUT,
        )
    except Exception as exc:
        print(f"  [{model}] latency probe ERROR: {exc}")
        return None
    elapsed = time.time() - t0
    choice = resp.choices[0]
    content = choice.message.content or ""
    rc = getattr(choice.message, "reasoning_content", None) or ""
    print(
        f"  [{model}] latency={elapsed:.1f}s  finish={choice.finish_reason!r}  "
        f"content_len={len(content)}  reasoning_len={len(rc)}"
    )
    return elapsed


def probe_function_calling(model: str) -> bool:
    """Send a tools=[...] request and check whether tool_calls comes back."""
    t0 = time.time()
    try:
        resp = _client().chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "帮我找一下关于 Q3 预算评审会议的邮件"},
            ],
            tools=[_SEARCH_TOOL],
            temperature=0,
            max_tokens=1500,
            timeout=PROBE_TIMEOUT,
        )
    except Exception as exc:
        print(f"  [{model}] function calling ERROR: {exc}")
        return False
    elapsed = time.time() - t0
    msg = resp.choices[0].message
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        for tc in tool_calls:
            print(
                f"  [{model}] tool_call OK ({elapsed:.1f}s): "
                f"{tc.function.name}({tc.function.arguments})"
            )
        return True
    print(
        f"  [{model}] NO tool_calls ({elapsed:.1f}s)  "
        f"finish={resp.choices[0].finish_reason!r}  content={msg.content!r}"
    )
    return False


def main():
    if not cfg.DEEPSEEK_API_KEY:
        print("DEEPSEEK_API_KEY not set — fill .env first.")
        sys.exit(1)

    configured = cfg.DEEPSEEK_MODEL
    candidates = [configured]
    if "deepseek-chat" not in candidates:
        candidates.append("deepseek-chat")

    print(f"\n=== Config ===")
    print(f"  base_url        = {cfg.DEEPSEEK_BASE_URL}")
    print(f"  configured model= {configured}")
    print(f"  LLM_TIMEOUT     = {cfg.LLM_TIMEOUT}s  (current production value)")

    print(f"\n=== 1. Latency (plain chat, realistic prompt) ===")
    latencies = {m: probe_latency(m) for m in candidates}

    print(f"\n=== 2. Function calling (tools param → tool_calls) ===")
    fc_support = {m: probe_function_calling(m) for m in candidates}

    print(f"\n=== Verdict ===")
    for m in candidates:
        lat = latencies.get(m)
        lat_s = f"{lat:.1f}s" if lat is not None else "n/a"
        fc = "YES" if fc_support.get(m) else "NO"
        print(f"  {m:24s} function_calling={fc:4s} latency={lat_s}")

    # Recommendation
    print(f"\n=== Recommendation ===")
    max_lat = max((v for v in latencies.values() if v is not None), default=None)
    if max_lat is not None:
        suggested = int(max_lat * 2) + 5
        verdict = "OK" if cfg.LLM_TIMEOUT >= max_lat else "TOO LOW — calls will time out"
        print(f"  LLM_TIMEOUT: current {cfg.LLM_TIMEOUT}s vs measured max {max_lat:.1f}s → {verdict}")
        print(f"               suggest LLM_TIMEOUT >= {suggested}s (2x measured max + margin)")
    planning_model = next(
        (m for m in candidates if fc_support.get(m)), None
    )
    if planning_model:
        print(f"  Agent planning model: use {planning_model!r} (function calling confirmed working)")
    else:
        print(f"  WARNING: no tested model returned tool_calls — agent loop needs a")
        print(f"           manual JSON-protocol fallback instead of native function calling.")


if __name__ == "__main__":
    main()
