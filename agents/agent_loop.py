"""
Agent loop — a function-calling, ReAct-style loop over the email tools.

The planner LLM (deepseek-chat by default; see docs/agent_loop_decisions.md
Step 0) is called with TOOL_SCHEMAS.  While it returns tool_calls we execute
them and feed the results back as tool messages; when it returns plain content
that content is the final answer.

Guardrails here are deliberately minimal — a hard AGENT_MAX_STEPS cap.  Loop
detection, argument validation and tool-error feedback are added in Step 5.
"""
import json
import logging
from typing import List

from openai import OpenAI

from agents.tools import TOOL_SCHEMAS, call_tool
from models.schemas import AgentRequest, AgentResponse
import config.settings as cfg

logger = logging.getLogger(__name__)

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=cfg.DEEPSEEK_API_KEY, base_url=cfg.DEEPSEEK_BASE_URL)
    return _client


_SYSTEM = """你是一个邮件智能助手 Agent，可以调用工具完成用户的任务。

可用工具：检索邮件、获取邮件详情、主题摘要、起草回信、邮件统计。

工作方式：
- 先判断完成任务需要哪些步骤，再逐步调用工具；多步任务需要多轮工具调用。
- 每拿到一次工具结果，判断信息是否已经足够。
- 信息足够后，用中文给出清晰、有依据的最终回答，此时不要再调用工具。"""


def _serialize_tool_calls(tool_calls) -> list:
    """Re-encode SDK tool_call objects as plain dicts to append back to messages."""
    return [
        {
            "id": tc.id,
            "type": "function",
            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
        }
        for tc in tool_calls
    ]


def run_agent_loop(request: AgentRequest, memory=None) -> AgentResponse:
    """Run the function-calling agent loop and return the final answer.

    metadata['steps'] records every tool invocation (name + arguments) for
    transparency and agent-level evaluation (Step 6).
    """
    messages = [{"role": "system", "content": _SYSTEM}]
    if memory is not None:
        messages.extend(memory.to_messages())
    messages.append({"role": "user", "content": request.query})

    steps: List[dict] = []
    client = _get_client()

    for step in range(cfg.AGENT_MAX_STEPS):
        resp = client.chat.completions.create(
            model=cfg.AGENT_PLANNER_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            temperature=0,
            max_tokens=1500,
            timeout=cfg.LLM_TIMEOUT,
        )
        msg = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        if not tool_calls:
            answer = (msg.content or "").strip() or "未能生成回答。"
            return AgentResponse(answer=answer, sources=[], metadata={"steps": steps})

        # Echo the assistant turn (with its tool_calls) back into the history,
        # then append one tool-role message per call — required protocol order.
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": _serialize_tool_calls(tool_calls),
        })
        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                # Malformed arguments degrade to an empty call rather than crash;
                # Step 5 will feed the error back to the model explicitly.
                logger.warning(f"[agent] malformed arguments for {name}: {tc.function.arguments!r}")
                args = {}
            logger.info(f"[agent step {step + 1}] tool={name} args={args}")
            result = call_tool(name, args)
            steps.append({"tool": name, "arguments": args})
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False, default=str),
            })

    # AGENT_MAX_STEPS exhausted — force a final answer with tools disabled so
    # the model must synthesize from whatever it has gathered.
    logger.warning(f"Agent hit AGENT_MAX_STEPS={cfg.AGENT_MAX_STEPS}, forcing final answer")
    final = client.chat.completions.create(
        model=cfg.AGENT_PLANNER_MODEL,
        messages=messages + [{
            "role": "user",
            "content": "已达到最大步数，请基于已获取的信息直接给出最终回答，不要再调用工具。",
        }],
        temperature=0,
        max_tokens=1500,
        timeout=cfg.LLM_TIMEOUT,
    )
    answer = (final.choices[0].message.content or "").strip() or "未能在限定步数内完成任务。"
    return AgentResponse(
        answer=answer,
        sources=[],
        metadata={"steps": steps, "max_steps_reached": True},
    )
