"""Quick debug: probe DeepSeek with the eval scoring prompt and dump the raw response."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
import config.settings as cfg

PROMPT = """你是RAG系统评测专家。请给下面的检索问答打分。

【问题】上周关于Q3预算的会议纪要发件人是谁？

【检索到的上下文片段】
[1] 主题: Q3预算评审会议纪要 / 发件人: alice@example.com / 日期: 2024-09-12
会议讨论了Q3各部门的预算分配，市场部追加50万，研发部维持原案。
[2] 主题: Re: Q3预算 / 发件人: bob@example.com
请参考上周纪要。

【系统生成的答案】
上周Q3预算评审会议纪要的发件人是 alice@example.com（2024-09-12 发送）。

请根据以下三个维度打分（每个维度0.00到1.00之间的小数）：
1. answer_relevancy：答案与问题的相关程度（1.0=完全切题）
2. faithfulness：答案是否有上下文依据（1.0=完全有据可查）
3. context_precision：上下文中真正有用的片段比例（1.0=全部有用）

请直接输出JSON，例如：
{"answer_relevancy": 0.85, "faithfulness": 0.90, "context_precision": 0.75}"""


def probe(model: str, max_tokens: int = 256, **extra):
    print(f"\n{'='*70}\nMODEL={model} max_tokens={max_tokens} extra={extra}\n{'='*70}")
    client = OpenAI(api_key=cfg.DEEPSEEK_API_KEY, base_url=cfg.DEEPSEEK_BASE_URL)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
            temperature=0.1,
            max_tokens=max_tokens,
            **extra,
        )
    except Exception as exc:
        print(f"!! API call raised: {type(exc).__name__}: {exc}")
        return
    choice = resp.choices[0]
    msg = choice.message
    print(f"finish_reason = {choice.finish_reason!r}")
    print(f"content       = {msg.content!r}")
    rc = getattr(msg, "reasoning_content", None)
    if rc:
        print(f"reasoning_content (first 500) = {rc[:500]!r}")
    if resp.usage:
        print(f"usage = prompt={resp.usage.prompt_tokens} completion={resp.usage.completion_tokens} total={resp.usage.total_tokens}")
    try:
        print(f"raw model_dump (truncated) = {json.dumps(resp.model_dump(), ensure_ascii=False)[:800]}")
    except Exception:
        pass


if __name__ == "__main__":
    print(f"Configured DEEPSEEK_MODEL = {cfg.DEEPSEEK_MODEL!r}")
    print(f"Configured DEEPSEEK_BASE_URL = {cfg.DEEPSEEK_BASE_URL!r}")
    probe(cfg.DEEPSEEK_MODEL, max_tokens=1500)
    probe(cfg.DEEPSEEK_MODEL, max_tokens=2048)
    probe(cfg.DEEPSEEK_MODEL, max_tokens=1500, response_format={"type": "json_object"})
