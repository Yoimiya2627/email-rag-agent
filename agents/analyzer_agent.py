"""
AnalyzerAgent: computes statistical summaries (top senders, label distribution,
daily volume) and uses LLM to interpret results in natural language.

`compute_email_stats()` is a module-level function so the agent tool layer
(agents/tools.py) can reuse the exact same stats computation without reaching
into the agent.
"""
import json
import logging
from collections import Counter
from typing import Any, Dict

from openai import OpenAI, APITimeoutError

from models.schemas import AgentRequest, AgentResponse
from core.embedder import get_all_metadata as get_all_chunks
from agents.runtime import remaining_timeout, RunCancelled, ContextBudgetExceeded
from core.memory import build_model_messages
from core.model_outcomes import display_text, outcome_metadata, text_from_choice
import config.settings as cfg

from core.model_clients import get_model_client, create_completion, ModelBudgetExceeded

logger = logging.getLogger(__name__)


def compute_email_stats() -> Dict[str, Any]:
    """Deterministic whole-index statistics; natural-language scope is not inferred."""
    from core.evidence_pages import compute_scoped_stats
    return compute_scoped_stats(get_all_chunks())


class AnalyzerAgent:
    def __init__(self):
        self._client = get_model_client(factory=OpenAI)

    def run(self, request: AgentRequest, memory=None) -> AgentResponse:
        remaining_timeout(cfg.LLM_TIMEOUT)
        stats = compute_email_stats()
        stats_json = json.dumps(stats, ensure_ascii=False, indent=2)

        history = memory.to_messages() if memory is not None else None
        messages = build_model_messages(
            "你是邮件数据分析专家。根据统计数据，用清晰易懂的语言回答用户的分析问题，"
            "只按 coverage 中的明确范围解释；未解析用户自然语言范围，不能把全索引统计冒充指定期间或指定集合。"
            "存在未读附件、解码疑点或同步不完整时明确说明，日期较新也不自动代表结论仍有效。",
            f"邮件统计数据如下：\n{stats_json}\n\n用户问题：{request.query}", history, stage="analyze", model=cfg.DEEPSEEK_MODEL, model_revision=getattr(cfg, "MODEL_REVISION", None), max_output_tokens=1000, original_request=request.query)
        resp = create_completion(self._client, stage="analyze",
            model=cfg.DEEPSEEK_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=1000,
            timeout=remaining_timeout(cfg.LLM_TIMEOUT),
        )
        remaining_timeout(cfg.LLM_TIMEOUT)
        answer = text_from_choice(resp.choices[0] if resp.choices else None)
        return AgentResponse(
            answer=display_text(answer),
            sources=[],
            metadata={**stats, **outcome_metadata(answer)},
        )
