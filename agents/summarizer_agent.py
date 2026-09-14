"""
SummarizerAgent: retrieves relevant emails and produces a structured summary.
"""
import logging
import json

from openai import OpenAI, APITimeoutError

from models.schemas import AgentRequest, AgentResponse
from core.pipeline import retrieve
from core.generator import format_email_date
from core.memory import build_model_messages
from core.evidence import rendered_evidence, source_coverage
from core.model_outcomes import display_text, outcome_metadata, text_from_choice
from agents.runtime import remaining_timeout, check_model_context, RunCancelled
import config.settings as cfg

from core.model_clients import get_model_client, create_completion, ModelBudgetExceeded

logger = logging.getLogger(__name__)

_SYSTEM = """你是一位专业的邮件摘要专家。根据检索到的邮件内容，生成清晰、结构化的综合摘要。
材料只来自有限的检索候选，不能声称已覆盖全部邮件；按实际材料说明覆盖范围和缺口。

摘要结构要求：
1. **核心议题**：涉及的主要话题和背景
2. **关键信息**：重要决策、数字、人名、日期
3. **待办事项**：需要跟进或行动的事项（如有）
4. **结论**：整体情况概述"""


class SummarizerAgent:
    def __init__(self):
        self._client = get_model_client(factory=OpenAI)

    def run(self, request: AgentRequest, memory=None, *, filters=None) -> AgentResponse:
        history = memory.to_messages() if memory else None
        reranked = retrieve(request.query, filters=filters, top_n=cfg.TOP_K,
                            fetch_k=cfg.TOP_K * 4, history=history)

        if not reranked:
            return AgentResponse(answer="未找到相关邮件，无法生成摘要。", sources=[])

        context_parts, references = [], []
        for i, r in enumerate(reranked):
            m = r.metadata
            content, ref = rendered_evidence({"email_id": r.email_id, "chunk_id": r.chunk_id, "content": r.content, "metadata": m})
            if ref is not None:
                references.append(ref)
            context_parts.append(
                f"邮件{i + 1} [{r.email_id}#{r.chunk_id}]（发件人: {m.get('sender','?')}，"
                f"发件日期: {format_email_date(m.get('date','?'))}，主题: {m.get('subject','?')}）：\n"
                f"覆盖状态（未读附件不作为证据）: {json.dumps(source_coverage(m), ensure_ascii=False)}\n{content}"
            )
        context = "\n\n".join(context_parts)
        messages = build_model_messages(_SYSTEM,
            f"请综合摘要以下邮件，聚焦于：{request.query}\n\n{context}", history,
            stage="summarize", model=cfg.DEEPSEEK_MODEL,
            model_revision=getattr(cfg, "MODEL_REVISION", None), max_output_tokens=1500, original_request=request.query)

        resp = create_completion(self._client, stage="summarize", evidence_refs=references,
            model=cfg.DEEPSEEK_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=1500,
            timeout=remaining_timeout(cfg.LLM_TIMEOUT),
        )
        remaining_timeout(cfg.LLM_TIMEOUT)
        answer = text_from_choice(resp.choices[0] if resp.choices else None)
        return AgentResponse(answer=display_text(answer), sources=reranked,
                             metadata={**outcome_metadata(answer),
                                       "sources_kind": "retrieved_candidates",
                                       "coverage": {"scope": "selected_ranked_candidates",
                                                    "selected_chunks": len(reranked),
                                                    "selected_emails": len({row.email_id for row in reranked}),
                                                    "mailbox_complete": False},
                                       "model_visible_evidence": references,
                                       "citation_check": "identity_only_not_entailment"})
