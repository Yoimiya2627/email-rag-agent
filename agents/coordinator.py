"""
Coordinator: uses LLM intent recognition (not keyword matching) to route
requests to the appropriate specialist agent.
"""
import json
import logging

from openai import OpenAI, APITimeoutError

from models.schemas import AgentRequest, AgentResponse, IntentType
import config.settings as cfg
from core.memory import build_model_messages
from agents.runtime import remaining_timeout, RunCancelled, ContextBudgetExceeded

from core.model_clients import get_model_client, create_completion, ModelBudgetExceeded

logger = logging.getLogger(__name__)

_client = None

_INTENT_SYSTEM = """你是一个邮件助手的任务协调器，负责分析用户输入并判断其意图类型。

意图类型及判断标准：
- retrieve：用户想查找/搜索特定邮件、信息或内容（如"找一下关于X的邮件"、"有没有提到Y"）
- summarize：用户想获得邮件摘要或综述（如"总结最近邮件"、"X话题都讨论了什么"）
- write_reply：用户想要回复邮件或起草回信（如"帮我回复这封邮件"、"写一封拒绝邮件"）
- analyze：用户想要统计或数据分析（如"谁发邮件最多"、"标签分布"、"每日邮件量"）
- general：其他一般性问题

返回格式（严格JSON）：{"intent": "<类型>", "reason": "<简短判断理由>"}"""


def _get_client() -> OpenAI:
    global _client
    _client = get_model_client(legacy=_client, factory=OpenAI)
    return _client


def classify_intent(query: str, history=None) -> IntentType:
    messages = build_model_messages(_INTENT_SYSTEM, query, history, stage="intent", model=cfg.DEEPSEEK_MODEL, model_revision=getattr(cfg, "MODEL_REVISION", None), max_output_tokens=1500)
    try:
        resp = create_completion(_get_client(), stage="intent",
            model=cfg.DEEPSEEK_MODEL,
            messages=messages,
            temperature=0,
            # 推理模型预留推理 + 答案空间，避免 content 为空
            max_tokens=1500,
            timeout=remaining_timeout(cfg.LLM_TIMEOUT),
        )
        choice = resp.choices[0]
        raw = (choice.message.content or "").strip()
        if not raw:
            rc = getattr(choice.message, "reasoning_content", None) or ""
            if rc and "{" in rc and "}" in rc:
                raw = rc
            else:
                raise ValueError(f"Empty intent response (finish_reason={choice.finish_reason!r})")
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()
        s, e = raw.find("{"), raw.rfind("}") + 1
        if s >= 0 and e > s:
            raw = raw[s:e]
        data = json.loads(raw)
        return IntentType(data["intent"])
    except (TimeoutError, APITimeoutError, RunCancelled, ModelBudgetExceeded, ContextBudgetExceeded):
        raise
    except Exception as exc:
        logger.warning("Intent classification failed; error_type=%s; defaulting to general", type(exc).__name__)
        return IntentType.GENERAL


def route(request: AgentRequest, memory=None, *, intent: IntentType | None = None) -> AgentResponse:
    from agents.retriever_agent import RetrieverAgent
    from agents.summarizer_agent import SummarizerAgent
    from agents.writer_agent import WriterAgent
    from agents.analyzer_agent import AnalyzerAgent

    history = memory.to_messages() if memory else None
    if intent is None:
        intent = classify_intent(request.query, history=history) if history else classify_intent(request.query)
    else:
        intent = IntentType(intent)
    logger.info("Intent=%s", intent.value)

    agent_map = {
        IntentType.RETRIEVE: RetrieverAgent,
        IntentType.SUMMARIZE: SummarizerAgent,
        IntentType.WRITE_REPLY: WriterAgent,
        IntentType.ANALYZE: AnalyzerAgent,
        IntentType.GENERAL: RetrieverAgent,
    }
    agent = agent_map[intent]()
    response = agent.run(request, memory=memory)
    response.intent = intent
    return response
