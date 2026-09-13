"""
Coordinator: handles common social/status turns locally and uses model intent
recognition for other requests before routing to a specialist agent.
"""
import json
import logging

from openai import OpenAI, APITimeoutError

from models.schemas import AgentRequest, AgentResponse, IntentType
import config.settings as cfg
from core.memory import build_model_messages
from agents.runtime import remaining_timeout, RunCancelled, ContextBudgetExceeded

from core.model_clients import get_model_client, create_completion, ModelBudgetExceeded
from core.model_outcomes import text_from_choice
from agents.general_agent import GeneralAgent, direct_general_response

logger = logging.getLogger(__name__)

_client = None

_INTENT_SYSTEM = """你是一个邮件助手的任务协调器，负责分析用户输入并判断其意图类型。

意图类型及判断标准：
- retrieve：用户想查找/搜索特定邮件、信息或内容（如"找一下关于X的邮件"、"有没有提到Y"）
- summarize：用户想获得邮件摘要或综述（如"总结最近邮件"、"X话题都讨论了什么"）
- write_reply：用户想要回复邮件或起草回信（如"帮我回复这封邮件"、"写一封拒绝邮件"）
- analyze：用户想要统计或数据分析（如"谁发邮件最多"、"标签分布"、"每日邮件量"）
- general：问候、致谢、助手使用说明，或尚未说明具体邮件任务、需要澄清的输入；不需要检索邮箱
- mailbox_status：询问邮箱接入、连接、授权配置、同步进度、已同步数量或助手能否读取真实邮件；这是应用状态，不是查找邮件正文。用户明确要求搜索、总结或修改具体邮件时，仍按实际任务分类。

只根据当前消息判断是否提出邮件任务。历史仅用于理解明确的追问，不要把独立的问候或致谢当成继续处理历史邮件。
问候与明确任务同时出现（如“你好，帮我找一下报价邮件”）时，按具体邮件任务分类。

返回格式（严格JSON）：{"intent": "<类型>", "reason": "<简短判断理由>"}"""


def _get_client() -> OpenAI:
    global _client
    _client = get_model_client(legacy=_client, factory=OpenAI)
    return _client


def classify_intent(query: str, history=None) -> IntentType:
    remaining_timeout(cfg.LLM_TIMEOUT)
    if direct_general_response(query) is not None:
        return IntentType.GENERAL
    from agents.mailbox_status_agent import is_mailbox_status_question
    if is_mailbox_status_question(query):
        return IntentType.MAILBOX_STATUS
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
        choice = text_from_choice(resp.choices[0] if resp.choices else None)
        if choice.completion_status != "complete":
            raise ValueError("Incomplete intent response")
        raw = str(choice)
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
    from agents.mailbox_status_agent import MailboxStatusAgent

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
        IntentType.GENERAL: GeneralAgent,
        IntentType.MAILBOX_STATUS: MailboxStatusAgent,
    }
    agent = agent_map[intent]()
    response = agent.run(request, memory=memory)
    response.intent = intent
    return response
