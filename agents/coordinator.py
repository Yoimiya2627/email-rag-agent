"""
Coordinator: uses LLM intent recognition (not keyword matching) to route
requests to the appropriate specialist agent.
"""
import json
import logging

from openai import OpenAI

from models.schemas import AgentRequest, AgentResponse, IntentType
import config.settings as cfg

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
    if _client is None:
        _client = OpenAI(api_key=cfg.DEEPSEEK_API_KEY, base_url=cfg.DEEPSEEK_BASE_URL)
    return _client


def classify_intent(query: str) -> IntentType:
    try:
        resp = _get_client().chat.completions.create(
            model=cfg.DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": _INTENT_SYSTEM},
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=128,
        )
        raw = resp.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()
        data = json.loads(raw)
        return IntentType(data["intent"])
    except Exception as exc:
        logger.warning(f"Intent classification failed: {exc}, defaulting to general")
        return IntentType.GENERAL


def route(request: AgentRequest, memory=None) -> AgentResponse:
    from agents.retriever_agent import RetrieverAgent
    from agents.summarizer_agent import SummarizerAgent
    from agents.writer_agent import WriterAgent
    from agents.analyzer_agent import AnalyzerAgent

    intent = classify_intent(request.query)
    logger.info(f"Intent={intent.value!r} | query={request.query[:60]!r}")

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
