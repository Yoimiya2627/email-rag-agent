"""Non-retrieval replies for conversation and requests needing clarification."""
import unicodedata

from core.model_outcomes import outcome_metadata
from models.schemas import AgentRequest, AgentResponse, IntentType
from core.mail_capabilities import GREETING, CHAT_SCOPE_EXPLANATION


_GREETING = GREETING
_THANKS = "不客气！需要查询同步状态或对导入资料提问时，告诉我就可以。"
_HELP = (
    "我可以对已导入的邮件进行搜索、摘要、统计和回复草稿撰写，并提供原文核对。"
    "例如：找一下关于某个项目的邮件、总结某个话题，或帮我起草回复。"
    "163 邮箱面板支持后台只读同步、正文和附件解析，以及独立的本地关键词搜索，具体覆盖情况可查看解析报告。"
    + CHAT_SCOPE_EXPLANATION + "自动移动归档邮件尚未支持。"
)
_CLARIFY = (
    "请具体说明你想查询的邮箱状态，或想对哪些导入资料进行问答、总结和草稿撰写。"
    "我还没有根据这条消息检索或修改邮件。" + CHAT_SCOPE_EXPLANATION
)
_REPLIES = {
    **dict.fromkeys(("你好", "您好", "你好呀", "你好啊", "嗨", "哈喽", "hello", "hi", "hey",
                     "早上好", "上午好", "中午好", "下午好", "晚上好", "在吗", "在么"), _GREETING),
    **dict.fromkeys(("谢谢", "谢谢你", "多谢", "感谢", "谢谢啦", "thanks", "thank you"), _THANKS),
    **dict.fromkeys(("帮助", "使用帮助", "你是谁", "你能做什么", "你会做什么", "你有什么功能",
                     "有哪些功能", "怎么使用", "怎么用", "help", "what can you do"), _HELP),
}


def _direct_text(query: str):
    # Match the complete utterance, never a greeting prefix in a mail request.
    value = unicodedata.normalize("NFKC", query).strip().casefold()
    value = value.strip(" \t\r\n!?.,。！？…~～")
    return _REPLIES.get(value)


def _response(answer: str) -> AgentResponse:
    return AgentResponse(intent=IntentType.GENERAL, answer=answer, sources=[],
                         metadata={**outcome_metadata(answer), "retrieval_performed": False,
                                   "actual_tool_calls": 0, "steps": []})


def direct_general_response(query: str) -> AgentResponse | None:
    """A small, exact fast path; other inputs still use intent/planner routing."""
    answer = _direct_text(query)
    return _response(answer) if answer is not None else None


class GeneralAgent:
    def run(self, request: AgentRequest, memory=None) -> AgentResponse:
        # Historical email excerpts are unnecessary for greetings or asking the
        # user to specify a task. Do not send them to a model on this path.
        return direct_general_response(request.query) or _response(_CLARIFY)
