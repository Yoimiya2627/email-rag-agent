"""Backend routing for mailbox metadata questions across chat modes."""
import re
import unicodedata

from core.mailbox_status import read_mailbox_status, format_mailbox_status
from core.model_outcomes import outcome_metadata
from models.schemas import AgentResponse, IntentType


def _text(query):
    return unicodedata.normalize('NFKC', query).casefold().strip().strip('?!？！。.')


def status_candidate(query):
    """Guard semantic classification against quoted mail and compound actions."""
    text = _text(query)
    if len(text) > 200 or re.search(r'["\'“”‘’「」《》\n]|如果|然后|顺便|并且|再帮', text):
        return False
    if re.search(r'查找|搜索|总结|摘要|起草|回复|删除|移动|归档|发一封|发邮件|search|summari[sz]e|delete|draft|reply|archive', text):
        return False
    mailbox = re.search(r'163|qq|邮箱|邮件|收件箱|收信|同步|mail|inbox|sync', text)
    status = re.search(r'连接|连上|连好|连通|接入|接上|接好|接通|授权|同步|收取|下载|解析|读取|读到|看到|查看|访问|能读|能看|多少|几封|状态|情况|进度|工作|正常|connect|access|read|see|sync|how many|status', text)
    return bool(mailbox and status)


def is_mailbox_status_question(query):
    """Deterministic common questions; unfamiliar phrasing has a semantic route."""
    if not status_candidate(query):
        return False
    text = re.sub(r'\s+', '', _text(query))
    prefix = r'(?:(?:请问[,，]?|请|帮我|麻烦)?(?:你)?(?:现在|目前|已经|已)?(?:是否|有没有|能不能|可不可以|可以|能)?(?:直接)?)'
    provider = r'(?:(?:网易)?163|qq)'
    mailbox = r'(?:(?:我(?:的)?)?(?:' + provider + r'(?:的)?(?:邮箱|邮件)?|邮箱|邮件|收件箱)(?:里|里面|中)?(?:的(?:邮件|内容))?)'
    end = r'(?:了|上|好|成功|完成)?(?:了)?(?:吗|么|没|没有|了吗|了没|了没有|不)?'
    patterns = [
        prefix + r'(?:连接|连|接入|接上|接通|绑定)(?:上|好|到|成功)?(?:了)?' + mailbox + end,
        prefix + mailbox + r'(?:现在|目前|已经|是否|有没有)?(?:连接|连|接入|接上|接通|绑定)(?:上|好|成功)?' + end,
        prefix + r'(?:看到|看见|看得到|看|读取|读到|读|查看|访问)' + mailbox + end,
        prefix + mailbox + r'(?:现在|目前)?(?:有|已同步|同步|收取|下载|解析)(?:了)?(?:多少|几)(?:封)?(?:邮件)?' + end,
        prefix + r'(?:同步|收取|下载|解析)(?:了)?(?:多少|几)(?:封)?(?:邮件)?' + end,
        prefix + mailbox + r'?(?:的)?(?:连接|同步|收取|授权|接入)?(?:状态|情况|进度)(?:怎么样|如何|正常)?' + end,
        prefix + mailbox + r'?(?:同步|收取|下载)(?:好|完|完成|正常)?' + end,
    ]
    if any(re.fullmatch(pattern, text) for pattern in patterns):
        return True
    english = _text(query)
    return bool(re.fullmatch(
        r'(?:can|could) you (?:already |now |directly )?(?:see|read|access) my (?:(?:163|qq) )?(?:emails?|mailbox|inbox)(?: now| yet)?'
        r'|(?:are you |is my (?:163 |qq )?mailbox )?connected(?: to (?:my )?(?:163|qq|mailbox))?(?: yet| now)?'
        r'|how many (?:emails?|messages?) (?:have (?:you |been )?synced|are synced)', english))


def requested_provider(query):
    providers = set(re.findall(r'(?<![a-z0-9])(?:163|qq)(?![a-z0-9])', _text(query)))
    return next(iter(providers)) if len(providers) == 1 else ('multiple' if providers else None)


def mailbox_status_response(request, owner):
    snapshot = read_mailbox_status(owner, request.mailbox_account_id, requested_provider(request.query))
    answer = format_mailbox_status(snapshot)
    return AgentResponse(intent=IntentType.MAILBOX_STATUS, answer=answer, sources=[], metadata={
        **outcome_metadata(answer), 'retrieval_performed': False, 'actual_tool_calls': 0,
        'steps': [], 'local_only': True, 'mailbox_status': snapshot,
        # Displayable in history, but never reuse stale status or private counts
        # as model context, summaries or evidence in a later question.
        'exclude_from_model_context': True,
    })


def direct_mailbox_status_response(request, owner):
    if is_mailbox_status_question(request.query):
        return mailbox_status_response(request, owner)
    if status_candidate(request.query):
        from agents.coordinator import classify_intent
        # Classification receives only the user's question, not mailbox state,
        # credentials, mail content, or conversation history.
        if classify_intent(request.query) == IntentType.MAILBOX_STATUS:
            return mailbox_status_response(request, owner)
    return None


class MailboxStatusAgent:
    def run(self, request, memory=None):
        from agents.runtime import current_run
        run = current_run()
        if run is None:
            raise ValueError('Mailbox status requires an authenticated run context')
        return mailbox_status_response(request, run.owner_id)
