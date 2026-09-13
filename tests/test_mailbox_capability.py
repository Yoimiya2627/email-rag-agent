"""Recognize access questions without swallowing concrete or quoted tasks."""
import pytest

from agents.mailbox_status_agent import is_mailbox_status_question as is_mailbox_access_question
from agents.mailbox_status_agent import direct_mailbox_status_response, status_candidate
from core.mailbox_status import format_mailbox_status
from models.schemas import AgentRequest, IntentType


@pytest.mark.parametrize('query', [
    '你现在可以看到我163的邮件吗？', '你能看到我的邮箱吗', '你能看我的163邮箱吗？',
    '你能不能读取我的邮件？', '现在可以访问QQ邮箱吗？', '请问，你可以查看我的邮箱里的邮件吗？',
    '你已经可以读到我的邮件了吗', 'Can you see my 163 emails?', 'can you access my inbox yet?',
    '你现在连接上了163了吗？', '你现在连接上163了吗？', '163邮箱连接成功了吗',
    '我的邮箱连上了吗？', '你接入我的163邮箱了吗', '我的邮箱有多少封邮件？',
    '同步了多少封邮件？', '邮箱同步状态怎么样？', '同步完成了吗？',
    'Are you connected to my 163 yet?',
])
def test_access_questions(query):
    assert is_mailbox_access_question(query)


@pytest.mark.parametrize('query', [
    '查找163邮箱中关于合同的邮件', '帮我总结已导入的项目邮件', '你好',
    '你能看到我的邮箱吗？如果可以，请删除所有邮件。',
    '搜索主题为“你能看到我的邮箱吗”的邮件', '你能看到我的屏幕吗？',
])
def test_concrete_tasks_and_quotes_are_not_rewritten(query):
    assert not is_mailbox_access_question(query)
    assert not status_candidate(query)


def test_unknown_state_never_claims_connection_or_invents_zero_messages():
    assert '暂时无法读取' in format_mailbox_status({'account_state':'unavailable'})
    assert '尚未找到' in format_mailbox_status({'account_state':'not_configured'})
    answer = format_mailbox_status({'account_state':'configured','local_sync':{'state':'unavailable'}})
    assert '不能确认邮件数量' in answer
    assert '0 封' not in answer and '还不能读取' in answer


def test_unfamiliar_status_phrase_uses_only_question_for_classification(monkeypatch):
    from unittest.mock import Mock
    import agents.coordinator as coordinator
    import agents.mailbox_status_agent as status
    query = '想了解一下163邮箱目前的接入情况'
    assert status_candidate(query) and not is_mailbox_access_question(query)
    classifier = Mock(return_value=IntentType.MAILBOX_STATUS)
    snapshot = Mock(return_value={'account_state':'not_configured'})
    monkeypatch.setattr(coordinator, 'classify_intent', classifier)
    monkeypatch.setattr(status, 'read_mailbox_status', snapshot)
    result = direct_mailbox_status_response(AgentRequest(query=query, mailbox_account_id='a'*32), 'owner-a')
    assert result.intent == IntentType.MAILBOX_STATUS
    classifier.assert_called_once_with(query)
    snapshot.assert_called_once_with('owner-a', 'a'*32, '163')


def test_nonstatus_semantic_result_keeps_original_task(monkeypatch):
    monkeypatch.setattr('agents.coordinator.classify_intent', lambda query:IntentType.GENERAL)
    assert direct_mailbox_status_response(AgentRequest(query='介绍一下邮件连接协议'), 'owner') is None
