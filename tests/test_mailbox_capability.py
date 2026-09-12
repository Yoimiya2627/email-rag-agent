"""Recognize access questions without swallowing concrete or quoted tasks."""
import pytest

from frontend.mailbox_capability import is_mailbox_access_question, mailbox_access_answer


@pytest.mark.parametrize('query', [
    '你现在可以看到我163的邮件吗？', '你能看到我的邮箱吗', '你能看我的163邮箱吗？',
    '你能不能读取我的邮件？', '现在可以访问QQ邮箱吗？', '请问，你可以查看我的邮箱里的邮件吗？',
    '你已经可以读到我的邮件了吗', 'Can you see my 163 emails?', 'can you access my inbox yet?',
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


def test_unknown_state_never_claims_connection_or_invents_zero_messages():
    def no_read(*args):
        pytest.fail('No account, no report request')
    assert '暂时无法读取' in mailbox_access_answer(None, None, no_read)
    assert '没有选中已连接' in mailbox_access_answer({'accounts':[]}, None, no_read)
    answer = mailbox_access_answer({'accounts':[{'id':'a'}]}, 'a', lambda path:None)
    assert '无法确认本地同步数量' in answer
    assert '0 封' not in answer and '不能读取' in answer
