"""Run the real Streamlit app and inspect mailbox controls and safe text output."""
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

st_testing = pytest.importorskip('streamlit.testing.v1')
APP = Path(__file__).resolve().parents[1] / 'frontend' / 'app.py'
ACCOUNT = {'id':'a'*32, 'address':'fixture@163.com', 'display_name':'离线邮箱',
           'provider':'163', 'credential_version':1}
HTML = '<script>alert("unsafe")</script><img src="https://tracking.invalid/pixel">'


def response(value):
    return SimpleNamespace(ok=True, status_code=200, json=lambda:value, raise_for_status=lambda:None)


class MailboxHTTP:
    def __init__(self, *, configured=True, parsed=True):
        self.accounts = [ACCOUNT] if configured else []
        self.parsed = parsed
        self.posts = []
        self.gets = []
        self.job_status = 'running'
        self.schedule={'enabled':False,'folders':['INBOX'],'interval_seconds':300,'max_messages':100,
                       'revision':0,'last_job_id':None,'worker_enabled':True,'state':'disabled'}

    def get(self, url, **kwargs):
        self.gets.append((url, kwargs))
        if url.endswith('/mailboxes'):
            return response({'accounts':self.accounts, 'local_only':True})
        if '/mailboxes/' in url:
            if url.endswith('/schedule'):
                return response(dict(self.schedule))
            if url.endswith('/search'):
                return response({'items':[{'message_key':'message-1','subject':'合成邮件','snippet':HTML}],
                                 'diagnostics':{'local_only':True}})
            if url.endswith('/report'):
                return response({'folders':[{'name':'INBOX', 'remote_count':9}], 'remote_snapshot_count':9,
                                 'parsed':5, 'failed':1, 'not_downloaded':3, 'body_empty':1,
                                 'decode_suspect':2, 'attachments':{'unsupported':1},
                                 'issues':{'attachment:unsupported_format':1}} if self.parsed else {'folders':[]})
            if url.endswith('/messages'):
                return response({'items':[{'key':'message-1', 'subject':'合成邮件', 'folder':'INBOX', 'uid':42}],
                                 'total':1, 'next_offset':None})
            return response({'email':{'sender':'sender@example.test', 'date':'2026-09-12', 'body':HTML,
                                     'source':{'provider':'163'}, 'decode_quality':{'status':'suspect'},
                                     'attachments':[{'filename':'fixture.html', 'status':'parsed', 'text':HTML,
                                                     'locations':[{'page':1}]}]}})
        if url.endswith('/health'):
            return response({'status':'ok'})
        if url.endswith('/index/status'):
            return response({'email_count':0, 'chunk_count':0})
        if '/jobs/' in url:
            return response({'id':'local-sync-job', 'kind':'imap_sync', 'status':self.job_status,
                             'progress':{'attempted':6,'parsed':5,'failed':1}})
        return response({'jobs':[], 'sessions':[], 'facts':[], 'turns':[], 'approvals':[]})

    def post(self, url, **kwargs):
        self.posts.append((url, kwargs.get('json')))
        if url.endswith('/mailboxes'):
            self.accounts = [ACCOUNT]
            return response(ACCOUNT)
        if url.endswith('/connect'):
            return response({'folders':[{'name':'INBOX', 'display_name':'收件箱', 'selectable':True},
                                        {'name':'Archive', 'display_name':'归档', 'selectable':True},
                                        {'name':'Parent', 'display_name':'不可选父目录', 'selectable':False}]})
        if url.endswith('/sync'):
            return response({'id':'local-sync-job', 'kind':'imap_sync', 'status':'queued'})
        if url.endswith('/schedule'):
            self.schedule.update(kwargs['json'],revision=self.schedule['revision']+1)
            self.schedule['state']='waiting' if self.schedule['enabled'] else 'disabled'
            return response(dict(self.schedule))
        raise AssertionError('Mailbox UI unexpectedly called another POST endpoint: '+url)


def _mail_app(workspace='我的邮箱'):
    app = st_testing.AppTest.from_file(str(APP), default_timeout=15)
    app.session_state['workspace_page'] = workspace
    return app


def button(app, label):
    return next(item for item in app.button if item.label == label)


def navigate(app, page):
    next(item for item in app.radio if item.label == '工作区').set_value(page).run()
    assert not app.exception


def test_chat_home_is_simple_and_mail_opens_only_when_requested():
    http = MailboxHTTP()
    with patch('requests.get',side_effect=http.get),patch('requests.post',side_effect=http.post):
        app = st_testing.AppTest.from_file(str(APP), default_timeout=15)
        # Stale flags from an earlier sync must not display global job logs here.
        app.session_state['job_submitted'] = True
        app.session_state['show_jobs'] = True
        app.run()
        assert not app.exception
        assert any(item.value == '助手对话' for item in app.title)
        assert len(app.chat_input) == 1 and not app.chat_input[0].disabled
        assert not any(url.endswith('/jobs') or url.endswith('/index/status') or url.endswith('/messages') for url,_ in http.gets)
        assert not any(item.label in {'邮件数据路径','搜索邮件'} for item in app.text_input)
        assert not any(item.label == '任务进度与恢复' for item in app.expander)
        button(app,'查看我的邮件').click().run()
        assert not app.exception
        assert not app.chat_input
        assert any(item.value == '我的邮箱' for item in app.title)
        assert not any('<div class="mail-body">' in item.value for item in app.markdown)
        button(app,'合成邮件').click().run()
        assert any('<div class="mail-body">' in item.value for item in app.markdown)
        navigate(app,'问答工作台')
        assert len(app.chat_input) == 1 and http.posts == []


CHAT_MODES = [
    ('普通（多 Agent 路由）', False, '/chat'),
    ('普通（多 Agent 路由）', True, '/chat/stream'),
    ('Self-RAG（反思工作流）', False, '/chat/graph'),
    ('Agent（自主工具调用）', False, '/jobs/agent'),
]


class StatusChatHTTP(MailboxHTTP):
    """Synthetic backend replies keep UI routing separate from status logic."""

    def __init__(self):
        super().__init__()
        self.jobs = {}

    def post(self, url, **kwargs):
        payload = kwargs['json']
        self.posts.append((url, payload))
        assert url.endswith(('/chat', '/chat/stream', '/chat/graph', '/jobs/agent'))
        count = 5 if payload['mailbox_account_id'] == ACCOUNT['id'] else 12
        result = {
            'answer': f'本地已解析 {count} 封。当前 AI 对话还不能读取这些真实邮件。',
            'intent': 'mailbox_status', 'sources': [],
            'metadata': {'session_id': payload['session_id'], 'local_only': True,
                         'mailbox_status': {'account_state': 'configured', 'parsed': count}},
        }
        if url.endswith('/jobs/agent'):
            identifier = f'status-job-{len(self.jobs)}'
            self.jobs[identifier] = {'id': identifier, 'kind': 'agent', 'status': 'succeeded',
                                     'result': result}
            return response({'id': identifier, 'kind': 'agent', 'status': 'queued'})
        if url.endswith('/chat/stream'):
            stream = response(result)
            stream.status_reply = result
            return nullcontext(stream)
        return response(result)

    def get(self, url, **kwargs):
        identifier = url.rsplit('/', 1)[-1]
        if '/jobs/' in url and identifier in self.jobs:
            self.gets.append((url, kwargs))
            return response(self.jobs[identifier])
        return super().get(url, **kwargs)


def status_sse_client(stream):
    result = stream.status_reply
    events = [{'intent': result['intent']}, {'token': result['answer']},
              {'sources': [], 'metadata': result['metadata']}]
    return SimpleNamespace(events=lambda: iter([
        *(SimpleNamespace(data=json.dumps(event)) for event in events),
        SimpleNamespace(data='[DONE]'),
    ]))


@pytest.mark.parametrize('mode,use_stream,endpoint', CHAT_MODES)
def test_status_chat_uses_backend_with_selected_account_and_no_mail_content(mode, use_stream, endpoint):
    http = StatusChatHTTP()
    second = {**ACCOUNT, 'id': 'b'*32, 'address': 'second@163.com', 'display_name': '第二个邮箱'}
    http.accounts.append(second)
    with patch('requests.get', side_effect=http.get), patch('requests.post', side_effect=http.post), \
         patch('sseclient.SSEClient', side_effect=status_sse_client):
        app = _mail_app('问答工作台').run()
        next(item for item in app.radio if item.label == '问答模式').set_value(mode).run()
        next(item for item in app.toggle if item.label == '流式输出').set_value(use_stream).run()
        session_id = app.session_state['session_id']
        for account, count, question in [
            (ACCOUNT, 5, '你现在连接上了163了吗？'),
            (second, 12, '你现在可以看到我163的邮件吗？'),
        ]:
            next(item for item in app.selectbox if item.label == '当前邮箱').set_value(account['id']).run()
            app.chat_input[0].set_value(question).run()
            assert not app.exception
            assert http.posts[-1][0].endswith(endpoint)
            payload = http.posts[-1][1]
            assert set(payload) == {'query', 'session_id', 'operation_key', 'mailbox_account_id'}
            assert payload['query'] == question
            assert payload['session_id'] == session_id
            assert payload['mailbox_account_id'] == account['id']
            assert payload['operation_key']
            answer = app.session_state['messages'][-1]
            assert answer['role'] == 'assistant'
            assert f'{count} 封' in answer['content'] and 'AI 对话还不能读取' in answer['content']
            assert answer['intent'] == 'mailbox_status'
            assert answer['sources'] == []
            assert 'local_status' not in answer
            app.run()
            assert not app.exception
        assert len(http.posts) == 2
        assert http.posts[0][1]['operation_key'] != http.posts[1][1]['operation_key']
        assert len(app.session_state['messages']) == 4
        assert not any('/messages' in url or url.endswith(('/report', '/search')) for url, _ in http.gets)
        sent = json.dumps(http.posts, ensure_ascii=False)
        assert all(value not in sent for value in [ACCOUNT['address'], second['address'], HTML, '合成邮件'])


@pytest.mark.parametrize('mode,use_stream,endpoint', CHAT_MODES)
def test_status_chat_backend_offline_does_not_invent_mailbox_connection_state(mode, use_stream, endpoint):
    import requests
    http = MailboxHTTP()

    def offline_post(url, **kwargs):
        http.posts.append((url, kwargs['json']))
        raise requests.exceptions.ConnectionError('offline fixture')

    with patch('requests.get', side_effect=http.get), patch('requests.post', side_effect=offline_post):
        app = _mail_app('问答工作台').run()
        next(item for item in app.radio if item.label == '问答模式').set_value(mode).run()
        next(item for item in app.toggle if item.label == '流式输出').set_value(use_stream).run()
        app.chat_input[0].set_value('你现在连接上了163了吗？').run()
        assert not app.exception
        assert len(http.posts) == 1 and http.posts[0][0].endswith(endpoint)
        assert http.posts[0][1]['mailbox_account_id'] == ACCOUNT['id']
        notices = ' '.join(item.value for item in [*app.error, *app.warning])
        assert '请求失败' in notices if use_stream else '无法连接到后端服务' in notices
        answer = app.session_state['messages'][-1]
        assert all(phrase not in answer['content'] for phrase in ['未配置邮箱', '没有连接163', '已连接163', '5 封'])
        assert not any('/messages' in url or url.endswith(('/report', '/search')) for url, _ in http.gets)
        if mode.startswith('Agent'):
            assert app.session_state['unconfirmed_submission'] == http.posts[0][1]


@pytest.mark.parametrize('configured', [True, False])
def test_embedded_chat_shares_conversation_without_adding_mail_context(configured):
    http = MailboxHTTP(configured=configured)
    def chat_post(url, **kwargs):
        http.posts.append((url, kwargs.get('json')))
        assert url.endswith('/chat')
        return response({'answer':'合成对话回复', 'intent':'general', 'sources':[]})

    with patch('requests.get', side_effect=http.get), patch('requests.post', side_effect=chat_post):
        app = _mail_app('问答工作台').run()
        assert not app.exception and http.posts == []
        if configured:
            navigate(app, '我的邮箱')
            button(app, '合成邮件').click().run()
            navigate(app, '问答工作台')
        app.chat_input[0].set_value('你好').run()
        assert not app.exception
        assert any(item.value == '合成对话回复' for item in app.markdown)
        assert len(http.posts) == 1
        payload = http.posts[0][1]
        expected_fields = {'query', 'session_id', 'operation_key'}
        if configured:
            expected_fields.add('mailbox_account_id')
            assert payload['mailbox_account_id'] == ACCOUNT['id']
        assert set(payload) == expected_fields
        assert payload['query'] == '你好'
        session_id = app.session_state['session_id']
        messages = list(app.session_state['messages'])
        button(app, '打开高级工具').click().run()
        assert not app.exception
        assert app.session_state['workspace_page'] == '问答工作台'
        assert app.session_state['session_id'] == session_id
        assert app.session_state['messages'] == messages
        navigate(app, '同步与设置')
        navigate(app, '问答工作台')
        button(app, '返回助手对话').click().run()
        assert app.session_state['session_id'] == session_id
        assert app.session_state['messages'] == messages
        assert len(http.posts) == 1
        button(app, '新会话').click().run()
        assert not app.exception
        assert app.session_state['session_id'] != session_id
        assert app.session_state['messages'] == []
        if configured:
            assert app.session_state['mail_selected_'+ACCOUNT['id']] == 'message-1'


def test_chat_mode_and_history_survive_account_and_workspace_switches():
    http = MailboxHTTP()
    second = {**ACCOUNT, 'id':'b'*32, 'address':'second@163.com', 'display_name':'第二个邮箱'}
    http.accounts.append(second)
    with patch('requests.get', side_effect=http.get), patch('requests.post', side_effect=http.post):
        app = _mail_app('问答工作台').run()
        mode = next(item for item in app.radio if item.label == '问答模式')
        mode.set_value('Agent（自主工具调用）').run()
        app.toggle[0].set_value(True).run()
        session_id = app.session_state['session_id']
        app.session_state['messages'] = [{'role':'user', 'content':'合成历史记录'}]
        next(item for item in app.selectbox if item.label == '当前邮箱').set_value(second['id']).run()
        navigate(app, '同步与设置')
        navigate(app, '问答工作台')
        app.run()
        assert next(item for item in app.radio if item.label == '问答模式').value == 'Agent（自主工具调用）'
        assert app.toggle[0].value is True
        assert app.session_state['session_id'] == session_id
        assert app.session_state['messages'] == [{'role':'user', 'content':'合成历史记录'}]
        assert http.posts == []


def test_empty_workspace_leads_to_account_setup_without_network_actions():
    http = MailboxHTTP(configured=False,parsed=False)
    with patch('requests.get',side_effect=http.get),patch('requests.post',side_effect=http.post):
        app = _mail_app().run()
        assert not app.exception
        button(app,'连接我的邮箱').click().run()
        assert not app.exception
        assert app.session_state['workspace_page'] == '同步与设置'
        assert any(item.label == '客户端授权码' for item in app.text_input)
        assert http.posts == []


def test_account_selection_and_search_state_survive_workspace_navigation():
    http = MailboxHTTP()
    second = {**ACCOUNT,'id':'b'*32,'address':'second@163.com','display_name':'第二个邮箱'}
    http.accounts.append(second)
    with patch('requests.get',side_effect=http.get),patch('requests.post',side_effect=http.post):
        app = _mail_app().run()
        next(item for item in app.text_input if item.label == '搜索邮件').set_value('发票')
        next(item for item in app.checkbox if item.label == '只搜索未读邮件').check()
        button(app,'搜索').click().run()
        next(item for item in app.selectbox if item.label == '当前邮箱').set_value(second['id']).run()
        assert not app.exception
        assert app.session_state['mail_account_id'] == second['id']
        assert not any(item.label == '返回全部邮件' for item in app.button)
        assert any(url.endswith('/'+second['id']+'/messages') for url,_ in http.gets)
        navigate(app,'同步与设置')
        navigate(app,'我的邮箱')
        assert next(item for item in app.selectbox if item.label == '当前邮箱').value == second['id']
        next(item for item in app.selectbox if item.label == '当前邮箱').set_value(ACCOUNT['id']).run()
        assert not app.exception
        assert any(item.label == '返回全部邮件' for item in app.button)
        assert app.session_state['imap_search_request_'+ACCOUNT['id']]['q'] == '发票'
        assert next(item for item in app.text_input if item.label == '搜索邮件').value == '发票'
        assert next(item for item in app.checkbox if item.label == '只搜索未读邮件').value is True
        button(app,'返回全部邮件').click().run()
        assert not app.exception
        assert next(item for item in app.text_input if item.label == '搜索邮件').value == ''
        assert next(item for item in app.checkbox if item.label == '只搜索未读邮件').value is False
        assert http.posts == []


def test_mail_subject_markdown_cannot_create_active_links_or_images():
    from frontend.mailbox_view import _plain_label
    assert _plain_label('![pixel](https://tracking.invalid/pixel)') == r'\!\[pixel\]\(https://tracking\.invalid/pixel\)'


def test_inbox_fetch_failure_is_not_rendered_as_empty_mailbox():
    http = MailboxHTTP()
    old_get = http.get
    def failed_get(url, **kwargs):
        if url.endswith('/messages'):
            raise RuntimeError('offline fixture')
        return old_get(url,**kwargs)
    with patch('requests.get',side_effect=failed_get),patch('requests.post',side_effect=http.post):
        app = _mail_app().run()
        assert not app.exception
        assert any('邮件列表暂时无法加载' in item.value for item in app.error)
        assert not any('这里还没有邮件' in item.value for item in app.info)


@pytest.mark.parametrize('suffix',['/mailboxes','/schedule','/report'])
def test_settings_read_failures_are_visible_and_do_not_invent_zero_counts(suffix):
    http = MailboxHTTP()
    old_get = http.get
    def failed_get(url, **kwargs):
        if url.endswith(suffix):
            raise RuntimeError('offline fixture')
        return old_get(url,**kwargs)
    with patch('requests.get',side_effect=failed_get),patch('requests.post',side_effect=http.post):
        app = _mail_app()
        app.session_state['workspace_page'] = '同步与设置'
        app.run()
        assert not app.exception
        assert any('暂时无法读取' in item.value for item in app.error)
        assert not app.metric
        assert http.posts == []


def test_remote_removal_moves_browse_cursor_back_to_valid_page():
    http = MailboxHTTP()
    old_get = http.get
    def get_page(url,**kwargs):
        if url.endswith('/messages'):
            http.gets.append((url,kwargs))
            if (kwargs.get('params') or {}).get('offset',0)>=20:
                return response({'items':[],'total':20,'next_offset':None})
        return old_get(url,**kwargs)
    with patch('requests.get',side_effect=get_page),patch('requests.post',side_effect=http.post):
        app = _mail_app()
        app.session_state['mail_page_'+ACCOUNT['id']] = 20
        app.run()
        assert not app.exception
        assert app.session_state['mail_page_'+ACCOUNT['id']] == 0
        assert any(item.label == '合成邮件' for item in app.button)
        assert not any('这里还没有邮件' in item.value for item in app.info)


def test_readable_table_display_uses_only_matching_metadata_and_escapes_cells():
    from frontend.mailbox_view import body_for_display
    text='[Table synthetic row synthetic:r1 status=complete] metadata'
    row={'start':2,'end':2+len(text),'text':text,'cells':[{'text':'<img src="https://invalid">','headers':['费用']}]}
    email={'body':'前言'+text+'结尾','table_rows':[row]}
    displayed=body_for_display(email)
    assert 'synthetic:r1' not in displayed and 'mail-table-row' in displayed
    assert '<img' not in displayed and '&lt;img' in displayed
    assert '费用' in displayed and '前言' in displayed and '结尾' in displayed
    assert email['body']=='前言'+text+'结尾'
    row['text']='mismatched metadata'
    assert 'synthetic:r1' in body_for_display(email)


def test_real_account_form_masks_code_and_submits_only_mailbox_endpoint():
    http = MailboxHTTP(configured=False, parsed=False)
    with patch('requests.get', side_effect=http.get), patch('requests.post', side_effect=http.post):
        app = _mail_app()
        app.session_state['workspace_page'] = '同步与设置'
        app.run()
        assert not app.exception
        address = next(item for item in app.text_input if item.label == '163 邮箱地址')
        code = next(item for item in app.text_input if item.label == '客户端授权码')
        assert code.proto.type == code.proto.PASSWORD
        address.set_value('fixture@163.com')
        code.set_value('synthetic-secret-only')
        button(app, '加密保存账号').click().run()
        assert not app.exception
        assert http.posts[0][0].endswith('/mailboxes')
        assert http.posts[0][1] == {'address':'fixture@163.com', 'authorization_code':'synthetic-secret-only',
                                  'display_name':''}
        rendered = [*app.markdown, *app.caption, *app.success, *app.error, *app.text]
        assert all('synthetic-secret-only' not in str(item.value) for item in rendered)
        assert any('账号已加密保存' in item.value for item in app.success)


def test_real_connection_and_sync_controls_submit_fixed_account_scope():
    http = MailboxHTTP(parsed=False)
    with patch('requests.get', side_effect=http.get), patch('requests.post', side_effect=http.post):
        app = _mail_app()
        app.session_state['workspace_page'] = '同步与设置'
        app.run()
        button(app, '测试连接并读取文件夹').click().run()
        assert not app.exception
        folders = next(item for item in app.multiselect if item.label == '同步文件夹')
        assert folders.options == ['收件箱', '归档']
        assert folders.value == ['INBOX','Archive']
        button(app, '开始本地同步与解析').click().run()
        assert not app.exception
        assert http.posts[-1][0].endswith('/mailboxes/'+ACCOUNT['id']+'/sync')
        request = http.posts[-1][1]
        assert request['folders'] == ['INBOX','Archive']
        assert request['max_messages'] == 100
        assert request['retry_failed'] is False
        assert request['operation_key'].startswith('imap-')
        assert 'authorization_code' not in request
        assert app.session_state['imap_job_'+ACCOUNT['id']] == 'local-sync-job'
        assert any('正在同步' in item.value for item in app.info)


def test_real_report_and_mail_html_are_displayed_safely_on_separate_pages():
    from html import escape
    http = MailboxHTTP()
    with patch('requests.get', side_effect=http.get), patch('requests.post', side_effect=http.post):
        app = _mail_app().run()
        assert not app.exception
        assert not app.chat_input
        button(app, '合成邮件').click().run()
        assert any('mail-body' in item.value and escape(HTML) in item.value for item in app.markdown)
        attachment = next(item for item in app.text_area if item.label == '附件提取文本')
        assert attachment.value == HTML and attachment.disabled
        assert not app.get('imgs') and not app.get('iframe')
        assert all(HTML not in item.value for item in app.markdown)
        assert all('tracking.invalid' not in url for url, _ in http.gets)
        assert http.posts == []
        next(item for item in app.radio if item.label == '工作区').set_value('同步与设置').run()
        values = {item.label:item.value for item in app.metric}
        assert {key:values[key] for key in ['已扫描邮件','解析成功','解析/下载失败','尚未下载']} == {
            '已扫描邮件':'9', '解析成功':'5', '解析/下载失败':'1', '尚未下载':'3'}
        assert any('不表示每个附件或字符都完整' in item.value for item in app.caption)


def test_completed_sync_refreshes_report_and_stops_polling_without_resubmission():
    import streamlit as st
    http = MailboxHTTP()
    with patch('requests.get', side_effect=http.get), patch('requests.post', side_effect=http.post), \
         patch('streamlit.fragment', wraps=st.fragment) as fragment:
        app = _mail_app()
        app.session_state['workspace_page'] = '同步与设置'
        app.run()
        button(app, '测试连接并读取文件夹').click().run()
        button(app, '开始本地同步与解析').click().run()
        assert not app.exception
        assert fragment.call_args.kwargs['run_every'] == 2
        http.job_status = 'succeeded'
        app.run()
        assert not app.exception
        assert fragment.call_args.kwargs['run_every'] is None
        assert app.session_state['imap_terminal_job_'+ACCOUNT['id']] == 'local-sync-job'
        assert any('本次处理完成' in item.value for item in app.info)
        assert any('本次已处理 6 封' in item.value for item in app.caption)
        assert sum(url.endswith('/sync') for url, _ in http.posts) == 1
        assert any(item.label == '解析成功' and item.value == '5' for item in app.metric)


def test_auto_sync_configuration_and_pause_use_only_mailbox_routes():
    http=MailboxHTTP()
    with patch('requests.get',side_effect=http.get),patch('requests.post',side_effect=http.post):
        app=_mail_app()
        app.session_state['workspace_page']='同步与设置'
        app.run()
        button(app,'测试连接并读取文件夹').click().run()
        next(item for item in app.checkbox if item.label=='启用后台自动同步').check()
        button(app,'保存自动同步设置').click().run()
        assert not app.exception
        assert http.schedule['enabled'] and http.schedule['interval_seconds']==300
        assert http.schedule['folders']==['INBOX','Archive']
        button(app,'暂停自动同步').click().run()
        assert not app.exception
        assert not http.schedule['enabled']
        assert all('/mailboxes/' in url for url,_ in http.posts)


def test_local_search_uses_get_and_keeps_snippet_and_body_plain():
    http=MailboxHTTP()
    with patch('requests.get',side_effect=http.get),patch('requests.post',side_effect=http.post):
        app=_mail_app().run()
        next(item for item in app.text_input if item.label=='搜索邮件').set_value('发票')
        button(app,'搜索').click().run()
        assert not app.exception
        searches=[kwargs['params'] for url,kwargs in http.gets if url.endswith('/search')]
        assert searches and searches[-1]['q']=='发票'
        assert all(HTML not in item.value for item in app.markdown)
        assert all(item.disabled for item in app.text_area)
        assert http.posts==[]


def test_disabled_worker_warns_without_automatic_polling():
    import streamlit as st
    http=MailboxHTTP()
    http.schedule.update(enabled=True,worker_enabled=False,revision=1,state='waiting',next_due=1000)
    with patch('requests.get',side_effect=http.get),patch('requests.post',side_effect=http.post), \
         patch('streamlit.fragment',wraps=st.fragment) as fragment:
        app=_mail_app().run()
        assert not app.exception
        assert fragment.call_args.kwargs['run_every'] is None
        assert any('后台自动同步服务未启用' in item.value for item in app.warning)
        assert not any('后台自动同步已启用' in item.value for item in app.caption)
        assert http.posts==[]


class MultipleJobHTTP(MailboxHTTP):
    def __init__(self, *, worker_enabled=False, enabled=True, manual_status='running'):
        super().__init__()
        self.schedule.update(enabled=enabled,worker_enabled=worker_enabled,revision=1,
                             last_job_id='auto-new',state='waiting' if enabled else 'disabled')
        self.jobs={'manual-old':{'id':'manual-old','created':1,'status':manual_status},
                   'auto-new':{'id':'auto-new','created':2,'status':'succeeded'}}

    def get(self,url,**kwargs):
        key=url.rsplit('/',1)[-1]
        if '/jobs/' in url and key in self.jobs:
            self.gets.append((url,kwargs))
            return response(dict(self.jobs[key]))
        return super().get(url,**kwargs)


def test_disabled_worker_keeps_manual_running_job_refreshing_until_completion():
    import streamlit as st
    http=MultipleJobHTTP()
    with patch('requests.get',side_effect=http.get),patch('requests.post',side_effect=http.post), \
         patch('streamlit.fragment',wraps=st.fragment) as fragment:
        app=_mail_app()
        app.session_state['imap_job_'+ACCOUNT['id']]='manual-old'
        app.run()
        assert not app.exception
        assert fragment.call_args.kwargs['run_every']==2
        assert any('正在同步' in item.value for item in app.info)
        assert any('后台自动同步服务未启用' in item.value for item in app.warning)
        http.jobs['manual-old']['status']='succeeded'
        app.run()
        assert not app.exception
        assert fragment.call_args.kwargs['run_every'] is None
        assert set(app.session_state['imap_terminal_jobs_'+ACCOUNT['id']])=={'manual-old','auto-new'}
        assert http.posts==[]


def test_paused_schedule_with_old_manual_and_new_auto_terminal_jobs_stops_polling():
    import streamlit as st
    http=MultipleJobHTTP(worker_enabled=True,enabled=False,manual_status='succeeded')
    with patch('requests.get',side_effect=http.get),patch('requests.post',side_effect=http.post), \
         patch('streamlit.fragment',wraps=st.fragment) as fragment:
        app=_mail_app()
        app.session_state['imap_job_'+ACCOUNT['id']]='manual-old'
        app.session_state['imap_terminal_job_'+ACCOUNT['id']]='auto-new'
        app.run()
        assert not app.exception
        assert fragment.call_args.kwargs['run_every'] is None
        assert any('本次处理完成' in item.value for item in app.info)
        assert http.posts==[]
