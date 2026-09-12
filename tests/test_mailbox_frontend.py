"""Run the real Streamlit app and inspect mailbox controls and safe text output."""
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


def button(app, label):
    return next(item for item in app.button if item.label == label)


def test_real_account_form_masks_code_and_submits_only_mailbox_endpoint():
    http = MailboxHTTP(configured=False, parsed=False)
    with patch('requests.get', side_effect=http.get), patch('requests.post', side_effect=http.post):
        app = st_testing.AppTest.from_file(str(APP), default_timeout=15).run()
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
        app = st_testing.AppTest.from_file(str(APP), default_timeout=15).run()
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


def test_real_report_renders_coverage_and_mail_html_only_as_disabled_text():
    http = MailboxHTTP()
    with patch('requests.get', side_effect=http.get), patch('requests.post', side_effect=http.post):
        app = st_testing.AppTest.from_file(str(APP), default_timeout=15).run()
        assert not app.exception
        values = {item.label:item.value for item in app.metric}
        assert {key:values[key] for key in ['已扫描邮件','解析成功','解析/下载失败','尚未下载']} == {
            '已扫描邮件':'9', '解析成功':'5', '解析/下载失败':'1', '尚未下载':'3'}
        assert any('不表示每个附件或字符都完整' in item.value for item in app.caption)
        body = next(item for item in app.text_area if item.label == '解析后的正文')
        attachment = next(item for item in app.text_area if item.label == '附件提取文本')
        assert body.value == attachment.value == HTML
        assert body.disabled and attachment.disabled
        assert not app.get('imgs')
        assert not app.get('iframe')
        assert all(HTML not in item.value for item in app.markdown)
        assert all('tracking.invalid' not in url for url, _ in http.gets)
        assert http.posts == []


def test_completed_sync_refreshes_report_and_stops_polling_without_resubmission():
    import streamlit as st
    http = MailboxHTTP()
    with patch('requests.get', side_effect=http.get), patch('requests.post', side_effect=http.post), \
         patch('streamlit.fragment', wraps=st.fragment) as fragment:
        app = st_testing.AppTest.from_file(str(APP), default_timeout=15).run()
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
        app=st_testing.AppTest.from_file(str(APP),default_timeout=15).run()
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
        app=st_testing.AppTest.from_file(str(APP),default_timeout=15).run()
        next(item for item in app.text_input if item.label=='搜索关键词').set_value('发票')
        button(app,'本地搜索').click().run()
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
        app=st_testing.AppTest.from_file(str(APP),default_timeout=15).run()
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
        app=st_testing.AppTest.from_file(str(APP),default_timeout=15)
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
        app=st_testing.AppTest.from_file(str(APP),default_timeout=15)
        app.session_state['imap_job_'+ACCOUNT['id']]='manual-old'
        app.session_state['imap_terminal_job_'+ACCOUNT['id']]='auto-new'
        app.run()
        assert not app.exception
        assert fragment.call_args.kwargs['run_every'] is None
        assert any('本次处理完成' in item.value for item in app.info)
        assert http.posts==[]
