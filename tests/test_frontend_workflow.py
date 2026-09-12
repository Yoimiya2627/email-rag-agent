"""Exercise the actual Streamlit script with HTTP replaced, not the UI helpers."""
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest

st_testing = pytest.importorskip('streamlit.testing.v1')


def response(value):
    return NS(ok=True,status_code=200,json=lambda:value,raise_for_status=lambda:None)


def get(url,**kwargs):
    if url.endswith('/health'):
        return response({'status':'ok'})
    if url.endswith('/index/status'):
        return response({'email_count':0,'chunk_count':0})
    if url.endswith('/jobs'):
        return response({'jobs':[]})
    return response({'sessions':[],'facts':[],'turns':[],'approvals':[]})


@pytest.mark.parametrize('advanced', [False, True])
def test_normal_chat_renders_response_without_uninitialized_result(advanced):
    path=Path(__file__).resolve().parents[1]/'frontend'/'app.py'
    with patch('requests.get',side_effect=get),patch('requests.post',return_value=response({
        'answer':'synthetic answer','intent':'general','sources':[],
        'metadata':{'status':'incomplete','completion_status':'incomplete','finish_reason':'length'}})):
        app=st_testing.AppTest.from_file(str(path),default_timeout=15)
        app.session_state['workspace_page']='问答工作台'
        app.session_state['advanced_chat']=advanced
        app.run()
        assert not app.exception
        app.chat_input[0].set_value('hello').run()
        assert not app.exception
        assert any('synthetic answer' in element.value for element in app.markdown)
        assert any('incomplete' in str(element.value) for element in app.warning)


@pytest.mark.parametrize('advanced', [False, True])
def test_agent_submission_can_enable_jobs_after_sidebar_widgets_exist(advanced):
    path=Path(__file__).resolve().parents[1]/'frontend'/'app.py'
    with patch('requests.get',side_effect=get),patch('requests.post',return_value=response({
        'id':'job-synthetic','kind':'agent','status':'queued','resumable':False})):
        app=st_testing.AppTest.from_file(str(path),default_timeout=15)
        app.session_state['workspace_page']='问答工作台'
        app.session_state['advanced_chat']=advanced
        app.run()
        # The mode control belongs to the sidebar; use its actual options.
        control=next(item for item in app.radio if any(str(option).startswith('Agent') for option in item.options))
        control.set_value(next(option for option in control.options if option.startswith('Agent'))).run()
        app.chat_input[0].set_value('draft a reply').run()
        assert not app.exception
        assert app.session_state['job_submitted'] is True


@pytest.mark.parametrize('action', ['检查上次任务状态', '使用原操作标识重试提交'])
def test_recovered_submission_returns_answer_to_the_original_conversation(action):
    path = Path(__file__).resolve().parents[1]/'frontend'/'app.py'
    pending = {'session_id':'recovered-session', 'operation_key':'original-operation', 'query':'合成请求'}
    job = {'id':'recovered-job', 'kind':'agent', 'status':'succeeded',
           'result':{'answer':'恢复后的合成回答', 'metadata':{'session_id':'recovered-session'}}}
    def job_get(url, **kwargs):
        if '/jobs/' in url:
            return response(job)
        return get(url, **kwargs)
    with patch('requests.get', side_effect=job_get), patch('requests.post', return_value=response(job)) as post:
        app = st_testing.AppTest.from_file(str(path), default_timeout=15)
        app.session_state['session_id'] = pending['session_id']
        app.session_state['messages'] = [{'role':'user', 'content':pending['query']}]
        app.session_state['unconfirmed_submission'] = pending
        app.run()
        assert not app.exception
        next(item for item in app.button if item.label == action).click().run()
        app.run()
        assert not app.exception
        assert app.session_state['agent_job_by_session'][pending['session_id']] == job['id']
        assert [item['content'] for item in app.session_state['messages']] == ['合成请求', '恢复后的合成回答']
        assert not app.chat_input[0].disabled
        if action == '检查上次任务状态':
            post.assert_not_called()
        else:
            post.assert_called_once()
            assert post.call_args.kwargs['json'] == pending


@pytest.mark.parametrize('binding, expected', [
    ({'provider': 'gmail', 'account_id': 'owner@example.test'}, '草稿所属邮箱：owner@example.test'),
    ({'provider': 'simulated'}, '执行方式：本地模拟'),
    ({}, '旧审批没有账号绑定'),
], ids=['gmail', 'simulated', 'legacy'])
def test_approval_displays_bound_destination_without_execution(binding, expected):
    path = Path(__file__).resolve().parents[1] / 'frontend' / 'app.py'
    def approvals_get(url, **kwargs):
        if url.endswith('/agent/approvals'):
            return response({'approvals': [{
                'approval_id': 'approval-ui', 'status': 'pending', 'payload_hash': 'synthetic',
                'payload': {'to': ['reader@example.test'], 'subject': 'Fixture', 'body': 'Draft',
                            'execution_binding': binding},
            }]})
        return get(url, **kwargs)
    with patch('requests.get', side_effect=approvals_get), patch('requests.post') as post:
        app = st_testing.AppTest.from_file(str(path), default_timeout=15)
        app.session_state['workspace_page'] = '问答工作台'
        app.session_state['advanced_chat'] = True
        app.run()
        next(box for box in app.checkbox if box.label == '加载审批待办').check().run()
        assert not app.exception
        assert any(expected in str(element.value) for element in [*app.markdown, *app.caption])
        post.assert_not_called()


def test_saved_sessions_and_jobs_can_reach_later_pages():
    path=Path(__file__).resolve().parents[1]/'frontend'/'app.py'
    requests_seen=[]
    def paged_get(url,**kwargs):
        params=kwargs.get('params') or {}
        requests_seen.append((url,dict(params)))
        offset=params.get('offset',0)
        if url.endswith('/chat/sessions'):
            return response({'sessions':[{'session_id':f's{n}','turn_count':1} for n in range(offset,min(offset+50,115))],
                             'next_offset':offset+50 if offset+50<115 else None})
        if url.endswith('/jobs'):
            return response({'jobs':[{'id':f'job{n}','kind':'agent','status':'succeeded'} for n in range(offset,min(offset+10,12))],
                             'next_offset':offset+10 if offset+10<12 else None})
        return get(url,**kwargs)
    with patch('requests.get',side_effect=paged_get):
        app=st_testing.AppTest.from_file(str(path),default_timeout=15)
        app.session_state['workspace_page']='问答工作台'
        app.session_state['advanced_chat'] = True
        app.run()
        next(box for box in app.checkbox if box.label=='查看已保存会话').check().run()
        next(button for button in app.button if button.label=='下一页会话').click().run()
        assert not app.exception
        assert next(box for box in app.selectbox if box.label=='选择会话').options[0].startswith('s50')
        next(button for button in app.button if button.label=='下一页会话').click().run()
        assert next(box for box in app.selectbox if box.label=='选择会话').options[0].startswith('s100')
        next(box for box in app.checkbox if box.label=='显示后台任务').check().run()
        next(button for button in app.button if button.label=='下一页任务').click().run()
        assert not app.exception
        assert any(url.endswith('/jobs') and params.get('offset')==10 for url,params in requests_seen)


def test_exact_evidence_reread_displays_original_and_table_context():
    from core.evidence import text_hash
    path=Path(__file__).resolve().parents[1]/'frontend'/'app.py'
    ref={'email_id':'e','chunk_id':'c','source_version':'g','visible_start':7,'visible_end':15,
         'visible_hash':text_hash('evidence')}
    seen=[]
    def post(url,**kwargs):
        seen.append((url,kwargs.get('json')))
        if '/evidence/reread' in url:
            return response({'body':'evidence','read_start':7,'read_end':15,'has_more':False,
                'verification_complete':True,'verified_start':7,'verified_end':15,
                'chunks':[{'table_context':'net total; partial_row=true'}]})
        return response({'answer':'Supported [e#c]','sources':[],
                         'metadata':{'status':'success','model_visible_evidence':[ref],'cited_evidence':[ref]}})
    with patch('requests.get',side_effect=get),patch('requests.post',side_effect=post):
        app=st_testing.AppTest.from_file(str(path),default_timeout=15)
        app.session_state['workspace_page']='问答工作台'
        app.session_state['advanced_chat'] = True
        app.run()
        app.chat_input[0].set_value('fixture').run()
        # Once persisted in UI history, its stable widget key owns the read.
        app.run()
        next(button for button in app.button if button.label=='核验并回读原文').click().run()
        assert not app.exception
        assert any(element.value=='evidence' for element in app.text)
        assert any('partial_row=true' in element.value for element in app.text)
        assert next(body for url,body in seen if '/evidence/reread' in url)==ref
