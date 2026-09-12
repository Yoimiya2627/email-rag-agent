"""Run the real Streamlit context controls against deterministic HTTP fixtures."""
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import pytest

st_testing=pytest.importorskip('streamlit.testing.v1')
APP=Path(__file__).resolve().parents[1]/'frontend/app.py'


def response(value):
    return NS(ok=True,status_code=200,json=lambda:value,raise_for_status=lambda:None)


def test_context_task_switch_and_constraint_revoke_use_displayed_versions():
    seen=[]
    state={'revision':7,'task_id':'a','task_state':{'task_id':'a','goal':'订单 A-7','objects':['A-7']},
           'user_events':[{'event_type':'current_user_correction','text':'预算改为8000'}],
           'candidates':[],'summary':None}
    facts=[{'key':'budget','value':'8000','version':3,'source_turn_id':'t','task_id':'a','scope':'task'}]
    def get(url,**kwargs):
        if url.endswith('/health'):return response({'status':'ok'})
        if url.endswith('/index/status'):return response({'email_count':0,'chunk_count':0})
        if url.endswith('/chat/context'):return response(state)
        if url.endswith('/chat/tasks'):return response({'tasks':[{'task_id':'a'},{'task_id':'b'}]})
        if url.endswith('/chat/facts'):return response({'facts':facts})
        if url.endswith('/chat/history'):return response({'turns':[{'turn_id':'t','query':'预算8000','answer':'ok'}]})
        return response({'jobs':[],'sessions':[],'turns':[]})
    def post(url,**kwargs):
        seen.append((url,kwargs.get('json')))
        if url.endswith('/chat/tasks/select'):
            state.update(revision=8,task_id='b',task_state={'task_id':'b','goal':'订单 B-8'})
        if url.endswith('/chat/facts/revoke'):facts.clear()
        return response({'status':'success'})
    with patch('requests.get',side_effect=get),patch('requests.post',side_effect=post):
        app=st_testing.AppTest.from_file(str(APP),default_timeout=15)
        app.session_state['session_id']='s'
        app.run()
        next(box for box in app.checkbox if box.label=='查看当前任务与记忆').check().run()
        assert not app.exception
        assert any('预算改为8000' in item.value for item in app.info)
        next(box for box in app.checkbox if box.label=='设置或切换任务').check().run()
        next(box for box in app.selectbox if box.label=='已有任务').select('b').run()
        next(button for button in app.button if button.label=='切换到所选任务').click().run()
        assert next(body for url,body in seen if url.endswith('/chat/tasks/select'))=={'session_id':'s','task_id':'b','expected_revision':7}
        next(box for box in app.checkbox if box.label=='管理显式任务约束').check().run()
        next(button for button in app.button if button.label=='撤销约束').click().run()
        assert not app.exception
        assert next(body for url,body in seen if url.endswith('/chat/facts/revoke'))['expected_version']==3


def test_history_window_original_read_and_continuation_survive_rerun():
    seen=[]
    def get(url,**kwargs):
        params=kwargs.get('params') or {};seen.append((url,params))
        if url.endswith('/health'):return response({'status':'ok'})
        if url.endswith('/index/status'):return response({'email_count':0,'chunk_count':0})
        if url.endswith('/chat/history/search'):
            return response({'turns':[{'turn_id':'t','query':'Q-728','answer':'背景'*500+'不含税8000',
                'metadata':{'status':'partial'},'hits':[{'text':'Q-728不含税8000'}]}]})
        if url.endswith('/chat/history/turn'):
            return response({'text':'原文后半段' if params.get('offset') else '原文第一段',
                'has_more':not params.get('offset'),'next_offset':1200})
        return response({'jobs':[],'sessions':[],'turns':[]})
    with patch('requests.get',side_effect=get):
        app=st_testing.AppTest.from_file(str(APP),default_timeout=15)
        app.session_state['session_id']='s';app.run()
        next(item for item in app.text_input if item.label=='历史关键词').set_value('Q-728').run()
        next(button for button in app.button if button.label=='搜索这段会话').click().run()
        assert any('不含税8000' in item.value for item in app.text)
        next(item for item in app.selectbox if item.label=='展开原文').select('answer').run()
        next(button for button in app.button if button.label=='读取这一轮原文').click().run()
        next(button for button in app.button if button.label=='下一段历史原文').click().run()
        assert not app.exception
        assert any(item.value=='原文后半段' for item in app.text)
        assert any(url.endswith('/chat/history/turn') and params.get('offset')==1200 for url,params in seen)
