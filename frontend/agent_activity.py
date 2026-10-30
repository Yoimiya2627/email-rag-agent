"""Display only the Agent job explicitly bound to the current conversation."""

_TERMINAL = {'succeeded', 'failed', 'cancelled', 'interrupted', 'incomplete', 'invalid'}


def render_agent_activity(st, get, post) -> bool:
    sid = st.session_state.get('session_id')
    job_id = (st.session_state.get('agent_job_by_session') or {}).get(sid)
    if not sid or not job_id:
        return False
    states = dict(st.session_state.get('agent_activity_states') or {})
    previous = states.get(job_id)

    def still_current():
        return (st.session_state.get('session_id') == sid and
                (st.session_state.get('agent_job_by_session') or {}).get(sid) == job_id)

    @st.fragment(run_every=None if previous in _TERMINAL else 2)
    def activity():
        if not still_current():
            return
        job = get('/jobs/' + job_id)
        # A fragment may finish after a navigation or session replacement.
        if not still_current():
            return
        if not isinstance(job, dict) or job.get('error'):
            st.warning('暂时无法读取任务状态，已保留当前任务。请稍后重试。')
            return
        result = job.get('result') or {}
        metadata = result.get('metadata') or {}
        if (job.get('id') != job_id or job.get('kind') != 'agent' or
                (metadata.get('session_id') and metadata['session_id'] != sid)):
            status = 'invalid'
        else:
            status = job.get('status')
        if status == 'succeeded' and isinstance(result.get('answer'), str) and result['answer'].strip():
            delivered = set(st.session_state.get('agent_activity_delivered') or [])
            messages = list(st.session_state.get('messages') or [])
            if job_id not in delivered:
                if not any(message.get('agent_job_id') == job_id for message in messages):
                    messages.append({'role': 'assistant', 'content': result['answer'],
                                     'intent': result.get('intent', ''), 'sources': result.get('sources') or [],
                                     'extra_metadata': metadata, 'agent_job_id': job_id})
                    st.session_state['messages'] = messages
                delivered.add(job_id)
                st.session_state['agent_activity_delivered'] = sorted(delivered)
                updated = dict(st.session_state.get('agent_activity_states') or {})
                updated[job_id] = status
                st.session_state['agent_activity_states'] = updated
                st.rerun(scope='app')
        updated = dict(st.session_state.get('agent_activity_states') or {})
        old_status = updated.get(job_id)
        updated[job_id] = status
        st.session_state['agent_activity_states'] = updated
        # Refresh both the timer and the outer composer's busy state.
        if (old_status in _TERMINAL) != (status in _TERMINAL):
            st.rerun(scope='app')
        if status in {'queued', 'running'}:
            st.caption('正在等待助手开始处理…' if status == 'queued' else '助手正在处理你的请求…')
            if job.get('cancel_requested'):
                st.info('停止请求已送达，正在等待当前步骤结束。')
            elif st.button('停止本次任务', key='agent_activity_cancel_' + job_id):
                response = post('/jobs/' + job_id + '/cancel')
                if not response or response.get('error'):
                    st.warning('停止请求未确认，请稍后核对任务状态。')
                else:
                    st.info('停止请求已送达，正在等待当前步骤结束。')
        elif status != 'succeeded' or not isinstance(result.get('answer'), str) or not result['answer'].strip():
            labels = {'failed': '任务失败', 'cancelled': '任务已停止', 'interrupted': '任务被中断',
                      'incomplete': '任务尚未完成', 'succeeded': '任务结束，但没有返回回答',
                      'invalid': '任务与当前会话不匹配'}
            st.warning(labels.get(status, '暂时无法确认任务状态') + '。请到高级工具查看详情或恢复任务。')

    activity()
    if not still_current():
        return False
    return (st.session_state.get('agent_activity_states') or {}).get(job_id) not in _TERMINAL
