"""Human-readable context inspection. No model payloads or debug JSON by default."""


def render_context_metrics(st, metadata):
    material=(metadata or {}).get('session_context') or {}
    metrics=(metadata or {}).get('context_metrics') or {}
    if not material and not metrics:
        return
    with st.expander('本次回答使用了哪些上下文',expanded=False):
        st.caption(f"相关历史来源：{len(material.get('source_turn_ids',[]))} 轮 · "
                   f"未装入材料：{material.get('omitted_count',0)} 项")
        if material.get('required_omissions'):
            st.warning('必要要求超出预算，无法完整继续。请缩小本次任务或检查过长约束。')
        reason_labels={
            'purpose_budget':'本次用途的材料预算不足', 'required_material_over_budget':'必要要求超出预算',
            'duplicate':'重复材料', 'recent_window_duplicate':'近期对话已包含该来源',
            'current_request_duplicate':'当前请求已完整保留', 'purpose_or_represented_source':'本阶段无需重复装入',
            'inactive_constraint':'旧约束已失效', 'inactive_user_event':'旧修正已被替代或处理',
            'missing_source':'缺少原始来源', 'scope_mismatch':'来源不属于当前会话范围',
            'invalid_source_range':'来源位置无法核验', 'source_hash_mismatch':'原文内容已变化',
            'invalid_summary':'摘要格式不可用', 'invalid_summary_entry':'摘要条目无法核验',
            'invalid_user_event':'用户修正无法核验', 'summary_coverage_label_budget':'摘要覆盖说明超出预算',
            'active_user_event_char_limit':'有效修正过长，需整理约束',
            'active_user_event_count_limit':'有效修正过多，需整理约束'}
        reasons={reason_labels.get(item.get('reason'),'材料未通过当前选择规则')
                 for item in material.get('omissions',[]) if isinstance(item,dict)}
        if reasons:
            st.caption('遗漏原因：'+'、'.join(sorted(reasons)))
        summary=metrics.get('summary') or {}
        if summary.get('status'):
            labels={'generated':'已更新','published':'已更新','ready':'可用','reused':'复用有效摘要','cached':'复用有效摘要',
                    'failed':'生成失败，使用可用回退','attempt_limit':'已达本输入的尝试上限',
                    'not_due':'尚未达到更新轮数','below_threshold':'尚未达到更新轮数','empty':'暂无可摘要的轮次','skipped':'未触发更新',
                    'conflict':'期间来源变化，未发布','stale':'已失效','unavailable':'暂不可用'}
            st.caption('会话摘要状态：'+labels.get(summary['status'],'本次未发布新摘要'))
        if metrics.get('result_storage_degraded'):
            st.warning('工具结果暂未完整保存；请使用仍可用的原文读取入口。')
        if metrics.get('dropped_history_turns'):
            st.caption(f"预算裁减：移除 {metrics['dropped_history_turns']} 个旧问答对。")
        st.caption('历史回答与摘要供理解任务；引用邮件时仍需核验原文。')


def render_task_context(st,get,post,sid):
    if not st.checkbox('查看当前任务与记忆',key='context-inspect-'+sid):
        return
    state=get('/chat/context',params={'session_id':sid})
    if state is None:
        st.warning('任务记忆暂时无法加载。')
        return
    task=state.get('task_state') or {}
    st.write('当前任务：'+str(task.get('goal') or '尚未设置独立任务目标'))
    if task.get('objects'):
        st.caption('对象：'+'、'.join(str(value) for value in task['objects']))
    for issue in task.get('open_questions',[]):
        st.text('待解决：'+str(issue))
    for job in state.get('execution_progress',[]):
        st.caption(f"执行记录 {job.get('id','')}：{job.get('status','unknown')}（来自后台任务）")
    for event in state.get('user_events',[]):
        if event.get('event_type')!='current_user_request':
            st.info('用户修正：'+event.get('text',''))
    summary=state.get('summary')
    if summary:
        st.caption(f"摘要覆盖轮次：{summary.get('covered_start_seq','?')}–{summary.get('covered_seq','?')}。摘要属于派生资料。")
        for name,label in [('goals','目标'),('decisions','讨论结论'),('conflicts','冲突'),('open_questions','未决问题'),('coverage','证据缺口')]:
            for item in (summary.get('sections') or {}).get(name,[]):
                st.write(label+'：'+item.get('text',''))
                st.caption('来源：'+'、'.join(item.get('source_turn_ids',[])))
    else:
        st.caption('当前没有有效语义摘要；历史原文仍可查询。')
    candidates=state.get('candidates') or []
    for item in candidates:
        st.write(f"待核对记忆：{item['key']} = {item['value']}")
        st.caption('来源：'+item.get('source_turn_id',''))
        payload={key:item.get(key) for key in ('key','source_turn_id','scope','task_id')}
        payload.update(session_id=sid,expected_version=item['version'])
        if st.button('确认这条记忆',key=f"confirm-{sid}-{item['key']}-{item['version']}"):
            result=post('/chat/facts/confirm',payload)
            if result.get('error'): st.error(result['error'])
            else: st.rerun()
        if st.button('撤销这条候选',key=f"reject-{sid}-{item['key']}-{item['version']}"):
            result=post('/chat/facts/revoke',payload)
            if result.get('error'): st.error(result['error'])
            else: st.rerun()
    if not st.checkbox('设置或切换任务',key='context-tasks-'+sid):
        return
    tasks=(get('/chat/tasks',params={'session_id':sid}) or {}).get('tasks',[])
    if tasks:
        choice=st.selectbox('已有任务',[item['task_id'] for item in tasks],key='task-choice-'+sid)
        if st.button('切换到所选任务',key='task-switch-'+sid):
            result=post('/chat/tasks/select',{'session_id':sid,'task_id':choice,'expected_revision':state.get('revision',0)})
            if result.get('error'): st.error(result['error'])
            else: st.rerun()
    page=get('/chat/history',params={'session_id':sid,'limit':100}) or {}
    rows=page.get('turns',[])
    if rows:
        source=st.selectbox('任务目标来自哪次要求',[row['turn_id'] for row in rows],
            format_func=lambda tid:next(row['query'][:80] for row in rows if row['turn_id']==tid),key='task-source-'+sid)
        goal=st.text_input('任务目标',max_chars=2000,key='task-goal-'+sid)
        objects=st.text_input('任务对象（用逗号分隔）',max_chars=2000,key='task-objects-'+sid)
        if st.button('建立新任务',key='task-create-'+sid) and goal.strip():
            import uuid
            result=post('/chat/tasks',{'session_id':sid,'task_id':uuid.uuid4().hex,'goal':goal,
                'objects':[value.strip() for value in objects.replace('，',',').split(',') if value.strip()],
                'source_turn_id':source,'expected_revision':state.get('revision',0)})
            if result.get('error'): st.error(result['error'])
            else: st.rerun()


def render_history_hit(st,get,sid,row):
    tid=row['turn_id']
    st.caption('来源轮次：'+tid+' · 状态：'+str((row.get('metadata') or {}).get('status','success')))
    if row.get('hits'):
        for hit in row['hits']:
            st.text(hit['text'])
    else:
        st.text(row['query'][:400]);st.text(row['answer'][:400])
    field=st.selectbox('展开原文',['query','answer'],key='turn-field-'+sid+'-'+tid,
                       format_func=lambda value:'用户要求' if value=='query' else '历史助手回答')
    key='turn-page-'+sid+'-'+tid+'-'+field
    def read(offset=0):
        return get('/chat/history/turn',params={'session_id':sid,'turn_id':tid,'field':field,'offset':offset,'limit':1200})
    if st.button('读取这一轮原文',key=key+'-read'):
        st.session_state[key]=read()
    page=st.session_state.get(key)
    if page:
        st.text(page.get('text',''))
        if page.get('has_more') and st.button('下一段历史原文',key=key+'-next'):
            st.session_state[key]=read(page.get('next_offset',page.get('next_start',0)))
            st.rerun()
