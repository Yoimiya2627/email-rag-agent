import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import requests

import config.settings as cfg
from frontend.client import StreamAccumulator, http_error
from frontend.evidence_view import render_evidence
from frontend.context_view import render_context_metrics, render_task_context, render_history_hit

API_URL = cfg.API_URL


def _headers():
    return {"Authorization": f"Bearer {cfg.API_AUTH_TOKEN}"} if cfg.API_AUTH_TOKEN else {}

st.set_page_config(
    page_title="邮件智能助手",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── helpers ──────────────────────────────────────────────────────────────────

def _post(endpoint: str, payload: dict = None, timeout: float | None = None) -> dict:
    timeout = cfg.AGENT_RUN_TIMEOUT + 15 if timeout is None else timeout
    try:
        r = requests.post(f"{API_URL}{endpoint}", json=payload or {}, timeout=timeout, headers=_headers())
        if not r.ok:
            return http_error(r)
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"error": "无法连接到后端服务，请先启动 API 服务器"}
    except requests.exceptions.Timeout:
        return {"error": "请求超时，请先检查任务或审批状态，避免重复执行。", "status_code": 504}
    except (requests.exceptions.RequestException, ValueError):
        return {"error": "请求失败或响应无效，请检查服务状态。"}


def _get(endpoint: str, timeout: int = 5, params=None) -> dict | None:
    try:
        r = requests.get(f"{API_URL}{endpoint}", timeout=timeout, headers=_headers(),params=params)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _restore_turns(rows, *, append=False):
    messages = list(st.session_state.get('messages',[])) if append else []
    for row in rows:
        messages.append({'role':'user','content':row['query'],'turn_id':row['turn_id']})
        messages.append({'role':'assistant','content':row['answer'],
                        'extra_metadata':row.get('metadata',{}),'turn_id':row['turn_id']})
    st.session_state['messages'] = messages


def _render_session_controls():
    st.subheader('会话记录')
    if st.button('新会话',use_container_width=True):
        import uuid
        st.session_state.update(session_id=str(uuid.uuid4()),messages=[],history_after=0,history_more=False)
    if st.checkbox('查看已保存会话',key='show_saved_sessions'):
        offset = st.session_state.get('session_list_offset',0)
        data = _get('/chat/sessions',params={'limit':50,'offset':offset}) or {}
        rows = data.get('sessions',[])
        if offset and st.button('上一页会话'):
            st.session_state['session_list_offset'] = max(0,offset-50)
            st.rerun()
        if data.get('next_offset') is not None and st.button('下一页会话'):
            st.session_state['session_list_offset'] = data['next_offset']
            st.rerun()
        if rows:
            chosen = st.selectbox('选择会话',[row['session_id'] for row in rows],
                format_func=lambda sid:next(f"{r['session_id'][:12]} · {r['turn_count']} 轮" for r in rows if r['session_id']==sid))
            if st.button('恢复会话'):
                page = _get('/chat/history',params={'session_id':chosen,'limit':50})
                if page is not None:
                    st.session_state.update(session_id=chosen,history_after=page['next_after'],history_more=page['has_more'])
                    _restore_turns(page['turns'])
            if st.session_state.get('history_more') and st.button('加载后续记录'):
                page = _get('/chat/history',params={'session_id':st.session_state['session_id'],
                    'after':st.session_state.get('history_after',0),'limit':50})
                if page is not None:
                    st.session_state.update(history_after=page['next_after'],history_more=page['has_more'])
                    _restore_turns(page['turns'],append=True)
        else:
            st.caption('暂无可恢复的会话。')


def _render_index_metrics(progress=None, metrics=None):
    """Show only bounded numeric diagnostics and known stage labels."""
    import math
    progress = progress if isinstance(progress, dict) else {}
    metrics = metrics if isinstance(metrics, dict) else {}
    counts = metrics.get('counts') if isinstance(metrics.get('counts'),dict) else {}
    def number(values, *names):
        for name in names:
            value = values.get(name)
            if type(value) in (int,float) and 0 <= value <= 1e15 and math.isfinite(value):
                return value
        return None
    completed = number(progress,'completed_chunks')
    total = number(progress,'total_chunks')
    if total is None:
        total = number(counts,'input_chunks')
    if completed is None and metrics.get('outcome') in ('published','unchanged'):
        completed = total
    if completed is not None and total is not None and total > 0:
        st.progress(min(1.0,completed/total),text=f'已处理 {completed:g} / {total:g} 个片段')
    values = []
    stages = {'validating':'校验输入','indexing':'构建索引','embedding':'计算向量','copying':'复用索引',
              'verifying':'验证索引','publishing':'发布索引','unchanged':'索引无需更新','reusing':'复用向量',
              'index_copy':'复用索引','index_embed':'计算向量','index_verify':'验证索引',
              'index_complete':'索引完成','index_unchanged':'索引无需更新'}
    if isinstance(progress.get('stage'),str) and progress['stage'] in stages:
        values.append(stages[progress['stage']])
    for key,label in (('reused_chunks','复用'),('embedded_chunks','新编码'),('copied_chunks','复制')):
        value = number(progress,key)
        if value is None:
            value = number(counts,key)
        if value is not None:
            values.append(f'{label} {value:g} 个片段')
    rate = number(progress,'chunks_per_second','rate_chunks_per_second')
    remaining = number(progress,'remaining_seconds','eta_seconds')
    if rate is not None:
        values.append(f'{rate:.1f} 片段/秒')
    if remaining is not None:
        values.append(f'预计剩余 {remaining:.1f} 秒')
    outcomes = {'unchanged':'索引未变化，已跳过重建','published':'索引已发布','failed':'索引未完成',
                'cancelled':'索引已停止','pending':'等待索引结果'}
    if isinstance(metrics.get('outcome'),str) and metrics['outcome'] in outcomes:
        values.append(outcomes[metrics['outcome']])
    elapsed = number(metrics,'total_seconds')
    if elapsed is None:
        elapsed = number(progress,'elapsed_seconds')
    if elapsed is not None:
        values.append(f'总耗时 {elapsed:.2f} 秒')
    if values:
        st.caption(' · '.join(values))
    deltas = []
    for key,label in (('new_emails','新增'),('changed_emails','正文变化'),('metadata_only_emails','仅元数据变化'),
                      ('unchanged_emails','未变化'),('deleted_emails','删除')):
        value = number(counts,key)
        if value is not None:
            deltas.append(f'{label} {value:g} 封')
    if deltas:
        st.caption(' · '.join(deltas))


@st.fragment(run_every=2)
def _render_jobs():
    if not (st.session_state.get('show_jobs') or st.session_state.get('job_submitted')):
        return
    offset = st.session_state.get('job_list_offset',0)
    data = _get('/jobs',params={'limit':10,'offset':offset})
    if data is None:
        st.warning('任务状态暂时不可用，请稍后刷新。')
        return
    with st.expander('任务进度与恢复',expanded=True):
        if offset and st.button('上一页任务'):
            st.session_state['job_list_offset'] = max(0,offset-10)
            st.rerun()
        if data.get('next_offset') is not None and st.button('下一页任务'):
            st.session_state['job_list_offset'] = data['next_offset']
            st.rerun()
        for job in data.get('jobs',[]):
            st.write(f"{job['kind']} · {job['status']} · {job['id'][:12]}")
            if job['status']=='succeeded' and job.get('cancel_requested'):
                st.caption('停止请求到达时，本次结果已经完成并保存。')
            progress = job.get('progress') or {}
            if job.get('kind') == 'index':
                _render_index_metrics(progress,(job.get('result') or {}).get('index_metrics'))
            elif progress:
                st.caption(' · '.join(f'{key}: {value}' for key,value in progress.items()))
            if job['status'] in {'queued','running'}:
                if job.get('cancel_requested'):
                    st.info('停止请求已送达；正在等待当前调用返回，期间不会启动后续操作。')
                elif st.button('停止后续执行',key=f"cancel-{job['id']}"):
                    result = _post(f"/jobs/{job['id']}/cancel")
                    if result.get('error'):
                        st.error(result['error'])
            elif job.get('resumable'):
                if st.button('从保存的检查点继续',key=f"resume-{job['id']}"):
                    result = _post(f"/jobs/{job['id']}/resume")
                    if result.get('error'):
                        st.error(result['error'])
            elif job['status'] in {'interrupted','failed','incomplete','cancelled'}:
                if job.get('resume_block_reason')=='recovery_budget_exhausted':
                    st.warning('这项任务已达到恢复次数上限。请核对已有结果，再明确创建新任务。')
                else:
                    st.warning('没有可安全恢复的检查点。请先核对已有步骤和审批状态。')
            result = job.get('result') or {}
            if result.get('answer'):
                st.markdown(result['answer'])
                _render_metadata(result.get('metadata'))
                render_evidence(st,result.get('sources',[]),result.get('metadata') or {},
                                _get,_post,'job-'+job['id'])
                if st.button('载入这次会话记录',key=f"load-job-{job['id']}"):
                    sid = result.get('metadata',{}).get('session_id')
                    page = _get('/chat/history',params={'session_id':sid,'limit':100}) if sid else None
                    if page is not None:
                        st.session_state.update(session_id=sid,history_after=page['next_after'],history_more=page['has_more'])
                        _restore_turns(page['turns'])
                        st.rerun()
            elif job.get('error_code'):
                st.caption(f"错误类型：{job['error_code']} · 任务ID：{job['id']}")


def _render_history_tools():
    sid = st.session_state.get('session_id')
    if not sid:
        return
    with st.expander('回查历史与记录任务约束',expanded=False):
        render_task_context(st,_get,_post,sid)
        term = st.text_input('历史关键词',max_chars=200)
        if st.button('搜索这段会话') and term.strip():
            result = _get('/chat/history/search',params={'session_id':sid,'query':term})
            st.session_state['history-search-'+sid]=(result or {}).get('turns',[])
        for row in st.session_state.get('history-search-'+sid,[]):
            render_history_hit(st,_get,sid,row)
        if st.checkbox('管理显式任务约束',key='manage_facts'):
            source_after = st.session_state.get('fact_source_after_'+sid,0)
            page = _get('/chat/history',params={'session_id':sid,'limit':100,'after':source_after}) or {}
            rows = page.get('turns',[])
            if source_after and st.button('约束来源回到第一页'):
                st.session_state['fact_source_after_'+sid] = 0
                st.rerun()
            if page.get('has_more') and st.button('下一页约束来源'):
                st.session_state['fact_source_after_'+sid] = page['next_after']
                st.rerun()
            facts = (_get('/chat/facts',params={'session_id':sid}) or {}).get('facts',[])
            for fact in facts:
                st.write(f"{fact['key']}（版本 {fact['version']}）：{fact['value']}")
                st.caption(f"来源：{fact['source_turn_id']}。任务资料不替代审批。")
                if st.button('撤销约束',key=f"revoke-{sid}-{fact.get('scope','task')}-{fact['key']}-{fact['version']}"):
                    result=_post('/chat/facts/revoke',{'session_id':sid,'key':fact['key'],
                        'source_turn_id':fact['source_turn_id'],'expected_version':fact['version'],
                        'scope':fact.get('scope','task'),'task_id':fact.get('task_id')})
                    if result.get('error'): st.error(result['error'])
                    else: st.rerun()
            if rows:
                source = st.selectbox('约束来自哪次用户要求',[row['turn_id'] for row in rows],
                    format_func=lambda tid:next(row['query'][:80] for row in rows if row['turn_id']==tid))
                key = st.text_input('约束名称',max_chars=100)
                value = st.text_area('约束内容',max_chars=2000)
                if st.button('保存约束') and key.strip():
                    version = next((fact['version'] for fact in facts if fact['key']==key),0)
                    result = _post('/chat/facts',{'session_id':sid,'key':key,'value':value,
                        'source_turn_id':source,'expected_version':version})
                    if result.get('error'):
                        st.error(result['error'])
                    else:
                        st.success('约束已保存，将随相关会话提供给模型。')


def _render_submission_recovery():
    pending = st.session_state.get('unconfirmed_submission')
    if not pending:
        return
    with st.expander('上次任务提交未确认',expanded=True):
        st.caption('先检查是否已经建立任务。同一次提交会复用操作标识，已建立的任务不会再次执行。')
        if st.button('检查上次任务状态'):
            job = _get('/jobs/operations/'+pending['operation_key'])
            if job:
                st.session_state['job_submitted'] = True
                st.session_state.pop('unconfirmed_submission',None)
                st.info(f"已找到任务 {job['id']}：{job['status']}")
            else:
                st.warning('暂未查到或服务不可用；不会自动重新提交。')
        if st.button('使用原操作标识重试提交'):
            job = _post('/jobs/agent',pending,timeout=10)
            if job.get('error'):
                st.error(job['error'])
            else:
                st.session_state['job_submitted'] = True
                st.session_state.pop('unconfirmed_submission',None)
                st.info(f"任务 {job['id']}：{job['status']}")


def _render_approvals():
    with st.expander('草稿审批与结果核对', expanded=False):
        st.caption('批准按审批绑定的执行方式和邮箱账号执行；账号或授权变化后需重新创建审批。现有功能不会直接发送邮件。')
        if not st.checkbox('加载审批待办', key='load_approvals'):
            return
        status_filter = st.selectbox('审批状态',['pending','unknown','executing'])
        page_key = 'approval_cursor_'+status_filter
        data = _get('/agent/approvals',params={'status':status_filter,'limit':50,
                                              'cursor':st.session_state.get(page_key)})
        if data is None:
            st.warning('审批列表加载失败。')
            return
        pending = [row for row in data.get('approvals', [])
                   if row.get('status') in {'pending', 'executing', 'unknown'}]
        if not pending:
            st.caption('没有待审批或待核对的动作。')
        if st.session_state.get(page_key) and st.button('返回第一页'):
            st.session_state[page_key] = None
            st.rerun()
        if data.get('next_cursor') and st.button('下一页审批'):
            st.session_state[page_key] = data['next_cursor']
            st.rerun()
        for item in pending[:50]:
            identifier = item['approval_id']
            payload = item.get('payload') or {}
            with st.expander(f"{item.get('status')} · {payload.get('subject', '无主题')}"):
                binding = payload.get('execution_binding') or {}
                if binding.get('provider') == 'gmail':
                    st.write('草稿所属邮箱：' + str(binding.get('account_id', '未知')))
                elif binding.get('provider') == 'simulated':
                    st.caption('执行方式：本地模拟，不会创建远端草稿。')
                else:
                    st.caption('旧审批没有账号绑定，不能用于创建 Gmail 草稿；请重新创建审批。')
                st.write('收件人：' + ', '.join(payload.get('to', [])))
                st.text(payload.get('body', ''))
                st.caption(f"审批：{identifier} · 内容校验：{item.get('payload_hash', '')}")
                if item.get('expires_at'):
                    st.caption(f"有效期（Unix时间）：{item['expires_at']}")
                if item['status'] == 'pending':
                    note = st.text_input('备注', key=f'note-{identifier}', max_chars=2000)
                    approve, reject = st.columns(2)
                    action = None
                    if approve.button('批准创建草稿', key=f'approve-{identifier}'):
                        action = 'approve'
                    if reject.button('拒绝', key=f'reject-{identifier}'):
                        action = 'reject'
                    if action:
                        response = _post(f'/agent/approvals/{identifier}/{action}', {'note':note})
                        if response.get('error'):
                            st.error(response['error'])
                        else:
                            st.write({'status':response.get('status'), 'result':response.get('result')})
                else:
                    st.warning('不要重复提交。请先在邮箱草稿中核对这一次操作的结果。')
                    choice = st.selectbox('核对结论', ['仍无法确认', '已找到草稿', '已确认没有执行'], key=f'outcome-{identifier}')
                    evidence = st.text_area('核对依据', key=f'evidence-{identifier}', max_chars=2000)
                    draft_id = st.text_input('已找到的草稿ID', key=f'draft-{identifier}')
                    stopped = st.checkbox('已确认原执行进程停止，不会继续创建草稿', key=f'stopped-{identifier}')
                    if st.button('保存核对结果', key=f'reconcile-{identifier}'):
                        outcome = {'仍无法确认':'unresolved','已找到草稿':'succeeded','已确认没有执行':'not_executed'}[choice]
                        response = _post(f'/agent/approvals/{identifier}/reconcile',
                            {'outcome':outcome,'evidence':evidence,'expected_payload_hash':item.get('payload_hash'),
                             'execution_stopped':stopped,'result':{'draft_id':draft_id,'sent':False} if draft_id else None})
                        if response.get('error'):
                            st.error(response['error'])
                        else:
                            st.write({'status':response.get('status'),'execution_state':response.get('execution_state')})


def _render_metadata(metadata: dict | None):
    """Render assistant-message metadata — agent tool-call steps, or stats."""
    if not metadata:
        return
    render_context_metrics(st,metadata)
    status = metadata.get('status', 'success')
    if status not in {'success', 'approval_required'}:
        st.warning(f"任务状态：{status}。此结果尚未完整完成，请先核对已有结果。")
    elif status == 'approval_required':
        st.info('草稿请求待人工审批。')
    if metadata.get('run_id'):
        st.caption(f"任务编号：{metadata['run_id']}")
    if metadata.get('finish_reason') == 'length' or metadata.get('output_truncated'):
        st.caption('内容受到输出预算限制。')
    if metadata.get('error_code'):
        st.caption(f"错误代码：{metadata['error_code']}")
    steps = metadata.get("steps")
    if steps:
        with st.expander(f"🛠️ Agent 工具调用（{len(steps)} 步）", expanded=False):
            for i, s in enumerate(steps, 1):
                blocked = "  ⚠️ 调用被拦截" if s.get("blocked") else ""
                st.markdown(f"**{i}. `{s.get('tool', '?')}`** · {s.get('status', 'unknown')}{blocked}")
                if s.get("error_code"):
                    st.caption(s['error_code'])
            if metadata.get("max_steps_reached"):
                st.caption("⚠️ 已达到最大步数上限")
            for approval_id in metadata.get('pending_approval_ids', []):
                st.caption(f'待人工确认的审批编号：{approval_id}')
        return
    if not (metadata.get("top5_senders") or metadata.get("label_distribution")
            or metadata.get("daily_counts")):
        return
    with st.expander("📊 统计数据", expanded=False):
        top5 = metadata.get("top5_senders", [])
        if top5:
            st.subheader("发件人 Top 5")
            max_count = max((i.get("count", 1) if isinstance(i, dict) else i[1]) for i in top5)
            for item in top5:
                sender = item.get("sender", "?") if isinstance(item, dict) else item[0]
                count = item.get("count", 0) if isinstance(item, dict) else item[1]
                st.progress(count / max_count, text=f"{sender}：{count} 封")
        label_dist = metadata.get("label_distribution", {})
        if label_dist:
            st.subheader("标签分布")
            st.bar_chart(label_dist)
        daily = metadata.get("daily_counts", [])
        if daily:
            st.subheader("每日邮件量")
            chart_data = {
                (d["date"] if isinstance(d, dict) else d[0]):
                (d["count"] if isinstance(d, dict) else d[1])
                for d in daily
            }
            st.bar_chart(chart_data)


INTENT_LABELS = {
    "retrieve": "🔍 检索",
    "summarize": "📝 摘要",
    "write_reply": "✉️ 回复草稿",
    "analyze": "📊 统计分析",
    "general": "💬 对话 / 需求澄清",
}

QUICK_QUESTIONS = [
    "最近有哪些重要邮件？",
    "帮我总结一下所有邮件",
    "谁给我发邮件最多？",
    "有什么待回复的邮件吗？",
    "请分析最近的邮件标签分布",
    "帮我回复关于项目进度的邮件",
]

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📧 邮件智能助手")
    st.caption("基于 RAG + Multi-Agent 架构")
    st.divider()

    # API status
    health = _get("/health")
    if health:
        st.success("✅ API 已连接")
        if st.button('在 API 中预热模型',help='首次预热可能下载嵌入模型；不会读取邮箱或调用远端生成模型。'):
            result = _post('/warmup',timeout=10)
            if result.get('error'):
                st.error(result['error'])
            else:
                st.info(f"预热状态：{result.get('status')}")
        status = _get("/index/status")
        if status:
            ec = status.get("email_count", 0)
            cc = status.get("chunk_count", 0)
            if status.get('requires_rebuild'):
                st.warning('已有旧索引需要显式重建。请先备份，再使用项目的索引重建命令。')
            elif cc:
                st.info(f"📚 已索引 **{ec}** 封邮件 / **{cc}** 个片段")
            else:
                st.warning("⚠️ 索引为空，请先点击下方 “索引邮件”")
    else:
        st.error("❌ 后端未连接，请运行：\n`uvicorn api.main:app --reload`")

    st.divider()
    st.subheader("📂 数据管理")
    data_path = st.text_input(
        "邮件数据路径",
        value="./data/emails.json",
        help="JSON 文件路径，相对于项目根目录",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗂️ 索引邮件", use_container_width=True):
            result = _post("/jobs/index", {"data_path": data_path}, timeout=10)
            if result.get("error"):
                st.error(result["error"])
            else:
                st.session_state['job_submitted'] = True
                st.info('索引任务已建立，可在任务进度中查看或停止。')
    with col2:
        if st.button("🗑️ 清除索引", use_container_width=True):
            result = _post("/index/clear")
            if result.get("error"):
                st.error(result["error"])
            elif result.get("success"):
                st.success("索引已清除")

    st.divider()
    _render_session_controls()
    st.checkbox('显示后台任务',key='show_jobs')
    st.divider()
    st.subheader("⚡ 快捷问题")
    for q in QUICK_QUESTIONS:
        if st.button(q, use_container_width=True, key=f"qk_{q}"):
            st.session_state["pending_query"] = q

    st.divider()
    st.subheader("⚙️ 模式")
    mode = st.radio(
        "问答模式",
        ["普通（多 Agent 路由）", "Self-RAG（反思工作流）", "Agent（自主工具调用）"],
        help=(
            "普通=意图路由到专家 agent；"
            "Self-RAG=LangGraph 反思重试；"
            "Agent=function-calling 自主规划并多轮调用工具"
        ),
    )
    use_stream = st.toggle("流式输出", value=False, help="SSE 流式返回 token（仅普通模式）")

    st.divider()
    if st.button("🗑️ 清空对话", use_container_width=True):
        sid = st.session_state.get("session_id")
        try:
            if sid:
                response = requests.delete(f"{API_URL}/chat/history", params={"session_id": sid}, timeout=5, headers=_headers())
                response.raise_for_status()
            import uuid
            st.session_state["session_id"] = str(uuid.uuid4())
            st.session_state["messages"] = []
        except Exception:
            st.error('清空失败，可能仍有任务在执行；保留当前对话，请稍后重试。')

# ── main chat area ────────────────────────────────────────────────────────────

st.title("邮件智能问答")
st.caption("支持检索、摘要、回复撰写、统计分析")
st.caption('聊天使用侧栏中的问答索引；下方 163 邮箱使用独立的本地全文索引，真实邮件不会自动进入模型问答。')
from frontend.mailbox_view import render_mailboxes
render_mailboxes(st, _get, _post)
_render_approvals()
_render_jobs()
_render_submission_recovery()

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "session_id" not in st.session_state:
    import uuid
    st.session_state["session_id"] = str(uuid.uuid4())

_render_history_tools()

# Render history
for message_index, msg in enumerate(st.session_state["messages"]):
    with st.chat_message(msg["role"]):
        if msg.get("intent"):
            st.caption(f"意图识别：{INTENT_LABELS.get(msg['intent'], msg['intent'])}")
        st.markdown(msg["content"])

        render_evidence(st,msg.get('sources',[]),msg.get('extra_metadata') or {},
                        _get,_post,'history-'+str(message_index))

        _render_metadata(msg.get("extra_metadata"))

# Handle pending quick query
pending = st.session_state.pop("pending_query", None)
user_input = st.chat_input("输入你的问题，例如：最近有哪些重要邮件？") or pending

if user_input:
    session_id = st.session_state["session_id"]
    # Show user bubble
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # Call API and show assistant bubble
    import uuid
    payload = {"query": user_input, "session_id": session_id, "operation_key":str(uuid.uuid4())}
    if mode.startswith("Agent"):
        endpoint = "/chat/agent"
    elif mode.startswith("Self-RAG"):
        endpoint = "/chat/graph"
    else:
        endpoint = "/chat"

    with st.chat_message("assistant"):
        if use_stream and mode.startswith("普通"):
            # SSE streaming
            import sseclient
            intent_caption = st.empty()
            answer_placeholder = st.empty()
            full_answer = ""
            intent_value = ""
            streamed_sources = []
            stream_done = False
            accumulator = StreamAccumulator()
            try:
                with requests.post(
                    f"{API_URL}/chat/stream",
                    headers=_headers(),
                    json=payload,
                    stream=True,
                    timeout=cfg.AGENT_RUN_TIMEOUT + 15,
                ) as resp:
                    resp.raise_for_status()
                    client_sse = sseclient.SSEClient(resp)
                    for event in client_sse.events():
                        if event.data == "[DONE]":
                            stream_done = True
                            accumulator.feed('[DONE]')
                            break
                        import json as _json
                        token_data = _json.loads(event.data)
                        accumulator.feed(token_data)
                        if "sources" in token_data:
                            streamed_sources = token_data['sources']
                        if "intent" in token_data:
                            intent_value = token_data["intent"]
                            label = INTENT_LABELS.get(intent_value, intent_value)
                            intent_caption.caption(f"意图识别：{label}")
                        elif "token" in token_data:
                            full_answer += token_data["token"]
                            answer_placeholder.markdown(full_answer + "▌")
                answer_placeholder.markdown(full_answer)
                # 标记 _streamed，让下方通用渲染分支跳过重复渲染
                result = accumulator.result()
            except Exception as exc:
                accumulator.error = '流式请求失败，请先检查会话状态。'
                result = accumulator.result()
                answer_placeholder.markdown(accumulator.answer)
        else:
            with st.spinner("思考中…"):
                if mode.startswith('Agent'):
                    st.session_state['unconfirmed_submission'] = dict(payload)
                    result = _post('/jobs/agent',payload,timeout=10)
                    if not result.get('error'):
                        st.session_state.pop('unconfirmed_submission',None)
                        st.session_state['job_submitted'] = True
                        st.info(f"任务已建立：{result['id']}。可在任务进度中停止或查看结果。")
                        result = {'_queued':True}
                else:
                    result = _post(endpoint, payload)

        if result.get('warning'):
            st.warning(result['warning'])
        if result.get('_queued'):
            st.rerun()
        elif result.get("error"):
            st.error(result["error"])
            st.session_state["messages"].append(
                {"role": "assistant", "content": result["error"]}
            )
        else:
            intent = result.get("intent", "")
            answer = result.get("answer", "抱歉，未能生成回答。")

            # 流式模式：placeholder 已渲染答案 + intent caption，跳过下面的重复渲染；
            # 非流式模式：在这里渲染 caption 和答案。
            if not result.get("_streamed"):
                if intent:
                    st.caption(f"意图识别：{INTENT_LABELS.get(intent, intent)}")
                st.markdown(answer)

            sources = result.get("sources", [])
            render_evidence(st,sources,result.get('metadata') or {},_get,_post,'current-response')

            _render_metadata(result.get("metadata"))

            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": answer,
                    "intent": intent,
                    "sources": sources,
                    "extra_metadata": result.get("metadata"),
                }
            )
