"""Local mailbox setup and honest extraction coverage; never invokes chat."""
import uuid


def _render_sync_progress(st, get, account_id):
    job_id = st.session_state.get('imap_job_'+account_id)
    if not job_id:
        return
    terminal_key = 'imap_terminal_job_'+account_id
    polling = st.session_state.get(terminal_key) != job_id

    @st.fragment(run_every=2 if polling else None)
    def progress():
        job = get('/jobs/'+job_id) or {}
        status = job.get('status')
        terminal = status in {'succeeded','failed','cancelled','interrupted','incomplete'}
        if terminal and polling:
            st.session_state[terminal_key] = job_id
            # Refresh the final report once, then remove the polling timer.
            st.rerun()
        labels = {'queued':'等待执行','running':'正在同步','succeeded':'本次处理完成',
                  'failed':'任务失败','cancelled':'已停止','interrupted':'服务中断','incomplete':'尚未完成'}
        st.info(labels.get(status,'正在查询任务状态'))
        metrics = job.get('progress') or {}
        if metrics:
            st.caption(f"本次已处理 {metrics.get('attempted',0)} 封 · "
                       f"解析成功 {metrics.get('parsed',0)} 封 · 失败 {metrics.get('failed',0)} 封")
        if status in {'queued','running'}:
            st.caption('进度自动刷新；任务结束后会自动更新下方报告。')

    progress()


def render_mailboxes(st, get, post):
    with st.expander('163 邮箱 · 本地收取与解析', expanded=st.session_state.get('show_mailboxes',False)):
        st.caption('邮件在本机收取和解析，不发送给 DeepSeek，也不自动加入问答索引。读取不会更改邮箱已读状态。')
        accounts = (get('/mailboxes') or {}).get('accounts',[])
        with st.expander('添加或更新 163 账号', expanded=not accounts):
            st.caption('先在 163 网页邮箱开启 IMAP/SMTP 服务并生成客户端授权码。授权码在本机按 Windows 用户加密保存。')
            with st.form('imap_account', clear_on_submit=True):
                address = st.text_input('163 邮箱地址', placeholder='你的邮箱@163.com',max_chars=128)
                code = st.text_input('客户端授权码', type='password',max_chars=512,
                                     help='填写客户端授权码，不是网页登录密码。无需发到聊天里。')
                name = st.text_input('账号备注（可选）',max_chars=80)
                saved = st.form_submit_button('加密保存账号')
            if saved:
                result = post('/mailboxes', {'address':address,'authorization_code':code,'display_name':name},timeout=10)
                # The form clears on submission; never copy the code into other session state.
                if result.get('error'):
                    st.error(result['error'])
                else:
                    st.session_state['show_mailboxes']=True
                    st.success('账号已加密保存。接下来测试连接并选择同步范围。')
                    accounts = (get('/mailboxes') or {}).get('accounts',[])
        if not accounts:
            return
        account_id = st.selectbox('当前查看的邮箱',[row['id'] for row in accounts],
            format_func=lambda value: next((row.get('display_name') or row['address']) for row in accounts if row['id']==value),
            key='imap_selected_account')
        prefix = '/mailboxes/'+account_id
        cached = 'imap_folders_'+account_id
        if st.button('测试连接并读取文件夹',key='imap_connect'):
            with st.spinner('正在连接 163…'):
                result = post(prefix+'/connect', timeout=100)
            if result.get('error'):
                st.error(result['error'])
            else:
                st.session_state[cached] = result['folders']
                st.success('只读连接成功。')
        folders = [row for row in st.session_state.get(cached,[]) if row.get('selectable')]
        if folders:
            names = [row['name'] for row in folders]
            chosen = st.multiselect('同步文件夹',names, default=['INBOX'] if 'INBOX' in names else names[:1],
                format_func=lambda name: next(row['display_name'] for row in folders if row['name']==name),
                key='imap_scope_'+account_id)
            count = st.number_input('本次最多处理邮件数',min_value=1,max_value=2000,value=100,step=50,
                                    help='优先处理尚未下载的邮件，重复同步会继续补齐。报告会显示剩余数量。')
            retry = st.checkbox('重新尝试此前多次失败的邮件',key='imap_retry_'+account_id)
            if st.button('开始本地同步与解析',disabled=not chosen):
                request={'folders':chosen,'max_messages':int(count),'retry_failed':retry,'operation_key':'imap-'+uuid.uuid4().hex}
                result=post(prefix+'/sync',request,timeout=10)
                if result.get('error'):
                    st.error(result['error'])
                else:
                    st.session_state['job_submitted']=True
                    st.session_state['imap_job_'+account_id]=result['id']
                    st.success('同步任务已启动。可在任务进度中停止、刷新或继续。')
        _render_sync_progress(st, get, account_id)
        if st.button('刷新同步与解析报告'):
            st.rerun()
        report=get(prefix+'/report') or {}
        if not report.get('folders'):
            st.caption('尚未同步。')
            return
        cols=st.columns(4)
        for col,label,key in zip(cols,['已扫描邮件','解析成功','解析/下载失败','尚未下载'],
                                 ['remote_snapshot_count','parsed','failed','not_downloaded']):
            col.metric(label,report.get(key,0))
        st.caption('范围为已扫描文件夹的最近快照；“解析成功”表示已产出记录，不表示每个附件或字符都完整。')
        st.write({'正文有内容':report.get('body_nonempty',0),'正文为空':report.get('body_empty',0),
                  '解码需核查':report.get('decode_suspect',0),'附件解析状态':report.get('attachments',{})})
        if report.get('issues'):
            st.write('需要核查的项目',report['issues'])
        st.dataframe(report['folders'],hide_index=True,use_container_width=True)
        only_failed=st.checkbox('只看失败记录',key='imap_failed_only_'+account_id)
        page_key='imap_page_'+account_id+str(only_failed)
        offset=st.session_state.get(page_key,0)
        page=get(prefix+'/messages',params={'offset':offset,'limit':25,'failures_only':only_failed}) or {}
        left,right=st.columns(2)
        if left.button('上一页邮件',disabled=offset==0):
            st.session_state[page_key]=max(0,offset-25)
            st.rerun()
        if right.button('下一页邮件',disabled=page.get('next_offset') is None):
            st.session_state[page_key]=page['next_offset']
            st.rerun()
        rows=page.get('items',[])
        if not rows:
            st.caption('这个范围没有邮件记录。')
            return
        selected=st.selectbox('查看已解析邮件',[row['key'] for row in rows],
            format_func=lambda value: next(f"{r.get('subject') or '(无主题/解析失败)'} · {r['folder']} · UID {r['uid']}" for r in rows if r['key']==value))
        record=get(prefix+'/messages/'+selected) or {}
        email=record.get('email')
        if not email:
            st.warning('这封邮件尚未成功解析：'+str(record.get('error_code') or 'unknown'))
            return
        # Plain text widgets avoid rendering mail HTML, remote images or links.
        st.text('发件人：'+str(email.get('sender',''))+'\n日期：'+str(email.get('date','')))
        st.text_area('解析后的正文',email.get('body',''),height=260,disabled=True,key='imap_body_'+selected)
        with st.expander('正文解析质量与来源'):
            st.json({'source':email.get('source',{}),'decode_quality':email.get('decode_quality',{})})
        for index,part in enumerate(email.get('attachments',[])):
            with st.expander(f"附件 {index+1}：{part.get('filename') or '(未命名)'} · {part.get('status','not_read')}"):
                st.json({k:v for k,v in part.items() if k not in {'text','locations'}})
                if part.get('text'):
                    st.text_area('附件提取文本',part['text'],height=220,disabled=True,key=f'imap_attachment_{selected}_{index}')
                if part.get('locations'):
                    st.json(part['locations'])
