"""Local mailbox setup and honest extraction coverage; never invokes chat."""
import uuid
from datetime import datetime


def _render_sync_progress(st, get, account_id, schedule=None):
    schedule = schedule or {}
    manual_id = st.session_state.get('imap_job_'+account_id)
    if not manual_id and not schedule.get('last_job_id') and not schedule.get('enabled'):
        return
    terminal_key = 'imap_terminal_job_'+account_id
    terminal_ids_key = 'imap_terminal_jobs_'+account_id
    tracked_ids = {value for value in (manual_id,schedule.get('last_job_id')) if value}
    known_terminal = set(st.session_state.get(terminal_ids_key,[]))
    auto_polling = bool(schedule.get('enabled')) and schedule.get('worker_enabled',True)
    polling = auto_polling or bool(tracked_ids-known_terminal)

    @st.fragment(run_every=(5 if auto_polling else 2) if polling else None)
    def progress():
        current = (get('/mailboxes/'+account_id+'/schedule') or schedule) if schedule.get('revision') else schedule
        if (bool(current.get('enabled')) and current.get('worker_enabled',True)) != auto_polling:
            st.rerun()
        if current.get('enabled'):
            if current.get('worker_enabled') is False:
                st.warning('后台自动同步服务未启用，已保存的计划当前不会执行。手动同步任务仍可单独运行。')
            elif current.get('state') == 'blocked':
                st.warning('连续同步失败，自动同步已暂停。检查网络或授权后重新保存设置可恢复。')
            else:
                due = current.get('next_due')
                when = datetime.fromtimestamp(due).strftime('%m-%d %H:%M:%S') if due else '即将执行'
                st.caption('后台自动同步已启用 · 下次执行：'+when)
            if current.get('failure_count'):
                st.caption('当前连续失败次数：'+str(current['failure_count'])+'；重试间隔会逐步延长。')
        ids = list(dict.fromkeys(value for value in (manual_id,current.get('last_job_id')) if value))
        jobs = [get('/jobs/'+value) or {} for value in ids]
        if not jobs:
            return
        terminal_ids = {job['id'] for job in jobs if job.get('id') and
                        job.get('status') in {'succeeded','failed','cancelled','interrupted','incomplete'}}
        active_jobs = [job for job in jobs if job.get('status') in {'queued','running'}]
        job = max(active_jobs or jobs,key=lambda value:value.get('created',0))
        job_id = job.get('id')
        status = job.get('status')
        terminal = status in {'succeeded','failed','cancelled','interrupted','incomplete'}
        if terminal_ids != set(st.session_state.get(terminal_ids_key,[])):
            st.session_state[terminal_ids_key] = sorted(terminal_ids)
            if terminal:
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


def _render_message(st, get, prefix, selected, namespace='imap'):
    record=get(prefix+'/messages/'+selected) or {}
    email=record.get('email')
    if not email:
        st.warning('这封邮件尚未成功解析：'+str(record.get('error_code') or 'unknown'))
        return
    flags=record.get('flags') or email.get('labels',[])
    st.caption(('已读' if '\\Seen' in flags else '未读')+' · '+('星标' if '\\Flagged' in flags else '未加星标')+
               ' · '+str(record.get('folder','')))
    st.text('发件人：'+str(email.get('sender',''))+'\n日期：'+str(email.get('date','')))
    st.text_area('解析后的正文',email.get('body',''),height=260,disabled=True,key=namespace+'_body_'+selected)
    with st.expander('正文解析质量与来源'):
        st.json({'source':email.get('source',{}),'decode_quality':email.get('decode_quality',{})})
    for index,part in enumerate(email.get('attachments',[])):
        with st.expander(f"附件 {index+1}：{part.get('filename') or '(未命名)'} · {part.get('status','not_read')}"):
            st.json({k:v for k,v in part.items() if k not in {'text','locations'}})
            if part.get('text'):
                st.text_area('附件提取文本',part['text'],height=220,disabled=True,key=f'{namespace}_attachment_{selected}_{index}')
            if part.get('locations'):
                st.json(part['locations'])


def _render_search(st, get, prefix, account_id, report):
    status=report.get('local_search') or {}
    st.caption(f"本地全文索引：{status.get('indexed_count',0)} 封邮件。关键词搜索覆盖主题、地址、正文和已提取附件文本，不调用模型。")
    with st.expander('搜索本地真实邮件'):
        with st.form('imap_search_'+account_id):
            query=st.text_input('搜索关键词',max_chars=500,placeholder='例如：发票、项目名称、联系人邮箱')
            chosen=st.multiselect('搜索文件夹（留空为全部已同步）',[row['name'] for row in report.get('folders',[])])
            unread=st.checkbox('只搜索未读邮件')
            starred=st.checkbox('只搜索星标邮件')
            submitted=st.form_submit_button('本地搜索')
        request_key='imap_search_request_'+account_id
        if submitted:
            st.session_state[request_key]=dict(q=query.strip(),unread_only=unread,starred_only=starred,
                **({'folders':chosen} if chosen else {})) if query.strip() else None
        params=st.session_state.get(request_key)
        if params:
            found=get(prefix+'/search',params=params)
            if found is None or found.get('error'):
                st.error('本地搜索未完成，请检查搜索词或索引状态后重试。')
                return
            rows=found.get('items',[])
            if not rows:
                st.info('当前已同步范围没有匹配结果。')
            if (found.get('diagnostics') or {}).get('truncated'):
                st.caption('结果达到本次显示或资源上限，可缩小搜索范围。')
            if rows:
                selected=st.selectbox('查看搜索结果',[row['message_key'] for row in rows],
                    format_func=lambda key:next(row['subject'] or '(无主题)' for row in rows if row['message_key']==key))
                hit=next(row for row in rows if row['message_key']==selected)
                st.text(hit.get('snippet',''))
                _render_message(st,get,prefix,selected,namespace='imap_search')


def render_mailboxes(st, get, post):
    with st.expander('163 邮箱 · 本地收取与解析', expanded=st.session_state.get('show_mailboxes',False)):
        st.caption('邮件在本机收取、解析并建立独立全文索引，不发送给 DeepSeek。读取不会更改邮箱已读状态。')
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
        schedule=get(prefix+'/schedule') or {}
        if st.session_state.pop('imap_reset_auto_'+account_id,False):
            st.session_state['imap_auto_'+account_id]=bool(schedule.get('enabled'))
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
            defaults = [name for name in schedule.get('folders',[]) if name in names] if schedule.get('revision') else [
                row['name'] for row in folders if not {'\\junk','\\trash','\\drafts'}.intersection(flag.lower() for flag in row.get('flags',[]))
                and row['display_name'].casefold() not in {'垃圾邮件','已删除','草稿箱','junk','spam','trash','drafts'}]
            chosen = st.multiselect('同步文件夹',names, default=defaults or names[:1],
                format_func=lambda name: next(row['display_name'] for row in folders if row['name']==name),
                key='imap_scope_'+account_id)
            count = st.number_input('本次最多处理邮件数',min_value=1,max_value=2000,value=int(schedule.get('max_messages',100)),step=50,
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
            with st.expander('自动同步与历史回补',expanded=bool(schedule.get('enabled'))):
                st.caption('后台服务运行时，关闭网页仍会同步；电脑关机或休眠后会在恢复运行时补齐。历史邮件每批完成后继续，补齐后按设定间隔检查新增和状态变化。')
                enabled=st.checkbox('启用后台自动同步',value=bool(schedule.get('enabled')),key='imap_auto_'+account_id)
                interval=st.number_input('增量检查间隔（分钟）',min_value=1,max_value=1440,
                    value=max(1,int(schedule.get('interval_seconds',300))//60),step=1,key='imap_interval_'+account_id)
                st.caption('使用上方选定文件夹和单批数量；垃圾箱、已删除与草稿可按需要单独选择。')
                if st.button('保存自动同步设置',disabled=not chosen):
                    result=post(prefix+'/schedule',{'enabled':enabled,'folders':chosen,
                        'interval_seconds':int(interval)*60,'max_messages':int(count)},timeout=100)
                    if result.get('error'):
                        st.error(result['error'])
                    else:
                        st.success('自动同步设置已保存。当前批次如正在执行，会完成后应用新设置。')
                        schedule=result
        if schedule.get('enabled') and st.button('暂停自动同步',key='imap_pause_'+account_id):
            result=post(prefix+'/schedule',{key:schedule[key] for key in ('folders','interval_seconds','max_messages')} | {'enabled':False})
            if result.get('error'):
                st.error(result['error'])
            else:
                st.session_state['imap_reset_auto_'+account_id]=True
                st.rerun()
        _render_sync_progress(st, get, account_id, schedule)
        if st.button('刷新同步与解析报告'):
            st.rerun()
        report=get(prefix+'/report') or {}
        if not report.get('folders'):
            st.caption('尚未同步。')
            return
        _render_search(st,get,prefix,account_id,report)
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
        _render_message(st,get,prefix,selected)
