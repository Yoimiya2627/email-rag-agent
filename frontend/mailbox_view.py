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


def folder_label(name):
    if str(name).upper() == 'INBOX':
        return '收件箱'
    from agents.imap_readonly import _decode_utf7
    try:
        return _decode_utf7(name)
    except (ValueError, RuntimeError, UnicodeError):
        return str(name)


def _plain_label(value, maximum=110):
    """Native button labels accept Markdown; escape all mail-authored syntax."""
    import re
    value = ' '.join(str(value or '').split())[:maximum]
    return re.sub(r'([\\`*_{}\[\]()#+.!|<>~\-])', r'\\\1', value)


def _date(value):
    try:
        return datetime.fromisoformat(str(value).replace('Z','+00:00')).astimezone().strftime('%m-%d %H:%M')
    except (ValueError, TypeError):
        return '日期未知'


def _select(st, key, value):
    st.session_state[key] = value


def _go_settings(st):
    st.session_state['workspace_page'] = '同步与设置'


def _clear_search(st, aid):
    st.session_state['imap_search_request_'+aid] = None
    for prefix,value in [('mail_query_',''),('mail_search_folders_',[]),('mail_search_unread_',False),('mail_search_starred_',False)]:
        st.session_state[prefix+aid] = value


def body_for_display(email):
    """Present verified normalized table rows without internal source IDs.

    Original text and extraction metadata remain unchanged in local storage.
    An unverified/malformed row falls back to escaped text, never HTML.
    """
    from html import escape
    import json

    def canonical_text(row):
        if row.get('cells') == []:
            table_id, status = row.get('table_id'), row.get('status')
            if (not isinstance(table_id, str) or not isinstance(status, str)
                    or row.get('row_id') != table_id + ':r0'):
                return None
            return '[Table ' + table_id + ' status=' + status + '; empty table]'
        if isinstance(row.get('text'), str):
            return row['text']
        # The MIME parser stores spans and structured cells, deliberately
        # omitting the redundant normalized row text. Reconstruct that exact
        # representation before trusting a span; never strip marker-like prose.
        try:
            fields = []
            for cell in row['cells']:
                label = '/'.join(cell['headers']) or ('header' if cell['header'] else f"column {cell['column']}")
                relation = f" {cell['rowspan']}x{cell['colspan']}" if cell['rowspan'] != 1 or cell['colspan'] != 1 else ''
                carried = ' carried' if cell.get('carried') else ''
                fields.append(f"{cell['source_id']}{relation}{carried} " + json.dumps(label, ensure_ascii=False)
                              + '=' + json.dumps(cell['text'], ensure_ascii=False))
            from core.cleaner import _normalize_whitespace
            return _normalize_whitespace('[Table ' + row['table_id'] + ' row ' + row['row_id']
                                         + ' status=' + row['status'] + '] ' + ' | '.join(fields))
        except (KeyError, TypeError, ValueError):
            return None

    body = email.get('body') or '这封邮件没有可显示的正文。'
    rows = [row for row in email.get('table_rows',[]) if isinstance(row,dict)
            and type(row.get('start')) is int and type(row.get('end')) is int]
    result, cursor = [], 0
    for row in sorted(rows,key=lambda row:row['start']):
        start, end = row['start'], row['end']
        cells = row.get('cells')
        if (not cursor <= start < end <= len(body)
                or not isinstance(cells,list)
                or any(not isinstance(cell,dict) or not isinstance(cell.get('text'),str) for cell in cells)):
            continue
        if canonical_text(row) != body[start:end]:
            continue
        result.append(escape(body[cursor:start]))
        if not cells:
            # An exactly verified empty table has no cell content to display.
            # Keep a human-readable note: extraction limits can also yield an
            # empty record, so do not silently promise the source was blank.
            result.append('（表格未提取到可显示内容）')
            cursor = end
            continue
        result.append('<div class="mail-table-row">')
        for cell in cells:
            headers = cell.get('headers',[])
            label = ' / '.join(value for value in headers if isinstance(value,str)) if isinstance(headers,list) else ''
            result.append('<div class="mail-table-cell">'+('<small>'+escape(label)+'</small>' if label else '')+escape(cell['text'])+'</div>')
        result.append('</div>')
        cursor = end
    result.append(escape(body[cursor:]))
    return ''.join(result)


def _start_sync(st, post, account_id, folders, maximum=100, retry=False):
    result = post('/mailboxes/'+account_id+'/sync', {
        'folders':folders,'max_messages':int(maximum),'retry_failed':retry,
        'operation_key':'imap-'+uuid.uuid4().hex},timeout=10)
    if result.get('error'):
        st.error(result['error'])
    else:
        st.session_state['job_submitted'] = True
        st.session_state['imap_job_'+account_id] = result['id']
        st.success('同步已开始，已有邮件仍可阅读。')


def _sync_description(schedule):
    if schedule.get('enabled') and schedule.get('worker_enabled') is False:
        return '后台同步服务未启用'
    if schedule.get('state') == 'blocked':
        return '自动同步已暂停 · 需要检查设置'
    if schedule.get('enabled'):
        return f"自动同步 · 每 {max(1,int(schedule.get('interval_seconds',300))//60)} 分钟检查"
    return '自动同步未开启'


def _render_message(st, get, prefix, selected, namespace='imap'):
    from html import escape
    record = get(prefix+'/messages/'+selected)
    if record is None:
        st.error('邮件暂时无法读取，请刷新页面后重试。')
        return
    email = record.get('email')
    if not email:
        st.warning('这封邮件尚未成功解析，可前往「同步与设置」查看失败原因并重试。')
        return
    # Mail content is always escaped, including subjects and addresses. Never
    # render the original HTML, Markdown links, images or tracking resources.
    st.markdown('<h2 class="mail-subject">'+escape(email.get('subject') or '无主题')+'</h2>',unsafe_allow_html=True)
    flags = record.get('flags') or email.get('labels',[])
    st.caption(('已读' if '\\Seen' in flags else '未读')+' · '+('已加星标 · ' if '\\Flagged' in flags else '')+
               _plain_label(folder_label(record.get('folder','')))+' · '+_date(email.get('date')))
    st.markdown('<div class="mail-address"><b>发件人</b> '+escape(str(email.get('sender','')))+'</div>',unsafe_allow_html=True)
    if email.get('recipients'):
        st.markdown('<div class="mail-address"><b>收件人</b> '+escape(', '.join(email['recipients']))+'</div>',unsafe_allow_html=True)
    st.divider()
    with st.container(height=470,border=False,key=namespace+'_body_scroll_'+selected):
        st.markdown('<div class="mail-body">'+body_for_display(email)+'</div>',unsafe_allow_html=True)
    parts = email.get('attachments',[])
    if parts:
        st.caption(f'附件 · {len(parts)} 个')
    for index,part in enumerate(parts):
        state = {'complete':'已提取','partial':'部分提取','unsupported':'暂不支持'}.get(part.get('status'),'未完整提取')
        with st.expander(_plain_label(part.get('filename') or '未命名附件')+' · '+state):
            if part.get('status') != 'complete':
                st.caption('该附件的提取结果可能不完整，原始附件仍保存在邮件原件中。')
            if part.get('text'):
                st.text_area('附件提取文本',part['text'],height=200,disabled=True,key=f'{namespace}_attachment_{selected}_{index}')
            with st.expander('查看附件解析详情'):
                st.json({k:v for k,v in part.items() if k not in {'text','locations'}})
                if part.get('locations'):
                    st.json(part['locations'])
    with st.expander('邮件来源与解析详情'):
        st.caption('查看邮件不会改变服务器上的已读状态。')
        st.json({'source':email.get('source',{}),'decode_quality':email.get('decode_quality',{})})


def _render_inbox(st, get, post, account, schedule, report):
    aid, prefix = account['id'], '/mailboxes/'+account['id']
    with st.container(key='mail_header'):
        heading, action = st.columns([4,1],vertical_alignment='center')
        with heading:
            st.title('我的邮箱')
            last = report.get('last_run') or {}
            finished = last.get('finished')
            stamp = datetime.fromtimestamp(finished).strftime('%m-%d %H:%M') if finished else '尚未同步'
            st.caption(f"已同步 {report.get('parsed',0)} 封 · {_sync_description(schedule)} · 最近同步 {stamp}")
        with action:
            if st.button('立即同步',use_container_width=True,type='primary'):
                scope = schedule.get('folders') or [r['name'] for r in report.get('folders',[])] or ['INBOX']
                _start_sync(st,post,aid,scope,schedule.get('max_messages',100))
    if report.get('failed') or report.get('not_downloaded'):
        st.warning(f"还有 {report.get('not_downloaded',0)} 封未下载、{report.get('failed',0)} 封未成功解析，可在「同步与设置」查看。")
    if schedule.get('state') == 'blocked' or (schedule.get('enabled') and schedule.get('worker_enabled') is False):
        st.warning(_sync_description(schedule)+'，请前往「同步与设置」处理。')
    request_key = 'imap_search_request_'+aid
    saved_search = st.session_state.get(request_key) or {}
    with st.form('imap_search_'+aid, border=False):
        search, submit, filters = st.columns([5,1,1],vertical_alignment='bottom')
        with search:
            query = st.text_input('搜索邮件',value=saved_search.get('q',''),max_chars=500,placeholder='搜索主题、发件人、正文或附件',label_visibility='collapsed',key='mail_query_'+aid)
        with submit:
            submitted = st.form_submit_button('搜索',use_container_width=True)
        with filters, st.popover('筛选', use_container_width=True):
            scope, flags = st.columns([2,1])
            with scope:
                folder_names = [r['name'] for r in report.get('folders',[])]
                chosen = st.multiselect('搜索文件夹（留空为全部已同步）',folder_names,
                    default=[name for name in saved_search.get('folders',[]) if name in folder_names],
                    format_func=folder_label,key='mail_search_folders_'+aid)
            with flags:
                unread = st.checkbox('只搜索未读邮件',value=saved_search.get('unread_only',False),key='mail_search_unread_'+aid)
                starred = st.checkbox('只搜索星标邮件',value=saved_search.get('starred_only',False),key='mail_search_starred_'+aid)
    if submitted:
        st.session_state[request_key] = dict(q=query.strip(),unread_only=unread,starred_only=starred,
            **({'folders':chosen} if chosen else {})) if query.strip() else None
        st.session_state['mail_page_'+aid] = 0
    params = st.session_state.get(request_key)
    if params:
        st.caption('当前显示搜索结果；搜索覆盖主题、地址、正文和已提取附件文本。')
        st.button('返回全部邮件',on_click=_clear_search,args=(st,aid))
    offset = st.session_state.get('mail_page_'+aid,0)
    page = get(prefix+'/search',params=params) if params else get(prefix+'/messages',params={'offset':offset,'limit':20})
    if page is None or page.get('error'):
        st.error('邮件列表暂时无法加载，请刷新后重试。')
        return
    if not params and offset > 0 and offset >= page.get('total',0):
        st.session_state['mail_page_'+aid] = max(0,(page.get('total',0)-1)//20*20)
        st.rerun()
    rows = [dict(row,key=row['message_key']) for row in page.get('items',[])] if params else page.get('items',[])
    selected_key = 'mail_selected_'+aid
    available = {row['key'] for row in rows}
    if st.session_state.get(selected_key) not in available:
        st.session_state[selected_key] = None
    listing, reading = st.columns([.38,.62],gap='medium')
    with listing, st.container(key='mail_list'):
        st.subheader('搜索结果' if params else '全部邮件')
        st.caption(f"显示 {len(rows)} 封" if params else f"共 {page.get('total',0)} 封 · 按邮件时间排列")
        if params and page.get('diagnostics',{}).get('truncated'):
            st.caption('结果较多，请缩小搜索范围以查看更多匹配邮件。')
        if not rows:
            st.info('没有匹配邮件，试试其他关键词。' if params else '这里还没有邮件，点击「立即同步」开始收取。')
        with st.container(height=540,border=False,key='mail_rows'):
            for row in rows:
                with st.container(key='mail_row_'+row['key']):
                    is_selected = row['key'] == st.session_state[selected_key]
                    title = _plain_label(row.get('subject') or '无主题')
                    if st.button(title,key='mail_open_'+aid+'_'+row['key'],use_container_width=True,
                                 type='primary' if is_selected else 'secondary',
                                 on_click=_select,args=(st,selected_key,row['key'])):
                        pass
                    st.caption(_plain_label(row.get('sender') or '未知发件人',55))
                    st.caption(_date(row.get('date'))+' · '+_plain_label(folder_label(row.get('folder',''))))
        if not params:
            left, right = st.columns(2)
            if left.button('上一页',disabled=offset==0,use_container_width=True):
                st.session_state['mail_page_'+aid] = max(0,offset-20)
                st.rerun()
            if right.button('下一页',disabled=page.get('next_offset') is None,use_container_width=True):
                st.session_state['mail_page_'+aid] = page['next_offset']
                st.rerun()
    with reading, st.container(key='mail_reader'):
        if st.session_state[selected_key]:
            _render_message(st,get,prefix,st.session_state[selected_key])
        else:
            st.subheader('从左侧选择一封邮件')
            st.caption('邮件正文和附件提取结果会显示在这里。')
    with st.expander('同步动态',expanded=False):
        _render_sync_progress(st,get,aid,schedule)


def _render_account_form(st, get, post, configured):
    with st.expander('添加邮箱 / 更新授权',expanded=not configured):
        st.caption('目前支持 163 邮箱；QQ 等邮箱将在后续接入。所有账号将在左侧统一切换。')
        with st.form('imap_account',clear_on_submit=True):
            st.text_input('邮箱服务',value='网易 163',disabled=True)
            address = st.text_input('163 邮箱地址',placeholder='你的邮箱@163.com',max_chars=128)
            code = st.text_input('客户端授权码',type='password',max_chars=512,
                help='在163网页邮箱开启IMAP并生成客户端授权码。不是网页登录密码，无需发到聊天里。')
            name = st.text_input('账号备注（可选）',max_chars=80,placeholder='例如：工作邮箱、个人邮箱')
            saved = st.form_submit_button('加密保存账号',type='primary')
        if saved:
            result = post('/mailboxes',{'address':address,'authorization_code':code,'display_name':name},timeout=10)
            if result.get('error'):
                st.error(result['error'])
            else:
                st.success('账号已加密保存。测试连接后即可同步邮件。')
                return result
    return None


def _render_settings(st, get, post, account, accounts):
    st.title('邮箱设置')
    st.caption('管理账号与邮件同步。')
    if st.session_state.pop('mail_account_saved_notice', False):
        st.success('账号已加密保存。测试连接后即可同步邮件。')
    if account:
        sync_tab, account_tab, report_tab = st.tabs(['收取设置', '邮箱账号', '同步记录'])
    else:
        account_tab = st.container()
    with account_tab:
        if account:
            st.text(account.get('address', ''))
        saved = _render_account_form(st,get,post,bool(accounts))
        if saved:
            # The persistent account choice is separate from the sidebar widget.
            st.session_state['mail_account_id'] = saved['id']
            st.session_state['mail_account_saved_notice'] = True
            st.rerun()
    if not account:
        return
    aid, prefix = account['id'], '/mailboxes/'+account['id']
    with sync_tab:
        st.subheader('邮件收取')
        st.text(account.get('address',''))
        schedule = get(prefix+'/schedule')
        if schedule is None:
            st.error('同步设置暂时无法读取，请刷新后重试。已保存的设置不会被覆盖。')
            return
        if st.session_state.pop('imap_reset_auto_'+aid,False):
            st.session_state['imap_auto_'+aid] = bool(schedule.get('enabled'))
        cached = 'imap_folders_'+aid
        if st.button('测试连接并读取文件夹',key='imap_connect'):
            with st.spinner('正在连接邮箱…'):
                result = post(prefix+'/connect',timeout=100)
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
            chosen = st.multiselect('同步文件夹',names,default=defaults or names[:1],
                format_func=lambda name:next(row.get('display_name') or folder_label(name) for row in folders if row['name']==name),key='imap_scope_'+aid)
            count = st.number_input('每批最多处理邮件数',min_value=1,max_value=2000,value=int(schedule.get('max_messages',100)),step=50,key='imap_batch_'+aid)
            retry = st.checkbox('重新尝试此前多次失败的邮件',key='imap_retry_'+aid)
            if st.button('开始本地同步与解析',disabled=not chosen):
                _start_sync(st,post,aid,chosen,count,retry)
            st.divider()
            st.subheader('自动同步')
            st.caption('后台服务运行时，关闭浏览器仍会同步。历史邮件会分批补齐，之后按间隔检查新邮件和状态变化。')
            enabled = st.checkbox('启用后台自动同步',value=bool(schedule.get('enabled')),key='imap_auto_'+aid)
            interval = st.number_input('增量检查间隔（分钟）',min_value=1,max_value=1440,
                value=max(1,int(schedule.get('interval_seconds',300))//60),key='imap_interval_'+aid)
            if st.button('保存自动同步设置',disabled=not chosen,type='primary'):
                result = post(prefix+'/schedule',{'enabled':enabled,'folders':chosen,'interval_seconds':int(interval)*60,'max_messages':int(count)},timeout=100)
                if result.get('error'):
                    st.error(result['error'])
                else:
                    st.success('自动同步设置已保存。正在执行的批次会完成后应用新设置。')
                    schedule = result
        else:
            st.caption('点击上方测试连接，可查看文件夹并调整同步范围。已保存的自动同步仍按原设置运行。')
        if schedule.get('enabled') and st.button('暂停自动同步',key='imap_pause_'+aid):
            result = post(prefix+'/schedule',{key:schedule[key] for key in ('folders','interval_seconds','max_messages')} | {'enabled':False})
            if result.get('error'):
                st.error(result['error'])
            else:
                st.session_state['imap_reset_auto_'+aid] = True
                st.rerun()
    with report_tab:
        _render_sync_progress(st,get,aid,schedule)
        if st.button('刷新同步与解析报告'):
            st.rerun()
        with st.expander('同步与解析报告'):
            report = get(prefix+'/report')
            if report is None:
                st.error('同步报告暂时无法读取，请稍后刷新。')
                return
            cols = st.columns(4)
            for col,label,key in zip(cols,['已扫描邮件','解析成功','解析/下载失败','尚未下载'],
                    ['remote_snapshot_count','parsed','failed','not_downloaded']):
                col.metric(label,report.get(key,0))
            st.caption('范围为已扫描文件夹的最近快照；解析成功不表示每个附件或字符都完整。')
            if report.get('issues'):
                st.write('需要核查的项目',report['issues'])
            st.write({'正文为空':report.get('body_empty',0),'解码需核查':report.get('decode_suspect',0),'附件解析状态':report.get('attachments',{})})
            if report.get('folders'):
                st.dataframe([dict(row,name=folder_label(row['name'])) for row in report['folders']],hide_index=True,use_container_width=True)
            failed = get(prefix+'/messages',params={'failures_only':True,'limit':25})
            if failed is None:
                st.warning('失败记录暂时无法读取，请稍后刷新。')
                return
            if failed.get('items'):
                st.caption(f"未成功解析 {failed.get('total',0)} 封；下方最多显示25条。")
                st.dataframe([{key:row.get(key) for key in ('subject','folder','error_code')} for row in failed['items']],hide_index=True,use_container_width=True)


def render_mailboxes(st, get, post, *, view='inbox', accounts=None, account_id=None):
    accounts = accounts if accounts is not None else (get('/mailboxes') or {}).get('accounts',[])
    account = next((row for row in accounts if row['id']==account_id),accounts[0] if accounts else None)
    if view == 'settings':
        _render_settings(st,get,post,account,accounts)
        return
    if account is None:
        st.title('我的邮箱')
        st.subheader('把你的邮箱放到这里')
        st.write('连接邮箱后，在左侧切换账号，在这里阅读、搜索邮件并查看同步状态。')
        st.caption('当前可连接网易163；QQ等邮箱将陆续接入。')
        st.button('连接我的邮箱',type='primary',on_click=_go_settings,args=(st,))
        return
    prefix = '/mailboxes/'+account['id']
    report, schedule = get(prefix+'/report'), get(prefix+'/schedule')
    if report is None or schedule is None:
        st.title('我的邮箱')
        st.error('无法读取邮箱状态，请确认本地服务正在运行后刷新。')
        return
    _render_inbox(st,get,post,account,schedule,report)
