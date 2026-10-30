"""Evidence display is independent of chat submission and approval controls."""
import hashlib
import json
from urllib.parse import urlencode


def _coverage_caption(st, coverage):
    if coverage.get('attachment_inventory_status')=='unknown':
        st.caption('附件清单未知，不能据此判断是否包含附件。')
    elif coverage.get('unread_attachments'):
        st.caption(f"附件 {coverage.get('attachment_count',0)} 个，其中 {coverage['unread_attachments']} 个尚未读取或解析。")
    elif coverage.get('attachment_count')==0:
        st.caption('来源记录的附件清单为空。')
    elif coverage.get('attachment_count') is not None:
        st.caption(f"来源记录包含 {coverage['attachment_count']} 个已解析附件。")
    if coverage.get('decode_status') in {'suspect','no_signal'}:
        st.caption('正文解码存在疑点，请核对原件。')


def render_evidence(st,sources,metadata,get,post,key):
    refs = metadata.get('model_visible_evidence') or []
    cited = metadata.get('cited_evidence') or []
    candidates = {(s['email_id'],s['chunk_id']):s for s in sources}
    entries = refs or [{'email_id':s['email_id'],'chunk_id':s['chunk_id']} for s in sources]
    if not entries:
        return
    with st.expander(f'证据核对 · {len(refs)} 段模型可见范围 / {len(sources)} 条检索候选',expanded=False):
        st.caption('可见范围表示提供给模型的原文；引用及版本匹配不能单独证明结论正确。附件和邮箱同步覆盖以来源记录为准。')
        ordinal = st.selectbox('选择来源范围',list(range(len(entries))),key=key+'-source',
            format_func=lambda i:f"{entries[i]['email_id']}#{entries[i]['chunk_id']} · {entries[i].get('visible_start','?')}–{entries[i].get('visible_end','?')}")
        ref = entries[ordinal]
        source = candidates.get((ref['email_id'],ref['chunk_id']),{})
        details = source.get('metadata') or {}
        from core.evidence import source_coverage
        coverage = source_coverage(details)
        _coverage_caption(st,coverage)
        st.text(' · '.join(str(details.get(field,'')) for field in ('sender','date','subject')))
        st.caption('回答引用了此范围。' if ref in cited else '此条没有经过回答引用匹配；请核对正文。')
        token = key+'-'+hashlib.sha256(json.dumps(ref,sort_keys=True).encode()).hexdigest()[:16]
        valid = all(ref.get(field) is not None for field in ('source_version','visible_start','visible_end','visible_hash'))
        if not valid:
            st.caption('这条记录没有可验证的可见范围，以下仅为检索预览。')
            st.text(source.get('content','')[:1200])
        else:
            st.caption(f"版本：{ref['source_version']} · 原文字符 {ref['visible_start']}–{ref['visible_end']} · 校验 {ref['visible_hash']}")
        def read(start=None):
            if valid:
                payload = {name:ref[name] for name in ('email_id','chunk_id','source_version','source_sha256','chunk_sha256','visible_start','visible_end','visible_hash') if name in ref}
                suffix = '?'+urlencode({'start':start}) if start is not None else ''
                return post('/evidence/reread'+suffix,payload,timeout=15)
            return get('/evidence/email',params={'email_id':ref['email_id'],'chunk_id':ref['chunk_id'],
                'start':start or 0,'limit':1200}) or {'error':'原文读取失败，请检查服务状态。'}
        if st.button('核验并回读原文' if valid else '读取当前索引正文',key=token+'-read'):
            st.session_state[token] = read()
        page = st.session_state.get(token)
        if page:
            if page.get('error'):
                st.warning(page['error'])
                st.caption('版本或校验不匹配时，旧引用不会自动替换成新内容。')
            else:
                st.text(page.get('body',''))
                for chunk in page.get('chunks',[]):
                    if chunk.get('table_context'):
                        st.text(chunk['table_context'])
                st.caption(f"当前展示 {page.get('read_start')}–{page.get('read_end')} 字符。")
                _coverage_caption(st,page.get('coverage',{}))
                if page.get('verification_complete'):
                    st.caption(f"完整核验范围 {page['verified_start']}–{page['verified_end']}，本页只是其中一段。")
                beginning = ref['visible_start'] if valid else 0
                if page.get('read_start',0)>beginning and st.button('上一段原文',key=token+'-prev'):
                    st.session_state[token] = read(max(beginning,page['read_start']-1200))
                    st.rerun()
                if page.get('has_more') and st.button('下一段原文',key=token+'-next'):
                    st.session_state[token] = read(page['next_start'])
                    st.rerun()
