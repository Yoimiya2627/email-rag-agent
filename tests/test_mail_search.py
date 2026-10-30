"""Offline mailbox FTS tests. Every database and message is synthetic."""
import json
import sqlite3

import pytest

import core.mail_search as search_mod
from core.mail_search import MailSearchError, ensure, search, status, update


def database(path=':memory:'):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute('''CREATE TABLE messages (
        key TEXT PRIMARY KEY, folder TEXT NOT NULL, present INTEGER NOT NULL DEFAULT 1,
        status TEXT NOT NULL DEFAULT 'parsed',email TEXT,flags TEXT NOT NULL DEFAULT '[]')''')
    db.commit()
    return db


def email(body='预算调整', **overrides):
    return {'subject': 'Synthetic subject', 'sender': 'sender@example.test',
            'recipients': ['recipient@example.test'], 'cc': [], 'date': '2026-09-12T00:00:00+00:00',
            'body': body, 'body_format': 'plain', 'attachments': [], **overrides}


def save(db, key, value, *, folder='INBOX', flags=(), present=1, state='parsed', index=True):
    db.execute('INSERT OR REPLACE INTO messages VALUES (?,?,?,?,?,?)',
               (key, folder, present, state, json.dumps(value), json.dumps(list(flags))))
    if index:
        update(db, key, value if state == 'parsed' else None)


def keys(result):
    return {item['message_key'] for item in result['items']}


def test_initial_backfill_and_two_character_chinese(tmp_path):
    db = database(tmp_path/'first.sqlite3')
    save(db, 'one', email(), index=False)
    save(db, 'failed', email(), state='failed', index=False)
    save(db, 'absent', email(), present=0, index=False)
    result = ensure(db)
    assert result['rebuilt'] and result['indexed_count'] == 1
    assert keys(search(db, '预算')) == {'one'}
    assert status(db)['indexed_count'] == 1
    assert not ensure(db)['rebuilt']
    db.close()


def test_connections_do_not_mix_accounts(tmp_path):
    first, second = database(tmp_path/'a.sqlite3'), database(tmp_path/'b.sqlite3')
    try:
        ensure(first); ensure(second)
        save(first, 'same-key', email('甲方预算'))
        save(second, 'same-key', email('乙方预算'))
        assert keys(search(first, '甲方')) == {'same-key'}
        assert not search(second, '甲方')['items']
        assert keys(search(second, '乙方')) == {'same-key'}
        assert not search(first, '乙方')['items']
    finally:
        first.close();second.close()


@pytest.mark.parametrize('field,value,query', [
    ('subject', 'Project Zebra', 'Zebra'),
    ('sender', 'unique.sender@example.test', 'unique.sender@example.test'),
    ('recipients', ['unique.recipient@example.test'], 'unique.recipient@example.test'),
    ('cc', ['copy.target@example.test'], 'copy.target@example.test'),
    ('body', '正文包含报销凭证', '报销'),
    ('attachments', [{'status': 'complete', 'text': '附件中独有合同编号'}], '合同'),
    ('attachments', [{'status': 'partial', 'text': '已提取部分交付内容'}], '交付'),
])
def test_each_text_source_is_searchable(field, value, query):
    db = database();ensure(db)
    try:
        save(db, 'fixture', email(**{field:value}))
        result = search(db, query)
        assert keys(result) == {'fixture'}
        item = result['items'][0]
        assert item['snippet_format'] == 'plain_text'
        assert len(item['snippet']) <= search_mod.SNIPPET_CHARS
        if field == 'attachments': assert item['snippet_field'] == 'attachments'
    finally:db.close()


def test_binary_and_unread_attachment_fields_are_not_indexed():
    db = database();ensure(db)
    try:
        value=email(attachments=[{'status':'unsupported','text':'unreadsecret','raw':'binarysecret'},
                                 {'status':'complete','text':'allowedtext','raw':'binarysecret'}],
                    source={'raw_html':'rawhtmlsecret'})
        save(db,'fixture',value)
        for query in ('unreadsecret','binarysecret','rawhtmlsecret'):
            assert not search(db,query)['items']
        assert keys(search(db,'allowedtext'))=={'fixture'}
        with pytest.raises(MailSearchError,match='body_not_normalized_text'):
            update(db,'fixture',email('<p>not normalized</p>',body_format='html'))
        assert keys(search(db,'allowedtext'))=={'fixture'}
    finally:db.close()


def test_present_status_and_deleted_flag_are_checked_at_search_time():
    db=database();ensure(db)
    try:
        for key in ('present','absent','failed','deleted'):save(db,key,email())
        db.execute("UPDATE messages SET present=0 WHERE key='absent'")
        db.execute("UPDATE messages SET status='failed' WHERE key='failed'")
        db.execute('UPDATE messages SET flags=? WHERE key=?',(json.dumps(['\\Deleted']),'deleted'))
        assert keys(search(db,'预算'))=={'present'}
        assert status(db)['indexed_count']==1
        update(db,'present',None)
        assert not search(db,'预算')['items']
        assert status(db)['status']=='incomplete'
    finally:db.close()


def test_folder_unread_starred_filters_and_live_flag_updates():
    db=database();ensure(db)
    try:
        save(db,'unread-starred',email(),flags=['\\Flagged'])
        save(db,'read-starred',email(),flags=['\\Seen','\\Flagged'])
        save(db,'archive',email(),folder='Archive',flags=[])
        assert keys(search(db,'预算',unread_only=True,starred_only=True))=={'unread-starred'}
        assert keys(search(db,'预算',folders=['inbox']))=={'unread-starred','read-starred'}
        assert keys(search(db,'预算',folders=['Archive'],unread_only=True))=={'archive'}
        db.execute('UPDATE messages SET flags=? WHERE key=?',(json.dumps(['\\seen','\\flagged']),'unread-starred'))
        assert not search(db,'预算',folders=['INBOX'],unread_only=True)['items']
        db.execute("UPDATE messages SET flags='malformed' WHERE key='read-starred'")
        assert keys(search(db,'预算',starred_only=True))=={'unread-starred'}
    finally:db.close()


def test_query_syntax_cannot_inject_sql_or_fts_operators():
    db=database();ensure(db)
    try:
        save(db,'budget',email('预算内容'))
        save(db,'literal',email('literal OR sentinel'))
        assert keys(search(db,'OR'))=={'literal'}
        assert keys(search(db,'" OR *'))=={'literal'}  # OR is a literal word, not an operator.
        for query in ["x' OR 1=1 --", 'NOT 预算', 'body:预算', '预算 NEAR secret']:
            assert not search(db,query)['items']
        assert not search(db,'预算',folders=["INBOX') OR 1=1 --"])['items']
        assert db.execute('SELECT count(*) FROM messages').fetchone()[0]==2
        assert keys(search(db,'预算'))=={'budget'}
    finally:db.close()


@pytest.mark.parametrize('query',['',' \n\t', 'x'*501])
def test_invalid_queries_rejected(query):
    db=database();ensure(db)
    try:
        with pytest.raises(MailSearchError,match='invalid_query'):search(db,query)
    finally:db.close()


@pytest.mark.parametrize('query',['*','"','(){}[]','！？，'])
def test_symbol_only_query_is_empty_without_scan(query):
    db=database();ensure(db)
    try:
        result=search(db,query)
        assert result['items']==[]
        assert result['diagnostics']['reason']=='no_searchable_terms'
        assert result['diagnostics']['fallback_scan'] is False
    finally:db.close()


def test_update_and_delete_participate_in_caller_transaction():
    db=database();ensure(db);save(db,'one',email('oldtoken'));db.commit()
    try:
        db.execute('BEGIN')
        save(db,'one',email('newtoken'))
        assert keys(search(db,'newtoken'))=={'one'}
        db.rollback()
        assert keys(search(db,'oldtoken'))=={'one'}
        assert not search(db,'newtoken')['items']
        update(db,'one',None)
        assert not search(db,'oldtoken')['items']
        db.rollback()
        assert keys(search(db,'oldtoken'))=={'one'}
    finally:db.close()


def test_failed_update_preserves_previous_terms_and_outer_transaction():
    db=database();ensure(db);save(db,'one',email('oldtoken'));db.commit()
    try:
        db.execute('BEGIN')
        db.execute("UPDATE messages SET folder='Archive' WHERE key='one'")
        with pytest.raises(MailSearchError,match='invalid_document_text'):
            update(db,'one',email(body=object()))
        assert keys(search(db,'oldtoken',folders=['Archive']))=={'one'}
        assert db.in_transaction
        db.rollback()
        assert keys(search(db,'oldtoken',folders=['INBOX']))=={'one'}
    finally:db.close()


def test_failed_migration_never_claims_indexed_and_can_retry():
    db=database()
    try:
        save(db,'good',email(),index=False)
        db.execute("INSERT INTO messages VALUES ('bad','INBOX',1,'parsed','invalid json','[]')")
        db.commit()
        with pytest.raises(MailSearchError,match='invalid_stored_document'):ensure(db)
        assert status(db)['status']=='not_ready'
        assert not db.execute("SELECT 1 FROM sqlite_master WHERE name='mail_search_documents'").fetchone()
        db.execute("DELETE FROM messages WHERE key='bad'")
        assert ensure(db)['indexed_count']==1
    finally:db.close()


def test_version_upgrade_rebuilds_and_rolls_back_to_previous_version(monkeypatch):
    db=database();ensure(db);save(db,'one',email('oldtoken'));db.commit()
    old_version=search_mod.INDEX_VERSION
    try:
        monkeypatch.setattr(search_mod,'INDEX_VERSION','synthetic-new-version')
        assert ensure(db)['rebuilt']
        assert keys(search(db,'oldtoken'))=={'one'}
        db.rollback()
        assert db.execute("SELECT value FROM mail_search_meta WHERE key='version'").fetchone()[0]==old_version
        monkeypatch.setattr(search_mod,'INDEX_VERSION',old_version)
        assert keys(search(db,'oldtoken'))=={'one'}
    finally:db.close()


def test_fts_unavailable_is_explicit_and_does_not_scan():
    class NoFTS(sqlite3.Connection):
        def execute(self,sql,*args,**kwargs):
            if sql.startswith('CREATE VIRTUAL TABLE'):
                raise sqlite3.OperationalError('no such module: fts5')
            return super().execute(sql,*args,**kwargs)
    db=sqlite3.connect(':memory:',factory=NoFTS)
    try:
        db.execute("CREATE TABLE messages(key TEXT PRIMARY KEY,email TEXT,present INTEGER,status TEXT,folder TEXT,flags TEXT)")
        with pytest.raises(MailSearchError,match='fts_unavailable'):ensure(db)
        assert status(db)['status']=='not_ready'
        with pytest.raises(MailSearchError,match='index_not_ready'):search(db,'预算')
    finally:db.close()


def test_result_limit_time_budget_and_document_budget(monkeypatch):
    db=database();ensure(db)
    try:
        for i in range(3):save(db,str(i),email())
        result=search(db,'预算',limit=1)
        assert len(result['items'])==1 and result['diagnostics']['truncated']
        monkeypatch.setattr(search_mod,'SEARCH_SECONDS',-1)
        result=search(db,'预算')
        assert not result['items'] and result['diagnostics']['reason']=='search_time_limit'
        monkeypatch.setattr(search_mod,'MAX_DOCUMENT_CHARS',3)
        with pytest.raises(MailSearchError,match='document_text_limit'):update(db,'0',email())
    finally:db.close()


def test_sql_write_failure_rolls_back_document_and_fts_deletions():
    db=database();ensure(db);save(db,'one',email('oldtoken'));db.commit()
    try:
        db.execute("""CREATE TRIGGER reject_index_insert BEFORE INSERT ON mail_search_documents
                      BEGIN SELECT RAISE(ABORT,'synthetic storage failure'); END""")
        with pytest.raises(MailSearchError,match='index_write_failed'):
            update(db,'one',email('newtoken'))
        assert keys(search(db,'oldtoken'))=={'one'}
        assert not search(db,'newtoken')['items']
        assert status(db)['indexed_count']==1
    finally:db.close()


def test_search_result_character_budget_and_progress_hook_cleanup(monkeypatch):
    db=database();ensure(db);save(db,'one',email())
    try:
        monkeypatch.setattr(search_mod,'MAX_RESULT_SCAN_CHARS',1)
        result=search(db,'预算')
        assert result['items']==[]
        assert result['diagnostics']['reason']=='result_text_budget'
        assert result['diagnostics']['truncated']
        # Search must remove its progress hook on every exit.
        assert db.execute('SELECT count(*) FROM messages').fetchone()[0]==1
    finally:db.close()


def test_text_widget_values_remain_plain_and_header_output_is_bounded():
    db=database();ensure(db)
    try:
        save(db,'one',email('fixture <script>literal text</script>',subject='fixture '+('x'*1200)))
        result=search(db,'fixture')['items'][0]
        assert result['snippet_format']=='plain_text'
        assert '<script>' in result['snippet']
        assert len(result['subject'])==1000
        assert result['display_headers_truncated']
    finally:db.close()


def test_valid_mime_many_recipients_does_not_fail_indexing():
    from email.message import EmailMessage
    from core.imap_mime import parse_imap_message
    message=EmailMessage()
    message['To']=', '.join(f'fixture{i}@example.test' for i in range(1500))
    message.set_content('Synthetic bounded recipient fixture')
    parsed=parse_imap_message(message.as_bytes(),account_id='fixture',folder='INBOX',uidvalidity='1',uid=1)
    assert len(parsed.recipients)==1500
    db=database();ensure(db)
    try:
        save(db,'bulk',parsed.model_dump())
        assert keys(search(db,'fixture1499@example.test'))=={'bulk'}
    finally:db.close()


def test_valid_mime_body_can_have_more_than_250000_distinct_chinese_terms():
    from core.imap_mime import MAX_BODY_CHARS
    # Each separated two-character word contributes one distinct CJK bigram.
    alphabet=[chr(0x4e00+i) for i in range(501)]
    body=' '.join(a+b for a in alphabet for b in alphabet[:500])
    assert len(body)<MAX_BODY_CHARS
    assert 501*500>250_000
    db=database();ensure(db)
    try:
        save(db,'large-cjk',email(body))
        assert keys(search(db,alphabet[-1]+alphabet[499]))=={'large-cjk'}
    finally:db.close()
