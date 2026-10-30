import hashlib
import sqlite3
import pytest
from core.session_repository import SessionRepository,SessionConflictError
from api.sessions import SessionStore,SessionBusyError


def append(repo,query='预算5000美元',answer='ok',owner='o',sid='s',turn_id=None):
    import uuid
    turn_id=turn_id or uuid.uuid4().hex
    repo.append_turns(owner,sid,[{'turn_id':turn_id,'query':query,'answer':answer,'include_in_context':True}],expected_revision=repo.revision(owner,sid))
    return turn_id


def test_history_chinese_exact_ids_hit_windows_and_scope(tmp_path):
    repo=SessionRepository(tmp_path/'s.db')
    target=append(repo,'项目讨论','前言'*300+'退款订单 A-102 联系 aa@example.com')
    append(repo,'退款订单 A102')
    append(repo,'退款订单 A-1020')
    append(repo,'其他','secret',owner='x')
    for _ in range(35): append(repo,'闲聊')
    hits=repo.search_history('o','s','退款')
    assert target in {r['turn_id'] for r in hits}
    result=repo.search_history('o','s','A-102')
    assert [r['turn_id'] for r in result]==[target]
    h=result[0]['hits'][0]
    assert h['start']>200 and 'A-102' in h['text']
    assert h['sha256']==hashlib.sha256(result[0][h['field']][h['start']:h['end']].encode()).hexdigest()
    assert repo.search_history('o','s','secret')==[]
    assert repo.search_history('o','s','" OR * NOT (')==[]
    assert repo.search_history('o','s','aa@example.com')[0]['turn_id']==target
    repo.fts_available=False
    assert repo.search_history('o','s','退款')[0]['retrieval']['degraded_reason']=='fts_unavailable'


def test_epoch_tombstone_reuse_and_old_append(tmp_path):
    repo=SessionRepository(tmp_path/'s.db');append(repo)
    revision=repo.revision('o','s');epoch=repo.context_epoch('o','s')
    repo.delete('o','s')
    assert repo.context_epoch('o','s')==epoch+1
    with pytest.raises(SessionConflictError):
        repo.append_turns('o','s',[{'turn_id':'late','query':'late','answer':'late'}],expected_revision=revision)
    with pytest.raises(SessionConflictError):repo.validate_context_epoch('o','s',epoch)
    append(repo)
    assert repo.context_epoch('o','s')==epoch+1
    assert len(repo.history('o','s')['turns'])==1


def test_current_request_inside_turn_and_active_delete(tmp_path):
    store=SessionStore(path=tmp_path/'s.db')
    with store.turn('o','s') as memory:
        store.record_current_request('o','s',text='hello',request_id='r1')
        memory.append_turn('hello','world')
    assert len(store.history('o','s')['turns'])==1
    with pytest.raises(SessionConflictError):
        with store.turn('o','s') as memory:
            memory.append_turn('late','result')
            store.clear('o','s',invalidate_active=True)
            with pytest.raises(SessionBusyError):
                with store.turn('o','s'):pass
    assert store.history('o','s')['turns']==[]
    with store.turn('o','s') as memory:memory.append_turn('new','new')
    assert len(store.history('o','s')['turns'])==1


def test_facts_candidates_corrections_and_task_scope(tmp_path):
    repo=SessionRepository(tmp_path/'s.db');source=append(repo)
    repo.set_task_fact('o','s','budget',5000,source_turn_id=source,expected_version=0,explicit_user=True)
    repo.set_task_fact('o','s','language','中文',source_turn_id=source,expected_version=0,explicit_user=True)
    repo.record_current_request('o','s',text='预算改为6000美元',request_id='change')
    assert [f['key'] for f in repo.task_facts('o','s')]==['language']
    assert repo.context_state('o','s')['user_events'][0]['event_type']=='current_user_correction'
    assert [f for f in repo.task_facts('o','s',include_history=True) if f['key']=='budget'][0]['status']=='superseded'
    repo.set_task_fact('o','s','budget',6000,source_turn_id=source,expected_version=1,status='candidate')
    assert not any(f['key']=='budget' for f in repo.task_facts('o','s'))
    repo.confirm_task_fact('o','s','budget',source_turn_id=source,expected_version=2,explicit_user=True)
    assert next(f for f in repo.task_facts('o','s') if f['key']=='budget')['value']==6000
    repo.update_task('o','s',task_id='order2',goal='新订单',source_turn_id=source,expected_revision=repo.revision('o','s'),explicit_user=True)
    assert repo.task_facts('o','s')==[]
    assert repo.context_state('o','s')['user_events']==[]
    with pytest.raises(SessionConflictError):
        repo.select_task('o','s',task_id='order2',expected_revision=0,explicit_user=True)


def test_migration_keeps_all_legacy_facts_and_versions(tmp_path):
    path=tmp_path/'s.db';repo=SessionRepository(path);source=append(repo)
    with repo._connect() as db:
        db.execute('INSERT INTO task_facts VALUES (?,?,?,?,?,?,?,?)',('o','s','budget',3,'constraint','5000',source,'now'))
        db.execute('INSERT INTO task_facts VALUES (?,?,?,?,?,?,?,?)',('o','s','language',1,'constraint','"中文"',source,'now'))
    repo=SessionRepository(path)
    repo.set_task_fact('o','s','budget',6000,source_turn_id=source,expected_version=3,explicit_user=True)
    assert {f['key']:f['value'] for f in repo.task_facts('o','s')}=={'budget':6000,'language':'中文'}
    repo=SessionRepository(path)
    assert next(f for f in repo.task_facts('o','s') if f['key']=='budget')['value']==6000


def test_read_page_and_delete_fts(tmp_path):
    repo=SessionRepository(tmp_path/'s.db');tid=append(repo,answer='段落'*3000)
    page=repo.get_turn_page('o','s',tid,offset=2500,limit=500,expected_epoch=0)
    assert page['start']==2500 and page['next_offset']==3000
    assert page['sha256']==hashlib.sha256(page['text'].encode()).hexdigest()
    with pytest.raises(KeyError):repo.get_turn_page('x','s',tid)
    repo.delete('o','s')
    with repo._connect() as db:
        assert db.execute('SELECT count(*) FROM history_fts').fetchone()[0]==0
        assert db.execute('SELECT count(*) FROM context_turn_tasks').fetchone()[0]==0


def test_legacy_database_backup_and_reindex(tmp_path):
    path=tmp_path/'legacy.db'
    with sqlite3.connect(path) as db:
        db.executescript("CREATE TABLE sessions(owner_id TEXT,session_id TEXT,revision INTEGER,updated_at TEXT,PRIMARY KEY(owner_id,session_id));")
        db.execute('INSERT INTO sessions VALUES (?,?,?,?)',('o','s',0,'now'))
    repo=SessionRepository(path)
    backup=path.with_suffix('.db.pre-context.bak')
    assert backup.exists()
    assert backup.with_suffix('.bak.sha256').read_text().strip()==hashlib.sha256(backup.read_bytes()).hexdigest()
    tid=append(repo,'独立检索测试')
    assert repo.rebuild_history_index()['indexed_turns']==1
    assert repo.search_history('o','s','检索')[0]['turn_id']==tid


def test_request_idempotency_does_not_rewind_cache_revision(tmp_path):
    store=SessionStore(path=tmp_path/'s.db')
    with store.turn('o','s') as memory:
        store.record_current_request('o','s',text='same',request_id='same')
        memory.append_turn('same','first')
    with store.turn('o','s') as memory:
        store.record_current_request('o','s',text='same',request_id='same')
        memory.append_turn('same','second')
    assert len(store.history('o','s')['turns'])==2


def test_correction_survives_ordinary_request_window(tmp_path):
    repo=SessionRepository(tmp_path/'s.db');append(repo)
    repo.record_current_request('o','s',text='预算改为6000美元',request_id='correction')
    for i in range(30):repo.record_current_request('o','s',text='继续处理',request_id=f'normal-{i}')
    events=repo.context_state('o','s')['user_events']
    assert len(events)==6 and any(e['event_id']=='correction' for e in events)


def test_index_failure_does_not_discard_transcript_and_rebuild_recovers(tmp_path,monkeypatch):
    import core.history_index as index
    repo=SessionRepository(tmp_path/'s.db')
    original=index.index_turn
    def broken(*args):raise sqlite3.OperationalError('fts tokenizer failed')
    monkeypatch.setattr(index,'index_turn',broken)
    tid=append(repo,'退款索引')
    assert repo.get_turn('o','s',tid)['query']=='退款索引'
    assert repo.search_history('o','s','退款')[0]['retrieval']['index_incomplete']
    monkeypatch.setattr(index,'index_turn',original)
    repo.rebuild_history_index()
    assert not repo.search_history('o','s','退款')[0]['retrieval']['index_incomplete']


def test_task_switch_filters_recent_and_automatic_history(tmp_path):
    repo=SessionRepository(tmp_path/'s.db');old=append(repo,'旧预算5000')
    repo.update_task('o','s',task_id='new',source_turn_id=old,expected_revision=repo.revision('o','s'),explicit_user=True)
    assert repo.recent_context('o','s')==[]
    new=append(repo,'新预算6000')
    assert [t['turn_id'] for t in repo.recent_context('o','s')]==[new]
    assert [t['turn_id'] for t in repo.search_history('o','s','预算',task_id='new')]==[new]
    assert len(repo.search_history('o','s','预算'))==2


def test_future_schema_is_rejected_without_mutation(tmp_path):
    path=tmp_path/'s.db';repo=SessionRepository(path);append(repo)
    with repo._connect() as db:db.execute('UPDATE context_schema SET version=999')
    with pytest.raises(ValueError,match='unsupported context schema'):SessionRepository(path)


def test_correction_replacement_and_resolution(tmp_path):
    repo=SessionRepository(tmp_path/'s.db');tid=append(repo)
    repo.record_current_request('o','s',text='预算改为6000',request_id='first')
    repo.record_current_request('o','s',text='预算改为7000',request_id='second')
    events=repo.context_state('o','s')['user_events']
    assert next(e for e in events if e['event_id']=='first')['status']=='superseded'
    assert next(e for e in events if e['event_id']=='second')['requires_protection']
    repo.set_task_fact('o','s','budget',7000,source_turn_id=tid,expected_version=0,explicit_user=True)
    assert not any(e['requires_protection'] for e in repo.context_state('o','s')['user_events'])


def test_existing_fts_trigger_dropped_when_fts_unavailable(tmp_path):
    from contextlib import contextmanager
    path=tmp_path/'s.db';repo=SessionRepository(path);append(repo)
    class NoFts(SessionRepository):
        @contextmanager
        def _connect(self):
            with super()._connect() as db:
                class Proxy:
                    def execute(self,sql,*args):
                        if 'CREATE VIRTUAL TABLE' in sql:raise sqlite3.OperationalError('no such module: fts5')
                        return db.execute(sql,*args)
                    def __getattr__(self,name):return getattr(db,name)
                yield Proxy()
    degraded=NoFts(path)
    assert not degraded.fts_available
    degraded.delete('o','s')
    assert degraded.history('o','s')['turns']==[]
    with degraded._connect() as db:
        assert db.execute("SELECT 1 FROM sqlite_master WHERE name='history_delete'").fetchone() is None
    assert degraded.context_epoch('o','s')==1


def test_search_diagnostics_are_thread_local_scalars(tmp_path):
    import threading
    repo=SessionRepository(tmp_path/'s.db');append(repo)
    repo.search_history('o','s','预算')
    original=repo.last_search_diagnostics.copy()
    output=[]
    def other():
        assert repo.last_search_diagnostics=={}
        repo.last_history_diagnostics={'degraded_reason':'test'}
        output.append(repo.last_search_diagnostics)
    worker=threading.Thread(target=other);worker.start();worker.join()
    assert output==[{'degraded_reason':'test'}] and repo.last_search_diagnostics==original
    assert all(isinstance(v,(str,int,float,bool,type(None))) for v in original.values())


def test_active_correction_does_not_age_out_after_unrelated_corrections(tmp_path):
    repo=SessionRepository(tmp_path/'s.db');tid=append(repo)
    repo.record_current_request('o','s',text='预算改为6000美元',request_id='budget-change')
    for i in range(25):
        key=f'field_{i}_end'
        repo.set_task_fact('o','s',key,'old',source_turn_id=tid,expected_version=0,explicit_user=True)
        repo.record_current_request('o','s',text=f'{key}改为new',request_id=f'change-{i}')
    state=repo.context_state('o','s')
    assert len([e for e in state['user_events'] if e['requires_protection']])==26
    budget=next(e for e in state['user_events'] if e['event_id']=='budget-change')
    assert budget['requires_protection'] and budget['affected_keys']==['budget']
    assert not state['required_omissions']


def test_active_correction_storage_overflow_is_explicit(tmp_path):
    repo=SessionRepository(tmp_path/'s.db');tid=append(repo)
    repo.record_current_request('o','s',text='预算改为'+('非常长的约束'*20000),request_id='long-budget')
    state=repo.context_state('o','s')
    assert state['user_event_overflow']
    assert state['required_omissions'][0]['reason']=='active_user_event_char_limit'
    assert state['user_event_chars']<=100000
    assert not state['user_events']


def test_active_correction_count_overflow_is_explicit(tmp_path):
    repo=SessionRepository(tmp_path/'s.db');tid=append(repo)
    for i in range(101):
        key=f'field_{i}_end'
        repo.set_task_fact('o','s',key,'old',source_turn_id=tid,expected_version=0,explicit_user=True)
        repo.record_current_request('o','s',text=f'{key}改为new',request_id=f'change-{i}')
    state=repo.context_state('o','s')
    assert len(state['user_events'])==100 and state['user_event_overflow']
    assert state['required_omissions'][0]['reason']=='active_user_event_count_limit'
