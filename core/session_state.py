"""Versioned, session-scoped task memory. Derived records confer no authority."""
import hashlib
import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).isoformat()


SCHEMA = '''
CREATE TABLE IF NOT EXISTS context_turn_tasks(owner_id TEXT NOT NULL,session_id TEXT NOT NULL,turn_id TEXT NOT NULL,task_id TEXT NOT NULL,
 PRIMARY KEY(owner_id,session_id,turn_id),FOREIGN KEY(owner_id,session_id,turn_id) REFERENCES session_turns(owner_id,session_id,turn_id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS candidate_attempts(owner_id TEXT NOT NULL,session_id TEXT NOT NULL,attempt_id TEXT NOT NULL,
 task_id TEXT NOT NULL,source_revision INTEGER NOT NULL,epoch INTEGER NOT NULL,input_sha256 TEXT NOT NULL,status TEXT NOT NULL,
 error TEXT,model_id TEXT NOT NULL,model_revision TEXT NOT NULL,created_at TEXT NOT NULL,
 PRIMARY KEY(owner_id,session_id,attempt_id),FOREIGN KEY(owner_id,session_id) REFERENCES sessions ON DELETE CASCADE);
CREATE INDEX IF NOT EXISTS candidate_attempt_lookup ON candidate_attempts(owner_id,session_id,task_id,epoch,input_sha256);
CREATE TABLE IF NOT EXISTS context_schema(version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS session_epochs(owner_id TEXT NOT NULL,session_id TEXT NOT NULL,
 epoch INTEGER NOT NULL DEFAULT 0,revision INTEGER NOT NULL DEFAULT 0,PRIMARY KEY(owner_id,session_id));
CREATE TABLE IF NOT EXISTS context_event_changes(owner_id TEXT NOT NULL,session_id TEXT NOT NULL,event_id TEXT NOT NULL,
 fact_key TEXT NOT NULL,status TEXT NOT NULL,seen_seq INTEGER NOT NULL DEFAULT 0,PRIMARY KEY(owner_id,session_id,event_id,fact_key),
 FOREIGN KEY(owner_id,session_id,event_id) REFERENCES context_events(owner_id,session_id,event_id) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS context_tasks(owner_id TEXT NOT NULL,session_id TEXT NOT NULL,task_id TEXT NOT NULL,
 revision INTEGER NOT NULL,state TEXT NOT NULL,active INTEGER NOT NULL,PRIMARY KEY(owner_id,session_id,task_id),
 FOREIGN KEY(owner_id,session_id) REFERENCES sessions ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS context_facts(owner_id TEXT NOT NULL,session_id TEXT NOT NULL,task_id TEXT NOT NULL,
 scope TEXT NOT NULL,fact_key TEXT NOT NULL,version INTEGER NOT NULL,status TEXT NOT NULL,kind TEXT NOT NULL,
 value TEXT NOT NULL,source_turn_id TEXT NOT NULL,source_quote TEXT NOT NULL,updated_at TEXT NOT NULL,
 PRIMARY KEY(owner_id,session_id,task_id,scope,fact_key,version),
 FOREIGN KEY(owner_id,session_id) REFERENCES sessions ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS context_events(owner_id TEXT NOT NULL,session_id TEXT NOT NULL,event_id TEXT NOT NULL,
 task_id TEXT NOT NULL,event_type TEXT NOT NULL,text TEXT NOT NULL,revision INTEGER NOT NULL,created_at TEXT NOT NULL,
 PRIMARY KEY(owner_id,session_id,event_id),FOREIGN KEY(owner_id,session_id) REFERENCES sessions ON DELETE CASCADE);
CREATE INDEX IF NOT EXISTS context_event_revision ON context_events(owner_id,session_id,task_id,revision DESC);
CREATE INDEX IF NOT EXISTS context_event_active ON context_event_changes(owner_id,session_id,status,event_id);
CREATE TABLE IF NOT EXISTS semantic_summaries(owner_id TEXT NOT NULL,session_id TEXT NOT NULL,summary_id TEXT NOT NULL,
 task_id TEXT NOT NULL,source_revision INTEGER NOT NULL,epoch INTEGER NOT NULL,covered_seq INTEGER NOT NULL,
 status TEXT NOT NULL,payload TEXT NOT NULL,created_at TEXT NOT NULL,
 PRIMARY KEY(owner_id,session_id,summary_id),FOREIGN KEY(owner_id,session_id) REFERENCES sessions ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS summary_attempts(owner_id TEXT NOT NULL,session_id TEXT NOT NULL,attempt_id TEXT NOT NULL,
 task_id TEXT NOT NULL,source_revision INTEGER NOT NULL,epoch INTEGER NOT NULL,status TEXT NOT NULL,
 error TEXT,created_at TEXT NOT NULL,input_sha256 TEXT NOT NULL DEFAULT '',PRIMARY KEY(owner_id,session_id,attempt_id),
 FOREIGN KEY(owner_id,session_id) REFERENCES sessions ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS history_index_failures(seq INTEGER PRIMARY KEY,reason TEXT NOT NULL,created_at TEXT NOT NULL,
 FOREIGN KEY(seq) REFERENCES session_turns(seq) ON DELETE CASCADE);
CREATE TABLE IF NOT EXISTS history_documents(seq INTEGER PRIMARY KEY,owner_id TEXT NOT NULL,session_id TEXT NOT NULL,
 analyzer TEXT NOT NULL,FOREIGN KEY(seq) REFERENCES session_turns(seq) ON DELETE CASCADE);
'''


class ContextRepositoryMixin:
    def _init_context(self):
        from core.history_index import index_turn
        with self._connect() as db:
            db.executescript(SCHEMA)
            db.execute('INSERT INTO context_schema SELECT 1 WHERE NOT EXISTS (SELECT 1 FROM context_schema)')
            if 'input_sha256' not in {row[1] for row in db.execute('PRAGMA table_info(summary_attempts)')}:
                db.execute("ALTER TABLE summary_attempts ADD COLUMN input_sha256 TEXT NOT NULL DEFAULT ''")
            if 'seen_seq' not in {row[1] for row in db.execute('PRAGMA table_info(context_event_changes)')}:
                db.execute('ALTER TABLE context_event_changes ADD COLUMN seen_seq INTEGER NOT NULL DEFAULT 0')
            db.execute('INSERT OR IGNORE INTO session_epochs SELECT owner_id,session_id,0,revision FROM sessions')
            db.execute("INSERT OR IGNORE INTO context_turn_tasks SELECT owner_id,session_id,turn_id,'default' FROM session_turns")
            db.execute("""INSERT OR IGNORE INTO context_facts
                SELECT f.owner_id,f.session_id,'default','task',f.fact_key,f.version,
                CASE WHEN f.version=(SELECT max(g.version) FROM task_facts g WHERE g.owner_id=f.owner_id AND g.session_id=f.session_id AND g.fact_key=f.fact_key)
                THEN 'active' ELSE 'superseded' END,f.kind,f.value,f.source_turn_id,t.query,f.updated_at
                FROM task_facts f JOIN session_turns t ON t.owner_id=f.owner_id AND t.session_id=f.session_id AND t.turn_id=f.source_turn_id""")
            self.fts_available = True
            try:
                db.execute('CREATE VIRTUAL TABLE IF NOT EXISTS history_fts USING fts5(terms)')
                db.execute('''CREATE TRIGGER IF NOT EXISTS history_delete AFTER DELETE ON session_turns
                    BEGIN DELETE FROM history_fts WHERE rowid=old.seq; END''')
                # Initial migration uses a bounded batch; later explicit rebuild
                # fills the remainder, and search exposes index lag diagnostics.
                import time
                deadline=time.monotonic()+0.5
                db.set_progress_handler(lambda:int(time.monotonic()>deadline),1000)
                try:
                    pending=db.execute('SELECT t.seq,length(t.query)+length(t.answer) AS chars FROM session_turns t LEFT JOIN history_documents d ON d.seq=t.seq WHERE d.seq IS NULL ORDER BY t.seq LIMIT 100').fetchall()
                except sqlite3.OperationalError as exc:
                    if 'interrupt' not in str(exc).lower():raise
                    pending=[]
                    self.history_index_degraded='migration_scan_time_budget_exhausted'
                finally:
                    db.set_progress_handler(None,0)
                total=0
                for item in pending:
                    if total+item['chars']>2_000_000:break
                    row=db.execute('SELECT * FROM session_turns WHERE seq=?',(item['seq'],)).fetchone()
                    db.execute('SAVEPOINT migrate_history_index')
                    try:
                        index_turn(db,row)
                        db.execute('RELEASE migrate_history_index')
                    except (sqlite3.OperationalError,ValueError) as exc:
                        db.execute('ROLLBACK TO migrate_history_index');db.execute('RELEASE migrate_history_index')
                        if any(word in str(exc).lower() for word in ('disk','readonly','locked','i/o','malformed')):raise
                        db.execute('INSERT OR REPLACE INTO history_index_failures VALUES (?,?,?)',(row['seq'],type(exc).__name__,now()))
                        self.history_index_degraded=type(exc).__name__
                    total+=item['chars']
            except sqlite3.OperationalError as exc:
                if 'fts5' not in str(exc).lower() and 'no such module' not in str(exc).lower():
                    raise
                self.fts_available = False
                db.execute('DROP TRIGGER IF EXISTS history_delete')

    def context_epoch(self, owner_id, session_id):
        from core.session_repository import _key
        with self._connect() as db:
            row = db.execute('SELECT epoch FROM session_epochs WHERE owner_id=? AND session_id=?',_key(owner_id,session_id)).fetchone()
        return row[0] if row else 0

    deletion_epoch = context_epoch

    def validate_context_epoch(self, owner_id, session_id, expected_epoch):
        from core.session_repository import SessionConflictError
        with self._connect() as db:
            exists=db.execute('SELECT 1 FROM sessions WHERE owner_id=? AND session_id=?',(owner_id,session_id)).fetchone()
        if not exists or type(expected_epoch) is not int or self.context_epoch(owner_id,session_id) != expected_epoch:
            raise SessionConflictError('session deletion epoch changed')
        return True

    def _bump_context(self, db, identity, *, invalidate=True):
        db.execute('UPDATE sessions SET revision=revision+1,updated_at=? WHERE owner_id=? AND session_id=?',(now(),*identity))
        if invalidate:
            db.execute('UPDATE semantic_summaries SET status="stale" WHERE owner_id=? AND session_id=? AND status="valid"',identity)
        db.execute('UPDATE session_epochs SET revision=(SELECT revision FROM sessions WHERE owner_id=? AND session_id=?) WHERE owner_id=? AND session_id=?',(*identity,*identity))
        return db.execute('SELECT revision FROM sessions WHERE owner_id=? AND session_id=?',identity).fetchone()[0]

    def _require_session(self, db, identity, expected_revision=None):
        from core.session_repository import SessionConflictError
        row=db.execute('SELECT revision FROM sessions WHERE owner_id=? AND session_id=?',identity).fetchone()
        if row is None:
            raise KeyError('session not found')
        if expected_revision is not None and row[0]!=expected_revision:
            raise SessionConflictError('session changed; reload before retrying')
        return row[0]

    def _active_task(self, db, identity):
        row=db.execute('SELECT task_id FROM context_tasks WHERE owner_id=? AND session_id=? AND active=1',identity).fetchone()
        return row[0] if row else 'default'

    def update_task(self, owner_id, session_id, *, task_id, goal='', objects=None, open_questions=None,
                    source_turn_id, expected_revision, explicit_user=False, progress=None):
        from core.session_repository import _key
        identity=_key(owner_id,session_id)
        _key(owner_id,task_id)
        if explicit_user is not True:
            raise PermissionError('task updates require an explicit user action')
        state={'task_id':task_id,'goal':goal,'objects':objects or [],'open_questions':open_questions or [],
               'progress':progress or [],'source_turn_id':source_turn_id,'authority':'user_note_not_execution_permission'}
        if not isinstance(goal,str) or len(json.dumps(state,ensure_ascii=False))>16000:
            raise ValueError('invalid or oversized task state')
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            self._require_session(db,identity,expected_revision)
            source=db.execute('SELECT query FROM session_turns WHERE owner_id=? AND session_id=? AND turn_id=?',(*identity,source_turn_id)).fetchone()
            if not source or not source[0].strip():
                raise KeyError('source user turn not found')
            revision=self._bump_context(db,identity)
            state.update(revision=revision,source_quote=source[0])
            db.execute('UPDATE context_tasks SET active=0 WHERE owner_id=? AND session_id=?',identity)
            db.execute('INSERT OR REPLACE INTO context_tasks VALUES (?,?,?,?,?,1)',(*identity,task_id,revision,json.dumps(state,ensure_ascii=False)))
        return state

    def set_task_fact(self, owner_id, session_id, key, value, *, source_turn_id, expected_version,
                      explicit_user=False, kind='constraint', scope='task', task_id=None, status='active'):
        from core.session_repository import _key,SessionConflictError
        identity=_key(owner_id,session_id)
        _key(owner_id,key)
        if explicit_user is not True and status!='candidate':
            raise PermissionError('task facts require an explicit user action')
        if scope not in {'task','session'} or status not in {'active','candidate','revoked'} or kind not in {'fact','constraint'} or type(expected_version) is not int or expected_version<0:
            raise ValueError('invalid fact update')
        encoded=json.dumps(value,ensure_ascii=False,allow_nan=False)
        if len(encoded)>8000:
            raise ValueError('task fact exceeds storage budget')
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            self._require_session(db,identity)
            task_id='' if scope=='session' else (task_id or self._active_task(db,identity))
            source=db.execute('SELECT query FROM session_turns WHERE owner_id=? AND session_id=? AND turn_id=?',(*identity,source_turn_id)).fetchone()
            if not source or not source[0].strip():
                raise KeyError('source user turn not found')
            params=(*identity,task_id,scope,key)
            old=db.execute('SELECT max(version) FROM context_facts WHERE owner_id=? AND session_id=? AND task_id=? AND scope=? AND fact_key=?',params).fetchone()[0] or 0
            if old!=expected_version:
                raise SessionConflictError('task fact changed; inspect its current version')
            # Pending candidates never displace confirmed facts. A confirmed
            # replacement supersedes every prior active/candidate version.
            if status!='candidate':
                db.execute('UPDATE context_facts SET status="superseded" WHERE owner_id=? AND session_id=? AND task_id=? AND scope=? AND fact_key=? AND status IN ("active","candidate")',params)
            if status in {'active','revoked'}:
                db.execute('UPDATE context_event_changes SET status="resolved" WHERE owner_id=? AND session_id=? AND fact_key=? AND event_id IN (SELECT event_id FROM context_events WHERE owner_id=? AND session_id=? AND task_id=?)',(*identity,key,*identity,task_id))
            stamp=now()
            db.execute('INSERT INTO context_facts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                       (*identity,task_id,scope,key,old+1,status,kind,encoded,source_turn_id,source[0],stamp))
            self._bump_context(db,identity)
        return {'key':key,'version':old+1,'kind':kind,'value':value,'source_turn_id':source_turn_id,
                'source_quote':source[0],'updated_at':stamp,'task_id':task_id,'scope':scope,'status':status,
                'authority':'user_note_not_execution_permission'}

    def task_facts(self, owner_id, session_id, *, include_history=False, task_id=None, include_candidates=False):
        from core.session_repository import _key
        identity=_key(owner_id,session_id)
        with self._connect() as db:
            task_id=task_id or self._active_task(db,identity)
            rows=db.execute('SELECT * FROM context_facts WHERE owner_id=? AND session_id=? AND (scope="session" OR task_id=?) ORDER BY fact_key,version',(*identity,task_id)).fetchall()
            # Migrated legacy notes keep their original sources and versions.
            legacy=db.execute('SELECT * FROM task_facts WHERE owner_id=? AND session_id=? ORDER BY fact_key,version',identity).fetchall() if not rows and task_id=='default' else []
        values=[]
        for row in rows:
            item=dict(row); item['key']=item.pop('fact_key'); item['value']=json.loads(item['value'])
            item.pop('owner_id');item.pop('session_id');item['authority']='user_note_not_execution_permission'
            if include_history or item['status']=='active' or (include_candidates and item['status']=='candidate'):
                values.append(item)
        if legacy:
            for row in legacy:
                item={'key':row['fact_key'],'version':row['version'],'kind':row['kind'],'value':json.loads(row['value']),
                      'source_turn_id':row['source_turn_id'],'updated_at':row['updated_at'],'status':'active','scope':'task','task_id':'default',
                      'authority':'user_note_not_execution_permission'}
                values.append(item)
            if not include_history:
                values=list({item['key']:item for item in values}.values())
        return values

    def revoke_task_fact(self, owner_id, session_id, key, *, source_turn_id, expected_version, explicit_user=False, **kwargs):
        return self.set_task_fact(owner_id,session_id,key,None,source_turn_id=source_turn_id,
            expected_version=expected_version,explicit_user=explicit_user,status='revoked',**kwargs)

    def record_current_request(self, owner_id, session_id, *, text, request_id, expected_revision=None):
        """Persist user wording; conservatively suppress constraints on correction.

        This is a trusted API hook, not a model-callable memory-writing tool.
        Ambiguous corrections remain events and cannot establish a new fact.
        """
        from core.session_repository import _key,SessionConflictError
        identity=_key(owner_id,session_id);_key(owner_id,request_id)
        if not isinstance(text,str) or not text.strip() or len(text)>200000:
            raise ValueError('invalid current request')
        correction=bool(re.search(r'改为|改成|更正|取消|撤销|不要再|不再|换个任务|新任务|instead|correction|no longer|revoke|cancel',text,re.I))
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            old=db.execute('SELECT * FROM context_events WHERE owner_id=? AND session_id=? AND event_id=?',(*identity,request_id)).fetchone()
            if old:
                if old['text']!=text: raise SessionConflictError('request ID already has different text')
                event=dict(old)
                event['event_revision']=event['revision']
                event['revision']=self._require_session(db,identity,expected_revision)
                return event
            self._require_session(db,identity,expected_revision)
            task_id=self._active_task(db,identity)
            revision=self._bump_context(db,identity,invalidate=correction)
            kind='current_user_correction' if correction else 'current_user_request'
            db.execute('INSERT INTO context_events VALUES (?,?,?,?,?,?,?,?)',(*identity,request_id,task_id,kind,text,revision,now()))
            if correction:
                aliases={'budget':['预算','金额','budget'],'date':['日期','截止','date','deadline'],
                         'language':['语言','中文','英文','language'],'send':['发送','send']}
                rows=db.execute('SELECT DISTINCT fact_key FROM context_facts WHERE owner_id=? AND session_id=? AND (task_id=? OR scope="session") AND kind="constraint" AND status="active"',(*identity,task_id)).fetchall()
                affected={key for key,words in aliases.items() if any(word.casefold() in text.casefold() for word in [key,*words])}
                for row in rows:
                    key=row[0]
                    words=[key]+aliases.get(key.casefold(),[])
                    if any(word.casefold() in text.casefold() for word in words):
                        affected.add(key)
                        db.execute('UPDATE context_facts SET status="superseded" WHERE owner_id=? AND session_id=? AND (task_id=? OR scope="session") AND fact_key=? AND kind="constraint" AND status="active"',(*identity,task_id,key))
                for key in affected:
                    db.execute('UPDATE context_event_changes SET status="superseded" WHERE owner_id=? AND session_id=? AND fact_key=? AND event_id IN (SELECT event_id FROM context_events WHERE owner_id=? AND session_id=? AND task_id=?)',(*identity,key,*identity,task_id))
                    db.execute('INSERT INTO context_event_changes VALUES (?,?,?,?,?,?)',(*identity,request_id,key,'active',db.execute('SELECT coalesce(max(seq),0) FROM session_turns WHERE owner_id=? AND session_id=?',identity).fetchone()[0]))
        return {'event_id':request_id,'task_id':task_id,'event_type':kind,'text':text,'revision':revision}

    def context_state(self, owner_id, session_id):
        from core.session_repository import _key
        identity=_key(owner_id,session_id)
        with self._connect() as db:
            task_id=self._active_task(db,identity)
            task_row=db.execute('SELECT state FROM context_tasks WHERE owner_id=? AND session_id=? AND task_id=?',(*identity,task_id)).fetchone()
            # Active corrections are state, not a recent-history window. Select
            # their IDs first so old still-effective requirements cannot age out.
            active_predicate="EXISTS (SELECT 1 FROM context_event_changes c WHERE c.owner_id=e.owner_id AND c.session_id=e.session_id AND c.event_id=e.event_id AND c.status='active')"
            projection='SELECT e.event_id,e.revision,length(e.text) AS text_chars FROM context_events e WHERE e.owner_id=? AND e.session_id=? AND e.task_id=? '
            active=db.execute(projection+'AND '+active_predicate+' ORDER BY e.revision DESC LIMIT 101',(*identity,task_id)).fetchall()
            inactive=db.execute(projection+'AND e.event_type="current_user_correction" AND NOT '+active_predicate+' ORDER BY e.revision DESC LIMIT 20',(*identity,task_id)).fetchall()
            ordinary=db.execute(projection+'AND e.event_type="current_user_request" ORDER BY e.revision DESC LIMIT 5',(*identity,task_id)).fetchall()
            events=[];required_omissions=[];event_chars=0
            if len(active)>100:
                required_omissions.append({'material_id':'active_user_events','reason':'active_user_event_count_limit',
                    'required':True,'limit':100,'omitted_count_at_least':len(active)-100})
            for is_active,rows in ((True,active[:100]),(False,inactive),(False,ordinary)):
                for item in rows:
                    if event_chars+item['text_chars']>100000:
                        if is_active:
                            required_omissions.append({'material_id':'event:'+item['event_id'],
                                'reason':'active_user_event_char_limit','required':True,'limit':100000})
                        continue
                    row=db.execute('SELECT event_id,event_type,text,revision,task_id FROM context_events WHERE owner_id=? AND session_id=? AND event_id=?',(*identity,item['event_id'])).fetchone()
                    event=dict(row);event_chars+=item['text_chars']
                    changes=db.execute('SELECT fact_key,status FROM context_event_changes WHERE owner_id=? AND session_id=? AND event_id=?',(*identity,event['event_id'])).fetchall()
                    event['affected_keys']=[c['fact_key'] for c in changes if c['status']=='active']
                    event['status']='active' if event['affected_keys'] else ('superseded' if changes else 'unresolved')
                    event['requires_protection']=event['event_type']=='current_user_correction' and event['status']=='active'
                    events.append(event)
            events.sort(key=lambda event:event['revision'],reverse=True)
        facts=self.task_facts(*identity,include_candidates=True)
        return {'revision':self.revision(*identity),'deletion_epoch':self.context_epoch(*identity),'task_id':task_id,
                'task_state':json.loads(task_row[0]) if task_row else {'task_id':task_id,'revision':0},
                'facts':[f for f in facts if f['status']=='active'],'candidates':[f for f in facts if f['status']=='candidate'],
                'events':[dict(e) for e in events],
                'required_omissions':required_omissions,'user_event_overflow':bool(required_omissions),
                'user_event_chars':event_chars,
                'user_events':[dict(e) for e in events if e['task_id']==task_id],'summary':self.get_semantic_summary(*identity)}

    def get_semantic_summary(self, owner_id, session_id):
        from core.session_summary import get_summary
        return get_summary(self,owner_id,session_id)

    def generate_summary(self, owner_id, session_id, *, generate, budget=None, force=False, **kwargs):
        from core.session_summary import generate_summary
        return generate_summary(self,owner_id,session_id,generate=generate,budget=budget,force=force,**kwargs)

    def summary_attempts(self, owner_id, session_id, *, limit=20):
        from core.session_repository import _key,_limit
        with self._connect() as db:
            rows=db.execute('SELECT * FROM summary_attempts WHERE owner_id=? AND session_id=? ORDER BY created_at DESC LIMIT ?',(*_key(owner_id,session_id),_limit(limit))).fetchall()
        return [{k:r[k] for k in ('attempt_id','task_id','source_revision','epoch','status','error','created_at','input_sha256')} for r in rows]

    def rebuild_history_index(self, *, batch_size=100, max_chars=2_000_000, after=0):
        from core.history_index import index_turn
        import time
        if type(batch_size) is not int or not 1<=batch_size<=1000 or type(max_chars) is not int or not 1<=max_chars<=10_000_000 or type(after) is not int or after<0:
            raise ValueError('invalid rebuild limits')
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE')
            rows=db.execute('SELECT seq,length(query)+length(answer) AS chars FROM session_turns WHERE seq>? ORDER BY seq LIMIT ?',(after,batch_size+1)).fetchall()
            count=0;processed=0;size=0;last=after;deadline=time.monotonic()+1.0
            for item in rows[:batch_size]:
                if item['chars']>max_chars:
                    db.execute('INSERT OR REPLACE INTO history_index_failures VALUES (?,?,?)',(item['seq'],'document_exceeds_rebuild_budget',now()))
                    last=item['seq'];processed+=1;continue
                if size+item['chars']>max_chars or time.monotonic()>deadline:break
                row=db.execute('SELECT * FROM session_turns WHERE seq=?',(item['seq'],)).fetchone()
                index_turn(db,row)
                db.execute('DELETE FROM history_index_failures WHERE seq=?',(item['seq'],))
                count+=1;processed+=1;size+=item['chars'];last=item['seq']
        return {'indexed_turns':count,'analyzer_version':'cjk-bigram-v1','has_more':processed<len(rows),
                'next_after':last,'scan_chars':size,'degraded_reason':'rebuild_budget_exhausted' if count<len(rows) else None,'processed_turns':processed}

    def list_tasks(self, owner_id, session_id):
        from core.session_repository import _key
        with self._connect() as db:
            rows=db.execute('SELECT task_id,revision,state,active FROM context_tasks WHERE owner_id=? AND session_id=? ORDER BY revision DESC LIMIT 200',_key(owner_id,session_id)).fetchall()
        return [{**json.loads(row['state']),'active':bool(row['active'])} for row in rows]

    def select_task(self, owner_id, session_id, *, task_id, expected_revision, explicit_user=False):
        from core.session_repository import _key
        if explicit_user is not True: raise PermissionError('task selection requires explicit user action')
        identity=_key(owner_id,session_id)
        with self._connect() as db:
            db.execute('BEGIN IMMEDIATE');self._require_session(db,identity,expected_revision)
            row=db.execute('SELECT state FROM context_tasks WHERE owner_id=? AND session_id=? AND task_id=?',(*identity,task_id)).fetchone()
            if row is None: raise KeyError('task not found')
            db.execute('UPDATE context_tasks SET active=0 WHERE owner_id=? AND session_id=?',identity)
            revision=self._bump_context(db,identity)
            state=json.loads(row[0]);state['revision']=revision
            db.execute('UPDATE context_tasks SET active=1,revision=?,state=? WHERE owner_id=? AND session_id=? AND task_id=?',(revision,json.dumps(state,ensure_ascii=False),*identity,task_id))
        return state

    def confirm_task_fact(self, owner_id, session_id, key, *, expected_version, source_turn_id, explicit_user=False, scope='task', task_id=None):
        candidates=self.task_facts(owner_id,session_id,include_history=True,task_id=task_id)
        fact=next((f for f in candidates if f['key']==key and f['version']==expected_version and f['scope']==scope and f['status']=='candidate'),None)
        if fact is None: raise KeyError('candidate not found')
        return self.set_task_fact(owner_id,session_id,key,fact['value'],source_turn_id=source_turn_id,
            expected_version=expected_version,explicit_user=explicit_user,scope=scope,task_id=task_id,kind=fact['kind'])

    def delete_task_fact(self, owner_id, session_id, key, **kwargs):
        return self.revoke_task_fact(owner_id,session_id,key,**kwargs)

    def extract_candidates(self, owner_id, session_id, **kwargs):
        from core.session_candidates import extract_candidates
        return extract_candidates(self,owner_id,session_id,**kwargs)
