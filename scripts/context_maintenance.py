"""Offline context migration rehearsal and idempotent derived-state cleanup."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from scripts.state_maintenance import _stopped, _open, _validate
from scripts.bundle_files import reject_links


def transcript_fingerprint(db):
    digest=hashlib.sha256();count=0
    for row in db.execute('SELECT owner_id,session_id,turn_id,query,answer,metadata,include_context,created_at FROM session_turns ORDER BY seq'):
        digest.update(json.dumps(list(row),ensure_ascii=False,separators=(',',':')).encode('utf-8'))
        digest.update(b'\n');count+=1
    return {'turns':count,'sha256':digest.hexdigest()}


def migrate_copy(source,destination,*,service_stopped=False):
    """Rehearse additive migration on a fresh SQLite backup, preserving originals."""
    _stopped(service_stopped)
    source=reject_links(source);destination=Path(destination).absolute()
    if destination.exists(): raise ValueError('migration destination must be new')
    ancestor=destination.parent
    while not ancestor.exists(): ancestor=ancestor.parent
    reject_links(ancestor)
    from scripts.context_paths import check_context_path
    check=check_context_path(destination,probe=True)
    if check['status']=='error': raise ValueError('context destination is not writable')
    destination.parent.mkdir(parents=True,exist_ok=True)
    before_db=_open(source,readonly=True)
    target=sqlite3.connect(destination)
    try:
        _validate(before_db,'sessions');before=transcript_fingerprint(before_db)
        before_db.backup(target)
    finally:
        target.close();before_db.close()
    from core.session_repository import SessionRepository
    repo=SessionRepository(destination)
    with repo._connect() as db:
        _validate(db,'sessions');after=transcript_fingerprint(db)
        versions=[row[0] for row in db.execute('SELECT version FROM context_schema')]
    if before!=after: raise ValueError('migration changed original transcripts')
    report={'status':'verified_copy','original_preserved':True,'schema_versions':versions,
            'transcript':after,'fts_available':repo.fts_available,'path_check':check}
    destination.with_suffix(destination.suffix+'.migration.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def _cleanup_rows(db, *, jobs=False):
    """Keyset batches bound payload memory while this connection mutates rows."""
    query = ("SELECT rowid,id,owner,request,checkpoint,error_code FROM jobs WHERE kind='agent' AND rowid>? ORDER BY rowid LIMIT 8"
             if jobs else "SELECT rowid,id,owner,session,epoch FROM tool_results WHERE rowid>? ORDER BY rowid LIMIT 256")
    after = 0
    while True:
        rows = db.execute(query, (after,)).fetchall()
        if not rows:
            return
        after = rows[-1][0]
        for row in rows:
            yield row[1:]


def cleanup_context(sessions_path,*,tool_results_path=None,jobs_path=None,apply=False,service_stopped=False):
    """Use session tombstones to clean other stores after an interrupted delete.

    Stopped writers are required; each store commits independently and repeating
    this operation is safe. No approvals are granted or operations retried.
    """
    _stopped(service_stopped)
    if type(apply) is not bool: raise ValueError('apply must be boolean')
    report={'dry_run':not apply,'tool_results_eligible':0,'jobs_eligible':0,'changed':0}
    sessions=_open(sessions_path,readonly=True)
    try:
        _validate(sessions,'sessions')
        names={row[0] for row in sessions.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'session_epochs' not in names: raise ValueError('rehearse context migration before derived cleanup')
        def state(owner,sid):
            return sessions.execute('''SELECT e.epoch,s.session_id FROM session_epochs e LEFT JOIN sessions s
                ON s.owner_id=e.owner_id AND s.session_id=e.session_id WHERE e.owner_id=? AND e.session_id=?''',(owner,sid)).fetchone()
        if tool_results_path:
            results=_open(tool_results_path,readonly=not apply)
            try:
                _validate(results,'tool_results')
                with results:
                    for rid,owner,sid,epoch in _cleanup_rows(results):
                        current=state(owner,sid)
                        if current is None or current[1] is None or current[0]!=epoch:
                            report['tool_results_eligible']+=1
                            if apply:
                                results.execute('DELETE FROM tool_results WHERE id=?',(rid,));report['changed']+=1
            finally: results.close()
        if jobs_path:
            jobs=_open(jobs_path,readonly=not apply)
            try:
                _validate(jobs,'jobs')
                with jobs:
                    for job_id,owner,request,checkpoint,error in _cleanup_rows(jobs,jobs=True):
                        req=json.loads(request);sid=req.get('session_id')
                        if not sid: continue
                        current=state(owner,sid);cp=json.loads(checkpoint) if checkpoint else {}
                        old_epoch=req.get('_session_epoch',cp.get('context_epoch',0))
                        stale=(current is not None and (current[1] is None or old_epoch!=current[0]))
                        if stale and error!='session_invalidated':
                            report['jobs_eligible']+=1
                            if apply:
                                jobs.execute("""UPDATE jobs SET status='cancelled',request=?,result=NULL,checkpoint=NULL,
                                    progress='{}',cancel_requested=1,error_code='session_invalidated' WHERE id=? AND owner=?""",
                                    (json.dumps({'session_id':sid}),job_id,owner));report['changed']+=1
            finally: jobs.close()
    finally: sessions.close()
    return report


def rebuild_index(path,*,service_stopped=False,max_batches=1000):
    _stopped(service_stopped)
    if type(max_batches) is not int or not 1<=max_batches<=100000:
        raise ValueError('invalid rebuild batch limit')
    from core.session_repository import SessionRepository
    repo=SessionRepository(reject_links(path))
    after=total=processed=0;has_more=True
    for batch in range(max_batches):
        result=repo.rebuild_history_index(after=after)
        total+=result['indexed_turns'];processed+=result['processed_turns'];has_more=result['has_more']
        if not has_more:break
        if result['next_after']<=after:break
        after=result['next_after']
    with repo._connect() as db:
        failures=db.execute('SELECT count(*) FROM history_index_failures').fetchone()[0]
    return {'indexed_turns':total,'processed_turns':processed,'has_more':has_more,
            'next_after':result['next_after'],'index_failures':failures,
            'status':'complete' if not has_more and not failures else 'partial','batches':batch+1}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    migrate=sub.add_parser('migrate-copy');migrate.add_argument('--source',required=True);migrate.add_argument('--destination',required=True)
    rebuild=sub.add_parser('rebuild-index');rebuild.add_argument('--sessions',required=True)
    rebuild.add_argument('--max-batches',type=int,default=1000)
    clean=sub.add_parser('cleanup');clean.add_argument('--sessions',required=True)
    clean.add_argument('--tool-results');clean.add_argument('--jobs');clean.add_argument('--apply',action='store_true')
    for command in (migrate,rebuild,clean): command.add_argument('--service-stopped',action='store_true')
    args=parser.parse_args(argv);_stopped(args.service_stopped)
    if args.command=='migrate-copy': result=migrate_copy(args.source,args.destination,service_stopped=True)
    elif args.command=='cleanup':
        result=cleanup_context(args.sessions,tool_results_path=args.tool_results,jobs_path=args.jobs,apply=args.apply,service_stopped=True)
    else:
        result=rebuild_index(args.sessions,service_stopped=True,max_batches=args.max_batches)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
