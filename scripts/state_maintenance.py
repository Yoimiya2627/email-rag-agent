"""Explicit local SQLite backup, quarantine restore and retention. No config/OAuth imports."""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time
try:
    from scripts.bundle_files import (reject_links,inventory,copy_artifacts,validate_artifacts,restore_artifacts)
except ModuleNotFoundError:
    from bundle_files import (reject_links,inventory,copy_artifacts,validate_artifacts,restore_artifacts)

TABLES={'approvals':{'approvals','approval_metadata','approval_reconciliations'},
        'jobs':{'jobs'},'sessions':{'sessions','session_turns','task_facts','turn_evidence'},
        'tool_results':{'tool_results'}}


def _stopped(value):
    if value is not True:
        raise ValueError('Stop API, UI, MCP and workers, then explicitly attest service_stopped=True')


def _open(path, *, readonly=False, snapshot=False):
    path=reject_links(path)
    if not path.is_file(): raise ValueError('state database must already exist')
    if snapshot:
        # A bundle member is a standalone, hashed SQLite snapshot. Never replay
        # sidecar writes that were not covered by that member's checksum.
        if any(Path(str(path)+suffix).exists() for suffix in ('-wal','-shm','-journal')):
            raise ValueError('Backup SQLite snapshot has unexpected sidecar files')
    mode='?mode=ro&immutable=1' if snapshot else ('?mode=ro' if readonly else '?mode=rw')
    db=sqlite3.connect(path.as_uri()+mode,uri=True,timeout=5)
    db.execute('PRAGMA foreign_keys=ON')
    return db


def _validate(db,kind):
    if kind not in TABLES: raise ValueError('unsupported state kind')
    tables={row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not TABLES[kind].issubset(tables): raise ValueError('database schema does not match state kind')
    if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok': raise ValueError('SQLite integrity check failed')


def _digest(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda:source.read(1_048_576),b''): digest.update(chunk)
    return digest.hexdigest()


def backup_states(sources: dict, destination, *, service_stopped=False, artifacts=None,
                  max_files=100000,max_bytes=10*1024**3):
    """Each SQLite backup is consistent; cross-store consistency needs stopped writers."""
    _stopped(service_stopped)
    artifacts=artifacts or {}
    if (not sources and not artifacts) or not set(sources).issubset(TABLES): raise ValueError('specify known state or artifact kinds')
    plans=inventory(artifacts,max_files=max_files,max_bytes=max_bytes)
    paths={kind:reject_links(path) for kind,path in sources.items()}
    if len(set(paths.values()))!=len(paths): raise ValueError('state kinds must use distinct databases')
    snapshot_sizes={}
    for kind,path in paths.items():
        with closing(_open(path,readonly=True)) as db:
            _validate(db,kind)
            # page_count includes committed WAL pages absent from the main file.
            snapshot_sizes[kind]=db.execute('PRAGMA page_count').fetchone()[0]*db.execute('PRAGMA page_size').fetchone()[0]
    artifact_bytes=sum(plan['total_bytes'] for plan in plans.values())
    if sum(snapshot_sizes.values())+artifact_bytes>max_bytes:
        raise ValueError('Backup exceeds its byte budget')
    destination=Path(destination).absolute()
    # Validate existing ancestors before creating a new destination.
    ancestor=destination.parent
    while not ancestor.exists(): ancestor=ancestor.parent
    reject_links(ancestor)
    destination.mkdir(parents=True,exist_ok=False)
    manifest={'format':1,'created_at':datetime.now(timezone.utc).isoformat(),'service_stopped_attested':True,'states':{},'artifacts':plans}
    for kind,path in paths.items():
        output=destination/(kind+'.sqlite3')
        source=_open(path,readonly=True)
        target=sqlite3.connect(output)
        try:
            page_size=source.execute('PRAGMA page_size').fetchone()[0]
            available=max_bytes-artifact_bytes-sum(item['bytes'] for item in manifest['states'].values())
            def bound_snapshot(status,remaining,total):
                if total*page_size>available:
                    raise ValueError('SQLite snapshot grew beyond the backup byte budget')
            source.backup(target,pages=1,progress=bound_snapshot)
            _validate(target,kind)
        finally:
            target.close(); source.close()
        manifest['states'][kind]={'file':output.name,'sha256':_digest(output),'bytes':output.stat().st_size}
    if sum(item['bytes'] for item in manifest['states'].values())+sum(plan['total_bytes'] for plan in plans.values())>max_bytes:
        raise ValueError('SQLite snapshot exceeds the backup byte budget')
    copy_artifacts(artifacts,plans,destination)
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest


def restore_states(backup_directory,destination,*,service_stopped=False,max_files=100000,max_bytes=10*1024**3):
    """Restore only to a new directory; never replace configured live databases."""
    _stopped(service_stopped)
    backup_directory=reject_links(backup_directory)
    manifest_path=reject_links(backup_directory/'manifest.json')
    if manifest_path.stat().st_size>64*1024**2: raise ValueError('Backup manifest exceeds its read budget')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('format')!=1 or not (manifest.get('states') or manifest.get('artifacts')) or not set(manifest.get('states',{})).issubset(TABLES):
        raise ValueError('invalid backup manifest')
    plans=validate_artifacts(backup_directory,manifest.get('artifacts',{}),max_files=max_files,max_bytes=max_bytes)
    stored_bytes=sum(item['bytes'] for plan in plans.values() for item in plan['files'])
    for kind,item in manifest['states'].items():
        if item.get('file')!=kind+'.sqlite3': raise ValueError('invalid backup member name')
        path=backup_directory/item['file']
        reject_links(path)
        stored_bytes+=path.stat().st_size
        if stored_bytes>max_bytes: raise ValueError('Restore exceeds its byte budget')
        if _digest(path)!=item.get('sha256'): raise ValueError('backup checksum mismatch')
        with closing(_open(path,readonly=True,snapshot=True)) as db: _validate(db,kind)
    destination=Path(destination).absolute()
    ancestor=destination.parent
    while not ancestor.exists(): ancestor=ancestor.parent
    reject_links(ancestor)
    destination.mkdir(parents=True,exist_ok=False)
    report={'format':1,'quarantined_approvals':0,'disabled_job_checkpoints':0,'states':{}}
    for kind,item in manifest['states'].items():
        output=destination/item['file']
        source=_open(backup_directory/item['file'],readonly=True,snapshot=True)
        target=sqlite3.connect(output)
        try:
            source.backup(target)
            with target:
                if kind=='approvals':
                    report['quarantined_approvals']=target.execute("UPDATE approvals SET status='unknown',execution_state='unknown',claim_token=NULL,error_code='restored_requires_reconciliation',updated_at=? WHERE status IN ('pending','executing','unknown')",(time.time(),)).rowcount
                elif kind=='jobs':
                    report['disabled_job_checkpoints']=target.execute('SELECT COUNT(*) FROM jobs WHERE checkpoint IS NOT NULL').fetchone()[0]
                    target.execute("UPDATE jobs SET checkpoint=NULL,cancel_requested=1,status=CASE WHEN status IN ('queued','running') THEN 'cancelled' ELSE status END,error_code='restored_execution_disabled'")
                elif kind=='sessions':
                    names={row[0] for row in target.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                    if 'session_epochs' in names:
                        target.execute('UPDATE session_epochs SET epoch=epoch+1,revision=revision+1')
                        target.execute('UPDATE sessions SET revision=revision+1')
                    if 'semantic_summaries' in names:
                        target.execute("UPDATE semantic_summaries SET status='stale'")
                elif kind=='tool_results':
                    # Restored snapshots do not silently become resumable live
                    # tool references. The original backup still has its hashes.
                    target.execute('UPDATE tool_results SET expires=0')
            _validate(target,kind)
        finally:
            target.close(); source.close()
        report['states'][kind]={'file':output.name,'sha256':_digest(output)}
    restore_artifacts(backup_directory,plans,destination)
    report['artifacts']=plans
    (destination/'restore-report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def retain_private_state(path,kind,*,owner_id,before,apply=False,service_stopped=False):
    """Offline job tombstone/session retention; dry-run is the default."""
    _stopped(service_stopped)
    if kind not in {'jobs','sessions'}: raise ValueError('use ApprovalStore.retain for approvals')
    if not isinstance(owner_id,str) or not owner_id or len(owner_id)>256: raise ValueError('invalid owner')
    if type(before) not in (float,int) or not math.isfinite(before) or not 0<=before<=time.time():
        raise ValueError('before must be a past Unix timestamp')
    if type(apply) is not bool: raise ValueError('apply must be boolean')
    db=_open(path,readonly=not apply)
    try:
        _validate(db,kind)
        with db:
            if apply: db.execute('BEGIN IMMEDIATE')
            if kind=='jobs':
                where="owner=? AND updated<? AND status NOT IN ('running','queued','interrupted')"
                args=(owner_id,before)
                count=db.execute('SELECT COUNT(*) FROM jobs WHERE '+where,args).fetchone()[0]
                if apply:
                    db.execute("UPDATE jobs SET request='{}',result=NULL,checkpoint=NULL,progress='{}',error_code='retention_redacted' WHERE "+where,args)
            else:
                cutoff=datetime.fromtimestamp(before,timezone.utc).isoformat()
                args=(owner_id,cutoff)
                count=db.execute('SELECT COUNT(*) FROM sessions WHERE owner_id=? AND updated_at<?',args).fetchone()[0]
                if apply:
                    names={row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                    if 'session_epochs' in names:
                        db.execute('''UPDATE session_epochs SET epoch=epoch+1,revision=revision+1
                            WHERE owner_id=? AND session_id IN
                            (SELECT session_id FROM sessions WHERE owner_id=? AND updated_at<?)''',
                            (owner_id,owner_id,cutoff))
                    db.execute('DELETE FROM sessions WHERE owner_id=? AND updated_at<?',args)
        return {'kind':kind,'dry_run':not apply,'eligible':count,'changed':count if apply else 0}
    finally: db.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    backup=sub.add_parser('backup')
    for kind in TABLES: backup.add_argument('--'+kind)
    backup.add_argument('--corpus')
    backup.add_argument('--raw-dir')
    backup.add_argument('--index-dir')
    backup.add_argument('--destination',required=True)
    restore=sub.add_parser('restore')
    restore.add_argument('--backup-directory',required=True)
    restore.add_argument('--destination',required=True)
    retain=sub.add_parser('retain')
    retain.add_argument('--kind',choices=['jobs','sessions'],required=True)
    retain.add_argument('--path',required=True)
    retain.add_argument('--owner',required=True)
    retain.add_argument('--before',type=float,required=True)
    retain.add_argument('--apply',action='store_true')
    for command in (backup,restore,retain): command.add_argument('--service-stopped',action='store_true')
    for command in (backup,restore):
        command.add_argument('--max-files',type=int,default=100000)
        command.add_argument('--max-bytes',type=int,default=10*1024**3)
    args=parser.parse_args(argv)
    if args.command=='backup':
        result=backup_states({kind:getattr(args,kind) for kind in TABLES if getattr(args,kind)},args.destination,
            service_stopped=args.service_stopped,artifacts={kind:path for kind,path in
                {'corpus':args.corpus,'raw':args.raw_dir,'index':args.index_dir}.items() if path},
            max_files=args.max_files,max_bytes=args.max_bytes)
    elif args.command=='restore':
        result=restore_states(args.backup_directory,args.destination,service_stopped=args.service_stopped,
                              max_files=args.max_files,max_bytes=args.max_bytes)
    else:
        result=retain_private_state(args.path,args.kind,owner_id=args.owner,before=args.before,apply=args.apply,service_stopped=args.service_stopped)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
