import json
import os
from pathlib import Path
import subprocess
import sys
import sqlite3
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
from scripts.state_maintenance import backup_states,restore_states


def test_unmanifested_sqlite_wal_is_rejected_before_restore(tmp_path):
    from core.jobs import JobStore
    store=JobStore(tmp_path/'jobs.sqlite3')
    store.create('local','agent',{'q':'one'},'one')
    backup=tmp_path/'backup'
    backup_states({'jobs':store.path},backup,service_stopped=True)
    member=backup/'jobs.sqlite3'
    with sqlite3.connect(member) as db:
        db.execute('PRAGMA journal_mode=WAL')
    # Record the legitimate standalone snapshot after setting journal mode.
    from scripts.bundle_files import digest
    manifest=json.loads((backup/'manifest.json').read_text())
    manifest['states']['jobs']['sha256']=digest(member)
    (backup/'manifest.json').write_text(json.dumps(manifest))
    db=sqlite3.connect(member)
    try:
        db.execute('PRAGMA wal_autocheckpoint=0')
        db.execute("UPDATE jobs SET request='UNMANIFESTED_WAL_ROW'")
        db.commit()
        assert digest(member)==manifest['states']['jobs']['sha256']
        with pytest.raises(ValueError,match='sidecar'):
            restore_states(backup,tmp_path/'restore',service_stopped=True)
        assert not (tmp_path/'restore').exists()
    finally:
        db.close()


def test_sqlite_wal_size_counts_before_backup_directory_creation(tmp_path):
    from core.jobs import JobStore
    store=JobStore(tmp_path/'jobs.sqlite3')
    db=sqlite3.connect(store.path)
    try:
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('PRAGMA wal_autocheckpoint=0')
        db.execute('CREATE TABLE payload(data BLOB)')
        db.execute('INSERT INTO payload VALUES (zeroblob(500000))')
        db.commit()
        assert store.path.stat().st_size<50000
        with pytest.raises(ValueError,match='budget'):
            backup_states({'jobs':store.path},tmp_path/'backup',service_stopped=True,max_bytes=50000)
        assert not (tmp_path/'backup').exists()
    finally:
        db.close()


def test_bundle_file_budget_stops_lazy_directory_scan_and_closes_iterator(tmp_path,monkeypatch):
    from scripts.bundle_files import inventory
    root=tmp_path/'raw';root.mkdir()
    for number in range(2):
        (root/f'f{number}').write_text('x')
    visited=[];closed=[]
    original=os.scandir
    @contextmanager
    def scan(path):
        if Path(path)!=root:
            with original(path) as entries:
                yield entries
            return
        def entries():
            for number in range(1000000):
                visited.append(number)
                yield SimpleNamespace(name=f'f{number}',is_dir=lambda **kw:False)
        try:
            yield entries()
        finally:
            closed.append(True)
    monkeypatch.setattr(os,'scandir',scan)
    with pytest.raises(ValueError,match='budget'):
        inventory({'raw':root},max_files=1)
    assert visited==[0,1] and closed==[True]


def test_corpus_raw_and_index_files_restore_exactly(tmp_path):
    corpus=tmp_path/'emails.json';corpus.write_text('[{"id":"synthetic"}]')
    raw=tmp_path/'raw';raw.mkdir();(raw/'e.json').write_text('{"body":"private"}')
    index=tmp_path/'index';index.mkdir();(index/'index.dat').write_bytes(b'\x00\x01synthetic')
    destination=tmp_path/'backup'
    manifest=backup_states({},destination,service_stopped=True,artifacts={'corpus':corpus,'raw':raw,'index':index})
    assert set(manifest['artifacts'])=={'corpus','raw','index'}
    restored=tmp_path/'restore'
    restore_states(destination,restored,service_stopped=True)
    assert (restored/'artifacts'/'corpus'/'data.json').read_bytes()==corpus.read_bytes()
    assert (restored/'artifacts'/'raw'/'e.json').read_bytes()==(raw/'e.json').read_bytes()
    assert (restored/'artifacts'/'index'/'index.dat').read_bytes()==(index/'index.dat').read_bytes()


def test_budget_and_changed_file_fail_before_restore_destination_exists(tmp_path):
    corpus=tmp_path/'data.json';corpus.write_text('private body')
    with pytest.raises(ValueError,match='budget'):
        backup_states({},tmp_path/'oversize',service_stopped=True,artifacts={'corpus':corpus},max_bytes=3)
    assert not (tmp_path/'oversize').exists()
    backup=tmp_path/'backup'
    backup_states({},backup,service_stopped=True,artifacts={'corpus':corpus})
    (backup/'artifacts'/'corpus'/'data.json').write_text('different')
    with pytest.raises(ValueError,match='checksum'):
        restore_states(backup,tmp_path/'restore',service_stopped=True)
    assert not (tmp_path/'restore').exists()


def test_manifest_parent_traversal_is_rejected_before_writing(tmp_path):
    raw=tmp_path/'raw';raw.mkdir();(raw/'e.json').write_text('{}')
    backup=tmp_path/'backup'
    backup_states({},backup,service_stopped=True,artifacts={'raw':raw})
    manifest=json.loads((backup/'manifest.json').read_text())
    manifest['artifacts']['raw']['files'][0]['path']='../../outside.json'
    (backup/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='member path'):
        restore_states(backup,tmp_path/'restore',service_stopped=True)
    assert not (tmp_path/'restore').exists()


def test_symlink_source_is_rejected(tmp_path):
    source=tmp_path/'data.json';source.write_text('{}')
    link=tmp_path/'alias.json'
    try:
        link.symlink_to(source)
    except OSError:
        pytest.skip('Creating symlinks is not enabled for this Windows account')
    with pytest.raises(ValueError,match='symlinks'):
        backup_states({},tmp_path/'backup',service_stopped=True,artifacts={'corpus':link})
    assert not (tmp_path/'backup').exists()


def test_actual_chroma_generation_recovers_in_fresh_process(tmp_path):
    pytest.importorskip('chromadb')
    root=Path(__file__).resolve().parents[1]
    script='''
import json,os,sys
os.environ.update(CHROMA_PERSIST_DIR=sys.argv[1],EMBEDDING_MODEL='synthetic',EMBEDDING_MODEL_REVISION='test-commit',EMBEDDING_DIMENSION='2',PYTHON_DOTENV_DISABLED='1')
from core import embedder
from core.chunker import chunk_email
from models.schemas import Email
embedder.embed_texts=lambda texts:[[1.0,0.0] for text in texts]
if sys.argv[2]=='build':
    client=embedder._get_client()
    create=client.get_or_create_collection
    def persisted_collection(*args,**kwargs):
        kwargs['metadata']={**kwargs.get('metadata',{}),'hnsw:batch_size':3,'hnsw:sync_threshold':3}
        return create(*args,**kwargs)
    client.get_or_create_collection=persisted_collection
    chunks=[]
    for number in range(3):
        email=Email(id=str(number),subject='synthetic',sender='a@example.invalid',recipients=[],date='2026-09-10',body='Synthetic evidence. '*10)
        chunks.extend(chunk_email(email))
    embedder.index_chunks(chunks,replace=True)
stats=embedder.verify_collection_readiness()
print(json.dumps({'count':stats['chunk_count'],'email_count':stats['email_count']}))
'''
    source=tmp_path/'chroma'
    def run(path,mode):
        result=subprocess.run([sys.executable,'-c',script,str(path),mode],cwd=root,
                              capture_output=True,text=True,timeout=30)
        assert result.returncode==0,result.stderr[-3000:]
        return json.loads(result.stdout.strip().splitlines()[-1])
    before=run(source,'build')
    assert before['email_count']==3 and before['count']>=3
    assert list(source.rglob('header.bin')), 'Synthetic build must flush a native HNSW index'
    backup_states({},tmp_path/'backup',service_stopped=True,artifacts={'index':source})
    restore_states(tmp_path/'backup',tmp_path/'restored',service_stopped=True)
    assert run(tmp_path/'restored'/'artifacts'/'index','read')==before
