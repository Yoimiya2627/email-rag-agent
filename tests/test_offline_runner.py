"""Process exit must release SDK handles before isolated state is removed."""
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import offline_tests as runner


@pytest.mark.parametrize('test_exit,expected', [(0,2),(1,1),(5,5)])
def test_cleanup_failure_is_visible_without_masking_test_exit(tmp_path, monkeypatch, capsys, test_exit, expected):
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a,**kw: SimpleNamespace(returncode=test_exit))
    def failed_cleanup(*args):
        raise PermissionError('synthetic open handle')
    monkeypatch.setattr(runner, '_cleanup_state', failed_cleanup)
    assert runner.main(['--work-dir',str(tmp_path),'-q']) == expected
    assert 'cleanup failed: PermissionError; retained state:' in capsys.readouterr().err


def test_cleanup_rejects_path_outside_recorded_root(tmp_path):
    work = tmp_path / 'work'
    work.mkdir()
    outside = tmp_path / 'email-agent-checks-outside'
    outside.mkdir()
    with pytest.raises(RuntimeError,match='unsafe'):
        runner._cleanup_state(outside,work)
    assert outside.is_dir()


@pytest.mark.parametrize('fail', [False, True])
def test_child_process_releases_real_chroma_and_preserves_exit(tmp_path, fail):
    pytest.importorskip('chromadb')
    case = tmp_path / 'test_sdk_child.py'
    case.write_text('''import os
import socket
from pathlib import Path
import chromadb
from chromadb.config import Settings
import pytest

# Deliberately retain this real client until process exit, just as the suite
# caches its default client. Chroma uses a synthetic supplied vector only.
client = chromadb.PersistentClient(path=os.environ['CHROMA_PERSIST_DIR'],
    settings=Settings(anonymized_telemetry=False))
def test_sdk_and_guards():
    # Python 3.11 on Windows names the stdlib implementation
    # _fallback_socketpair; asyncio needs it even in an offline suite.
    left, right = socket.socketpair()
    with left, right:
        left.send(b'offline')
        assert right.recv(7) == b'offline'
    collection = client.get_or_create_collection('synthetic',embedding_function=None)
    collection.upsert(ids=['one'],embeddings=[[1.,0.,0.]],documents=['synthetic'])
    assert collection.count() == 1
    with pytest.raises(RuntimeError,match='network connection'):
        socket.socket().connect(('127.0.0.1',1))
    with pytest.raises(RuntimeError,match='credentials'):
        Path(__file__).with_name('.env').read_text()
    from scripts.offline_tests import ROOT
    with pytest.raises(RuntimeError,match='project state'):
        (ROOT/'offline-runner-must-not-write.tmp').write_text('forbidden')
''' + ('    assert False, "synthetic test failure"\n' if fail else ''), encoding='utf-8')
    work = tmp_path / 'runner-state'
    result = subprocess.run([sys.executable,'-B',str(runner.ROOT/'scripts/offline_tests.py'),
                             '--work-dir',str(work),str(case),'-q','--tb=short'],
                            capture_output=True,text=True,timeout=90)
    assert result.returncode == (1 if fail else 0), result.stdout + result.stderr
    assert not list(work.iterdir()), result.stderr
    assert 'cleanup failed' not in result.stderr
