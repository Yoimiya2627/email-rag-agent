"""Synthetic parser process trees only; temporary paths come from pytest."""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from core.parse_process import parser_process, ParserProcessError


def alive(pid):
    if os.name == 'nt':
        api = ctypes.WinDLL('kernel32', use_last_error=True)
        api.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        api.OpenProcess.restype = ctypes.c_void_p
        api.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        api.WaitForSingleObject.restype = ctypes.c_ulong
        api.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = api.OpenProcess(0x100000, False, pid)
        if not handle:
            return False
        try:
            return api.WaitForSingleObject(handle, 0) == 258
        finally:
            api.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        stat = Path(f'/proc/{pid}/stat')
        return not stat.exists() or stat.read_text().split()[2] != 'Z'
    except ProcessLookupError:
        return False


def wait_for(path, process):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if path.is_file():
            try:
                return json.loads(path.read_text())
            except (ValueError, OSError):
                pass
        time.sleep(.02)
    pytest.fail(f'synthetic process did not signal readiness, status={process.poll()}')


@pytest.mark.parametrize('parent_exits', [False, True])
def test_entire_tree_stops_including_venv_launcher(tmp_path, parent_exits):
    # The child publishes its own real PID, rather than its venv launcher PID.
    child_marker, parent_marker = tmp_path/'child.json', tmp_path/'parent.json'
    child_code = f'import os,time,json; from pathlib import Path; Path({str(child_marker)!r}).write_text(json.dumps(os.getpid())); time.sleep(60)'
    parent_code = ('import os,time,json,subprocess,sys; from pathlib import Path; '
                   f'p=subprocess.Popen([sys.executable,"-c",{child_code!r}]); '
                   f'Path({str(parent_marker)!r}).write_text(json.dumps([os.getpid(),p.pid])); '
                   + ('time.sleep(.1)' if parent_exits else 'time.sleep(60)'))
    with parser_process([sys.executable, '-c', parent_code], cwd=tmp_path) as process:
        parent_pid, child_launcher = wait_for(parent_marker, process)
        child_pid = wait_for(child_marker, process)
        assert alive(child_pid)
        if parent_exits:
            assert process.wait(timeout=10) == 0
            assert alive(child_pid), 'test must exercise a descendant surviving its parent'
    assert process.poll() is not None
    assert not any(alive(pid) for pid in (parent_pid, child_launcher, child_pid))


def test_exception_cancellation_cleans_process(tmp_path):
    marker = tmp_path/'ready.json'
    command = [sys.executable, '-c', f'import os,time,json; from pathlib import Path; Path({str(marker)!r}).write_text(json.dumps(os.getpid())); time.sleep(60)']
    with pytest.raises(TimeoutError, match='synthetic deadline'):
        with parser_process(command, cwd=tmp_path) as process:
            real_pid = wait_for(marker, process)
            raise TimeoutError('synthetic deadline')
    assert process.poll() is not None
    assert not alive(real_pid)


def test_normal_completion_can_read_result_inside_context(tmp_path):
    with parser_process([sys.executable, '-c', 'print("synthetic-result")'], cwd=tmp_path, stdout=subprocess.PIPE) as process:
        output, _ = process.communicate(timeout=10)
        assert process.returncode == 0
        assert output.strip() == b'synthetic-result'


def test_missing_executable_safe_failure(tmp_path):
    with pytest.raises(ParserProcessError, match='parser_process_start_failed'):
        with parser_process([str(tmp_path/'absent-private-command.exe')], cwd=tmp_path):
            pytest.fail('must not yield')


@pytest.mark.parametrize('command', [[], 'python', [None], ['python', 'bad\x00arg']])
def test_bad_command_fails_before_start(tmp_path, command):
    with pytest.raises(ParserProcessError, match='invalid_parser_command'):
        with parser_process(command, cwd=tmp_path):
            pytest.fail('must not yield')


@pytest.mark.skipif(os.name != 'nt', reason='Windows Job Object contract')
@pytest.mark.parametrize('stage', ['assign', 'resume'])
def test_initialization_failure_kills_suspended_process_before_user_code(tmp_path, monkeypatch, stage):
    import core.parse_process as module
    created = []
    original = module.subprocess.Popen

    def popen(*args, **kwargs):
        process = original(*args, **kwargs)
        created.append(process)
        return process

    def fail(*args):
        raise OSError('private startup diagnostics')

    monkeypatch.setattr(module.subprocess, 'Popen', popen)
    monkeypatch.setattr(module._WindowsJob, stage, fail)
    marker = tmp_path/'must-not-run'
    with pytest.raises(ParserProcessError, match='parser_process_start_failed'):
        with parser_process([sys.executable, '-c', f'from pathlib import Path; Path({str(marker)!r}).write_text("ran")'], cwd=tmp_path):
            pytest.fail('must not yield')
    assert len(created) == 1
    assert created[0].poll() is not None
    assert not marker.exists()


@pytest.mark.skipif(os.name != 'nt', reason='Windows Job Object memory limit')
def test_memory_limit_rejects_large_allocation(tmp_path):
    marker = tmp_path/'memory.json'
    code = f'''import json
from pathlib import Path
try:
    data = bytearray(900 * 1024 * 1024)
except MemoryError:
    Path({str(marker)!r}).write_text(json.dumps('limited'))
else:
    Path({str(marker)!r}).write_text(json.dumps('unlimited'))
'''
    with parser_process([sys.executable, '-c', code], cwd=tmp_path) as process:
        process.wait(timeout=15)
        assert process.returncode == 0
        assert json.loads(marker.read_text()) == 'limited'
