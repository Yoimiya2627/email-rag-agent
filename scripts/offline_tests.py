"""Run regression tests with isolated local state and no model/provider network."""
from __future__ import annotations
import argparse
import inspect
import socket
import shutil
import stat
import subprocess
import os
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parent.parent


def _run_child(state, pytest_args):
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(ROOT))
    import dotenv
    dotenv.load_dotenv=lambda *a,**kw:False
    dotenv.dotenv_values=lambda *a,**kw:{}
    tempfile.tempdir=str(state)
    for key,filename in {'EMAIL_DATA_PATH':'emails.json','CHROMA_PERSIST_DIR':'chroma','APPROVAL_STORE_PATH':'approvals.sqlite3',
                         'AGENT_TRACE_LOG_PATH':'trace.jsonl','MCP_AUDIT_LOG_PATH':'audit.jsonl',
                         'MCP_SERVER_AUDIT_LOG_PATH':'mcp-server.jsonl','SESSION_STORE_PATH':'sessions.sqlite3',
                         'JOB_STORE_PATH':'jobs.sqlite3','TOOL_RESULT_STORE_PATH':'tool_results.sqlite3',
                         'MAIL_ACCOUNTS_PATH':'mail-accounts.sqlite3','IMAP_DATA_ROOT':'imap-mail',
                         'GMAIL_CREDENTIALS_PATH':'no-client.json',
                         'GMAIL_TOKEN_PATH':'no-token.json','GMAIL_READONLY_TOKEN_PATH':'no-read-token.json',
                         'GMAIL_SYNC_OUTPUT_PATH':'emails.json','GMAIL_SYNC_STATE_PATH':'sync.json'}.items():
        os.environ[key]=str(state/filename)
    (state/'emails.json').write_text('[]',encoding='utf-8')
    os.environ.update(DEEPSEEK_API_KEY='offline-placeholder',API_AUTH_TOKEN='',API_OWNER_ID='local',
                      MCP_AUTH_TOKEN='',MCP_OWNER_ID='local',MAIL_PROVIDER='simulated',AGENT_TOOL_BACKEND='local',
                      ENABLE_AGENT_TRACE='false',ENABLE_MCP_AUDIT='false',ENABLE_REAL_EMAIL_SEND='false',
                      PYTHONDONTWRITEBYTECODE='1')
    def guard(event,arguments):
        if event in {'os.remove','os.rmdir','os.mkdir','os.rename'}:
            candidates=arguments[:2] if event=='os.rename' else arguments[:1]
            for candidate in candidates:
                if isinstance(candidate,(str,bytes,os.PathLike)):
                    target=Path(os.fsdecode(candidate)).resolve()
                    if target==ROOT or target.is_relative_to(ROOT):
                        raise RuntimeError('offline test attempted to mutate project files')
        if event=='socket.connect':
            # asyncio implements socketpair with a private loopback pair on Windows.
            if any(frame.function in {'socketpair', '_fallback_socketpair'}
                   and Path(frame.filename).resolve() == Path(socket.__file__).resolve()
                   for frame in inspect.stack(context=0)):
                return
            raise RuntimeError('offline test attempted a network connection')
        if event=='open' and isinstance(arguments[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(arguments[0])).resolve()
            mode,flags=arguments[1],arguments[2]
            write=(isinstance(mode,str) and any(char in mode for char in 'wax+')) or (isinstance(flags,int) and flags&3!=0)
            if write and (path==ROOT or path.is_relative_to(ROOT)):
                raise RuntimeError('offline test attempted to write project state')
            if not write and (path.name=='.env' or 'credentials' in path.parts) and not path.is_relative_to(state):
                raise RuntimeError('offline test attempted to read credentials')
    sys.addaudithook(guard)
    import pytest
    return pytest.main(['-p','no:cacheprovider','--basetemp',str(state/'pytest')]+(pytest_args or ['tests','-q']))


def _cleanup_state(state, work):
    # Only remove the exact directory this parent created; refuse replacement
    # by a link/junction or a path that now resolves outside its known parent.
    if (state.parent != work or state.resolve() != state or not state.name.startswith('email-agent-checks-')
            or state == ROOT or state.is_relative_to(ROOT)):
        raise RuntimeError('unsafe test state cleanup path')
    attributes = getattr(state.lstat(), 'st_file_attributes', 0)
    if state.is_symlink() or attributes & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400):
        raise RuntimeError('test state root was replaced with a link')
    shutil.rmtree(state)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir',default=None)
    parser.add_argument('--_state-dir',default=None,help=argparse.SUPPRESS)
    args,pytest_args=parser.parse_known_args(argv)
    if args._state_dir:
        state=Path(args._state_dir).resolve()
        if state == ROOT or state.is_relative_to(ROOT) or not state.is_dir():
            parser.error('worker state must be an existing directory outside the project')
        return _run_child(state,pytest_args)
    work=Path(args.work_dir or tempfile.gettempdir()).resolve()
    if work == ROOT or work.is_relative_to(ROOT):
        parser.error('test work directory must be outside the project')
    work.mkdir(parents=True,exist_ok=True)
    state=Path(tempfile.mkdtemp(prefix='email-agent-checks-',dir=work)).resolve()
    result=2
    cleanup_failed=False
    try:
        # pytest/Chroma only live in the child. Waiting for process exit releases
        # SDK-owned SQLite handles before the parent attempts Windows cleanup.
        result=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),
                               '--_state-dir',str(state),*pytest_args],check=False).returncode
    finally:
        try:
            _cleanup_state(state,work)
        except Exception as exc:
            cleanup_failed=True
            print(f'offline test cleanup failed: {type(exc).__name__}; retained state: {state}',file=sys.stderr)
    # A cleanup failure is visible even when tests pass, but never overwrites
    # an actual failing pytest exit code with a secondary cleanup exception.
    return result if result else (2 if cleanup_failed else 0)



if __name__=='__main__':
    raise SystemExit(main())
