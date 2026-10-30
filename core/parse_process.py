"""Own a parser process tree for exactly one context, including its children.

Windows starts suspended, assigns a kill-on-close Job Object, then resumes the
primary thread. Both each process and the whole job have a 768 MiB memory cap.
POSIX uses a new session/process group; it does not impose a memory cap.
"""
from contextlib import contextmanager
import ctypes
from ctypes import wintypes
import os
import signal
import subprocess
import time


MEMORY_LIMIT_BYTES = 768 * 1024 * 1024


class ParserProcessError(RuntimeError):
    """Stable messages only: command arguments may identify private mail."""


class _BasicLimits(ctypes.Structure):
    _fields_ = [('process_time', ctypes.c_longlong), ('job_time', ctypes.c_longlong),
                ('flags', wintypes.DWORD), ('min_working_set', ctypes.c_size_t),
                ('max_working_set', ctypes.c_size_t), ('active_limit', wintypes.DWORD),
                ('affinity', ctypes.c_size_t), ('priority', wintypes.DWORD),
                ('scheduling', wintypes.DWORD)]


class _IOCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in
                ('read_ops', 'write_ops', 'other_ops', 'read_bytes', 'write_bytes', 'other_bytes')]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [('basic', _BasicLimits), ('io', _IOCounters),
                ('process_memory', ctypes.c_size_t), ('job_memory', ctypes.c_size_t),
                ('peak_process_memory', ctypes.c_size_t), ('peak_job_memory', ctypes.c_size_t)]


class _Accounting(ctypes.Structure):
    _fields_ = [('user_time', ctypes.c_longlong), ('kernel_time', ctypes.c_longlong),
                ('period_user_time', ctypes.c_longlong), ('period_kernel_time', ctypes.c_longlong),
                ('page_faults', wintypes.DWORD), ('total_processes', wintypes.DWORD),
                ('active_processes', wintypes.DWORD), ('terminated_processes', wintypes.DWORD)]


class _ThreadEntry(ctypes.Structure):
    _fields_ = [('size', wintypes.DWORD), ('usage', wintypes.DWORD),
                ('thread_id', wintypes.DWORD), ('owner_pid', wintypes.DWORD),
                ('base_priority', wintypes.LONG), ('delta_priority', wintypes.LONG),
                ('flags', wintypes.DWORD)]


def _kernel():
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    signatures = {
        'CreateJobObjectW': ([ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
        'SetInformationJobObject': ([wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD], wintypes.BOOL),
        'QueryInformationJobObject': ([wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p], wintypes.BOOL),
        'AssignProcessToJobObject': ([wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
        'TerminateJobObject': ([wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
        'CreateToolhelp32Snapshot': ([wintypes.DWORD, wintypes.DWORD], wintypes.HANDLE),
        'Thread32First': ([wintypes.HANDLE, ctypes.POINTER(_ThreadEntry)], wintypes.BOOL),
        'Thread32Next': ([wintypes.HANDLE, ctypes.POINTER(_ThreadEntry)], wintypes.BOOL),
        'OpenThread': ([wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
        'ResumeThread': ([wintypes.HANDLE], wintypes.DWORD),
        'OpenProcess': ([wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
        'WaitForSingleObject': ([wintypes.HANDLE, wintypes.DWORD], wintypes.DWORD),
        'CloseHandle': ([wintypes.HANDLE], wintypes.BOOL),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(api, name)
        function.argtypes, function.restype = arguments, result
    return api


class _WindowsJob:
    def __init__(self):
        self.api = _kernel()
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise ParserProcessError('parser_job_create_failed')
        limits = _ExtendedLimits()
        # KILL_ON_JOB_CLOSE | PROCESS_MEMORY | JOB_MEMORY. No breakaway flags.
        limits.basic.flags = 0x2000 | 0x100 | 0x200
        limits.process_memory = limits.job_memory = MEMORY_LIMIT_BYTES
        if not self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            self.close()
            raise ParserProcessError('parser_job_limits_failed')

    def assign(self, process):
        if not self.api.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise ParserProcessError('parser_job_assignment_failed')

    def resume(self, process):
        # Popen closes the primary thread handle. Locate that one thread while
        # CREATE_SUSPENDED guarantees no user code or children have run yet.
        snapshot = self.api.CreateToolhelp32Snapshot(4, 0)  # TH32CS_SNAPTHREAD
        if snapshot == ctypes.c_void_p(-1).value:
            raise ParserProcessError('parser_thread_snapshot_failed')
        try:
            entry = _ThreadEntry()
            entry.size = ctypes.sizeof(entry)
            found = []
            more = self.api.Thread32First(snapshot, ctypes.byref(entry))
            while more:
                if entry.owner_pid == process.pid:
                    found.append(entry.thread_id)
                entry.size = ctypes.sizeof(entry)
                more = self.api.Thread32Next(snapshot, ctypes.byref(entry))
            if len(found) != 1:
                raise ParserProcessError('parser_primary_thread_missing')
            thread = self.api.OpenThread(2, False, found[0])  # THREAD_SUSPEND_RESUME
            if not thread:
                raise ParserProcessError('parser_thread_open_failed')
            try:
                if self.api.ResumeThread(thread) != 1:
                    raise ParserProcessError('parser_thread_resume_failed')
            finally:
                self.api.CloseHandle(thread)
        finally:
            self.api.CloseHandle(snapshot)

    def stop(self):
        if not self.handle:
            return
        handles = []
        try:
            # Keep real process handles, rather than polling recyclable PIDs.
            # Job accounting reaches zero slightly before handles signal exit.
            capacity = 16
            while True:
                buffer = ctypes.create_string_buffer(8 + capacity * ctypes.sizeof(ctypes.c_size_t))
                if self.api.QueryInformationJobObject(self.handle, 3, buffer, len(buffer), None):
                    count = wintypes.DWORD.from_buffer(buffer, 4).value
                    identifiers = (ctypes.c_size_t * count).from_buffer(buffer, 8)
                    for pid in identifiers:
                        handle = self.api.OpenProcess(0x100000, False, pid)
                        if handle:
                            handles.append(handle)
                    break
                if ctypes.get_last_error() != 234 or capacity >= 65536:  # ERROR_MORE_DATA
                    raise ParserProcessError('parser_job_status_failed')
                capacity *= 2
            if not self.api.TerminateJobObject(self.handle, 1):
                raise ParserProcessError('parser_job_termination_failed')
            deadline = time.monotonic() + 5
            while True:
                state = _Accounting()
                if not self.api.QueryInformationJobObject(self.handle, 1, ctypes.byref(state), ctypes.sizeof(state), None):
                    raise ParserProcessError('parser_job_status_failed')
                if state.active_processes == 0:
                    break
                if time.monotonic() >= deadline:
                    raise ParserProcessError('parser_job_exit_timeout')
                time.sleep(.01)
            for handle in handles:
                milliseconds = max(0, int((deadline - time.monotonic()) * 1000))
                if self.api.WaitForSingleObject(handle, milliseconds) != 0:
                    raise ParserProcessError('parser_job_exit_timeout')
        finally:
            for handle in handles:
                self.api.CloseHandle(handle)

    def close(self):
        if self.handle:
            handle, self.handle = self.handle, None
            self.api.CloseHandle(handle)


@contextmanager
def parser_process(command, *, cwd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL):
    """Yield Popen; completion, cancellation and initialization errors clean up.

    The caller controls polling and its deadline. A parent that exits normally
    does not authorize surviving descendants: they are stopped on context exit.
    """
    if (not isinstance(command, list) or not command
            or any(not isinstance(arg, str) or '\x00' in arg for arg in command)):
        raise ParserProcessError('invalid_parser_command')
    job, process = None, None
    try:
        if os.name == 'nt':
            job = _WindowsJob()
            try:
                process = subprocess.Popen(command, cwd=cwd, stdin=subprocess.DEVNULL,
                    stdout=stdout, stderr=stderr, close_fds=True,
                    creationflags=subprocess.CREATE_NO_WINDOW | 4)  # CREATE_SUSPENDED
                job.assign(process)
                job.resume(process)
            except Exception:
                raise ParserProcessError('parser_process_start_failed') from None
        else:
            try:
                process = subprocess.Popen(command, cwd=cwd, stdin=subprocess.DEVNULL,
                    stdout=stdout, stderr=stderr, close_fds=True, start_new_session=True)
            except Exception:
                raise ParserProcessError('parser_process_start_failed') from None
        yield process
    finally:
        try:
            if job is not None:
                try:
                    job.stop()
                finally:
                    job.close()  # kill-on-close is also a fallback if stop fails.
            elif process is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        finally:
            if process is not None:
                # Handles assignment failure, where the suspended process was
                # never a member of the job. Never locate a process by name.
                if process.poll() is None:
                    process.kill()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    raise ParserProcessError('parser_process_exit_timeout') from None
