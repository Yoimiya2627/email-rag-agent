"""Exclusive evaluation output ownership for cooperating local writers.

The sidecar is held for the whole run. Existing evidence is never reused.
Identity/content checks detect outside changes but are not an OS lock against
arbitrary non-cooperating writers between the check and atomic replacement.
Crash leftovers are deliberately not reclaimed automatically.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid


class OutputOwnershipError(RuntimeError):
    pass


class EvaluationOutput:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.lock_path = self.path.with_name(self.path.name + '.writer-lock')
        self._token = uuid.uuid4().hex.encode('ascii')
        self._lock_identity = None
        self._identity = None
        self._digest = None
        self._active = False
        self.updates = 0

    @staticmethod
    def _identity_of(stat):
        return stat.st_dev, stat.st_ino

    @staticmethod
    def _encode(payload):
        return (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # O_EXCL rejects existing files and symlinks, including stale locks.
        try:
            with self.lock_path.open('xb') as lock:
                self._lock_identity = self._identity_of(os.fstat(lock.fileno()))
                lock.write(self._token)
                lock.flush()
                os.fsync(lock.fileno())
            data = self._encode({'complete': False, 'batch_status': 'initializing'})
            with self.path.open('xb') as output:
                self._identity = self._identity_of(os.fstat(output.fileno()))
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            self._digest = hashlib.sha256(data).digest()
            self._active = True
            return self
        except BaseException:
            self.close()
            raise

    def _check_owned(self):
        if not self._active:
            raise OutputOwnershipError('evaluation output is not owned by this writer')
        try:
            with self.lock_path.open('rb') as lock:
                valid_lock = (self._identity_of(os.fstat(lock.fileno())) == self._lock_identity
                              and lock.read() == self._token)
            with self.path.open('rb') as output:
                valid_output = (self._identity_of(os.fstat(output.fileno())) == self._identity
                                and hashlib.sha256(output.read()).digest() == self._digest)
            if not valid_lock or not valid_output:
                raise OutputOwnershipError('evaluation output ownership was lost')
        except OSError as exc:
            self._active = False
            raise OutputOwnershipError('evaluation output ownership was lost') from exc
        except OutputOwnershipError:
            self._active = False
            raise

    def write(self, payload):
        data = self._encode(payload)
        self._check_owned()
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='wb', dir=self.path.parent,
                    prefix=self.path.name + '.', suffix='.tmp', delete=False) as output:
                temporary = Path(output.name)
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
                identity = self._identity_of(os.fstat(output.fileno()))
            self._check_owned()
            temporary.replace(self.path)
            self._identity = identity
            self._digest = hashlib.sha256(data).digest()
            self.updates += 1
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def close(self):
        self._active = False
        if self._lock_identity is None:
            return
        try:
            with self.lock_path.open('rb') as lock:
                owned = (self._identity_of(os.fstat(lock.fileno())) == self._lock_identity
                         and lock.read() == self._token)
            if owned:
                self.lock_path.unlink()
        except FileNotFoundError:
            pass
        finally:
            self._lock_identity = None

    def __exit__(self, *args):
        self.close()
