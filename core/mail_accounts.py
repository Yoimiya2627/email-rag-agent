"""Local 163 account registry; authorization codes are bound to Windows users.

Only ``credentials`` exposes a decrypted code, for internal IMAP callers. The
injected protector is intended for isolated tests; production uses user DPAPI.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Protocol
import uuid

IMAP_HOST = "imap.163.com"
IMAP_PORT = 993
_ADDRESS = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}@163\.com", re.ASCII)
_ACCOUNT_ID = re.compile(r"[0-9a-f]{32}", re.ASCII)


class Protector(Protocol):
    def protect(self, data: bytes, entropy: bytes) -> bytes: ...
    def unprotect(self, data: bytes, entropy: bytes) -> bytes: ...


class WindowsDPAPIProtector:
    """Use current-user DPAPI, without machine scope or interactive prompts."""

    def __init__(self):
        if os.name != "nt":
            raise RuntimeError("unsupported_secure_storage")
        from ctypes import wintypes

        class Blob(ctypes.Structure):
            _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_ubyte))]

        self._blob = Blob
        self._crypt = ctypes.WinDLL("crypt32", use_last_error=True)
        self._kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        blob_ptr = ctypes.POINTER(Blob)
        self._crypt.CryptProtectData.argtypes = [blob_ptr, wintypes.LPCWSTR, blob_ptr,
                                                 ctypes.c_void_p, ctypes.c_void_p,
                                                 wintypes.DWORD, blob_ptr]
        self._crypt.CryptUnprotectData.argtypes = [blob_ptr, ctypes.c_void_p, blob_ptr,
                                                   ctypes.c_void_p, ctypes.c_void_p,
                                                   wintypes.DWORD, blob_ptr]
        self._crypt.CryptProtectData.restype = wintypes.BOOL
        self._crypt.CryptUnprotectData.restype = wintypes.BOOL
        self._kernel.LocalFree.argtypes = [ctypes.c_void_p]
        self._kernel.LocalFree.restype = ctypes.c_void_p

    def _transform(self, data: bytes, entropy: bytes, *, decrypt: bool) -> bytes:
        data_buffer = ctypes.create_string_buffer(data)
        entropy_buffer = ctypes.create_string_buffer(entropy)
        data_blob = self._blob(len(data), ctypes.cast(data_buffer, ctypes.POINTER(ctypes.c_ubyte)))
        entropy_blob = self._blob(len(entropy), ctypes.cast(entropy_buffer, ctypes.POINTER(ctypes.c_ubyte)))
        output = self._blob()
        function = self._crypt.CryptUnprotectData if decrypt else self._crypt.CryptProtectData
        try:
            if not function(ctypes.byref(data_blob), None, ctypes.byref(entropy_blob),
                            None, None, 0x01, ctypes.byref(output)):
                raise RuntimeError("secure_storage_failed")
            return ctypes.string_at(output.data, output.size)
        finally:
            if output.data:
                self._kernel.LocalFree(ctypes.cast(output.data, ctypes.c_void_p))

    def protect(self, data: bytes, entropy: bytes) -> bytes:
        return self._transform(data, entropy, decrypt=False)

    def unprotect(self, data: bytes, entropy: bytes) -> bytes:
        return self._transform(data, entropy, decrypt=True)


def _owner(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError("invalid_owner")
    return value


def _account_id(value: str) -> str:
    if not isinstance(value, str) or _ACCOUNT_ID.fullmatch(value) is None:
        raise ValueError("invalid_account_id")
    return value


def _entropy(owner: str, account_id: str) -> bytes:
    return hashlib.sha256(json.dumps(["mail-account-v1", owner, account_id],
                                     ensure_ascii=True).encode("ascii")).digest()


def get_data_dir(base_root: str | Path, owner: str, account_id: str) -> Path:
    """Return an isolated directory without putting email addresses in paths."""
    owner_hash = hashlib.sha256(_owner(owner).encode("utf-8")).hexdigest()
    return Path(base_root) / owner_hash / _account_id(account_id)


class MailAccountStore:
    def __init__(self, path: str | Path, *, protector: Protector | None = None):
        self._protector = protector if protector is not None else WindowsDPAPIProtector()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS mail_accounts (
                id TEXT PRIMARY KEY,
                owner TEXT NOT NULL,
                address TEXT NOT NULL,
                display_name TEXT NOT NULL,
                encrypted_code BLOB NOT NULL,
                credential_version INTEGER NOT NULL,
                created TEXT NOT NULL,
                updated TEXT NOT NULL,
                UNIQUE(owner, address)
            )""")

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(str(self.path), timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _public(row) -> dict:
        return {"id": row["id"], "address": row["address"],
                "display_name": row["display_name"], "provider": "163",
                "credential_version": row["credential_version"],
                "created": row["created"], "updated": row["updated"]}

    def upsert(self, owner: str, address: str, authorization_code: str,
               display_name: str = "") -> dict:
        owner = _owner(owner)
        if not isinstance(address, str) or _ADDRESS.fullmatch(address.strip().lower()) is None:
            raise ValueError("invalid_163_address")
        address = address.strip().lower()
        if (not isinstance(authorization_code, str) or not authorization_code
                or len(authorization_code) > 512
                or any(character.isspace() or ord(character) < 32 or ord(character) == 127
                       for character in authorization_code)):
            raise ValueError("invalid_authorization_code")
        if (not isinstance(display_name, str) or len(display_name) > 128
                or any(ord(character) < 32 or ord(character) == 127 for character in display_name)):
            raise ValueError("invalid_display_name")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            previous = connection.execute("SELECT * FROM mail_accounts WHERE owner=? AND address=?",
                                          (owner, address)).fetchone()
            account_id = previous["id"] if previous else uuid.uuid4().hex
            now = datetime.now(timezone.utc).isoformat()
            created = previous["created"] if previous else now
            version = previous["credential_version"] + 1 if previous else 1
            try:
                encrypted = self._protector.protect(authorization_code.encode("utf-8"),
                                                    _entropy(owner, account_id))
                if not isinstance(encrypted, bytes) or not encrypted:
                    raise ValueError("invalid_ciphertext")
            except Exception:
                raise RuntimeError("secure_storage_failed") from None
            connection.execute("""INSERT INTO mail_accounts
                (id,owner,address,display_name,encrypted_code,credential_version,created,updated)
                VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(owner,address) DO UPDATE SET
                display_name=excluded.display_name, encrypted_code=excluded.encrypted_code,
                credential_version=excluded.credential_version, updated=excluded.updated""",
                (account_id, owner, address, display_name.strip(), sqlite3.Binary(encrypted),
                 version, created, now))
            row = connection.execute("SELECT * FROM mail_accounts WHERE id=?", (account_id,)).fetchone()
            return self._public(row)

    def list(self, owner: str) -> list[dict]:
        with self._connect() as connection:
            return [self._public(row) for row in connection.execute(
                "SELECT * FROM mail_accounts WHERE owner=? ORDER BY created,id", (_owner(owner),))]

    def get(self, owner: str, account_id: str) -> dict:
        with self._connect() as connection:
            return self._public(self._row(connection, owner, account_id))

    @staticmethod
    def _row(connection, owner: str, account_id: str):
        row = connection.execute("SELECT * FROM mail_accounts WHERE owner=? AND id=?",
                                 (_owner(owner), _account_id(account_id))).fetchone()
        if row is None:
            raise KeyError("account_not_found")
        return row

    def credentials(self, owner: str, account_id: str) -> tuple[dict, str]:
        with self._connect() as connection:
            row = self._row(connection, owner, account_id)
            public = self._public(row)
            try:
                code = self._protector.unprotect(bytes(row["encrypted_code"]),
                                                _entropy(owner, account_id)).decode("utf-8")
            except Exception:
                raise RuntimeError("secure_storage_failed") from None
            return public, code
