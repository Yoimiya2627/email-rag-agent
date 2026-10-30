"""Offline account isolation and secure storage regression coverage."""
import hashlib
import json
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from core.mail_accounts import MailAccountStore, WindowsDPAPIProtector, get_data_dir


class TestProtector:
    __test__ = False

    def protect(self, data, entropy):
        # A reversible test double, not a production encryption implementation.
        payload = bytes(value ^ 0xA5 for value in data)
        return hashlib.sha256(entropy + payload).digest() + payload

    def unprotect(self, data, entropy):
        payload = data[32:]
        if data[:32] != hashlib.sha256(entropy + payload).digest():
            raise ValueError("invalid_entropy")
        return bytes(value ^ 0xA5 for value in payload)


@pytest.fixture
def store(tmp_path):
    return MailAccountStore(tmp_path / "accounts.sqlite3", protector=TestProtector())


def test_normalized_upsert_keeps_id_and_advances_credentials(store):
    original = store.upsert("alice", " My.Mail@163.COM ", "first-code", "邮箱")
    rebound = store.upsert("alice", "my.mail@163.com", "second-code", "新邮箱")
    assert original["address"] == "my.mail@163.com"
    assert rebound["id"] == original["id"]
    assert rebound["credential_version"] == 2
    assert rebound["created"] == original["created"]
    assert rebound["updated"] >= original["updated"]
    assert rebound["display_name"] == "新邮箱"
    assert store.credentials("alice", rebound["id"]) == (rebound, "second-code")
    assert store.list("alice") == [rebound]
    reopened = MailAccountStore(store.path, protector=TestProtector())
    assert reopened.get("alice", rebound["id"]) == rebound


def test_owner_isolation_and_secret_not_in_public_or_database(store):
    secret = "never-store-this-authorization-code"
    first = store.upsert("alice", "mail@163.com", secret)
    second = store.upsert("bob", "mail@163.com", "different-code")
    assert first["id"] != second["id"]
    assert "mail" not in first["id"]
    assert set(first) == {"id", "address", "display_name", "provider", "credential_version", "created", "updated"}
    assert store.list("bob") == [second]
    with pytest.raises(KeyError):
        store.get("bob", first["id"])
    with pytest.raises(KeyError):
        store.credentials("bob", first["id"])
    assert secret not in json.dumps(first)
    assert secret not in repr(store)
    assert secret.encode() not in store.path.read_bytes()
    with sqlite3.connect(store.path) as connection:
        kind = connection.execute("SELECT typeof(encrypted_code) FROM mail_accounts").fetchone()[0]
    assert kind == "blob"


@pytest.mark.parametrize("address", ["mail", "mail@126.com", "mail@163.com.evil", "mail@evil@163.com",
                                    "../mail@163.com", "邮箱@163.com", "mail\nname@163.com", "@163.com"])
def test_rejects_invalid_provider_addresses(store, address):
    with pytest.raises(ValueError, match="^invalid_163_address$"):
        store.upsert("alice", address, "secret")
    assert store.list("alice") == []


def test_protector_failure_rolls_back_and_sanitizes_exception(store):
    first = store.upsert("alice", "mail@163.com", "old-code")
    class FailingProtector(TestProtector):
        def protect(self, data, entropy):
            raise ValueError(data.decode())
    store._protector = FailingProtector()
    with pytest.raises(RuntimeError, match="^secure_storage_failed$") as failure:
        store.upsert("alice", "mail@163.com", "sensitive-code", "should roll back")
    assert "sensitive-code" not in str(failure.value)
    assert failure.value.__suppress_context__
    assert store.get("alice", first["id"]) == first
    assert store.credentials("alice", first["id"])[1] == "old-code"


def test_database_failure_rolls_back_entire_rebind(store):
    first = store.upsert("alice", "mail@163.com", "old-code")
    with sqlite3.connect(store.path) as connection:
        connection.execute("CREATE TRIGGER fail_update BEFORE UPDATE ON mail_accounts BEGIN SELECT RAISE(ABORT, 'test_failure'); END")
    with pytest.raises(sqlite3.IntegrityError, match="test_failure"):
        store.upsert("alice", "mail@163.com", "new-code", "modified")
    assert store.get("alice", first["id"]) == first
    assert store.credentials("alice", first["id"])[1] == "old-code"


def test_concurrent_rebinds_have_distinct_versions(store):
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda n: store.upsert("alice", "mail@163.com", f"code-{n}"), range(8)))
    assert len({result["id"] for result in results}) == 1
    assert sorted(result["credential_version"] for result in results) == list(range(1, 9))
    assert store.list("alice")[0]["credential_version"] == 8


def test_ciphertext_cannot_be_moved_between_owners_or_accounts(store):
    first = store.upsert("alice", "mail@163.com", "first-code")
    second = store.upsert("bob", "mail@163.com", "second-code")
    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE mail_accounts SET encrypted_code=(SELECT encrypted_code FROM mail_accounts WHERE id=?) WHERE id=?",
                           (first["id"], second["id"]))
    with pytest.raises(RuntimeError, match="^secure_storage_failed$"):
        store.credentials("bob", second["id"])


def test_data_paths_hide_address_and_reject_traversal(tmp_path):
    account_id = "a" * 32
    path = get_data_dir(tmp_path, "../../alice@163.com", account_id)
    assert path.parent.parent == tmp_path
    assert "@" not in str(path.relative_to(tmp_path))
    assert path != get_data_dir(tmp_path, "bob", account_id)
    for invalid in ("../escape", "A" * 32, "", "a/b"):
        with pytest.raises(ValueError, match="invalid_account_id"):
            get_data_dir(tmp_path, "alice", invalid)


def test_non_windows_default_fails_without_creating_database(monkeypatch, tmp_path):
    monkeypatch.setattr("core.mail_accounts.os.name", "posix")
    with pytest.raises(RuntimeError, match="^unsupported_secure_storage$"):
        MailAccountStore(tmp_path / "uncreated.sqlite3")
    assert not (tmp_path / "uncreated.sqlite3").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI requires Windows")
def test_real_windows_dpapi_roundtrip_and_entropy_binding(tmp_path):
    protector = WindowsDPAPIProtector()
    synthetic = b"offline-synthetic-authorization-code"
    encrypted = protector.protect(synthetic, b"owner-and-account")
    assert synthetic not in encrypted
    assert protector.unprotect(encrypted, b"owner-and-account") == synthetic
    with pytest.raises(RuntimeError, match="^secure_storage_failed$"):
        protector.unprotect(encrypted, b"different-account")
    real_store = MailAccountStore(tmp_path / "dpapi.sqlite3")
    public = real_store.upsert("offline-user", "synthetic@163.com", synthetic.decode())
    assert real_store.credentials("offline-user", public["id"]) == (public, synthetic.decode())
    assert synthetic not in real_store.path.read_bytes()
