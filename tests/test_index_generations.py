"""Real local Chroma, deterministic vectors; never downloads models."""
import threading
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest

chromadb = pytest.importorskip("chromadb")
from chromadb.config import Settings
import config.settings as cfg
from core import embedder as db, index_manifest as manifests
from core.filters import FilterSpec
from models.schemas import EmailChunk


def chunk(email="a", index=0, text="synthetic searchable evidence", **metadata):
    return EmailChunk(chunk_id=f"{email}_{index}", email_id=email, chunk_index=index,
                      content=text, metadata={"source_start": 0, "source_end": len(text), "sender": "Alice <alice@example.com>", "date": "2026-01-03",
                                              "labels": '["INBOX","IMPORTANT"]', **metadata})


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(cfg, "EMBEDDING_MODEL_REVISION", "offline-v1", raising=False)
    monkeypatch.setattr(cfg, "EMBEDDING_DIMENSION", 2, raising=False)
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"), settings=Settings(anonymized_telemetry=False))
    monkeypatch.setattr(db, "_get_client", lambda: client)
    monkeypatch.setattr(db, "_collection", None)
    monkeypatch.setattr(db, "resolved_model_revision", lambda: "offline-v1")
    monkeypatch.setattr(db, "embed_texts", lambda texts: [[1., 0.] for _ in texts])
    try:
        yield client
    finally:
        # Chroma retains each persistent System globally until close(); garbage
        # collection alone leaves HNSW files open across the full test suite.
        client.close()
        from chromadb.api.shared_system_client import SharedSystemClient
        assert client._identifier not in SharedSystemClient._identifier_to_system


def active():
    return manifests.read_active_manifest()["generation"]


def test_real_build_queries_and_all_filter_contracts(store):
    db.index_chunks([chunk(), chunk("b", sender="Bob", labels='["INBOX"]')], replace=True)
    state = manifests.read_active_manifest()
    assert state["chunk_count"] == state["email_count"] == 2
    scope = FilterSpec.from_mapping({"sender": "ALICE", "labels": ["INBOX", "IMPORTANT"], "date_hint": "2026-01-01 to 2026-01-04"}, now=datetime(2026, 1, 5, tzinfo=timezone.utc))
    assert [row["email_id"] for row in db.search_similar("evidence", top_k=2, filters=scope)] == ["a"]
    assert [row["metadata"]["email_id"] for row in db.get_filtered_chunks(scope)] == ["a"]


def test_readiness_reads_published_collection_and_rejects_count_mismatch(store):
    db.index_chunks([chunk(),chunk('b')],replace=True)
    assert db.verify_collection_readiness()['chunk_count']==2
    collection=store.get_collection(manifests.read_active_manifest()['collection'])
    collection.delete(ids=['a_0'])
    assert db.get_collection_stats()['chunk_count']==2  # Manifest still looks healthy.
    with pytest.raises(manifests.IndexCompatibilityError,match='count'):
        db.verify_collection_readiness()


def test_readiness_propagates_native_storage_failure(store):
    db.index_chunks([chunk()],replace=True)
    collection=db._get_collection()
    with patch.object(collection,'count',side_effect=RuntimeError('synthetic HNSW failure')):
        with pytest.raises(RuntimeError,match='HNSW'):
            db.verify_collection_readiness()


def test_upsert_preserves_omitted_and_removes_stale_tail(store):
    db.index_chunks([chunk(), chunk(index=1), chunk("keep")], replace=True)
    old = active()
    db.index_chunks([chunk(text="replacement")])
    assert active() != old
    rows = db.get_all_chunks()
    assert {row["chunk_id"] for row in rows} == {"a_0", "keep_0"}
    assert store.get_collection(manifests.read_generation(old)["collection"]).count() == 3


def test_late_embedding_failure_preserves_active_and_resume_is_idempotent(store):
    db.index_chunks([chunk("old")], replace=True)
    old = active()
    with patch.object(db, "embed_texts", side_effect=[[[1., 0.]], RuntimeError("offline failure")]):
        with pytest.raises(RuntimeError):
            db.index_chunks((chunk(str(i)) for i in range(3)), batch_size=1, replace=True)
    assert active() == old
    paused = next(row for row in manifests.list_generations() if row["status"] == "paused")
    assert paused["indexed_chunks"] == 1
    assert db.resume_index_generation(paused["generation"]) == 3
    assert active() == paused["generation"]
    assert db.resume_index_generation(paused["generation"]) == 3
    assert db.get_collection_stats()["chunk_count"] == 3


def test_resume_refuses_changed_input_and_changed_active(store):
    db.index_chunks([chunk("old")], replace=True)
    with patch.object(db, "embed_texts", side_effect=RuntimeError("offline")):
        with pytest.raises(RuntimeError):
            db.index_chunks([chunk("new")], replace=True)
    paused = next(row for row in manifests.list_generations() if row["status"] == "paused")["generation"]
    path = manifests.source_path(paused)
    original = path.read_bytes()
    path.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="changed"):
        db.resume_index_generation(paused)
    path.write_bytes(original)
    db.index_chunks([chunk("different")], replace=True)
    with pytest.raises(manifests.IndexBuildConflict):
        db.resume_index_generation(paused)


def test_atomic_pointer_failure_can_resume_ready_generation(store):
    db.index_chunks([chunk("old")], replace=True)
    old = active()
    real = manifests.atomic_json
    def fail_pointer(path, value):
        if path.name == "active.json":
            raise OSError("injected pointer replacement failure")
        return real(path, value)
    with patch.object(manifests, "atomic_json", side_effect=fail_pointer):
        with pytest.raises(OSError):
            db.index_chunks([chunk("new")], replace=True)
    assert active() == old
    ready = next(row for row in manifests.list_generations() if not row["active"])
    assert ready["status"] == "ready"
    assert db.resume_index_generation(ready["generation"]) == 1


def test_pin_read_generation_clear_and_rollback(store):
    db.index_chunks([chunk("old")], replace=True)
    old = active()
    with db.index_snapshot():
        assert db.get_corpus_revision() == old
        db.index_chunks([chunk("new")], replace=True)
        assert db.get_corpus_revision() == old
        assert db.get_all_chunks()[0]["chunk_id"] == "old_0"
    assert db.get_all_chunks()[0]["chunk_id"] == "new_0"
    db.clear_collection()
    assert db.get_collection_stats()["chunk_count"] == 0
    assert db.resume_index_generation(active()) == 0
    assert db.rollback_generation(old)["chunk_count"] == 1
    assert active() == old


@pytest.mark.parametrize("setting,value", [("EMBEDDING_MODEL", "different-model"), ("EMBEDDING_MODEL_REVISION", "offline-v2"), ("CHUNK_SIZE", 99)])
def test_same_dimension_incompatibility_refused_then_explicit_rebuild(store, monkeypatch, setting, value):
    db.index_chunks([chunk()], replace=True)
    old = active()
    monkeypatch.setattr(cfg, setting, value)
    with pytest.raises(manifests.IndexCompatibilityError):
        db.get_all_chunks()
    with pytest.raises(manifests.IndexCompatibilityError):
        db.index_chunks([chunk("new")])
    assert manifests.read_active_manifest(False)["generation"] == old
    db.index_chunks([chunk("new")], replace=True)
    assert active() != old


def test_legacy_collection_preserved_and_explicit_rebuild_required(store):
    legacy = db._legacy_collection()
    legacy.upsert(ids=["legacy"], documents=["old"], metadatas=[{"email_id": "legacy"}], embeddings=[[1., 0.]])
    assert db.get_collection_stats()["email_count"] is None
    with pytest.raises(manifests.IndexCompatibilityError):
        db.get_all_chunks()
    with pytest.raises(manifests.IndexCompatibilityError):
        db.index_chunks([chunk()])
    db.index_chunks([chunk()], replace=True)
    assert legacy.count() == 1


def test_vector_validation_and_bounded_batches(store):
    for vector in ([float("nan"), 0.], [1.], [True, 0.]):
        with patch.object(db, "embed_texts", return_value=[vector]):
            with pytest.raises(ValueError):
                db.index_chunks([chunk()], replace=True)
        assert manifests.read_active_manifest() is None
    with patch.object(db, "embed_texts", side_effect=lambda texts: [[1., 0.] for _ in texts]) as encode:
        db.index_chunks((chunk(str(i)) for i in range(7)), batch_size=3, replace=True)
    assert [len(call.args[0]) for call in encode.call_args_list] == [3, 3, 1]


def test_metadata_and_stats_never_fetch_corpus_bodies(store, monkeypatch):
    db.index_chunks([chunk(), chunk("b")], replace=True)
    collection = db._get_collection()
    with patch.object(collection, "get", side_effect=AssertionError("stats must not scan")):
        assert db.get_collection_stats()["email_count"] == 2
    with patch.object(collection, "get", wraps=collection.get) as get:
        assert len(db.get_all_metadata()) == 2
    assert all(call.kwargs["include"] == ["metadatas"] for call in get.call_args_list)
    monkeypatch.setattr(cfg, "STATS_METADATA_SCAN_LIMIT", 1, raising=False)
    with pytest.raises(ValueError):
        db.get_all_metadata()


def test_rollback_checks_stored_content_integrity(store):
    db.index_chunks([chunk("old")], replace=True)
    old = active()
    db.index_chunks([chunk("new")], replace=True)
    previous = store.get_collection(manifests.read_generation(old)["collection"])
    previous.update(ids=["old_0"], documents=["tampered"], embeddings=[[1., 0.]])
    with pytest.raises(ValueError, match="verification"):
        db.rollback_generation(old)


def test_unknown_model_revision_fails_before_publication(store):
    db.index_chunks([chunk("old")], replace=True)
    old = active()
    with patch.object(db, "resolved_model_revision", return_value=None):
        with pytest.raises(manifests.IndexCompatibilityError, match="revision is unknown"):
            db.index_chunks([chunk("new")], replace=True)
    assert active() == old
    assert db.get_all_chunks()[0]["chunk_id"] == "old_0"


def test_retained_generation_survives_client_restart_and_rollback(store, monkeypatch):
    db.index_chunks([chunk("old")], replace=True)
    old = active()
    db.index_chunks([chunk("new")], replace=True)
    db.clear_collection()
    store.close()
    reopened = chromadb.PersistentClient(path=cfg.CHROMA_PERSIST_DIR,
                                        settings=Settings(anonymized_telemetry=False))
    monkeypatch.setattr(db, "_get_client", lambda: reopened)
    monkeypatch.setattr(db, "_collection", None)
    try:
        assert db.rollback_generation(old)["chunk_count"] == 1
        assert db.get_all_chunks()[0]["chunk_id"] == "old_0"
        assert db.search_similar("evidence", top_k=1)[0]["email_id"] == "old"
    finally:
        reopened.close()


def test_checkpoint_cancel_preserves_old_and_resumes(store):
    from agents.runtime import RunContext, RunCancelled, use_run_context
    db.index_chunks([chunk("old")], replace=True)
    old = active()
    event = threading.Event()
    checkpoints = []
    def checkpoint(payload):
        checkpoints.append(payload)
        if payload["indexed_chunks"] == 1:
            event.set()
    run = RunContext(cancel_event=event, checkpoint_callback=checkpoint)
    with use_run_context(run), pytest.raises(RunCancelled):
        db.index_chunks([chunk("new"), chunk("second")], batch_size=1, replace=True)
    assert active() == old
    assert checkpoints[-1]["safe"] is True and checkpoints[-1]["kind"] == "index"
    assert db.resume_index_generation(checkpoints[-1]["generation_id"]) == 2


def test_rollback_after_hnsw_reader_cache_eviction(tmp_path, monkeypatch):
    # Exercise real persistence with a tiny reader cache, instead of relying on
    # old generations remaining in memory throughout one test process.
    from chromadb.api.rust import RustBindingsAPI
    real_start = RustBindingsAPI.start
    def limited_start(api):
        api.hnsw_cache_size = 2
        return real_start(api)
    monkeypatch.setattr(cfg, "CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(cfg, "EMBEDDING_MODEL_REVISION", "offline-v1", raising=False)
    monkeypatch.setattr(cfg, "EMBEDDING_DIMENSION", 2, raising=False)
    with patch.object(RustBindingsAPI, "start", limited_start):
        client = chromadb.PersistentClient(path=cfg.CHROMA_PERSIST_DIR,
                                          settings=Settings(anonymized_telemetry=False))
    monkeypatch.setattr(db, "_get_client", lambda: client)
    monkeypatch.setattr(db, "_collection", None)
    monkeypatch.setattr(db, "resolved_model_revision", lambda: "offline-v1")
    monkeypatch.setattr(db, "embed_texts", lambda texts: [[1., 0.] for _ in texts])
    try:
        db.index_chunks([chunk("old")], replace=True)
        old = active()
        for index in range(5):
            db.index_chunks([chunk(f"new{index}")], replace=True)
            assert db.search_similar("evidence", top_k=1)[0]["email_id"] == f"new{index}"
        assert db.rollback_generation(old)["chunk_count"] == 1
        assert db.search_similar("evidence", top_k=1)[0]["email_id"] == "old"
    finally:
        client.close()
