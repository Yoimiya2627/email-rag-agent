"""Behavioral delta contracts with deterministic vectors and real storage option."""
import hashlib
from unittest.mock import Mock

import pytest
import config.settings as cfg
from core import embedder as db, index_manifest as manifests
from core.index_metrics import collect_index_metrics
from models.schemas import EmailChunk
from tests.index_store_helpers import MemoryClient


def item(email="a", index=0, text=None, **metadata):
    text = text if text is not None else f"Evidence {email} {index}"
    return EmailChunk(chunk_id=f"{email}_{index}", email_id=email, chunk_index=index, content=text,
        metadata={"source_start": 0, "source_end": len(text), "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
                  "source_length": len(text), "labels": '["INBOX"]', **metadata})


def vectors(texts):
    return [[float(hashlib.sha256(t.encode()).digest()[0]), 1.] for t in texts]


@pytest.fixture(params=["memory", "chroma"])
def index_store(request, tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "CHROMA_PERSIST_DIR", str(tmp_path / "db"))
    monkeypatch.setattr(cfg, "EMBEDDING_MODEL_REVISION", "offline-revision")
    monkeypatch.setattr(cfg, "EMBEDDING_DIMENSION", 2)
    if request.param == "chroma":
        chroma = pytest.importorskip("chromadb")
        from chromadb.config import Settings
        client = chroma.PersistentClient(path=cfg.CHROMA_PERSIST_DIR, settings=Settings(anonymized_telemetry=False))
    else:
        client = MemoryClient()
    monkeypatch.setattr(db, "_get_client", lambda: client)
    monkeypatch.setattr(db, "_collection", None)
    monkeypatch.setattr(db, "resolved_model_revision", lambda: cfg.EMBEDDING_MODEL_REVISION)
    encode = Mock(side_effect=vectors)
    monkeypatch.setattr(db, "embed_texts", encode)
    monkeypatch.setattr(db, "_get_model", Mock(side_effect=AssertionError("no real model in regression")))
    yield client, encode
    if request.param == "chroma":
        client.close()


def run(chunks, **kwargs):
    with collect_index_metrics() as metrics:
        count = db.index_chunks(chunks, **kwargs)
    return count, metrics.to_dict()


def active():
    return manifests.read_active_manifest()["generation"]


@pytest.mark.parametrize("replace", [False, True])
def test_identical_input_no_model_no_generation_no_cache_revision_change(index_store, replace):
    client, encode = index_store
    chunks = [item(), item("b")]
    run(chunks, replace=True)
    old, revision = active(), db.get_corpus_revision()
    encode.reset_mock()
    _, report = run(chunks, replace=replace)
    assert report["outcome"] == "unchanged"
    assert report["counts"]["unchanged_emails"] == 2
    assert report["counts"]["embedded_chunks"] == 0
    assert active() == old and db.get_corpus_revision() == revision
    assert len(manifests.list_generations()) == 1
    assert len(list((manifests.root_path() / "sources").glob("*.jsonl"))) == 1
    encode.assert_not_called()


def test_partial_unchanged_preserves_omitted_emails(index_store):
    _, encode = index_store
    run([item(), item("b")], replace=True)
    old = active()
    encode.reset_mock()
    _, report = run([item()])
    assert report["outcome"] == "unchanged" and active() == old
    assert {c["metadata"]["email_id"] for c in db.get_all_chunks()} == {"a", "b"}
    encode.assert_not_called()


def test_metadata_changes_reuse_vectors_and_refresh_source_fields(index_store):
    _, encode = index_store
    run([item()], replace=True)
    old = active()
    encode.reset_mock()
    _, report = run([item(labels='["STARRED"]', source_start=27, source_end=39, source_sha256="new source")])
    assert active() != old
    assert report["counts"]["metadata_only_emails"] == 1
    row = db.get_all_chunks()[0]
    assert row["metadata"]["labels"] == '["STARRED"]'
    assert row["metadata"]["source_start"] == 27
    assert row["metadata"]["source_sha256"] == "new source"
    encode.assert_not_called()


def test_only_changed_text_encoded_and_batch_order_retained(index_store):
    _, encode = index_store
    original = [item(index=i) for i in range(3)]
    run(original, replace=True)
    encode.reset_mock()
    changed = [original[0], item(index=1, text="Revised evidence"), original[2]]
    _, report = run(changed)
    encode.assert_called_once_with(["Revised evidence"])
    assert report["counts"]["reused_chunks"] == 2
    assert report["counts"]["embedded_chunks"] == 1
    collection = db._get_collection()
    for chunk in changed:
        assert list(collection.get(ids=[chunk.chunk_id], include=["embeddings"])["embeddings"][0]) == pytest.approx(vectors([chunk.content])[0], rel=1e-6)


def test_changed_ordinals_reuse_exact_text_but_refresh_provenance(index_store):
    _, encode = index_store
    run([item(index=0, text="first"), item(index=1, text="second")], replace=True)
    encode.reset_mock()
    _, report = run([item(index=0, text="inserted"), item(index=1, text="first", source_start=9),
                     item(index=2, text="second", source_start=15)])
    encode.assert_called_once_with(["inserted"])
    assert report["counts"]["reused_chunks"] == 2
    rows = {row["chunk_id"]: row for row in db.get_all_chunks()}
    assert rows["a_1"]["content"] == "first" and rows["a_1"]["metadata"]["source_start"] == 9
    assert rows["a_2"]["content"] == "second" and rows["a_2"]["metadata"]["source_start"] == 15


def test_removed_tail_and_snapshot_deletion_need_no_embedding(index_store):
    _, encode = index_store
    run([item(), item(index=1), item("b")], replace=True)
    encode.reset_mock()
    _, report = run([item()], replace=True)
    assert report["outcome"] == "published"
    assert report["counts"]["deleted_emails"] == 1
    assert [c["chunk_id"] for c in db.get_all_chunks()] == ["a_0"]
    encode.assert_not_called()


def test_reuse_is_scoped_to_same_email(index_store):
    _, encode = index_store
    run([item("a", text="same")], replace=True)
    encode.reset_mock()
    run([item("b", text="same")], replace=True)
    encode.assert_called_once_with(["same"])
    assert db.get_all_chunks()[0]["metadata"]["email_id"] == "b"


def test_force_reembed_and_incompatible_model_create_fresh_vectors(index_store, monkeypatch):
    _, encode = index_store
    run([item()], replace=True)
    old = active()
    encode.reset_mock()
    run([item()], force_reembed=True)
    assert active() != old
    encode.assert_called_once()
    encode.reset_mock()
    monkeypatch.setattr(cfg, "EMBEDDING_MODEL_REVISION", "different-revision")
    with pytest.raises(manifests.IndexCompatibilityError):
        run([item()])
    run([item()], replace=True)
    encode.assert_called_once()
    assert manifests.read_active_manifest()["resolved_model_revision"] == "different-revision"


def test_invalid_late_input_does_not_reuse_or_publish(index_store):
    _, encode = index_store
    run([item()], replace=True)
    old = active()
    encode.reset_mock()
    def bad_input():
        yield item()
        raise ValueError("invalid trailing input")
    with pytest.raises(ValueError):
        run(bad_input())
    assert active() == old and len(manifests.list_generations()) == 1
    encode.assert_not_called()


@pytest.mark.parametrize("forced", [False, True])
def test_corrupted_retained_base_refuses_publication(index_store, forced):
    client, encode = index_store
    run([item(), item("b")], replace=True)
    old = active()
    collection = db._get_collection()
    collection.upsert(ids=["b_0"], documents=["tampered"], embeddings=[[1., 0.]],
                      metadatas=[{**item("b").metadata, "email_id": "b", "chunk_index": 0, "index_generation": old}])
    encode.reset_mock()
    with pytest.raises(ValueError, match="verification"):
        run([item(text="changed")], force_reembed=forced)
    assert active() == old
    encode.assert_not_called()


def test_reused_batch_checkpoint_survives_later_failure_and_resumes(index_store):
    _, encode = index_store
    run([item()], replace=True)
    old = active()
    encode.side_effect = RuntimeError("simulated failure")
    with pytest.raises(RuntimeError):
        run([item(labels='["STARRED"]'), item("new")], batch_size=1)
    assert active() == old
    paused = next(m for m in manifests.list_generations() if m["status"] == "paused")
    assert paused["indexed_chunks"] == 1
    encode.side_effect = vectors
    encode.reset_mock()
    with collect_index_metrics() as metrics:
        assert db.resume_index_generation(paused["generation"]) == 2
    encode.assert_called_once_with([item("new").content])
    assert metrics.outcome == "published"
    assert {r["metadata"]["email_id"] for r in db.get_all_chunks()} == {"a", "new"}


def test_repeated_empty_clear_is_noop(index_store):
    _, encode = index_store
    run([item()], replace=True)
    db.clear_collection()
    old = active()
    encode.reset_mock()
    with collect_index_metrics() as metrics:
        db.clear_collection()
    assert metrics.outcome == "unchanged" and active() == old
    encode.assert_not_called()
