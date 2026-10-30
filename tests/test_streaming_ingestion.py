import gc
import json
import tracemalloc
from pathlib import Path
from unittest.mock import patch

import pytest
import config.settings as cfg
from core.ingestion import prepare_email_chunks, EmailPlan
from core.json_stream import iter_email_records
from core import embedder as db, index_manifest as manifests
from models.schemas import EmailChunk
from tests.index_store_helpers import MemoryClient


def record(identifier="a", body="normal evidence"):
    return {"id": identifier, "subject": "title", "sender": "a@example.com", "recipients": [], "date": "2026-01-01", "body": body}


@pytest.mark.parametrize("raw", ['{"item":{}}', '[null]', '[1]', '["scalar"]', '[]', '[{},]', '[{}] trailing', '[{}] []', '[', '[{"body":"broken}', '[{}', 'null'])
def test_streaming_rejects_invalid_container_and_tail_without_spool_leaks(tmp_path, raw):
    path = tmp_path / "source.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError):
        prepare_email_chunks(path)


def test_streaming_escaped_unicode_and_block_boundaries_are_repeatable(tmp_path):
    path = tmp_path / "source.json"
    records = [record("a", 'quote \\" {{中文}} ' * 7000), record("b", "tail fact not approved 789.01")]
    path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    emails, chunks = prepare_email_chunks(path)
    assert isinstance(emails, EmailPlan) and len(emails) == 2
    assert emails[1].body == records[1]["body"]
    assert len(list(chunks)) == len(chunks) == len(list(chunks))
    spool = emails.path
    assert spool.exists() and not hasattr(emails, "emails")
    emails.close()
    assert not spool.exists()


@pytest.mark.parametrize("setting,limit,records", [
    ("MAX_EMAIL_RECORD_BYTES", 100, [record(body="private-canary" * 100)]),
    ("MAX_INDEX_INPUT_EMAILS", 1, [record("a"), record("b")]),
    ("MAX_INDEX_INPUT_CHUNKS", 1, [record("a"), record("b")]),
    ("MAX_INDEX_INPUT_BYTES", 100, [record()]),
])
def test_budgets_are_safe_errors_before_index_calls(tmp_path, monkeypatch, setting, limit, records):
    monkeypatch.setattr(cfg, setting, limit, raising=False)
    path = tmp_path / "input.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    with patch.object(db, "_get_client", side_effect=AssertionError("must not initialize index")), pytest.raises(ValueError) as failure:
        prepare_email_chunks(path)
    assert "private-canary" not in str(failure.value)


def test_bad_last_record_and_duplicate_ids_never_reach_index(tmp_path):
    path = tmp_path / "input.json"
    for rows in ([record("a"), record("b"), {**record("c"), "body": {"private-canary": 1}}],
                 [record("a"), record("b"), record("a")]):
        path.write_text(json.dumps(rows), encoding="utf-8")
        with pytest.raises(ValueError) as failure:
            prepare_email_chunks(path)
        assert "record 3" in str(failure.value) and "private-canary" not in str(failure.value)


def test_prepared_configuration_change_refused_before_index_creation(tmp_path, monkeypatch):
    path = tmp_path / "input.json"
    path.write_text(json.dumps([record()]), encoding="utf-8")
    emails, chunks = prepare_email_chunks(path)
    monkeypatch.setattr(cfg, "CHUNK_SIZE", cfg.CHUNK_SIZE + 1)
    with patch.object(db, "_get_client", side_effect=AssertionError("must not initialize index")), pytest.raises(manifests.IndexCompatibilityError):
        db.index_chunks(chunks)
    emails.close()


def test_direct_chunk_generator_budget_preserves_active_generation(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "CHROMA_PERSIST_DIR", str(tmp_path / "index"))
    monkeypatch.setattr(cfg, "EMBEDDING_MODEL_REVISION", "offline-v1", raising=False)
    client = MemoryClient()
    with patch.object(db, "_get_client", return_value=client), patch.object(db, "embed_texts", side_effect=lambda values: [[1., 0.] for _ in values]):
        db.index_chunks([EmailChunk(chunk_id="old", email_id="old", chunk_index=0, content="old")], replace=True)
        old = db.get_corpus_revision()
        writes = client.upserts
        monkeypatch.setattr(cfg, "MAX_INDEX_INPUT_CHUNKS", 2, raising=False)
        with pytest.raises(ValueError, match="MAX_INDEX_INPUT_CHUNKS"):
            db.index_chunks((EmailChunk(chunk_id=str(i), email_id=str(i), chunk_index=0, content="x") for i in range(3)))
        assert db.get_corpus_revision() == old and client.upserts == writes


def test_preflight_memory_does_not_scale_with_total_body_bytes(tmp_path):
    def peak(count):
        path = tmp_path / f"{count}.json"
        with path.open("w", encoding="utf-8") as output:
            output.write("[")
            for index in range(count):
                if index:
                    output.write(",")
                json.dump(record(str(index), "bounded content " * 4000), output)
            output.write("]")
        gc.collect()
        tracemalloc.start()
        emails, chunks = prepare_email_chunks(path)
        _, measured = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert len(emails) == count and len(chunks) > count
        emails.close()
        return measured
    small, large = peak(12), peak(120)
    # Ten times the source text may add bounded parser/ID bookkeeping, never
    # a second full normalized corpus. This is Python allocation, not RSS/GPU.
    assert large < small * 2 + 1_000_000, (small, large)
