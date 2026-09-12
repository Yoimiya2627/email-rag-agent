"""Offline contracts for embedding execution settings and phase boundaries."""
from contextlib import contextmanager
import sys
from types import SimpleNamespace

import pytest

import config.settings as cfg
from core import embedder


@pytest.fixture
def encoder(monkeypatch):
    events = []

    class FakeModel:
        def __init__(self, name, **options):
            events.append(("load", name, options))

        def encode(self, texts, **options):
            events.append(("encode", texts, options))
            return SimpleNamespace(tolist=lambda: [[1.0, 0.0] for _ in texts])

    @contextmanager
    def measure(name):
        events.append(("start", name))
        try:
            yield
        finally:
            events.append(("end", name))

    monkeypatch.setattr(embedder, "_model", None)
    monkeypatch.setattr(embedder, "_model_key", None)
    monkeypatch.setattr(embedder, "measure_stage", measure)
    monkeypatch.setattr(cfg, "EMBEDDING_BATCH_SIZE", 32)
    monkeypatch.setattr(cfg, "EMBEDDING_CPU_THREADS", 0)
    monkeypatch.setattr(cfg, "EMBEDDING_DEVICE", "cpu")
    monkeypatch.setattr(cfg, "EMBEDDING_MODEL_REVISION", "offline-revision")
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=FakeModel))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda n: events.append(("threads", n))))
    return events


@pytest.mark.parametrize("batch_size", [1, 32, 512])
def test_explicit_microbatch_and_nonoverlapping_phases(monkeypatch, encoder, batch_size):
    monkeypatch.setattr(cfg, "EMBEDDING_BATCH_SIZE", batch_size)
    assert embedder.embed_texts(["a", "b"]) == [[1.0, 0.0], [1.0, 0.0]]
    assert [event[0] for event in encoder] == ["start", "load", "end", "start", "encode", "end"]
    assert encoder[0] == ("start", "model_load")
    assert encoder[2:4] == [("end", "model_load"), ("start", "embedding")]
    assert encoder[4][2] == {"batch_size": batch_size, "normalize_embeddings": True, "show_progress_bar": False}
    assert encoder[1][2] == {"device": "cpu", "revision": "offline-revision"}


@pytest.mark.parametrize("setting,value", [
    ("EMBEDDING_BATCH_SIZE", 0), ("EMBEDDING_BATCH_SIZE", 513),
    ("EMBEDDING_BATCH_SIZE", 1.5), ("EMBEDDING_BATCH_SIZE", True),
    ("EMBEDDING_CPU_THREADS", -1), ("EMBEDDING_CPU_THREADS", 1.5),
    ("EMBEDDING_CPU_THREADS", True),
])
def test_invalid_settings_rejected_before_loading(monkeypatch, encoder, setting, value):
    monkeypatch.setattr(cfg, setting, value)
    with pytest.raises(ValueError, match=setting):
        embedder.embed_texts(["a"])
    with pytest.raises(ValueError, match=setting):
        embedder._get_model()
    assert encoder == []


def test_threads_applied_once_and_setting_change_invalidates_model(monkeypatch, encoder):
    monkeypatch.setattr(cfg, "EMBEDDING_CPU_THREADS", 2)
    embedder.embed_texts(["a"])
    model = embedder._get_model()
    embedder.embed_texts(["b"])
    assert [event for event in encoder if event[0] == "threads"] == [("threads", 2)]
    assert sum(event[0] == "load" for event in encoder) == 1
    monkeypatch.setattr(cfg, "EMBEDDING_BATCH_SIZE", 64)
    assert embedder._get_model() is model
    monkeypatch.setattr(cfg, "EMBEDDING_CPU_THREADS", 4)
    assert embedder._get_model() is not model
    assert [event for event in encoder if event[0] == "threads"] == [("threads", 2), ("threads", 4)]


def test_force_reembed_wrapper_preserves_integer_return(monkeypatch):
    from core import index_generation
    calls = []

    def build(chunks, **options):
        calls.append((chunks, options))
        return 7

    monkeypatch.setattr(index_generation, "build_index", build)
    chunks = iter(["sentinel"])
    assert embedder.index_chunks(chunks, batch_size=8, replace=True, force_reembed=True) == 7
    assert calls == [(chunks, {"batch_size": 8, "replace": True, "force_reembed": True})]
