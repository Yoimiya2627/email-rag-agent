import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType

import config.settings as cfg
import pytest
from core import embedder as db


def test_embedding_constructor_is_singleflight_and_configuration_scoped(monkeypatch):
    module = ModuleType("sentence_transformers")
    calls = []
    def constructor(*args, **kwargs):
        calls.append((args, kwargs))
        time.sleep(0.015)
        return object()
    module.SentenceTransformer = constructor
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setattr(db, "_model", None)
    monkeypatch.setattr(db, "_model_key", None)
    monkeypatch.setattr(cfg, "EMBEDDING_MODEL_REVISION", "offline-v1", raising=False)
    with ThreadPoolExecutor(max_workers=8) as pool:
        models = list(pool.map(lambda _: db._get_model(), range(16)))
    assert len(calls) == 1 and all(model is models[0] for model in models)
    monkeypatch.setattr(cfg, "EMBEDDING_MODEL_REVISION", "offline-v2")
    assert db._get_model() is not models[0]
    assert len(calls) == 2 and calls[-1][1]["revision"] == "offline-v2"


def test_chroma_constructor_is_singleflight_and_directory_scoped(monkeypatch, tmp_path):
    module, config = ModuleType("chromadb"), ModuleType("chromadb.config")
    calls = []
    def constructor(**kwargs):
        calls.append(kwargs)
        time.sleep(0.015)
        return object()
    module.PersistentClient = constructor
    config.Settings = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "chromadb", module)
    monkeypatch.setitem(sys.modules, "chromadb.config", config)
    monkeypatch.setattr(db, "_client", None)
    monkeypatch.setattr(db, "_client_key", None)
    monkeypatch.setattr(cfg, "CHROMA_PERSIST_DIR", str(tmp_path / "first"))
    with ThreadPoolExecutor(max_workers=8) as pool:
        clients = list(pool.map(lambda _: db._get_client(), range(16)))
    assert len(calls) == 1 and all(client is clients[0] for client in clients)
    monkeypatch.setattr(cfg, "CHROMA_PERSIST_DIR", str(tmp_path / "second"))
    assert db._get_client() is not clients[0]
    assert len(calls) == 2


def test_cross_encoder_singleflight_and_failed_constructor_can_retry(monkeypatch):
    from core import reranker
    module = ModuleType("sentence_transformers")
    calls = []
    def constructor(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise OSError("synthetic local load failure")
        time.sleep(0.015)
        return object()
    module.CrossEncoder = constructor
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setattr(reranker, "_cross_encoder", None)
    monkeypatch.setattr(reranker, "_cross_encoder_key", None)
    with pytest.raises(OSError):
        reranker._get_cross_encoder()
    assert reranker._cross_encoder is None
    with ThreadPoolExecutor(max_workers=8) as pool:
        models = list(pool.map(lambda _: reranker._get_cross_encoder(), range(16)))
    assert len(calls) == 2 and all(model is models[0] for model in models)
