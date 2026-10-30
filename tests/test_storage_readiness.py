"""Native storage must be usable before the service advertises readiness."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import api.main as api
from core import storage_paths
from core.storage_paths import validate_chroma_path
from scripts import doctor


def test_windows_chroma_path_rejects_unicode_without_creating_directory(tmp_path):
    path = tmp_path / '中文' / 'chroma'
    with pytest.raises(ValueError, match='ASCII-only resolved path'):
        validate_chroma_path(path, platform_name='nt')
    assert not path.exists()
    assert validate_chroma_path(path, platform_name='posix') == path.resolve()
    assert validate_chroma_path(tmp_path/'chroma', platform_name='nt') == (tmp_path/'chroma').resolve()


def test_doctor_reports_windows_chroma_path_error(monkeypatch, tmp_path):
    monkeypatch.setattr(doctor.metadata,'version',lambda name:'fixture')
    real_validate = validate_chroma_path
    monkeypatch.setattr(storage_paths,'validate_chroma_path',lambda path:real_validate(path,platform_name='nt'))
    result = doctor.diagnose(SimpleNamespace(CHROMA_PERSIST_DIR=tmp_path/'中文'), profile='test',root=tmp_path)
    check = next(row for row in result['checks'] if row['check']=='filesystem:CHROMA_ASCII_PATH')
    assert not result['configuration_valid']
    assert check['status']=='error' and 'E:/email-agent-runtime/chroma_db' in check['detail']


def test_chroma_path_is_checked_before_native_client_creation(monkeypatch, tmp_path):
    from core import embedder
    monkeypatch.setattr(embedder.cfg,'CHROMA_PERSIST_DIR',str(tmp_path/'中文'))
    monkeypatch.setattr(embedder,'validate_chroma_path',lambda path:validate_chroma_path(path,platform_name='nt'))
    # Invalid paths must fail even in the lightweight test environment where
    # the optional native Chroma package is not installed.
    with patch.dict('sys.modules', {'chromadb': None}):
        with pytest.raises(ValueError,match='ASCII-only'):
            embedder._get_client()


def test_ready_refuses_unreadable_storage_even_after_successful_warmup():
    with patch.object(api.readiness,'snapshot',return_value={'status':'ready','components':{}}), \
         patch.object(api,'get_collection_stats',return_value={'chunk_count':5001,'requires_rebuild':False}), \
         patch.object(api,'verify_collection_readiness',side_effect=RuntimeError('synthetic unreadable HNSW')):
        response = TestClient(api.app).get('/ready')
    assert response.status_code==503
    assert response.json()['ready'] is False
    assert response.json()['index']=={'status':'unavailable','error_code':'RuntimeError'}


def test_warmup_verifies_storage_before_model_initialization():
    with patch.object(api,'verify_collection_readiness',side_effect=RuntimeError('synthetic count mismatch')), \
         patch('core.embedder.embed_texts') as embedding, patch('core.generator._get_client') as model:
        with pytest.raises(RuntimeError,match='count mismatch'):
            api._warm_components()
    embedding.assert_not_called()
    model.assert_not_called()


def test_ready_accepts_readable_matching_storage_after_warmup():
    stats={'chunk_count':3,'requires_rebuild':False,'storage_verified':True}
    with patch.object(api.readiness,'snapshot',return_value={'status':'ready','components':{}}), \
         patch.object(api,'verify_collection_readiness',return_value=stats):
        response=TestClient(api.app).get('/ready')
    assert response.status_code==200 and response.json()['index']['storage_verified'] is True
