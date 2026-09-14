from __future__ import annotations

import logging
import copy
import hashlib
import sqlite3
import uuid
import threading
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import List, Optional

from models.schemas import EmailChunk
from core.filters import FilterSpec, FilterCoverageError
from core import index_manifest as manifests
from core.index_metrics import measure_stage, index_operation, set_outcome, add_count
from core.storage_paths import validate_chroma_path
import config.settings as cfg

logger = logging.getLogger(__name__)

_model: Optional[SentenceTransformer] = None
_client: Optional[chromadb.PersistentClient] = None
_collection = None
_model_key = None
_client_key = None
_collection_key = None
_model_lock = threading.Lock()
_client_lock = threading.Lock()
_collection_lock = threading.Lock()
_snapshot = ContextVar("email_index_snapshot", default=None)


def _encoding_options() -> tuple[int, int]:
    batch_size = getattr(cfg, "EMBEDDING_BATCH_SIZE", 32)
    cpu_threads = getattr(cfg, "EMBEDDING_CPU_THREADS", 0)
    if type(batch_size) is not int or not 1 <= batch_size <= 512:
        raise ValueError("EMBEDDING_BATCH_SIZE must be an integer between 1 and 512")
    if type(cpu_threads) is not int or cpu_threads < 0:
        raise ValueError("EMBEDDING_CPU_THREADS must be a nonnegative integer (0 uses runtime defaults)")
    return batch_size, cpu_threads


def _get_model() -> SentenceTransformer:
    global _model, _model_key
    _, cpu_threads = _encoding_options()
    revision = getattr(cfg, "EMBEDDING_MODEL_REVISION", None) or None
    key = (cfg.EMBEDDING_MODEL, cfg.EMBEDDING_DEVICE, revision, cpu_threads)
    with _model_lock:
        if _model is None or (_model_key is not None and _model_key != key):
            with measure_stage("model_load"):
                if cpu_threads:
                    # PyTorch's intra-op setting is process-wide. Set it only
                    # during initialization, never on every embedding request.
                    import torch
                    torch.set_num_threads(cpu_threads)
                from sentence_transformers import SentenceTransformer
                logger.info("Loading embedding model")
                options = {"revision": revision} if revision else {}
                _model = SentenceTransformer(cfg.EMBEDDING_MODEL, device=cfg.EMBEDDING_DEVICE, **options)
        _model_key = key
        return _model


def resolved_model_revision():
    if _model is not None:
        try:
            value = getattr(_model[0].auto_model.config, "_commit_hash", None)
            if isinstance(value, str) and value:
                return value
        except (AttributeError, IndexError, KeyError, TypeError):
            pass
    return getattr(cfg, "EMBEDDING_MODEL_REVISION", None) or None


def _get_client():
    global _client, _client_key
    key = str(validate_chroma_path(cfg.CHROMA_PERSIST_DIR))
    with _client_lock:
        if _client is None or (_client_key is not None and _client_key != key):
            import chromadb
            from chromadb.config import Settings
            _client = chromadb.PersistentClient(path=key, settings=Settings(anonymized_telemetry=False))
        _client_key = key
        return _client


def _descriptor():
    pinned = _snapshot.get()
    key = (str(Path(cfg.CHROMA_PERSIST_DIR).resolve()), cfg.CHROMA_COLLECTION)
    if pinned is not None:
        if pinned[0] != key:
            raise manifests.IndexCompatibilityError("Index configuration changed during this read operation")
        return pinned
    return key, manifests.read_active_manifest()


@contextmanager
def index_snapshot():
    """Pin one immutable generation across all retrieval branches in a request."""
    token = _snapshot.set(_descriptor())
    try:
        yield
    finally:
        _snapshot.reset(token)


def current_index_manifest():
    """Return an isolated manifest copy from the current retrieval snapshot."""
    return copy.deepcopy(_descriptor()[1])


def _legacy_collection():
    return _get_client().get_or_create_collection(name=cfg.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"}, embedding_function=None)


def _get_collection():
    global _collection, _collection_key
    key, manifest = _descriptor()
    name = manifest["collection"] if manifest else cfg.CHROMA_COLLECTION
    full_key = (*key, name)
    with _collection_lock:
        if _collection is None or _collection_key != full_key:
            _collection = (_get_client().get_collection(name=name, embedding_function=None)
                           if manifest else _legacy_collection())
            _collection_key = full_key
        collection = _collection
    if manifest is None and collection.count():
        raise manifests.IndexCompatibilityError("Legacy index has no compatibility manifest; rebuild with --clear (the original collection will be retained)")
    return collection


def _revision_path() -> Path:
    return Path(cfg.CHROMA_PERSIST_DIR) / "corpus_revision.sqlite3"


def _write_marker_path() -> Path:
    name = hashlib.sha256(cfg.CHROMA_COLLECTION.encode("utf-8")).hexdigest()[:16]
    return _revision_path().parent / f".corpus-{name}.writing"


def get_corpus_revision() -> str:
    """Persisted revision catches same-count updates made by other processes."""
    _, manifest = _descriptor()
    if manifest is not None:
        return manifest["generation"]
    marker_base = _write_marker_path()
    if any(marker_base.parent.glob(marker_base.name + "*")):
        # Also covers an abrupt writer crash. Never reuse a BM25 snapshot until
        # another completed indexing operation clears the persisted marker.
        return "unstable-" + uuid.uuid4().hex
    path = _revision_path()
    if not path.exists():
        return "unversioned"
    # Read-only connection: checking cache freshness never creates a database.
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=60)
    try:
        if not connection.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'corpus_revision'").fetchone():
            return "unversioned"
        row = connection.execute("SELECT revision FROM corpus_revision WHERE collection = ?",
                                 (cfg.CHROMA_COLLECTION,)).fetchone()
        return row[0] if row else "unversioned"
    finally:
        connection.close()


@contextmanager
def _corpus_write():
    """Serialize this app's index writers and invalidate caches even on failure.

    Chroma changes are not rolled back by this SQLite transaction. A failed
    write may leave partial data, but readers will never retain its old revision.
    """
    path = _revision_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=60)
    try:
        connection.execute("CREATE TABLE IF NOT EXISTS corpus_revision "
                           "(collection TEXT PRIMARY KEY, revision TEXT NOT NULL)")
        connection.commit()
        connection.execute("BEGIN IMMEDIATE")
        marker_base = _write_marker_path()
        marker = marker_base.with_name(marker_base.name + "." + uuid.uuid4().hex)
        marker.write_text(uuid.uuid4().hex, encoding="ascii")
        # The SQLite writer lock proves previous markers are no longer active.
        for old_marker in marker_base.parent.glob(marker_base.name + "*"):
            if old_marker != marker:
                old_marker.unlink(missing_ok=True)
        try:
            yield
        finally:
            connection.execute("INSERT OR REPLACE INTO corpus_revision VALUES (?, ?)",
                               (cfg.CHROMA_COLLECTION, uuid.uuid4().hex))
            connection.commit()
            marker.unlink(missing_ok=True)
    finally:
        connection.close()


def embed_texts(texts: List[str]) -> List[List[float]]:
    batch_size, _ = _encoding_options()
    model = _get_model()
    with measure_stage("embedding"):
        vectors = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()


def index_chunks(chunks, batch_size: int = 64, *, replace: bool = False, force_reembed: bool = False) -> int:
    """Build and verify an isolated generation, then atomically publish it."""
    from core.index_generation import build_index
    return build_index(chunks, batch_size=batch_size, replace=replace, force_reembed=force_reembed)


def resume_index_generation(generation: str) -> int:
    from core.index_generation import resume_index_generation as resume
    return resume(generation)


def rollback_generation(generation: str) -> dict:
    from core.index_generation import rollback_generation as rollback
    return rollback(generation)


def _filter_selectors(collection, scope: FilterSpec, *, max_chunks=None, max_characters=None) -> list[dict]:
    """Resolve legacy string metadata without loading any message documents.

    Chroma versions supported by the app do not share substring/JSON-label
    predicates. Enumerate bounded metadata pages, then query exact chunk keys.
    Exceeding the budget raises a coverage error, never a false empty result.
    """
    limit = int(getattr(cfg, "FILTER_METADATA_SCAN_LIMIT", 100_000))
    page_size = int(getattr(cfg, "FILTER_METADATA_PAGE_SIZE", 500))
    if limit < 1 or page_size < 1:
        raise ValueError("filter metadata budgets must be positive")
    count = collection.count()
    if count > limit:
        raise FilterCoverageError("Filter scope exceeds the metadata scan budget; narrow the indexed corpus or increase FILTER_METADATA_SCAN_LIMIT")
    selectors = []
    characters = 0
    offset = 0
    while offset < count:
        from agents.runtime import remaining_timeout
        remaining_timeout(getattr(cfg, "LLM_TIMEOUT", 60))
        page = collection.get(include=["metadatas"], limit=min(page_size, count - offset), offset=offset)
        ids, metadata = page.get("ids", []), page.get("metadatas", [])
        if not ids or len(ids) != len(metadata) or len(ids) > min(page_size, count - offset):
            raise FilterCoverageError("Index metadata changed or could not be completely enumerated; retry the search")
        for meta in metadata:
            if not scope.matches(meta):
                continue
            email_id, index = meta.get("email_id"), meta.get("chunk_index")
            if not isinstance(email_id, str) or not email_id or type(index) is not int:
                raise FilterCoverageError("Matching index metadata lacks stable chunk keys; reindex before filtered search")
            if max_chunks is not None and len(selectors) >= max_chunks:
                raise FilterCoverageError("Filtered lexical scope exceeds its chunk budget; narrow the filter or increase FILTER_LEXICAL_MAX_CHUNKS")
            if max_characters is not None:
                start, end = meta.get("source_start"), meta.get("source_end")
                if type(start) is not int or type(end) is not int or not 0 <= start < end:
                    raise FilterCoverageError("Matching index lacks chunk length metadata; reindex before bounded lexical search")
                characters += end - start
                if characters > max_characters:
                    raise FilterCoverageError("Filtered lexical scope exceeds its character budget; narrow the filter or increase FILTER_LEXICAL_CHAR_LIMIT")
            selectors.append({"$and": [{"email_id": {"$eq": email_id}}, {"chunk_index": {"$eq": index}}]})
        offset += len(ids)
    return selectors


@index_snapshot()
def get_filtered_chunks(scope: FilterSpec) -> List[dict]:
    """Load only the complete matching lexical scope within explicit budgets."""
    collection = _get_collection()
    revision = get_corpus_revision()
    if revision.startswith("unstable-"):
        raise FilterCoverageError("Index update is in progress; retry filtered search after it completes")
    max_chunks = int(getattr(cfg, "FILTER_LEXICAL_MAX_CHUNKS", 10_000))
    max_chars = int(getattr(cfg, "FILTER_LEXICAL_CHAR_LIMIT", 10_000_000))
    if max_chunks < 1 or max_chars < 1:
        raise ValueError("filtered lexical budgets must be positive")
    selectors = _filter_selectors(collection, scope, max_chunks=max_chunks, max_characters=max_chars)
    chunks = []
    characters = 0
    for offset in range(0, len(selectors), 100):
        from agents.runtime import remaining_timeout
        remaining_timeout(getattr(cfg, "LLM_TIMEOUT", 60))
        batch = selectors[offset:offset + 100]
        predicate = batch[0] if len(batch) == 1 else {"$or": batch}
        result = collection.get(where=predicate, include=["documents", "metadatas"], limit=len(batch) + 1)
        ids, docs, metas = result["ids"], result["documents"], result["metadatas"]
        if len(ids) != len(batch) or len(docs) != len(ids) or len(metas) != len(ids):
            raise FilterCoverageError("Filtered index keys changed or are ambiguous; retry or reindex")
        for cid, content, meta in zip(ids, docs, metas):
            start, end = (meta or {}).get("source_start"), (meta or {}).get("source_end")
            if (not scope.matches(meta) or not isinstance(content, str) or type(start) is not int
                    or type(end) is not int or len(content) != end - start):
                raise FilterCoverageError("Filtered index content differs from its metadata; reindex before searching")
            characters += len(content)
            if characters > max_chars:
                raise FilterCoverageError("Filtered lexical content exceeded its declared budget")
            chunks.append({"chunk_id": cid, "content": content, "metadata": meta})
    if get_corpus_revision() != revision:
        raise FilterCoverageError("Index changed during lexical scope loading; retry the search")
    return chunks


@index_snapshot()
def search_similar(query: str, top_k: int = None, *, filters: FilterSpec | None = None,
                   where: dict | None = None) -> List[dict]:
    top_k = top_k or cfg.TOP_K
    collection = _get_collection()
    count = collection.count()
    if count == 0:
        return []
    if filters is not None and where is not None:
        raise ValueError("provide either a filter scope or a native where predicate")
    revision = get_corpus_revision() if filters and filters.active else None
    if revision and revision.startswith("unstable-"):
        raise FilterCoverageError("Index update is in progress; retry filtered search after it completes")
    selectors = _filter_selectors(collection, filters) if filters and filters.active else None
    if selectors == []:
        if get_corpus_revision() != revision:
            raise FilterCoverageError("Index changed during filter selection; retry the search")
        return []
    query_vec = embed_texts([query])[0]
    _, manifest = _descriptor()
    if manifest is not None:
        if len(query_vec) != manifest["embedding_dimension"]:
            raise manifests.IndexCompatibilityError("Query embedding dimension differs from the active index; rebuild with the intended model")
        resolved = resolved_model_revision()
        if not resolved or resolved != manifest.get("resolved_model_revision"):
            raise manifests.IndexCompatibilityError("Embedding model revision is unknown or differs from the index; pin EMBEDDING_MODEL_REVISION and rebuild")
    batches = [None]
    if selectors is not None:
        batch_size = int(getattr(cfg, "FILTER_VECTOR_BATCH_SIZE", 100))
        if batch_size < 1:
            raise ValueError("FILTER_VECTOR_BATCH_SIZE must be positive")
        batches = [selectors[i:i + batch_size] for i in range(0, len(selectors), batch_size)]
    items = []
    for batch in batches:
        from agents.runtime import remaining_timeout
        remaining_timeout(getattr(cfg, "LLM_TIMEOUT", 60))
        predicate = (batch[0] if len(batch) == 1 else {"$or": batch}) if batch else where
        options = {"where": predicate} if predicate else {}
        results = collection.query(query_embeddings=[query_vec],
            n_results=min(top_k, len(batch) if batch else count),
            include=["documents", "metadatas", "distances"], **options)
        for chunk_id, doc, meta, dist in zip(results["ids"][0], results["documents"][0],
                                            results["metadatas"][0], results["distances"][0]):
            if filters is not None and not filters.matches(meta):
                raise FilterCoverageError("Index filter snapshot changed; retry the search")
            items.append({"chunk_id": chunk_id, "email_id": meta.get("email_id", ""),
                          "content": doc, "score": float(1.0 - dist), "metadata": meta})
        # Keep memory bounded by top-k plus one query batch.
        items = sorted({item["chunk_id"]: item for item in items}.values(),
                       key=lambda item: (-item["score"], item["chunk_id"]))[:top_k]
    if revision is not None and get_corpus_revision() != revision:
        raise FilterCoverageError("Index changed during filtered search; retry the search")
    return items


def get_collection_count() -> int:
    """Cheap count probe — does not fetch any documents."""
    return _get_collection().count()


@index_snapshot()
def get_all_chunks() -> List[dict]:
    collection = _get_collection()
    count = collection.count()
    if count == 0:
        return []
    limit = int(getattr(cfg, "BM25_MAX_CHUNKS", 100_000))
    char_limit = int(getattr(cfg, "BM25_CHAR_LIMIT", 10_000_000))
    _, manifest = _descriptor()
    if count > limit or (manifest and manifest.get("character_count", 0) > char_limit):
        raise FilterCoverageError("Lexical corpus exceeds its memory budget; use narrower filters or increase BM25 budgets")
    items = []
    characters = 0
    for offset in range(0, count, 500):
        from agents.runtime import remaining_timeout
        remaining_timeout(60)
        result = collection.get(include=["documents", "metadatas"], limit=min(500, count - offset), offset=offset)
        if not result["ids"]:
            raise FilterCoverageError("Lexical corpus enumeration stopped before all chunks were read")
        for chunk_id, doc, meta in zip(result["ids"], result["documents"], result["metadatas"]):
            characters += len(doc)
            if characters > char_limit:
                raise FilterCoverageError("Lexical corpus exceeded its character budget")
            items.append({"chunk_id": chunk_id, "content": doc, "metadata": meta})
    return items


@index_snapshot()
def get_all_metadata() -> List[dict]:
    """Bounded metadata-only rows for deterministic corpus statistics."""
    collection = _get_collection()
    count = collection.count()
    limit = int(getattr(cfg, "STATS_METADATA_SCAN_LIMIT", 100_000))
    if count > limit:
        raise FilterCoverageError("Statistics exceed the metadata scan budget; increase STATS_METADATA_SCAN_LIMIT")
    rows = []
    for offset in range(0, count, 500):
        from agents.runtime import remaining_timeout
        remaining_timeout(60)
        result = collection.get(include=["metadatas"], limit=min(500, count - offset), offset=offset)
        rows.extend({"chunk_id": cid, "metadata": meta} for cid, meta in zip(result["ids"], result["metadatas"]))
    if len(rows) != count:
        raise FilterCoverageError("Statistics could not enumerate complete metadata")
    return rows


@index_snapshot()
def get_email_chunk(email_id: str, chunk_index: int) -> dict | None:
    """Read one exact chunk without loading the rest of a potentially large mail."""
    if not isinstance(email_id, str) or not email_id.strip():
        raise ValueError("email_id must be a non-empty string")
    if type(chunk_index) is not int or chunk_index < 0:
        raise ValueError("chunk_index must be a non-negative integer")
    result = _get_collection().get(
        where={"$and": [{"email_id": email_id}, {"chunk_index": chunk_index}]},
        include=["documents", "metadatas"], limit=2)
    if not result["ids"]:
        return None
    if not len(result["ids"]) == len(result["documents"]) == len(result["metadatas"]) == 1:
        raise ValueError("Indexed email chunk identity is ambiguous")
    meta = result["metadatas"][0]
    if meta.get("email_id") != email_id or meta.get("chunk_index") != chunk_index:
        raise ValueError("Indexed email chunk identity mismatch")
    return {"chunk_id": result["ids"][0], "email_id": email_id,
            "content": result["documents"][0], "metadata": meta}


def get_email_chunks(email_id: str) -> List[dict]:
    """Lookup by indexed email_id; never scan documents belonging to other mail."""
    if not isinstance(email_id, str) or not email_id.strip():
        raise ValueError("email_id must be a non-empty string")
    result = _get_collection().get(where={"email_id": email_id}, include=["documents", "metadatas"])
    chunks = [
        {"chunk_id": cid, "email_id": email_id, "content": doc,
         "score": 0.0, "metadata": meta or {}}
        for cid, doc, meta in zip(result["ids"], result["documents"], result["metadatas"])
    ]
    return sorted(chunks, key=lambda chunk: chunk["metadata"].get("chunk_index", 0))


def get_indexed_email(email_id: str) -> dict:
    """Return cleaned indexed text, not the original MIME email.

    Source offsets remove only proven overlap. Historical chunks without offsets
    remain readable, but their reconstruction is explicitly marked approximate.
    """
    chunks = get_email_chunks(email_id)
    if not chunks:
        return {"error": "email_id not found"}
    metadata = chunks[0]["metadata"]
    body = ""
    exact = True
    for chunk in chunks:
        meta, part = chunk["metadata"], chunk["content"]
        start, end = meta.get("source_start"), meta.get("source_end")
        if (type(start) is not int or type(end) is not int
                or not 0 <= start <= len(body) < end or end - start != len(part)
                or meta.get("source_sha256") != metadata.get("source_sha256")
                or meta.get("source_length") != metadata.get("source_length")
                or body[start:] != part[:len(body) - start]):
            exact = False
            break
        body += part[len(body) - start:]
    exact = (exact and len(body) == metadata.get("source_length")
             and hashlib.sha256(body.encode("utf-8")).hexdigest() == metadata.get("source_sha256"))
    if not exact:
        # There is no safe way to infer whether matching text was original or
        # synthetic overlap in an old index. Preserve all evidence for reindexing.
        body = "\n\n".join(chunk["content"] for chunk in chunks)
    return {"email_id": email_id, "subject": metadata.get("subject", ""),
            "sender": metadata.get("sender", ""), "date": metadata.get("date", ""),
            "body": body, "body_source": "indexed_chunks", "chunks": chunks,
            "body_format": metadata.get("body_format", "plain"),
            "reconstruction_exact": exact, "reindex_required": not exact}


def verify_collection_readiness() -> dict:
    """Read the published native collection; a manifest alone is not readiness."""
    validate_chroma_path(cfg.CHROMA_PERSIST_DIR)
    with index_snapshot():
        _, manifest = _descriptor()
        if manifest is None:
            raise manifests.IndexCompatibilityError('No published index generation is available')
        count = _get_collection().count()
        if type(count) is not int or count != manifest['chunk_count'] or count <= 0:
            raise manifests.IndexCompatibilityError('Published collection count does not match a nonempty manifest')
        return {'chunk_count':count, 'email_count':manifest['email_count'],
                'collection':cfg.CHROMA_COLLECTION, 'generation':manifest['generation'],
                'requires_rebuild':False, 'compatibility_status':'compatible', 'storage_verified':True}


def get_collection_stats() -> dict:
    manifest = manifests.read_active_manifest(validate_compatibility=False)
    if manifest is not None:
        compatible = manifest["config_fingerprint"] == manifests.configuration_fingerprint()
        return {"chunk_count": manifest["chunk_count"], "email_count": manifest["email_count"],
                "collection": cfg.CHROMA_COLLECTION, "generation": manifest["generation"],
                "requires_rebuild": not compatible, "compatibility_status": "compatible" if compatible else "incompatible"}
    count = _legacy_collection().count()
    return {"chunk_count": count, "email_count": None if count else 0,
            "collection": cfg.CHROMA_COLLECTION, "generation": None,
            "requires_rebuild": bool(count), "compatibility_status": "legacy" if count else "empty"}


@index_operation
def clear_collection():
    global _collection, _collection_key
    with _corpus_write():
        # Publish an empty, compatible generation; retain old data for readers
        # and explicit rollback rather than deleting a collection they may hold.
        active = manifests.read_active_manifest(validate_compatibility=False)
        if active and active.get("chunk_count") == 0 and active.get("config_fingerprint") == manifests.configuration_fingerprint():
            from core.index_generation import verify_generation
            metrics = verify_generation(active)
            if any(active.get(key) != value for key, value in metrics.items()):
                raise ValueError("Empty generation verification failed")
            set_outcome("unchanged")
            return
        generation = uuid.uuid4().hex
        _get_client().create_collection(name="idx_" + generation,
            metadata={"hnsw:space": "cosine"}, embedding_function=None)
        configuration = manifests.configuration()
        empty_hash = hashlib.sha256(manifests.canonical_bytes([0, "0" * 64])).hexdigest()
        state = {"generation": generation, "collection": "idx_" + generation, "status": "ready",
                 "created_at": manifests.utc_now(), "configuration": configuration,
                 "config_fingerprint": manifests.configuration_fingerprint(configuration),
                 "chunk_count": 0, "email_count": 0, "character_count": 0,
                 "corpus_sha256": empty_hash, "embedding_dimension": None,
                 "resolved_model_revision": resolved_model_revision()}
        with measure_stage("publish"):
            manifests.activate(state, expected_generation=(active or {}).get("generation"))
        add_count("deleted_emails", (active or {}).get("email_count", 0) or 0)
        set_outcome("published")
        _collection = None
        _collection_key = None
    logger.info("Published empty index generation")


def _get_embedding_fn():
    """Return a LangChain-compatible embedding function wrapping our local model."""
    class _EmbeddingFn:
        def embed_documents(self, texts):
            return embed_texts(texts)

        def embed_query(self, text):
            return embed_texts([text])[0]

    return _EmbeddingFn()
