from __future__ import annotations

import logging
import re
import threading
import heapq
from typing import List, Optional, Tuple

from models.schemas import SearchResult
from core.embedder import search_similar, get_all_chunks, get_collection_count, get_corpus_revision, get_filtered_chunks
from core.filters import FilterSpec, FilterCoverageError
import config.settings as cfg

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    # Handle both Chinese characters and English words/numbers
    return re.findall(r"[一-鿿]|[a-zA-Z0-9]+", text.lower())


# BM25 index cache. With 5000 emails, rebuilding the index from scratch on
# every query (ChromaDB full read + tokenize all chunks + Okapi build) is the
# dominant retrieval cost. Cache the index and invalidate when the persisted
# corpus revision or chunk count changes, including same-count updates.
_bm25_lock = threading.Lock()
_bm25_cache = None
_filtered_bm25_cache = None
# tuple: ((corpus_revision, chunk_count), bm25_index, all_chunks, corpus_texts)


def _get_bm25_index() -> Optional[Tuple[BM25Okapi, List[dict], List[str]]]:
    """Return cached (bm25, all_chunks, corpus); rebuild after corpus changes.

    The fast path reads the cheap collection count and SQLite revision. Only
    fetch all documents and rebuild Okapi when either differs from the cache.
    """
    global _bm25_cache
    n = get_collection_count()
    if n == 0:
        return None
    version = (get_corpus_revision(), n)
    with _bm25_lock:
        if _bm25_cache is not None and _bm25_cache[0] == version:
            return _bm25_cache[1], _bm25_cache[2], _bm25_cache[3]
        logger.info(f"Building BM25 index over {n} chunks")
        all_chunks = get_all_chunks()
        if not all_chunks:
            return None
        corpus = [c["content"] for c in all_chunks]
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi([_tokenize(doc) for doc in corpus])
        # If a writer completed during this fetch, this snapshot is usable only
        # for this request; the next request must fetch a fresh corpus.
        _bm25_cache = ((version if get_corpus_revision() == version[0] else None),
                       bm25, all_chunks, corpus)
        return bm25, all_chunks, corpus


def invalidate_bm25_cache() -> None:
    """Call after re-indexing emails so the next query rebuilds BM25."""
    global _bm25_cache, _filtered_bm25_cache
    with _bm25_lock:
        _bm25_cache = None
        _filtered_bm25_cache = None


def _get_filtered_bm25_index(scope: FilterSpec):
    """One bounded scope cache; filtered requests do not load the whole corpus."""
    global _filtered_bm25_cache
    version = (get_corpus_revision(), get_collection_count(), scope,
               getattr(cfg, "FILTER_LEXICAL_MAX_CHUNKS", 10_000),
               getattr(cfg, "FILTER_LEXICAL_CHAR_LIMIT", 10_000_000),
               getattr(cfg, "FILTER_METADATA_SCAN_LIMIT", 100_000))
    with _bm25_lock:
        if _filtered_bm25_cache is not None and _filtered_bm25_cache[0] == version:
            return _filtered_bm25_cache[1:]
        chunks = get_filtered_chunks(scope)
        if not chunks:
            return None
        corpus = [chunk["content"] for chunk in chunks]
        tokenized = [_tokenize(doc) for doc in corpus]
        if not any(tokenized):
            return None
        from rank_bm25 import BM25Okapi
        index = BM25Okapi(tokenized)
        if get_corpus_revision() != version[0]:
            raise FilterCoverageError("Index changed while building lexical scope; retry the search")
        _filtered_bm25_cache = (version, index, chunks, corpus)
        return index, chunks, corpus


def vector_search(query: str, top_k: int = None, *, filters: FilterSpec | None = None) -> List[SearchResult]:
    top_k = top_k or cfg.TOP_K
    raw = search_similar(query, top_k=top_k, **({"filters": filters} if filters and filters.active else {}))
    return [
        SearchResult(
            chunk_id=item["chunk_id"],
            email_id=item["email_id"],
            content=item["content"],
            score=item["score"],
            metadata=item["metadata"],
        )
        for item in raw
    ]


def bm25_search(query: str, top_k: int = None, *, filters: FilterSpec | None = None) -> List[SearchResult]:
    top_k = top_k or cfg.TOP_K
    cached = _get_filtered_bm25_index(filters) if filters and filters.active else _get_bm25_index()
    if cached is None:
        return []
    bm25, all_chunks, corpus = cached
    scores = bm25.get_scores(_tokenize(query))

    query_tokens = set(_tokenize(query))
    candidates = (i for i in range(len(scores))
                  if (filters is None or filters.matches(all_chunks[i]["metadata"]))
                  and (not filters or not filters.active or query_tokens.intersection(_tokenize(corpus[i]))))
    top_indices = heapq.nlargest(top_k, candidates, key=lambda i: scores[i])
    results = []
    for idx in top_indices:
        # Okapi scores can be nonpositive in tiny scopes. Scoped candidates
        # already have literal overlap, which is a hit; RRF consumes its rank.
        if not (filters and filters.active) and scores[idx] <= 0:
            continue
        meta = all_chunks[idx]["metadata"]
        results.append(
            SearchResult(
                chunk_id=all_chunks[idx]["chunk_id"],
                email_id=meta.get("email_id", ""),
                content=corpus[idx],
                score=float(scores[idx]),
                metadata=meta,
            )
        )
    return results


def hybrid_search(query: str, top_k: int = None, *, filters: FilterSpec | None = None) -> List[SearchResult]:
    """Reciprocal Rank Fusion of vector + BM25 results. Respects ENABLE_BM25 / ENABLE_RRF flags."""
    top_k = top_k or cfg.TOP_K
    options = {"filters": filters} if filters and filters.active else {}
    revision = get_corpus_revision() if options else None
    vec_results = vector_search(query, top_k=top_k, **options)

    if not cfg.ENABLE_BM25:
        return vec_results[:top_k]

    bm25_results = bm25_search(query, top_k=top_k, **options)
    if revision is not None and get_corpus_revision() != revision:
        raise FilterCoverageError("Index changed between retrieval branches; retry the search")

    if not cfg.ENABLE_RRF:
        # Simple score merge without RRF weights
        result_map: dict[str, SearchResult] = {r.chunk_id: r for r in bm25_results}
        result_map.update({r.chunk_id: r for r in vec_results})
        merged = sorted(result_map.values(), key=lambda r: r.score, reverse=True)
        return merged[:top_k]

    RRF_K = 60
    fused_scores: dict[str, float] = {}
    rrf_map: dict[str, SearchResult] = {}

    for rank, r in enumerate(vec_results):
        key = r.chunk_id
        fused_scores[key] = fused_scores.get(key, 0.0) + cfg.VECTOR_WEIGHT / (rank + RRF_K)
        rrf_map[key] = r

    for rank, r in enumerate(bm25_results):
        key = r.chunk_id
        fused_scores[key] = fused_scores.get(key, 0.0) + cfg.BM25_WEIGHT / (rank + RRF_K)
        rrf_map.setdefault(key, r)

    sorted_keys = sorted(fused_scores, key=lambda x: fused_scores[x], reverse=True)[:top_k]
    return [
        SearchResult(
            chunk_id=rrf_map[k].chunk_id,
            email_id=rrf_map[k].email_id,
            content=rrf_map[k].content,
            score=fused_scores[k],
            metadata=rrf_map[k].metadata,
        )
        for k in sorted_keys
    ]
