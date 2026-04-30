import logging
import re
from typing import List

from rank_bm25 import BM25Okapi

from models.schemas import SearchResult
from core.embedder import search_similar, get_all_chunks
import config.settings as cfg

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    # Handle both Chinese characters and English words/numbers
    return re.findall(r"[一-鿿]|[a-zA-Z0-9]+", text.lower())


def vector_search(query: str, top_k: int = None) -> List[SearchResult]:
    top_k = top_k or cfg.TOP_K
    raw = search_similar(query, top_k=top_k)
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


def bm25_search(query: str, top_k: int = None) -> List[SearchResult]:
    top_k = top_k or cfg.TOP_K
    all_chunks = get_all_chunks()
    if not all_chunks:
        return []

    corpus = [c["content"] for c in all_chunks]
    bm25 = BM25Okapi([_tokenize(doc) for doc in corpus])
    scores = bm25.get_scores(_tokenize(query))

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    results = []
    for idx in top_indices:
        if scores[idx] <= 0:
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


def hybrid_search(query: str, top_k: int = None) -> List[SearchResult]:
    """Reciprocal Rank Fusion of vector + BM25 results. Respects ENABLE_BM25 / ENABLE_RRF flags."""
    top_k = top_k or cfg.TOP_K
    vec_results = vector_search(query, top_k=top_k)

    if not cfg.ENABLE_BM25:
        return vec_results[:top_k]

    bm25_results = bm25_search(query, top_k=top_k)

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
