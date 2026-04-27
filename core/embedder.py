import logging
from typing import List, Optional

import chromadb
from sentence_transformers import SentenceTransformer

from models.schemas import EmailChunk
import config.settings as cfg

logger = logging.getLogger(__name__)

_model: Optional[SentenceTransformer] = None
_client: Optional[chromadb.PersistentClient] = None
_collection = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {cfg.EMBEDDING_MODEL}")
        _model = SentenceTransformer(cfg.EMBEDDING_MODEL, device=cfg.EMBEDDING_DEVICE)
    return _model


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=cfg.CHROMA_PERSIST_DIR)
        _collection = _client.get_or_create_collection(
            name=cfg.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def embed_texts(texts: List[str]) -> List[List[float]]:
    model = _get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def index_chunks(chunks: List[EmailChunk], batch_size: int = 64) -> int:
    collection = _get_collection()
    total = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        ids = [c.chunk_id for c in batch]
        texts = [c.content for c in batch]
        metadatas = [
            {**c.metadata, "email_id": c.email_id, "chunk_index": c.chunk_index}
            for c in batch
        ]
        embeddings = embed_texts(texts)
        collection.upsert(
            ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas
        )
        total += len(batch)
        logger.info(f"Indexed batch {i // batch_size + 1}: {total}/{len(chunks)} chunks")
    return total


def search_similar(query: str, top_k: int = None) -> List[dict]:
    top_k = top_k or cfg.TOP_K
    collection = _get_collection()
    count = collection.count()
    if count == 0:
        return []
    n = min(top_k, count)
    query_vec = embed_texts([query])[0]
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    items = []
    for chunk_id, doc, meta, dist in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        items.append(
            {
                "chunk_id": chunk_id,
                "email_id": meta.get("email_id", ""),
                "content": doc,
                "score": float(1.0 - dist),
                "metadata": meta,
            }
        )
    return items


def get_all_chunks() -> List[dict]:
    collection = _get_collection()
    if collection.count() == 0:
        return []
    result = collection.get(include=["documents", "metadatas"])
    items = []
    for chunk_id, doc, meta in zip(
        result["ids"], result["documents"], result["metadatas"]
    ):
        items.append({"chunk_id": chunk_id, "content": doc, "metadata": meta})
    return items


def clear_collection():
    global _collection
    client = chromadb.PersistentClient(path=cfg.CHROMA_PERSIST_DIR)
    client.delete_collection(cfg.CHROMA_COLLECTION)
    _collection = None
    logger.info("ChromaDB collection cleared")
