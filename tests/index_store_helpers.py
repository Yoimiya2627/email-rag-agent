"""Small deterministic storage double; real Chroma contracts live separately."""
from copy import deepcopy


class MemoryCollection:
    def __init__(self, owner):
        self.owner, self.rows, self.events = owner, {}, []

    def count(self):
        return len(self.rows)

    def upsert(self, ids, documents, metadatas, embeddings):
        self.owner.upserts += 1
        if self.owner.fail_upsert == self.owner.upserts:
            raise OSError("synthetic upsert failure")
        self.events.append(("upsert", list(ids)))
        for cid, doc, meta, vector in zip(ids, documents, metadatas, embeddings):
            self.rows[cid] = {"content": doc, "email_id": meta["email_id"], "metadata": deepcopy(meta), "embedding": list(vector)}

    def get(self, where=None, include=None, limit=None, offset=0, ids=None):
        selected = [cid for cid, row in self.rows.items() if (ids is None or cid in ids)
                    and (not where or all(row["metadata"].get(k) == v for k, v in where.items()))]
        selected = selected[offset:offset + limit if limit is not None else None]
        return {"ids": selected, "documents": [self.rows[cid]["content"] for cid in selected],
                "metadatas": [deepcopy(self.rows[cid]["metadata"]) for cid in selected],
                "embeddings": [list(self.rows[cid]["embedding"]) for cid in selected]}


class MemoryClient:
    def __init__(self):
        self.collections, self.upserts, self.fail_upsert = {}, 0, None

    def get_or_create_collection(self, name, **kwargs):
        return self.collections.setdefault(name, MemoryCollection(self))

    def create_collection(self, name, **kwargs):
        assert name not in self.collections
        return self.get_or_create_collection(name)

    def get_collection(self, name, **kwargs):
        return self.collections[name]
