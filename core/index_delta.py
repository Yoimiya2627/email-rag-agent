"""Ephemeral, disk-backed lookup scoped to one verified base generation.

Only vectors from the same email and exact encoded text are reused. Source
metadata is always taken from the new input, never from the cached vector row.
"""
from contextlib import contextmanager
import hashlib
import json
import sqlite3
import tempfile
from pathlib import Path

from agents.runtime import remaining_timeout
from core import index_manifest as manifests
from core.index_metrics import add_count, measure_stage
from models.schemas import EmailChunk


def text_key(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_metadata(chunk):
    return {**chunk.metadata, "email_id": chunk.email_id, "chunk_index": chunk.chunk_index}


class ReuseLookup:
    def __init__(self, connection):
        self.connection = connection

    def vector(self, chunk):
        row = self.connection.execute(
            "SELECT document, vector FROM previous WHERE email=? AND text_hash=? LIMIT 1",
            (chunk.email_id, text_key(chunk.content))).fetchone()
        if row is None or row[0] != chunk.content:
            return None
        return json.loads(row[1])

    def classify(self, source_path, replace):
        from core.index_generation import _record_digest
        db = self.connection
        with Path(source_path).open("rb") as source:
            for line in source:
                remaining_timeout(60)
                chunk = EmailChunk.model_validate_json(line)
                db.execute("INSERT INTO incoming VALUES(?,?,?,?)", (chunk.chunk_id, chunk.email_id,
                    text_key(chunk.content), _record_digest(chunk.chunk_id, chunk.content, chunk_metadata(chunk)).hex()))
        db.commit()
        # Full joins count stale tails as changes, even if all incoming chunks
        # individually equal old rows. In upsert mode omitted emails stay intact.
        db.execute("CREATE TEMP TABLE emails AS SELECT DISTINCT email FROM incoming")
        new = db.execute("SELECT count(*) FROM emails e WHERE NOT EXISTS(SELECT 1 FROM previous p WHERE p.email=e.email)").fetchone()[0]
        unchanged = metadata_only = changed = 0
        for (email,) in db.execute("SELECT email FROM emails"):
            remaining_timeout(60)
            old_count = db.execute("SELECT count(*) FROM previous WHERE email=?", (email,)).fetchone()[0]
            if not old_count:
                continue
            new_count = db.execute("SELECT count(*) FROM incoming WHERE email=?", (email,)).fetchone()[0]
            same_text, same_record = db.execute("""SELECT
                coalesce(sum(p.email=i.email AND p.text_hash=i.text_hash),0),
                coalesce(sum(p.email=i.email AND p.record_hash=i.record_hash),0)
                FROM incoming i LEFT JOIN previous p ON p.cid=i.cid WHERE i.email=?""", (email,)).fetchone()
            if old_count == new_count == same_record:
                unchanged += 1
            elif old_count == new_count == same_text:
                metadata_only += 1
            else:
                changed += 1
        deleted = db.execute("SELECT count(DISTINCT email) FROM previous p WHERE NOT EXISTS(SELECT 1 FROM incoming i WHERE i.email=p.email)").fetchone()[0] if replace else 0
        for name, value in (("new_emails", new), ("changed_emails", changed),
                            ("metadata_only_emails", metadata_only), ("unchanged_emails", unchanged),
                            ("deleted_emails", deleted)):
            add_count(name, value)
        return new == changed == metadata_only == deleted == 0


@contextmanager
def verified_reuse_lookup(state, collection):
    """Scan and verify the base once, keeping large documents/vectors on disk."""
    from core.index_generation import _record_digest, _validated_vectors
    with tempfile.TemporaryDirectory(prefix="email-index-reuse-") as directory:
        db = sqlite3.connect(str(Path(directory) / "lookup.sqlite3"))
        try:
            db.execute("PRAGMA cache_size=-8192")
            db.execute("CREATE TABLE previous(cid TEXT PRIMARY KEY,email TEXT NOT NULL,text_hash TEXT NOT NULL,record_hash TEXT NOT NULL,document TEXT NOT NULL,vector TEXT NOT NULL)")
            db.execute("CREATE INDEX previous_email_text ON previous(email,text_hash)")
            db.execute("CREATE TABLE incoming(cid TEXT PRIMARY KEY,email TEXT NOT NULL,text_hash TEXT NOT NULL,record_hash TEXT NOT NULL)")
            db.execute("CREATE INDEX incoming_email ON incoming(email)")
            with measure_stage("compare"):
                offset, digest_sum, characters = 0, 0, 0
                total = collection.count()
                if total != state["chunk_count"]:
                    raise ValueError("Base generation coverage changed; reuse refused")
                while offset < total:
                    remaining_timeout(60)
                    page = collection.get(include=["documents", "metadatas", "embeddings"], limit=min(256, total-offset), offset=offset)
                    ids, docs, metas = page["ids"], page["documents"], page["metadatas"]
                    if not ids or len(ids) != len(docs) or len(ids) != len(metas):
                        raise ValueError("Base generation enumeration is incomplete")
                    vectors, _ = _validated_vectors(page["embeddings"], len(ids), state["embedding_dimension"])
                    for cid, doc, meta, vector in zip(ids, docs, metas, vectors):
                        if meta.get("index_generation") != state["generation"]:
                            raise ValueError("Base generation identity changed; reuse refused")
                        digest = _record_digest(cid, doc, meta)
                        digest_sum = (digest_sum + int.from_bytes(digest, "big")) % (1 << 256)
                        characters += len(doc)
                        db.execute("INSERT INTO previous VALUES(?,?,?,?,?,?)", (cid, meta["email_id"], text_key(doc), digest.hex(), doc, json.dumps(vector)))
                    offset += len(ids)
                actual = hashlib.sha256(manifests.canonical_bytes([total, f"{digest_sum:064x}"])).hexdigest()
                emails = db.execute("SELECT count(DISTINCT email) FROM previous").fetchone()[0]
                if actual != state["corpus_sha256"] or characters != state["character_count"] or emails != state["email_count"]:
                    raise ValueError("Base generation content verification failed; reuse refused")
                db.commit()
            yield ReuseLookup(db)
        finally:
            db.close()
