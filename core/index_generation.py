"""Resumable staging builds. Active generations are never modified in place."""
from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from pathlib import Path
from contextlib import nullcontext

from core import index_manifest as manifests
from models.schemas import EmailChunk
from agents.runtime import current_run, remaining_timeout
from core.index_delta import verified_reuse_lookup
from core.index_metrics import (add_count, current_metrics, index_operation,
                               measure_stage, set_outcome, timed_stage)


def _backend():
    from core import embedder
    return embedder


def _notify(state, stage="indexing"):
    run = current_run()
    if run:
        run.checkpoint({"generation": state["generation"], "generation_id": state["generation"], "kind": "index", "safe": True, "phase": state["phase"],
                        "indexed_chunks": state["indexed_chunks"], "source_offset": state["source_offset"],
                        "resume_supported": bool(state.get("source_complete"))})
        report = current_metrics()
        elapsed = report.elapsed_seconds if report else 0.
        processed = (report.counts.get("processed_chunks", 0) if report else 0)
        rate = processed / elapsed if elapsed > 0 and processed else None
        remaining = max(0, state["source_chunk_count"] - state["indexed_chunks"])
        # This estimate is only for the input-processing phase. Copy/verify and
        # cold model loading are not extrapolated from a single encoded batch.
        eta = remaining / rate if rate and processed >= 2 * state["batch_size"] and state["phase"] == "embed" else None
        run.progress("index_" + state["phase"], completed_chunks=state["indexed_chunks"],
            total_chunks=state["source_chunk_count"], reused_chunks=state.get("reused_chunks", 0),
            embedded_chunks=state.get("embedded_chunks", 0), elapsed_seconds=round(elapsed, 2),
            chunks_per_second=round(rate, 2) if rate else None, remaining_seconds=round(eta, 1) if eta is not None else None)


def _save(state):
    report = current_metrics()
    if report:
        # A checkpoint is not a final performance report. Final publication
        # timing is returned to the caller after the immutable manifest switch.
        state["checkpoint_metrics"] = {**report.to_dict(), "checkpoint_phase": state["phase"]}
    manifests.write_generation(state)
    _notify(state)


def _file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            remaining_timeout(60)
            digest.update(block)
    return digest.hexdigest()


def _record_digest(cid, document, metadata):
    clean = {key: value for key, value in metadata.items() if key != "index_generation"}
    return hashlib.sha256(manifests.canonical_bytes([cid, document, clean])).digest()


def _validated_vectors(vectors, expected_count, dimension=None):
    if len(vectors) != expected_count:
        raise ValueError("Embedding count does not match chunk count")
    output = []
    for vector in vectors:
        values = vector.tolist() if hasattr(vector, "tolist") else vector
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError("Embedding vectors must be nonempty")
        if dimension is None:
            dimension = len(values)
        if len(values) != dimension or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ValueError("Embedding dimension or numeric values are invalid")
        output.append([float(value) for value in values])
    return output, dimension


def _collection(name):
    return _backend()._get_client().get_collection(name=name, embedding_function=None)


@timed_stage("write")
def _upsert(collection, **values):
    collection.upsert(**values)


def _batch_vectors(batch, state, reuse):
    backend = _backend()
    with measure_stage("compare"):
        vectors = [reuse.vector(chunk) if reuse else None for chunk in batch]
    missing = [i for i, vector in enumerate(vectors) if vector is None]
    if missing:
        add_count("embedding_batches", 1)
        with measure_stage("embedding"):
            encoded = backend.embed_texts([batch[i].content for i in missing])
        encoded, dimension = _validated_vectors(encoded, len(missing), state["embedding_dimension"])
        add_count("embedded_chunks", len(missing))
        revision = backend.resolved_model_revision()
        if not isinstance(revision, str) or not revision.strip():
            raise manifests.IndexCompatibilityError("Embedding model revision is unknown; pin EMBEDDING_MODEL_REVISION and rebuild")
        if state["resolved_model_revision"] is not None and revision != state["resolved_model_revision"]:
            raise manifests.IndexCompatibilityError("Embedding model revision changed during the build")
        state["resolved_model_revision"] = revision
        for i, vector in zip(missing, encoded):
            vectors[i] = vector
    vectors, dimension = _validated_vectors(vectors, len(batch), state["embedding_dimension"])
    configured = state["configuration"].get("embedding_dimension")
    if configured and dimension != configured:
        raise ValueError("Embedding dimension differs from configured manifest dimension")
    add_count("reused_chunks", len(batch) - len(missing))
    return vectors, dimension, len(missing)


@timed_stage("verify")
def verify_generation(state, *, batch_size=256):
    """Verify full stored coverage in bounded pages before publication/rollback."""
    collection = _collection(state["collection"])
    count = collection.count()
    email_ids = set()
    digest_sum = 0
    characters = 0
    offset = 0
    while offset < count:
        remaining_timeout(60)
        page = collection.get(include=["documents", "metadatas", "embeddings"], limit=min(batch_size, count - offset), offset=offset)
        ids, docs, metas = page["ids"], page["documents"], page["metadatas"]
        if not ids or len(ids) != len(docs) or len(ids) != len(metas):
            raise ValueError("Generation verification could not enumerate all chunks")
        _validated_vectors(page["embeddings"], len(ids), state["embedding_dimension"])
        for cid, doc, meta in zip(ids, docs, metas):
            if meta.get("index_generation") != state["generation"]:
                raise ValueError("Generation contains chunks from another build")
            email_ids.add(meta["email_id"])
            characters += len(doc)
            digest_sum = (digest_sum + int.from_bytes(_record_digest(cid, doc, meta), "big")) % (1 << 256)
        offset += len(ids)
    return {"chunk_count": count, "email_count": len(email_ids), "character_count": characters,
            "corpus_sha256": hashlib.sha256(manifests.canonical_bytes([count, f"{digest_sum:064x}"])).hexdigest()}


@index_operation
def build_index(chunks, batch_size=64, *, replace=False, force_reembed=False):
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be positive")
    start_fingerprint = manifests.configuration_fingerprint()
    options = getattr(chunks, "options", None)
    prepared_fingerprint = getattr(chunks, "config_fingerprint", None)
    if prepared_fingerprint is not None and prepared_fingerprint != manifests.configuration_fingerprint():
        raise manifests.IndexCompatibilityError("Index configuration changed after preparation; prepare input again")
    if options is not None:
        configuration = manifests.configuration()
        if any(options.get(key) != configuration[key] for key in ("chunk_size", "chunk_overlap", "min_chunk_size")):
            raise manifests.IndexCompatibilityError("Chunk configuration changed after preparation; prepare input again")
    backend = _backend()
    with backend._corpus_write():
        active = manifests.read_active_manifest(validate_compatibility=not replace)
        if not replace and active is None:
            legacy = backend._legacy_collection()
            if legacy.count():
                raise manifests.IndexCompatibilityError("Legacy index has no manifest; rebuild with --clear to preserve it and create a compatible generation")
        generation = uuid.uuid4().hex
        source_path = manifests.source_path(generation)
        source_path.parent.mkdir(parents=True, exist_ok=True)
        ids, email_ids = set(), set()
        count = 0
        try:
            with measure_stage("source_spool"), source_path.open("xb") as output:
                for chunk in chunks:
                    remaining_timeout(60)
                    import config.settings as cfg
                    if count >= int(getattr(cfg, "MAX_INDEX_INPUT_CHUNKS", 250000)):
                        raise ValueError("Index input exceeds MAX_INDEX_INPUT_CHUNKS")
                    if not isinstance(chunk, EmailChunk) or not chunk.chunk_id.strip() or not chunk.email_id.strip() or chunk.chunk_id in ids:
                        raise ValueError("Index input contains invalid or duplicate chunk identifiers")
                    if len(chunk.chunk_id) > 512 or len(chunk.email_id) > 512:
                        raise ValueError("Index chunk identifiers exceed the 512-character budget")
                    if not chunk.content:
                        raise ValueError("Index input contains an empty chunk")
                    if len(chunk.content) > int(getattr(cfg, "MAX_EMAIL_RECORD_BYTES", 16000000)):
                        raise ValueError("Index chunk exceeds record budget")
                    ids.add(chunk.chunk_id)
                    email_ids.add(chunk.email_id)
                    record = manifests.canonical_bytes(chunk.model_dump()) + b"\n"
                    if len(record) > int(getattr(cfg, "MAX_EMAIL_RECORD_BYTES", 16000000)):
                        raise ValueError("Index chunk exceeds record budget")
                    if output.tell() + len(record) > int(getattr(cfg, "MAX_INDEX_INPUT_BYTES", 2000000000)):
                        raise ValueError("Index spool exceeds MAX_INDEX_INPUT_BYTES")
                    output.write(record)
                    count += 1
                output.flush()
                os.fsync(output.fileno())
            if manifests.configuration_fingerprint() != start_fingerprint:
                raise manifests.IndexCompatibilityError("Index configuration changed during input preparation")
            if not count:
                source_path.unlink(missing_ok=True)
                if replace:
                    raise ValueError("replacement index input must contain chunks")
                set_outcome("unchanged")
                return 0
        except BaseException:
            source_path.unlink(missing_ok=True)
            raise
        add_count("input_chunks", count)
        add_count("input_emails", len(email_ids))
        if not active or not active.get("chunk_count"):
            add_count("new_emails", len(email_ids))
        configuration = manifests.configuration()
        # A complete replace changes membership; it does not force re-encoding.
        # Unknown or incompatible old model identities never enter reuse.
        reusable = bool(active and not force_reembed and
                        active.get("config_fingerprint") == manifests.configuration_fingerprint(configuration) and
                        isinstance(active.get("resolved_model_revision"), str) and active["resolved_model_revision"].strip() and
                        active.get("embedding_dimension"))
        lookup_context = verified_reuse_lookup(active, _collection(active["collection"])) if reusable else nullcontext(None)
        try:
            with lookup_context as reuse:
                if reuse:
                    with measure_stage("compare"):
                        unchanged = reuse.classify(source_path, replace)
                    if unchanged:
                        remaining_timeout(60)
                        manifests.check_compatibility(active)
                        set_outcome("unchanged")
                        add_count("reused_chunks", count)
                        add_count("embedded_chunks", 0)
                        run = current_run()
                        if run:
                            run.progress("index_unchanged", completed_chunks=count, total_chunks=count,
                                         reused_chunks=count, embedded_chunks=0)
                        source_path.unlink(missing_ok=True)
                        return count
                return _create_and_resume(generation, source_path, count, email_ids, configuration,
                    active, batch_size, replace, force_reembed, reuse)
        except BaseException:
            # Once a manifest exists the source is a checkpoint. Before that,
            # the file is an uncommitted private preflight spool.
            if not manifests.generation_path(generation).exists():
                source_path.unlink(missing_ok=True)
            raise


def _create_and_resume(generation, source_path, count, email_ids, configuration,
                       active, batch_size, replace, force_reembed, reuse):
    state = {"generation": generation, "collection": "idx_" + generation,
             "status": "building", "phase": "copy", "created_at": manifests.utc_now(),
             "configuration": configuration, "config_fingerprint": manifests.configuration_fingerprint(configuration),
             "base_generation": (active or {}).get("generation"),
             "base_collection": (active or {}).get("collection"), "replace": bool(replace),
             "batch_size": batch_size, "source_complete": True, "reuse_enabled": reuse is not None,
             "force_reembed": bool(force_reembed), "reused_chunks": 0, "embedded_chunks": 0,
             "source_sha256": _file_digest(source_path), "source_chunk_count": count,
             "updated_email_ids": sorted(email_ids), "source_offset": 0, "copy_offset": 0,
             "indexed_chunks": 0, "copied_chunks": 0, "embedding_dimension": (active or {}).get("embedding_dimension") if not replace or reuse else None,
             "expected_digest_sum": "0",
             "resolved_model_revision": (active or {}).get("resolved_model_revision") if not replace or reuse else None}
    manifests.write_generation(state)
    return _resume_locked(state, reuse)


def _resume_locked(state, reuse=None):
    current = manifests.read_active_manifest(validate_compatibility=False)
    if reuse is None and state.get("reuse_enabled") and (current or {}).get("generation") != state["generation"]:
        if (current or {}).get("generation") != state["base_generation"]:
            raise manifests.IndexBuildConflict("Active index changed while the build was paused; create a new build")
        manifests.check_compatibility(current)
        if current.get("resolved_model_revision") != state.get("resolved_model_revision"):
            raise manifests.IndexCompatibilityError("Base embedding model identity changed")
        with verified_reuse_lookup(current, _collection(current["collection"])) as lookup:
            return _resume_body(state, lookup)
    if reuse is None and current and not state.get("replace", False) and current["generation"] != state["generation"]:
        if current["generation"] != state["base_generation"]:
            raise manifests.IndexBuildConflict("Active index changed while the build was paused; create a new build")
        # Forced encoding applies only to incoming chunks. Retained emails
        # still come from the base and must pass integrity checks on every run.
        metrics = verify_generation(current)
        if any(current.get(key) != value for key, value in metrics.items()):
            raise ValueError("Base generation verification failed; retained copy refused")
    return _resume_body(state, reuse)


def _resume_body(state, reuse):
    backend = _backend()
    manifests.check_compatibility(state)
    current = manifests.read_active_manifest(validate_compatibility=False)
    if (current or {}).get("generation") == state["generation"]:
        set_outcome("unchanged")
        return state.get("source_chunk_count", 0)
    if (current or {}).get("generation") != state["base_generation"]:
        raise manifests.IndexBuildConflict("Active index changed while the build was paused; create a new build")
    path = manifests.source_path(state["generation"])
    with measure_stage("source_verify"):
        if not state.get("source_complete") or _file_digest(path) != state["source_sha256"]:
            raise ValueError("Stored index input is incomplete or changed; create a new build")
    stage = backend._get_client().get_or_create_collection(name=state["collection"],
        metadata={"hnsw:space": "cosine"}, embedding_function=None)
    batch_size = state["batch_size"]
    updated = set(state["updated_email_ids"])
    try:
        if state["status"] == "ready":
            metrics = verify_generation(state)
            if any(metrics[key] != state[key] for key in metrics):
                raise ValueError("Ready generation verification failed")
            with measure_stage("publish"):
                manifests.activate(state, expected_generation=state["base_generation"])
            set_outcome("published")
            return state["source_chunk_count"]
        state["status"] = "building"
        if state["phase"] == "copy":
            if state["base_collection"] and not state["replace"]:
                previous = _collection(state["base_collection"])
                total = previous.count()
                while state["copy_offset"] < total:
                    remaining_timeout(60)
                    with measure_stage("copy"):
                        page = previous.get(include=["documents", "metadatas", "embeddings"],
                                            limit=batch_size, offset=state["copy_offset"])
                    if not page["ids"]:
                        raise ValueError("Previous generation enumeration stopped before its recorded count")
                    keep = [i for i, meta in enumerate(page["metadatas"]) if meta["email_id"] not in updated]
                    if keep:
                        vectors, _ = _validated_vectors([page["embeddings"][i] for i in keep], len(keep), state["embedding_dimension"])
                        _upsert(stage, ids=[page["ids"][i] for i in keep], embeddings=vectors,
                            documents=[page["documents"][i] for i in keep],
                            metadatas=[{**page["metadatas"][i], "index_generation": state["generation"]} for i in keep])
                        state["expected_digest_sum"] = str((int(state["expected_digest_sum"]) + sum(
                            int.from_bytes(_record_digest(page["ids"][i], page["documents"][i], page["metadatas"][i]), "big")
                            for i in keep)) % (1 << 256))
                    state["copied_chunks"] += len(keep)
                    add_count("copied_chunks", len(keep))
                    state["copy_offset"] += len(page["ids"])
                    _save(state)
            state["phase"] = "embed"
            _save(state)
        if state["phase"] == "embed":
            with path.open("rb") as source:
                source.seek(state["source_offset"])
                while True:
                    remaining_timeout(60)
                    batch = []
                    for _ in range(batch_size):
                        line = source.readline()
                        if not line:
                            break
                        try:
                            batch.append(EmailChunk.model_validate(json.loads(line)))
                        except (ValueError, TypeError):
                            raise ValueError("Stored index input is invalid") from None
                    if not batch:
                        break
                    vectors, dimension, embedded = _batch_vectors(batch, state, reuse)
                    _upsert(stage, ids=[c.chunk_id for c in batch], embeddings=vectors,
                        documents=[c.content for c in batch],
                        metadatas=[{**c.metadata, "email_id": c.email_id, "chunk_index": c.chunk_index,
                                    "index_generation": state["generation"]} for c in batch])
                    state["expected_digest_sum"] = str((int(state["expected_digest_sum"]) + sum(
                        int.from_bytes(_record_digest(c.chunk_id, c.content,
                            {**c.metadata, "email_id": c.email_id, "chunk_index": c.chunk_index}), "big")
                        for c in batch)) % (1 << 256))
                    state["embedding_dimension"] = dimension
                    state["embedded_chunks"] = state.get("embedded_chunks", 0) + embedded
                    state["reused_chunks"] = state.get("reused_chunks", 0) + len(batch) - embedded
                    state["source_offset"] = source.tell()
                    state["indexed_chunks"] += len(batch)
                    add_count("processed_chunks", len(batch))
                    _save(state)
            state["phase"] = "verify"
            _save(state)
        if state["phase"] == "verify":
            metrics = verify_generation(state, batch_size=batch_size)
            expected_count = state["copied_chunks"] + state["source_chunk_count"]
            expected_hash = hashlib.sha256(manifests.canonical_bytes([expected_count,
                            f'{int(state["expected_digest_sum"]):064x}'])).hexdigest()
            if metrics["chunk_count"] != expected_count or metrics["corpus_sha256"] != expected_hash:
                raise ValueError("Staging generation is incomplete; activation refused")
            state.update(metrics)
            state["status"] = "ready"
            state["phase"] = "complete"
            # Check cancellation immediately before publication. No await/model
            # call occurs between this check and the atomic pointer replacement.
            remaining_timeout(60)
            with measure_stage("publish"):
                manifests.activate(state, expected_generation=state["base_generation"])
            set_outcome("published")
            run = current_run()
            if run:
                run.checkpoint({"generation": state["generation"], "generation_id": state["generation"], "kind": "index", "safe": True, "phase": "complete",
                                "indexed_chunks": state["indexed_chunks"], "resume_supported": True})
            return state["source_chunk_count"]
        raise ValueError("Stored generation phase is not resumable")
    except BaseException as exc:
        # Persist a safe resume point, never exception text or body fragments.
        # Do not mutate a ready manifest after its hash may have been published.
        if state.get("status") != "ready":
            state["status"] = "paused"
            state["last_error_type"] = type(exc).__name__
            manifests.write_generation(state)
        raise


@index_operation
def resume_index_generation(generation: str) -> int:
    with _backend()._corpus_write():
        return _resume_locked(manifests.read_generation(generation))


def rollback_generation(generation: str) -> dict:
    with _backend()._corpus_write():
        state = manifests.read_generation(generation)
        manifests.check_compatibility(state)
        if state.get("status") != "ready":
            raise ValueError("Only ready generations may be restored")
        metrics = verify_generation(state)
        if any(state.get(key) != value for key, value in metrics.items()):
            raise ValueError("Generation verification failed; rollback refused")
        current = manifests.read_active_manifest(validate_compatibility=False)
        manifests.activate(state, expected_generation=(current or {}).get("generation"))
        return {"generation": generation, **metrics}
