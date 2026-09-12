"""Durable index generations and compatibility metadata; no model/storage SDK."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import config.settings as cfg


class IndexCompatibilityError(ValueError):
    pass


class IndexBuildConflict(ValueError):
    pass


def canonical_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def root_path() -> Path:
    key = hashlib.sha256(cfg.CHROMA_COLLECTION.encode("utf-8")).hexdigest()[:16]
    return Path(cfg.CHROMA_PERSIST_DIR) / "index_manifests" / key


def generation_path(generation: str) -> Path:
    if not isinstance(generation, str) or not re.fullmatch(r"[a-f0-9]{32}", generation):
        raise ValueError("Invalid index generation identifier")
    return root_path() / "generations" / (generation + ".json")


def source_path(generation: str) -> Path:
    generation_path(generation)  # validate before constructing any path
    return root_path() / "sources" / (generation + ".chunks.jsonl")


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".index-", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            output.write(canonical_bytes(value))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        temporary = None
        if os.name != "nt":
            descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def configuration() -> dict:
    project = Path(__file__).resolve().parent.parent
    files = ("core/cleaner.py", "core/html_tables.py", "core/chunker.py")
    return {"schema_version": 1, "embedding_model": cfg.EMBEDDING_MODEL,
            "embedding_revision": getattr(cfg, "EMBEDDING_MODEL_REVISION", None) or None,
            "embedding_dimension": getattr(cfg, "EMBEDDING_DIMENSION", None) or None,
            "normalize_embeddings": True,
            "chunk_size": cfg.CHUNK_SIZE, "chunk_overlap": cfg.CHUNK_OVERLAP,
            "min_chunk_size": cfg.MIN_CHUNK_SIZE,
            "source_versions": {name: hashlib.sha256((project / name).read_bytes()).hexdigest() for name in files}}


def configuration_fingerprint(configuration_value=None) -> str:
    return hashlib.sha256(canonical_bytes(configuration_value or configuration())).hexdigest()


def check_compatibility(manifest: dict) -> None:
    if manifest.get("config_fingerprint") != configuration_fingerprint():
        raise IndexCompatibilityError("Index configuration/model or cleaner/chunker version changed; rebuild a new generation before searching")
    if manifest.get("configuration", {}).get("normalize_embeddings") is not True:
        raise IndexCompatibilityError("Index normalization is incompatible; rebuild before searching")


def read_generation(generation: str) -> dict:
    try:
        value = json.loads(generation_path(generation).read_text(encoding="utf-8"))
    except (ValueError, UnicodeError):
        raise IndexCompatibilityError("Index generation manifest is invalid") from None
    if not isinstance(value, dict) or value.get("generation") != generation:
        raise IndexCompatibilityError("Index generation manifest identity is invalid")
    return value


def write_generation(manifest: dict) -> None:
    atomic_json(generation_path(manifest["generation"]), manifest)


def read_active_manifest(validate_compatibility: bool = True) -> dict | None:
    path = root_path() / "active.json"
    if not path.exists():
        return None
    try:
        pointer = json.loads(path.read_text(encoding="utf-8"))
        value = read_generation(pointer["generation"])
        expected = hashlib.sha256(canonical_bytes(value)).hexdigest()
        if value.get("status") != "ready" or pointer.get("manifest_sha256") != expected:
            raise IndexCompatibilityError("Active index manifest integrity check failed")
    except (ValueError, TypeError, KeyError, UnicodeError, FileNotFoundError):
        raise IndexCompatibilityError("Active index manifest is invalid; inspect generations before recovery") from None
    if validate_compatibility:
        check_compatibility(value)
    return value


def activate(manifest: dict, *, expected_generation: str | None) -> None:
    """Compare then atomically publish. Caller must hold the index writer lock."""
    current = read_active_manifest(validate_compatibility=False)
    if (current or {}).get("generation") != expected_generation:
        raise IndexBuildConflict("Active index changed since this build started; rebuild against the current generation")
    if manifest.get("status") != "ready":
        raise ValueError("Only complete index generations can be activated")
    check_compatibility(manifest)
    # Ready manifests are immutable. Failed pointer replacement leaves the old
    # pointer plus a complete, inspectable generation available for recovery.
    write_generation(manifest)
    atomic_json(root_path() / "active.json", {"generation": manifest["generation"],
                "manifest_sha256": hashlib.sha256(canonical_bytes(manifest)).hexdigest()})


def list_generations() -> list[dict]:
    active = read_active_manifest(validate_compatibility=False)
    result = []
    for path in sorted((root_path() / "generations").glob("*.json")):
        value = read_generation(path.stem)
        result.append({key: value.get(key) for key in ("generation", "collection", "status", "created_at",
                       "chunk_count", "email_count", "config_fingerprint", "corpus_sha256", "indexed_chunks")}
                      | {"active": value["generation"] == (active or {}).get("generation")})
    return sorted(result, key=lambda item: item.get("created_at") or "", reverse=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
