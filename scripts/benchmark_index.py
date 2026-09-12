"""Offline readiness check; explicit, isolated real indexing benchmark."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = ("torch", "sentence-transformers", "chromadb", "pydantic", "python-dotenv", "ijson")


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be zero or positive")
    return number


def encoder_batch(value):
    number = positive(value)
    if number > 512:
        raise argparse.ArgumentTypeError("must be between 1 and 512")
    return number


def failure(exc):
    """Exceptions can contain email/provider text; persist only their type."""
    return {"error_type": type(exc).__name__,
            "error_code": "cancelled" if type(exc).__name__ in {"KeyboardInterrupt", "RunCancelled"} else "failed"}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def readiness(model):
    """Inspect distribution metadata and paths only; never import model libraries."""
    versions, missing = {}, []
    for package in DEPENDENCIES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            missing.append(package)
    cache = Path(os.environ.get("HF_HUB_CACHE", str(Path(os.environ.get(
        "HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub")))
    local = Path(model).is_dir()
    snapshots = cache / ("models--" + model.replace("/", "--")) / "snapshots"
    present = local or (snapshots.is_dir() and any(snapshots.iterdir()))
    return {"python": sys.executable, "python_version": platform.python_version(),
            "platform": platform.platform(), "dependencies": versions, "missing_dependencies": missing,
            "model": model, "model_files_detected": present,
            "model_readiness": "unverified: file presence is not proof of complete weights or compatible runtime",
            "ready_to_attempt": not missing and present,
            "action": ("Use an existing interpreter with these dependencies: " + ", ".join(missing)) if missing else
                      ("Provide an already cached model or --model LOCAL_DIRECTORY; no download was attempted." if not present else
                       "Run --run --input EMAILS.json to test the actual runtime offline.")}


def freeze_input(source, target, max_bytes):
    digest, size = hashlib.sha256(), 0
    with Path(source).open("rb") as reader, Path(target).open("xb") as writer:
        while block := reader.read(1024 * 1024):
            size += len(block)
            if size > max_bytes:
                raise ValueError("Input exceeds --max-input-mb; choose a bounded sample or increase the explicit limit")
            writer.write(block)
            digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": size}


def worker_environment(job):
    env = os.environ.copy()
    env.update({"CHROMA_PERSIST_DIR": job["index_dir"], "CHROMA_COLLECTION": "benchmark_emails",
                "EMAIL_DATA_PATH": job["input"], "EMBEDDING_MODEL": job["model"],
                "EMBEDDING_DEVICE": job["device"], "EMBEDDING_MODEL_REVISION": job["revision"] or "",
                "EMBEDDING_BATCH_SIZE": str(job["embedding_batch_size"]),
                "EMBEDDING_CPU_THREADS": str(job["cpu_threads"]),
                "MAX_INDEX_INPUT_EMAILS": str(job["max_emails"]),
                "MAX_INDEX_INPUT_CHUNKS": str(job["max_chunks"]),
                "ANONYMIZED_TELEMETRY": "False", "HF_HUB_DISABLE_TELEMETRY": "1"})
    if not job["allow_download"]:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    return env


def execute_worker(job):
    # Also override already-imported settings: no path can fall back to the production index.
    os.environ.update(worker_environment(job))
    sys.path.insert(0, str(ROOT))
    import config.settings as cfg
    cfg.CHROMA_PERSIST_DIR = job["index_dir"]
    cfg.CHROMA_COLLECTION = "benchmark_emails"
    cfg.EMAIL_DATA_PATH = job["input"]
    from core import embedder
    from core.ingestion import prepare_email_chunks
    from core.index_metrics import collect_index_metrics
    from core.index_manifest import configuration

    original_embed = embedder.embed_texts
    observations = {"embedding_calls": 0, "embedding_texts": 0}

    def observed_embed(texts):
        observations["embedding_calls"] += 1
        observations["embedding_texts"] += len(texts)
        return original_embed(texts)

    embedder.embed_texts = observed_embed
    phases = []
    result = {"status": "running", "pid": os.getpid(), "mode": job["mode"],
              "force_reembed": job["force_reembed"], "configuration": configuration(), "phases": phases}
    try:
        for phase in ("initial", "unchanged_repeat"):
            observations.update(embedding_calls=0, embedding_texts=0)
            emails = None
            metrics = None
            started = time.perf_counter()
            phase_result = {"phase": phase, "status": "running",
                            "model_process_state": "cold" if phase == "initial" else "warm_from_initial",
                            "input_sha256": job["input_sha256"]}
            phases.append(phase_result)
            print(f"[{job['mode']}] {phase}: started", file=sys.stderr, flush=True)
            if job.get("result_path"):
                atomic_json(job["result_path"], result)
            try:
                with collect_index_metrics() as metrics:
                    try:
                        emails, chunks = prepare_email_chunks(job["input"])
                        phase_result.update(input_emails=len(emails), input_chunks=len(chunks))
                        count = embedder.index_chunks(chunks, batch_size=job["write_batch_size"],
                                                      force_reembed=job["force_reembed"])
                        stats = embedder.get_collection_stats()
                        phase_result.update(index_return_count=count, published_chunks=stats["chunk_count"])
                        if stats["chunk_count"] != len(chunks):
                            raise RuntimeError("Published chunk count differs from the frozen input")
                    finally:
                        if emails is not None:
                            emails.close()
                phase_result["status"] = "ok"
            except BaseException as exc:
                phase_result.update(status="error", **failure(exc))
                result.update(status="error", **failure(exc))
            finally:
                phase_result.update(observations)
                phase_result["total_seconds"] = time.perf_counter() - started
                if metrics is not None:
                    phase_result["metrics"] = metrics.to_dict()
                if job.get("result_path"):
                    atomic_json(job["result_path"], result)
                print(f"[{job['mode']}] {phase}: {phase_result['status']} "
                      f"({phase_result['total_seconds']:.2f}s, embedding calls={observations['embedding_calls']})",
                      file=sys.stderr, flush=True)
            if result["status"] == "error":
                return result
        result["status"] = "ok"
        return result
    finally:
        embedder.embed_texts = original_embed


def run_benchmark(args, check):
    report = {"schema_version": 1, "status": "running", "environment": check,
              "baseline": "same-code force_reembed algorithm baseline; not a historical version",
              "model_download_allowed": args.allow_download,
              "model_file_state": "download_allowed_unmeasured" if args.allow_download else "cached_only_offline",
              "model_load_timing": "Model initialization includes any permitted download; download duration is not separated.",
              "runtime_options": {"model": args.model, "revision": args.revision, "device": args.device,
                                  "embedding_batch_size": args.embedding_batch_size,
                                  "cpu_threads": args.cpu_threads, "write_batch_size": args.write_batch_size,
                                  "max_emails": args.max_emails, "max_chunks": args.max_chunks,
                                  "max_input_mb": args.max_input_mb,
                                  "timeout_seconds_per_process": args.timeout_seconds}, "runs": []}
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="email-index-benchmark-") as scratch:
            scratch = Path(scratch)
            frozen = scratch / "frozen.json"
            report["input"] = freeze_input(args.input, frozen, args.max_input_mb * 1024 * 1024)
            for mode, forced in (("forced_baseline", True), ("incremental", False)):
                job = {"mode": mode, "force_reembed": forced, "input": str(frozen),
                       "input_sha256": report["input"]["sha256"], "index_dir": str(scratch / mode),
                       "model": args.model, "device": args.device, "revision": args.revision,
                       "embedding_batch_size": args.embedding_batch_size, "cpu_threads": args.cpu_threads,
                       "write_batch_size": args.write_batch_size, "max_emails": args.max_emails,
                       "max_chunks": args.max_chunks, "allow_download": args.allow_download}
                job_path, result_path = scratch / (mode + "-job.json"), scratch / (mode + "-result.json")
                job["result_path"] = str(result_path)
                atomic_json(job_path, job)
                try:
                    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(job_path),
                                    "--output", str(result_path)], cwd=ROOT, env=worker_environment(job),
                                   check=False, timeout=args.timeout_seconds)
                except (subprocess.TimeoutExpired, KeyboardInterrupt):
                    if result_path.exists():
                        report["runs"].append(json.loads(result_path.read_text(encoding="utf-8")))
                    raise
                if not result_path.exists():
                    raise RuntimeError("Benchmark worker exited without a result")
                result = json.loads(result_path.read_text(encoding="utf-8"))
                report["runs"].append(result)
                if result["status"] != "ok":
                    raise RuntimeError("Benchmark worker failed")
            counts = {p["input_chunks"] for r in report["runs"] for p in r["phases"]}
            if len(counts) != 1:
                raise RuntimeError("Input chunk counts differ between scenarios")
            report["status"] = "ok"
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status="error", **failure(exc))
    report["orchestration_seconds"] = time.perf_counter() - started
    atomic_json(args.output, report)
    return 0 if report["status"] == "ok" else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Default: metadata-only offline readiness check")
    mode.add_argument("--run", action="store_true", help="Run real indexing in private temporary directories")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "work" / "index-benchmark.json")
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"))
    parser.add_argument("--device", default=os.getenv("EMBEDDING_DEVICE", "cpu"))
    parser.add_argument("--revision", default=os.getenv("EMBEDDING_MODEL_REVISION"))
    parser.add_argument("--embedding-batch-size", type=encoder_batch, default=32)
    parser.add_argument("--cpu-threads", type=nonnegative, default=0)
    parser.add_argument("--write-batch-size", type=positive, default=64)
    parser.add_argument("--max-input-mb", type=positive, default=32)
    parser.add_argument("--max-emails", type=positive, default=10000)
    parser.add_argument("--max-chunks", type=positive, default=10000)
    parser.add_argument("--timeout-seconds", type=positive, default=1800)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        try:
            result = execute_worker(json.loads(args.worker.read_text(encoding="utf-8")))
        except (Exception, KeyboardInterrupt) as exc:
            result = {"status": "error", **failure(exc)}
        atomic_json(args.output, result)
        return 0 if result["status"] == "ok" else 2
    check = readiness(args.model)
    if not args.run:
        print(json.dumps(check, ensure_ascii=False, indent=2))
        return 0 if check["ready_to_attempt"] else 2
    if args.input is None:
        parser.error("--run requires --input; production EMAIL_DATA_PATH is never selected automatically")
    if args.output.resolve() == args.input.resolve():
        parser.error("--output must differ from --input")
    if check["missing_dependencies"]:
        atomic_json(args.output, {"status": "not_ready", "environment": check})
        print(check["action"], file=sys.stderr)
        return 2
    status = run_benchmark(args, check)
    print(f"Benchmark report: {args.output.resolve()}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
