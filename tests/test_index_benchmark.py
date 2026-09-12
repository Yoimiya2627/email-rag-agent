"""Benchmark orchestration tests need no model, Chroma runtime or network."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_index.py"
spec = importlib.util.spec_from_file_location("benchmark_index", SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_check_does_not_import_heavy_modules_or_settings():
    source = '''
import sys, runpy
class Block:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'sentence_transformers', 'chromadb', 'config', 'huggingface_hub'}:
            raise AssertionError('forbidden import: ' + fullname)
sys.meta_path.insert(0, Block())
sys.argv = [sys.argv[1], '--check']
runpy.run_path(sys.argv[0], run_name='__main__')
'''
    result = subprocess.run([sys.executable, "-c", source, str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode in (0, 2), result.stderr
    assert "missing_dependencies" in json.loads(result.stdout)


def test_isolated_process_jobs_share_frozen_input(tmp_path, monkeypatch):
    source = tmp_path / "emails.json"
    source.write_text('[{"body":"a bounded sample"}]', encoding="utf-8")
    original = source.read_bytes()
    output = tmp_path / "report.json"
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "production"))
    monkeypatch.setenv("EMAIL_DATA_PATH", str(tmp_path / "production.json"))
    monkeypatch.setattr(benchmark, "readiness", lambda model: {"missing_dependencies": []})
    calls = []

    def fake_process(command, **options):
        job = json.loads(Path(command[3]).read_text(encoding="utf-8"))
        assert command[:2] == [sys.executable, str(SCRIPT)]
        assert Path(job["input"]).read_bytes() == original
        assert options["env"]["CHROMA_PERSIST_DIR"] == job["index_dir"]
        assert options["env"]["EMAIL_DATA_PATH"] == job["input"]
        assert options["env"]["HF_HUB_OFFLINE"] == "1"
        assert options["env"]["TRANSFORMERS_OFFLINE"] == "1"
        assert options["env"]["EMBEDDING_BATCH_SIZE"] == "7"
        assert options["env"]["EMBEDDING_CPU_THREADS"] == "2"
        assert not Path(job["index_dir"]).exists()
        calls.append(job)
        benchmark.atomic_json(command[5], {"status": "ok", "mode": job["mode"],
            "phases": [{"input_chunks": 3, "input_sha256": job["input_sha256"]}]})

    monkeypatch.setattr(benchmark.subprocess, "run", fake_process)
    assert benchmark.main(["--run", "--input", str(source), "--output", str(output),
                           "--embedding-batch-size", "7", "--cpu-threads", "2"]) == 0
    assert len(calls) == 2
    assert calls[0]["index_dir"] != calls[1]["index_dir"]
    assert calls[0]["force_reembed"] is True and calls[1]["force_reembed"] is False
    assert calls[0]["input_sha256"] == calls[1]["input_sha256"]
    assert source.read_bytes() == original
    assert not Path(calls[0]["input"]).exists()
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "ok"


def test_input_limit_and_existing_source_are_safe(tmp_path):
    source, frozen = tmp_path / "source.json", tmp_path / "frozen.json"
    source.write_bytes(b"12345")
    with pytest.raises(ValueError, match="exceeds"):
        benchmark.freeze_input(source, frozen, 4)
    assert source.read_bytes() == b"12345"


@pytest.mark.parametrize("arguments", [
    ["--embedding-batch-size", "0"], ["--cpu-threads", "-1"], ["--write-batch-size", "0"],
    ["--max-input-mb", "0"], ["--timeout-seconds", "0"],
    ["--embedding-batch-size", "513"],
])
def test_reject_invalid_limits(arguments):
    with pytest.raises(SystemExit) as exc:
        benchmark.main(arguments)
    assert exc.value.code == 2


def test_worker_failure_is_reported_atomically(tmp_path, monkeypatch):
    source, output = tmp_path / "source.json", tmp_path / "report.json"
    source.write_text("[]")
    monkeypatch.setattr(benchmark, "readiness", lambda model: {"missing_dependencies": []})

    def fake_failure(command, **kwargs):
        benchmark.atomic_json(command[5], {"status": "error", "error_type": "OSError", "error_code": "failed"})

    monkeypatch.setattr(benchmark.subprocess, "run", fake_failure)
    assert benchmark.main(["--run", "--input", str(source), "--output", str(output)]) == 2
    report = json.loads(output.read_text())
    assert report["status"] == "error"
    assert report["runs"][0]["error_type"] == "OSError"
    assert report["error_type"] == "RuntimeError"
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("force", [False, True])
@pytest.mark.parametrize("error_phase,error_type", [(None, ValueError), ("unchanged_repeat", ValueError),
                                                  ("initial", ValueError), ("unchanged_repeat", KeyboardInterrupt)])
def test_worker_counts_real_calls_and_restores_observer(tmp_path, monkeypatch, force, error_phase, error_type):
    cfg = ModuleType("config.settings")
    cfg.CHROMA_PERSIST_DIR = "production"
    config = ModuleType("config")
    config.settings = cfg
    core = ModuleType("core")
    backend = ModuleType("core.embedder")
    backend.embed_texts = lambda texts: [[1.0] for _ in texts]
    original = backend.embed_texts
    invocations, plans = [], []

    def index_chunks(chunks, *, batch_size, force_reembed):
        assert cfg.CHROMA_PERSIST_DIR == str(tmp_path / "private")
        assert force_reembed is force
        if not invocations or force_reembed:
            backend.embed_texts(["first", "second"])
        invocations.append(force_reembed)
        if (error_phase == "initial" and len(invocations) == 1) or (error_phase == "unchanged_repeat" and len(invocations) == 2):
            raise error_type("secret email body from provider")
        return len(chunks)

    backend.index_chunks = index_chunks
    backend.get_collection_stats = lambda: {"chunk_count": 2}
    core.embedder = backend
    ingestion = ModuleType("core.ingestion")

    class Plan(list):
        closed = False

        def close(self):
            self.closed = True

    def prepare(path):
        plan = Plan([1])
        plans.append(plan)
        return plan, [1, 2]

    ingestion.prepare_email_chunks = prepare
    metrics = ModuleType("core.index_metrics")

    @contextmanager
    def collect():
        state = {"stages_seconds": {"embedding": 0.01}, "counts": {}, "outcome": "pending"}
        try:
            yield SimpleNamespace(to_dict=lambda: state.copy())
        except BaseException as exc:
            state["outcome"] = "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
            raise

    metrics.collect_index_metrics = collect
    manifest = ModuleType("core.index_manifest")
    manifest.configuration = lambda: {"model": "fake"}
    for name, module in {"config": config, "config.settings": cfg, "core": core,
                         "core.embedder": backend, "core.ingestion": ingestion,
                         "core.index_metrics": metrics, "core.index_manifest": manifest}.items():
        monkeypatch.setitem(sys.modules, name, module)
    # execute_worker modifies its environment; isolate that change from the test runner.
    monkeypatch.setattr(benchmark.os, "environ", dict(benchmark.os.environ))
    monkeypatch.setattr(benchmark.sys, "path", list(sys.path))
    job = {"index_dir": str(tmp_path / "private"), "input": str(tmp_path / "frozen.json"),
           "model": "fake", "device": "cpu", "revision": None, "embedding_batch_size": 2,
           "cpu_threads": 0, "max_emails": 10, "max_chunks": 10, "allow_download": False,
           "input_sha256": "frozen-hash", "write_batch_size": 64,
           "mode": "test", "force_reembed": force}
    result = benchmark.execute_worker(job)
    expected_calls = [1] if error_phase == "initial" else [1, int(force)]
    assert [phase["embedding_calls"] for phase in result["phases"]] == expected_calls
    assert [phase["embedding_texts"] for phase in result["phases"]] == [2 * n for n in expected_calls]
    if error_phase:
        assert result["status"] == "error"
        assert result["phases"][-1]["metrics"]["outcome"] == ("cancelled" if error_type is KeyboardInterrupt else "failed")
        assert result["phases"][-1]["error_type"] == error_type.__name__
        assert "secret" not in json.dumps(result)
    assert all(plan.closed for plan in plans)
    assert backend.embed_texts is original


def test_orchestrator_sanitizes_error_message(tmp_path, monkeypatch):
    source, output = tmp_path / "source.json", tmp_path / "report.json"
    source.write_text("[]")
    monkeypatch.setattr(benchmark, "readiness", lambda model: {"missing_dependencies": []})

    def fail_process(*args, **kwargs):
        raise OSError("secret email text must never enter report")

    monkeypatch.setattr(benchmark.subprocess, "run", fail_process)
    assert benchmark.main(["--run", "--input", str(source), "--output", str(output)]) == 2
    assert "secret" not in output.read_text()
    assert json.loads(output.read_text())["error_type"] == "OSError"


def test_timeout_keeps_last_worker_checkpoint(tmp_path, monkeypatch):
    source, output = tmp_path / "source.json", tmp_path / "report.json"
    source.write_text("[]")
    monkeypatch.setattr(benchmark, "readiness", lambda model: {"missing_dependencies": []})

    def timeout(command, **kwargs):
        benchmark.atomic_json(command[5], {"status": "running", "phases": [
            {"phase": "initial", "status": "ok", "embedding_calls": 1},
            {"phase": "unchanged_repeat", "status": "running"}]})
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(benchmark.subprocess, "run", timeout)
    assert benchmark.main(["--run", "--input", str(source), "--output", str(output)]) == 2
    report = json.loads(output.read_text())
    assert report["error_type"] == "TimeoutExpired"
    assert report["runs"][0]["phases"][0]["embedding_calls"] == 1
    assert report["runs"][0]["phases"][1]["status"] == "running"
