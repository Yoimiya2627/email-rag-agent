"""Evaluate retrieval context_recall against manually labeled gold chunks.

RAGAS-style metrics in ``run_ragas_eval.py`` use an LLM judge.  This script
adds a deterministic retrieval metric: for each question, compare retrieved
``chunk_id`` values with human-labeled ``gold_chunk_ids``.

Gold file shape:
[
  {
    "id": "gold_001",
    "question": "...",
    "ground_truth": "...",
    "source_email_ids": ["..."],
    "gold_chunk_ids": ["email_1_chunk_0"],
    "notes": "optional"
  }
]
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
from importlib import metadata as package_metadata
import json
import logging
import sys
from pathlib import Path
from typing import Callable, Iterable

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.eval_output import EvaluationOutput, OutputOwnershipError
from models.schemas import SearchResult
from agents.runtime import ContextBudgetExceeded, RunCancelled, RunDeadlineExceeded
from core.model_clients import ModelBudgetExceeded
from scripts.run_ragas_eval import TESTSET_PATH, VERSION_FLAGS, apply_flags

GOLD_PATH = Path(__file__).parent.parent / "data" / "gold_chunks.json"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "eval_results" / "context_recall.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_CONTROL_ERRORS = (
    RunCancelled, RunDeadlineExceeded, ContextBudgetExceeded, ModelBudgetExceeded,
    KeyboardInterrupt, SystemExit, GeneratorExit,
)


def _contains_control_error(exc: BaseException) -> bool:
    """Wrapped cancellation/budget failures must still stop the evaluation."""
    from core.model_errors import find_permanent_model_error
    if find_permanent_model_error(exc) is not None:
        return True
    seen = set()
    pending = [exc]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, _CONTROL_ERRORS):
            return True
        pending.extend(error for error in (current.__cause__, current.__context__) if error is not None)
    return False


def _reference_time(value: datetime | None = None) -> datetime:
    if value is None:
        from core.pipeline import _now

        value = _now()
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("reference_time must be a timezone-aware datetime")
    return value


def _parse_reference_time(raw: str) -> datetime:
    try:
        if isinstance(raw, str) and raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"  # Python 3.10 does not parse the UTC Z suffix.
        return _reference_time(datetime.fromisoformat(raw))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("reference time must be an ISO datetime with a timezone") from exc


def _run_metadata(reference_time: datetime, gold_path: str | Path) -> dict:
    """Best-effort reproducibility metadata, without credentials or env contents."""
    metadata = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "reference_time": reference_time.isoformat(),
    }
    try:
        metadata["gold_sha256"] = hashlib.sha256(Path(gold_path).read_bytes()).hexdigest()
    except OSError as exc:
        metadata["gold_hash_error_type"] = type(exc).__name__
    try:
        import config.settings as cfg

        metadata["settings"] = {
            name: getattr(cfg, name, None)
            for name in (
                "DEEPSEEK_MODEL", "MODEL_REVISION", "EMBEDDING_MODEL", "EMBEDDING_MODEL_REVISION",
                "CROSS_ENCODER_MODEL", "CROSS_ENCODER_DEVICE", "CROSS_ENCODER_MAX_LENGTH",
                "TOP_K", "RERANK_TOP_N", "RERANK_INPUT_CHAR_LIMIT",
                "RETRIEVAL_TIMEZONE", "RETRIEVAL_TIMEZONE_OFFSET_HOURS",
            )
        }
    except Exception as exc:
        metadata["settings_error_type"] = type(exc).__name__
    packages = {}
    for name in ("sentence-transformers", "transformers", "torch", "chromadb"):
        try:
            packages[name] = package_metadata.version(name)
        except Exception:
            packages[name] = None
    metadata["packages"] = packages
    return metadata


def _load_json(path: str | Path) -> list:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON array")
    return payload


def _dump_json(writer: EvaluationOutput, payload: object) -> None:
    writer.write(payload)


def build_gold_template(testset: list, limit: int | None = None) -> list[dict]:
    items = testset[:limit] if limit else testset
    template = []
    for idx, item in enumerate(items, start=1):
        template.append(
            {
                "id": f"gold_{idx:03d}",
                "question": item.get("question", ""),
                "ground_truth": item.get("ground_truth", ""),
                "source_email_ids": item.get("email_ids", []),
                "gold_chunk_ids": [],
                "notes": "Fill gold_chunk_ids after inspecting retrieved/source chunks.",
            }
        )
    return template


def score_context_recall_case(case: dict, retrieved: Iterable[SearchResult]) -> dict:
    gold_chunk_ids = [str(item) for item in case.get("gold_chunk_ids", []) if str(item)]
    retrieved_chunk_ids = [item.chunk_id for item in retrieved]
    retrieved_set = set(retrieved_chunk_ids)
    hit_chunk_ids = [chunk_id for chunk_id in gold_chunk_ids if chunk_id in retrieved_set]
    gold_count = len(gold_chunk_ids)
    hit_count = len(hit_chunk_ids)
    context_recall = round(hit_count / gold_count, 4) if gold_count else 0.0

    return {
        "id": case.get("id", ""),
        "question": case.get("question", ""),
        "gold_chunk_ids": gold_chunk_ids,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "hit_chunk_ids": hit_chunk_ids,
        "hit_count": hit_count,
        "gold_count": gold_count,
        "context_recall": context_recall,
        "has_any_hit": hit_count > 0,
        "has_perfect_recall": gold_count > 0 and hit_count == gold_count,
    }


def evaluate_context_recall_cases(
    cases: list[dict],
    retrieve_fn: Callable[[str], list[SearchResult]],
    limit: int | None = None,
    *,
    diagnostics_fn: Callable[[], dict] | None = None,
    progress_fn: Callable[[list[dict]], None] | None = None,
) -> dict:
    selected = cases[:limit] if limit else cases
    records = []
    for case in selected:
        question = case["question"]
        try:
            retrieved = list(retrieve_fn(question))
        except Exception as exc:
            if _contains_control_error(exc):
                raise
            record = score_context_recall_case(case, [])
            record.update(
                status="error",
                error_type=type(exc).__name__,
                cause_type=type(exc.__cause__).__name__ if exc.__cause__ is not None else None,
            )
            logger.warning("context recall case %s failed (%s)", case.get("id", ""), type(exc).__name__)
        else:
            record = score_context_recall_case(case, retrieved)
            record["status"] = "ok" if retrieved else "empty"
        if diagnostics_fn is not None:
            record["diagnostics"] = copy.deepcopy(diagnostics_fn())
        records.append(record)
        if progress_fn is not None:
            progress_fn(copy.deepcopy(records))

    n = len(records)
    mean_context_recall = (
        round(sum(record["context_recall"] for record in records) / n, 4) if n else 0.0
    )
    chunk_hit_rate = (
        round(sum(1 for record in records if record["has_any_hit"]) / n, 4) if n else 0.0
    )
    perfect_recall_rate = (
        round(sum(1 for record in records if record["has_perfect_recall"]) / n, 4)
        if n
        else 0.0
    )
    return {
        "summary": {
            "n": n,
            "mean_context_recall": mean_context_recall,
            "chunk_hit_rate": chunk_hit_rate,
            "perfect_recall_rate": perfect_recall_rate,
            "ok_count": sum(record["status"] == "ok" for record in records),
            "empty_count": sum(record["status"] == "empty" for record in records),
            "error_count": sum(record["status"] == "error" for record in records),
        },
        "records": records,
    }


def _parse_versions(raw: str) -> list[str]:
    versions = [item.strip().upper() for item in raw.split(",") if item.strip()]
    if not versions or len(set(versions)) != len(versions):
        raise ValueError('versions must be non-empty and unique')
    unknown = [version for version in versions if version not in VERSION_FLAGS]
    if unknown:
        raise ValueError(f"Unknown versions: {', '.join(unknown)}")
    return versions


def evaluate_version(
    version: str, cases: list[dict], limit: int | None = None, *, reference_time: datetime | None = None,
    progress_fn: Callable[[list[dict]], None] | None = None,
) -> dict:
    fixed_time = _reference_time(reference_time)
    apply_flags(VERSION_FLAGS[version])
    from core.pipeline import retrieve
    from core.reranker import reset_circuit_breaker

    reset_circuit_breaker()
    diagnostics = {}

    def retrieve_with_diagnostics(query: str):
        diagnostics.clear()
        rows = retrieve(query, diagnostics=diagnostics, reference_time=fixed_time)
        if VERSION_FLAGS[version]['ENABLE_RERANKER']:
            from scripts.paired_acceptance import validate_rerank_evidence
            validate_rerank_evidence(diagnostics.get('reranker'),
                expected_backend=VERSION_FLAGS[version]['RERANKER_BACKEND'],
                candidate_count=diagnostics.get('rerank_candidate_count'))
        return rows

    report = evaluate_context_recall_cases(
        cases, retrieve_fn=retrieve_with_diagnostics, limit=limit, diagnostics_fn=lambda: diagnostics,
        progress_fn=progress_fn,
    )
    return {
        "version": version,
        "flags": VERSION_FLAGS[version],
        "reference_time": fixed_time.isoformat(),
        **report,
    }


def _print_table(results: list[dict]) -> None:
    print(f"{'Version':<8} {'n':>4} {'context_recall':>16} {'hit_rate':>10} {'perfect':>10} {'empty':>6} {'errors':>6}")
    print("-" * 76)
    for result in results:
        summary = result["summary"]
        print(
            f"{result['version']:<8} {summary['n']:>4} "
            f"{summary['mean_context_recall']:>16.4f} "
            f"{summary['chunk_hit_rate']:>10.4f} "
            f"{summary['perfect_recall_rate']:>10.4f} "
            f"{summary['empty_count']:>6} {summary['error_count']:>6}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default=str(GOLD_PATH), help="Gold chunk JSON path")
    parser.add_argument("--versions", default="V2,V7", help="Comma-separated versions to evaluate")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument(
        "--reference-time", type=_parse_reference_time, default=None,
        help="Freeze relative dates at this ISO datetime with timezone; defaults to one shared run-start time",
    )
    parser.add_argument(
        "--init-template",
        action="store_true",
        help="Create a gold chunk template from the RAGAS testset and exit",
    )
    parser.add_argument("--template-from", default=str(TESTSET_PATH))
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    destination = Path(args.gold if args.init_template else args.output)
    if destination.exists():
        parser.error('output already exists; choose a new path to preserve evidence')
    if args.init_template:
        template = build_gold_template(_load_json(args.template_from), limit=args.limit)
        try:
            with EvaluationOutput(args.gold) as writer:
                _dump_json(writer, template)
        except FileExistsError:
            parser.error('output or its ownership lock already exists; choose a new path')
        logger.info("wrote gold chunk template -> %s", args.gold)
        return 0

    cases = _load_json(args.gold)
    from scripts.paired_acceptance import validate_gold_cases
    try:
        validate_gold_cases(cases)
    except ValueError as exc:
        parser.error(str(exc))
    versions = _parse_versions(args.versions)
    try:
        with EvaluationOutput(destination) as writer:
            try:
                return _run_owned(args, cases, versions, writer)
            except OutputOwnershipError:
                return 2
            except BaseException as exc:
                try:
                    if writer.updates == 0:
                        writer.write({'complete': False, 'batch_status': 'aborted',
                                      'abort': {'error_type': type(exc).__name__}})
                except OutputOwnershipError:
                    pass
                if isinstance(exc, Exception):
                    return 2
                raise
    except FileExistsError:
        parser.error('output or its ownership lock already exists; choose a new path')
    except OSError:
        return 2


def _run_owned(args, cases, versions, writer):
    reference_time = _reference_time(args.reference_time)
    run_metadata = _run_metadata(reference_time, args.gold)
    from scripts.paired_acceptance import _source_hashes
    from core.model_errors import find_permanent_model_error
    root = Path(__file__).resolve().parents[1]
    source_hashes = _source_hashes(root)
    run_metadata['source_sha256'] = source_hashes
    selected = cases[:args.limit] if args.limit else cases
    payload = {"schema_version": 2, "gold_path": args.gold, "run": run_metadata,
               "complete": False, "full_suite": len(selected) == len(cases),
               "planned_n": len(selected), "batch_status": "running", "results": []}
    _dump_json(writer, payload)
    completed_summaries = {}
    try:
        for version in versions:
            partial = {"version": version, "summary": None, "records": []}
            payload['results'].append(partial)
            def checkpoint(records):
                partial['records'] = records
                _dump_json(writer, payload)
            result = evaluate_version(version, cases, limit=args.limit, reference_time=reference_time,
                                      progress_fn=checkpoint)
            # Until every requested version finishes, all aggregate scores remain absent.
            partial.update(result)
            partial['summary'] = None
            completed_summaries[version] = result['summary']
        if _source_hashes(root) != source_hashes:
            raise RuntimeError('project source changed during evaluation')
        if hashlib.sha256(Path(args.gold).read_bytes()).hexdigest() != run_metadata['gold_sha256']:
            raise RuntimeError('gold changed during evaluation')
        for result in payload['results']:
            result['summary'] = completed_summaries[result['version']]
        payload.update(complete=True, batch_status='completed')
        _dump_json(writer, payload)
    except OutputOwnershipError:
        raise
    except BaseException as exc:
        provider_error = find_permanent_model_error(exc)
        payload['complete'] = False
        payload['batch_status'] = 'stopped_provider_error' if provider_error else 'aborted'
        payload['abort'] = {'error_type': type(exc).__name__}
        if provider_error:
            payload['provider_error'] = provider_error.as_dict()
        for result in payload['results']:
            result['summary'] = None
        _dump_json(writer, payload)
        if isinstance(exc, Exception):
            return 2
        raise
    _print_table(payload['results'])
    logger.info("saved context_recall report -> %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
