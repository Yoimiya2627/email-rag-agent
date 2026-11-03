"""Evaluate V2/V7 on one extraction and one candidate pool per gold case.

This is an acceptance artifact, not a replacement production entry point.
The flags are fixed to the V2/V7 common retrieval configuration. With query
rewrite disabled and no history, the production reranker scores the original
question. Disabling rerank and requesting top_n=fetch_k preserves the entire
post-filter/body-candidate sequence; V2 truncates it and V7 reranks that same
sequence. An invariant check rejects any hidden candidate truncation.

Example (uses the configured DeepSeek service once per question):
  python -B paired_acceptance.py --output paired-final.json

Existing output files are never overwritten when starting a new run. During
this run, its own checkpoint is atomically replaced after every case. Partial
reports have complete=false and no aggregate score; they cannot be mistaken
for a complete 30-case evaluation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib import metadata as package_metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval_output import EvaluationOutput, OutputOwnershipError


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = '2026-11-02T12:00:00+08:00'
SOURCE_DIRECTORIES = ('agents', 'api', 'core', 'config', 'eval', 'frontend', 'langchain_version', 'models', 'scripts')
DEPENDENCY_FILES = ('pyproject.toml', 'uv.lock', 'poetry.lock', 'Pipfile', 'Pipfile.lock')
SETTING_NAMES = (
    'DEEPSEEK_MODEL', 'MODEL_REVISION', 'DEEPSEEK_THINKING_MODE',
    'AGENT_PLANNER_MODEL',
    'EMBEDDING_MODEL', 'EMBEDDING_MODEL_REVISION', 'EMBEDDING_DEVICE',
    'EMBEDDING_CPU_THREADS', 'CROSS_ENCODER_MODEL', 'CROSS_ENCODER_DEVICE',
    'CROSS_ENCODER_MAX_LENGTH', 'TOP_K', 'RERANK_TOP_N', 'RERANK_INPUT_CHAR_LIMIT',
    'RETRIEVAL_TIMEZONE', 'RETRIEVAL_TIMEZONE_OFFSET_HOURS',
    'MODEL_RUN_TOKEN_LIMIT', 'MODEL_RUN_COST_LIMIT', 'AGENT_RUN_TIMEOUT',
)
COMMON_FLAGS = dict(ENABLE_BM25=True, ENABLE_RRF=True, ENABLE_QUERY_REWRITE=False)


class AcceptanceInvariantError(RuntimeError):
    """Stop if this adapter no longer matches the production candidate stage."""


def validate_gold_cases(cases, *, require_case_ids: bool = False) -> None:
    """Validate evaluation labels; template generation intentionally permits blanks."""
    if not isinstance(cases, list) or not cases:
        raise ValueError('gold must be a non-empty JSON array')
    seen = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get('question'), str) or not case['question'].strip():
            raise ValueError('each gold question must be a non-empty string')
        if require_case_ids or 'id' in case:
            identity = case.get('id')
            if not isinstance(identity, str) or not identity.strip() or identity in seen:
                raise ValueError('case ids must be non-empty unique strings')
            seen.add(identity)
        labels = case.get('gold_chunk_ids')
        if (not isinstance(labels, list) or not labels
                or any(not isinstance(label, str) or not label.strip() for label in labels)
                or len(set(labels)) != len(labels)):
            raise ValueError('gold chunk ids must be non-empty unique strings')


class RerankerDegradedError(RuntimeError):
    """A production fallback returned results without completing reranking."""


class RerankerEvidenceError(RuntimeError):
    """Diagnostic evidence cannot establish the requested reranking method."""


def validate_rerank_evidence(details, *, expected_backend: str, candidate_count=None) -> None:
    if not isinstance(details, dict):
        raise RerankerEvidenceError('missing reranker diagnostics')
    status, reason = details.get('status'), details.get('reason')
    if status == 'failed' or (status == 'skipped' and reason == 'circuit_open'):
        raise RerankerDegradedError('reranker failed or its circuit is open')
    if status == 'succeeded' and details.get('backend') == expected_backend:
        return
    if status == 'skipped' and type(candidate_count) is int:
        if (reason == 'empty_candidates' and candidate_count == 0
                or reason == 'insufficient_candidates' and candidate_count == 1):
            return
    raise RerankerEvidenceError('reranker diagnostics do not establish the requested backend or a valid skip')


def _parse_reference(raw: str) -> datetime:
    try:
        if isinstance(raw, str) and raw.endswith('Z'):
            raw = raw[:-1] + '+00:00'  # Python 3.10 does not parse the UTC Z suffix.
        value = datetime.fromisoformat(raw)
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError('timezone required')
        return value
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError('reference time requires an ISO datetime with timezone') from exc


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes(root: Path) -> dict:
    paths = {path for directory in SOURCE_DIRECTORIES for path in (root / directory).rglob('*.py')}
    paths.update(root / name for name in DEPENDENCY_FILES if (root / name).is_file())
    paths.update(root.glob('requirements*.txt'))
    paths.update(root.glob('*.py'))
    return {path.relative_to(root).as_posix(): _hash(path) for path in sorted(paths)}


def _control_error(exc: BaseException, control_types: tuple) -> bool:
    # A wrapper must not turn cancellation/deadline/budget exhaustion into a
    # scored ordinary failure. Traverse both chains without following cycles.
    from core.model_errors import find_permanent_model_error
    if find_permanent_model_error(exc) is not None:
        return True
    pending, seen = [exc], set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, (AcceptanceInvariantError, *control_types)):
            return True
        pending.extend(item for item in (current.__cause__, current.__context__) if item is not None)
    return False


def _error_score(case: dict, exc: Exception, score_fn, stage: str) -> dict:
    result = score_fn(case, [])
    result.update(status='error', error_type=type(exc).__name__,
                  cause_type=type(exc.__cause__).__name__ if exc.__cause__ is not None else None,
                  error_stage=stage)
    return result


def _scored(case: dict, rows: list, score_fn) -> dict:
    return {**score_fn(case, rows), 'status': 'ok' if rows else 'empty'}


def evaluate_pair(case: dict, runtime, reference_time: datetime, *, fetch_k: int,
                  top_n: int, deadline_seconds: float, record: dict | None = None) -> dict:
    """Dependency-injected case runner, also exercised by offline acceptance checks."""
    record = {} if record is None else record
    details, reranker_details = {}, {}
    record.update(id=case['id'], question=case['question'], complete=False,
                  diagnostics=details, v7_reranker=reranker_details, versions={})
    start = time.monotonic()
    context = runtime.RunContext(deadline=start + deadline_seconds)
    try:
        with runtime.use_run_context(context):
            for name, value in COMMON_FLAGS.items():
                setattr(runtime.cfg, name, value)
            runtime.cfg.ENABLE_RERANKER = False
            runtime.cfg.RERANKER_BACKEND = 'llm'  # Disabled, exactly as in V2.
            try:
                candidates = list(runtime.retrieve(
                    case['question'], top_n=fetch_k, fetch_k=fetch_k,
                    reference_time=reference_time, diagnostics=details,
                ))
            except Exception as exc:
                if _control_error(exc, runtime.control_types):
                    raise
                stage = details.get('stage', 'retrieve')
                for version in ('V2', 'V7'):
                    record['versions'][version] = _error_score(case, exc, runtime.score, stage)
            else:
                ids = [row.chunk_id for row in candidates]
                if ids != details.get('rerank_candidate_ids'):
                    raise AcceptanceInvariantError('retrieve did not return the complete rerank candidate sequence')
                if details.get('scoring_query') != case['question']:
                    raise AcceptanceInvariantError('unexpected scoring query under the fixed common flags')
                record['candidate_ids'] = ids
                record['versions']['V2'] = _scored(case, candidates[:top_n], runtime.score)
                if 'filters' in details:
                    clock = datetime.fromisoformat(details['reference_time'])
                    scope = runtime.FilterSpec.from_mapping(details['filters'], now=clock)
                    record['gold_scope_checks'] = [
                        {'email_id': email_id, 'date': runtime.emails[email_id]['date'],
                         'subject': runtime.emails[email_id]['subject'],
                         'matches': scope.matches(runtime.emails[email_id])}
                        for email_id in case.get('source_email_ids', [])
                        if email_id in runtime.emails
                    ]
                runtime.cfg.ENABLE_RERANKER = True
                runtime.cfg.RERANKER_BACKEND = 'cross_encoder'
                try:
                    ranked = list(runtime.rerank(
                        case['question'], list(candidates), top_n=top_n,
                        diagnostics=reranker_details,
                    ))
                    try:
                        validate_rerank_evidence(reranker_details, expected_backend='cross_encoder',
                                                candidate_count=len(candidates))
                    except RerankerEvidenceError as exc:
                        raise AcceptanceInvariantError(str(exc)) from exc
                except Exception as exc:
                    if _control_error(exc, runtime.control_types):
                        raise
                    record['versions']['V7'] = _error_score(case, exc, runtime.score, 'rerank')
                else:
                    record['versions']['V7'] = _scored(case, ranked, runtime.score)
            record['complete'] = True
            return record
    finally:
        record['model_usage'] = runtime.model_metrics_snapshot(context)
        record['duration_seconds'] = round(time.monotonic() - start, 3)


def summarize(records: list[dict]) -> dict:
    n = len(records)
    return {
        'n': n,
        'mean_context_recall': round(sum(row['context_recall'] for row in records) / n, 4) if n else 0.0,
        'chunk_hit_rate': round(sum(row['has_any_hit'] for row in records) / n, 4) if n else 0.0,
        'perfect_recall_rate': round(sum(row['has_perfect_recall'] for row in records) / n, 4) if n else 0.0,
        'ok_count': sum(row['status'] == 'ok' for row in records),
        'empty_count': sum(row['status'] == 'empty' for row in records),
        'error_count': sum(row['status'] == 'error' for row in records),
    }


def _checkpoint(writer: EvaluationOutput, report: dict) -> None:
    report['completed_n'] = sum(case['complete'] for case in report['paired_cases'])
    report['results'] = []
    for version in ('V2', 'V7'):
        records = [case['versions'][version] for case in report['paired_cases']
                   if case['complete'] and version in case['versions']]
        report['results'].append({
            'version': version,
            'flags': {**COMMON_FLAGS, 'ENABLE_RERANKER': version == 'V7',
                      'RERANKER_BACKEND': 'cross_encoder' if version == 'V7' else 'llm'},
            'summary': summarize(records) if report['complete'] else None,
            'records': records,
        })
    writer.write(report)


def _validate_import_sources(root: Path) -> None:
    root = root.resolve()
    for name, module in tuple(sys.modules.items()):
        if name.split('.')[0] not in SOURCE_DIRECTORIES:
            continue
        filename = getattr(module, '__file__', None)
        if filename and not Path(filename).resolve().is_relative_to(root):
            raise AcceptanceInvariantError(f'project module imported outside selected root: {name}')


def _runtime(root: Path):
    _validate_import_sources(root)
    sys.path.insert(0, str(root))
    import config.settings as cfg
    # Process-local workaround only. Do not change the user's environment file.
    os.environ['NO_PROXY'] = ','.join(
        '::1' if item.strip() == '[::1]' else item.strip()
        for item in os.environ.get('NO_PROXY', '').split(',')
    )
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    from agents.runtime import (RunContext, use_run_context, RunCancelled,
                                RunDeadlineExceeded, ContextBudgetExceeded)
    from core.model_clients import ModelBudgetExceeded, model_metrics_snapshot
    from core.pipeline import retrieve
    from core.reranker import rerank, reset_circuit_breaker
    from core.embedder import index_snapshot, current_index_manifest, get_corpus_revision
    from core.filters import FilterSpec
    from scripts.evaluate_context_recall import score_context_recall_case
    _validate_import_sources(root)
    corpus = Path(cfg.EMAIL_DATA_PATH)
    corpus = corpus if corpus.is_absolute() else root / corpus
    emails = json.loads(corpus.read_text(encoding='utf-8'))
    if not isinstance(emails, list):
        raise ValueError('acceptance corpus must be a JSON array')
    return SimpleNamespace(
        cfg=cfg, RunContext=RunContext, use_run_context=use_run_context,
        control_types=(RunCancelled, RunDeadlineExceeded, ContextBudgetExceeded, ModelBudgetExceeded),
        model_metrics_snapshot=model_metrics_snapshot, retrieve=retrieve, rerank=rerank,
        reset_circuit_breaker=reset_circuit_breaker, score=score_context_recall_case,
        index_snapshot=index_snapshot, current_index_manifest=current_index_manifest,
        get_corpus_revision=get_corpus_revision, FilterSpec=FilterSpec, corpus_path=corpus,
        emails={row['id']: row for row in emails},
    )


def _resolved_revisions() -> dict:
    # Inspect already loaded models only; never trigger downloads or initialization.
    from core import embedder, reranker
    model = getattr(reranker, '_cross_encoder', None)
    revision = getattr(getattr(getattr(model, 'model', None), 'config', None), '_commit_hash', None)
    return {'embedding': embedder.resolved_model_revision(),
            'cross_encoder': revision if isinstance(revision, str) else None}


def _provider_endpoint(cfg) -> str | None:
    endpoint = urlsplit(getattr(cfg, 'DEEPSEEK_BASE_URL', ''))
    # Deliberately omit credentials, query strings and fragments.
    if not endpoint.hostname:
        return None
    return f'{endpoint.scheme}://{endpoint.hostname}' + (f':{endpoint.port}' if endpoint.port else '')


def _metadata(root: Path, gold: Path, runtime, reference_time: datetime) -> dict:
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, check=True,
                            capture_output=True, text=True).stdout.strip()
    packages = {}
    for name in ('sentence-transformers', 'transformers', 'torch', 'chromadb', 'openai'):
        try:
            packages[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            packages[name] = None
    manifest = runtime.current_index_manifest()
    return {
        'started_at': datetime.now(timezone.utc).isoformat(),
        'reference_time': reference_time.isoformat(), 'git_head': commit,
        'root': str(root), 'python_executable': sys.executable,
        'git_status': subprocess.run(['git', 'status', '--porcelain'], cwd=root, check=True, capture_output=True, text=True).stdout.splitlines(),
        'source_sha256': _source_hashes(root), 'acceptance_script_sha256': _hash(Path(__file__)),
        'gold_sha256': _hash(gold), 'corpus_sha256': _hash(runtime.corpus_path),
        'corpus_revision': runtime.get_corpus_revision(),
        'manifest': {key: manifest.get(key) for key in (
            'generation', 'collection', 'corpus_sha256', 'source_sha256', 'chunk_count',
            'email_count', 'config_fingerprint', 'resolved_model_revision',
        )} if manifest else None,
        'settings': {name: getattr(runtime.cfg, name, None) for name in SETTING_NAMES},
        'provider_endpoint_origin': _provider_endpoint(runtime.cfg),
        'packages': packages,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--gold', type=Path, default=None)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('paired_acceptance.json'))
    parser.add_argument('--reference-time', type=_parse_reference,
                        default=_parse_reference(DEFAULT_REFERENCE))
    parser.add_argument('--limit', type=_positive_int, default=None,
                        help='Optional smoke-run limit; a limited report is not a full-suite result')
    parser.add_argument('--case-timeout', type=_positive_int, default=180)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    gold = (args.gold or root / 'data/gold_chunks.json').resolve()
    output = args.output.resolve()
    if output.exists():
        parser.error('output already exists; choose a new file to preserve previous evidence')
    cases = json.loads(gold.read_text(encoding='utf-8'))
    try:
        validate_gold_cases(cases, require_case_ids=True)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        with EvaluationOutput(output) as writer:
            try:
                return _run_owned(args, root, gold, output, cases, writer)
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


def _run_owned(args, root, gold, output, cases, writer):
    selected = cases[:args.limit] if args.limit is not None else cases
    os.chdir(root)
    runtime = _runtime(root)
    fetch_k, top_n = runtime.cfg.TOP_K * 4, runtime.cfg.RERANK_TOP_N
    if type(fetch_k) is not int or type(top_n) is not int or not 1 <= top_n <= fetch_k:
        raise ValueError('requires positive top_n <= fetch_k')
    output.parent.mkdir(parents=True, exist_ok=True)
    flag_names = (*COMMON_FLAGS, 'ENABLE_RERANKER', 'RERANKER_BACKEND')
    original_flags = {name: getattr(runtime.cfg, name) for name in flag_names}
    report = dict(schema_version=1, mode='shared_candidates', complete=False,
                  full_suite=len(selected) == len(cases), planned_n=len(selected), batch_status='running',
                  gold_path=str(gold), top_n=top_n, fetch_k=fetch_k,
                  paired_cases=[], results=[])
    try:
        with runtime.index_snapshot():
            report['run'] = _metadata(root, gold, runtime, args.reference_time)
            runtime.reset_circuit_breaker()
            _checkpoint(writer, report)
            for case in selected:
                record = {}
                report['paired_cases'].append(record)
                evaluate_pair(case, runtime, args.reference_time, fetch_k=fetch_k,
                              top_n=top_n, deadline_seconds=args.case_timeout, record=record)
                _checkpoint(writer, report)
                print(json.dumps({
                    'id': case['id'], 'completed': len(report['paired_cases']), 'planned': len(selected),
                    'V2': record['versions']['V2']['context_recall'],
                    'V7': record['versions']['V7']['context_recall'],
                    'status_V2': record['versions']['V2']['status'],
                    'status_V7': record['versions']['V7']['status'],
                    'model_calls': record['model_usage']['call_count'],
                }, ensure_ascii=False), flush=True)
            if _source_hashes(root) != report['run']['source_sha256']:
                raise AcceptanceInvariantError('project source changed during acceptance')
            if _hash(gold) != report['run']['gold_sha256'] or _hash(runtime.corpus_path) != report['run']['corpus_sha256']:
                raise AcceptanceInvariantError('gold or source corpus changed during acceptance')
            _validate_import_sources(root)
            report['run']['resolved_model_revisions'] = _resolved_revisions()
            report['batch_status'] = 'completed'
            report['complete'] = True
            report['finished_at'] = datetime.now(timezone.utc).isoformat()
            _checkpoint(writer, report)
    except OutputOwnershipError:
        raise
    except BaseException as exc:
        from core.model_errors import find_permanent_model_error
        provider_error = find_permanent_model_error(exc)
        report['complete'] = False
        report.pop('finished_at', None)
        report['batch_status'] = 'stopped_provider_error' if provider_error else 'aborted'
        if provider_error:
            report['provider_error'] = provider_error.as_dict()
        report['abort'] = {'error_type': type(exc).__name__,
                           'cause_type': type(exc.__cause__).__name__ if exc.__cause__ else None}
        _checkpoint(writer, report)
        if isinstance(exc, Exception):
            return 2
        raise
    finally:
        for name, value in original_flags.items():
            setattr(runtime.cfg, name, value)
    for result in report['results']:
        print(json.dumps({'version': result['version'], **result['summary']}, ensure_ascii=False), flush=True)
    print(f'Saved complete paired report: {output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
