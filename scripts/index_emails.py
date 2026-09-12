"""Index emails into ChromaDB from the command line.

Mirrors the /index HTTP endpoint in api/main.py but works without a running
server, so `make index` can run as a one-shot during onboarding.

Usage:
    python scripts/index_emails.py                  # use cfg.EMAIL_DATA_PATH
    python scripts/index_emails.py --data-path ...  # custom JSON
    python scripts/index_emails.py --clear          # replace the existing corpus
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config.settings as cfg
from core.ingestion import prepare_email_chunks
from core.embedder import index_chunks, get_collection_stats, rollback_generation, resume_index_generation
from core.index_manifest import list_generations, atomic_json
from models.schemas import Email

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _report_destination(parser, target, data_path):
    if target in (None, '-'):
        return target
    path = Path(target).expanduser().resolve()
    sources = {Path(data_path or cfg.EMAIL_DATA_PATH).expanduser().resolve(),
               Path(cfg.EMAIL_DATA_PATH).expanduser().resolve()}
    if any(path == source or (path.exists() and source.exists() and path.samefile(source)) for source in sources):
        parser.error('--json-report must not overwrite an input email corpus')
    if path.is_dir():
        parser.error('--json-report must name a file')
    return path


def _write_report(target, payload):
    if target == '-':
        print(json.dumps(payload,ensure_ascii=False,indent=2))
    else:
        atomic_json(target,payload)


def main():
    parser = argparse.ArgumentParser(description="Index emails into ChromaDB")
    parser.add_argument(
        "--data-path",
        default=None,
        help=f"Path to emails JSON (default: {cfg.EMAIL_DATA_PATH})",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Replace the corpus after input validation and successful indexing",
    )
    maintenance = parser.add_mutually_exclusive_group()
    maintenance.add_argument("--list-generations", action="store_true", help="List retained ready/paused index generations")
    maintenance.add_argument("--rollback", metavar="GENERATION", help="Verify and reactivate a retained compatible generation")
    maintenance.add_argument("--resume-generation", metavar="GENERATION", help="Resume a paused generation from its verified local input")
    parser.add_argument('--force-reembed', action='store_true', help='Explicitly re-encode all supplied chunks instead of reusing verified vectors')
    parser.add_argument('--json-report', nargs='?', const='-', default=None, metavar='PATH',
                        help='Write indexing counts/timings as JSON (omit PATH for stdout)')
    args = parser.parse_args()
    report_target = _report_destination(parser,args.json_report,args.data_path)
    if args.force_reembed and (args.resume_generation or args.rollback or args.list_generations):
        parser.error('--force-reembed applies to a new indexing run, not generation maintenance')

    if args.list_generations:
        print(json.dumps(list_generations(), ensure_ascii=False, indent=2))
        return
    if args.rollback:
        print(json.dumps(rollback_generation(args.rollback), ensure_ascii=False, indent=2))
        return
    if args.resume_generation:
        from core.index_metrics import collect_index_metrics
        report = None
        count = None
        try:
            with collect_index_metrics() as report:
                count = resume_index_generation(args.resume_generation)
            if report.outcome != 'unchanged':
                from core.retriever import invalidate_bm25_cache
                invalidate_bm25_cache()
            payload = {'indexed_chunks':count,'generation':args.resume_generation,'index_metrics':report.to_dict()}
            if report_target != '-':
                print(json.dumps(payload,ensure_ascii=False,indent=2))
        finally:
            if report_target and report is not None:
                payload = {'generation':args.resume_generation,'index_metrics':report.to_dict()}
                if count is not None:
                    payload['indexed_chunks'] = count
                _write_report(report_target,payload)
        return

    from core.index_metrics import collect_index_metrics
    report = None
    try:
        with collect_index_metrics() as report:
            emails, all_chunks = prepare_email_chunks(args.data_path)
            try:
                email_count = len(emails)
                logger.info('Loaded and validated %s emails; prepared %s chunks', email_count, len(all_chunks))
                count = index_chunks(all_chunks, replace=args.clear,
                                     **({'force_reembed':True} if args.force_reembed else {}))
            finally:
                close = getattr(emails,'close',None)
                if close:
                    close()
        if report.outcome != 'unchanged':
            from core.retriever import invalidate_bm25_cache
            invalidate_bm25_cache()
        stats = get_collection_stats()
        logger.info('Index %s: %s input emails, %s returned chunks, %s collection chunks; %.2fs',
                    report.outcome, email_count, count, stats.get('chunk_count','?'),report.to_dict()['total_seconds'])
        logger.info('Index work counts: %s', json.dumps(report.to_dict().get('counts',{}),sort_keys=True))
    finally:
        if report_target and report is not None:
            _write_report(report_target,{'index_metrics':report.to_dict()})


if __name__ == "__main__":
    main()
