"""Index emails into ChromaDB from the command line.

Mirrors the /index HTTP endpoint in api/main.py but works without a running
server, so `make index` can run as a one-shot during onboarding.

Usage:
    python scripts/index_emails.py                  # use cfg.EMAIL_DATA_PATH
    python scripts/index_emails.py --data-path ...  # custom JSON
    python scripts/index_emails.py --clear          # wipe collection first
"""
import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config.settings as cfg
from core.loader import load_emails
from core.cleaner import clean_email
from core.chunker import chunk_email
from core.embedder import index_chunks, clear_collection, get_collection_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


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
        help="Clear the existing collection before indexing",
    )
    args = parser.parse_args()

    if args.clear:
        logger.info("Clearing existing collection ...")
        clear_collection()

    t0 = time.time()
    emails = load_emails(args.data_path)
    logger.info(f"Loaded {len(emails)} emails — chunking ...")

    all_chunks = []
    for email in emails:
        cleaned = clean_email(email)
        all_chunks.extend(chunk_email(cleaned))
    logger.info(f"Produced {len(all_chunks)} chunks — embedding + writing to ChromaDB ...")

    count = index_chunks(all_chunks)

    # Drop BM25 cache so the next query rebuilds it over the new corpus.
    from core.retriever import invalidate_bm25_cache
    invalidate_bm25_cache()

    stats = get_collection_stats()
    elapsed = time.time() - t0
    logger.info(
        f"Done in {elapsed:.1f}s — {count} new chunks indexed, "
        f"{stats.get('chunk_count', '?')} chunks total in collection."
    )


if __name__ == "__main__":
    main()
