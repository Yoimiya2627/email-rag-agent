"""Preload the bge-m3 embedding model so the first `make run` doesn't stall.

sentence-transformers downloads the model on first use (~570MB for bge-m3).
Without preloading, the first incoming request to /index or /chat blocks for
several minutes silently. Running this script during `make install` moves
that wait into the install step where the user expects it.

Idempotent — if the model is already cached, this is a few seconds at most.
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config.settings as cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info(
        f"Preloading embedding model {cfg.EMBEDDING_MODEL} (device={cfg.EMBEDDING_DEVICE}) ..."
    )
    logger.info("First run downloads ~570MB and may take several minutes.")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(cfg.EMBEDDING_MODEL, device=cfg.EMBEDDING_DEVICE)
    # Force a tiny encode so any lazy initialization (tokenizer, weights) finishes now.
    _ = model.encode(["hello"], show_progress_bar=False)
    logger.info("Embedding model ready.")


if __name__ == "__main__":
    main()
