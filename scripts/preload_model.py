"""Explicit embedding preparation; uses configured model/revision/endpoint.

This command may download weights. Dependency installation does not invoke it.
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
    logger.info("Explicit model preparation may download weights; size depends on the selected model.")

    from sentence_transformers import SentenceTransformer

    revision = getattr(cfg, 'EMBEDDING_MODEL_REVISION', None)
    options = {'revision':revision} if revision else {}
    model = SentenceTransformer(cfg.EMBEDDING_MODEL, device=cfg.EMBEDDING_DEVICE, **options)
    # Force a tiny encode so any lazy initialization (tokenizer, weights) finishes now.
    _ = model.encode(["hello"], show_progress_bar=False)
    logger.info("Embedding model ready.")


if __name__ == "__main__":
    main()
