import json
import logging
from pathlib import Path
from typing import List

from models.schemas import Email
import config.settings as cfg

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = {"id", "subject", "sender", "recipients", "date", "body"}


def load_emails(path: str = None) -> List[Email]:
    data_path = Path(path or cfg.EMAIL_DATA_PATH)
    if not data_path.exists():
        raise FileNotFoundError(f"Email data file not found: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        raise ValueError("Email data must be a JSON array")

    emails: List[Email] = []
    for i, item in enumerate(raw_data):
        missing = REQUIRED_FIELDS - set(item.keys())
        if missing:
            logger.warning(f"Email[{i}] missing fields {missing}, skipping")
            continue
        try:
            emails.append(Email(**item))
        except Exception as exc:
            logger.warning(f"Email[{i}] parse error: {exc}, skipping")

    logger.info(f"Loaded {len(emails)} valid emails from {data_path}")
    return emails
