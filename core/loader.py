import json
import logging
from pathlib import Path
from typing import List

from models.schemas import Email
import config.settings as cfg

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = {"id", "subject", "sender", "recipients", "date", "body"}


def load_emails(path: str = None) -> List[Email]:
    """Validate the complete corpus before callers perform any index mutation.

    Import errors contain only record positions, never Pydantic input values.
    Partial import is deliberately not an implicit fallback.
    """
    data_path = Path(path or cfg.EMAIL_DATA_PATH)
    if not data_path.exists():
        raise FileNotFoundError(f"Email data file not found: {data_path}")

    try:
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except (json.JSONDecodeError, UnicodeError):
        raise ValueError("Invalid email input: expected UTF-8 JSON") from None
    return validate_email_records(raw_data)


def validate_email_records(raw_data) -> List[Email]:
    """Shared strict record contract for API, CLI and offline ingestion."""
    if not isinstance(raw_data, list) or not raw_data:
        raise ValueError("Invalid email input: expected a nonempty JSON array")

    emails: List[Email] = []
    identifiers = set()
    for i, item in enumerate(raw_data, 1):
        email = validate_email_record(item, i)
        if not email.id.strip() or email.id in identifiers:
            raise ValueError(f"Invalid or duplicate email ID at record {i}")
        identifiers.add(email.id)
        emails.append(email)
    logger.info("Validated %s email records", len(emails))
    return emails


def validate_email_record(item, position):
    if not isinstance(item, dict) or REQUIRED_FIELDS - item.keys():
        raise ValueError(f"Invalid email input at record {position}")
    try:
        email = Email.model_validate(item)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid email input at record {position}") from None
    if not email.id.strip():
        raise ValueError(f"Invalid email ID at record {position}")
    if len(email.id) > 512:
        raise ValueError(f"Email ID exceeds the 512-character budget at record {position}")
    return email
