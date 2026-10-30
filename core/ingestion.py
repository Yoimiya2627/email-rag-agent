"""Strict streaming preflight into a private spool; no active-index mutation."""
from pathlib import Path
import json
import os
import tempfile
import weakref
from models.schemas import Email
from core.json_stream import iter_email_records

from core.loader import validate_email_record
from core.cleaner import clean_email
from core.chunker import chunk_email
from agents.runtime import remaining_timeout, current_run
import config.settings as cfg
from core.index_manifest import configuration_fingerprint
from core.index_metrics import timed_stage


class EmailPlan:
    """Repeatable normalized records, disk-backed rather than corpus-sized RAM."""
    def __init__(self, directory, path, count):
        self.path, self.count = path, count
        self._cleanup = weakref.finalize(self, directory.cleanup)

    def __len__(self):
        return self.count

    def __iter__(self):
        with self.path.open("r", encoding="utf-8") as source:
            for line in source:
                remaining_timeout(60)
                yield Email.model_validate_json(line)

    def __getitem__(self, position):
        if not isinstance(position, int) or position < 0:
            raise IndexError("EmailPlan supports nonnegative record positions")
        for index, email in enumerate(self):
            if index == position:
                return email
        raise IndexError(position)

    def close(self):
        self._cleanup()


class ChunkPlan:
    """Hold a normalized-email spool and emit one bounded email's chunks."""
    def __init__(self, emails, count, options, fingerprint):
        self.emails, self.count = emails, count
        self.config_fingerprint, self.options = fingerprint, options

    def __len__(self):
        return self.count

    def __iter__(self):
        for email in self.emails:
            remaining_timeout(60)
            yield from chunk_email(email, **self.options)


@timed_stage("preprocessing")
def prepare_email_chunks(data_path: str | Path | None = None):
    directory = tempfile.TemporaryDirectory(prefix="email-index-prepare-")
    spool = Path(directory.name) / "normalized.jsonl"
    count, email_count, identifiers = 0, 0, set()
    options = {"chunk_size": cfg.CHUNK_SIZE, "chunk_overlap": cfg.CHUNK_OVERLAP,
               "min_chunk_size": cfg.MIN_CHUNK_SIZE}
    fingerprint = configuration_fingerprint()
    try:
        with spool.open("x", encoding="utf-8", newline="\n") as output:
            try:
                os.chmod(spool, 0o600)
            except OSError:
                pass  # Windows temp directory inherits the user's ACL.
            for position, record in enumerate(iter_email_records(data_path or cfg.EMAIL_DATA_PATH), 1):
                remaining_timeout(60)
                if position > int(getattr(cfg, "MAX_INDEX_INPUT_EMAILS", 100000)):
                    raise ValueError("Index input exceeds MAX_INDEX_INPUT_EMAILS")
                email = validate_email_record(record, position)
                if email.id in identifiers:
                    raise ValueError(f"Invalid or duplicate email ID at record {position}")
                identifiers.add(email.id)
                try:
                    cleaned = clean_email(email)
                    # Bound one-email splitting before constructing its spans.
                    per_email_limit = int(getattr(cfg, "MAX_EMAIL_CHUNKS", 10000))
                    step = options["chunk_size"] - options["chunk_overlap"]
                    if step <= 0 or (len(cleaned.body) + len(cleaned.subject) + 10) // step > per_email_limit:
                        raise ValueError("Email exceeds MAX_EMAIL_CHUNKS")
                    produced = len(chunk_email(cleaned, **options))
                    if produced > per_email_limit:
                        raise ValueError("Email exceeds MAX_EMAIL_CHUNKS")
                    count += produced
                except (ValueError, TypeError, KeyError, AttributeError, UnicodeError):
                    raise ValueError(f"Invalid email content or per-email chunk budget at record {position}") from None
                if count > int(getattr(cfg, "MAX_INDEX_INPUT_CHUNKS", 250000)):
                    raise ValueError("Index input exceeds MAX_INDEX_INPUT_CHUNKS")
                output.write(cleaned.model_dump_json() + "\n")
                email_count = position
                run = current_run()
                if run:
                    run.progress("preparing_index", completed_emails=position, total_emails=None)
            output.flush()
            os.fsync(output.fileno())
        if not email_count:
            raise ValueError("Invalid email input: expected a nonempty JSON array")
        if not count:
            raise ValueError("Invalid email input: no indexable chunks were produced")
        if configuration_fingerprint() != fingerprint:
            raise ValueError("Index configuration changed during preparation; prepare again")
        emails = EmailPlan(directory, spool, email_count)
        return emails, ChunkPlan(emails, count, options, fingerprint)
    except BaseException:
        directory.cleanup()
        raise
