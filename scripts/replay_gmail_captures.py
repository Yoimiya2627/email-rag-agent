"""Offline, repeatable reparse of retained Gmail full-JSON captures."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agents.gmail_readonly import GmailReadOnlyProvider, MailContentError, parser_version
from agents.runtime import remaining_timeout
from core.loader import validate_email_records
from scripts.sync_gmail_readonly import _dump_json
from core.capture_selection import read_json, safe_path, selected_capture, LegacyVersions


def replay_captures(raw_dir, output_path, *, report_path=None, original_path=None, dry_run=True, checkpoint_dir=None,
                    selection_dir=None):
    raw, output = Path(raw_dir).resolve(), Path(output_path).resolve()
    report = Path(report_path).resolve() if report_path else output.with_suffix(output.suffix + ".report.json")
    checkpoint = Path(checkpoint_dir).resolve() if checkpoint_dir else output.with_suffix(output.suffix + ".replay")
    targets = [output, report, checkpoint]
    if any(path == raw or raw in path.parents or path in raw.parents for path in targets):
        raise ValueError("Replay outputs and checkpoints must be outside the original capture directory")
    if len(set(targets)) != 3 or any(a in b.parents or b in a.parents for i, a in enumerate(targets) for b in targets[i + 1:]):
        raise ValueError("Replay output, report and checkpoint must be separate")
    protected = [Path(value).resolve() for value in (original_path, selection_dir) if value is not None]
    if any(a == b or a in b.parents or b in a.parents for a in targets for b in protected):
        raise ValueError("Replay outputs must not overlap the original corpus or selection directory")
    if output.exists():
        raise ValueError("Replay output must be a new file; never overwrite the existing corpus")
    original = {}
    if original_path:
        original = {email.id: email for email in validate_email_records(json.loads(Path(original_path).read_text(encoding="utf-8")))}
    reader, results, rows, seen = GmailReadOnlyProvider(), [], [], set()
    legacy_versions = LegacyVersions(raw)
    version = parser_version()
    files = sorted(raw.glob("*.json"))
    if not files:
        raise ValueError("Capture directory contains no Gmail JSON files")
    for file in files:
        remaining_timeout(60)
        digest, selected_name, selection_method = None, file.name, None
        try:
            envelope, root_hash = read_json(safe_path(raw, file.name))
            selected, envelope, digest, selection_method = selected_capture(raw, file, envelope, root_hash,
                selection_dir=selection_dir)
            if selection_method == 'legacy_root':
                try:
                    reader.email_from_capture(envelope)
                except (ValueError, MailContentError) as exc:
                    if getattr(exc, 'code', None) != 'missing_captured_body_data':
                        raise
                    selected, envelope, digest, selection_method = legacy_versions.find_complete(envelope, reader)
            selected_name = selected.relative_to(raw).as_posix()
            cache = checkpoint / (hashlib.sha256((selected_name + digest + version).encode()).hexdigest() + ".json")
            if cache.exists():
                from models.schemas import Email
                email = Email.model_validate(json.loads(cache.read_text(encoding="utf-8")))
            else:
                email = reader.email_from_capture(envelope)
                if not dry_run:
                    _dump_json(cache, email.model_dump())
            if email.id in seen:
                raise ValueError("duplicate_message_id")
            seen.add(email.id)
            results.append(email.model_dump())
            old = original.get(email.id)
            record = {"capture_file": selected_name, "root_file": file.name, "selection_method": selection_method,
                      "capture_sha256": digest, "email_id": email.id,
                      "status": "parsed", "body_sha256": hashlib.sha256(email.body.encode()).hexdigest(),
                      "body_chars": len(email.body), "decode_quality": email.decode_quality}
            if old:
                record.update({"body_changed": old.body != email.body,
                    "previous_body_sha256": hashlib.sha256(old.body.encode()).hexdigest(),
                    "numbers_changed": re.findall(r"\d+(?:[.,]\d+)*", old.body) != re.findall(r"\d+(?:[.,]\d+)*", email.body),
                    "negation_changed": re.findall(r"\b(?:not|no|never)\b|不|未|无", old.body, re.I) != re.findall(r"\b(?:not|no|never)\b|不|未|无", email.body, re.I)})
            rows.append(record)
        except (ValueError, TypeError, KeyError, UnicodeError, RuntimeError, OSError) as exc:
            from agents.runtime import RunCancelled, RunDeadlineExceeded
            if isinstance(exc, (RunCancelled, RunDeadlineExceeded)):
                raise
            rows.append({"capture_file": selected_name, "root_file": file.name,
                         "capture_sha256": digest, "status": "failed",
                         "error_code": getattr(exc, "code", type(exc).__name__)})
    summary = {"parser_version": version, "dry_run": dry_run, "input_count": len(files),
               "parsed_count": len(results), "failed_count": sum(row["status"] == "failed" for row in rows),
               "source_format": "gmail-full-json-selected-body-parts", "captures_unchanged": True, "files": rows}
    _dump_json(report, summary)
    if not dry_run:
        if summary["failed_count"]:
            raise ValueError("Replay contains failed captures; report/checkpoints retained, corpus not published")
        validate_email_records(results)
        _dump_json(output, results)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    parser.add_argument("--original-corpus")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--selection-dir", help="Explicit per-message selection records; defaults to raw-dir/selections")
    parser.add_argument("--write", action="store_true", help="Publish new corpus only after every capture parses; default is dry-run")
    args = parser.parse_args()
    result = replay_captures(args.raw_dir, args.output, report_path=args.report,
        original_path=args.original_corpus, checkpoint_dir=args.checkpoint_dir, dry_run=not args.write,
        selection_dir=args.selection_dir)
    print(json.dumps({key: value for key, value in result.items() if key != "files"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
