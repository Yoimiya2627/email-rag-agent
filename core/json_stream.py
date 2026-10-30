"""Strict JSON-array streaming with a byte/depth gate before parser allocation."""
from pathlib import Path

import config.settings as cfg
from agents.runtime import remaining_timeout


class _BoundedReader:
    def __init__(self, stream):
        self.stream = stream
        self.depth, self.record_bytes, self.total = 0, 0, 0
        self.quoted, self.escaped, self.started = False, False, False

    def read(self, size=-1):
        if size == 0:
            return b""
        remaining_timeout(60)
        data = self.stream.read(min(size, 65536) if size > 0 else 65536)
        self.total += len(data)
        if self.total > int(getattr(cfg, "MAX_INDEX_INPUT_BYTES", 2000000000)):
            raise ValueError("Index input exceeds MAX_INDEX_INPUT_BYTES")
        for byte in data:
            if not self.started:
                if byte in b" \r\n\t":
                    continue
                if byte != ord("["):
                    raise ValueError("Invalid email input: expected a nonempty JSON array")
                self.started, self.depth = True, 1
                continue
            if self.depth >= 2:
                self.record_bytes += 1
                if self.record_bytes > int(getattr(cfg, "MAX_EMAIL_RECORD_BYTES", 16000000)):
                    raise ValueError("Email record exceeds MAX_EMAIL_RECORD_BYTES")
            if self.quoted:
                if self.escaped:
                    self.escaped = False
                elif byte == 92:
                    self.escaped = True
                elif byte == 34:
                    self.quoted = False
                continue
            if self.depth == 1 and byte not in b" \r\n\t,]{":
                raise ValueError("Invalid email input: every array item must be an object")
            if byte == 34:
                self.quoted = True
            elif byte in (123, 91):
                if self.depth == 1:
                    self.record_bytes = 1
                self.depth += 1
                if self.depth > int(getattr(cfg, "MAX_EMAIL_JSON_DEPTH", 64)):
                    raise ValueError("Email record exceeds MAX_EMAIL_JSON_DEPTH")
            elif byte in (125, 93):
                self.depth -= 1
        return data


def iter_email_records(path):
    import ijson
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Email data file not found: {path}")
    try:
        with path.open("rb") as source:
            events = ijson.parse(_BoundedReader(source), use_float=True)
            first = next(events, None)
            if first != ("", "start_array", None):
                raise ValueError("Invalid email input: expected a nonempty JSON array")
            yield from ijson.items(events, "item")
    except (ijson.JSONError, UnicodeError, OverflowError):
        raise ValueError("Invalid email input: expected UTF-8 JSON") from None
