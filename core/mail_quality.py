"""Bounded MIME accounting and conservative decoding signals; no guessing."""
import re
import time
from dataclasses import dataclass, field

import config.settings as cfg
from agents.runtime import remaining_timeout


class MimeBudgetExceeded(ValueError):
    pass


@dataclass
class MimeBudget:
    started: float = field(default_factory=time.monotonic)
    decoded_bytes: int = 0
    encoded_bytes: int = 0

    def check(self):
        remaining_timeout(60)
        if time.monotonic() - self.started > float(getattr(cfg, "MIME_PARSE_SECONDS", 15)):
            raise MimeBudgetExceeded("mime_time_limit")

    def inspect(self, payload):
        stack, nodes = [(payload, 0)], 0
        header_chars = 0
        while stack:
            self.check()
            part, depth = stack.pop()
            nodes += 1
            if nodes > int(getattr(cfg, "MIME_PART_LIMIT", 1000)):
                raise MimeBudgetExceeded("mime_part_limit")
            if depth > int(getattr(cfg, "MIME_DEPTH_LIMIT", 32)):
                raise MimeBudgetExceeded("mime_depth_limit")
            if not isinstance(part, dict):
                raise MimeBudgetExceeded("invalid_mime_part")
            header_chars += sum(len(str(part.get(key) or "")) for key in ("filename", "mimeType", "partId"))
            for header in part.get("headers") or []:
                if not isinstance(header, dict):
                    raise MimeBudgetExceeded("invalid_mime_part")
                header_chars += len(str(header.get("name", ""))) + len(str(header.get("value", "")))
            if header_chars > int(getattr(cfg, "MIME_HEADER_CHAR_LIMIT", 64000)):
                raise MimeBudgetExceeded("mime_header_limit")
            data = (part.get("body") or {}).get("data") or ""
            if not isinstance(data, str):
                raise MimeBudgetExceeded("invalid_body_encoding")
            # All inline payload strings already arrived in JSON; reject before
            # base64 padding/byte allocation, including discarded alternatives.
            self.encoded_bytes += len(data)
            if self.encoded_bytes > int(getattr(cfg, "MIME_ENCODED_BYTE_LIMIT", 16000000)):
                raise MimeBudgetExceeded("mime_encoded_limit")
            children = part.get("parts") or []
            if not isinstance(children, list) or len(children) + nodes > int(getattr(cfg, "MIME_PART_LIMIT", 1000)):
                raise MimeBudgetExceeded("mime_part_limit")
            stack.extend((child, depth + 1) for child in children)

    def before_decode(self, data, declared_size=0):
        self.check()
        maximum = int(getattr(cfg, "MIME_DECODED_BYTE_LIMIT", 8000000))
        if declared_size < 0 or declared_size > maximum - self.decoded_bytes:
            raise MimeBudgetExceeded("mime_decoded_limit")
        if len(data) > int(getattr(cfg, "MIME_ENCODED_BYTE_LIMIT", 16000000)):
            raise MimeBudgetExceeded("mime_encoded_limit")
        if (len(data.rstrip("=")) * 3 // 4) > maximum - self.decoded_bytes:
            raise MimeBudgetExceeded("mime_decoded_limit")

    def decoded(self, size):
        self.decoded_bytes += size
        if self.decoded_bytes > int(getattr(cfg, "MIME_DECODED_BYTE_LIMIT", 8000000)):
            raise MimeBudgetExceeded("mime_decoded_limit")


def text_quality(text):
    signals = []
    replacements = text.count("\ufffd")
    controls = sum(ord(ch) < 32 and ch not in "\n\r\t" for ch in text)
    if replacements:
        signals.append("unicode_replacement_character")
    if controls:
        signals.append("unexpected_control_character")
    if re.search(r"(?:Ã[\u0080-\u00bf]|Â[\u0080-\u00bf]|â[\u0080-\u00bf]|ðŸ)", text):
        signals.append("possible_utf8_mojibake")
    if re.search(r"=\?[^?]+\?[bBqQ]\?", text):
        signals.append("undecoded_encoded_word")
    return {"status": "suspect" if signals else "no_signal", "signals": signals,
            "replacement_count": replacements, "control_count": controls,
            "corrected": False}
