import re
import logging
from models.schemas import Email

logger = logging.getLogger(__name__)

_HTML_TAGS = re.compile(r"<[^>]+>")
_HTML_ENTITIES = re.compile(r"&(?:[a-zA-Z]+|#\d+);")
_MULTI_NEWLINES = re.compile(r"\n{3,}")
_MULTI_SPACES = re.compile(r" {2,}")

# Common email signature openers (both Chinese and English)
_SIG_PATTERNS = [
    r"^--\s*$",
    r"^Best regards?,?\s*$",
    r"^Thanks?,?\s*$",
    r"^Sincerely,?\s*$",
    r"^Cheers,?\s*$",
    r"^Regards?,?\s*$",
    r"^此致[\s,，]*$",
    r"^谢谢[\s!！]*$",
    r"^祝好[\s,，]*$",
    r"^顺颂商祺[\s,，]*$",
]
_SIG_RE = re.compile(
    "|".join(f"(?:{p})" for p in _SIG_PATTERNS),
    re.IGNORECASE | re.MULTILINE,
)


def _strip_html(text: str) -> str:
    text = _HTML_TAGS.sub(" ", text)
    text = _HTML_ENTITIES.sub(" ", text)
    return text


def _remove_signature(text: str) -> str:
    match = _SIG_RE.search(text)
    if match:
        text = text[: match.start()]
    return text


def _normalize_whitespace(text: str) -> str:
    text = _MULTI_NEWLINES.sub("\n\n", text)
    text = _MULTI_SPACES.sub(" ", text)
    return text.strip()


def clean_body(body: str) -> str:
    body = _strip_html(body)
    body = _remove_signature(body)
    body = _normalize_whitespace(body)
    return body


def clean_email(email: Email) -> Email:
    return email.model_copy(update={"body": clean_body(email.body)})
