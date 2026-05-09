import json
import re
import logging
from typing import List

from models.schemas import Email, EmailChunk
import config.settings as cfg

logger = logging.getLogger(__name__)


def _split_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]


def _force_split(text: str, size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    step = max(1, size - overlap)
    while start < len(text):
        chunks.append(text[start : start + size])
        start += step
    return chunks


def chunk_text(
    text: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
    min_chunk_size: int = None,
) -> List[str]:
    size = cfg.CHUNK_SIZE if chunk_size is None else chunk_size
    overlap = cfg.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
    min_size = cfg.MIN_CHUNK_SIZE if min_chunk_size is None else min_chunk_size

    paragraphs = _split_paragraphs(text)
    raw_chunks: List[str] = []
    buffer = ""

    for para in paragraphs:
        if len(para) > size:
            if buffer:
                raw_chunks.append(buffer.strip())
                buffer = ""
            raw_chunks.extend(_force_split(para, size, overlap))
        elif len(buffer) + len(para) + 2 <= size:
            buffer = (buffer + "\n\n" + para).strip() if buffer else para
        else:
            if buffer:
                raw_chunks.append(buffer.strip())
            tail = buffer[-overlap:] if overlap and buffer else ""
            buffer = (tail + "\n\n" + para).strip() if tail else para

    if buffer:
        raw_chunks.append(buffer.strip())

    # merge chunks that are too short into the previous one
    merged: List[str] = []
    for chunk in raw_chunks:
        if len(chunk) < min_size and merged:
            merged[-1] = merged[-1] + "\n\n" + chunk
        else:
            merged.append(chunk)

    return merged


def chunk_email(email: Email) -> List[EmailChunk]:
    text = f"Subject: {email.subject}\n\n{email.body}"
    parts = chunk_text(text)
    chunks = []
    for i, part in enumerate(parts):
        chunks.append(
            EmailChunk(
                chunk_id=f"{email.id}_chunk_{i}",
                email_id=email.id,
                content=part,
                chunk_index=i,
                metadata={
                    "subject": email.subject,
                    "sender": email.sender,
                    "date": email.date,
                    # ChromaDB metadata values must be primitives; serialize lists
                    "labels": json.dumps(email.labels, ensure_ascii=False),
                    "recipients": json.dumps(email.recipients, ensure_ascii=False),
                    "thread_id": email.thread_id or "",
                },
            )
        )
    return chunks
