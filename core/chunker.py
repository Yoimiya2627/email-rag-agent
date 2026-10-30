import hashlib
import json
import re
from bisect import bisect_right
from typing import List

from models.schemas import Email, EmailChunk
import config.settings as cfg


def _validate_options(size: int, overlap: int, min_size: int) -> None:
    if any(isinstance(value, bool) or not isinstance(value, int)
           for value in (size, overlap, min_size)):
        raise ValueError("chunk options must be integers")
    if size <= 0 or not 0 <= overlap < size or min_size < 0:
        raise ValueError("require chunk_size > chunk_overlap >= 0 and min_chunk_size >= 0")


def _force_split(text: str, size: int, overlap: int) -> List[str]:
    _validate_options(size, overlap, 0)
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def _chunk_spans(text: str, size: int, overlap: int, min_size: int) -> list[tuple[int, int]]:
    """Slice the source without manufacturing separators or merging overlaps.

    Offsets, rather than matching text, distinguish intentional overlap from
    legitimate repeated words. A short tail merges by extending its predecessor.
    """
    _validate_options(size, overlap, min_size)
    if not text.strip():
        return []
    boundaries = [match.end() for match in re.finditer(r"\n\n+", text)]
    spans = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = bisect_right(boundaries, end) - 1
            if boundary >= 0 and boundaries[boundary] > start + overlap:
                end = boundaries[boundary]
        if end == len(text) and end - start < min_size and spans:
            spans[-1] = (spans[-1][0], end)
        else:
            spans.append((start, end))
        if end == len(text):
            break
        start = end - overlap
    return spans


def chunk_text(text: str, chunk_size: int = None, chunk_overlap: int = None,
               min_chunk_size: int = None) -> List[str]:
    size = cfg.CHUNK_SIZE if chunk_size is None else chunk_size
    overlap = cfg.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
    min_size = cfg.MIN_CHUNK_SIZE if min_chunk_size is None else min_chunk_size
    return [text[start:end] for start, end in _chunk_spans(text, size, overlap, min_size)]


def chunk_email(email: Email, *, chunk_size=None, chunk_overlap=None, min_chunk_size=None) -> List[EmailChunk]:
    size = cfg.CHUNK_SIZE if chunk_size is None else chunk_size
    overlap = cfg.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
    minimum = cfg.MIN_CHUNK_SIZE if min_chunk_size is None else min_chunk_size
    text = f"Subject: {email.subject}\n\n{email.body}"
    prefix = len(text) - len(email.body)
    rows = [{**row, 'start': row['start'] + prefix, 'end': row['end'] + prefix}
            for row in email.table_rows]
    if rows:
        _validate_options(size, overlap, minimum)
        spans = []
        cursor = 0
        for row in rows:
            if not cursor <= row['start'] < row['end'] <= len(text):
                raise ValueError('invalid or overlapping table row offsets')
            if cursor < row['start']:
                spans.extend((cursor + a, cursor + b) for a, b in
                             _chunk_spans(text[cursor:row['start']], size, overlap, 0)
                             or [(0, row['start'] - cursor)])
            # Each row is an atomic unit unless it exceeds the hard bound.
            length = row['end'] - row['start']
            for offset in range(0, length, size):
                spans.append((row['start'] + offset, min(row['end'], row['start'] + offset + size)))
            cursor = row['end']
        if cursor < len(text):
            spans.extend((cursor + a, cursor + b) for a, b in
                         _chunk_spans(text[cursor:], size, overlap, 0)
                         or [(0, len(text) - cursor)])
    else:
        spans = _chunk_spans(text, size, overlap, minimum)
    source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    row_ends = [row['end'] for row in rows]

    def intersecting_rows(start, end):
        # Row spans are ordered and disjoint. Avoid a full scan for every chunk,
        # including the separators between thousands of independent tables.
        index = bisect_right(row_ends, start)
        while index < len(rows) and rows[index]['start'] < end:
            yield rows[index]
            index += 1

    return [EmailChunk(
        chunk_id=f"{email.id}_chunk_{i}", email_id=email.id,
        content=text[start:end], chunk_index=i,
        metadata={
            "subject": email.subject, "sender": email.sender, "date": email.date,
            "labels": json.dumps(email.labels, ensure_ascii=False),
            "recipients": json.dumps(email.recipients, ensure_ascii=False),
            "thread_id": email.thread_id or "", "body_format": email.body_format,
            "sender_name": email.sender_name, "message_id": email.message_id, "in_reply_to": email.in_reply_to,
            **{key: json.dumps(getattr(email, key), ensure_ascii=False) for key in
               ("cc", "references", "label_names", "attachments", "source", "decode_quality")},
            "source_start": start, "source_end": end, "source_length": len(text),
            "source_sha256": source_hash,
            **({'table_context': json.dumps([
                {'table_id': row['table_id'], 'row_id': row['row_id'], 'status': row['status'],
                 'partial_row': start > row['start'] or end < row['end'],
                 'cells': [{key: cell[key] for key in ('source_id', 'row', 'column', 'rowspan', 'colspan', 'headers')}
                           for cell in row['cells']]}
                for row in intersecting_rows(start, end)], ensure_ascii=False)} if rows else {}),
        },
    ) for i, (start, end) in enumerate(spans)]
