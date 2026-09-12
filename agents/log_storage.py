"""Bounded local JSONL storage for the supported single-process service."""
from __future__ import annotations

import json
from pathlib import Path
import threading

_LOCK = threading.RLock()


def append_jsonl(path, row, *, max_bytes=5_000_000, backups=3):
    if type(max_bytes) is not int or max_bytes<256 or type(backups) is not int or not 1<=backups<=20:
        raise ValueError('invalid log rotation budget')
    path = Path(path)
    encoded = (json.dumps(row,ensure_ascii=False,default=str)+'\n').encode('utf-8')
    if len(encoded)>max_bytes:
        raise ValueError('event exceeds log byte budget')
    with _LOCK:
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists() and path.stat().st_size+len(encoded)>max_bytes:
            oldest=path.with_name(path.name+f'.{backups}')
            oldest.unlink(missing_ok=True)
            for index in range(backups-1,0,-1):
                source=path.with_name(path.name+f'.{index}')
                if source.exists(): source.replace(path.with_name(path.name+f'.{index+1}'))
            path.replace(path.with_name(path.name+'.1'))
        with path.open('ab') as target: target.write(encoded)


def read_jsonl_tail(path, *, limit=1000, max_bytes=1_000_000, predicate=None):
    """Read only the active file's byte-bounded suffix, oldest to newest.

    Filters apply inside this suffix; fewer than limit matches is expected.
    Truncated/invalid lines are ignored. Rotated history is not auto-loaded.
    """
    if type(limit) is not int or not 1<=limit<=10000 or type(max_bytes) is not int or not 1<=max_bytes<=10_000_000:
        raise ValueError('invalid log read budget')
    path=Path(path)
    with _LOCK:
        try:
            with path.open('rb') as source:
                source.seek(0,2)
                size=source.tell()
                start=max(0,size-max_bytes)
                at_boundary = not start
                if start:
                    source.seek(start-1)
                    at_boundary = source.read(1) in {b'\n',b'\r'}
                source.seek(start)
                data=source.read(max_bytes)
        except FileNotFoundError:
            return []
    lines=data.splitlines()
    if not at_boundary and lines: lines=lines[1:]
    rows=[]
    for line in lines:
        try: row=json.loads(line)
        except (ValueError,UnicodeDecodeError): continue
        if isinstance(row,dict) and (predicate is None or predicate(row)):
            rows.append(row)
    return rows[-limit:]
