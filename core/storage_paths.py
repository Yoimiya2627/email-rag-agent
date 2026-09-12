"""Filesystem constraints checked before opening native index storage."""
import os
from pathlib import Path


def validate_chroma_path(path, *, platform_name=None):
    resolved = Path(path).expanduser().resolve()
    if (os.name if platform_name is None else platform_name) == 'nt' and not str(resolved).isascii():
        raise ValueError('Windows Chroma storage requires an ASCII-only resolved path; '
                         'set CHROMA_PERSIST_DIR to a directory such as E:/email-agent-runtime/chroma_db')
    return resolved
