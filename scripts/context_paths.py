"""Portable context-store path diagnostics; an actual write probe is opt-in."""
from pathlib import Path
import os
import tempfile


def check_context_path(value, *, probe=False, windows=None):
    path = Path(value).expanduser().absolute()
    windows = os.name == 'nt' if windows is None else windows
    # Allow the longest SQLite/atomic migration sidecar name used by context
    # stores. Windows long-path support requires more than a new Python alone.
    projected_length = len(str(path)) + 52
    status = 'warning' if windows and projected_length >= 260 else 'ok'
    ancestor = path.parent
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    writable = ancestor.is_dir() and os.access(ancestor, os.W_OK)
    result = {'status':status if writable else 'error', 'path_chars':len(str(path)),
              'projected_sidecar_chars':projected_length, 'write_probe':'not_requested',
              'long_path_support':'not_assumed', 'writable_parent':writable}
    if probe and writable:
        try:
            with tempfile.TemporaryFile(prefix='ctx-', dir=ancestor) as handle:
                handle.write(b'context-path-probe'); handle.flush(); os.fsync(handle.fileno())
            result['write_probe'] = 'passed'
        except OSError as exc:
            result.update(status='error', write_probe='failed', error_code=type(exc).__name__)
    return result
