"""Per-message capture selection, with exact source identity and bounded reads."""
import hashlib
import json
import stat
from pathlib import Path, PurePosixPath

from agents.runtime import remaining_timeout
from agents.gmail_readonly import MailContentError


class CaptureSelectionError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


def read_json(path, *, limit=64 * 1024 * 1024):
    with Path(path).open('rb') as source:
        raw = source.read(limit + 1)
    if len(raw) > limit:
        raise CaptureSelectionError('capture_file_too_large')
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise CaptureSelectionError('invalid_capture_object')
    return value, hashlib.sha256(raw).hexdigest()


def safe_path(root, relative):
    if (not isinstance(relative, str) or not relative or '\\' in relative or ':' in relative
            or any(part in ('', '.', '..') for part in relative.split('/'))):
        raise CaptureSelectionError('unsafe_capture_path')
    rel = PurePosixPath(relative)
    if rel.is_absolute():
        raise CaptureSelectionError('unsafe_capture_path')
    root = Path(root).resolve()
    path = root.joinpath(*rel.parts)
    if not path.resolve().is_relative_to(root):
        raise CaptureSelectionError('unsafe_capture_path')
    node = path
    while node != root:
        attributes = getattr(node.lstat(), 'st_file_attributes', 0)
        if node.is_symlink() or attributes & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400):
            raise CaptureSelectionError('unsafe_capture_path')
        node = node.parent
    return path


def selection_record(raw_dir, root_file, capture_file, message_id):
    root = Path(raw_dir).resolve()
    root_name = Path(root_file).relative_to(root).as_posix()
    capture_name = Path(capture_file).relative_to(root).as_posix()
    _, root_hash = read_json(safe_path(root, root_name))
    _, capture_hash = read_json(safe_path(root, capture_name))
    return {'schema_version': 1, 'message_id': message_id, 'root_file': root_name,
            'root_sha256': root_hash, 'capture_file': capture_name, 'capture_sha256': capture_hash}


def selected_capture(raw_dir, root_file, envelope, root_hash, *, selection_dir=None):
    root = Path(raw_dir).resolve()
    directory = Path(selection_dir).resolve() if selection_dir else root / 'selections'
    if selection_dir is None and (directory.exists() or directory.is_symlink()):
        safe_path(root, 'selections')
    entry_path = directory / root_file.name
    if not entry_path.exists():
        if entry_path.is_symlink():
            raise CaptureSelectionError('unsafe_capture_path')
        if selection_dir is not None:
            raise CaptureSelectionError('capture_selection_missing')
        return root_file, envelope, root_hash, 'legacy_root'
    entry_path = safe_path(root, 'selections/' + root_file.name) if selection_dir is None else safe_path(directory, root_file.name)
    record, _ = read_json(entry_path, limit=16_384)
    message = envelope.get('message')
    if (not isinstance(message, dict) or type(record.get('schema_version')) is not int or record['schema_version'] != 1
            or record.get('root_file') != root_file.name or record.get('root_sha256') != root_hash
            or record.get('message_id') != message.get('id')):
        raise CaptureSelectionError('capture_selection_stale')
    name = record.get('capture_file')
    if not isinstance(name, str) or not (name == root_file.name or name.startswith('versions/')):
        raise CaptureSelectionError('unsafe_capture_path')
    path = safe_path(root, name)
    selected, digest = read_json(path)
    if digest != record.get('capture_sha256') or selected.get('message') != envelope.get('message'):
        raise CaptureSelectionError('capture_selection_integrity_failed')
    return path, selected, digest, 'manifest'


def _message_key(envelope):
    return hashlib.sha256(json.dumps(envelope.get('message'), ensure_ascii=False,
        sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


class LegacyVersions:
    """Index version messages once; never guess across different complete bodies."""
    def __init__(self, root):
        self.root, self.index = Path(root), None

    def find_complete(self, original, reader):
        if self.index is None:
            self.index = {}
            for candidate in sorted((self.root / 'versions').glob('*.json')):
                remaining_timeout(60)
                path = safe_path(self.root, candidate.relative_to(self.root).as_posix())
                envelope, digest = read_json(path)
                self.index.setdefault(_message_key(envelope), []).append((path, digest))
        choices = {}
        for path, original_digest in self.index.get(_message_key(original), []):
            remaining_timeout(60)
            envelope, digest = read_json(path)
            if digest != original_digest or envelope.get('message') != original.get('message'):
                raise CaptureSelectionError('capture_selection_integrity_failed')
            try:
                email = reader.email_from_capture(envelope)
            except (ValueError, MailContentError):
                continue
            # Capture timestamps can differ for equivalent content. Account and
            # original body bytes may not: ambiguity requires explicit selection.
            identity = (email.source.get('raw_sha256'), email.source.get('account_id'))
            choices.setdefault(identity, (path, envelope, digest, 'unique_legacy_version'))
        if len(choices) > 1:
            raise CaptureSelectionError('ambiguous_capture_versions')
        if not choices:
            raise CaptureSelectionError('missing_captured_body_data')
        return next(iter(choices.values()))
