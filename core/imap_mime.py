"""RFC822 IMAP parsing with explicit source identity and extraction coverage."""
from __future__ import annotations

import base64
import binascii
import hashlib
import json
import quopri
import re
from datetime import datetime, timezone
from email import policy
from email.header import decode_header
from email.parser import BytesParser
from email.utils import getaddresses, parsedate_to_datetime
from pathlib import Path

from core.attachment_text import MAX_TEXT_CHARS, decode_text, extract_attachment_text
from core.cleaner import HTML_INPUT_LIMIT, html_to_structured_text
from models.schemas import Email

MAX_RAW_BYTES = 25 * 1024 * 1024
MAX_PARTS = 256
MAX_DEPTH = 24
MAX_HEADER_CHARS = 128_000
MAX_BODY_CHARS = 800_000
MAX_ATTACHMENT_TEXT_TOTAL = 800_000


class MailParseError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def parser_version() -> str:
    from core import attachment_text, cleaner, html_tables
    from models import schemas
    digest = hashlib.sha256(b''.join(Path(module).read_bytes() for module in
        (__file__, attachment_text.__file__, cleaner.__file__, html_tables.__file__, schemas.__file__))).hexdigest()
    return 'imap-rfc822-v1:' + digest


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()


def _header(part, name, warnings):
    # compat32's get() wraps unencoded 8-bit headers and str() replaces bytes.
    # raw_items retains surrogate-escaped octets so fallback decoding is audited.
    values = [value for key, value in part.raw_items() if key.lower() == name]
    if len(values) > 1 and name not in {'to', 'cc'}:
        warnings.append('duplicate_header_first_used:' + name)
    raw = ', '.join(map(str, values)) if name in {'to', 'cc'} else str(values[0]) if values else ''
    pieces = []
    try:
        decoded = decode_header(raw)
    except (ValueError, LookupError):
        warnings.append('header_decode_failed:' + name)
        return raw
    for value, charset in decoded:
        if isinstance(value, bytes):
            text, issues, _ = decode_text(value, charset)
            warnings.extend(issue + ':' + name for issue in issues)
        else:
            text = value
            if any(0xDC80 <= ord(char) <= 0xDCFF for char in text):
                text, issues, _ = decode_text(text.encode('ascii', 'surrogateescape'))
                warnings.extend(issue + ':' + name for issue in issues)
        pieces.append(text)
        if '\ufffd' in text:
            warnings.append('replacement_character_present:' + name)
    return ''.join(pieces)


def _filename(part, warnings):
    value = part.get_param('filename', header='content-disposition')
    if value is None: value = part.get_param('name', header='content-type')
    if value is None: return ''
    if isinstance(value, tuple):
        charset, _, encoded = value
        try:
            text, issues, _ = decode_text(encoded.encode('latin-1'), charset)
        except UnicodeError:
            text, issues = encoded, ['filename_parameter_decode_failed']
        warnings.extend(issues)
    else:
        text = str(value)
    pieces = []
    try:
        for fragment, charset in decode_header(text):
            if isinstance(fragment, bytes):
                fragment, issues, _ = decode_text(fragment, charset)
                warnings.extend(issues)
            pieces.append(fragment)
        text = ''.join(pieces)
    except (LookupError, ValueError):
        warnings.append('filename_header_decode_failed')
    # This is a display name only; never a filesystem path.
    if len(text) > 1024:
        warnings.append('filename_display_limit')
        text = text[:1024]
    if '\ufffd' in text:
        warnings.append('filename_replacement_character_present')
    return text


def _payload_bytes(part):
    warnings = []
    payload = part.get_payload()
    if isinstance(payload, list):
        return b'\r\n'.join(item.as_bytes(policy=policy.SMTP) for item in payload), ['reconstructed_rfc822_bytes']
    if not isinstance(payload, str):
        return b'', ['missing_payload']
    try:
        encoded = payload.encode('ascii', 'surrogateescape')
    except UnicodeError:
        encoded = payload.encode('utf-8', 'replace')
        warnings.append('payload_reencoded')
    transfer = str(part.get('content-transfer-encoding', '7bit')).strip().lower()
    if transfer == 'base64':
        compact = re.sub(rb'[\t\r\n ]+', b'', encoded)
        try:
            return base64.b64decode(compact, validate=True), warnings
        except (ValueError, binascii.Error):
            warnings.append('invalid_base64')
            try:
                # Preserve a recoverable prefix, but never call it complete.
                cleaned = re.sub(rb'[^A-Za-z0-9+/=]', b'', compact)
                return base64.b64decode(cleaned + b'=' * (-len(cleaned) % 4)), warnings
            except (ValueError, binascii.Error):
                return b'', warnings + ['transfer_decode_failed']
    if transfer == 'quoted-printable':
        if re.search(rb'=(?![0-9A-Fa-f]{2}|\r?\n)', encoded):
            warnings.append('invalid_quoted_printable')
        return quopri.decodestring(encoded), warnings
    if transfer not in {'7bit', '8bit', 'binary', ''}:
        warnings.append('unsupported_transfer_encoding')
    elif transfer in {'7bit', ''} and any(value > 127 for value in encoded):
        warnings.append('non_ascii_in_7bit_payload')
    return encoded, warnings


def _date(value, internal_date, warnings):
    for candidate, origin in ((value, 'date_header'), (internal_date, 'imap_internaldate')):
        if not candidate: continue
        moment = None
        try:
            moment = parsedate_to_datetime(candidate)
        except (ValueError, TypeError, OverflowError):
            try:
                moment = datetime.fromisoformat(candidate.replace('Z', '+00:00'))
            except (ValueError, TypeError):
                try: moment = datetime.strptime(candidate.strip('"'), '%d-%b-%Y %H:%M:%S %z')
                except (ValueError, TypeError): pass
        if moment is not None and moment.tzinfo is not None:
            if origin != 'date_header': warnings.append('date_header_missing_or_invalid_used_internaldate')
            return moment.astimezone(timezone.utc).isoformat(), origin
        warnings.append('invalid_or_timezone_missing_' + origin)
    warnings.append('date_unavailable')
    return '', 'unavailable'


def parse_imap_message(raw: bytes, *, account_id: str, folder: str, uidvalidity: str,
                       uid: int, internal_date: str = '', flags=None) -> Email:
    """Parse local bytes only. Unsafe MIME structure fails with a stable code.

    Identity includes IMAP account/folder/UIDVALIDITY/UID, not mutable headers.
    References are account-scoped; exact raw MIME is archived by the caller.
    Attachment text is never appended to parent body. No network is performed.
    """
    if not isinstance(raw, bytes) or not raw:
        raise MailParseError('invalid_raw_message')
    if len(raw) > MAX_RAW_BYTES:
        raise MailParseError('raw_message_byte_limit')
    if (not isinstance(account_id, str) or not account_id.strip() or len(account_id) > 512
            or not isinstance(folder, str) or not folder or len(folder) > 1024
            or not isinstance(uidvalidity, str) or not uidvalidity.isascii() or not uidvalidity.isdecimal()
            or not 1 <= int(uidvalidity) <= 0xFFFFFFFF
            or type(uid) is not int or not 1 <= uid <= 0xFFFFFFFF):
        raise MailParseError('invalid_imap_identity')
    account = account_id.strip()
    folder_key = 'INBOX' if folder.upper() == 'INBOX' else folder
    validity = str(int(uidvalidity))
    if flags is None: flags = []
    if (not isinstance(internal_date, str) or not isinstance(flags, (list, tuple)) or len(flags) > 100
            or any(not isinstance(item, str) or len(item) > 200 for item in flags)):
        raise MailParseError('invalid_imap_metadata')
    try:
        message = BytesParser(policy=policy.compat32).parsebytes(raw)
    except (RecursionError, ValueError, IndexError) as exc:
        raise MailParseError('unsafe_mime_structure') from exc
    nodes, warnings = [], []
    stack, header_chars = [(message, '1', 0)], 0
    while stack:
        part, part_id, depth = stack.pop()
        nodes.append((part, part_id))
        if len(nodes) > MAX_PARTS: raise MailParseError('mime_part_limit')
        if depth > MAX_DEPTH: raise MailParseError('mime_depth_limit')
        header_chars += sum(len(name) + len(str(value)) for name, value in part.raw_items())
        if header_chars > MAX_HEADER_CHARS: raise MailParseError('mime_header_limit')
        for name in ('content-type', 'content-transfer-encoding', 'content-disposition'):
            if len(part.get_all(name, [])) > 1:
                raise MailParseError('ambiguous_mime_headers')
        for defect in part.defects:
            warnings.append('mime_defect:' + type(defect).__name__ + ':' + part_id)
        if part.is_multipart():
            children = part.get_payload()
            if len(children) + len(nodes) + len(stack) > MAX_PARTS:
                raise MailParseError('mime_part_limit')
            stack.extend((child, part_id + '.' + str(number), depth + 1)
                         for number, child in reversed(list(enumerate(children, 1))))
        elif part.get_content_maintype() == 'multipart':
            raise MailParseError('multipart_boundary_unreadable')
    paths = {id(part): part_id for part, part_id in nodes}
    filenames = {}
    filename_warnings = {}
    for part, part_id in nodes:
        issues = []
        filenames[part_id] = _filename(part, issues)
        filename_warnings[part_id] = issues
    def is_attachment(part):
        return (part.get_content_disposition() == 'attachment' or bool(filenames[paths[id(part)]])
                or part.get_content_type() == 'message/rfc822')
    selection = []
    def body_parts(part):
        if is_attachment(part): return []
        kind = part.get_content_type()
        if not part.is_multipart():
            return [part] if kind in {'text/plain', 'text/html'} else []
        children = part.get_payload()
        if kind == 'multipart/alternative':
            candidates = [(child, body_parts(child)) for child in children]
            candidates = [(child, pieces) for child, pieces in candidates if pieces]
            if not candidates: return []
            preferred = [candidate for candidate in candidates
                         if any(piece.get_content_type() == 'text/html' for piece in candidate[1])]
            child, pieces = (preferred or candidates)[-1]
            selection.append({'container': paths[id(part)], 'selected': paths[id(child)],
                              'reason': 'html_preferred_preserves_tables_single_alternative'})
            return pieces
        if kind == 'multipart/related':
            start = part.get_param('start')
            matches = [child for child in children if str(child.get('content-id', '')).strip() == str(start).strip()]
            if start and not matches:
                warnings.append('related_root_missing:' + paths[id(part)])
            child = matches[0] if matches else children[0] if children else None
            return body_parts(child) if child is not None else []
        return [leaf for child in children for leaf in body_parts(child)]
    selected = body_parts(message)
    selected_ids = {id(part) for part in selected}
    chunks, table_rows, body_info = [], [], []
    length = 0
    for part in selected:
        part_id = paths[id(part)]
        data, issues = _payload_bytes(part)
        text, decode_issues, codec = decode_text(data, part.get_content_charset())
        issues.extend(decode_issues)
        rows = []
        if part.get_content_type() == 'text/html':
            if len(text) > HTML_INPUT_LIMIT: issues.append('html_input_limit')
            text, rows = html_to_structured_text(text, table_prefix='p' + _digest(part_id)[:12] + 't')
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        offset = length + (2 if chunks else 0)
        remaining = max(0, MAX_BODY_CHARS - offset)
        if len(text) > remaining:
            text = text[:remaining]
            issues.append('body_text_limit')
        if text:
            chunks.append(text)
            table_rows.extend({**row, 'start': row['start'] + offset, 'end': row['end'] + offset,
                               'part_id': part_id} for row in rows if row['end'] <= len(text))
            length = offset + len(text)
        body_info.append({'part_id': part_id, 'mime_type': part.get_content_type(), 'charset_used': codec,
                          'decoded_size': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                          'status': 'partial' if issues else 'complete', 'warnings': issues})
        warnings.extend(issue + ':' + part_id for issue in issues)
    attachments = []
    attachment_text_remaining = MAX_ATTACHMENT_TEXT_TOTAL
    def inventory(part, resource=False):
        nonlocal attachment_text_remaining
        part_id, kind = paths[id(part)], part.get_content_type()
        attached = is_attachment(part)
        if attached or (not part.is_multipart() and id(part) not in selected_ids
                        and (resource or kind not in {'text/plain', 'text/html'})):
            data, issues = _payload_bytes(part)
            issues.extend(filename_warnings[part_id])
            if attachment_text_remaining > 0:
                extracted = extract_attachment_text(data, filename=filenames[part_id], mime_type=kind,
                    charset=part.get_content_charset(), max_chars=min(MAX_TEXT_CHARS, attachment_text_remaining))
            else:
                extracted = {'status': 'not_read', 'reason': 'message_attachment_text_limit', 'text': '',
                             'locations': [], 'warnings': [], 'text_extracted': False}
            attachment_text_remaining -= len(extracted['text'])
            extracted['warnings'] = list(dict.fromkeys(issues + extracted.get('warnings', [])))
            if issues and extracted['status'] == 'complete':
                extracted.update(status='partial', reason='mime_decode_warning')
            attachments.append({**extracted, 'part_id': part_id, 'filename': filenames[part_id], 'mime_type': kind,
                'size': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                'hash_basis': 'reconstructed_rfc822_bytes' if part.is_multipart() else 'decoded_attachment_bytes',
                'disposition': part.get_content_disposition() or '', 'content_id': str(part.get('content-id', '')),
                'role': 'forwarded_message' if kind == 'message/rfc822' else 'attachment' if attached else 'inline_resource'})
            return
        if part.is_multipart():
            for child in part.get_payload():
                child_path = paths[id(child)]
                contains_body = any(paths[id(selected_part)] == child_path or
                                    paths[id(selected_part)].startswith(child_path + '.') for selected_part in selected)
                inventory(child, resource or (kind == 'multipart/related' and not contains_body))
    inventory(message)
    subject = _header(message, 'subject', warnings)
    from_value = _header(message, 'from', warnings)
    from_addresses = getaddresses([from_value])
    if len(from_addresses) > 1: warnings.append('multiple_from_addresses_first_used')
    to_values = [_header(message, 'to', warnings)]
    cc_values = [_header(message, 'cc', warnings)]
    raw_message_id = str(message.get('message-id', '')).strip()
    raw_reply = re.findall(r'<[^<>\s]+>', ' '.join(map(str, message.get_all('in-reply-to', []))))
    raw_references = re.findall(r'<[^<>\s]+>', ' '.join(map(str, message.get_all('references', []))))
    def reference(value):
        return 'imap-msg-' + _digest([account, value.strip()]) if value else ''
    valid_id = re.fullmatch(r'<[^<>\s]+>', raw_message_id)
    if raw_message_id and not valid_id: warnings.append('invalid_message_id')
    message_ref = reference(raw_message_id) if valid_id else ''
    references = [reference(value) for value in raw_references]
    in_reply_to = reference(raw_reply[0]) if raw_reply else ''
    email_id = 'imap-' + _digest([account, folder_key, validity, uid])
    thread_id = references[0] if references else in_reply_to or message_ref or email_id
    date, date_origin = _date(_header(message, 'date', warnings), internal_date, warnings)
    warnings = list(dict.fromkeys(warnings))
    body = '\n\n'.join(chunks)
    body_warnings = [issue for item in body_info for issue in item['warnings']]
    return Email(id=email_id, subject=subject, sender=from_addresses[0][1] if from_addresses else '',
        sender_name=from_addresses[0][0] if from_addresses else '', recipients=[address for _, address in getaddresses(to_values) if address],
        cc=[address for _, address in getaddresses(cc_values) if address], date=date, body=body, body_format='plain',
        table_rows=table_rows, labels=list(flags), label_names=list(flags), thread_id=thread_id,
        message_id=message_ref, in_reply_to=in_reply_to, references=references, attachments=attachments,
        source={'provider': 'imap', 'format': 'rfc822', 'account_id': account, 'folder': folder,
            'uidvalidity': validity, 'uid': uid, 'raw_sha256': hashlib.sha256(raw).hexdigest(), 'raw_size': len(raw),
            'parser_version': parser_version(), 'internal_date': internal_date, 'date_origin': date_origin,
            'raw_message_id': raw_message_id, 'raw_in_reply_to': raw_reply, 'raw_references': raw_references,
            'body_parts': body_info, 'body_selection': selection, 'mime_part_count': len(nodes),
            'body_status': 'not_read' if not selected else 'partial' if body_warnings or
                any(issue.startswith('mime_defect:') for issue in warnings) else 'complete',
            'attachment_inventory_status': 'complete', 'warnings': warnings,
            'coverage': 'selected_mime_body_and_separate_attachment_inventory'},
        decode_quality={'status': 'suspect' if warnings else 'ok', 'warnings': warnings,
                        'body': {'status': 'suspect' if body_warnings else 'ok', 'warnings': body_warnings},
                        'subject': {'status': 'suspect' if any(issue.endswith(':subject') for issue in warnings) else 'ok'}})
