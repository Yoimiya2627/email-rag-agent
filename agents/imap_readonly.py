"""163 IMAP reader: verified TLS, EXAMINE, and bounded UID BODY.PEEK only.

This transport does not parse MIME, persist mail, or change mailbox state.
Folder ``name`` values are IMAP modified UTF-7 wire names; ``display_name`` is
Unicode. Exceptions contain stable codes, never server text or credentials.
"""
from __future__ import annotations

import base64
import binascii
import imaplib
import math
import re
import ssl


class ImapReadError(RuntimeError):
    def __init__(self, code='imap_read_failed'):
        self.code = code if isinstance(code, str) and re.fullmatch(r'[a-z_]{1,64}', code) else 'imap_read_failed'
        super().__init__(self.code)


class ImapMessageMissing(ImapReadError):
    def __init__(self):
        super().__init__('message_missing')


class ImapMessageTooLarge(ImapReadError):
    def __init__(self):
        super().__init__('message_too_large')


class _BoundedIMAP4SSL(imaplib.IMAP4_SSL):
    def __init__(self, *args, max_literal_bytes, **kwargs):
        self._max_literal_bytes = max_literal_bytes
        super().__init__(*args, **kwargs)

    def read(self, size):
        # imaplib calls read with the server's literal count before allocation.
        if size > self._max_literal_bytes:
            raise ImapMessageTooLarge()
        return super().read(size)


def _encode_utf7(value):
    pieces, pending = [], []

    def flush():
        if pending:
            encoded = base64.b64encode(''.join(pending).encode('utf-16-be')).decode('ascii')
            pieces.append('&' + encoded.rstrip('=').replace('/', ',') + '-')
            pending.clear()

    for char in value:
        if ' ' <= char <= '~':
            flush()
            pieces.append('&-' if char == '&' else char)
        else:
            pending.append(char)
    flush()
    return ''.join(pieces)


def _decode_utf7(value):
    pieces, index = [], 0
    try:
        value.encode('ascii')
        while index < len(value):
            if value[index] != '&':
                pieces.append(value[index]); index += 1
                continue
            end = value.find('-', index + 1)
            if end < 0:
                raise ValueError()
            encoded = value[index+1:end]
            if not encoded:
                pieces.append('&')
            else:
                if not re.fullmatch(r'[A-Za-z0-9+,]+', encoded):
                    raise ValueError()
                raw = base64.b64decode(encoded.replace(',', '/') + '=' * (-len(encoded) % 4), validate=True)
                pieces.append(raw.decode('utf-16-be'))
            index = end + 1
        result = ''.join(pieces)
        if _encode_utf7(result) != value or any(ord(c) < 32 or ord(c) == 127 for c in result):
            raise ValueError()
        return result
    except (UnicodeError, ValueError, binascii.Error):
        raise ImapReadError('invalid_folder_encoding') from None


def _quoted_folder(name):
    if not isinstance(name, str) or not name or len(name) > 4096 or any(ord(c) < 32 or ord(c) == 127 for c in name):
        raise ImapReadError('invalid_folder')
    try:
        name.encode('ascii')
    except UnicodeError:
        try:
            name = _encode_utf7(name)
        except UnicodeError:
            raise ImapReadError('invalid_folder_encoding') from None
    else:
        _decode_utf7(name)  # ASCII names supplied by LIST must be valid wire names.
    return '"' + name.replace('\\', '\\\\').replace('"', '\\"') + '"'


def _unquote(value):
    if value.startswith(b'"'):
        if not re.fullmatch(rb'"(?:[^"\\\r\n]|\\["\\])*"', value):
            raise ImapReadError('invalid_folder_list')
        return re.sub(rb'\\(["\\])', rb'\1', value[1:-1])
    if not value or any(c in value for c in (b' ', b'\r', b'\n', b'(', b')', b'{')):
        raise ImapReadError('invalid_folder_list')
    return value


_LIST = re.compile(rb'^\(([^)]*)\) +(?:NIL|"(?:[^"\\]|\\.)*"|[^ ]+) +(.+)$')
_UID = re.compile(rb'\bUID +(\d+)\b', re.I)
_SIZE = re.compile(rb'\bRFC822\.SIZE +(\d+)\b', re.I)
_DATE = re.compile(rb'\bINTERNALDATE +"([^"\r\n]*)"', re.I)
_FLAGS = re.compile(rb'\bFLAGS +\(([^)]*)\)', re.I)


def _decimal(value, *, code, max_digits=10):
    if not isinstance(value, bytes) or not value.isdigit() or len(value) > max_digits:
        raise ImapReadError(code)
    return int(value)


class ImapReadOnlyProvider:
    provider = 'imap_163_readonly'
    host, port = 'imap.163.com', 993

    def __init__(self, address, authorization_code, *, timeout=30, max_message_bytes=25_000_000, client_factory=None):
        if (not isinstance(address, str) or len(address) > 254
                or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._+\-]*@163\.com', address, re.I)):
            raise ImapReadError('invalid_account')
        if (not isinstance(authorization_code, str) or not authorization_code
                or len(authorization_code) > 1024 or any(ord(c) < 32 or ord(c) == 127 for c in authorization_code)):
            raise ImapReadError('invalid_authorization_code')
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ImapReadError('invalid_timeout')
        if type(max_message_bytes) is not int or not 1 <= max_message_bytes <= 100_000_000:
            raise ImapReadError('invalid_size_limit')
        self.address = address.casefold()
        self._authorization_code = authorization_code
        self.timeout, self.max_message_bytes = timeout, max_message_bytes
        self._factory, self._client, self._selected = client_factory, None, None
        self._capabilities = set()

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.close()

    def _call(self, code, function, *args, **kwargs):
        try:
            response = function(*args, **kwargs)
        except ImapReadError:
            self.close()
            raise
        except Exception:
            self.close()
            raise ImapReadError(code) from None
        if not isinstance(response, tuple) or len(response) != 2 or str(response[0]).upper() != 'OK':
            raise ImapReadError(code)
        if not isinstance(response[1], (list, tuple)):
            raise ImapReadError('invalid_response')
        return response[1]

    def connect(self):
        if self._client is not None:
            return self
        try:
            options = {'ssl_context':ssl.create_default_context(), 'timeout':self.timeout}
            if self._factory is None:
                self._client = _BoundedIMAP4SSL(self.host, self.port,
                    max_literal_bytes=self.max_message_bytes + 1, **options)
            else:
                self._client = self._factory(self.host, self.port, **options)
            self._call('authentication_failed', self._client.login, self.address, self._authorization_code)
            values = self._call('capability_failed', self._client.capability)
            if any(not isinstance(row, bytes) for row in values):
                raise ImapReadError('invalid_capabilities')
            self._capabilities = set(b' '.join(values).upper().split())
            if b'ID' in self._capabilities:
                self._call('client_id_failed', self._client.xatom, 'ID', '("name" "Email RAG Agent" "version" "1.0")')
            return self
        except ImapReadError:
            self.close()
            raise
        except Exception:
            self.close()
            raise ImapReadError('connection_failed') from None

    def close(self):
        client, self._client, self._selected = self._client, None, None
        if client is not None:
            try:
                client.logout()  # Never CLOSE: it may expunge a writable mailbox.
            except Exception:
                try:
                    client.shutdown()
                except Exception:
                    pass

    def _connected(self):
        if self._client is None:
            raise ImapReadError('not_connected')
        return self._client

    def describe_account(self):
        self._connected()
        return {'account_id':self.address, 'provider':self.provider, 'host':self.host, 'port':self.port, 'read_only':True}

    def list_folders(self):
        client = self._connected()
        rows = self._call('folder_list_failed', client.list, '""', '"*"')
        folders, seen = [], set()
        for row in rows:
            if row in (None, b''):
                continue
            literal = None
            if isinstance(row, tuple) and len(row) == 2:
                row, literal = row
            if not isinstance(row, bytes) or len(row) > 16384:
                raise ImapReadError('invalid_folder_list')
            match = _LIST.fullmatch(row)
            if match is None:
                raise ImapReadError('invalid_folder_list')
            flags, value = match.groups()
            if literal is not None:
                size = re.fullmatch(rb'\{(\d+)\}', value)
                if not size or not isinstance(literal, bytes) or len(literal) != _decimal(size[1], code='invalid_folder_list'):
                    raise ImapReadError('invalid_folder_list')
                value = literal
            else:
                value = _unquote(value)
            try:
                name = value.decode('ascii')
                flag_names = [flag.decode('ascii') for flag in flags.split()]
            except UnicodeError:
                raise ImapReadError('invalid_folder_encoding') from None
            display = _decode_utf7(name)
            _quoted_folder(name)
            if name in seen:
                continue
            seen.add(name)
            folders.append({'name':name, 'display_name':display, 'selectable':not any(flag.casefold() == '\\noselect' for flag in flag_names), 'flags':flag_names})
            if len(folders) > 10000:
                raise ImapReadError('too_many_folders')
        return folders

    def select_folder(self, name):
        client = self._connected()
        quoted = _quoted_folder(name)
        self._selected = None
        data = self._call('folder_select_failed', client.select, quoted, readonly=True)
        try:
            if len(data) != 1 or not isinstance(data[0], bytes) or not data[0].isdigit():
                raise ValueError()
            exists = int(data[0])
            response = client.response('UIDVALIDITY')
            if (not isinstance(response, tuple) or len(response) != 2 or str(response[0]).upper() != 'UIDVALIDITY'
                    or len(response[1]) != 1 or not isinstance(response[1][0], bytes) or not response[1][0].isdigit()):
                raise ValueError()
            validity = int(response[1][0])
            if not 1 <= validity <= 2**32-1:
                raise ValueError()
        except Exception:
            raise ImapReadError('invalid_mailbox_state') from None
        self._selected = {'uidvalidity':str(validity), 'exists':exists}
        return dict(self._selected)

    def _mailbox(self):
        client = self._connected()
        if self._selected is None:
            raise ImapReadError('folder_not_selected')
        return client

    def list_uids(self):
        rows = self._call('uid_search_failed', self._mailbox().uid, 'SEARCH', None, 'ALL')
        if len(rows) != 1 or not isinstance(rows[0], bytes):
            raise ImapReadError('invalid_uid_list')
        tokens = rows[0].split()
        if len(tokens) > 1_000_000:
            raise ImapReadError('invalid_uid_list')
        uids = [_decimal(token, code='invalid_uid_list') for token in tokens]
        if any(not 1 <= uid <= 2**32-1 for uid in uids):
            raise ImapReadError('invalid_uid_list')
        return sorted(set(uids))

    def _fetch_parts(self, rows, uid, *, body, allow_reported_size_mismatch=False, full_body=False):
        headers, literals = [], []
        for row in rows:
            if row in (None, b'', b')'):
                continue
            if isinstance(row, tuple) and len(row) == 2 and all(isinstance(part, bytes) for part in row):
                header, literal = row
                if not body:
                    raise ImapReadError('unexpected_literal')
                literals.append(literal)
                headers.append(header)
            elif isinstance(row, bytes):
                headers.append(row)
            else:
                raise ImapReadError('invalid_fetch_response')
        if not headers:
            raise ImapMessageMissing()
        envelope = b' '.join(headers)
        uids = _UID.findall(envelope)
        if not uids or any(_decimal(value, code='unexpected_uid') != uid for value in uids):
            raise ImapReadError('unexpected_uid')
        sizes, dates, flags = _SIZE.findall(envelope), _DATE.findall(envelope), _FLAGS.findall(envelope)
        if len(sizes) != 1 or len(dates) != 1 or len(flags) != 1:
            raise ImapReadError('incomplete_fetch_metadata')
        size = _decimal(sizes[0], code='invalid_message_size')
        if size > self.max_message_bytes:
            raise ImapMessageTooLarge()
        if size <= 0:
            raise ImapReadError('invalid_message_size')
        try:
            date = dates[0].decode('ascii')
            flag_names = [flag.decode('ascii') for flag in flags[0].split()]
            if not re.fullmatch(r' ?\d{1,2}-[A-Za-z]{3}-\d{4} \d{2}:\d{2}:\d{2} [+-]\d{4}', date):
                raise ValueError()
        except (UnicodeError, ValueError):
            raise ImapReadError('invalid_fetch_metadata') from None
        raw = None
        if body:
            if len(literals) != 1:
                raise ImapReadError('incomplete_message')
            raw = literals[0]
            if len(raw) > self.max_message_bytes:
                raise ImapMessageTooLarge()
            marker_pattern = rb'BODY\[\] +\{(\d+)\}' if full_body else rb'BODY\[\](?:<0>)? +\{(\d+)\}'
            markers = re.findall(marker_pattern, envelope, re.I)
            if (len(markers) != 1 or _decimal(markers[0], code='incomplete_message') != len(raw)
                    or not raw or (not allow_reported_size_mismatch and len(raw) != size)):
                raise ImapReadError('incomplete_message')
        return {'uid':uid, 'raw':raw, 'size':size, 'internal_date':date, 'flags':flag_names}

    def fetch_message(self, uid):
        if type(uid) is not int or not 1 <= uid <= 2**32-1:
            raise ImapReadError('invalid_uid')
        client = self._mailbox()
        fields = '(UID RFC822.SIZE INTERNALDATE FLAGS)'
        before = self._fetch_parts(self._call('message_fetch_failed', client.uid, 'FETCH', str(uid), fields), uid, body=False)
        fields = f'(UID RFC822.SIZE INTERNALDATE FLAGS BODY.PEEK[]<0.{before["size"]+1}>)'
        after = self._fetch_parts(self._call('message_fetch_failed', client.uid, 'FETCH', str(uid), fields),
                                 uid, body=True, allow_reported_size_mismatch=True)
        if after['size'] != before['size'] or after['internal_date'] != before['internal_date']:
            raise ImapReadError('message_changed_during_fetch')
        if len(after['raw']) != after['size']:
            # Some 163 records overstate RFC822.SIZE. Never accept a shortened
            # ranged response on that assumption: independently request the
            # entire BODY, require its non-partial marker and exact literal
            # framing, and verify that it contains the previously read bytes.
            # The real client's allocation guard still caps this full literal.
            fields = '(UID RFC822.SIZE INTERNALDATE FLAGS BODY.PEEK[])'
            complete = self._fetch_parts(self._call('message_fetch_failed', client.uid, 'FETCH', str(uid), fields),
                                        uid, body=True, allow_reported_size_mismatch=True, full_body=True)
            if (complete['size'] != before['size'] or complete['internal_date'] != before['internal_date']
                    or not complete['raw'].startswith(after['raw'])):
                raise ImapReadError('message_changed_during_fetch')
            complete.update(size=len(complete['raw']), reported_size=before['size'],
                            size_mismatch=len(complete['raw']) != before['size'])
            after = complete
        return after
