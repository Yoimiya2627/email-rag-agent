"""Offline protocol contracts; fake mail and clients only."""
import imaplib
import ssl

import pytest

from agents.imap_readonly import (
    ImapReadOnlyProvider, ImapReadError, ImapMessageMissing,
    ImapMessageTooLarge, _BoundedIMAP4SSL,
)


RAW = b'Subject: synthetic\r\n\r\nSynthetic body.'
DATE = b'12-Sep-2026 12:30:00 +0800'


def metadata(uid=42, size=None, date=DATE):
    size = len(RAW) if size is None else size
    return b'1 (UID %d RFC822.SIZE %d INTERNALDATE "%s" FLAGS ())' % (uid, size, date)


def body_rows(raw=RAW, uid=42, size=None, date=DATE):
    head = metadata(uid, size, date)[:-1]
    return [(head + b' BODY[]<0> {%d}' % len(raw), raw), b')']


class FakeClient:
    def __init__(self):
        self.trace = []
        self.capabilities = b'IMAP4rev1 ID'
        self.folders = [b'(\\HasNoChildren) "/" "INBOX"']
        self.validity = ('UIDVALIDITY', [b'123'])
        self.search = [b'42']
        self.fetch = [[metadata()], body_rows()]
        self.failures = {}

    def record(self, command, *args, **kwargs):
        self.trace.append((command, args, kwargs))
        error = self.failures.get(command)
        if error:
            raise error

    def login(self, address, code):
        self.record('LOGIN', address, '<redacted>')
        return 'OK', [b'authenticated']

    def capability(self):
        self.record('CAPABILITY')
        return 'OK', [self.capabilities]

    def xatom(self, command, *args):
        self.record(command, *args)
        return 'OK', [b'ID completed']

    def list(self, *args):
        self.record('LIST', *args)
        return 'OK', self.folders

    def select(self, *args, **kwargs):
        self.record('EXAMINE' if kwargs.get('readonly') else 'SELECT', *args, **kwargs)
        return 'OK', [b'1']

    def response(self, code):
        self.record('RESPONSE', code)
        return self.validity

    def uid(self, command, *args):
        self.record('UID ' + command, *args)
        return 'OK', self.search if command == 'SEARCH' else self.fetch.pop(0)

    def logout(self):
        self.record('LOGOUT')
        return 'BYE', [b'goodbye']

    def shutdown(self):
        self.record('SHUTDOWN')


@pytest.fixture
def connected():
    fake = FakeClient()
    provider = ImapReadOnlyProvider('reader@163.com', 'synthetic-code', client_factory=lambda *a, **k: fake)
    provider.connect()
    yield provider, fake
    provider.close()


def test_verified_tls_and_readonly_command_trace():
    fake, construction = FakeClient(), []

    def factory(*args, **kwargs):
        construction.append((args, kwargs))
        return fake

    with ImapReadOnlyProvider('Reader@163.com', 'synthetic-code', timeout=7, client_factory=factory) as provider:
        assert provider.connect() is provider
        assert provider.describe_account()['read_only'] is True
        assert provider.describe_account()['account_id'] == 'reader@163.com'
        assert provider.list_folders()[0]['name'] == 'INBOX'
        assert provider.select_folder('INBOX') == {'uidvalidity': '123', 'exists': 1}
        assert provider.list_uids() == [42]
        assert provider.fetch_message(42) == dict(uid=42, raw=RAW, size=len(RAW), internal_date=DATE.decode(), flags=[])
    args, kwargs = construction[0]
    assert args == ('imap.163.com', 993)
    assert kwargs['timeout'] == 7
    assert kwargs['ssl_context'].check_hostname
    assert kwargs['ssl_context'].verify_mode == ssl.CERT_REQUIRED
    commands = [entry[0] for entry in fake.trace]
    assert commands == ['LOGIN', 'CAPABILITY', 'ID', 'LIST', 'EXAMINE', 'RESPONSE', 'UID SEARCH', 'UID FETCH', 'UID FETCH', 'LOGOUT']
    assert fake.trace[2][1] == ('("name" "Email RAG Agent" "version" "1.0")',)
    assert fake.trace[4][1:] == (('"INBOX"',), {'readonly': True})
    assert fake.trace[7][1] == ('42', '(UID RFC822.SIZE INTERNALDATE FLAGS)')
    assert fake.trace[8][1] == ('42', f'(UID RFC822.SIZE INTERNALDATE FLAGS BODY.PEEK[]<0.{len(RAW)+1}>)')
    provider.close()
    assert [e[0] for e in fake.trace].count('LOGOUT') == 1


def test_no_id_without_advertised_capability():
    fake = FakeClient()
    fake.capabilities = b'IMAP4rev1'
    with ImapReadOnlyProvider('reader@163.com', 'synthetic-code', client_factory=lambda *a, **k: fake):
        pass
    assert [e[0] for e in fake.trace] == ['LOGIN', 'CAPABILITY', 'LOGOUT']


def test_modified_utf7_quoted_and_literal_folder_names(connected):
    provider, fake = connected
    fake.folders = [
        b'(\\NoSelect) "/" "&U,BTFw-"',
        b'() "/" "A &- B"',
        b'() NIL "Quoted \\"name\\""',
        (b'() "/" {11}', b'Two folders'),
        b'() "/" "INBOX"', b'() "/" "INBOX"',
    ]
    folders = provider.list_folders()
    assert len(folders) == 5
    assert folders[0] == dict(name='&U,BTFw-', display_name='台北', selectable=False, flags=['\\NoSelect'])
    assert folders[1]['display_name'] == 'A & B'
    assert folders[2]['name'] == 'Quoted "name"'
    assert folders[3]['display_name'] == 'Two folders'
    provider.select_folder('台北')
    assert fake.trace[-2][1] == ('"&U,BTFw-"',)
    provider.select_folder(folders[2]['name'])
    assert fake.trace[-2][1] == ('"Quoted \\"name\\""',)


@pytest.mark.parametrize('name', ['INBOX\r\nSTORE 1 +FLAGS (\\Seen)', '', '\x00', '&invalid', '&AGE-', '\ud800'])
def test_invalid_folder_never_reaches_server(connected, name):
    provider, fake = connected
    before = list(fake.trace)
    with pytest.raises(ImapReadError):
        provider.select_folder(name)
    assert fake.trace == before


@pytest.mark.parametrize('row', [b'garbage', b'() "/" "&oops"', b'() "/" "bad\x00name"', (b'() "/" {9}', b'bad')])
def test_rejects_malformed_list(connected, row):
    provider, fake = connected
    fake.folders = [row]
    with pytest.raises(ImapReadError):
        provider.list_folders()


@pytest.mark.parametrize('uid', [True, 0, -1, 2**32, '42', '42:*', None])
def test_invalid_uid_no_command(connected, uid):
    provider, fake = connected
    before = list(fake.trace)
    with pytest.raises(ImapReadError, match='invalid_uid'):
        provider.fetch_message(uid)
    assert fake.trace == before


@pytest.mark.parametrize('data, expected', [(b'', []), (b'43 42 42', [42, 43])])
def test_uid_list_normalization(connected, data, expected):
    provider, fake = connected
    provider.select_folder('INBOX')
    fake.search = [data]
    assert provider.list_uids() == expected


@pytest.mark.parametrize('data', [[None], [b'0'], [b'4294967296'], [b'1:*'], [b'1', b'2']])
def test_invalid_uid_list(connected, data):
    provider, fake = connected
    provider.select_folder('INBOX')
    fake.search = data
    with pytest.raises(ImapReadError, match='invalid_uid_list'):
        provider.list_uids()


def test_invalid_uidvalidity_disables_previous_selection(connected):
    provider, fake = connected
    provider.select_folder('INBOX')
    fake.validity = ('UIDVALIDITY', [None])
    with pytest.raises(ImapReadError, match='invalid_mailbox_state'):
        provider.select_folder('Other')
    with pytest.raises(ImapReadError, match='folder_not_selected'):
        provider.list_uids()


@pytest.mark.parametrize('rows', [[], [None]])
def test_missing_uid(connected, rows):
    provider, fake = connected
    provider.select_folder('INBOX')
    fake.fetch = [rows]
    with pytest.raises(ImapMessageMissing):
        provider.fetch_message(42)


@pytest.mark.parametrize('stage', [0, 1])
def test_wrong_uid_rejected_at_each_stage(connected, stage):
    provider, fake = connected
    provider.select_folder('INBOX')
    fake.fetch[stage] = [metadata(uid=43)] if stage == 0 else body_rows(uid=43)
    with pytest.raises(ImapReadError, match='unexpected_uid'):
        provider.fetch_message(42)


def test_size_precheck_prevents_body_download(connected):
    provider, fake = connected
    provider.select_folder('INBOX')
    fake.fetch = [[metadata(size=provider.max_message_bytes + 1)]]
    with pytest.raises(ImapMessageTooLarge):
        provider.fetch_message(42)
    assert len([e for e in fake.trace if e[0] == 'UID FETCH']) == 1


@pytest.mark.parametrize('rows', [body_rows(raw=RAW[:-1]), [metadata()], [(b'1 (UID 42 BODY[] {5}', b'hello'), b')'],
                               [(metadata()[:-1] + b' BODY[] {999}', RAW), b')']])
def test_partial_or_incomplete_fetch_rejected(connected, rows):
    provider, fake = connected
    provider.select_folder('INBOX')
    fake.fetch[1] = rows
    # A repeated partial/ranged response cannot prove full-body completion.
    fake.fetch.append(rows)
    with pytest.raises(ImapReadError, match='incomplete_'):
        provider.fetch_message(42)


def test_changed_message_rejected(connected):
    provider, fake = connected
    provider.select_folder('INBOX')
    fake.fetch[1] = body_rows(raw=RAW + b'x', size=len(RAW) + 1)
    with pytest.raises(ImapReadError, match='message_changed_during_fetch'):
        provider.fetch_message(42)


def test_metadata_after_literal_supported(connected):
    provider, fake = connected
    provider.select_folder('INBOX')
    fake.fetch[1] = [(b'1 (BODY[] {%d}' % len(RAW), RAW), metadata()[3:]]
    assert provider.fetch_message(42)['raw'] == RAW


@pytest.mark.parametrize('command, code', [('LOGIN', 'authentication_failed'), ('CAPABILITY', 'capability_failed'), ('ID', 'client_id_failed')])
def test_connect_errors_sanitized_and_disconnected(command, code):
    fake = FakeClient()
    fake.failures[command] = imaplib.IMAP4.error('secret-code server private data')
    provider = ImapReadOnlyProvider('reader@163.com', 'secret-code', client_factory=lambda *a, **k: fake)
    with pytest.raises(ImapReadError) as caught:
        provider.connect()
    assert str(caught.value) == code
    assert 'secret' not in repr(caught.value)
    with pytest.raises(ImapReadError, match='not_connected'):
        provider.describe_account()


def test_network_error_clears_selection_and_logout_failure_shuts_down(connected):
    provider, fake = connected
    provider.select_folder('INBOX')
    fake.failures['UID FETCH'] = TimeoutError('server secret')
    fake.failures['LOGOUT'] = OSError('socket lost')
    with pytest.raises(ImapReadError, match='message_fetch_failed'):
        provider.fetch_message(42)
    assert fake.trace[-1][0] == 'SHUTDOWN'
    with pytest.raises(ImapReadError, match='not_connected'):
        provider.list_uids()


def test_non_ok_status_is_sanitized(connected):
    provider, fake = connected
    fake.list = lambda *args: ('NO', [b'private mailbox data'])
    with pytest.raises(ImapReadError) as caught:
        provider.list_folders()
    assert str(caught.value) == 'folder_list_failed'


def test_literal_allocation_bounded_before_read(monkeypatch):
    client = _BoundedIMAP4SSL.__new__(_BoundedIMAP4SSL)
    client._max_literal_bytes = 32
    reads = []
    monkeypatch.setattr(imaplib.IMAP4_SSL, 'read', lambda self, size: reads.append(size) or b'x' * size)
    with pytest.raises(ImapMessageTooLarge):
        client.read(10**12)
    assert reads == []
    assert client.read(32) == b'x' * 32


@pytest.mark.parametrize('address', ['reader', 'reader@gmail.com', 'reader@163.com.evil', 'reader@163.com\r\n', ' reader@163.com'])
def test_fixed_163_identity(address):
    with pytest.raises(ImapReadError, match='invalid_account'):
        ImapReadOnlyProvider(address, 'synthetic-code')


@pytest.mark.parametrize('operation', ['uid', 'size', 'marker', 'search', 'list'])
def test_untrusted_integer_fields_remain_safe_errors(connected, operation):
    provider, fake = connected
    provider.select_folder('INBOX')
    huge = b'9' * 5000
    if operation == 'search':
        fake.search = [huge]
        action = provider.list_uids
    elif operation == 'list':
        fake.folders = [(b'() "/" {' + huge + b'}', b'x')]
        action = provider.list_folders
    else:
        if operation == 'uid':
            fake.fetch[0] = [metadata().replace(b'UID 42', b'UID ' + huge)]
        elif operation == 'size':
            fake.fetch[0] = [metadata().replace(b'RFC822.SIZE ' + str(len(RAW)).encode(), b'RFC822.SIZE ' + huge)]
        else:
            fake.fetch[1] = [(metadata()[:-1] + b' BODY[] {' + huge + b'}', RAW), b')']
        action = lambda: provider.fetch_message(42)
    with pytest.raises(ImapReadError):
        action()


def test_missing_during_body_fetch(connected):
    provider, fake = connected
    provider.select_folder('INBOX')
    fake.fetch[1] = [None]
    with pytest.raises(ImapMessageMissing):
        provider.fetch_message(42)


def test_body_exceeding_limit_rejected_even_with_small_metadata(connected):
    provider, fake = connected
    provider.max_message_bytes = len(RAW)
    provider.select_folder('INBOX')
    fake.fetch[1] = body_rows(raw=RAW + b'oversized')
    with pytest.raises(ImapMessageTooLarge):
        provider.fetch_message(42)


def test_existing_flags_preserved_as_metadata(connected):
    provider, fake = connected
    provider.select_folder('INBOX')
    rows = body_rows()
    fake.fetch[1] = [(rows[0][0].replace(b'FLAGS ()', b'FLAGS (\\Seen \\Flagged Custom)'), RAW), b')']
    assert provider.fetch_message(42)['flags'] == ['\\Seen', '\\Flagged', 'Custom']


def test_exception_in_with_block_still_logs_out():
    fake = FakeClient()
    with pytest.raises(ValueError, match='caller error'):
        with ImapReadOnlyProvider('reader@163.com', 'synthetic-code', client_factory=lambda *a, **k: fake):
            raise ValueError('caller error')
    assert fake.trace[-1][0] == 'LOGOUT'


def test_factory_failure_is_sanitized():
    def factory(*args, **kwargs):
        raise OSError('private certificate or server information')
    with pytest.raises(ImapReadError) as caught:
        ImapReadOnlyProvider('reader@163.com', 'synthetic-code', client_factory=factory).connect()
    assert str(caught.value) == 'connection_failed'


@pytest.mark.parametrize('kwargs', [dict(timeout=0), dict(timeout=float('nan')), dict(timeout=True),
                                   dict(max_message_bytes=0), dict(max_message_bytes=True)])
def test_invalid_limits(kwargs):
    with pytest.raises(ImapReadError):
        ImapReadOnlyProvider('reader@163.com', 'synthetic-code', **kwargs)


def test_read_requires_connection_and_explicit_folder(connected):
    provider, _ = connected
    with pytest.raises(ImapReadError, match='folder_not_selected'):
        provider.list_uids()
    provider.close()
    with pytest.raises(ImapReadError, match='not_connected'):
        provider.fetch_message(42)


def full_body_rows(raw=RAW, uid=42, size=None, date=DATE):
    rows = body_rows(raw=raw, uid=uid, size=size, date=date)
    return [(rows[0][0].replace(b'BODY[]<0>', b'BODY[]'), raw), b')']


@pytest.mark.parametrize('difference', [4, 17])
def test_inaccurate_reported_size_requires_independent_full_body(connected, difference):
    provider, fake = connected
    provider.select_folder('INBOX')
    reported = len(RAW) + difference
    fake.fetch = [[metadata(size=reported)], body_rows(size=reported), full_body_rows(size=reported)]
    message = provider.fetch_message(42)
    assert message['raw'] == RAW
    assert message['size'] == len(RAW)
    assert message['reported_size'] == reported
    assert message['size_mismatch'] is True
    assert fake.trace[-1][1] == ('42', '(UID RFC822.SIZE INTERNALDATE FLAGS BODY.PEEK[])')
    assert len([entry for entry in fake.trace if entry[0] == 'UID FETCH']) == 3


def test_underreported_size_recovers_verified_full_body_with_prefix_check(connected):
    provider, fake = connected
    provider.select_folder('INBOX')
    reported = 10
    fake.fetch = [[metadata(size=reported)], body_rows(raw=RAW[:11], size=reported), full_body_rows(size=reported)]
    message = provider.fetch_message(42)
    assert message['raw'] == RAW
    assert message['reported_size'] == 10
    assert message['size'] == len(RAW)
    assert message['size_mismatch'] is True


def test_short_initial_fetch_can_recover_complete_matching_full_body(connected):
    provider, fake = connected
    provider.select_folder('INBOX')
    fake.fetch = [[metadata()], body_rows(raw=RAW[:-3]), full_body_rows()]
    message = provider.fetch_message(42)
    assert message['raw'] == RAW
    assert message['size_mismatch'] is False


@pytest.mark.parametrize('failure', ['offset', 'wrong_uid', 'changed_size', 'changed_date', 'changed_bytes', 'bad_literal', 'missing', 'too_large'])
def test_full_body_reverification_fails_closed(connected, failure):
    provider, fake = connected
    provider.select_folder('INBOX')
    reported = len(RAW) + 4
    full = full_body_rows(size=reported)
    expected = ImapReadError
    if failure == 'offset':
        full = body_rows(size=reported)
    elif failure == 'wrong_uid':
        full = full_body_rows(uid=43, size=reported)
    elif failure == 'changed_size':
        full = full_body_rows(size=reported+1)
    elif failure == 'changed_date':
        full = full_body_rows(size=reported, date=b'13-Sep-2026 12:30:00 +0800')
    elif failure == 'changed_bytes':
        full = full_body_rows(raw=b'X' + RAW[1:], size=reported)
    elif failure == 'bad_literal':
        full = [(full[0][0].replace(b'{'+str(len(RAW)).encode()+b'}', b'{999}'), RAW), b')']
    elif failure == 'missing':
        full = [None]
        expected = ImapMessageMissing
    elif failure == 'too_large':
        provider.max_message_bytes = reported
        full = full_body_rows(raw=RAW+b'oversized body', size=reported)
        expected = ImapMessageTooLarge
    fake.fetch = [[metadata(size=reported)], body_rows(size=reported), full]
    with pytest.raises(expected):
        provider.fetch_message(42)


def test_invalid_literal_framing_never_enters_size_fallback(connected):
    provider, fake = connected
    provider.select_folder('INBOX')
    fake.fetch = [[metadata()], [(metadata()[:-1] + b' BODY[]<0> {999}', RAW), b')']]
    with pytest.raises(ImapReadError, match='incomplete_message'):
        provider.fetch_message(42)
    assert len([entry for entry in fake.trace if entry[0] == 'UID FETCH']) == 2


def test_flags_batches_are_readonly_and_missing_uids_are_allowed(connected):
    provider, fake = connected
    provider.select_folder('INBOX')
    fake.fetch = [[b'1 (UID 1 FLAGS (\\Seen \\Flagged))', b'2 (FLAGS () UID 200)'],
                  [b'3 (uid 201 flags (Custom))'], [None]]
    assert provider.fetch_flags(list(range(1, 402))) == {1: ['\\Seen', '\\Flagged'], 200: [], 201: ['Custom']}
    requests = [entry for entry in fake.trace if entry[0] == 'UID FETCH']
    assert len(requests) == 3
    assert [len(entry[1][0].split(',')) for entry in requests] == [200, 200, 1]
    assert all(entry[1][1] == '(UID FLAGS)' for entry in requests)
    assert requests[-1][1][0] == '401'


def test_empty_flags_request_does_not_require_network():
    provider = ImapReadOnlyProvider('reader@163.com', 'synthetic-code')
    assert provider.fetch_flags([]) == {}


def test_duplicate_input_uids_and_consistent_responses(connected):
    provider, fake = connected
    provider.select_folder('INBOX')
    fake.fetch = [[b'1 (UID 42 FLAGS (\\Seen Custom))', b'1 (FLAGS (Custom \\Seen) UID 42)']]
    assert set(provider.fetch_flags([42, 42])[42]) == {'\\Seen', 'Custom'}
    assert fake.trace[-1][1] == ('42', '(UID FLAGS)')


@pytest.mark.parametrize('uids', [[True], [0], [-1], [2**32], ['42'], '42', None])
def test_invalid_flags_uid_inputs_no_commands(connected, uids):
    provider, fake = connected
    before = list(fake.trace)
    with pytest.raises(ImapReadError, match='invalid_uid_list'):
        provider.fetch_flags(uids)
    assert fake.trace == before


@pytest.mark.parametrize('rows', [
    [b'1 (UID 43 FLAGS ())'],
    [b'1 (UID 0 FLAGS ())'],
    [b'1 (UID 4294967296 FLAGS ())'],
    [b'1 (FLAGS ("UID 42"))'],
    [b'1 (UID 42 FLAGS ("quoted"))'],
    [b'1 (UID 42 FLAGS (\xff))'],
    [b'1 (UID 42 FLAGS (bad\x00flag))'],
    [b'1 (UID 42 FLAGS (one\ttwo))'],
    [b'4294967296 (UID 42 FLAGS ())'],
    [b'1 (UID 42 FLAGS (' + b'x'*201 + b'))'],
    [b'1 (UID 42 FLAGS (' + b'flag '*101 + b'))'],
    [b'1 (UID 42 FLAGS ()) BODY[] {5}'],
    [(b'1 (UID 42 FLAGS () BODY[] {5}', b'hello'), b')'],
    [b'1 (UID 42 FLAGS ())', b'garbage'],
    [b'1 (UID 42 FLAGS ())', b'1 (UID 42 FLAGS (\\Seen))'],
    [b'1 (UID 42 FLAGS ()) UID 42'],
    [b'1 (UID 42 FLAGS ())\r\n'],
])
def test_bad_flags_response_never_becomes_partial_snapshot(connected, rows):
    provider, fake = connected
    provider.select_folder('INBOX')
    fake.fetch = [rows]
    with pytest.raises(ImapReadError):
        provider.fetch_flags([42])


def test_uid_from_another_batch_rejected(connected):
    provider, fake = connected
    provider.select_folder('INBOX')
    fake.fetch = [[b'1 (UID 201 FLAGS ())']]
    with pytest.raises(ImapReadError, match='unexpected_uid'):
        provider.fetch_flags(list(range(1, 202)))


@pytest.mark.parametrize('failure', ['NO', 'disconnect'])
def test_failed_flags_fetch_cannot_establish_empty_snapshot(connected, failure):
    provider, fake = connected
    provider.select_folder('INBOX')
    if failure == 'NO':
        fake.uid = lambda *args: ('NO', [b'private diagnostics'])
    else:
        fake.failures['UID FETCH'] = TimeoutError('private diagnostics')
    with pytest.raises(ImapReadError, match='flags_fetch_failed'):
        provider.fetch_flags([42])


def test_cancel_after_first_flags_batch_prevents_further_requests(connected):
    from threading import Event
    from agents.runtime import RunContext, RunCancelled, use_run_context
    provider, fake = connected
    provider.select_folder('INBOX')
    cancelled = Event()

    def first_batch(*args):
        fake.record('UID FETCH', *args)
        cancelled.set()
        return 'OK', [b'1 (UID 1 FLAGS ())']

    fake.uid = first_batch
    with use_run_context(RunContext(cancel_event=cancelled)):
        with pytest.raises(RunCancelled):
            provider.fetch_flags(list(range(1, 402)))
    requests = [entry for entry in fake.trace if entry[0] == 'UID FETCH']
    assert len(requests) == 1
    assert len(requests[0][1][1].split(',')) == 200


def test_cancel_before_flags_scan_sends_no_fetch(connected):
    from threading import Event
    from agents.runtime import RunContext, RunCancelled, use_run_context
    provider, fake = connected
    provider.select_folder('INBOX')
    cancelled = Event()
    cancelled.set()
    before = list(fake.trace)
    with use_run_context(RunContext(cancel_event=cancelled)):
        with pytest.raises(RunCancelled):
            provider.fetch_flags([42])
    assert fake.trace == before
