"""Synthetic RFC822 fixtures: no mailbox or private mail access."""
import base64
import hashlib
from email.message import EmailMessage
from email import policy

import pytest

import core.imap_mime as mime
from core.imap_mime import MailParseError, parse_imap_message


def parse(raw, **kwargs):
    return parse_imap_message(raw, account_id=kwargs.pop('account_id', 'fixture@163.com'),
        folder=kwargs.pop('folder', 'INBOX'), uidvalidity=kwargs.pop('uidvalidity', '42'),
        uid=kwargs.pop('uid', 1), **kwargs)


def basic():
    message = EmailMessage()
    message['From'] = 'Fixture Sender <sender@example.test>'
    message['To'] = 'first@example.test, second@example.test'
    message['Date'] = 'Sat, 12 Sep 2026 10:00:00 +0800'
    message['Message-ID'] = '<fixture-1@example.test>'
    message['Subject'] = 'Synthetic subject'
    message.set_content('Synthetic body')
    return message


def test_locator_hash_namespace_and_raw_provenance():
    raw = basic().as_bytes(policy=policy.SMTP)
    row = parse(raw)
    assert row.body.strip() == 'Synthetic body'
    assert row.id == parse(raw, folder='inbox', uidvalidity='00042').id
    assert len({row.id, parse(raw, account_id='other@163.com').id, parse(raw, folder='Archive').id,
                parse(raw, uidvalidity='43').id, parse(raw, uid=2).id}) == 5
    assert row.source['raw_sha256'] == hashlib.sha256(raw).hexdigest()
    assert row.source['uid'] == 1
    assert row.source['parser_version'].startswith('imap-rfc822-v1:')
    assert row.date == '2026-09-12T02:00:00+00:00'
    assert row.recipients == ['first@example.test', 'second@example.test']


def test_rfc2047_gbk_headers_rfc2231_filename_and_gb18030_body():
    subject = base64.b64encode('采购确认'.encode('gbk')).decode()
    body = base64.b64encode('中文正文𠀀'.encode('gb18030')).decode()
    raw = (f'Subject: =?GBK?B?{subject}?=\r\nFrom: fixture@example.test\r\n'
           'MIME-Version: 1.0\r\nContent-Type: multipart/mixed; boundary="fixture"\r\n\r\n'
           '--fixture\r\nContent-Type: text/plain; charset=GB18030\r\n'
           f'Content-Transfer-Encoding: base64\r\n\r\n{body}\r\n'
           '--fixture\r\nContent-Type: text/plain; charset=utf-8\r\n'
           "Content-Disposition: attachment; filename*=utf-8''%E6%8A%A5%E4%BB%B7.txt\r\n\r\n"
           'attachment fixture\r\n--fixture--\r\n').encode()
    row = parse(raw)
    assert row.subject == '采购确认'
    assert row.body == '中文正文𠀀'
    assert row.attachments[0]['filename'] == '报价.txt'
    assert row.attachments[0]['status'] == 'complete'
    assert 'attachment fixture' not in row.body
    assert row.attachments[0]['text'].strip() == 'attachment fixture'


@pytest.mark.parametrize('charset,payload,expected,warning', [
    ('utf-8', '中文'.encode('gbk'), '中文', 'charset_fallback_used'),
    ('bogus-codec', '正文'.encode(), '正文', 'unknown_declared_charset'),
    ('utf-8', b'\xff', '\ufffd', 'decode_replacement_used'),
    ('', '中文'.encode(), '中文', 'charset_missing_inferred'),
])
def test_decode_fallbacks_are_reported(charset, payload, expected, warning):
    header = ('; charset=' + charset) if charset else ''
    raw = ('Content-Type: text/plain' + header + '\r\nContent-Transfer-Encoding: base64\r\n\r\n').encode() + base64.b64encode(payload)
    row = parse(raw)
    assert row.body == expected
    assert any(warning in issue for issue in row.source['warnings'])
    assert row.decode_quality['body']['status'] == 'suspect'


def test_quoted_printable_and_invalid_base64_report_quality():
    row = parse(b'Content-Type: text/plain; charset=utf-8\r\nContent-Transfer-Encoding: quoted-printable\r\n\r\n=E4=B8=AD=E6=96=87')
    assert row.body == '中文'
    broken = parse(b'Content-Type: text/plain; charset=utf-8\r\nContent-Transfer-Encoding: base64\r\n\r\nSGVsbG8$')
    assert broken.body == 'Hello'
    assert any('invalid_base64' in issue for issue in broken.source['warnings'])


def test_alternative_selects_one_html_body_preserving_table_spans():
    msg = basic()
    msg.set_content('Plain alternative should not duplicate')
    msg.add_alternative('<p>HTML preferred</p><table><tr><th>Item</th><th>Price</th></tr><tr><td>A</td><td>23</td></tr></table>', subtype='html')
    row = parse(msg.as_bytes())
    assert 'HTML preferred' in row.body
    assert 'Plain alternative' not in row.body
    assert row.table_rows
    assert all('23' in row.body[r['start']:r['end']] or 'Price' in row.body[r['start']:r['end']] for r in row.table_rows)
    assert not row.attachments
    assert row.source['body_selection']


def test_related_start_selects_root_and_keeps_resources_separate():
    raw = (b'Content-Type: multipart/related; boundary="x"; start="<root>"\r\n\r\n'
        b'--x\r\nContent-Type: image/png\r\nContent-ID: <image>\r\nContent-Transfer-Encoding: base64\r\n\r\nUE5H\r\n'
        b'--x\r\nContent-Type: text/html; charset=utf-8\r\nContent-ID: <root>\r\n\r\n<p>Only root text</p>\r\n--x--\r\n')
    row = parse(raw)
    assert row.body == 'Only root text'
    assert len(row.attachments) == 1
    assert row.attachments[0]['role'] == 'inline_resource'
    assert row.attachments[0]['reason'] == 'image_ocr_not_enabled'


def test_forwarded_message_is_inventory_not_parent_text():
    outer, inner = basic(), basic()
    inner.set_content('Forwarded private synthetic text')
    outer.add_attachment(inner)
    row = parse(outer.as_bytes())
    assert 'Forwarded private' not in row.body
    assert row.attachments[0]['role'] == 'forwarded_message'
    assert row.attachments[0]['status'] == 'not_read'
    assert row.attachments[0]['hash_basis'] == 'reconstructed_rfc822_bytes'


def test_reply_references_are_account_scoped_and_match_parent():
    parent = basic()
    reply = basic()
    reply.replace_header('Message-ID', '<fixture-2@example.test>')
    reply['References'] = '<fixture-1@example.test>'
    reply['In-Reply-To'] = '<fixture-1@example.test>'
    a, b = parse(parent.as_bytes()), parse(reply.as_bytes(), uid=2)
    assert b.in_reply_to == a.message_id == b.references[0]
    assert a.thread_id == b.thread_id
    assert parse(reply.as_bytes(), account_id='other@163.com').thread_id != b.thread_id


def test_missing_date_uses_internaldate_without_inventing_timezone():
    row = parse(b'Content-Type: text/plain\r\n\r\nfixture', internal_date='12-Sep-2026 10:00:00 +0800')
    assert row.date == '2026-09-12T02:00:00+00:00'
    assert row.source['date_origin'] == 'imap_internaldate'
    missing = parse(b'Date: Sat, 12 Sep 2026 10:00:00\r\n\r\nfixture')
    assert missing.date == ''
    assert 'date_unavailable' in missing.source['warnings']


@pytest.mark.parametrize('raw,code', [
    (b'', 'invalid_raw_message'),
    (b'Content-Type: text/plain\r\nContent-Type: text/html\r\n\r\nx', 'ambiguous_mime_headers'),
    (b'Content-Type: multipart/mixed; boundary=x\r\n\r\nno boundary', 'multipart_boundary_unreadable'),
])
def test_unsafe_structure_has_stable_errors(raw, code):
    with pytest.raises(MailParseError) as error: parse(raw)
    assert error.value.code == code


def test_budgets_and_filename_do_not_create_files(monkeypatch, tmp_path):
    msg = basic()
    msg.add_attachment(b'synthetic', maintype='text', subtype='plain', filename='../../escape.txt')
    row = parse(msg.as_bytes())
    assert row.attachments[0]['filename'] == '../../escape.txt'
    assert not list(tmp_path.iterdir())
    monkeypatch.setattr(mime, 'MAX_RAW_BYTES', 4)
    with pytest.raises(MailParseError, match='raw_message_byte_limit'): parse(msg.as_bytes())


def test_repeated_recipient_headers_are_not_silently_dropped():
    raw = b'To: a@example.test\r\nTo: b@example.test\r\n\r\nfixture'
    assert parse(raw).recipients == ['a@example.test', 'b@example.test']


def test_related_alternative_does_not_turn_unused_body_into_attachment():
    related = EmailMessage()
    related.make_related()
    alternative = EmailMessage()
    alternative.set_content('Unused plain')
    alternative.add_alternative('<p>Chosen HTML</p>', subtype='html')
    related.attach(alternative)
    related.add_related(b'PNG', maintype='image', subtype='png', cid='<fixture>')
    row = parse(related.as_bytes())
    assert row.body == 'Chosen HTML'
    assert len(row.attachments) == 1
    assert row.attachments[0]['mime_type'] == 'image/png'


def test_part_depth_and_body_budgets_are_explicit(monkeypatch):
    root = EmailMessage()
    root.make_mixed()
    child = EmailMessage()
    child.make_mixed()
    leaf = EmailMessage()
    leaf.set_content('abcdef')
    child.attach(leaf)
    root.attach(child)
    raw = root.as_bytes()
    monkeypatch.setattr(mime, 'MAX_BODY_CHARS', 3)
    row = parse(raw)
    assert row.body == 'abc'
    assert row.source['body_status'] == 'partial'
    monkeypatch.setattr(mime, 'MAX_DEPTH', 1)
    with pytest.raises(MailParseError, match='mime_depth_limit'): parse(raw)


def test_unencoded_utf8_header_recovers_octets_and_reports_inference():
    row = parse('Subject: 中文主题\r\n\r\nfixture'.encode())
    assert row.subject == '中文主题'
    assert 'charset_missing_inferred:subject' in row.source['warnings']
    assert row.decode_quality['subject']['status'] == 'suspect'
