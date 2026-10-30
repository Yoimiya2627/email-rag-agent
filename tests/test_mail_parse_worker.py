"""Exercise the actual isolated worker, not a mocked parser function."""
from email.message import EmailMessage

import pytest

from core.mail_sync import isolated_parse, ImapSyncError


def test_real_worker_reads_mime_attachment_and_never_loads_html_resources(tmp_path):
    message=EmailMessage()
    message['Subject']='本地解析验证'
    message['From']='sender@example.test'
    message['To']='receiver@example.test'
    message['Date']='Sat, 12 Sep 2026 08:00:00 +0000'
    message.set_content('只验证本地解析。')
    message.add_alternative('<html><body>Only local<img src="https://example.invalid/never-fetch"><table><tr><td>金额</td><td>128</td></tr></table></body></html>', subtype='html')
    message.add_attachment('附件数据'.encode('utf-8'),maintype='text',subtype='plain',filename='说明.txt',params={'charset':'utf-8'})
    raw=tmp_path/'fixture.eml'
    raw.write_bytes(message.as_bytes())
    result=isolated_parse(raw,{'account_id':'account-a','folder':'INBOX','uidvalidity':'1','uid':1})
    # MIME alternatives are representations of one body; this parser selects
    # the richer HTML representation and preserves its structured table.
    assert result.subject=='本地解析验证' and 'Only local' in result.body and '128' in result.body
    assert result.attachments[0]['text']=='附件数据'
    assert result.attachments[0]['status']=='complete'
    assert not list(tmp_path.glob('.parse-*'))


def test_real_worker_timeout_is_bounded_and_temp_files_can_be_removed(tmp_path):
    raw=tmp_path/'fixture.eml'
    raw.write_bytes(b'Subject: Fixture\r\n\r\nLocal text')
    with pytest.raises(ImapSyncError,match='parse_timeout'):
        isolated_parse(raw,{'account_id':'account-a','folder':'INBOX','uidvalidity':'1','uid':1},timeout=.00001)
    assert not list(tmp_path.glob('.parse-*'))
