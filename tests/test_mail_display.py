"""Presentation checks using the real HTML/MIME pipeline and synthetic mail."""
from copy import deepcopy
from email.message import EmailMessage
from html import escape

from core.imap_mime import parse_imap_message
from frontend.mailbox_view import body_for_display


def parsed_html(html):
    message = EmailMessage()
    message.set_content(html, subtype='html')
    return parse_imap_message(message.as_bytes(), account_id='synthetic', folder='INBOX',
                              uidvalidity='1', uid=1).model_dump()


def test_real_mime_nested_table_schema_renders_without_source_markers():
    email = parsed_html('<p>Before</p><table><tr><td>Outer<table><tr><td>Inner</td></tr></table>'
                        '</td><td>Value &lt;script&gt;</td></tr></table><p>After</p>')
    original = deepcopy(email)
    assert email['table_rows']
    assert 'text' not in email['table_rows'][0]
    assert 'status=nested_table_flattened' in email['body']
    displayed = body_for_display(email)
    assert '[Table ' not in displayed
    assert 'status=nested_table_flattened' not in displayed
    assert 'mail-table-cell' in displayed
    assert all(value in displayed for value in ('Before', 'Outer', 'Inner', 'After', '&lt;script&gt;'))
    assert '<script>' not in displayed
    assert email == original


def test_rowspan_headers_and_multiple_tables_preserve_content():
    email = parsed_html('<table><tr><th scope="col">Name</th><th scope="col">Price</th></tr>'
                        '<tr><td rowspan="2">Item</td><td>10</td></tr><tr><td>20</td></tr></table>'
                        '<p>Between</p><table><tr><td>Last</td></tr></table>')
    displayed = body_for_display(email)
    assert '[Table ' not in displayed
    assert displayed.count('class="mail-table-row"') == len(email['table_rows'])
    assert all(value in displayed for value in ('Name', 'Price', 'Item', '10', '20', 'Between', 'Last'))


def test_marker_like_user_prose_is_not_removed():
    literal = '[Table custom row custom:r1 status=nested_table_flattened] user-authored text'
    email = parsed_html('<p>' + literal + '</p><table><tr><td>Real cell</td></tr></table>')
    displayed = body_for_display(email)
    assert literal in displayed
    assert 'Real cell' in displayed
    assert displayed.count('[Table ') == 1


def test_changed_span_or_cell_metadata_falls_back_without_losing_body():
    for change in ('offset', 'cell', 'status', 'missing'):
        email = parsed_html('<p>Prefix</p><table><tr><td>Kept value</td></tr></table>')
        row = email['table_rows'][0]
        if change == 'offset':
            row['start'] += 1
        elif change == 'cell':
            row['cells'][0]['text'] = 'Different value'
        elif change == 'status':
            row['status'] = 'different'
        else:
            del row['cells'][0]['source_id']
        assert body_for_display(email) == escape(email['body'])


def test_plain_body_html_is_escaped_without_table_metadata():
    body = '<img src="https://invalid.example/private"> [Table x row y status=complete]'
    assert body_for_display({'body': body}) == escape(body)


def test_verified_empty_layout_table_uses_readable_note():
    email = parsed_html('<p>Before</p><table></table><p>After</p>')
    original = deepcopy(email)
    assert email['table_rows'][0]['cells'] == []
    assert '; empty table]' in email['body']
    displayed = body_for_display(email)
    assert '[Table ' not in displayed
    assert '表格未提取到可显示内容' in displayed
    assert 'Before' in displayed and 'After' in displayed
    assert email == original


def test_empty_table_user_literal_and_unverified_metadata_preserved():
    literal = '[Table user status=complete; empty table]'
    email = parsed_html('<p>' + literal + '</p><table></table>')
    displayed = body_for_display(email)
    assert literal in displayed
    assert displayed.count('[Table ') == 1
    email['table_rows'][0]['status'] = 'changed'
    assert body_for_display(email) == escape(email['body'])


def test_empty_cells_with_arbitrary_text_field_do_not_remove_prose():
    body = 'User-authored prose'
    email = {'body': body, 'table_rows': [{'cells': [], 'start': 0, 'end': len(body), 'text': body}]}
    assert body_for_display(email) == escape(body)
