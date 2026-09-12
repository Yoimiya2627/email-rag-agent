"""Passive local attachment extraction with synthetic documents."""
import io
import zipfile

import pytest

import core.attachment_text as attachment
from core.attachment_text import extract_attachment_text


def zip_document(files):
    target = io.BytesIO()
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, text in files.items(): archive.writestr(name, text)
    return target.getvalue()


@pytest.mark.parametrize('filename,mime,data', [
    ('a.txt', 'text/plain', b'fixture plain'),
    ('a.csv', 'text/csv', b'item,value\na,3'),
    ('a.json', 'application/json', b'{"fixture":true}'),
])
def test_plain_documents_preserve_text_and_locations(filename, mime, data):
    result = extract_attachment_text(data, filename=filename, mime_type=mime)
    assert result['status'] == 'complete'
    assert result['text'] == data.decode()
    assert result['locations'][0] == {'kind': 'text', 'start': 0, 'end': len(data)}
    assert result['size'] == len(data)


def test_application_ics_alias_extracts_same_text_as_calendar_without_interpretation():
    data = ('BEGIN:VCALENDAR\r\nVERSION:2.0\r\nBEGIN:VEVENT\r\n'
            'UID:synthetic-fixture\r\nSUMMARY:合成日历文本\r\n'
            'DESCRIPTION:First line\r\n continued line\r\n'
            'END:VEVENT\r\nEND:VCALENDAR\r\n').encode('utf-8')
    calendar = extract_attachment_text(data, mime_type='text/calendar', charset='utf-8')
    alias = extract_attachment_text(data, mime_type='application/ics', charset='utf-8')
    assert alias == calendar
    assert alias['status'] == 'complete'
    assert alias['text'] == data.decode('utf-8')
    assert alias['locations'] == [{'kind': 'text', 'start': 0, 'end': len(alias['text'])}]
    assert 'events' not in alias


def test_ics_extension_alone_does_not_force_binary_attachment_to_text():
    result = extract_attachment_text(b'\x00\xff\x80', filename='synthetic.ics',
                                     mime_type='application/octet-stream')
    assert result['status'] == 'unsupported'
    assert result['reason'] == 'unsupported_format'
    assert not result['text']


def test_html_removes_active_content_and_retains_tables():
    result = extract_attachment_text(b'<script>execute()</script><table><tr><td>A</td><td>23</td></tr></table>', filename='fixture.html')
    assert 'execute' not in result['text']
    assert '23' in result['text']
    assert result['table_rows']


def test_docx_extracts_paragraphs_and_marks_text_only():
    data = zip_document({'word/document.xml': '<w:document xmlns:w="urn:w"><w:body><w:p><w:r><w:t>Fixture paragraph</w:t></w:r></w:p></w:body></w:document>'})
    row = extract_attachment_text(data, filename='fixture.docx')
    assert row['text'] == 'Fixture paragraph'
    assert row['status'] == 'partial'
    assert row['locations'][0]['paragraph'] == 1


def test_xlsx_retains_cell_and_never_evaluates_formula():
    data = zip_document({'xl/workbook.xml': '<workbook/>',
        'xl/sharedStrings.xml': '<sst><si><t>Fixture cell</t></si></sst>',
        'xl/worksheets/sheet1.xml': '<worksheet><sheetData><row><c r="A1" t="s"><v>0</v></c><c r="B1"><f>HYPERLINK("https://invalid.test")</f><v>7</v></c></row></sheetData></worksheet>'})
    row = extract_attachment_text(data, filename='fixture.xlsx')
    assert row['text'] == 'Fixture cell\n7'
    assert row['locations'][1]['cell'] == 'B1'
    assert row['locations'][1]['cached_formula'] is True
    assert 'formula_not_evaluated_cached_value_only' in row['warnings']


@pytest.mark.parametrize('files,reason', [
    ({'../escape.xml': 'x'}, 'unsafe_archive_structure'),
    ({'word/document.xml': '<!DOCTYPE a [<!ENTITY x "evil">]><a>&x;</a>'}, 'xml_dtd_not_allowed'),
    ({'word/document.xml': 'x' * 100000}, 'archive_expansion_limit'),
])
def test_office_archive_limits_and_dtd_rejected(files, reason):
    row = extract_attachment_text(zip_document(files), filename='fixture.docx')
    assert row['status'] == 'not_read'
    assert row['reason'] == reason
    assert not row['text']


def test_text_budget_and_bad_encoding_are_not_reported_complete(monkeypatch):
    result = extract_attachment_text(b'abcdef', filename='a.txt', max_chars=3)
    assert result['text'] == 'abc' and result['status'] == 'partial'
    result = extract_attachment_text(b'\xff', filename='a.txt', charset='utf-8')
    assert result['status'] == 'partial'
    assert 'decode_replacement_used' in result['warnings']
    monkeypatch.setattr(attachment, 'MAX_ATTACHMENT_BYTES', 1)
    assert extract_attachment_text(b'abc', filename='a.txt')['reason'] == 'attachment_byte_limit'


def pdf_bytes(*, text=True, encrypted=False, pages=1):
    from pypdf import PdfWriter
    from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
    writer = PdfWriter()
    for _ in range(pages):
        page = writer.add_blank_page(width=200, height=200)
        if text:
            font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
            page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
            content = DecodedStreamObject()
            content.set_data(b'BT /F1 12 Tf 10 100 Td (Synthetic PDF text) Tj ET')
            page[NameObject('/Contents')] = writer._add_object(content)
    if encrypted: writer.encrypt('fixture-password')
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def test_pdf_text_page_locations_and_encryption():
    row = extract_attachment_text(pdf_bytes(), filename='fixture.pdf')
    assert row['status'] == 'complete'
    assert 'Synthetic PDF text' in row['text']
    assert row['locations'][0]['page'] == 1
    encrypted = extract_attachment_text(pdf_bytes(encrypted=True), filename='fixture.pdf')
    assert encrypted['status'] == 'not_read' and encrypted['reason'] == 'encrypted_pdf'


def test_pdf_blank_or_scanned_not_claimed_complete_and_page_limit(monkeypatch):
    row = extract_attachment_text(pdf_bytes(text=False), filename='fixture.pdf')
    assert row['status'] == 'not_read'
    assert row['pages_without_extractable_text'] == [1]
    assert 'ocr_not_enabled' in row['warnings'][0]
    monkeypatch.setattr(attachment, 'MAX_PDF_PAGES', 1)
    row = extract_attachment_text(pdf_bytes(pages=2), filename='fixture.pdf')
    assert row['status'] == 'partial'
    assert row['pages_read'] == 1


@pytest.mark.parametrize('filename,mime,reason', [
    ('a.png', 'image/png', 'image_ocr_not_enabled'),
    ('a.xls', 'application/vnd.ms-excel', 'legacy_or_macro_office_not_supported'),
    ('a.bin', 'application/octet-stream', 'unsupported_format'),
    ('a.eml', 'message/rfc822', 'forwarded_message_separate_from_parent'),
])
def test_unsupported_formats_report_reason(filename, mime, reason):
    row = extract_attachment_text(b'fixture', filename=filename, mime_type=mime)
    assert row['status'] in {'unsupported', 'not_read'}
    assert row['reason'] == reason
