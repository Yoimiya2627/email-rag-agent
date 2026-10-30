"""Bounded, passive attachment text extraction; never executes embedded content."""
from __future__ import annotations

import hashlib
import io
import re
import zipfile
from pathlib import PurePosixPath
from xml.etree import ElementTree as ET

from core.cleaner import HTML_INPUT_LIMIT, html_to_structured_text

MAX_ATTACHMENT_BYTES = 8 * 1024 * 1024
MAX_TEXT_CHARS = 200_000
MAX_ZIP_MEMBERS = 1024
MAX_ZIP_BYTES = 24 * 1024 * 1024
MAX_XML_BYTES = 8 * 1024 * 1024
MAX_ZIP_RATIO = 200
MAX_LOCATIONS = 5000
MAX_PDF_PAGES = 100
MAX_PDF_STREAM_BYTES = 8 * 1024 * 1024


def decode_text(data: bytes, charset: str | None = None) -> tuple[str, list[str], str]:
    """Decode strictly first; report guesses and every replacement fallback."""
    warnings = []
    declared = (charset or '').strip().lower()
    candidates = []
    if data.startswith((b'\xff\xfe', b'\xfe\xff')):
        candidates.append('utf-16')
    elif data.startswith(b'\xef\xbb\xbf'):
        candidates.append('utf-8-sig')
    if declared:
        candidates.append(declared)
    candidates.extend(['utf-8', 'gb18030'])
    for codec in dict.fromkeys(candidates):
        try:
            text = data.decode(codec, errors='strict')
        except LookupError:
            if codec == declared:
                warnings.append('unknown_declared_charset')
            continue
        except UnicodeError:
            if codec == declared:
                warnings.append('declared_charset_decode_failed')
            continue
        if not declared and any(value >= 128 for value in data):
            warnings.append('charset_missing_inferred')
        elif declared and codec != declared and not (codec.endswith('-sig') and declared == 'utf-8'):
            warnings.append('charset_fallback_used')
        if '\ufffd' in text:
            warnings.append('replacement_character_present')
        return text, list(dict.fromkeys(warnings)), codec
    warnings.append('decode_replacement_used')
    return data.decode('utf-8', errors='replace'), list(dict.fromkeys(warnings)), 'utf-8-replace'


class _ExtractionLimit(ValueError):
    pass


def _archive(data):
    archive = zipfile.ZipFile(io.BytesIO(data))
    infos = archive.infolist()
    if len(infos) > MAX_ZIP_MEMBERS or sum(item.file_size for item in infos) > MAX_ZIP_BYTES:
        archive.close()
        raise _ExtractionLimit('archive_expansion_limit')
    seen = set()
    for item in infos:
        name = item.filename
        if (name in seen or name.startswith(('/', '\\')) or '\\' in name
                or '..' in PurePosixPath(name).parts or ':' in name):
            archive.close()
            raise _ExtractionLimit('unsafe_archive_structure')
        seen.add(name)
        if item.flag_bits & 1:
            archive.close()
            raise _ExtractionLimit('encrypted_archive')
        if item.file_size > MAX_XML_BYTES or item.file_size > max(1, item.compress_size) * MAX_ZIP_RATIO:
            archive.close()
            raise _ExtractionLimit('archive_expansion_limit')
    return archive


def _xml(archive, name):
    with archive.open(name) as source:
        raw = source.read(MAX_XML_BYTES + 1)
    if len(raw) > MAX_XML_BYTES:
        raise _ExtractionLimit('xml_size_limit')
    # Reject DTDs/entities before parsing, including UTF-16/32 encodings.
    sniff = raw.replace(b'\x00', b'').upper()
    if b'<!DOCTYPE' in sniff or b'<!ENTITY' in sniff:
        raise _ExtractionLimit('xml_dtd_not_allowed')
    return ET.fromstring(raw)


def _local(tag):
    return tag.rsplit('}', 1)[-1]


class _Text:
    def __init__(self, maximum):
        self.maximum = maximum
        self.parts, self.locations = [], []
        self.length = 0
        self.truncated = False

    def add(self, value, location):
        if not value:
            return True
        separator = '\n' if self.parts else ''
        room = self.maximum - self.length - len(separator)
        if room <= 0 or len(self.locations) >= MAX_LOCATIONS:
            self.truncated = True
            return False
        piece = value[:room]
        start = self.length + len(separator)
        self.parts.append(separator + piece)
        self.length = start + len(piece)
        self.locations.append({**location, 'start': start, 'end': self.length})
        if len(piece) < len(value):
            self.truncated = True
            return False
        return True

    @property
    def text(self):
        return ''.join(self.parts)


def _office(data, extension, result, out):
    with _archive(data) as archive:
        names = set(archive.namelist())
        if extension == '.docx':
            if 'word/document.xml' not in names:
                raise ValueError('missing_office_document')
            document = _xml(archive, 'word/document.xml')
            for number, paragraph in enumerate((n for n in document.iter() if _local(n.tag) == 'p'), 1):
                values = []
                for node in paragraph.iter():
                    tag = _local(node.tag)
                    if tag == 't': values.append(node.text or '')
                    elif tag == 'tab': values.append('\t')
                    elif tag in {'br', 'cr'}: values.append('\n')
                if not out.add(''.join(values), {'part': 'word/document.xml', 'paragraph': number}):
                    break
            result['warnings'].append('office_layout_and_embedded_objects_not_extracted')
            if any(name.startswith(('word/header', 'word/footer', 'word/footnotes', 'word/endnotes', 'word/comments'))
                   or name.startswith('word/embeddings/') for name in names):
                result['warnings'].append('additional_document_parts_not_extracted')
        else:
            if 'xl/workbook.xml' not in names:
                raise ValueError('missing_office_workbook')
            strings = []
            if 'xl/sharedStrings.xml' in names:
                root = _xml(archive, 'xl/sharedStrings.xml')
                strings = [''.join(n.text or '' for n in item.iter() if _local(n.tag) == 't')
                           for item in root if _local(item.tag) == 'si']
            sheet_files = sorted(name for name in names if re.fullmatch(r'xl/worksheets/sheet\d+\.xml', name))
            if not sheet_files:
                raise ValueError('missing_office_sheets')
            for name in sheet_files:
                root = _xml(archive, name)
                for cell in (node for node in root.iter() if _local(node.tag) == 'c'):
                    address, kind = cell.get('r', ''), cell.get('t', '')
                    value_node = next((n for n in cell if _local(n.tag) == 'v'), None)
                    value = value_node.text or '' if value_node is not None else ''
                    formula = next((n for n in cell if _local(n.tag) == 'f'), None)
                    if kind == 's' and value:
                        try: value = strings[int(value)] if int(value) >= 0 else '[invalid shared string]'
                        except (ValueError, IndexError):
                            value = '[invalid shared string]'
                            result['warnings'].append('invalid_shared_string')
                    elif kind == 'inlineStr':
                        value = ''.join(n.text or '' for n in cell.iter() if _local(n.tag) == 't')
                    if formula is not None:
                        result['warnings'].append('formula_not_evaluated_cached_value_only')
                        if not value: value = '[formula has no cached value]'
                    if not out.add(value, {'part': name, 'cell': address, 'cached_formula': formula is not None}):
                        break
                if out.truncated: break
            result['warnings'].append('spreadsheet_styles_dates_and_objects_not_interpreted')
        # Text-only Office coverage cannot claim full document fidelity.
        result['status'] = 'partial'
        result['reason'] = 'office_text_only'


def _pdf(data, result, out):
    try:
        from pypdf import PdfReader, apply_configuration
    except ImportError:
        result.update(status='unsupported', reason='pdf_extractor_unavailable')
        return
    # Context-local pypdf limits; no global mutation and no image/OCR subprocess.
    # Use the caller's bounded worker for wall-clock and process resource limits.
    with apply_configuration(
            maximum_declared_stream_length=MAX_PDF_STREAM_BYTES,
            array_based_stream_maximum_output_length=MAX_PDF_STREAM_BYTES,
            zlib_maximum_output_length=MAX_PDF_STREAM_BYTES,
            zlib_maximum_recovery_input_length=1_000_000,
            lzw_maximum_output_length=MAX_PDF_STREAM_BYTES,
            run_length_maximum_output_length=MAX_PDF_STREAM_BYTES,
            page_tree_maximum_entries=2000, page_tree_maximum_depth=30,
            xform_maximum_invocations_per_extraction=200, jbig2dec_binary=None):
        try:
            reader = PdfReader(io.BytesIO(data), strict=True)
            if reader.is_encrypted:
                result.update(status='not_read', reason='encrypted_pdf')
                return
            count = len(reader.pages)
            result['total_pages'] = count
            result['pages_read'] = 0
            blank = []
            result.update(status='complete', reason='pdf_text_extracted')
            for index in range(min(count, MAX_PDF_PAGES)):
                try:
                    value = reader.pages[index].extract_text() or ''
                except Exception:
                    result['warnings'].append('pdf_page_extraction_failed:' + str(index + 1))
                    continue
                result['pages_read'] += 1
                if not value.strip(): blank.append(index + 1)
                if not out.add(value, {'page': index + 1}): break
            if blank:
                result['pages_without_extractable_text'] = blank
                result['warnings'].append('pages_without_text_may_be_scanned_or_blank_ocr_not_enabled')
            if count > MAX_PDF_PAGES:
                result['warnings'].append('pdf_page_limit')
            if result['warnings']:
                result.update(status='partial' if out.text else 'not_read',
                              reason='pdf_partial_text' if out.text else 'no_extractable_pdf_text')
        except Exception:
            result.update(status='partial' if out.text else 'failed', reason='invalid_or_resource_limited_pdf')


def extract_attachment_text(data: bytes, *, filename: str = '', mime_type: str = 'application/octet-stream',
                            charset: str | None = None, max_chars: int = MAX_TEXT_CHARS) -> dict:
    """Return explicit extraction coverage and bounded text with source locations.

    PDF uses bounded pypdf extraction when installed. Office ZIPs are never unpacked
    onto disk. Formulas, macros, links and embedded code are never executed.
    """
    if not isinstance(data, bytes) or type(max_chars) is not int or not 1 <= max_chars <= MAX_TEXT_CHARS:
        raise ValueError('invalid attachment extraction input')
    result = {'status': 'unsupported', 'reason': 'unsupported_format', 'text': '', 'locations': [],
              'warnings': [], 'text_extracted': False, 'size': len(data),
              'sha256': hashlib.sha256(data).hexdigest(), 'coverage': 'attachment_text_only'}
    if len(data) > MAX_ATTACHMENT_BYTES:
        return {**result, 'status': 'not_read', 'reason': 'attachment_byte_limit'}
    extension = PurePosixPath(filename.replace('\\', '/')).suffix.lower()
    kind = mime_type.lower().split(';', 1)[0].strip()
    if kind == 'message/rfc822':
        return {**result, 'status': 'not_read', 'reason': 'forwarded_message_separate_from_parent'}
    if kind.startswith('image/') or extension in {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.gif', '.webp', '.bmp'}:
        return {**result, 'reason': 'image_ocr_not_enabled'}
    if extension in {'.doc', '.xls', '.ppt', '.docm', '.xlsm', '.pptm'}:
        return {**result, 'reason': 'legacy_or_macro_office_not_supported'}
    out = _Text(max_chars)
    try:
        office = {'.docx', '.xlsx'}
        if kind == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document': extension = '.docx'
        if kind == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': extension = '.xlsx'
        if kind == 'application/pdf' or extension == '.pdf':
            _pdf(data, result, out)
        elif extension in office:
            _office(data, extension, result, out)
        elif (kind.startswith('text/') or kind in {'application/json', 'application/xml', 'application/csv', 'application/ics'}
              or extension in {'.txt', '.csv', '.json', '.xml', '.html', '.htm', '.md', '.log'}):
            text, warnings, codec = decode_text(data, charset)
            result['warnings'].extend(warnings)
            result['charset_used'] = codec
            if kind == 'text/html' or extension in {'.html', '.htm'}:
                too_long = len(text) > HTML_INPUT_LIMIT
                text, rows = html_to_structured_text(text)
                result['table_rows'] = [row for row in rows if row['end'] <= max_chars]
                if too_long: result['warnings'].append('html_input_limit')
            out.add(text, {'kind': 'text'})
            result.update(status='partial' if result['warnings'] else 'complete',
                          reason='decode_or_extraction_warning' if result['warnings'] else 'text_extracted')
    except _ExtractionLimit as exc:
        result.update(status='not_read', reason=str(exc))
    except (zipfile.BadZipFile, ET.ParseError, KeyError, ValueError, OSError, RuntimeError):
        result.update(status='failed', reason='invalid_or_unreadable_attachment')
    result.update(text=out.text, locations=out.locations, text_extracted=bool(out.text))
    result['warnings'] = list(dict.fromkeys(result['warnings']))
    if out.truncated:
        result.update(status='partial', reason='attachment_text_limit')
        result['warnings'].append('attachment_text_limit')
    return result
