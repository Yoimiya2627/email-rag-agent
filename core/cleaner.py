import re
import logging
from html.parser import HTMLParser
from models.schemas import Email

logger = logging.getLogger(__name__)
HTML_INPUT_LIMIT = 4_000_000
HTML_TABLE_LIMIT = 1024
HTML_TABLE_ROW_LIMIT = 4096
HTML_TABLE_OUTPUT_LIMIT = 4_000_000

_MULTI_NEWLINES = re.compile(r"\n{3,}")
_MULTI_SPACES = re.compile(r" {2,}")

# Only the explicit plain-text signature delimiter is evidence of a signature.
# Greetings such as Thanks/谢谢 and bare '--' can introduce business content.
_SIG_RE = re.compile(r"^-- $", re.MULTILINE)


class _BodyHTMLParser(HTMLParser):
    _blocks = {"p", "div", "section", "article", "header", "footer", "blockquote",
               "li", "ul", "ol", "table", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "hr"}
    _head_elements = {"base", "basefont", "bgsound", "link", "meta", "title",
                      "noscript", "noframes", "style", "template", "script"}

    def __init__(self, table_prefix='t'):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = []
        self.in_head = False
        self.pre_depth = 0
        self.table = None
        self.table_count = 0
        self.table_records = []
        self.hidden_elements = []
        self.table_prefix = table_prefix
        self.table_characters = 0
        self.output_stopped = False

    def handle_starttag(self, tag, attrs):
        if self.output_stopped:
            return
        attributes = dict(attrs)
        void = tag in {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'}
        style = re.sub(r'\s+', '', attributes.get('style', '') or '').lower()
        if self.hidden_elements or 'hidden' in attributes or attributes.get('aria-hidden') == 'true' or 'display:none' in style or 'visibility:hidden' in style:
            if not void:
                self.hidden_elements.append(tag)
            return
        if tag == "head" and not self.hidden:
            self.in_head = True
            return
        # HTML permits omission of </head> (and <body>). A body element ends
        # the metadata section; do not leave its text hidden indefinitely.
        if self.in_head and not self.hidden and tag not in self._head_elements | {"html"}:
            self.in_head = False
        if tag in {"script", "style", "title", "template"}:
            self.hidden.append(tag)
        if self.hidden or self.in_head:
            return
        if self.table is not None:
            self.table.start(tag, attrs)
            return
        if tag == 'table':
            from core.html_tables import TableBuilder
            self.table_count += 1
            if self.table_count > HTML_TABLE_LIMIT:
                self._stop_output('table_limit')
                return
            self.table = TableBuilder(self.table_count, self.table_prefix)
            return
        if tag in self._blocks:
            self.parts.append("\n\n")
        elif tag == "br":
            self.parts.append("\n")
        elif tag in {"td", "th"}:
            self.parts.append("\t")
        if tag == "pre":
            self.pre_depth += 1

    def handle_endtag(self, tag):
        if self.output_stopped:
            return
        if self.hidden_elements:
            if tag in self.hidden_elements:
                index = len(self.hidden_elements) - 1 - self.hidden_elements[::-1].index(tag)
                del self.hidden_elements[index:]
            return
        if self.hidden:
            if tag == self.hidden[-1]:
                self.hidden.pop()
            return
        if tag == "head":
            self.in_head = False
            return
        if self.table is not None:
            if self.table.end(tag):
                self._finish_table()
            return
        if tag == "pre":
            self.pre_depth = max(0, self.pre_depth - 1)
        if tag in self._blocks:
            self.parts.append("\n\n")

    def handle_data(self, data):
        if self.output_stopped:
            return
        if self.hidden or self.hidden_elements:
            return
        if self.in_head:
            if not data.strip():
                return
            # Visible text also starts an implicit body after head metadata.
            self.in_head = False
        if self.table is not None:
            self.table.data(data)
            return
        self.parts.append(data if self.pre_depth else re.sub(r"\s+", " ", data))

    def _finish_table(self, unclosed=False):
        records = self.table.render(unclosed)
        for record in records:
            if len(self.table_records) >= HTML_TABLE_ROW_LIMIT:
                self._stop_output('total_table_row_limit')
                break
            if self.table_characters + len(record['text']) > HTML_TABLE_OUTPUT_LIMIT:
                self._stop_output('total_table_output_limit')
                break
            self.table_records.append(record)
            self.table_characters += len(record['text'])
            self.parts.append('\n\n' + record['text'] + '\n\n')
        self.table = None

    def _stop_output(self, reason):
        self.output_stopped = True
        table_id = f'{self.table_prefix}{self.table_count}'
        marker = f'[HTML extraction incomplete: {reason}; remaining content omitted]'
        self.table_records.append({'table_id': table_id, 'row_id': table_id + ':limit',
                                   'cells': [], 'status': reason, 'text': marker})
        self.parts.append('\n\n' + marker + '\n\n')


def html_to_text(text: str) -> str:
    """Convert explicitly identified HTML once, preserving block boundaries."""
    return html_to_structured_text(text)[0]


def html_to_structured_text(text: str, table_prefix: str = 't') -> tuple[str, list[dict]]:
    """Return normalized visible text and body-relative canonical table row spans."""
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,40}', table_prefix):
        raise ValueError('invalid table_prefix')
    parser = _BodyHTMLParser(table_prefix)
    limit = HTML_INPUT_LIMIT
    parser.feed(text[:limit])
    parser.close()
    if parser.table is not None:
        parser._finish_table(unclosed=True)
    if len(text) > limit:
        parser.parts.append('\n\n[HTML extraction incomplete: input_limit]\n\n')
    value = "".join(parser.parts).replace("\xa0", " ")
    value = re.sub(r"[^\S\n]*\n[^\S\n]*", "\n", value)
    value = _normalize_whitespace(_MULTI_NEWLINES.sub("\n\n", value).strip())
    rows = []
    cursor = 0
    for record in parser.table_records:
        line = _normalize_whitespace(record['text'])
        start = value.find(line, cursor)
        if start < 0:
            raise ValueError('canonical table row lost during normalization')
        rows.append({key: item for key, item in record.items() if key != 'text'} |
                    {'start': start, 'end': start + len(line)})
        cursor = start + len(line)
    return value, rows


def _strip_html(text: str) -> str:
    return html_to_text(text)


def _remove_signature(text: str) -> str:
    match = _SIG_RE.search(text)
    if match and text[:match.start()].strip():
        text = text[: match.start()]
    return text


def _normalize_whitespace(text: str) -> str:
    text = _MULTI_NEWLINES.sub("\n\n", text)
    text = _MULTI_SPACES.sub(" ", text)
    return text.strip()


def clean_body(body: str, body_format: str = "plain") -> str:
    if body_format == "html":
        body = html_to_text(body)
    elif body_format != "plain":
        raise ValueError("body_format must be plain or html")
    body = body.replace("\r\n", "\n").replace("\r", "\n")
    body = _remove_signature(body)
    body = _normalize_whitespace(body)
    return body


def clean_email(email: Email) -> Email:
    if email.body_format == 'html':
        body, rows = html_to_structured_text(email.body)
    else:
        body, rows = email.body, email.table_rows
    cleaned = clean_body(body)
    # Rebase actual row strings after trimming/signature normalization. A row
    # removed by the explicit signature delimiter is no longer indexed.
    rebased = []
    cursor = 0
    for row in rows:
        line = _normalize_whitespace(body[row['start']:row['end']])
        start = cleaned.find(line, cursor)
        if start >= 0 and line:
            rebased.append({**row, 'start': start, 'end': start + len(line)})
            cursor = start + len(line)
    return email.model_copy(update={'body': cleaned, 'body_format': 'plain', 'table_rows': rebased})
