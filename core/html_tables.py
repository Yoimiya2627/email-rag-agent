"""Bounded, source-labelled table rendering for retrieval (not a browser DOM)."""
import json


class TableBuilder:
    MAX_CELLS = 1024
    MAX_COLUMNS = 64
    MAX_ROWS = 1024
    MAX_TEXT = 200_000

    def __init__(self, number, prefix='t'):
        self.number = number
        self.table_id = f'{prefix}{number}'
        self.rows = []
        self.row = None
        self.cell = None
        self.depth = 1
        self.count = 0
        self.text_count = 0
        self.issues = set()

    def start(self, tag, attrs):
        if tag == 'table':
            self.depth += 1
            self.issues.add('nested_table_flattened')
            self.data(' [nested table] ')
            return
        if self.depth > 1:
            if tag in {'td', 'th', 'tr', 'br'}:
                self.data(' ')
            return
        if tag == 'tr':
            if self.row is not None:
                self.issues.add('implicit_row_close')
            self._finish_row()
            self.row = []
        elif tag in {'td', 'th'}:
            self._finish_cell()
            if self.row is None:
                self.issues.add('missing_row')
                self.row = []
            attrs = dict(attrs)
            spans = {}
            for key in ('rowspan', 'colspan'):
                raw = attrs.get(key, '1')
                try:
                    value = int(raw)
                except (ValueError, TypeError):
                    value = 1
                    self.issues.add('invalid_span')
                if not 1 <= value <= self.MAX_COLUMNS:
                    self.issues.add('span_limit')
                spans[key] = min(self.MAX_COLUMNS, max(1, value))
            scope = (attrs.get('scope') or '').lower()
            if scope not in {'', 'row', 'col'}:
                self.issues.add('unsupported_header_scope')
            self.cell = dict(text='', header=tag == 'th', scope=scope,
                             html_id=attrs.get('id', ''), header_refs=(attrs.get('headers') or '').split(), **spans)
        elif tag in {'p', 'div', 'br', 'li'}:
            self.data(' ')

    def end(self, tag):
        if tag == 'table':
            self.depth -= 1
            if self.depth == 0:
                self._finish_row()
                return True
        if self.depth == 1:
            if tag in {'td', 'th'}:
                self._finish_cell()
            elif tag == 'tr':
                self._finish_row()
        return False

    def data(self, text):
        remaining = self.MAX_TEXT - self.text_count
        if len(text) > remaining:
            self.issues.add('text_limit')
        text = text[:max(0, remaining)]
        self.text_count += len(text)
        if self.cell is not None:
            self.cell['text'] += text
        elif text.strip():
            # Captions and malformed direct text remain evidence, not silence.
            self.issues.add('text_outside_cell')
            if self.row is None:
                self.row = []
            self.cell = dict(text=text, header=False, rowspan=1, colspan=1)

    def _finish_cell(self):
        if self.cell is None:
            return
        if self.count < self.MAX_CELLS and len(self.rows) < self.MAX_ROWS:
            self.cell['text'] = ' '.join(self.cell['text'].split())
            self.row.append(self.cell)
            self.count += 1
        else:
            self.issues.add('cell_limit')
        self.cell = None

    def _finish_row(self):
        self._finish_cell()
        if self.row:
            if len(self.rows) < self.MAX_ROWS:
                self.rows.append(self.row)
            else:
                self.issues.add('row_limit')
        self.row = None

    def render(self, unclosed=False):
        self._finish_row()
        if unclosed:
            self.issues.add('unclosed_table')
        occupied = {}
        headers = {}
        records = []
        rendered_cost = 0
        explicit_headers = {}
        ambiguous_ids = set()
        for row in self.rows:
            for cell in row:
                identifier = cell.get('html_id')
                if identifier:
                    if identifier in explicit_headers:
                        ambiguous_ids.add(identifier)
                    explicit_headers[identifier] = cell
        render_limit = min(1_000_000, max(100_000, self.text_count * 8 + self.count * 256))
        for r, row in enumerate(self.rows, 1):
            cells = []
            c = 1
            mixed_row = any(not cell['header'] for cell in row)
            def is_row_header(cell):
                return cell['header'] and (cell.get('scope') == 'row' or
                                          (not cell.get('scope') and mixed_row))
            row_headers = list(dict.fromkeys(
                [entry['text'][:160] for until, entry in occupied.values()
                 if until >= r and entry.get('row_header')]
                + [cell['text'][:160] for cell in row if is_row_header(cell)]))
            if any(len(cell['text']) > 160 for cell in row if is_row_header(cell)):
                self.issues.add('header_context_shortened')
            for cell in row:
                while c in occupied and occupied[c][0] >= r:
                    c += 1
                width = cell['colspan']
                # Require a contiguous free interval for a merged cell.
                while c <= self.MAX_COLUMNS and any(
                        col in occupied and occupied[col][0] >= r
                        for col in range(c, min(c + width, self.MAX_COLUMNS + 1))):
                    c += 1
                if c + width - 1 > self.MAX_COLUMNS:
                    self.issues.add('column_limit')
                    break
                source_id = f'{self.table_id}:r{r}:c{c}'
                entry = {**cell, 'source_id': source_id, 'row': r, 'column': c}
                entry['row_header'] = is_row_header(cell)
                context_headers = list(dict.fromkeys(
                    [item for col in range(c, c + width) for item in headers.get(col, [])]
                    + ([] if is_row_header(cell) else row_headers)))
                if cell.get('header_refs'):
                    context_headers = []
                    for reference in cell['header_refs'][:8]:
                        header = explicit_headers.get(reference)
                        if reference in ambiguous_ids or not header or not header['header']:
                            self.issues.add('unresolved_header_reference')
                        else:
                            context_headers.append(header['text'][:160])
                            if len(header['text']) > 160:
                                self.issues.add('header_context_shortened')
                    if len(cell['header_refs']) > 8:
                        self.issues.add('header_context_limit')
                if len(context_headers) > 8:
                    self.issues.add('header_context_limit')
                entry['headers'] = context_headers[:8]
                cells.append(entry)
                for col in range(c, c + width):
                    occupied[col] = (r + cell['rowspan'] - 1, entry)
                    if cell['header'] and not is_row_header(cell) and cell.get('scope') in {'', 'col'}:
                        if len(cell['text']) > 160:
                            self.issues.add('header_context_shortened')
                        if len(headers.get(col, [])) >= 8:
                            self.issues.add('header_context_limit')
                        headers[col] = (headers.get(col, []) + [cell['text'][:160]])[-8:]
                c += width
            carried = {entry['source_id']: entry for until, entry in occupied.values()
                       if until >= r and entry['row'] < r}
            for entry in carried.values():
                if len(entry['text']) > 160:
                    self.issues.add('carried_context_shortened')
                cells.append({**entry, 'text': entry['text'][:160], 'carried': True})
            rendered_cost += sum(len(cell['text']) + sum(map(len, cell['headers'])) + 256 for cell in cells)
            if rendered_cost > render_limit:
                self.issues.add('render_limit')
                break
            records.append({'table_id': self.table_id, 'row_id': f'{self.table_id}:r{r}',
                            'cells': cells})
        # Status is included in the normalized text so downstream prompts see it.
        status = ','.join(sorted(self.issues)) or 'complete'
        output = []
        for record in records:
            record['status'] = status
            fields = []
            for cell in record['cells']:
                label = '/'.join(cell['headers']) or ('header' if cell['header'] else f"column {cell['column']}")
                relation = f" {cell['rowspan']}x{cell['colspan']}" if cell['rowspan'] != 1 or cell['colspan'] != 1 else ''
                carried = ' carried' if cell.get('carried') else ''
                fields.append(f"{cell['source_id']}{relation}{carried} " + json.dumps(label, ensure_ascii=False)
                              + '=' + json.dumps(cell['text'], ensure_ascii=False))
            record['text'] = ('[Table ' + record['table_id'] + ' row ' + record['row_id']
                              + ' status=' + status + '] '
                              + ' | '.join(fields))
            output.append(record)
        if not output:
            output.append({'table_id': self.table_id, 'row_id': f'{self.table_id}:r0',
                           'cells': [], 'status': status,
                           'text': f'[Table {self.table_id} status={status}; empty table]'})
        return output
