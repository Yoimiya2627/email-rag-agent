"""Real HTML normalization/chunking/reconstruction; only storage is substituted."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from core.cleaner import clean_email, html_to_structured_text
from core.chunker import chunk_email
from core import embedder
from core.html_tables import TableBuilder
from models.schemas import Email
import config.settings as cfg


def mail(body):
    return Email(id='table', subject='Quote', sender='a@example.com', recipients=[],
                 date='2026-09-10', body=body, body_format='html')


class TableFidelityTests(unittest.TestCase):
    def roundtrip(self, html, size=800):
        normalized = clean_email(mail(html))
        self.assertEqual(clean_email(normalized).model_dump(), normalized.model_dump())
        with patch.multiple(cfg, CHUNK_SIZE=size, CHUNK_OVERLAP=5, MIN_CHUNK_SIZE=10):
            chunks = chunk_email(normalized)
        collection = SimpleNamespace(get=lambda **kw: {
            'ids': [c.chunk_id for c in reversed(chunks)],
            'documents': [c.content for c in reversed(chunks)],
            'metadatas': [{**c.metadata, 'chunk_index': c.chunk_index} for c in reversed(chunks)]})
        with patch.object(embedder, '_get_collection', return_value=collection):
            result = embedder.get_indexed_email(normalized.id)
        self.assertTrue(result['reconstruction_exact'])
        self.assertEqual(result['body'], 'Subject: Quote\n\n' + normalized.body)
        self.assertTrue(all(len(c.content) <= size for c in chunks))
        return normalized, chunks

    def test_headers_rows_and_merged_source_relationships(self):
        email, chunks = self.roundtrip('<p>Quote</p><table><tr><th>Item</th><th colspan="2">USD</th></tr>'
            '<tr><td rowspan="2">Plan A</td><td>3</td><td>700</td></tr>'
            '<tr><td>4</td><td>900</td></tr></table><p>Approved</p>')
        self.assertEqual(len(email.table_rows), 3)
        cells = email.table_rows[1]['cells']
        self.assertEqual(cells[1]['headers'], ['USD'])
        self.assertEqual(cells[0]['rowspan'], 2)
        carried = email.table_rows[2]['cells'][-1]
        self.assertTrue(carried['carried'])
        self.assertEqual(carried['source_id'], cells[0]['source_id'])
        self.assertEqual(email.table_rows[2]['cells'][0]['column'], 2)
        self.assertIn('"USD"="700"', email.body)
        for row in email.table_rows:
            containing = [c for c in chunks if c.metadata['source_start'] <= row['start'] + len('Subject: Quote\n\n')
                          and c.metadata['source_end'] >= row['end'] + len('Subject: Quote\n\n')]
            self.assertEqual(len(containing), 1)

    def test_multiple_tables_unique_source_ids_and_escape_literals(self):
        email, _ = self.roundtrip('<table><tr><th>A|B</th></tr><tr><td>1 &lt; 2 "x"</td></tr></table>'
                                 '<p>between</p><table><tr><td>second</td></tr></table>')
        self.assertEqual([r['table_id'] for r in email.table_rows], ['t1', 't1', 't2'])
        self.assertEqual(email.table_rows[1]['cells'][0]['text'], '1 < 2 "x"')
        self.assertIn('between', email.body)
        _, namespaced = html_to_structured_text('<table><tr><td>A</td></tr></table>', 'p3t')
        self.assertEqual(namespaced[0]['cells'][0]['source_id'], 'p3t1:r1:c1')

    def test_row_headers_do_not_contaminate_future_rows(self):
        email, _ = self.roundtrip('<table><tr><th>Item</th><th>Price</th></tr>'
            '<tr><th scope="row">Apples</th><td>10</td></tr>'
            '<tr><th scope="row">Pears</th><td>20</td></tr></table>')
        self.assertEqual(email.table_rows[1]['cells'][1]['headers'], ['Price', 'Apples'])
        self.assertEqual(email.table_rows[2]['cells'][0]['headers'], ['Item'])
        self.assertEqual(email.table_rows[2]['cells'][1]['headers'], ['Price', 'Pears'])

    def test_oversized_cell_fragments_explicit_context_and_exact_reconstruction(self):
        email, chunks = self.roundtrip('<table><tr><th>Amount</th></tr><tr><td>' + 'abcdef' * 2000
                                       + '</td></tr></table>', size=180)
        fragments = [row for c in chunks for row in json.loads(c.metadata['table_context']) if row['partial_row']]
        self.assertGreater(len(fragments), 20)
        self.assertTrue(all(row['row_id'] == 't1:r2' for row in fragments))
        self.assertTrue(all(row['cells'][0]['headers'] == ['Amount'] for row in fragments))
        self.assertEqual(email.table_rows[1]['cells'][0]['text'], 'abcdef' * 2000)

    def test_nested_hidden_and_unclosed_tables_mark_incomplete(self):
        email, _ = self.roundtrip('<table><tr><td>outer<table><tr><td>inner</td></tr></table>end</td>'
            '<td hidden>secret</td><td style="display: none">secret2</td><td><script>secret3</script>visible')
        self.assertNotIn('secret', email.body)
        self.assertIn('inner', email.body)
        self.assertIn('outer', email.body)
        self.assertIn('nested_table_flattened', email.body)
        self.assertIn('unclosed_table', email.body)

    def test_nested_same_tag_hidden_region_stays_hidden(self):
        email, _ = self.roundtrip('<div hidden><div>inner</div>secret</div><p>VISIBLE</p>'
                                 '<table><tr><td>visible<div hidden><div>x</div>secret2</div></td></tr></table>')
        self.assertNotIn('secret', email.body)
        self.assertNotIn('inner', email.body)
        self.assertIn('VISIBLE', email.body)

    def test_span_and_cell_limits_are_explicit(self):
        with patch.multiple(TableBuilder, MAX_CELLS=3, MAX_TEXT=30):
            email, _ = self.roundtrip('<table><tr><td colspan="999999999999">' + 'z' * 100
                + '</td></tr><tr><td>A</td><td>B</td><td>C</td><td>D</td></tr></table>')
        self.assertIn('span_limit', email.body)
        self.assertIn('cell_limit', email.body)
        self.assertIn('text_limit', email.body)
        self.assertLessEqual(sum(len(r['cells']) for r in email.table_rows), 3)

    def test_input_and_render_expansion_limits_are_explicit(self):
        with patch('core.cleaner.HTML_INPUT_LIMIT', 100):
            email, _ = self.roundtrip('<p>visible</p><table><tr><td>' + 'z' * 200 + '</td></tr></table>')
        self.assertIn('HTML extraction incomplete: input_limit', email.body)
        self.assertIn('unclosed_table', email.body)
        email, _ = self.roundtrip('<table><tr><th>' + 'H' * 160 + '</th></tr>'
                                 + '<tr><td>x</td></tr>' * 1000 + '</table>')
        self.assertIn('render_limit', email.body)
        self.assertLess(len(email.table_rows), 1001)

    def test_global_table_limits_keep_indexed_incomplete_marker(self):
        for option, maximum, reason in [('HTML_TABLE_LIMIT', 2, 'table_limit'),
                ('HTML_TABLE_ROW_LIMIT', 2, 'total_table_row_limit'),
                ('HTML_TABLE_OUTPUT_LIMIT', 160, 'total_table_output_limit')]:
            with self.subTest(option=option), patch('core.cleaner.' + option, maximum):
                email, chunks = self.roundtrip('<table><tr><td>x</td></tr></table>' * 10 + '<p>tail</p>')
            self.assertIn(reason, email.body)
            self.assertEqual(email.table_rows[-1]['status'], reason)
            self.assertIn('remaining content omitted', email.body)
            self.assertTrue(any(reason in c.content for c in chunks))

    def test_omitted_cell_and_row_endtags_preserve_values(self):
        email, _ = self.roundtrip('<table><tr><th>Name<th>Cost<tr><td>A<td>70<tr><td>B<td>90</table>')
        self.assertEqual(len(email.table_rows), 3)
        self.assertEqual([c['text'] for c in email.table_rows[2]['cells']], ['B', '90'])
        self.assertIn('implicit_row_close', email.body)


if __name__ == '__main__':
    unittest.main()
