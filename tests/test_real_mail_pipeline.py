"""Cross-module acceptance probes using only synthetic mail and temporary files."""
import base64
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import config.settings as cfg
from agents.gmail_readonly import GmailReadOnlyProvider, gmail_message_to_email
from core.cleaner import clean_email, HTML_INPUT_LIMIT
from core.chunker import chunk_email
from core import embedder
from core.evidence import table_context_note, evidence_text
from core.generator import build_context
from models.schemas import Email, SearchResult
from scripts.sync_gmail_readonly import sync_gmail_to_json
from scripts.index_emails import prepare_email_chunks


TABLE = ('<table><tr><th>项目</th><th>数量</th><th>金额</th></tr>'
         '<tr><td rowspan="2">方案 A</td><td>2</td><td>500</td></tr>'
         '<tr><td>3</td><td>700</td></tr></table>')


def part(kind, text):
    return {"mimeType": kind, "body": {"data": base64.urlsafe_b64encode(text.encode()).decode()}}


def mail(payload):
    return {"id": "quote", "internalDate": "1710000000000", "payload": payload}


class RealMailPipelineTests(unittest.TestCase):
    def test_html_tables_supplement_plain_without_duplicating_html_prose(self):
        source = mail({"mimeType": "multipart/alternative", "parts": [
            part("text/plain", "Please use the revised quote."),
            part("text/html", "<p>HTML prose duplicate</p>" + TABLE)]})
        converted = gmail_message_to_email(source)
        self.assertIn("Please use the revised quote.", converted.body)
        self.assertNotIn("HTML prose duplicate", converted.body)
        self.assertIn("HTML alternative", converted.body)
        row = converted.table_rows[-1]
        text = converted.body[row["start"]:row["end"]]
        self.assertIn('"金额"="700"', text)
        self.assertIn('"方案 A"', text)
        self.assertEqual(row["origin"], "html_alternative")
        cleaned = clean_email(Email.model_validate(converted.model_dump()))
        self.assertEqual(cleaned.model_dump(), clean_email(cleaned).model_dump())

    def test_mixed_html_parts_have_distinct_source_ids_and_valid_offsets(self):
        converted = gmail_message_to_email(mail({"mimeType": "multipart/mixed", "parts": [
            part("text/html", TABLE), part("text/plain", "middle"), part("text/html", TABLE)]}))
        identifiers = {row["table_id"] for row in converted.table_rows}
        self.assertEqual(len(identifiers), 2)
        for row in converted.table_rows:
            self.assertIn(row["row_id"], converted.body[row["start"]:row["end"]])

    def test_truncated_html_alternative_does_not_silently_drop_status(self):
        source = mail({"mimeType": "multipart/alternative", "parts": [
            part("text/plain", "short"), part("text/html", "x" * (HTML_INPUT_LIMIT + 1) + TABLE)]})
        converted = gmail_message_to_email(source)
        self.assertIn("input_limit", converted.body)
        self.assertNotIn("700", converted.body)

    def test_sync_archive_json_clean_chunks_and_exact_detail_roundtrip(self):
        source = mail(part("text/html", TABLE))
        class Provider(GmailReadOnlyProvider):
            def list_message_ids(self, **kwargs): return ["quote"]
            def get_message(self, identifier): return source
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus, state = root / "corpus.json", root / "state.json"
            result = sync_gmail_to_json(Provider(), corpus, state, "", 10)
            self.assertEqual((result["added"], result["failed"]), (1, 0))
            capture = json.loads((Path(str(corpus) + ".raw") /
                                 (hashlib.sha256(b"quote").hexdigest() + ".json")).read_text(encoding="utf-8"))
            self.assertEqual(capture["message"], source)
            with patch.multiple(cfg, CHUNK_SIZE=100, CHUNK_OVERLAP=20, MIN_CHUNK_SIZE=10):
                emails, chunks = prepare_email_chunks(corpus)
            normalized = clean_email(emails[0])
            rows = [{"content": chunk.content, "metadata": chunk.metadata} for chunk in chunks]
            with patch.object(embedder, "get_email_chunks", return_value=rows):
                detail = embedder.get_indexed_email("gmail_quote")
            self.assertTrue(detail["reconstruction_exact"])
            self.assertEqual(detail["body"], f"Subject: {normalized.subject}\n\n{normalized.body}")
            partial = next(chunk for chunk in chunks if any(
                row["partial_row"] for row in json.loads(chunk.metadata.get("table_context", "[]"))))
            context = build_context([SearchResult(chunk_id=partial.chunk_id, email_id=partial.email_id,
                                                   content=partial.content, metadata=partial.metadata, score=1)])
            self.assertIn('"partial_row":true', context)
            self.assertIn("部分行", context)

    def test_table_note_is_bounded_and_plain_evidence_unchanged(self):
        self.assertEqual(table_context_note({}), "")
        for raw in ["broken", "{}", '[{"cells":null}]', '"bad"', "x" * 200001]:
            self.assertIn("不完整", table_context_note({"table_context": raw}))
        raw = json.dumps([{"table_id": "t1", "row_id": "t1:r1", "status": "complete", "partial_row": True,
                           "cells": [{"headers": ["x" * 5000]}]}])
        self.assertLess(len(table_context_note({"table_context": raw})), 2000)

    def test_excerpt_and_generation_budget_mark_atomic_rows_partial(self):
        from agents.tools import _format_hit
        metadata = {"table_context": json.dumps([{"table_id": "t1", "row_id": "t1:r1",
                    "status": "complete", "partial_row": False, "cells": []}])}
        result = SearchResult(chunk_id="e_0", email_id="e", content="x" * 500, metadata=metadata, score=1)
        hit = _format_hit(result)
        self.assertTrue(hit["snippet_truncated"])
        self.assertIn('"partial_row":true', hit["table_context"])
        context = evidence_text(result.content, metadata, max_chars=300)
        self.assertLessEqual(len(context), 300)
        self.assertIn('"partial_row":true', context)
        self.assertEqual(evidence_text(result.content, metadata, max_chars=10), "")
        self.assertIn('"partial_row": false', metadata["table_context"])


if __name__ == "__main__":
    unittest.main()
