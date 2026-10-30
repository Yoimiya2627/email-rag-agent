"""Real input/SQLite preflight tests; synthetic embedding and Chroma leaves only."""
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import config.settings as cfg
import core.embedder as embedder
import core.ingestion as ingestion
from models.schemas import EmailChunk
from tests.index_store_helpers import MemoryClient
from scripts import index_emails as cli
from scripts.sync_gmail_readonly import index_email_json


def _record(identifier="new"):
    return {"id": identifier, "subject": "Valid", "sender": "a@example.com",
            "recipients": ["b@example.com"], "date": "2026-01-01",
            "body": "Valid email content"}


def _chunk(identifier, email_id):
    return EmailChunk(chunk_id=identifier, email_id=email_id, content="new content",
                      chunk_index=0, metadata={})


class FakeCollection:
    def __init__(self):
        self.rows = {
            "old_0": {"email_id": "old", "content": "previous head"},
            "old_1": {"email_id": "old", "content": "previous tail"},
            "omitted_0": {"email_id": "omitted", "content": "keep until success"},
        }
        self.events = []
        self.fail_upsert = None
        self.upserts = 0

    def count(self):
        return len(self.rows)

    def get(self, where=None, **kwargs):
        ids = [key for key, row in self.rows.items()
               if not where or row["email_id"] == where["email_id"]]
        return {"ids": ids, "metadatas": [{"email_id": self.rows[key]["email_id"]} for key in ids]}

    def upsert(self, ids, documents, metadatas, **kwargs):
        self.upserts += 1
        if self.upserts == self.fail_upsert:
            raise OSError("injected upsert failure")
        self.events.append(("upsert", list(ids)))
        for key, content, metadata in zip(ids, documents, metadatas):
            self.rows[key] = {"email_id": metadata["email_id"], "content": content}

    def delete(self, ids):
        self.events.append(("delete", list(ids)))
        for key in ids:
            self.rows.pop(key, None)


class IndexPreflightTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix="index_preflight_")))
        self.client = MemoryClient()
        self.stack.enter_context(patch.object(cfg, "CHROMA_PERSIST_DIR", str(self.directory / "chroma")))
        self.stack.enter_context(patch.object(embedder, "_get_client", return_value=self.client))
        self.stack.enter_context(patch.object(cfg, "EMBEDDING_MODEL_REVISION", "offline-v1", create=True))
        self.encode = self.stack.enter_context(patch.object(embedder, "embed_texts", side_effect=lambda texts: [[1., 0.] for _ in texts]))
        self.clear = self.stack.enter_context(patch.object(embedder, "clear_collection", side_effect=AssertionError("must not clear before replacement")))
        embedder.index_chunks([_chunk("old_0", "old"), _chunk("old_1", "old"), _chunk("omitted_0", "omitted")], replace=True)
        self.collection = embedder._get_collection()
        self.collection.events.clear()
        self.encode.reset_mock()
        self.path = self.directory / "emails.json"

    def invoke(self, entrypoint, path=None):
        path = path or self.path
        if entrypoint == "sync":
            return index_email_json(path, clear=True)
        with patch("sys.argv", ["index_emails.py", "--data-path", str(path), "--clear"]):
            return cli.main()

    def test_both_entrypoints_reject_bad_missing_empty_and_invalid_input_without_mutation(self):
        inputs = ["{bad json", "{}", "[]", "[{}]", "[null]",
                  json.dumps([_record(), {}]), json.dumps([_record(), _record()])]
        original = dict(self.collection.rows)
        for entrypoint in ("sync", "cli"):
            with self.subTest(entrypoint=entrypoint, invalid="missing"):
                with self.assertRaises(FileNotFoundError):
                    self.invoke(entrypoint, self.directory / "missing.json")
            for content in inputs:
                with self.subTest(entrypoint=entrypoint, invalid=content):
                    self.path.write_text(content, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        self.invoke(entrypoint)
                    self.assertEqual(self.collection.rows, original)
        self.assertEqual(self.collection.events, [])
        self.encode.assert_not_called()
        self.clear.assert_not_called()

    def test_no_chunks_and_cleaning_failure_preserve_existing_index(self):
        self.path.write_text(json.dumps([_record()]), encoding="utf-8")
        for entrypoint in ("sync", "cli"):
            with patch.object(ingestion, "chunk_email", return_value=[]):
                with self.assertRaisesRegex(ValueError, "no indexable chunks"):
                    self.invoke(entrypoint)
            with patch.object(ingestion, "clean_email", side_effect=ValueError("invalid cleaner input")):
                with self.assertRaises(ValueError):
                    self.invoke(entrypoint)
        self.assertEqual(self.collection.events, [])
        self.encode.assert_not_called()

    def test_embedding_failure_through_both_entrypoints_preserves_old_index(self):
        self.path.write_text(json.dumps([_record()]), encoding="utf-8")
        original = dict(self.collection.rows)
        self.encode.side_effect = RuntimeError("injected embedding failure")
        for entrypoint in ("sync", "cli"):
            with self.subTest(entrypoint=entrypoint):
                with self.assertRaises(RuntimeError):
                    self.invoke(entrypoint)
                self.assertEqual(self.collection.rows, original)
                self.assertEqual(self.collection.events, [])
        self.clear.assert_not_called()

    def test_later_embedding_batch_failure_still_precedes_every_write(self):
        self.encode.side_effect = [[[1., 0.]], RuntimeError("second batch failure")]
        with self.assertRaises(RuntimeError):
            embedder.index_chunks([_chunk("new_0", "new"), _chunk("second_0", "second")], batch_size=1, replace=True)
        self.assertEqual(self.collection.events, [])
        self.assertEqual(len(self.collection.rows), 3)

    def test_failed_upsert_does_not_delete_old_tail_or_omitted_emails(self):
        self.client.fail_upsert = self.client.upserts + 2
        with self.assertRaises(OSError):
            embedder.index_chunks([_chunk("old_0", "old"), _chunk("new_0", "new")], replace=True, batch_size=1)
        self.assertEqual(set(self.collection.rows), {"old_0", "old_1", "omitted_0"})
        self.assertEqual(self.collection.rows["old_0"]["content"], "new content")
        self.assertEqual(embedder._get_collection(), self.collection)
        self.assertFalse(any(event[0] == "delete" for event in self.collection.events))

    def test_replacement_deletes_stale_and_omitted_chunks_only_after_all_upserts(self):
        count = embedder.index_chunks([_chunk("old_0", "old"), _chunk("new_0", "new")], replace=True)
        self.assertEqual(count, 2)
        self.assertEqual(set(embedder._get_collection().rows), {"old_0", "new_0"})
        self.assertEqual(set(self.collection.rows), {"old_0", "old_1", "omitted_0"})
        self.assertEqual(self.collection.events, [])

    def test_default_upsert_retains_omitted_emails(self):
        embedder.index_chunks([_chunk("old_0", "old")])
        self.assertEqual(set(embedder._get_collection().rows), {"old_0", "omitted_0"})

    def test_empty_replacement_is_rejected_without_write(self):
        with self.assertRaisesRegex(ValueError, "must contain chunks"):
            embedder.index_chunks([], replace=True)
        self.assertEqual(embedder.index_chunks([]), 0)
        self.assertEqual(self.collection.events, [])
        self.encode.assert_not_called()

    def test_both_entrypoints_replace_successfully_without_a_separate_clear(self):
        self.path.write_text(json.dumps([_record()]), encoding="utf-8")
        for entrypoint in ("sync", "cli"):
            with self.subTest(entrypoint=entrypoint):
                self.invoke(entrypoint)
                self.assertTrue(self.collection.rows)
                self.assertTrue(all(row["email_id"] == "new" for row in embedder._get_collection().rows.values()))
        self.clear.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
