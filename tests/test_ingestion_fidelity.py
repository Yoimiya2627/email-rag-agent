"""Offline regression tests using the actual ingestion and reconstruction modules."""
import base64
import hashlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agents.gmail_readonly import gmail_message_to_email, GmailReadOnlyProvider
from agents.mail_providers import MailProviderError
from core.cleaner import clean_body, clean_email
from core.chunker import chunk_email, chunk_text, _force_split
from core import embedder
import config.settings as cfg
from models.schemas import Email


def part(mime, text="", **extra):
    return {"mimeType": mime, "body": {"data": base64.urlsafe_b64encode(text.encode()).decode()}, **extra}


def message(payload):
    return {"id": "m", "internalDate": "1710000000000", "payload": payload}


def email(body="body", **extra):
    return Email(id="e", subject="subject", sender="a@example.com", recipients=[], date="2026-09-09", body=body, **extra)


class FidelityTests(unittest.TestCase):
    def test_gmail_decodes_utf8_encoded_subject_and_indexes_readable_text(self):
        payload = part("text/plain", "Approved budget 500.", headers=[
            {"name": "Subject", "value": "=?UTF-8?B?6aKE566X5om55YeG?="}])
        converted = gmail_message_to_email(message(payload))
        self.assertEqual(converted.subject, "预算批准")
        self.assert_detail_roundtrip(converted, "Approved budget 500.")

    def test_gmail_decodes_mixed_plain_and_quoted_printable_subject(self):
        payload = part("text/plain", "Confirm Friday.", headers=[
            {"name": "Subject", "value": "Re: =?iso-8859-1?Q?caf=E9?= / delivery"}])
        converted = gmail_message_to_email(message(payload))
        self.assertEqual(converted.subject, "Re: café / delivery")
        self.assert_detail_roundtrip(converted, "Confirm Friday.")

    def test_gmail_preserves_plain_and_undecodable_subjects(self):
        subjects = ["Budget  update <USD 500>", "预算批准", "Re: =?x-unknown-charset?Q?caf=E9?=",
                    "=?UTF-8?B?/w==?=", "=?UTF-8?B?a?=", "=?UTF-8?B?%%%?="]
        for subject in subjects:
            with self.subTest(subject=subject):
                payload = part("text/plain", "body", headers=[{"name": "Subject", "value": subject}])
                self.assertEqual(gmail_message_to_email(message(payload)).subject, subject)

    def assert_detail_roundtrip(self, source, expected_body):
        normalized = clean_email(source)
        self.assertEqual(normalized.body, expected_body)
        with patch.multiple(cfg, CHUNK_SIZE=25, CHUNK_OVERLAP=4, MIN_CHUNK_SIZE=8):
            chunks = chunk_email(normalized)
        records = list(reversed(chunks))
        calls = []
        def lookup(**kwargs):
            calls.append(kwargs)
            return {"ids": [c.chunk_id for c in records],
                    "documents": [c.content for c in records],
                    "metadatas": [{**c.metadata, "chunk_index": c.chunk_index} for c in records]}
        with patch.object(embedder, "_get_collection", return_value=SimpleNamespace(get=lookup)):
            detail = embedder.get_indexed_email(normalized.id)
        self.assertEqual(calls[0]["where"], {"email_id": normalized.id})
        self.assertEqual(detail["body"], f"Subject: {normalized.subject}\n\n{expected_body}")
        self.assertTrue(detail["reconstruction_exact"])
        self.assertEqual(detail["body_format"], "plain")

    def test_implicit_head_close_survives_clean_chunk_and_details(self):
        expected = "Approved budget is 500.\n\nConfirm delivery Friday."
        for opening in ["<body>", ""]:
            html = ('<html><head><meta charset="utf-8"><title>Private metadata</title>'
                    '<style>p{color:red}</style>' + opening
                    + '<p>Approved budget is 500.</p><p>Confirm delivery Friday.</p></body></html>')
            with self.subTest(opening=opening):
                self.assert_detail_roundtrip(email(html, body_format="html"), expected)

    def test_gmail_implicit_head_close_preserves_normalized_body(self):
        html = '<html><head><meta charset="utf-8"><body><p>Thanks</p><p>Approve USD 500 by Friday.</p></body></html>'
        converted = gmail_message_to_email(message(part("text/html", html)))
        self.assertEqual(converted.body_format, "plain")
        self.assert_detail_roundtrip(converted, "Thanks\n\nApprove USD 500 by Friday.")

    def test_implicit_body_text_after_metadata_is_not_hidden(self):
        self.assertEqual(clean_body('<head><title>Hidden</title><meta charset="utf-8">Actual body', "html"), "Actual body")

    def test_thanks_and_signoffs_never_truncate_business_body(self):
        for greeting in ["Thanks", "谢谢！", "Best regards", "祝好", "--"]:
            body = (greeting + "\n\nThe revised quote looks good. Please confirm delivery by Friday "
                    "and keep the total below USD 500.\n\nAlice")
            with self.subTest(greeting=greeting):
                self.assert_detail_roundtrip(email(body), body)
                converted = gmail_message_to_email(message(part("text/plain", body)))
                self.assert_detail_roundtrip(converted, body)

    def test_only_explicit_signature_delimiter_after_body_is_removed(self):
        self.assertEqual(clean_body("Please confirm Friday.\n\n-- \nAlice"), "Please confirm Friday.")
        self.assertEqual(clean_body("-- \nPlease confirm Friday."), "-- \nPlease confirm Friday.")
        self.assertEqual(clean_body("Thanks\n\nPlease confirm Friday.\n\nRegards\nAlice"),
                         "Thanks\n\nPlease confirm Friday.\n\nRegards\nAlice")

    def test_html_horizontal_rules_separate_words_through_gmail_and_details(self):
        html = "<p>approved<hr>rejected</p>"
        self.assert_detail_roundtrip(email(html, body_format="html"), "approved\n\nrejected")
        self.assert_detail_roundtrip(gmail_message_to_email(message(part("text/html", html))), "approved\n\nrejected")

    def test_plain_brackets_addresses_and_entities_remain_literal(self):
        text = "Contact <alice@example.com>; 2 < 3 and 5 > 4; literal <b>bold</b> &amp; &#x41;."
        self.assertEqual(clean_body(text), text)
        self.assertEqual(clean_email(email(text)).body, text)

    def test_explicit_html_decodes_entities_preserves_paragraphs(self):
        value = clean_email(email("<p>A &amp; B &#x41;</p><p>next<br>line &lt;b&gt;</p><script>bad()</script>", body_format="html"))
        self.assertEqual(value.body, "A & B A\n\nnext\nline <b>")
        self.assertEqual(value.body_format, "plain")
        self.assertEqual(clean_email(value).body, value.body)

    def test_gmail_plain_html_looking_text_survives_cleanup_and_serialization(self):
        text = "Literal <p>example</p> &amp; <person@example.com>"
        converted = gmail_message_to_email(message(part("text/plain", text)))
        restored = Email.model_validate(converted.model_dump())
        self.assertEqual(clean_email(restored).body, text)
        self.assertEqual(restored.body_format, "plain")

    def test_gmail_html_normalized_once(self):
        converted = gmail_message_to_email(message(part("text/html", "<p>&lt;b&gt; &amp;amp;</p><p>Next</p>")))
        self.assertEqual(converted.body, "<b> &amp;\n\nNext")
        self.assertEqual(clean_email(converted).body, converted.body)

    def test_attachment_cannot_replace_html_body(self):
        payload = part("multipart/mixed", parts=[part("text/plain", "ATTACHMENT", filename="note.txt"), part("text/html", "<p>Actual</p>")])
        self.assertEqual(gmail_message_to_email(message(payload)).body, "Actual")

    def test_attachment_disposition_excludes_whole_subtree(self):
        attached = part("multipart/mixed", headers=[{"name": "Content-Disposition", "value": "attachment"}], parts=[part("text/plain", "secret")])
        payload = part("multipart/mixed", parts=[attached, part("text/plain", "Actual")])
        self.assertEqual(gmail_message_to_email(message(payload)).body, "Actual")

    def test_content_type_filename_excluded(self):
        attached = part("text/plain", "secret", headers=[{"name": "Content-Type", "value": 'text/plain; name="note.txt"'}])
        payload = part("multipart/mixed", parts=[attached, part("text/html", "<p>Actual</p>")])
        self.assertEqual(gmail_message_to_email(message(payload)).body, "Actual")

    def test_nested_alternative_and_mixed_keep_all_body_sections(self):
        alternative = part("multipart/alternative", parts=[part("text/html", "<b>duplicate</b>"), part("text/plain", "First")])
        nested = part("multipart/mixed", parts=[alternative, part("text/html", "<p>Second</p>"), part("text/plain", "Third")])
        self.assertEqual(gmail_message_to_email(message(nested)).body, "First\n\nSecond\n\nThird")

    def test_related_root_selected_by_start_and_resources_excluded(self):
        payload = part("multipart/related", headers=[{"name": "Content-Type", "value": 'multipart/related; start="<root>"'}], parts=[part("text/plain", "resource"), part("text/html", "<p>Actual</p>", headers=[{"name": "Content-ID", "value": "<root>"}])])
        self.assertEqual(gmail_message_to_email(message(payload)).body, "Actual")

    def test_forwarded_message_does_not_replace_parent(self):
        payload = part("multipart/mixed", parts=[part("message/rfc822", parts=[part("text/plain", "forwarded")]), part("text/plain", "parent")])
        self.assertEqual(gmail_message_to_email(message(payload)).body, "parent")

    def test_charset_decoding(self):
        payload = part("text/plain", headers=[{"name": "Content-Type", "value": "text/plain; charset=iso-8859-1"}])
        payload["body"]["data"] = base64.urlsafe_b64encode("caf\xe9".encode("latin1")).decode()
        self.assertEqual(gmail_message_to_email(message(payload)).body, "caf\xe9")

    def test_invalid_encoding_fails_instead_of_indexing_replacement_characters(self):
        payload = part("text/plain")
        payload["body"]["data"] = "/w=="
        with self.assertRaises(MailProviderError):
            gmail_message_to_email(message(payload))
        payload["body"]["data"] = "%%%"
        with self.assertRaises(MailProviderError):
            gmail_message_to_email(message(payload))

    def test_separated_body_fetches_only_non_attachment(self):
        payload = part("multipart/mixed", parts=[part("text/plain", filename="secret.txt", body={"attachmentId": "secret"}), part("text/plain", body={"attachmentId": "body"})])
        calls = []
        def loader(identifier):
            calls.append(identifier)
            return part("text/plain", "Actual")["body"]["data"]
        self.assertEqual(gmail_message_to_email(message(payload), loader).body, "Actual")
        self.assertEqual(calls, ["body"])
        with self.assertRaises(MailProviderError):
            gmail_message_to_email(message(payload))

    def test_force_split_stops_when_source_is_fully_covered(self):
        self.assertEqual(_force_split("0123456789", 10, 3), ["0123456789"])
        self.assertEqual(_force_split("0123456789abcdefg", 10, 3), ["0123456789", "789abcdefg"])

    def test_short_tail_merge_does_not_duplicate_overlap(self):
        text = "A" * 15 + "\n\nBB"
        self.assertEqual(chunk_text(text, 15, 2, 8), [text])
        self.assertEqual(chunk_text("0123456789ABC", 10, 3, 7), ["0123456789ABC"])

    def test_invalid_chunk_options_fail_fast(self):
        for values in [(0, 0, 1), (5, 5, 1), (5, -1, 1), (5, 2, -1), (True, 0, 0), (3, 0.5, 0)]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                chunk_text("data", *values)

    def test_round_trip_offsets_preserve_paragraphs_whitespace_and_true_repetitions(self):
        samples = ["abcdefgh" * 20, "repeat\n\nrepeat\n\nrepeat", " first \n\n\n second  \nlast ", "A" * 51, "汉字" * 35, ""]
        for body in samples:
            for size, overlap, minimum in [(10, 3, 5), (20, 0, 8), (10, 9, 0), (50, 5, 40)]:
                with self.subTest(body=body, size=size, overlap=overlap):
                    value = email(body)
                    text = f"Subject: {value.subject}\n\n{body}"
                    with patch.multiple(cfg, CHUNK_SIZE=size, CHUNK_OVERLAP=overlap, MIN_CHUNK_SIZE=minimum):
                        chunks = chunk_email(value)
                    rows = [{"chunk_id": c.chunk_id, "content": c.content, "metadata": c.metadata} for c in chunks]
                    for c in chunks:
                        self.assertEqual(c.content, text[c.metadata["source_start"]:c.metadata["source_end"]])
                    # A settings change after indexing must not change reconstruction.
                    with patch.object(embedder, "get_email_chunks", return_value=rows), patch.object(cfg, "CHUNK_OVERLAP", 999):
                        result = embedder.get_indexed_email("e")
                    self.assertEqual(result["body"], text)
                    self.assertTrue(result["reconstruction_exact"])

    def test_legacy_identical_boundary_text_is_never_guessed_away(self):
        rows = [{"content": "repeat", "metadata": {}}, {"content": "repeat", "metadata": {}}]
        with patch.object(embedder, "get_email_chunks", return_value=rows), patch.object(cfg, "CHUNK_OVERLAP", 6):
            result = embedder.get_indexed_email("e")
        self.assertEqual(result["body"], "repeat\n\nrepeat")
        self.assertTrue(result["reindex_required"])

    def test_missing_or_modified_chunks_never_claim_exact_reconstruction(self):
        with patch.multiple(cfg, CHUNK_SIZE=10, CHUNK_OVERLAP=2, MIN_CHUNK_SIZE=1):
            chunks = chunk_email(email("abcdefgh" * 10))
        rows = [{"content": c.content, "metadata": c.metadata} for c in chunks]
        for damaged in [rows[:-1], rows[1:], [rows[0], *rows[2:]], [{**rows[0], "content": "bad"}, *rows[1:]]]:
            with patch.object(embedder, "get_email_chunks", return_value=damaged):
                self.assertFalse(embedder.get_indexed_email("e")["reconstruction_exact"])


if __name__ == "__main__":
    unittest.main()
