from __future__ import annotations

from models.schemas import Email
from scripts.build_real_gmail_gold_template import (
    build_real_mail_gold_template,
    merge_existing_gold_labels,
)


def test_build_real_mail_gold_template_uses_real_email_chunk_ids():
    emails = [
        Email(
            id="gmail_m1",
            subject="Budget update",
            sender="alice@example.com",
            recipients=["bob@example.com"],
            date="2026-06-09T00:00:00+00:00",
            body="The renewal budget is approved for the Q3 vendor plan.",
            labels=["INBOX"],
            thread_id="thread-1",
        )
    ]

    template = build_real_mail_gold_template(emails, limit=5, chunks_per_email=2)

    assert template == [
        {
            "id": "real_gold_001",
            "question": "",
            "ground_truth": "",
            "source_email_ids": ["gmail_m1"],
            "gold_chunk_ids": ["gmail_m1_chunk_0"],
            "source_subject": "Budget update",
            "source_sender": "alice@example.com",
            "chunk_preview": (
                "Subject: Budget update\n"
                "From: alice@example.com\n"
                "To: bob@example.com\n"
                "Date: 2026-06-09T00:00:00+00:00\n"
                "Labels: INBOX\n"
                "Thread-ID: thread-1\n\n"
                "Body:\n"
                "The renewal budget is approved for the Q3 vendor plan."
            ),
            "notes": "Fill question and ground_truth for this real-mail chunk before running context recall.",
        }
    ]


def test_build_real_mail_gold_template_honors_limit_across_chunks():
    emails = [
        Email(
            id="gmail_m1",
            subject="One",
            sender="alice@example.com",
            recipients=["bob@example.com"],
            date="2026-06-09",
            body="first real message",
            labels=[],
        ),
        Email(
            id="gmail_m2",
            subject="Two",
            sender="carol@example.com",
            recipients=["bob@example.com"],
            date="2026-06-10",
            body="second real message",
            labels=[],
        ),
    ]

    template = build_real_mail_gold_template(emails, limit=1, chunks_per_email=1)

    assert [item["id"] for item in template] == ["real_gold_001"]
    assert template[0]["source_email_ids"] == ["gmail_m1"]


def test_build_real_mail_gold_template_skips_subject_only_chunks():
    emails = [
        Email(
            id="gmail_m1",
            subject="Long announcement",
            sender="alice@example.com",
            recipients=["bob@example.com"],
            date="2026-06-10",
            body="A" * 700,
            labels=[],
        )
    ]

    template = build_real_mail_gold_template(
        emails,
        limit=1,
        chunks_per_email=3,
        min_content_chars=120,
    )

    assert template[0]["gold_chunk_ids"] == ["gmail_m1_chunk_0"]
    assert template[0]["chunk_preview"].startswith("Subject: Long announcement")
    assert "\n\nBody:\nA" in template[0]["chunk_preview"]


def test_merge_existing_labels_preserves_question_and_ground_truth_on_same_chunk():
    generated = [
        {
            "id": "real_gold_001",
            "question": "",
            "ground_truth": "",
            "source_email_ids": ["gmail_m1"],
            "gold_chunk_ids": ["gmail_m1_chunk_0"],
            "chunk_preview": "fresh evidence preview",
            "notes": "Fill question and ground_truth for this real-mail chunk before running context recall.",
        }
    ]
    existing = [
        {
            "id": "real_gold_009",
            "question": "What changed in the account?",
            "ground_truth": "The recovery phone number changed.",
            "source_email_ids": ["gmail_m1"],
            "gold_chunk_ids": ["gmail_m1_chunk_0"],
            "chunk_preview": "old evidence preview",
            "notes": "old label",
        }
    ]

    merged, stats = merge_existing_gold_labels(generated, existing)

    assert merged[0]["id"] == "real_gold_001"
    assert merged[0]["question"] == "What changed in the account?"
    assert merged[0]["ground_truth"] == "The recovery phone number changed."
    assert merged[0]["gold_chunk_ids"] == ["gmail_m1_chunk_0"]
    assert merged[0]["chunk_preview"] == "fresh evidence preview"
    assert merged[0]["notes"] == "Labeled from a preserved real-mail annotation."
    assert stats == {"generated": 1, "preserved_labels": 1, "new_items": 0}


def test_merge_existing_labels_does_not_reuse_labels_for_other_chunks_in_same_email():
    generated = [
        {
            "id": "real_gold_002",
            "question": "",
            "ground_truth": "",
            "source_email_ids": ["gmail_m1"],
            "gold_chunk_ids": ["gmail_m1_chunk_1"],
            "chunk_preview": "fresh chunk 1 preview",
            "notes": "Fill question and ground_truth for this real-mail chunk before running context recall.",
        }
    ]
    existing = [
        {
            "id": "real_gold_001",
            "question": "What changed in the account?",
            "ground_truth": "The recovery phone number changed.",
            "source_email_ids": ["gmail_m1"],
            "gold_chunk_ids": ["gmail_m1_chunk_0"],
            "chunk_preview": "old chunk 0 preview",
            "notes": "old label",
        }
    ]

    merged, stats = merge_existing_gold_labels(generated, existing)

    assert merged[0]["question"] == ""
    assert merged[0]["ground_truth"] == ""
    assert merged[0]["gold_chunk_ids"] == ["gmail_m1_chunk_1"]
    assert merged[0]["chunk_preview"] == "fresh chunk 1 preview"
    assert stats == {"generated": 1, "preserved_labels": 0, "new_items": 1}
