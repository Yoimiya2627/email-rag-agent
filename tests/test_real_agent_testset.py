from __future__ import annotations

from scripts.build_real_agent_testset import build_real_agent_testset


def _gold_case(**overrides):
    base = {
        "id": "real_gold_001",
        "question": "What changed in the account security email?",
        "ground_truth": "A recovery phone number was changed.",
        "source_email_ids": ["gmail_m1"],
        "gold_chunk_ids": ["gmail_m1_chunk_0"],
        "source_subject": "Security alert",
        "source_sender": "no-reply@accounts.google.com",
        "chunk_preview": "Subject: Security alert\nBody: recovery phone changed",
        "notes": "Labeled from local annotation.",
    }
    base.update(overrides)
    return base


def test_build_real_agent_testset_generates_retrieval_and_detail_tasks():
    tasks = build_real_agent_testset([_gold_case()], limit=10, variants=["retrieval", "detail"])

    assert [task["id"] for task in tasks] == [
        "real_agent_retrieval_001",
        "real_agent_detail_001",
    ]
    assert tasks[0]["task_type"] == "real_retrieval"
    assert tasks[0]["expected_tools"] == ["search_emails"]
    assert tasks[0]["forbidden_tools"] == ["send_email"]
    assert tasks[0]["source_gold_id"] == "real_gold_001"
    assert tasks[0]["source_email_ids"] == ["gmail_m1"]
    assert tasks[0]["gold_chunk_ids"] == ["gmail_m1_chunk_0"]
    assert "What changed in the account security email?" in tasks[0]["task"]
    assert "A recovery phone number was changed." in tasks[0]["success_criteria"]

    assert tasks[1]["task_type"] == "real_detail_lookup"
    assert tasks[1]["expected_tools"] == ["search_emails", "get_email"]
    assert "Security alert" in tasks[1]["task"]
    assert "no-reply@accounts.google.com" in tasks[1]["task"]


def test_build_real_agent_testset_skips_unlabeled_gold_cases_and_honors_limit():
    tasks = build_real_agent_testset(
        [
            _gold_case(id="real_gold_001"),
            _gold_case(id="real_gold_002", question=""),
            _gold_case(id="real_gold_003", ground_truth=""),
        ],
        limit=1,
        variants=["retrieval", "detail"],
    )

    assert len(tasks) == 1
    assert tasks[0]["id"] == "real_agent_retrieval_001"


def test_build_real_agent_testset_filters_fragmentary_and_duplicate_real_gold_cases():
    good_first = _gold_case(
        id="real_gold_good_001",
        question=(
            'What does the email from noreply@example.com about "Security update" '
            "say regarding Your recovery phone number was changed and you should "
            "review recent account activity?"
        ),
        ground_truth=(
            "Your recovery phone number was changed and you should review recent "
            "account activity."
        ),
        source_email_ids=["gmail_good_1"],
    )
    bad_fragment = _gold_case(
        id="real_gold_bad_fragment",
        question=(
            'What does the email from googleaistudio-noreply@google.com about '
            '"[Gemini API] Build production ready managed agents with a single API call" '
            "say regarding ents directly in?"
        ),
        ground_truth="ents directly in...",
        source_email_ids=["gmail_bad_1"],
    )
    bad_boilerplate = _gold_case(
        id="real_gold_bad_boilerplate",
        question=(
            'What does the email from googleaistudio-noreply@google.com about '
            '"Build Android apps in minutes with simple prompts" say regarding '
            "Email not displaying correctly? View it?"
        ),
        ground_truth="Email not displaying correctly? View it...",
        source_email_ids=["gmail_bad_2"],
    )
    duplicate_same_email = _gold_case(
        id="real_gold_duplicate",
        question=(
            'What does the email from noreply@example.com about "Security update" '
            "say regarding This second chunk is valid but comes from the same source email?"
        ),
        ground_truth="This second chunk is valid but comes from the same source email.",
        source_email_ids=["gmail_good_1"],
    )
    good_second = _gold_case(
        id="real_gold_good_002",
        question=(
            'What does the email from billing@example.com about "Receipt" say regarding '
            "Your subscription receipt says the plan renews monthly until cancellation?"
        ),
        ground_truth="Your subscription receipt says the plan renews monthly until cancellation.",
        source_email_ids=["gmail_good_2"],
    )

    tasks = build_real_agent_testset(
        [bad_fragment, good_first, bad_boilerplate, duplicate_same_email, good_second],
        limit=10,
        variants=["retrieval"],
    )

    assert [task["source_gold_id"] for task in tasks] == [
        "real_gold_good_001",
        "real_gold_good_002",
    ]
    assert [task["id"] for task in tasks] == [
        "real_agent_retrieval_001",
        "real_agent_retrieval_002",
    ]


def test_build_real_agent_testset_rejects_unknown_variants():
    try:
        build_real_agent_testset([_gold_case()], variants=["send"])
    except ValueError as exc:
        assert "Unsupported real agent task variant" in str(exc)
    else:
        raise AssertionError("expected unsupported variant to fail")
