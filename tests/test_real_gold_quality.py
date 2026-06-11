from __future__ import annotations


def _gold_case(
    case_id: str,
    chunk_id: str,
    question: str = "What does the email say about budget approval?",
    ground_truth: str = "The budget was approved.",
    notes: str = "Human reviewed annotation.",
) -> dict:
    return {
        "id": case_id,
        "question": question,
        "ground_truth": ground_truth,
        "source_email_ids": [chunk_id.rsplit("_chunk_", 1)[0]],
        "gold_chunk_ids": [chunk_id],
        "notes": notes,
    }


def test_evaluate_gold_quality_accepts_clean_gold():
    from scripts.check_real_gold_quality import evaluate_gold_quality

    report = evaluate_gold_quality(
        [
            _gold_case("real_gold_001", "gmail_m1_chunk_0"),
            _gold_case("real_gold_002", "gmail_m2_chunk_0"),
        ],
        min_cases=2,
        max_local_annotation_ratio=1.0,
    )

    assert report["ok"] is True
    assert report["blockers"] == []
    assert report["summary"]["total_cases"] == 2
    assert report["summary"]["unique_gold_chunks"] == 2


def test_evaluate_gold_quality_blocks_structural_and_text_quality_issues():
    from scripts.check_real_gold_quality import evaluate_gold_quality

    report = evaluate_gold_quality(
        [
            _gold_case(
                "real_gold_001",
                "gmail_m1_chunk_0",
                question="What does chunk 0 say?",
            ),
            _gold_case(
                "real_gold_002",
                "gmail_m1_chunk_0",
                question="What does the email say?",
                ground_truth="This answer contains ???? mojibake.",
            ),
            _gold_case(
                "real_gold_003",
                "gmail_m3_chunk_0",
                question="",
                ground_truth="Missing question.",
            ),
        ],
        min_cases=4,
        max_local_annotation_ratio=1.0,
    )

    codes = {item["code"] for item in report["blockers"]}
    assert report["ok"] is False
    assert {
        "too_few_cases",
        "duplicate_gold_chunk_id",
        "unlabeled_case",
        "suspicious_text_artifact",
        "question_leaks_chunk_position",
    }.issubset(codes)
