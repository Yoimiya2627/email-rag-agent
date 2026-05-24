from __future__ import annotations

from models.schemas import SearchResult


def _result(chunk_id: str) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        email_id=chunk_id.split("_")[0],
        content=f"content {chunk_id}",
        score=1.0,
        metadata={},
    )


def test_build_gold_template_keeps_question_and_leaves_chunk_ids_blank():
    from scripts import evaluate_context_recall as recall

    template = recall.build_gold_template(
        [
            {
                "question": "When is the budget meeting?",
                "ground_truth": "Monday",
                "email_ids": ["email_1"],
            }
        ],
        limit=1,
    )

    assert template == [
        {
            "id": "gold_001",
            "question": "When is the budget meeting?",
            "ground_truth": "Monday",
            "source_email_ids": ["email_1"],
            "gold_chunk_ids": [],
            "notes": "Fill gold_chunk_ids after inspecting retrieved/source chunks.",
        }
    ]


def test_score_context_recall_case_counts_hits_against_gold_chunks():
    from scripts import evaluate_context_recall as recall

    scored = recall.score_context_recall_case(
        {
            "id": "gold_001",
            "question": "q",
            "gold_chunk_ids": ["c1", "c3"],
        },
        [_result("c1"), _result("c2")],
    )

    assert scored["context_recall"] == 0.5
    assert scored["hit_count"] == 1
    assert scored["gold_count"] == 2
    assert scored["retrieved_chunk_ids"] == ["c1", "c2"]
    assert scored["hit_chunk_ids"] == ["c1"]
    assert scored["has_any_hit"] is True
    assert scored["has_perfect_recall"] is False


def test_evaluate_context_recall_cases_aggregates_recall_rates():
    from scripts import evaluate_context_recall as recall

    cases = [
        {"id": "a", "question": "qa", "gold_chunk_ids": ["a1", "a2"]},
        {"id": "b", "question": "qb", "gold_chunk_ids": ["b1"]},
    ]

    def fake_retrieve(question: str):
        if question == "qa":
            return [_result("a1"), _result("x")]
        return [_result("b1")]

    report = recall.evaluate_context_recall_cases(cases, retrieve_fn=fake_retrieve)

    assert report["summary"] == {
        "n": 2,
        "mean_context_recall": 0.75,
        "chunk_hit_rate": 1.0,
        "perfect_recall_rate": 0.5,
    }
    assert [record["context_recall"] for record in report["records"]] == [0.5, 1.0]
