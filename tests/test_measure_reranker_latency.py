from __future__ import annotations

import pytest

from scripts import measure_reranker_latency as bench


def test_parse_versions_accepts_supported_reranker_targets():
    targets = bench.parse_versions("V2,V7")

    assert [target.version for target in targets] == ["V2", "V7"]
    assert targets[0].enable_reranker is False
    assert targets[1].backend == "cross_encoder"


def test_parse_versions_rejects_unknown_target():
    with pytest.raises(ValueError, match="Unknown reranker benchmark version"):
        bench.parse_versions("V2,V8")


def test_build_cases_uses_ground_truth_plus_distractors():
    testset = [
        {"question": "q1", "ground_truth": "answer one", "email_ids": ["email_1"]},
        {"question": "q2", "ground_truth": "answer two", "email_ids": ["email_2"]},
        {"question": "q3", "ground_truth": "answer three", "email_ids": ["email_3"]},
    ]

    cases = bench.build_cases(testset, limit=1, candidate_count=3)

    assert len(cases) == 1
    assert cases[0].query == "q1"
    assert [candidate.content for candidate in cases[0].candidates] == [
        "answer one",
        "answer two",
        "answer three",
    ]
    assert cases[0].candidates[0].metadata["source"] == "ground_truth"
    assert cases[0].candidates[1].metadata["source"] == "distractor"


def test_summarize_timings_reports_trimmed_mean_and_p95_ms():
    summary = bench.summarize_timings([0.01, 0.02, 0.03, 0.2])

    assert summary == {
        "n": 4,
        "raw_ms": [10.0, 20.0, 30.0, 200.0],
        "mean_trimmed_ms": 25.0,
        "median_ms": 25.0,
        "p95_ms": 200.0,
    }
