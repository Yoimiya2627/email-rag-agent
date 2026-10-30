"""Strict reranker provider contracts, with no real model or network calls."""
import json
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi.encoders import jsonable_encoder
from starlette.responses import JSONResponse

import config.settings as cfg
import core.reranker as reranker
from models.schemas import AgentResponse, SearchResult


class RerankerScoreContracts(unittest.TestCase):
    def setUp(self):
        self.rows = [SearchResult(email_id=f"e{i}", chunk_id=f"c{i}", content=f"Evidence {i}",
                                  score=0.1 * i, metadata={}) for i in (1, 2)]
        settings = patch.multiple(cfg, ENABLE_RERANKER=True, RERANKER_BACKEND="llm")
        settings.start()
        self.addCleanup(settings.stop)
        failures = patch.object(reranker, "_consecutive_failures", 0)
        failures.start()
        self.addCleanup(failures.stop)

    def llm_client(self, payload):
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs:
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)),
                                                     finish_reason="stop")]))))

    def assert_serializable(self, rows):
        response = JSONResponse(jsonable_encoder(AgentResponse(answer="Grounded answer", sources=rows)))
        self.assertEqual(response.status_code, 200)
        self.assertTrue(all(math.isfinite(row.score) for row in rows))

    def test_llm_valid_scores_sort_numerically_without_mutating_retrieval_scores(self):
        with patch.object(reranker, "_get_client", return_value=self.llm_client({"scores": [9, 10]})):
            result = reranker.rerank("budget", self.rows, top_n=2)
        self.assertEqual([row.email_id for row in result], ["e2", "e1"])
        self.assertEqual([row.score for row in result], [10, 9])
        self.assertEqual([row.score for row in self.rows], [0.1, 0.2])
        self.assertEqual(reranker._consecutive_failures, 0)
        self.assert_serializable(result)

    def test_llm_integer_valued_json_numbers_and_endpoints_are_accepted(self):
        with patch.object(reranker, "_get_client", return_value=self.llm_client({"scores": [0.0, 10.0]})):
            result = reranker.rerank("budget", self.rows, top_n=2)
        self.assertEqual([row.score for row in result], [10, 0])
        self.assertEqual(reranker._consecutive_failures, 0)

    def test_llm_invalid_scores_degrade_before_sort_and_remain_json_safe(self):
        cases = (["9", "10"], ["NaN", "1"], [float("nan"), 1], [float("inf"), 1],
                 [-float("inf"), 1], [True, 1], [False, 1], [-1, 1], [11, 1],
                 [9.5, 1], [None, 1], [[1], 2], [1], [1, 2, 3])
        for scores in cases:
            with self.subTest(scores=repr(scores)), patch.object(reranker, "_consecutive_failures", 0), \
                    patch.object(reranker, "_get_client", return_value=self.llm_client({"scores": scores})):
                result = reranker.rerank("budget", self.rows, top_n=2)
                self.assertEqual(result, self.rows)
                self.assertEqual(reranker._consecutive_failures, 1)
                self.assert_serializable(result)

    def test_llm_malformed_score_container_degrades(self):
        for payload in ([], {}, {"scores": None}, {"scores": "91"}, {"scores": {"0": 9, "1": 10}}):
            with self.subTest(payload=payload), patch.object(reranker, "_consecutive_failures", 0), \
                    patch.object(reranker, "_get_client", return_value=self.llm_client(payload)):
                result = reranker.rerank("budget", self.rows, top_n=2)
                self.assertEqual(result, self.rows)
                self.assertEqual(reranker._consecutive_failures, 1)

    def test_invalid_llm_scores_participate_in_existing_circuit_breaker(self):
        client = self.llm_client({"scores": ["9", "10"]})
        with patch.object(reranker, "_get_client", return_value=client) as get_client:
            for _ in range(reranker._FAILURE_THRESHOLD + 1):
                self.assertEqual(reranker.rerank("budget", self.rows, top_n=2), self.rows)
        self.assertEqual(get_client.call_count, reranker._FAILURE_THRESHOLD)
        self.assertEqual(reranker._consecutive_failures, reranker._FAILURE_THRESHOLD)

    def test_valid_llm_scores_reset_failure_counter(self):
        with patch.object(reranker, "_consecutive_failures", 2), patch.object(reranker, "_get_client",
                return_value=self.llm_client({"scores": [1, 2]})):
            result = reranker.rerank("budget", self.rows, top_n=2)
            self.assertEqual(reranker._consecutive_failures, 0)
        self.assertEqual(result[0].email_id, "e2")

    def test_cross_encoder_accepts_negative_and_unbounded_finite_logits(self):
        model = SimpleNamespace(predict=lambda pairs: [-4.25, 27.5])
        with patch.object(cfg, "RERANKER_BACKEND", "cross_encoder"), \
                patch.object(reranker, "_get_cross_encoder", return_value=model):
            result = reranker.rerank("budget", self.rows, top_n=2)
        self.assertEqual([row.score for row in result], [27.5, -4.25])
        self.assertEqual(reranker._consecutive_failures, 0)
        self.assert_serializable(result)

    def test_cross_encoder_array_conversion_preserves_numeric_output(self):
        array = SimpleNamespace(tolist=lambda: [-1.5, 2.25])
        with patch.object(cfg, "RERANKER_BACKEND", "cross_encoder"), patch.object(reranker,
                "_get_cross_encoder", return_value=SimpleNamespace(predict=lambda pairs: array)):
            result = reranker.rerank("budget", self.rows, top_n=2)
        self.assertEqual([row.score for row in result], [2.25, -1.5])
        self.assertEqual(reranker._consecutive_failures, 0)

    def test_cross_encoder_invalid_numbers_and_length_degrade(self):
        cases = ([float("nan"), 1], [float("inf"), 1], [-float("inf"), 1],
                 [True, 1], ["9", "10"], [None, 1], [[1], [2]], [1], [1, 2, 3], 1)
        for scores in cases:
            with self.subTest(scores=repr(scores)), patch.object(reranker, "_consecutive_failures", 0), \
                    patch.object(cfg, "RERANKER_BACKEND", "cross_encoder"), patch.object(reranker,
                    "_get_cross_encoder", return_value=SimpleNamespace(predict=lambda pairs: scores)):
                result = reranker.rerank("budget", self.rows, top_n=2)
                self.assertEqual(result, self.rows)
                self.assertEqual(reranker._consecutive_failures, 1)
                self.assert_serializable(result)


if __name__ == "__main__":
    unittest.main()
