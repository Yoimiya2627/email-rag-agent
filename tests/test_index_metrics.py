from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor
import pytest
from core.index_metrics import collect_index_metrics, measure_stage, add_count, current_metrics


def test_nested_stage_times_are_exclusive_and_sum_to_wall_time():
    with patch("core.index_metrics.time.perf_counter", side_effect=[0., 1., 2., 4., 6., 8.]):
        with collect_index_metrics() as report:
            with measure_stage("embedding"):
                with measure_stage("model_load"):
                    add_count("embedded_chunks", 3)
        data = report.to_dict()
    assert data["total_seconds"] == 8.
    assert data["stages_seconds"] == {"model_load": 2., "embedding": 3.}
    assert data["unaccounted_seconds"] == 3.


def test_error_report_never_contains_exception_text_and_releases_context():
    with pytest.raises(ValueError):
        with collect_index_metrics() as report:
            with measure_stage("write"):
                raise ValueError("private message body")
    assert report.outcome == "failed"
    assert "private" not in str(report.to_dict())
    assert current_metrics() is None


def test_independent_requests_do_not_share_metrics():
    def run(n):
        with collect_index_metrics() as report:
            with collect_index_metrics() as nested:
                assert report is nested
                add_count("input_chunks", n)
        return report.to_dict()["counts"]["input_chunks"]
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(run, [2, 7])) == [2, 7]
