"""All evaluation CLIs accept the same aware reference times on Python 3.10+."""
import argparse
from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture(params=["recall", "paired", "agent"])
def parse_reference(request):
    from scripts import evaluate_context_recall, paired_acceptance, run_agent_eval

    return {
        "recall": evaluate_context_recall._parse_reference_time,
        "paired": paired_acceptance._parse_reference,
        "agent": run_agent_eval._reference_time_arg,
    }[request.param]


@pytest.mark.parametrize("raw, expected", [
    ("2026-11-02T12:00:00Z", datetime(2026, 11, 2, 12, tzinfo=timezone.utc)),
    ("2026-11-02T12:00:00.123456Z", datetime(2026, 11, 2, 12, microsecond=123456, tzinfo=timezone.utc)),
    ("2026-11-02T12:00:00+08:00", datetime(2026, 11, 2, 12, tzinfo=timezone(timedelta(hours=8)))),
])
def test_reference_cli_preserves_instant_precision_and_offset(parse_reference, raw, expected):
    actual = parse_reference(raw)
    assert actual == expected
    assert actual.isoformat() == expected.isoformat()


@pytest.mark.parametrize("raw", [
    "2026-11-02",
    "2026-11-02T12:00:00",
    "2026-11-02T12:00:00+08:00Z",
    None,
])
def test_reference_cli_still_rejects_naive_or_malformed_values(parse_reference, raw):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_reference(raw)
