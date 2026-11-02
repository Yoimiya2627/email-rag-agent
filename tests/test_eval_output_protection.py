"""Evaluation evidence must not overwrite old reports or its own state."""
from unittest.mock import Mock
import sys

import pytest

from scripts import run_agent_eval as evaluation


@pytest.mark.parametrize('option', ['--output', '--report-output'])
def test_existing_report_is_rejected_before_state_or_client_creation(monkeypatch, tmp_path, option):
    existing = tmp_path / 'history.json'
    existing.write_bytes(b'historical evidence')
    run = tmp_path / 'new-run'
    client = Mock(side_effect=AssertionError('must validate before model client creation'))
    monkeypatch.setattr(evaluation, 'OpenAI', client)
    monkeypatch.setattr(sys, 'argv', ['eval', '--run-dir', str(run), option, str(existing)])
    with pytest.raises(SystemExit) as exc:
        evaluation.main()
    assert exc.value.code == 2
    assert existing.read_bytes() == b'historical evidence'
    assert not run.exists()
    client.assert_not_called()


@pytest.mark.parametrize('name', ['records.jsonl', 'sessions.sqlite3', 'approvals.sqlite3', 'traces.jsonl'])
def test_report_cannot_replace_its_own_evaluation_state(monkeypatch, tmp_path, name):
    run = tmp_path / 'new-run'
    monkeypatch.setattr(sys, 'argv', ['eval', '--run-dir', str(run), '--output', str(run / name)])
    with pytest.raises(SystemExit) as exc:
        evaluation.main()
    assert exc.value.code == 2 and not run.exists()


def test_json_and_markdown_outputs_cannot_alias(monkeypatch, tmp_path):
    run = tmp_path / 'new-run'
    report = tmp_path / 'report'
    monkeypatch.setattr(sys, 'argv', ['eval', '--run-dir', str(run), '--output', str(report),
                                    '--report-output', str(report)])
    with pytest.raises(SystemExit) as exc:
        evaluation.main()
    assert exc.value.code == 2 and not run.exists() and not report.exists()
