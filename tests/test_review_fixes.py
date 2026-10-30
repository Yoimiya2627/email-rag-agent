"""Regression coverage for incomplete filters and unread draft evidence."""
import json
from types import SimpleNamespace as NS

import pytest


def fake_client(response):
    return NS(chat=NS(completions=NS(create=lambda **kwargs: response)))


@pytest.mark.parametrize('reason,content,reasoning', [
    ('length', '{"sender":""}', ''),
    ('length', '', '{"sender":""}'),
    ('stop', '', '{"sender":""}'),
    (None, '{"sender":""}', ''),
    ('content_filter', '{"sender":""}', ''),
    ('tool_calls', '{"sender":""}', ''),
])
def test_filter_refuses_incomplete_or_reasoning_output(monkeypatch, reason, content, reasoning):
    import core.pipeline as pipeline
    response = NS(choices=[NS(finish_reason=reason, message=NS(content=content, reasoning_content=reasoning))])
    monkeypatch.setattr(pipeline, '_get_client', lambda: fake_client(response))
    with pytest.raises(ValueError, match='无法可靠解析'):
        pipeline.extract_filters('Find quotes from alice@example.test')


@pytest.mark.parametrize('fenced', [False, True])
def test_filter_uses_complete_final_json_and_ignores_reasoning(monkeypatch, fenced):
    import core.pipeline as pipeline
    payload = {'query': 'quote', 'sender': 'alice@example.test', 'date_hint': '', 'labels': []}
    raw = json.dumps(payload)
    if fenced:
        raw = '```json\n' + raw + '\n```'
    response = NS(choices=[NS(finish_reason='stop', message=NS(content=raw, reasoning_content='{"sender":"wrong"}'))])
    monkeypatch.setattr(pipeline, '_get_client', lambda: fake_client(response))
    assert pipeline.extract_filters('Find quotes from alice@example.test') == payload


def test_incomplete_query_rewrite_cannot_erase_original_constraints(monkeypatch):
    import core.pipeline as pipeline
    monkeypatch.setattr(pipeline.cfg, 'ENABLE_QUERY_REWRITE', True)
    response = NS(choices=[NS(finish_reason='length', message=NS(content='quote'))])
    monkeypatch.setattr(pipeline, '_get_client', lambda: fake_client(response))
    original = 'Find quotes from alice@example.test'
    assert pipeline.rewrite_query(original) == original


@pytest.mark.parametrize('source,expected', [
    ({'attachments': [{'filename': 'quote.pdf', 'status': 'not_read'}]}, 1),
    ({'attachments': []}, 0),
    ({}, None),
    ({'attachments': 'invalid json'}, None),
])
@pytest.mark.parametrize('indexed', [False, True])
def test_writer_keeps_source_coverage_in_actual_model_request(monkeypatch, source, expected, indexed):
    import agents.writer_agent as writer
    captured = {}
    def create(**kwargs):
        captured.update(kwargs)
        return NS(choices=[NS(finish_reason='stop', message=NS(content='Draft for review'))])
    monkeypatch.setattr(writer, '_get_client', lambda: NS(chat=NS(completions=NS(create=create))))
    email = {'body': 'Please review the attached quote.'}
    if indexed:
        email['chunks'] = [{'email_id': 'e', 'chunk_id': 'c', 'content': email['body'], 'metadata': source}]
    else:
        email.update(source)
    writer.draft_reply_for_email(email, 'Draft a reply about the terms')
    model_input = '\n'.join(row['content'] for row in captured['messages'])
    encoded = 'null' if expected is None else str(expected)
    assert '"unread_attachments": ' + encoded in model_input
    assert '不得确认未读附件中的条款' in model_input
    assert email['body'] in model_input


def test_writer_preserves_full_coverage_when_inventory_page_is_shortened(monkeypatch):
    import agents.writer_agent as writer
    captured = {}
    def create(**kwargs):
        captured.update(kwargs)
        return NS(choices=[NS(finish_reason='stop', message=NS(content='Draft'))])
    monkeypatch.setattr(writer, '_get_client', lambda: NS(chat=NS(completions=NS(create=create))))
    email = {'body': 'See attachments.', 'attachments': [{'filename': 'a.pdf', 'status': 'not_read'}],
             'coverage': {'attachment_inventory_status': 'available', 'attachment_inventory_present': True,
                          'attachment_count': 9, 'unread_attachments': 9}}
    writer.draft_reply_for_email(email)
    model_input = '\n'.join(row['content'] for row in captured['messages'])
    assert '"unread_attachments": 9' in model_input
