"""Real-file capture publication/replay with synthetic Gmail content."""
import base64
import copy
import hashlib
import json
from types import SimpleNamespace as NS

import pytest

from agents.gmail_readonly import GmailReadOnlyProvider
from core.capture_selection import selection_record
from scripts.replay_gmail_captures import replay_captures
from scripts.sync_gmail_readonly import sync_gmail_to_json
from tests.test_gmail_capture import message, Service


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')
    return path


def complete(body='Price 700'):
    provider = GmailReadOnlyProvider(service=Service(bodies={'body-1': base64.urlsafe_b64encode(body.encode()).decode()}))
    return provider.capture_message(message())


def archive(tmp_path):
    root = tmp_path / 'raw'
    base = save(root / 'legacy.json', {'format': 'gmail-full-v1', 'message': message(), 'body_data': {}, 'body_errors': {}})
    enriched = save(root / 'versions' / 'complete.json', complete())
    return root, base, enriched


def manifest(root, base, enriched, directory=None):
    return save((directory or root / 'selections') / base.name, selection_record(root, base, enriched, 'm'))


def test_new_sync_records_complete_version_before_replay(tmp_path):
    reader = GmailReadOnlyProvider(service=Service(bodies={'body-1': base64.urlsafe_b64encode(b'Price 700').decode()}))
    provider = NS(list_message_ids=lambda **kwargs: ['m'], get_message=lambda identifier: message(),
                  capture_message=reader.capture_message, email_from_capture=reader.email_from_capture)
    corpus, state, raw = tmp_path / 'corpus.json', tmp_path / 'state.json', tmp_path / 'raw'
    sync_gmail_to_json(provider, corpus, state, '', 10, raw_dir=raw)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.rglob('*.json')}
    result = replay_captures(raw, tmp_path / 'replayed.json', dry_run=False)
    assert (result['parsed_count'], result['failed_count']) == (1, 0)
    assert result['files'][0]['selection_method'] == 'manifest'
    assert result['files'][0]['capture_file'].startswith('versions/')
    assert json.loads((tmp_path / 'replayed.json').read_text())[0]['body'] == 'Price 700'
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.rglob('*.json')}


def test_legacy_root_uses_unique_complete_version_without_modifying_archive(tmp_path):
    root, base, enriched = archive(tmp_path)
    result = replay_captures(root, tmp_path / 'new.json')
    assert (result['parsed_count'], result['failed_count']) == (1, 0)
    assert result['files'][0]['selection_method'] == 'unique_legacy_version'
    assert not (root / 'selections').exists()


def test_equivalent_legacy_versions_do_not_duplicate_mail(tmp_path):
    root, base, enriched = archive(tmp_path)
    second = json.loads(enriched.read_text())
    second['captured_at'] = '2099-01-01T00:00:00Z'
    save(root / 'versions' / 'copy.json', second)
    result = replay_captures(root, tmp_path / 'new.json')
    assert (result['parsed_count'], result['failed_count']) == (1, 0)


@pytest.mark.parametrize('different', ['body', 'account'])
def test_ambiguous_legacy_version_refuses_publication_until_explicit_selection(tmp_path, different):
    root, base, enriched = archive(tmp_path)
    second = complete('Price 900') if different == 'body' else json.loads(enriched.read_text())
    if different == 'account':
        second['account_id'] = 'other@example.test'
    alternative = save(root / 'versions' / 'alternative.json', second)
    output = tmp_path / 'new.json'
    with pytest.raises(ValueError, match='failed captures'):
        replay_captures(root, output, dry_run=False)
    assert not output.exists()
    report = json.loads(output.with_suffix('.json.report.json').read_text())
    assert report['files'][0]['error_code'] == 'ambiguous_capture_versions'
    selections = tmp_path / 'chosen'
    manifest(root, base, enriched, selections)
    result = replay_captures(root, output, selection_dir=selections, dry_run=False)
    assert result['parsed_count'] == 1
    assert json.loads(output.read_text())[0]['body'] == 'Price 700'


@pytest.mark.parametrize('corruption', ['root', 'selected', 'missing', 'wrong_message'])
def test_manifest_integrity_checked_before_replay_or_cache(tmp_path, corruption):
    root, base, enriched = archive(tmp_path)
    selection = manifest(root, base, enriched)
    cache = tmp_path / 'cache'
    replay_captures(root, tmp_path / 'first.json', checkpoint_dir=cache, dry_run=False)
    if corruption == 'root':
        base.write_text(base.read_text() + ' ')
    elif corruption == 'selected':
        enriched.write_text(enriched.read_text() + ' ')
    elif corruption == 'missing':
        enriched.unlink()
    else:
        altered = json.loads(enriched.read_text())
        altered['message']['id'] = 'different'
        save(enriched, altered)
        manifest(root, base, enriched)
    output = tmp_path / 'second.json'
    with pytest.raises(ValueError, match='failed captures'):
        replay_captures(root, output, checkpoint_dir=cache, dry_run=False)
    assert not output.exists()


@pytest.mark.parametrize('path', ['../outside.json', 'versions/../../outside.json', 'versions/..\\outside.json',
                                  'C:/outside.json', '/outside.json'])
def test_selection_cannot_read_outside_raw_directory(tmp_path, path):
    root, base, enriched = archive(tmp_path)
    selection = manifest(root, base, enriched)
    record = json.loads(selection.read_text())
    record['capture_file'] = path
    save(selection, record)
    result = replay_captures(root, tmp_path / 'out.json')
    assert result['failed_count'] == 1
    assert result['files'][0]['error_code'] == 'unsafe_capture_path'


def test_explicit_selection_requires_every_message_and_protects_selection_files(tmp_path):
    root, base, enriched = archive(tmp_path)
    selections = tmp_path / 'chosen'
    selections.mkdir()
    result = replay_captures(root, tmp_path / 'out.json', selection_dir=selections)
    assert result['files'][0]['error_code'] == 'capture_selection_missing'
    with pytest.raises(ValueError, match='overlap'):
        replay_captures(root, selections / 'output.json', selection_dir=selections)
    with pytest.raises(ValueError, match='overlap'):
        replay_captures(root, tmp_path / 'out.json', report_path=tmp_path / 'original.json', original_path=tmp_path / 'original.json')


def test_legacy_selection_never_substitutes_a_different_message_version(tmp_path):
    root, base, enriched = archive(tmp_path)
    envelope = json.loads(enriched.read_text())
    envelope['message']['labelIds'] = ['CHANGED']
    save(enriched, envelope)
    result = replay_captures(root, tmp_path / 'out.json')
    assert result['failed_count'] == 1
    assert result['files'][0]['error_code'] == 'missing_captured_body_data'


def test_failed_selection_write_keeps_message_pending(tmp_path, monkeypatch):
    import scripts.sync_gmail_readonly as sync
    reader = GmailReadOnlyProvider(service=Service(bodies={'body-1': base64.urlsafe_b64encode(b'Price 700').decode()}))
    provider = NS(list_message_ids=lambda **kwargs: ['m'], get_message=lambda identifier: message(),
                  capture_message=reader.capture_message, email_from_capture=reader.email_from_capture)
    corpus, state, raw = tmp_path / 'corpus.json', tmp_path / 'state.json', tmp_path / 'raw'
    original = sync._dump_json
    def fail_selection(path, value):
        if path.parent.name == 'selections' and value['capture_file'].startswith('versions/'):
            raise OSError('synthetic storage failure')
        return original(path, value)
    monkeypatch.setattr(sync, '_dump_json', fail_selection)
    with pytest.raises(OSError):
        sync_gmail_to_json(provider, corpus, state, '', 10, raw_dir=raw)
    saved = json.loads(state.read_text())
    assert saved['status'] == 'aborted' and saved['pending_message_ids'] == ['m']
    assert saved['seen_message_ids'] == []
    assert replay_captures(raw, tmp_path / 'out.json')['failed_count'] == 1
