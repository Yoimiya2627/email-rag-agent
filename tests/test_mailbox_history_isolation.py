import json
from types import SimpleNamespace

from core.session_repository import SessionRepository
from core.session_summary import generate_summary, get_summary
from core.session_candidates import extract_candidates
from core.session_context import assemble_session_context


def test_private_status_excluded_from_models_but_visible_to_user(tmp_path, monkeypatch):
    repo = SessionRepository(tmp_path / 'sessions.db')
    for turn_id, metadata in [('private', {'exclude_from_model_context': True}),
                              ('failure', {'status': 'error'})]:
        repo.append_turns('o', 's', [{'turn_id': turn_id, 'query': turn_id + ' query',
            'answer': turn_id + ' answer', 'metadata': metadata, 'include_in_context': False}],
            expected_revision=repo.revision('o', 's'))
    assert repo.get_turn_page('o', 's', 'private')['text'] == 'private answer'
    assert repo.search_history('o', 's', 'private')

    seen = []
    def summarize(*, payload, budget):
        seen.extend(payload['turns'])
        return {'schema_version': 1, 'sections': {}}
    result = generate_summary(repo, 'o', 's', generate=summarize, min_turns=1)
    assert result['status'] == 'generated'
    assert [t['turn_id'] for t in seen] == ['failure']
    seen.clear()
    def candidates(*, payload, budget):
        seen.extend(payload['turns'])
        return {'schema_version': 1, 'candidates': []}
    assert extract_candidates(repo, 'o', 's', generate=candidates)['status'] == 'generated'
    assert [t['turn_id'] for t in seen] == ['failure']

    private = repo.search_history('o', 's', 'private')
    material = assemble_session_context(history_matches=private)
    assert 'private answer' not in json.dumps(material)
    assert 'private query' not in json.dumps(material)

    import agents.tools as tools
    monkeypatch.setattr(tools, '_history_scope', lambda: SimpleNamespace(
        session_repository=repo, owner_id='o', session_id='s', context_epoch=repo.context_epoch('o', 's')))
    assert tools.search_history('private')['items'] == []
    assert tools.search_history('failure')['items']
    assert 'private answer' not in json.dumps(tools.get_turn('private'))
    assert tools.get_turn('failure')['text'] == 'failure answer'

    # Previously generated summaries must not reintroduce a source that is
    # subsequently marked private, even when the original text is unchanged.
    with repo._connect() as db:
        db.execute("UPDATE session_turns SET metadata=? WHERE turn_id='failure'",
                   (json.dumps({'status': 'error', 'exclude_from_model_context': True}),))
    assert get_summary(repo, 'o', 's') is None
