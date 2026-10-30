"""Transport-independent response handling. Failed output is display-only."""
from core.model_outcomes import display_text, outcome_metadata


def response_metadata(response) -> dict:
    metadata = dict(response.metadata or {})
    inferred = outcome_metadata(response.answer)
    for key, value in inferred.items():
        metadata.setdefault(key, value)
    if not str(response.answer).strip():
        metadata.update(inferred)
    return metadata


def can_commit_answer(answer: str, metadata: dict) -> bool:
    return (bool(str(answer).strip())
            and metadata.get('completion_status', 'complete') == 'complete'
            and metadata.get('status', 'success') in {'success', 'approval_required'})


def finalize_response(response):
    response.metadata = response_metadata(response)
    response.answer = display_text(response.answer)
    return response
