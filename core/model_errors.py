"""Safe permanent model transport failures; never serialize provider responses."""
from openai import APIStatusError

_PERMANENT_STATUSES = frozenset({401, 402, 403, 404})
_STAGES = frozenset({"other", "agent", "agent_plan", "agent_finalize", "analyze", "intent",
    "graph_grade", "rewrite", "summarize", "write_reply", "generate", "stream_generate",
    "graph_generate", "filter", "rerank", "judge", "session_summary", "context_compress",
    "context_candidates"})


class PermanentModelProviderError(RuntimeError):
    error_code = "model_provider_error"

    def __init__(self, http_status: int, stage: str = "other"):
        if type(http_status) is not int or http_status not in _PERMANENT_STATUSES:
            raise ValueError("invalid permanent model provider status")
        self.http_status = http_status
        self.stage = stage if type(stage) is str and stage in _STAGES else "other"
        super().__init__("Model provider configuration or access failed.")

    def as_dict(self) -> dict:
        return {"http_status": self.http_status, "stage": self.stage,
                "error_code": self.error_code}


def provider_error_from_sdk(exc: BaseException, *, stage: str):
    """Call only at a known model SDK transport boundary, never a business tool."""
    if isinstance(exc, APIStatusError) and type(exc.status_code) is int and exc.status_code in _PERMANENT_STATUSES:
        return PermanentModelProviderError(exc.status_code, stage)
    return None


def find_permanent_model_error(exc: BaseException):
    """Recognize our typed provenance through wrappers, without inspecting text."""
    pending, visited = [exc], set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        if isinstance(current, PermanentModelProviderError):
            return current
        pending.extend(value for value in (current.__cause__, current.__context__)
                       if isinstance(value, BaseException))
    return None
