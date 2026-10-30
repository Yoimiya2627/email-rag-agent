"""Model completion semantics without provider, configuration or storage imports.

ModelText is string-compatible at existing internal call boundaries. JSON and
Pydantic string serialization discard its attributes: response builders MUST
copy outcome_metadata(text) into metadata before serializing. Bare nonempty
strings remain compatible with deterministic responses and test doubles.
"""
from __future__ import annotations

_FINISH_REASONS = {"stop", "length", "content_filter", "tool_calls", "function_call"}
_EMPTY_MESSAGE = "模型未返回有效回答，任务尚未完成。"


class ModelText(str):
    def __new__(cls, text: str, *, completion_status="complete", finish_reason=None,
                error_code=None):
        value = super().__new__(cls, text)
        if completion_status not in {"complete", "incomplete", "error"}:
            raise ValueError("invalid completion status")
        if not text.strip():
            completion_status, error_code = "error", "empty_model_response"
        value.completion_status = completion_status
        value.finish_reason = finish_reason if finish_reason in _FINISH_REASONS else None
        value.error_code = error_code
        return value


def text_from_choice(choice) -> ModelText:
    """Never treat a length/content-filtered response as a completed answer.

    The SDK provides finish_reason. A missing attribute is supported for legacy
    lightweight test doubles; an explicitly null reason is unfinished.
    """
    content = getattr(getattr(choice, "message", None), "content", None)
    text = content.strip() if isinstance(content, str) else ""
    reason = getattr(choice, "finish_reason", "stop")
    if not text:
        return ModelText("", completion_status="error", finish_reason=reason,
                         error_code="empty_model_response")
    if reason != "stop":
        return ModelText(text, completion_status="incomplete", finish_reason=reason,
                         error_code="model_output_incomplete")
    return ModelText(text, finish_reason=reason)


def outcome_metadata(text: str) -> dict:
    """Safe terminal metadata; never includes model text or provider diagnostics."""
    completion = getattr(text, "completion_status", "complete")
    code = getattr(text, "error_code", None)
    if not str(text).strip():
        completion, code = "error", "empty_model_response"
    status = {"complete": "success", "incomplete": "incomplete", "error": "error"}[completion]
    if completion == "error" and code == "empty_model_response":
        status = "empty_model_response"
    return {"status": status, "completion_status": completion,
            "finish_reason": getattr(text, "finish_reason", None), "error_code": code}


def display_text(text: str) -> str:
    return str(text) if str(text).strip() else _EMPTY_MESSAGE


def text_from_response(response) -> ModelText:
    """Restore outcome after an AgentResponse string field erased ModelText."""
    metadata = getattr(response, "metadata", None) or {}
    completion = metadata.get("completion_status")
    if completion is None:
        status = metadata.get("status", "success")
        completion = "complete" if status == "success" else "incomplete" if status in {"incomplete", "partial"} else "error"
    return ModelText(response.answer, completion_status=completion,
                     finish_reason=metadata.get("finish_reason"),
                     error_code=metadata.get("error_code"))


class ModelOutputError(RuntimeError):
    """A stream ended without a complete answer; partial_text is display-only.

    The exception message is safe for logs. Consumers must not replay tools or
    submit successful history on this exception. Deadlines/cancellation remain
    their original exceptions and must not be disguised as this outcome.
    """
    def __init__(self, partial_text="", *, completion_status="incomplete",
                 finish_reason=None, error_code="model_output_incomplete"):
        outcome = ModelText(partial_text, completion_status=completion_status,
                            finish_reason=finish_reason, error_code=error_code)
        self.partial_text = str(outcome)
        self.completion_status = outcome.completion_status
        self.finish_reason = outcome.finish_reason
        self.error_code = outcome.error_code
        self.metadata = outcome_metadata(outcome)
        super().__init__("Model returned no answer" if self.error_code == "empty_model_response"
                         else "Model answer is incomplete")
