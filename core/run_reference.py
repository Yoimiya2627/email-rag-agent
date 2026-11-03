"""Versioned Agent reference clock; wall-clock deadlines remain independent."""
from contextlib import contextmanager
from datetime import datetime, timezone

from core.tool_results import ToolResultUnavailable


def timezone_policy(tz):
    key = getattr(tz, "key", None)
    if key is not None:
        return {"kind": "iana", "name": key}
    delta = tz.utcoffset(None)
    return {"kind": "offset", "microseconds": (delta.days * 86400 + delta.seconds) * 1000000 + delta.microseconds}


def restore_reference(checkpoint, tz):
    if checkpoint is None:
        return None
    if "reference_clock" not in checkpoint:
        raise ToolResultUnavailable("checkpoint_reference_requires_restart")
    value = checkpoint["reference_clock"]
    if not isinstance(value, dict) or set(value) != {"version", "instant_utc", "timezone"}:
        raise ToolResultUnavailable("checkpoint_reference_invalid")
    if type(value["version"]) is not int or value["version"] != 1:
        raise ToolResultUnavailable("checkpoint_reference_version_unsupported")
    try:
        instant = datetime.fromisoformat(value["instant_utc"])
        if instant.tzinfo is None or instant.utcoffset() is None:
            raise ValueError()
        instant = instant.astimezone(timezone.utc)
    except (ValueError, TypeError, OverflowError):
        raise ToolResultUnavailable("checkpoint_reference_invalid") from None
    policy = value["timezone"]
    if not isinstance(policy, dict):
        raise ToolResultUnavailable("checkpoint_reference_invalid")
    if policy.get("kind") == "iana":
        valid = set(policy) == {"kind", "name"} and type(policy.get("name")) is str and bool(policy["name"])
    elif policy.get("kind") == "offset":
        valid = set(policy) == {"kind", "microseconds"} and type(policy.get("microseconds")) is int and abs(policy["microseconds"]) < 86400000000
    else:
        valid = False
    if not valid:
        raise ToolResultUnavailable("checkpoint_reference_invalid")
    if policy != timezone_policy(tz):
        raise ToolResultUnavailable("checkpoint_reference_timezone_changed")
    return instant


def prepare_reference(checkpoint=None):
    from core import pipeline
    tz = pipeline._retrieval_timezone()
    value = restore_reference(checkpoint, tz)
    try:
        value = pipeline._clock_reference(value) or pipeline._now()
    except ValueError:
        raise ToolResultUnavailable("checkpoint_reference_conflict" if checkpoint is not None else "run_reference_conflict") from None
    return value, {"version": 1, "instant_utc": value.astimezone(timezone.utc).isoformat(),
                   "timezone": timezone_policy(tz)}


@contextmanager
def run_reference(checkpoint=None, *, prepared=None):
    from core import pipeline
    value, contract = prepare_reference(checkpoint) if prepared is None else prepared
    if contract["timezone"] != timezone_policy(pipeline._retrieval_timezone()):
        raise ToolResultUnavailable("checkpoint_reference_timezone_changed")
    with pipeline._fixed_reference_time(value):
        yield contract


_REFERENCE_ERROR_CODES = frozenset({
    "checkpoint_reference_requires_restart",
    "checkpoint_reference_invalid",
    "checkpoint_reference_version_unsupported",
    "checkpoint_reference_timezone_changed",
    "checkpoint_reference_conflict",
    "run_reference_conflict",
})


def reference_error_code(exc):
    """Expose only known clock recovery codes, never arbitrary exception text."""
    if isinstance(exc, ToolResultUnavailable) and len(exc.args) == 1:
        code = exc.args[0]
        if type(code) is str and code in _REFERENCE_ERROR_CODES:
            return code
    return None
