"""Model/stage-aware context estimates without downloading arbitrary tokenizers."""
import json
import threading

_ADAPTERS = {}
_LOCK = threading.RLock()


def register_token_counter(model, revision, stage, counter, *, validation_samples):
    """Register a backend adapter after checking operator-supplied golden counts.

    Exact provider framing is still not claimed: counts receive explicit framing
    headroom. No wildcard model/revision matching or network lookup occurs.
    """
    if not all(isinstance(v, str) and v for v in (model, revision, stage)) or not callable(counter):
        raise ValueError('model, revision, stage and callable counter are required')
    if not validation_samples:
        raise ValueError('token counter requires validation samples')
    for raw, expected in validation_samples:
        actual = counter(raw)
        if type(expected) is not int or expected < 0 or type(actual) is not int or actual != expected:
            raise ValueError('token counter validation failed')
    with _LOCK:
        _ADAPTERS[(model, revision, stage)] = counter


def unregister_token_counter(model, revision, stage):
    with _LOCK:
        _ADAPTERS.pop((model, revision, stage), None)


def measure_context(messages, schemas=None, *, token_counter=None, model=None, model_revision=None, stage=None):
    raw = json.dumps([messages, schemas or []], ensure_ascii=False, default=str)
    matched = False
    if token_counter is None:
        with _LOCK:
            token_counter = _ADAPTERS.get((model, model_revision, stage))
        matched = token_counter is not None
    degraded = None
    if token_counter is None:
        tokens, method = len(raw.encode('utf-8')), 'utf8_bytes_upper_bound'
    else:
        try:
            tokens, method = token_counter(raw), 'validated_model_adapter' if matched else 'injected_tokenizer'
            if type(tokens) is not int or tokens < 0:
                raise ValueError('token counter must return a nonnegative integer')
        except Exception as exc:
            if not matched:
                raise
            tokens, method = len(raw.encode('utf-8')), 'utf8_bytes_upper_bound'
            degraded = type(exc).__name__
    tokens += 8 * len(messages) + 16
    return {'context_chars': len(raw), 'estimated_input_tokens': tokens,
            'token_estimation_method': method, 'usage_is_estimated': True,
            **({'token_counter_degraded':degraded} if degraded else {})}


def within_budget(usage, *, char_limit, token_limit=0, output_reserve=0):
    if type(char_limit) is not int or char_limit < 1:
        raise ValueError('context character limit must be positive')
    if type(token_limit) is not int or token_limit < 0 or type(output_reserve) is not int or output_reserve < 0:
        raise ValueError('invalid token capacity or output reserve')
    return (usage['context_chars'] <= char_limit and
            (not token_limit or usage['estimated_input_tokens'] + output_reserve <= token_limit))


def model_profile(model, stage=None):
    """Trusted operator profiles may narrow, never enlarge a run's capacity."""
    import config.settings as cfg
    profiles = getattr(cfg, 'MODEL_CONTEXT_PROFILES', {})
    if not isinstance(profiles, dict):
        raise ValueError('model context profiles must be a mapping')
    profile = profiles.get(model, {})
    if not isinstance(profile, dict):
        raise ValueError('invalid model context profile')
    stages = profile.get('stages', {})
    if not isinstance(stages, dict) or not isinstance(stages.get(stage, {}), dict):
        raise ValueError('invalid model stage profile')
    result = {**profile, **stages.get(stage, {})}
    capacity = result.get('context_tokens', 0)
    base_capacity = profile.get('context_tokens', 0)
    if type(base_capacity) is not int or base_capacity < 0:
        raise ValueError('invalid model context capacity')
    if type(capacity) is not int or capacity < 0:
        raise ValueError('invalid model context capacity')
    if base_capacity:
        capacity = min(base_capacity, capacity) if capacity else base_capacity
    revision = result.get('revision')
    if revision is None and model == getattr(cfg, 'DEEPSEEK_MODEL', None):
        revision = getattr(cfg, 'MODEL_REVISION', None)
    if revision is not None and (not isinstance(revision, str) or not revision):
        raise ValueError('invalid model revision')
    return {'context_tokens':capacity, 'revision':revision}
