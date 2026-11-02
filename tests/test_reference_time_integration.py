from datetime import date, datetime, timedelta, timezone

import pytest

from agents.execution_scope import ExecutionScope, use_execution_scope, evaluation_reference_instant
from core import pipeline

UTC = timezone.utc
T = datetime(2026, 4, 28, 4, tzinfo=UTC)


def scope(**kwargs):
    return ExecutionScope(owner_id="eval:clock", evaluation=True,
                          approval_store_path="unused", run_dir="unused", **kwargs)


def test_scope_contract_and_legacy_midnight():
    tz = timezone(timedelta(hours=8))
    assert evaluation_reference_instant(scope(evaluation_reference_date=date(2026,4,28)), tz) == datetime(2026,4,28,tzinfo=tz)
    with pytest.raises(ValueError):
        scope(evaluation_reference_date=date(2026,4,28), evaluation_reference_time=T)
    with pytest.raises(ValueError):
        scope(evaluation_reference_time=T.replace(tzinfo=None))
    with pytest.raises(ValueError):
        ExecutionScope(owner_id="ordinary", evaluation_reference_time=T)


def test_conflicts_are_instant_based_and_restore_after_exception():
    with use_execution_scope(scope(evaluation_reference_time=T)):
        with pipeline._fixed_reference_time(T.astimezone(timezone(timedelta(hours=-7)))):
            assert pipeline._now() == T
            with pytest.raises(ValueError):
                with pipeline._fixed_reference_time(T + timedelta(hours=1)):
                    pass
        with pytest.raises(RuntimeError):
            with pipeline._fixed_reference_time():
                raise RuntimeError("fixture")
        assert pipeline._reference_time.get() is None
        assert pipeline._now() == T
    assert pipeline._reference_time.get() is None


def test_scope_cannot_override_existing_retrieval_freeze():
    with pipeline._fixed_reference_time(T):
        with use_execution_scope(scope(evaluation_reference_time=T + timedelta(hours=1))):
            with pytest.raises(ValueError):
                pipeline._now()
        assert pipeline._now() == T


def test_recent_window_remains_half_open(monkeypatch):
    monkeypatch.setattr(pipeline.cfg, "RETRIEVAL_TIMEZONE", "")
    monkeypatch.setattr(pipeline.cfg, "RETRIEVAL_TIMEZONE_OFFSET_HOURS", 8)
    with use_execution_scope(scope(evaluation_reference_date=date(2026,4,28))):
        start, end = pipeline._date_window("最近")
        assert start == datetime(2026,4,14,tzinfo=timezone(timedelta(hours=8)))
        assert end == datetime(2026,4,28,tzinfo=timezone(timedelta(hours=8)))
        spec = pipeline.FilterSpec.from_mapping({"date_hint":"最近"}, now=pipeline._now())
        assert spec.matches({"date":start.isoformat()})
        assert not spec.matches({"date":end.isoformat()})


def test_dst_fold_compares_utc_instants():
    from zoneinfo import ZoneInfo
    tz = ZoneInfo("America/Los_Angeles")
    first = datetime(2026, 11, 1, 1, 30, tzinfo=tz, fold=0)
    second = first.replace(fold=1)
    with use_execution_scope(scope(evaluation_reference_time=first)):
        with pytest.raises(ValueError):
            with pipeline._fixed_reference_time(second):
                pass


def test_concurrent_contexts_and_cancellation_restore():
    import asyncio

    async def worker(value):
        with pipeline._fixed_reference_time(value):
            await asyncio.sleep(0)
            assert pipeline._now() == value
        assert pipeline._reference_time.get() is None

    async def run():
        await asyncio.gather(worker(T), worker(T + timedelta(days=1)))
        with pytest.raises(asyncio.CancelledError):
            with pipeline._fixed_reference_time(T):
                raise asyncio.CancelledError()
        assert pipeline._reference_time.get() is None

    asyncio.run(run())


@pytest.mark.parametrize("stage", ["rewrite", "filter"])
@pytest.mark.parametrize("wrapped", [False, True])
def test_pipeline_does_not_downgrade_permanent_provider_failure(monkeypatch, stage, wrapped):
    from core.model_errors import PermanentModelProviderError
    permanent = PermanentModelProviderError(404, stage)
    error = RuntimeError("wrapper") if wrapped else permanent
    if wrapped:
        error.__cause__ = permanent
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(pipeline, "create_completion", fail)
    monkeypatch.setattr(pipeline, "_get_client", lambda: object())
    monkeypatch.setattr(pipeline.cfg, "ENABLE_QUERY_REWRITE", True)
    fn = pipeline.rewrite_query if stage == "rewrite" else pipeline.extract_filters
    with pytest.raises(PermanentModelProviderError) as caught:
        fn("fixture")
    assert caught.value is permanent
