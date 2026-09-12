"""Request-local, body-free index diagnostics with exclusive phase timings."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
import time

_current = ContextVar("index_metrics", default=None)


@dataclass
class IndexMetrics:
    stages_seconds: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)
    outcome: str = "pending"
    started: float = field(default_factory=lambda: time.perf_counter())
    finished: float | None = None
    _stack: list = field(default_factory=list, repr=False)

    @property
    def elapsed_seconds(self):
        return (self.finished if self.finished is not None else time.perf_counter()) - self.started

    def to_dict(self):
        elapsed = self.elapsed_seconds
        return {"stages_seconds": {k: round(v, 6) for k, v in self.stages_seconds.items()},
                "counts": dict(self.counts), "outcome": self.outcome,
                "total_seconds": round(elapsed, 6),
                "unaccounted_seconds": round(max(0., elapsed - sum(self.stages_seconds.values())), 6)}


def current_metrics():
    return _current.get()


@contextmanager
def collect_index_metrics():
    existing = current_metrics()
    if existing is not None:
        yield existing
        return
    report = IndexMetrics()
    token = _current.set(report)
    try:
        yield report
    except BaseException as exc:
        # Do not include exception messages: upstream validation may carry text.
        report.outcome = "cancelled" if type(exc).__name__ in {"RunCancelled", "KeyboardInterrupt"} else "failed"
        raise
    finally:
        report.finished = time.perf_counter()
        _current.reset(token)


def add_count(name, value=1):
    report = current_metrics()
    if report is not None:
        report.counts[name] = report.counts.get(name, 0) + value


def set_outcome(value):
    report = current_metrics()
    if report is not None:
        report.outcome = value


@contextmanager
def measure_stage(name):
    report = current_metrics()
    if report is None:
        yield
        return
    frame = [time.perf_counter(), 0.]
    report._stack.append(frame)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - frame[0]
        report._stack.pop()
        report.stages_seconds[name] = report.stages_seconds.get(name, 0.) + max(0., elapsed - frame[1])
        if report._stack:
            report._stack[-1][1] += elapsed


def timed_stage(name):
    def decorate(function):
        @wraps(function)
        def measured(*args, **kwargs):
            with measure_stage(name):
                return function(*args, **kwargs)
        return measured
    return decorate


def index_operation(function):
    @wraps(function)
    def measured(*args, **kwargs):
        with collect_index_metrics():
            return function(*args, **kwargs)
    return measured
