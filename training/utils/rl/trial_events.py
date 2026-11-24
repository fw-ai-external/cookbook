"""Process-local accounting for harness trials that never reach a trajectory.

A completed trajectory carries its own TITO summary, so its timing reaches
WandB through the per-step trajectory merge. Attempts that fail — a broken
provider stream, a whole-trial-bound firing, a materialization error — produce
no trajectory at all, so without this accumulator a step that burned half its
wall clock on doomed trials looks exactly like a clean one.

The recorders are called from the client event loop (the rollout retry loop and
the trial driver), never from the artifact worker process: a counter
incremented in a child process would be discarded with that process. Values
are drained once per optimizer step by ``publish_tito_sidecar_metrics``.
"""

from __future__ import annotations

import re
import threading
from bisect import bisect_left
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

_COUNTER_ROOT = "tito/trial/"
_FAILED_PHASE_ROOT = "tito/trial_failed/"
_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")

_LOCK = threading.Lock()
_COUNTERS: dict[str, float] = {}
_DURATION_BUCKETS = (
    0.001,
    0.01,
    0.1,
    0.5,
    1.0,
    5.0,
    10.0,
    30.0,
    60.0,
    180.0,
    300.0,
    900.0,
)


@dataclass(slots=True)
class _Distribution:
    bins: list[int] | None
    count: int = 0
    total: float = 0.0
    minimum: float = float("inf")
    maximum: float = -float("inf")

    def observe(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)
        if self.bins is not None:
            self.bins[bisect_left(_DURATION_BUCKETS, value)] += 1

    def quantile_upper_bound(self, quantile: float) -> float:
        assert self.bins is not None
        total = 0
        for bound, count in zip(
            (*_DURATION_BUCKETS, self.maximum), self.bins, strict=True
        ):
            total += count
            if total >= self.count * quantile:
                return min(bound, self.maximum)
        raise AssertionError("duration histogram has an incorrect observation count")


_DISTRIBUTIONS: dict[str, _Distribution] = {}


def _slug(reason: str) -> str:
    value = _SLUG_PATTERN.sub("_", reason.strip().lower()).strip("_")
    return value or "unknown"


def _increment(name: str, amount: float = 1.0) -> None:
    with _LOCK:
        _COUNTERS[name] = _COUNTERS.get(name, 0.0) + amount


def _observe(name: str, value: float) -> None:
    with _LOCK:
        distribution = _DISTRIBUTIONS.get(name)
        if distribution is None:
            bins = (
                [0] * (len(_DURATION_BUCKETS) + 1)
                if name.endswith("_seconds")
                else None
            )
            distribution = _DISTRIBUTIONS[name] = _Distribution(bins=bins)
        distribution.observe(float(value))


def record_trial_attempt() -> None:
    """One Harbor trial attempt started (retries count separately)."""
    _increment(f"{_COUNTER_ROOT}attempts")


def record_trial_retry(reason: str) -> None:
    """A transient attempt failure that will be retried on a fresh trial."""
    _increment(f"{_COUNTER_ROOT}retries")
    _increment(f"{_COUNTER_ROOT}retry_reason/{_slug(reason)}")


def record_trial_discarded(reason: str) -> None:
    """A task abandoned after exhausting its attempts, or a fatal attempt."""
    _increment(f"{_COUNTER_ROOT}discarded")
    _increment(f"{_COUNTER_ROOT}discard_reason/{_slug(reason)}")


def record_trial_reward(source: Literal["verifier", "terminal_fallback"]) -> None:
    """One materialized trial with a verifier or configured terminal reward."""
    _increment(f"{_COUNTER_ROOT}reward_source/{source}")


def record_trial_timeout() -> None:
    """The whole-trial bound fired: a silent hang became a retryable failure."""
    _increment(f"{_COUNTER_ROOT}whole_trial_bound_firings")


def record_failed_trial_wall(seconds: float) -> None:
    """Wall clock burned by an attempt that produced no trajectory."""
    _observe(f"{_FAILED_PHASE_ROOT}trial_wall_seconds", seconds)


def record_failed_trial_phases(phases: Mapping[str, float]) -> None:
    """Harbor phase brackets of a trial that ran but yielded no trajectory."""
    for phase, value in phases.items():
        _observe(f"{_FAILED_PHASE_ROOT}{phase}", value)


def drain() -> dict[str, float]:
    """Return and clear everything recorded since the previous drain."""
    with _LOCK:
        counters = _COUNTERS
        distributions = _DISTRIBUTIONS
        _reset_locked()
    drained: dict[str, float] = dict(counters)
    for name, distribution in distributions.items():
        drained[f"{name}_count"] = float(distribution.count)
        drained[f"{name}_sum"] = distribution.total
        drained[f"{name}_min"] = distribution.minimum
        drained[f"{name}_max"] = distribution.maximum
        drained[f"{name}_mean"] = distribution.total / distribution.count
        if distribution.bins is not None:
            drained[f"{name}_p95_upper_bound"] = distribution.quantile_upper_bound(0.95)
            drained[f"{name}_p99_upper_bound"] = distribution.quantile_upper_bound(0.99)
    return drained


def _reset_locked() -> None:
    global _COUNTERS, _DISTRIBUTIONS
    _COUNTERS = {}
    _DISTRIBUTIONS = {}


def reset() -> None:
    """Drop everything recorded so far (test isolation)."""
    with _LOCK:
        _reset_locked()


__all__ = [
    "drain",
    "record_failed_trial_phases",
    "record_failed_trial_wall",
    "record_trial_attempt",
    "record_trial_discarded",
    "record_trial_retry",
    "record_trial_reward",
    "record_trial_timeout",
    "reset",
]
