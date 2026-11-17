"""Deterministic full and incremental sliding-window aggregators."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Literal

from .spec import Observation


PathName = Literal["full", "incremental"]
MEMORY_RECORD_BYTES = 32
PATH_OVERHEAD_BYTES = 4096


@dataclass(frozen=True)
class WindowStats:
    samples: int
    total_requests: int
    total_errors: int
    total_latency_ms: int
    avg_requests: int
    avg_latency_ms: int
    error_rate_ppm: int

    def metric_value(self, metric: str) -> int:
        if metric == "requests":
            return self.avg_requests
        if metric == "latency":
            return self.avg_latency_ms
        if metric == "error_rate":
            return self.error_rate_ppm
        raise ValueError(f"unsupported metric: {metric}")

    def to_dict(self) -> dict[str, int]:
        return {
            "samples": self.samples,
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "total_latency_ms": self.total_latency_ms,
            "avg_requests": self.avg_requests,
            "avg_latency_ms": self.avg_latency_ms,
            "error_rate_ppm": self.error_rate_ppm,
        }


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return (numerator + denominator - 1) // denominator


def _stats(
    samples: int,
    total_requests: int,
    total_errors: int,
    total_latency_ms: int,
) -> WindowStats:
    avg_requests = _ceil_div(total_requests, samples)
    avg_latency = _ceil_div(total_latency_ms, samples)
    error_rate_ppm = (
        _ceil_div(total_errors * 1_000_000, total_requests)
        if total_requests
        else 0
    )
    return WindowStats(
        samples=samples,
        total_requests=total_requests,
        total_errors=total_errors,
        total_latency_ms=total_latency_ms,
        avg_requests=avg_requests,
        avg_latency_ms=avg_latency,
        error_rate_ppm=error_rate_ppm,
    )


def choose_path(minute_count: int, threshold: int, requested: str = "auto") -> PathName:
    if requested not in {"auto", "full", "incremental"}:
        raise ValueError("path must be auto, full or incremental")
    if requested != "auto":
        return requested  # type: ignore[return-value]
    return "full" if minute_count < threshold else "incremental"


def estimate_peak_memory_bytes(
    path: PathName,
    workload_count: int,
    minute_count: int,
    window_size: int,
) -> int:
    """Stable estimate of aggregator-held observation bytes.

    The full path retains every observation for recomputation; the incremental
    path retains a fixed-size ring per workload plus one current row per
    workload.  Python object overhead is intentionally modeled as a fixed
    allowance so the number does not depend on hash randomization or platform.
    """

    if path == "full":
        rows = workload_count * minute_count
    else:
        rows = workload_count * (window_size + 1)
    return PATH_OVERHEAD_BYTES + rows * MEMORY_RECORD_BYTES


class Aggregator:
    def __init__(self, path: PathName, workload_ids: list[str], window_size: int):
        self.path = path
        self.workload_ids = tuple(sorted(workload_ids))
        self.window_size = window_size
        self._peak_rows = 0

    @property
    def peak_memory_estimate_bytes(self) -> int:
        return PATH_OVERHEAD_BYTES + self._peak_rows * MEMORY_RECORD_BYTES

    def add(self, workload_id: str, observation: Observation) -> WindowStats:
        raise NotImplementedError


class FullWindowAggregator(Aggregator):
    """Recompute each trailing window from retained history."""

    def __init__(self, workload_ids: list[str], window_size: int):
        super().__init__("full", workload_ids, window_size)
        self._history: dict[str, list[Observation]] = {
            workload_id: [] for workload_id in self.workload_ids
        }
        self._rows = 0

    def add(self, workload_id: str, observation: Observation) -> WindowStats:
        history = self._history[workload_id]
        history.append(observation)
        self._rows += 1
        self._peak_rows = max(self._peak_rows, self._rows)
        window = history[-self.window_size :]
        return _stats(
            len(window),
            sum(item.requests for item in window),
            sum(item.errors for item in window),
            sum(item.avg_latency_ms for item in window),
        )


class IncrementalWindowAggregator(Aggregator):
    """Single-pass fixed-size ring with maintained integer totals."""

    def __init__(self, workload_ids: list[str], window_size: int):
        super().__init__("incremental", workload_ids, window_size)
        self._rings: dict[str, Deque[Observation]] = {
            workload_id: deque() for workload_id in self.workload_ids
        }
        self._totals: dict[str, list[int]] = {
            workload_id: [0, 0, 0] for workload_id in self.workload_ids
        }
        self._rows = 0
        self._peak_rows = 0

    def add(self, workload_id: str, observation: Observation) -> WindowStats:
        ring = self._rings[workload_id]
        totals = self._totals[workload_id]
        evicted = len(ring) == self.window_size
        if evicted:
            old = ring.popleft()
            totals[0] -= old.requests
            totals[1] -= old.errors
            totals[2] -= old.avg_latency_ms
        ring.append(observation)
        totals[0] += observation.requests
        totals[1] += observation.errors
        totals[2] += observation.avg_latency_ms
        if not evicted:
            self._rows += 1
        self._peak_rows = max(self._peak_rows, self._rows)
        return _stats(len(ring), totals[0], totals[1], totals[2])


def make_aggregator(path: PathName, workload_ids: list[str], window_size: int) -> Aggregator:
    if path == "full":
        return FullWindowAggregator(workload_ids, window_size)
    return IncrementalWindowAggregator(workload_ids, window_size)
