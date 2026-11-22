"""Replica demand estimation: full recompute vs incremental ring window."""

from __future__ import annotations

from collections import deque

import numpy as np

from .errors import DivergenceError
from .models import ClusterSpec, MinuteEstimate, ServiceSpec, canonical_bytes

RULE_VERSION = "replica-int-v1"
MINUTE_MS = 60_000
_BASIS_POINTS = 10_000

# Field order used when locating the first divergent scalar.
_COMPARE_FIELDS = (
    "minute",
    "service_id",
    "requests",
    "per_replica_capacity",
    "raw_demand",
    "demand",
    "min_replicas",
    "max_replicas",
    "rule_version",
    "scale_action",
    "utilization_bp",
)


def per_replica_capacity(service: ServiceSpec) -> int:
    """Integer requests/minute one replica can absorb."""

    if service.latency_ms == 0:
        # A zero-cost request finishes immediately: each concurrency slot can
        # serve one request per millisecond within the 60_000 ms minute.
        return service.concurrency_per_replica * MINUTE_MS
    return (service.concurrency_per_replica * MINUTE_MS) // service.latency_ms


def estimate_service_minute(service: ServiceSpec, minute: int) -> MinuteEstimate:
    """Pure integer rule shared by both estimation paths."""

    requests = service.requests[minute]
    capacity = per_replica_capacity(service)

    if capacity == 0:
        # No serving capacity exists: demand is infeasible, expressed at the
        # service ceiling so clamping and rejection remain well defined.
        raw_demand = service.max_replicas
    elif requests == 0:
        raw_demand = 0
    else:
        raw_demand = (requests + capacity - 1) // capacity

    demand = min(service.max_replicas, max(service.min_replicas, raw_demand))

    if demand > 0 and capacity > 0:
        utilization = min(
            _BASIS_POINTS,
            (requests * _BASIS_POINTS) // (demand * capacity),
        )
    else:
        utilization = 0

    if raw_demand > demand:
        action = "saturate"
    elif utilization >= service.scale_up_threshold:
        action = "scale_up"
    elif requests > 0 and utilization <= service.scale_down_threshold:
        action = "scale_down"
    else:
        action = "hold"

    return MinuteEstimate(
        minute=minute,
        service_id=service.service_id,
        requests=requests,
        per_replica_capacity=capacity,
        raw_demand=raw_demand,
        demand=demand,
        min_replicas=service.min_replicas,
        max_replicas=service.max_replicas,
        rule_version=RULE_VERSION,
        scale_action=action,
        utilization_bp=utilization,
    )


def select_path(spec: ClusterSpec) -> tuple[str, str, int]:
    """Deterministic default path selection around an explicit boundary.

    Crossing rule is strict ``minutes > threshold`` so the boundary minute
    itself stays on the full path and the boundary value is auditable.
    """

    boundary = spec.incremental_threshold_minutes
    minutes = spec.minutes
    if minutes > boundary:
        return (
            "incremental",
            f"minutes={minutes} strictly exceeds boundary={boundary}",
            boundary,
        )
    return (
        "full",
        f"minutes={minutes} does not exceed boundary={boundary}",
        boundary,
    )


class FullEstimator:
    """Recomputes every minute independently with no carried state."""

    name = "full"

    def run(self, spec: ClusterSpec) -> tuple[tuple[MinuteEstimate, ...], ...]:
        rows: list[tuple[MinuteEstimate, ...]] = []
        for minute in range(spec.minutes):
            rows.append(tuple(
                estimate_service_minute(service, minute)
                for service in spec.services
            ))
        return tuple(rows)


class IncrementalEstimator:
    """Single-pass scan over a fixed-length NumPy ring buffer.

    The rolling window sum is updated incrementally (subtract the slot being
    overwritten before inserting the new load), reusing the previous minute's
    window state. Demand itself always comes from the shared pure integer
    rule, so both paths emit byte-identical decision rows.
    """

    name = "incremental"

    def __init__(self, window_size: int, fixed_point_scale: int) -> None:
        self.window_size = window_size
        self.fixed_point_scale = fixed_point_scale
        self._ring: np.ndarray | None = None
        self._window_sum: np.ndarray | None = None
        self.window_history: deque[tuple[int, tuple[int, ...], tuple[int, ...]]] = deque()

    def run(self, spec: ClusterSpec) -> tuple[tuple[MinuteEstimate, ...], ...]:
        service_count = len(spec.services)
        self._ring = np.zeros((self.window_size, service_count), dtype=np.int64)
        self._window_sum = np.zeros(service_count, dtype=np.int64)
        rows: list[tuple[MinuteEstimate, ...]] = []

        for minute in range(spec.minutes):
            load = np.fromiter(
                (service.requests[minute] for service in spec.services),
                dtype=np.int64,
                count=service_count,
            )
            slot = minute % self.window_size
            if minute >= self.window_size:
                self._window_sum -= self._ring[slot]
            self._ring[slot] = load
            self._window_sum += load
            observed = min(minute + 1, self.window_size)
            window_mean_fp = tuple(
                int(value * self.fixed_point_scale // observed)
                for value in self._window_sum.tolist()
            )
            self.window_history.append(
                (minute, tuple(int(v) for v in self._window_sum.tolist()), window_mean_fp)
            )
            rows.append(tuple(
                estimate_service_minute(service, minute)
                for service in spec.services
            ))
        return tuple(rows)


def estimates_digest(rows: tuple[tuple[MinuteEstimate, ...], ...]) -> str:
    """Stable digest over every scalar in service-id/minute order."""

    payload = [
        [estimate.as_tuple() for estimate in minute_rows]
        for minute_rows in rows
    ]
    return hashlib_sha256(payload)


def hashlib_sha256(payload: list) -> str:
    import hashlib

    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def assert_estimates_equal(
    full_rows: tuple[tuple[MinuteEstimate, ...], ...],
    incremental_rows: tuple[tuple[MinuteEstimate, ...], ...],
) -> None:
    """Raise E_DIVERGE at the first minute/service/field that differs."""

    if len(full_rows) != len(incremental_rows):
        raise DivergenceError(
            "estimation paths diverged on minute count",
            details={
                "minute": min(len(full_rows), len(incremental_rows)),
                "service_id": None,
                "field": "minute_count",
                "full": len(full_rows),
                "incremental": len(incremental_rows),
            },
        )
    for minute, (left_rows, right_rows) in enumerate(zip(full_rows, incremental_rows)):
        for left, right in zip(left_rows, right_rows):
            left_tuple = left.as_tuple()
            right_tuple = right.as_tuple()
            for field_name, left_value, right_value in zip(
                _COMPARE_FIELDS, left_tuple, right_tuple
            ):
                if left_value != right_value:
                    raise DivergenceError(
                        "estimation paths diverged",
                        details={
                            "minute": minute,
                            "service_id": left.service_id,
                            "field": field_name,
                            "full": left_value,
                            "incremental": right_value,
                        },
                    )


def run_both_paths(
    spec: ClusterSpec,
) -> tuple[
    tuple[tuple[MinuteEstimate, ...], ...],
    tuple[tuple[MinuteEstimate, ...], ...],
    IncrementalEstimator,
]:
    """Execute both paths and enforce byte-identical outputs."""

    full_rows = FullEstimator().run(spec)
    incremental = IncrementalEstimator(spec.window_size, spec.fixed_point_scale)
    incremental_rows = incremental.run(spec)
    assert_estimates_equal(full_rows, incremental_rows)
    return full_rows, incremental_rows, incremental
