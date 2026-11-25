from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .spec import ESTIMATION_RULE_VERSION


def ceil_div(numerator: int, denominator: int) -> int:
    return -((-numerator) // denominator)


def unclamped_replicas(
    request_count: int,
    concurrency_per_replica: int,
    scaling_factor_permille: int = 1000,
) -> int:
    """Integer fixed-point estimate; all rounding is integer ceiling."""

    if request_count < 0:
        raise ValueError("request_count must be non-negative")
    if concurrency_per_replica <= 0:
        raise ValueError("concurrency_per_replica must be positive")
    if scaling_factor_permille <= 0:
        raise ValueError("scaling_factor_permille must be positive")
    numerator = request_count * scaling_factor_permille
    denominator = concurrency_per_replica * 1000
    return ceil_div(numerator, denominator)


def clamp_replicas(value: int, min_replicas: int, max_replicas: int) -> int:
    return max(min_replicas, min(max_replicas, value))


def estimate_minute(
    request_count: int,
    concurrency_per_replica: int,
    min_replicas: int,
    max_replicas: int,
    scaling_factor_permille: int = 1000,
) -> tuple[int, int]:
    raw_demand = unclamped_replicas(
        request_count, concurrency_per_replica, scaling_factor_permille
    )
    return raw_demand, clamp_replicas(raw_demand, min_replicas, max_replicas)


def numpy_unclamped_replicas(
    request_counts: Sequence[int],
    concurrency_per_replica: int,
    scaling_factor_permille: int = 1000,
) -> list[int]:
    """Vectorized NumPy cross-check for deterministic sample/batch checks."""

    counts = np.asarray(request_counts, dtype=np.int64)
    numerator = counts * np.int64(scaling_factor_permille)
    denominator = np.int64(concurrency_per_replica * 1000)
    values = (numerator + denominator - np.int64(1)) // denominator
    return [int(value) for value in values]


def rule_version() -> str:
    return ESTIMATION_RULE_VERSION


def scaling_action(
    replica_delta: int,
    scale_up_threshold: int,
    scale_down_threshold: int,
) -> str:
    if replica_delta >= scale_up_threshold:
        return "scale_out"
    if -replica_delta >= scale_down_threshold:
        return "scale_in"
    return "hold"


def require_math_import() -> str:
    # Kept as a tiny explicit standard-library dependency marker for audits.
    return math.__name__
