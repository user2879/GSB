"""Integer fixed-point latency statistics.

Every sample is an integer millisecond duration measured with
``time.monotonic_ns()`` (floored to whole ms). Percentiles use the
nearest-rank method on an integer-sorted sample list, so all outputs are
exact integers; the fixed-point encoding then multiplies by 10**scale.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .config import INT64_MAX, INT64_MIN, saturate_int64


def measure_ms(clock_ns: Any = None) -> tuple[int, Any]:
    """Return (start_ns, end_fn); end_fn yields floored integer ms."""
    import time

    start = time.monotonic_ns()

    def end() -> int:
        return (time.monotonic_ns() - start) // 1_000_000

    return start, end


def nearest_rank_index(n: int, percentile: float) -> int:
    """0-based nearest-rank index for percentile in (0, 100]."""
    if n <= 0:
        raise ValueError("percentile requires at least one sample")
    rank = -(-percentile * n // 100)  # ceil(p*n/100)
    return min(n, max(1, rank)) - 1


def percentile(samples: Sequence[int], percentile_value: float = 95.0) -> int:
    if not samples:
        raise ValueError("percentile requires at least one sample")
    ordered = sorted(samples)
    return ordered[nearest_rank_index(len(ordered), percentile_value)]


@dataclass(frozen=True)
class LatencyStats:
    samples: int
    mean_fixed: int
    p95_fixed: int
    max_fixed: int
    scale: int
    # Worst-case absolute error in *decimal units* of the encoded values:
    # integer samples + floor(ms) => <=1 ms measurement floor plus exact
    # integer arithmetic afterwards.
    error_bound_decimal: str
    saturated: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "mean_fixed": self.mean_fixed,
            "p95_fixed": self.p95_fixed,
            "max_fixed": self.max_fixed,
            "scale": self.scale,
            "error_bound_decimal": self.error_bound_decimal,
            "saturated": self.saturated,
        }


def compute_stats(samples: Sequence[int], scale: int) -> LatencyStats:
    """Compute fixed-point stats with explicit error and saturation rules.

    Mean uses floor division (truncation toward zero for non-negative values),
    contributing an absolute mean error of strictly less than 10**-scale in
    decimal units. P95 and max are exact order statistics. Encoding
    saturation clamps to signed int64.
    """
    factor = 10 ** scale
    if not samples:
        return LatencyStats(
            samples=0, mean_fixed=0, p95_fixed=0, max_fixed=0, scale=scale,
            error_bound_decimal="0", saturated=False)
    total = sum(samples)
    n = len(samples)
    mean_exact_fixed = (total * factor) // n
    p95 = percentile(samples, 95.0) * factor
    max_value = max(samples) * factor
    raw_values = (mean_exact_fixed, p95, max_value)
    clamped = tuple(saturate_int64(v) for v in raw_values)
    saturated = any(v > INT64_MAX or v < INT64_MIN for v in raw_values)
    # floor division remainder guarantees mean error < one decimal unit.
    error_bound = f"1e-{scale}" if scale > 0 else "1"
    return LatencyStats(
        samples=n,
        mean_fixed=clamped[0],
        p95_fixed=clamped[1],
        max_fixed=clamped[2],
        scale=scale,
        error_bound_decimal=error_bound,
        saturated=saturated,
    )
