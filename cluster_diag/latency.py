from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

from .ledger import LatencyEvent


CLOCK_BASIS = "time.monotonic_ns"
TRUTH_SOURCE = "sqlite:latency_events"
STATS_SCHEMA_VERSION = "latency-stats-v1:unit=ms_x1000:mean=half-up:p95=nearest-rank"
QUANTIZATION_ERROR_UPPER_BOUND_MS_X1000 = 1


def measure_elapsed_us(started_ns: int, ended_ns: int) -> int:
    """Round a monotonic nanosecond span to integer microseconds.

    The truth record is a monotonic nanosecond interval. The reported fixed-point
    millisecond value is ``elapsed_ms_x1000``; one unit is 0.001 ms (1 us).
    """

    return max(0, (int(ended_ns) - int(started_ns) + 500) // 1000)


def _round_div(numerator: int, denominator: int) -> int:
    sign = -1 if numerator < 0 else 1
    return sign * ((abs(numerator) + denominator // 2) // denominator)


def _p95_nearest_rank(values: list[int]) -> int:
    ordered = sorted(values)
    rank = max(1, (95 * len(ordered) + 99) // 100)
    return ordered[rank - 1]


def aggregate_latency_events(events: Iterable[LatencyEvent]) -> dict[str, Any]:
    """Aggregate ledger latency truth using integer-only fixed-point arithmetic."""

    grouped: dict[tuple[str, str | None, int | None], list[LatencyEvent]] = defaultdict(list)
    for event in events:
        grouped[(event.kind, event.cluster_id, event.revision)].append(event)

    groups: list[dict[str, Any]] = []
    for (kind, cluster_id, revision), values in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][1] or "", item[0][2] if item[0][2] is not None else -1),
    ):
        samples = [event.elapsed_us for event in values]
        total = sum(samples)
        groups.append(
            {
                "kind": kind,
                "cluster_id": cluster_id,
                "revision": revision,
                "count": len(samples),
                "mean_ms_x1000": _round_div(total, len(samples)),
                "p95_ms_x1000": _p95_nearest_rank(samples),
                "max_ms_x1000": max(samples),
                "exceeded_count": sum(1 for event in values if event.exceeded),
                "clock_basis": CLOCK_BASIS,
                "truth_source": TRUTH_SOURCE,
            }
        )

    return {
        "schema": STATS_SCHEMA_VERSION,
        "clock_basis": CLOCK_BASIS,
        "truth_source": TRUTH_SOURCE,
        "unit": "milliseconds_x1000",
        "quantization": "monotonic_ns rounded to 1us; divisor 1000 per ms",
        "quantization_error_upper_bound_ms_x1000": QUANTIZATION_ERROR_UPPER_BOUND_MS_X1000,
        "groups": groups,
    }
