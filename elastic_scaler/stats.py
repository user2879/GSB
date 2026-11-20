"""Latency measurement and integer fixed-point aggregation.

Timing uses :func:`time.perf_counter_ns` (a monotonic clock). Each sample is
stored as integer microseconds via floor division, so the per-sample
quantization error is strictly below 1 microsecond; millisecond statistics are
derived deterministically from those integer samples.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

NS_PER_US = 1_000
US_PER_MS = 1_000
SCALE_MILLI = 1_000  # three-decimal fixed point


@dataclass(frozen=True)
class LatencyStats:
    samples: int
    mean_micro: int
    p95_micro: int
    max_micro: int
    quantization_error_bound_us: int

    def to_dict(self) -> dict[str, object]:
        return {
            "samples": self.samples,
            "mean_ms_fixed3": _fixed3(self.mean_micro),
            "p95_ms_fixed3": _fixed3(self.p95_micro),
            "max_ms_fixed3": _fixed3(self.max_micro),
            "quantization_error_bound_us": self.quantization_error_bound_us,
        }


class LatencyRecorder:
    def __init__(self) -> None:
        self._samples_us: list[int] = []

    def measure(self) -> "LatencySample":
        return LatencySample(self)

    def record_ns(self, elapsed_ns: int) -> int:
        sample_us = elapsed_ns // NS_PER_US  # floor -> error < 1 microsecond
        self._samples_us.append(sample_us)
        return sample_us

    def snapshot(self) -> LatencyStats:
        if not self._samples_us:
            return LatencyStats(0, 0, 0, 0, 1)
        ordered = sorted(self._samples_us)
        count = len(ordered)
        total = sum(ordered)
        mean_micro = total // count  # integer fixed-point mean (floor)
        # Nearest-rank P95: rank = ceil(0.95 * n), 1-based.
        rank = (95 * count + 99) // 100
        p95_micro = ordered[min(count, max(1, rank)) - 1]
        return LatencyStats(
            samples=count,
            mean_micro=mean_micro,
            p95_micro=p95_micro,
            max_micro=ordered[-1],
            quantization_error_bound_us=1,
        )


class LatencySample:
    def __init__(self, recorder: LatencyRecorder) -> None:
        self._recorder = recorder
        self._start_ns = 0
        self.elapsed_ns = 0
        self.elapsed_us = 0

    def __enter__(self) -> "LatencySample":
        self._start_ns = time.perf_counter_ns()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed_ns = time.perf_counter_ns() - self._start_ns
        self.elapsed_us = self._recorder.record_ns(self.elapsed_ns)


def _fixed3(micro: int) -> str:
    milli_scaled = micro * SCALE_MILLI // US_PER_MS
    whole, frac = divmod(milli_scaled, SCALE_MILLI)
    return f"{whole}.{frac:03d}"
