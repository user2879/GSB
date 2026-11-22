"""Configurable monotonic clocks. Wall time is never used for lease expiry."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from .errors import AppError, E_CONFIG_INVALID


CLOCK_SOURCES: dict[str, Callable[[], int]] = {
    "monotonic_ns": time.monotonic_ns,
    "monotonic": time.monotonic_ns,
    "process_monotonic": time.monotonic_ns,
    "perf_counter_ns": time.perf_counter_ns,
    "performance_counter": time.perf_counter_ns,
}


@dataclass(frozen=True)
class MonotonicClock:
    source: str = "monotonic_ns"

    def __post_init__(self) -> None:
        if self.source not in CLOCK_SOURCES:
            raise AppError(
                E_CONFIG_INVALID,
                f"unsupported monotonic clock source: {self.source}",
                stage="config",
                details={"clock_source": self.source, "allowed": sorted(CLOCK_SOURCES)},
            )

    def now_ns(self) -> int:
        return int(CLOCK_SOURCES[self.source]())

    def seconds_until(self, deadline_ns: int) -> float:
        return max(0.0, (deadline_ns - self.now_ns()) / 1_000_000_000)
