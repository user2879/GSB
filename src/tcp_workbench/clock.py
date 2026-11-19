"""Monotonic millisecond clocks.

Production uses the host's monotonic counter.  Tests and deterministic sample
replays use a manual counter which still represents local monotonic
milliseconds; it never uses wall-clock time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


def monotonic_ms() -> int:
    return time.monotonic_ns() // 1_000_000


@dataclass
class ManualClock:
    value_ms: int = 0

    def now_ms(self) -> int:
        return self.value_ms

    def set_ms(self, value: int) -> int:
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < self.value_ms
        ):
            raise ValueError(
                "manual clock must receive a non-decreasing integer millisecond value"
            )
        self.value_ms = value
        return value

    def advance_ms(self, delta: int) -> int:
        if not isinstance(delta, int) or isinstance(delta, bool) or delta < 0:
            raise ValueError("manual clock delta must be a non-negative integer")
        self.value_ms += delta
        return self.value_ms
