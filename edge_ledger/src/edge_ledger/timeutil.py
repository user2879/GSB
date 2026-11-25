from __future__ import annotations

import time
from dataclasses import dataclass


def monotonic_ms() -> int:
    """Return the real local monotonic clock in whole milliseconds."""
    return time.monotonic_ns() // 1_000_000


@dataclass(frozen=True)
class MonotonicClock:
    """Small injectable clock wrapper; production tests use the real clock."""

    def now_ms(self) -> int:
        return monotonic_ms()
