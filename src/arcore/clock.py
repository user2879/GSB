"""Monotonic clock abstraction.

All lease timing uses a monotonic source so it is immune to wall clock
adjustments. A controllable fake clock is provided for deterministic tests.
"""

import time


class MonotonicClock:
    """Millisecond monotonic clock backed by :func:`time.monotonic_ns`."""

    def now_ms(self) -> int:
        return time.monotonic_ns() // 1_000_000


class FakeClock:
    """Deterministic clock advanced explicitly (test only)."""

    def __init__(self, start_ms: int = 0):
        self._now = int(start_ms)

    def now_ms(self) -> int:
        return self._now

    def advance(self, delta_ms: int) -> int:
        if delta_ms < 0:
            raise ValueError("clock can only move forward")
        self._now += int(delta_ms)
        return self._now
