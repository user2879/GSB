"""Monotonic clock abstraction.

The scheduler times leases and queue waits exclusively through this
interface. Production uses the OS monotonic clock; tests inject a manual
clock so expiry boundaries are examined without real sleeping.
"""

import time
from typing import Protocol


class Clock(Protocol):
    def now_ms(self) -> int:
        ...


class MonotonicClock:
    """Millisecond view of :func:`time.monotonic_ns`."""

    def now_ms(self) -> int:
        return time.monotonic_ns() // 1_000_000


class ManualClock:
    """Deterministic clock whose time only advances when told to."""

    def __init__(self, start_ms: int = 0):
        self._now = int(start_ms)

    def now_ms(self) -> int:
        return self._now

    def advance(self, delta_ms: int) -> int:
        if delta_ms < 0:
            raise ValueError("clock cannot move backwards")
        self._now += int(delta_ms)
        return self._now
