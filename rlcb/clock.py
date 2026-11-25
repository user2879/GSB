"""Monotonic millisecond clock.

The production clock wraps ``time.monotonic_ns`` so lease expiry is decided by
real elapsed time rather than artificial sleeps. A controllable fake clock is
provided for deterministic replay tests without weakening the real-clock path.
"""

from __future__ import annotations

import time
from typing import Protocol


class MonotonicClock(Protocol):
    def now_ms(self) -> int: ...


class SystemMonotonicClock:
    """Real monotonic clock; milliseconds since an arbitrary local epoch."""

    __slots__ = ("_base_ns",)

    def __init__(self) -> None:
        self._base_ns = time.monotonic_ns()

    def now_ms(self) -> int:
        return (time.monotonic_ns() - self._base_ns) // 1_000_000


class FakeMonotonicClock:
    """Deterministic, manually advanced clock used by replay tests."""

    __slots__ = ("_now",)

    def __init__(self, start_ms: int = 0) -> None:
        self._now = int(start_ms)

    def now_ms(self) -> int:
        return self._now

    def advance(self, delta_ms: int) -> int:
        if delta_ms < 0:
            raise ValueError("clock cannot move backwards")
        self._now += int(delta_ms)
        return self._now

    def set(self, value_ms: int) -> int:
        if value_ms < self._now:
            raise ValueError("clock cannot move backwards")
        self._now = int(value_ms)
        return self._now
