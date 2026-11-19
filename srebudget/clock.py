"""Clock abstractions.

Production runs use ``time.monotonic_ns``.  Tests and deterministic example
scripts can use ``ManualClock`` or ``LogicalClock``; the state machine itself
never reads wall time directly.
"""

from __future__ import annotations

import time


class MonotonicClock:
    def now_ms(self) -> int:
        return time.monotonic_ns() // 1_000_000


class ManualClock:
    def __init__(self, start_ms: int = 0) -> None:
        self._now_ms = int(start_ms)

    def now_ms(self) -> int:
        return self._now_ms

    def set(self, value_ms: int) -> None:
        self._now_ms = int(value_ms)

    def advance(self, delta_ms: int) -> None:
        self._now_ms += int(delta_ms)


class LogicalClock(ManualClock):
    """Monotonic logical clock driven by trace timestamps during replay."""

    def follow(self, timestamp_ms: int) -> None:
        self.set(max(self.now_ms(), int(timestamp_ms)))
