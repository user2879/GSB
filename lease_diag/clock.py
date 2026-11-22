"""Clock abstraction.

``RealClock`` wraps the local monotonic clock (``time.monotonic_ns``) so that
lease expiry and socket scheduling are driven by a genuine monotonic timer,
while ``VirtualClock`` lets tests and deterministic replay advance time
explicitly without any sleeping.
"""
from __future__ import annotations

import time


class Clock:
    def now_ms(self) -> int:
        raise NotImplementedError

    def sleep_ms(self, ms: int) -> None:
        raise NotImplementedError


class RealClock(Clock):
    """Production clock: true local monotonic time in integer milliseconds."""

    def now_ms(self) -> int:
        return time.monotonic_ns() // 1_000_000

    def sleep_ms(self, ms: int) -> None:
        if ms > 0:
            time.sleep(ms / 1000.0)


class VirtualClock(Clock):
    """Deterministic clock advanced manually; never blocks."""

    def __init__(self, start_ms: int = 0) -> None:
        self._now = int(start_ms)

    def now_ms(self) -> int:
        return self._now

    def advance(self, ms: int) -> None:
        if ms < 0:
            raise ValueError("time cannot move backwards")
        self._now += int(ms)

    def sleep_ms(self, ms: int) -> None:
        self.advance(ms)
