"""Monotonic millisecond clocks.

The production clock wraps :func:`time.monotonic_ns` and therefore only ever
moves forward. A :class:`VirtualClock` is provided for deterministic replay and
unit tests; it exposes the same millisecond interface.
"""
from __future__ import annotations

import time
from typing import Callable


class MonotonicClock:
    """Wall monotonic clock reported in integer milliseconds."""

    def now_ms(self) -> int:
        return time.monotonic_ns() // 1_000_000

    def sleep_ms(self, ms: int) -> None:
        if ms > 0:
            time.sleep(ms / 1000.0)


class VirtualClock:
    """Deterministic clock whose time only advances by explicit request."""

    def __init__(self, start_ms: int = 0) -> None:
        self._t = int(start_ms)

    def now_ms(self) -> int:
        return self._t

    def advance(self, ms: int) -> int:
        if ms < 0:
            raise ValueError("virtual clock cannot move backwards")
        self._t += int(ms)
        return self._t

    def sleep_ms(self, ms: int) -> None:
        self.advance(ms)


ClockLike = Callable[[], int]
