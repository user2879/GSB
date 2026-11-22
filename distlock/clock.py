"""Monotonic clock abstraction.

The arbitration services judge lease expiry on *real* monotonic time
(:class:`MonotonicClock`). Unit tests and deterministic replay checks inject
:class:`FakeClock` so timing scenarios are fully reproducible without sleeping.
"""
from __future__ import annotations

import time
from typing import Protocol


class Clock(Protocol):
    def now_ms(self) -> int: ...

    def sleep_ms(self, ms: int) -> None: ...


class MonotonicClock:
    """Real process-local monotonic clock, integer milliseconds."""

    def now_ms(self) -> int:
        return time.monotonic_ns() // 1_000_000

    def sleep_ms(self, ms: int) -> None:
        if ms > 0:
            time.sleep(ms / 1000.0)


class FakeClock:
    """Deterministic manually-advanced clock (tests / offline replay)."""

    def __init__(self, start_ms: int = 0):
        self._t = int(start_ms)
        self._sleeps: list[int] = []

    def now_ms(self) -> int:
        return self._t

    def sleep_ms(self, ms: int) -> None:
        self._sleeps.append(int(ms))
        self._t += max(0, int(ms))

    def advance(self, ms: int) -> None:
        self._t += int(ms)

    @property
    def sleeps(self) -> list[int]:
        return self._sleeps
