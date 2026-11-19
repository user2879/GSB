"""Monotonic clock abstraction used by TTL and cache age calculations."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


class Clock:
    @property
    def source(self) -> str:
        raise NotImplementedError

    def monotonic_ns(self) -> int:
        raise NotImplementedError


class SystemMonotonicClock(Clock):
    @property
    def source(self) -> str:
        return "time.monotonic_ns()"

    def monotonic_ns(self) -> int:
        return time.monotonic_ns()


@dataclass
class FakeClock(Clock):
    """Deterministic clock used by tests and the repeatable demonstration."""

    now_ns: int = 1_000_000_000_000

    @property
    def source(self) -> str:
        return "dns_local.clock.FakeClock"

    def monotonic_ns(self) -> int:
        return self.now_ns

    def advance(self, seconds: float) -> int:
        if seconds < 0:
            raise ValueError("FakeClock cannot move backwards")
        self.now_ns += int(seconds * 1_000_000_000)
        return self.now_ns


class ThreadSafeClock(Clock):
    """Wraps a clock whose value can be advanced by a control thread."""

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._lock = threading.Lock()

    @property
    def source(self) -> str:
        return self._clock.source

    def monotonic_ns(self) -> int:
        with self._lock:
            return self._clock.monotonic_ns()

    def advance(self, seconds: float) -> int:
        if not isinstance(self._clock, FakeClock):
            raise TypeError("advance() is only available for FakeClock")
        with self._lock:
            return self._clock.advance(seconds)

