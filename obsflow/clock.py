"""Monotonic clock abstraction.

Real deployments use :func:`time.monotonic_ns`. Tests inject a deterministic
:class:`ManualClock` so that lease expiry is driven by *real* timing
semantics without any artificial sleeps in the production code paths.
"""
from __future__ import annotations

import time
from typing import Protocol


class MonotonicClock(Protocol):
    def now_ms(self) -> int:
        ...


class SystemClock:
    """Wall production clock backed by ``time.monotonic_ns``."""

    __slots__ = ()

    def now_ms(self) -> int:
        return time.monotonic_ns() // 1_000_000


class ManualClock:
    """Deterministic clock for tests; only moves when explicitly advanced."""

    __slots__ = ("_t",)

    def __init__(self, start_ms: int = 0):
        self._t = int(start_ms)

    def now_ms(self) -> int:
        return self._t

    def advance(self, delta_ms: int) -> int:
        if delta_ms < 0:
            raise ValueError("clock can only advance forward")
        self._t += int(delta_ms)
        return self._t
