"""Monotonic millisecond clock.

All expiry and latency decisions use :func:`monotonic_ms`, which wraps
``time.monotonic_ns`` truncated to integer milliseconds. Real wall-clock
sleeping is never used to emulate expiry: lease validity is a pure comparison
of monotonic timestamps, so a short-lived request stream can observe genuine
expiry without artificial delays inside the accounting logic.
"""
from __future__ import annotations

import time


def monotonic_ms() -> int:
    """Return a monotonically increasing integer millisecond timestamp."""
    return time.monotonic_ns() // 1_000_000


class Clock:
    """Thin clock wrapper that can be replaced in tests with a manual one."""

    def now_ms(self) -> int:
        return monotonic_ms()


class ManualClock(Clock):
    """Deterministic clock advanced explicitly (never sleeps)."""

    def __init__(self, start_ms: int = 1_000_000) -> None:
        self._now = int(start_ms)

    def now_ms(self) -> int:
        return self._now

    def advance(self, ms: int) -> int:
        if ms < 0:
            raise ValueError("clock may only move forward")
        self._now += int(ms)
        return self._now


__all__ = ["monotonic_ms", "Clock", "ManualClock"]
