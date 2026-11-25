"""Monotonic-clock helpers for queue wait measurements."""

from __future__ import annotations

import time


def monotonic_ms() -> int:
    """Return a non-decreasing millisecond reading from a monotonic clock."""

    return time.monotonic_ns() // 1_000_000
