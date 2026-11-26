"""Monotonic millisecond clock used for leases, waits and elapsed time."""

from __future__ import annotations

import time


def now_ms() -> int:
    """Return monotonic time in whole milliseconds (integer fixed point)."""
    return int(time.monotonic() * 1000.0)
