"""Monotonic millisecond clock.

All scheduling, lease and queue measurements use time.monotonic_ns so results
are immune to wall-clock adjustments. Wall clock is only used for human
readable log timestamps.
"""

import time


def now_ms() -> int:
    """Current monotonic time in whole milliseconds."""
    return time.monotonic_ns() // 1_000_000


def wall_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
