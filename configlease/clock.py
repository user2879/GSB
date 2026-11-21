"""Monotonic clock helpers.

Lease expiry is measured with the platform wide monotonic clock so it cannot
move backwards when the wall clock changes.  The same reference is used by the
nodes (same machine, same Windows ``QueryPerformanceCounter`` basis) and the
unit is milliseconds, matching the wire protocol.
"""
from __future__ import annotations

import time


def now_ms() -> int:
    """Return monotonic milliseconds since an unspecified but stable epoch."""
    return time.monotonic_ns() // 1_000_000


def deadline_from_ttl(ttl_ms: int) -> int:
    return now_ms() + int(ttl_ms)


def remaining_ms(deadline_ms: int) -> int:
    return int(deadline_ms) - now_ms()


def is_expired(deadline_ms: int) -> bool:
    return now_ms() >= int(deadline_ms)
