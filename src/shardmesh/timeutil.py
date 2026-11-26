"""Monotonic clock helpers.

Lease expiry always uses :func:`time.monotonic_ns`; wall clocks are never used
for scheduling decisions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


def now_ns() -> int:
    return time.monotonic_ns()


def now_ms() -> int:
    return now_ns() // 1_000_000


def elapsed_ms(start_ns: int) -> int:
    return max(0, (now_ns() - start_ns) // 1_000_000)


@dataclass(frozen=True)
class MonotonicStamp:
    ms: int
    ns: int


def stamp() -> MonotonicStamp:
    ns = now_ns()
    return MonotonicStamp(ms=ns // 1_000_000, ns=ns)
