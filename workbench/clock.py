"""Monotonic clock helpers. The declared clock base is time.monotonic_ns.

All queue/decide/forward interval stamps and GUI/recompute latency
observations use integer milliseconds derived from this clock.
"""

import time

CLOCK_BASE = "time.monotonic_ns//1_000_000"


def monotonic_ms():
    return time.monotonic_ns() // 1_000_000
