from __future__ import annotations

import time


def monotonic_ms() -> int:
    """Return CLOCK_MONOTONIC milliseconds.

    The integer division is truncation toward zero; monotonic values are
    nonnegative in practice for process-local durations.
    """

    return time.monotonic_ns() // 1_000_000


def monotonic_ns() -> int:
    return time.monotonic_ns()


def elapsed_ms(start_ns: int) -> int:
    return (time.monotonic_ns() - start_ns) // 1_000_000
