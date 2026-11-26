from __future__ import annotations

import time


def monotonic_ns() -> int:
    return time.monotonic_ns()


def monotonic_ms() -> int:
    """Floor the Linux monotonic clock to integer milliseconds."""

    return monotonic_ns() // 1_000_000


def elapsed_ms(start_ns: int, end_ns: int | None = None) -> int:
    end = monotonic_ns() if end_ns is None else end_ns
    return (end - start_ns) // 1_000_000
