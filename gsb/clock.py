from __future__ import annotations

import time


def monotonic_ms() -> int:
    """Return a strictly local monotonic timestamp in whole milliseconds."""

    return time.monotonic_ns() // 1_000_000


def monotonic_ms_precise() -> float:
    return time.monotonic_ns() / 1_000_000.0


def elapsed_ms(start_ns: int, end_ns: int | None = None) -> int:
    end = time.perf_counter_ns() if end_ns is None else end_ns
    return max(0, (end - start_ns) // 1_000_000)
