from __future__ import annotations

import time


def monotonic_ms() -> int:
    """Local CLOCK_MONOTONIC timestamp truncated to milliseconds."""
    return time.monotonic_ns() // 1_000_000


def elapsed_ms(start_ms: int, end_ms: int | None = None) -> int:
    end = monotonic_ms() if end_ms is None else end_ms
    return max(0, end - start_ms)
