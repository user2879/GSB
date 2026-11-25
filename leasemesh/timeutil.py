from __future__ import annotations

import time


def monotonic_ms() -> float:
    """Monotonic clock in fractional milliseconds."""
    return time.monotonic_ns() / 1_000_000.0


def elapsed_ms(start_ms: float) -> float:
    return max(0.0, monotonic_ms() - start_ms)


def iso_stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())
