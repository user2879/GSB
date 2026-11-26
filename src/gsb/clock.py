"""Monotonic clock helpers. All scheduling timeouts use real elapsed time."""

import time


def now_ms() -> int:
    """Integer monotonic milliseconds since an unspecified local epoch."""
    return int(time.monotonic_ns() // 1_000_000)


def sleep_ms(milliseconds) -> None:
    if milliseconds is None or milliseconds <= 0:
        return
    time.sleep(milliseconds / 1000.0)
