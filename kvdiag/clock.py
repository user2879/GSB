"""Local monotonic clock used for lease expiry and wait-event timing."""

import time


def mono_ms():
    """Current local monotonic clock in integer milliseconds."""
    return int(time.monotonic() * 1000)


def sleep_ms(ms):
    time.sleep(ms / 1000.0)
