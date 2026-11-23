"""Monotonic millisecond clock; injectable for deterministic tests."""

import threading
import time


def monotonic_ms() -> int:
    return time.monotonic_ns() // 1_000_000


class ManualClock:
    """Test clock; time only advances when told to."""

    def __init__(self, start_ms: int = 0):
        self._now = start_ms
        self._lock = threading.Lock()

    def __call__(self) -> int:
        with self._lock:
            return self._now

    def advance(self, ms: int) -> int:
        if ms < 0:
            raise ValueError("cannot move clock backwards")
        with self._lock:
            self._now += ms
            return self._now
