from __future__ import annotations

import time
from typing import Callable


class MonotonicClock:
    """Production clock. Milliseconds come from CLOCK_MONOTONIC."""

    def now_ms(self) -> int:
        return time.monotonic_ns() // 1_000_000


class ScriptedClock:
    """Deterministic clock for field-for-field rebuild tests.

    Time only moves when a test explicitly calls :meth:`advance`; this keeps
    admission and commit fields reproducible across blank rebuilds.
    """

    def __init__(self, start_ms: int = 1_700_000_000_000, step_ms: int = 0) -> None:
        self.value = start_ms
        self.step_ms = step_ms

    def now_ms(self) -> int:
        value = self.value
        if self.step_ms:
            self.value += self.step_ms
        return value

    def advance(self, ms: int) -> None:
        self.value += ms


Clock = MonotonicClock | ScriptedClock | Callable[[], int]


def clock_ms(clock: object) -> int:
    if callable(clock):
        return int(clock())
    return int(clock.now_ms())  # type: ignore[attr-defined]
