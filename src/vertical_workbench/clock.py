from __future__ import annotations

import time
from dataclasses import dataclass


class Clock:
    def now_ms(self) -> int:
        raise NotImplementedError


class MonotonicClock(Clock):
    """Real monotonic clock; never uses wall-clock jumps or sleep as reclamation."""

    def now_ms(self) -> int:
        return time.monotonic_ns() // 1_000_000


@dataclass
class ManualClock(Clock):
    current_ms: int = 1_000_000

    def now_ms(self) -> int:
        return self.current_ms

    def advance(self, milliseconds: int) -> int:
        if milliseconds < 0:
            raise ValueError("clock cannot move backwards")
        self.current_ms += milliseconds
        return self.current_ms
