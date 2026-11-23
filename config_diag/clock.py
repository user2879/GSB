from __future__ import annotations

import time
from dataclasses import dataclass


class MonotonicClock:
    def now_ns(self) -> int:
        return time.monotonic_ns()


@dataclass
class FrozenClock:
    value_ns: int

    def now_ns(self) -> int:
        return self.value_ns


@dataclass
class ManualClock:
    value_ns: int

    def now_ns(self) -> int:
        return self.value_ns

    def advance_ms(self, milliseconds: int) -> int:
        self.value_ns += milliseconds * 1_000_000
        return self.value_ns
