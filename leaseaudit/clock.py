from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol


class Clock(Protocol):
    def now_ms(self) -> int:
        ...

    def sleep_until_ms(self, target_ms: int) -> None:
        ...


@dataclass
class RealMonotonicClock:
    """Milliseconds from the local monotonic clock (never wall-clock time)."""

    def now_ms(self) -> int:
        return time.monotonic_ns() // 1_000_000

    def sleep_until_ms(self, target_ms: int) -> None:
        while True:
            remaining = target_ms - self.now_ms()
            if remaining <= 0:
                return
            time.sleep(min(remaining / 1000.0, 0.05))


@dataclass
class ManualClock:
    """Explicit monotonic clock used by deterministic unit tests."""

    value_ms: int = 1_000_000
    sleep_calls: list[int] = field(default_factory=list)

    def now_ms(self) -> int:
        return self.value_ms

    def advance(self, milliseconds: int) -> int:
        if milliseconds < 0:
            raise ValueError("monotonic clock cannot move backwards")
        self.value_ms += milliseconds
        return self.value_ms

    def sleep_until_ms(self, target_ms: int) -> None:
        if target_ms < self.value_ms:
            raise ValueError("monotonic clock cannot move backwards")
        self.sleep_calls.append(target_ms)
        self.value_ms = target_ms
