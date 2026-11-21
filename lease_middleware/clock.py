"""Clock abstraction.

Expiry is always judged with the local monotonic clock (``time.monotonic``)
so it cannot be fooled by wall-clock adjustments.  A wall-clock epoch in
milliseconds is stored alongside each lease purely for human readable
persistence; liveness decisions never read it.
"""
from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Instant:
    mono: float          # monotonic seconds
    epoch_ms: int        # wall clock milliseconds, display only

    @staticmethod
    def now() -> "Instant":
        return Instant(time.monotonic(), int(time.time() * 1000))


class Clock:
    def now(self) -> Instant:
        return Instant.now()

    def monotonic(self) -> float:
        return time.monotonic()
