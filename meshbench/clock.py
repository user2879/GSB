"""Clock sources.

Circuit-breaker timing always uses the real local monotonic clock
(:func:`mono_ms`), as required.  Window assignment uses a pluggable
:class:`Clock` that defaults to real wall/monotonic time but can be replaced by
a :class:`VirtualClock` so deterministic demos can span multiple windows.
"""

import time


def mono_ms() -> int:
    """Real monotonic clock in integer milliseconds (never goes backwards)."""
    return int(time.monotonic() * 1000.0)


def wall_ms() -> int:
    """Real wall-clock milliseconds since the Unix epoch."""
    return int(time.time() * 1000.0)


class Clock:
    """Default clock: wall time for windows, monotonic for elapsed timing."""

    def now_ms(self) -> int:
        return wall_ms()

    def elapsed_ms(self) -> int:
        return mono_ms()


class VirtualClock(Clock):
    """Deterministic, manually advanced clock for repeatable demos/tests."""

    def __init__(self, start_ms: int = 1_000_000):
        self._virtual = int(start_ms)
        self._mono_origin = mono_ms()
        self._virtual_origin = self._virtual

    def advance(self, ms: int) -> int:
        self._virtual += int(ms)
        return self._virtual

    def now_ms(self) -> int:
        return self._virtual

    def elapsed_ms(self) -> int:
        """Mirror virtual progress on a monotonic basis."""
        return self._mono_origin + (self._virtual - self._virtual_origin)
