"""Monotonic millisecond clocks.

Production uses :func:`time.monotonic_ns`; tests and replayable runs inject
:class:`DeterministicClock`.  Nodes receive the same injected schedule through
the ``EDGE_CLOCK_SCHEDULE`` environment variable (a path to a JSON file of
strictly increasing millisecond timestamps), which keeps receipt durations,
lease expiry and arbitration reproducible while the processes themselves are
still real OS processes talking over loopback TCP.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Protocol


class Clock(Protocol):
    def now_ms(self) -> int: ...


class SystemClock:
    """Real monotonic clock, millisecond resolution."""

    def now_ms(self) -> int:
        return time.monotonic_ns() // 1_000_000


class DeterministicClock:
    """Scripted clock.

    ``schedule`` is a list of millisecond values.  Every call to
    :meth:`now_ms` returns the next scheduled value, repeating the final one
    once the schedule is exhausted.  This allows a centre-side run and its
    node subprocesses to observe identical logical timelines.
    """

    def __init__(self, schedule: list[int]) -> None:
        if not schedule or any(b <= a for a, b in zip(schedule, schedule[1:])):
            raise ValueError("clock schedule must be non-empty and strictly increasing")
        self._schedule = list(schedule)
        self._idx = 0
        self._lock = threading.Lock()

    def now_ms(self) -> int:
        with self._lock:
            value = self._schedule[min(self._idx, len(self._schedule) - 1)]
            self._idx += 1
            return value

    def to_dict(self) -> dict[str, object]:
        return {"schedule": list(self._schedule), "idx": self._idx}


def node_clock() -> Clock:
    """Clock factory used inside node subprocesses."""

    path = os.environ.get("EDGE_CLOCK_SCHEDULE")
    if path:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return DeterministicClock(payload["schedule"])
    return SystemClock()
