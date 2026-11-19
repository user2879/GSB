"""Cancellation and deterministic in-flight spectrum frame budget."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from .errors import BusinessError


class CancelToken:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._reason = ""
        self._lock = threading.Lock()

    def cancel(self, reason: str) -> None:
        with self._lock:
            if not self._event.is_set():
                self._reason = reason
                self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        return self._reason

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise BusinessError("E_INTERRUPTED", self._reason or "job interrupted")


@dataclass(slots=True)
class BudgetStats:
    capacity_frames: int
    peak_in_flight_frames: int = 0
    total_waits: int = 0
    acquisitions: int = 0


class FrameBudget:
    """Global fair budget for transient spectrum rows.

    Waiters are ordered by ``priority`` and then by arrival. The lower-priority
    number follows the track declaration order, so an exhausted budget blocks
    later tracks instead of silently reducing frame/band precision.
    """

    def __init__(self, capacity_frames: int) -> None:
        if capacity_frames < 1:
            raise ValueError("capacity_frames must be positive")
        self.stats = BudgetStats(capacity_frames=capacity_frames)
        self._capacity = capacity_frames
        self._used = 0
        self._condition = threading.Condition()
        self._sequence = 0
        self._waiters: list[dict[str, object]] = []

    @property
    def capacity_frames(self) -> int:
        return self._capacity

    def acquire(self, frames: int, priority: int, token: CancelToken) -> None:
        if frames < 0 or frames > self._capacity:
            raise ValueError("requested frame count is outside budget capacity")
        with self._condition:
            self._sequence += 1
            waiter: dict[str, object] = {
                "priority": priority,
                "sequence": self._sequence,
                "frames": frames,
                "done": False,
            }
            self._waiters.append(waiter)
            waited = False
            try:
                while True:
                    token.raise_if_cancelled()
                    self._waiters.sort(key=lambda w: (int(w["priority"]), int(w["sequence"])))
                    head = self._waiters[0]
                    if head is waiter and self._used + frames <= self._capacity:
                        self._waiters.pop(0)
                        self._used += frames
                        self.stats.acquisitions += 1
                        self.stats.peak_in_flight_frames = max(self.stats.peak_in_flight_frames, self._used)
                        if waited:
                            self.stats.total_waits += 1
                        return
                    waited = True
                    self._condition.wait(timeout=0.05)
            except BaseException:
                if waiter in self._waiters:
                    self._waiters.remove(waiter)
                self._condition.notify_all()
                raise

    def release(self, frames: int) -> None:
        with self._condition:
            self._used -= frames
            if self._used < 0:
                raise RuntimeError("spectrum budget released more frames than acquired")
            self._condition.notify_all()
