from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class BudgetSnapshot:
    global_limit: int
    global_active: int
    global_peak: int
    services: dict[str, dict[str, int]]


class BudgetReporter:
    """Coalesces occupancy updates onto a daemon writer thread.

    Permit acquisition and release must never block on SQLite I/O. The latest
    snapshot is coalesced and persisted promptly by a separate thread.
    """

    def __init__(self, callback: Callable[[BudgetSnapshot], None]) -> None:
        self._callback = callback
        self._condition = threading.Condition()
        self._latest: BudgetSnapshot | None = None
        self._stop = False
        self._thread = threading.Thread(
            target=self._run,
            name="mesh-budget-reporter",
            daemon=True,
        )
        self._thread.start()

    def notify(self, snapshot: BudgetSnapshot) -> None:
        with self._condition:
            self._latest = snapshot
            self._condition.notify()

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._latest is None and not self._stop:
                    self._condition.wait()
                if self._stop and self._latest is None:
                    return
                snapshot = self._latest
                self._latest = None
            if snapshot is not None:
                try:
                    self._callback(snapshot)
                except Exception:
                    pass
            time.sleep(0.005)

    def close(self, timeout: float = 2.0) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        self._thread.join(timeout=timeout)


class ConcurrencyGate:
    """Atomically acquires a global and a per-service concurrency permit."""

    def __init__(
        self,
        global_limit: int,
        service_limits: dict[str, int],
        on_change: Callable[[BudgetSnapshot], None] | None = None,
    ) -> None:
        self.global_limit = global_limit
        self.service_limits = dict(service_limits)
        self._global_active = 0
        self._global_peak = 0
        self._service_active = {name: 0 for name in self.service_limits}
        self._service_peak = {name: 0 for name in self.service_limits}
        self._queue_waiters = 0
        self._condition = threading.Condition()
        self._reporter = BudgetReporter(on_change) if on_change else None

    def acquire(self, service: str) -> float:
        if service not in self.service_limits:
            raise KeyError(service)
        with self._condition:
            self._queue_waiters += 1
            try:
                while (
                    self._global_active >= self.global_limit
                    or self._service_active[service] >= self.service_limits[service]
                ):
                    self._condition.wait()
                self._global_active += 1
                self._service_active[service] += 1
                self._global_peak = max(self._global_peak, self._global_active)
                self._service_peak[service] = max(
                    self._service_peak[service], self._service_active[service]
                )
                acquired_at = time.monotonic_ns() / 1_000_000.0
            finally:
                self._queue_waiters -= 1
        self._emit()
        return acquired_at

    def release(self, service: str) -> float:
        with self._condition:
            if self._global_active <= 0 or self._service_active[service] <= 0:
                raise RuntimeError("release without a matching acquire")
            self._global_active -= 1
            self._service_active[service] -= 1
            released_at = time.monotonic_ns() / 1_000_000.0
            self._condition.notify_all()
        self._emit()
        return released_at

    def snapshot(self) -> BudgetSnapshot:
        with self._condition:
            return BudgetSnapshot(
                global_limit=self.global_limit,
                global_active=self._global_active,
                global_peak=self._global_peak,
                services={
                    name: {
                        "limit": self.service_limits[name],
                        "active": self._service_active[name],
                        "peak": self._service_peak[name],
                    }
                    for name in sorted(self.service_limits)
                },
            )

    @property
    def queued(self) -> int:
        with self._condition:
            return self._queue_waiters

    def _emit(self) -> None:
        if self._reporter is None:
            return
        self._reporter.notify(self.snapshot())

    def close(self) -> None:
        if self._reporter is not None:
            self._reporter.close()
