from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class BudgetSlot:
    service: str
    wait_start_mono_ms: int
    wait_end_mono_ms: int
    global_inflight: int
    service_inflight: int

    @property
    def wait_ms(self) -> int:
        return max(0, self.wait_end_mono_ms - self.wait_start_mono_ms)


class _Waiter:
    __slots__ = ("service", "priority", "wait_start_mono_ms", "granted", "granted_mono_ms")

    def __init__(self, service: str, priority: tuple[int, int], wait_start_mono_ms: int) -> None:
        self.service = service
        self.priority = priority
        self.wait_start_mono_ms = wait_start_mono_ms
        self.granted = False
        self.granted_mono_ms = 0


class BudgetScheduler:
    """Dual-budget scheduler with explicit wait timestamps.

    A granted slot represents an attempt that counts against both the global
    limit and the owning service's quota. Backoff and retry sleeps happen before
    calling :meth:`acquire`, so they never hold either budget.
    """

    def __init__(self, global_limit: int, service_quotas: dict[str, int]) -> None:
        if global_limit < 1:
            raise ValueError("global_limit must be positive")
        self.global_limit = global_limit
        self.service_quotas = dict(sorted(service_quotas.items()))
        self._condition = threading.Condition()
        self._waiters: list[_Waiter] = []
        self._global_inflight = 0
        self._service_inflight = {service: 0 for service in self.service_quotas}
        self._grants_global = 0
        self._grants_service = {service: 0 for service in self.service_quotas}
        self._peak_global = 0
        self._peak_service = {service: 0 for service in self.service_quotas}

    def _mono_ms(self) -> int:
        return time.monotonic_ns() // 1_000_000

    def _dispatch_locked(self) -> None:
        granted_any = False
        remaining: list[_Waiter] = []
        for waiter in sorted(self._waiters, key=lambda item: item.priority):
            service_inflight = self._service_inflight.get(waiter.service)
            quota = self.service_quotas.get(waiter.service)
            if (
                self._global_inflight < self.global_limit
                and service_inflight is not None
                and quota is not None
                and service_inflight < quota
            ):
                self._global_inflight += 1
                self._service_inflight[waiter.service] += 1
                self._grants_global += 1
                self._grants_service[waiter.service] += 1
                self._peak_global = max(self._peak_global, self._global_inflight)
                self._peak_service[waiter.service] = max(
                    self._peak_service[waiter.service],
                    self._service_inflight[waiter.service],
                )
                waiter.granted = True
                waiter.granted_mono_ms = self._mono_ms()
                granted_any = True
            else:
                remaining.append(waiter)
        self._waiters = remaining
        if granted_any:
            self._condition.notify_all()

    def acquire(self, service: str, priority: tuple[int, int]) -> BudgetSlot:
        wait_start = self._mono_ms()
        waiter = _Waiter(service, priority, wait_start)
        with self._condition:
            if service not in self.service_quotas:
                raise KeyError(f"unknown service quota for {service}")
            self._waiters.append(waiter)
            self._dispatch_locked()
            while not waiter.granted:
                if waiter not in self._waiters:
                    raise RuntimeError("budget waiter lost")
                self._condition.wait(timeout=0.05)
            wait_end = waiter.granted_mono_ms
            return BudgetSlot(
                service,
                wait_start,
                wait_end,
                self._global_inflight,
                self._service_inflight[service],
            )

    def release(self, slot: BudgetSlot) -> None:
        with self._condition:
            self._global_inflight -= 1
            self._service_inflight[slot.service] -= 1
            if self._global_inflight < 0 or self._service_inflight[slot.service] < 0:
                raise RuntimeError("budget released more than acquired")
            self._dispatch_locked()

    def snapshot(self) -> dict[str, object]:
        with self._condition:
            return {
                "global_limit": self.global_limit,
                "global_inflight": self._global_inflight,
                "global_grants": self._grants_global,
                "peak_global": self._peak_global,
                "service_quotas": dict(self.service_quotas),
                "service_inflight": dict(self._service_inflight),
                "service_grants": dict(self._grants_service),
                "peak_service": dict(self._peak_service),
                "waiting": len(self._waiters),
            }

    def set_usage(self, global_grants: int, service_grants: dict[str, int]) -> None:
        with self._condition:
            self._grants_global = max(self._grants_global, int(global_grants))
            for service in self.service_quotas:
                self._grants_service[service] = max(
                    self._grants_service[service], int(service_grants.get(service, 0))
                )
