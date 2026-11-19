from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Generic, Iterable, TypeVar


T = TypeVar("T")
R = TypeVar("R")


@dataclass
class SchedulingStats:
    max_global_active: int = 0
    max_service_active: dict[str, int] = field(default_factory=dict)
    queue_wait_ms: dict[str, int] = field(default_factory=dict)


class QuotaScheduler(Generic[T, R]):
    """Deterministic admission with global and per-service concurrency budgets.

    A fixed set of global workers repeatedly selects the earliest queued request
    whose service quota is available. Requests for a saturated service stay in
    the queue; no worker can enter an upstream call while bypassing its quota.
    With a one-slot service quota, request order is fully deterministic.
    """

    def __init__(self, global_limit: int, service_limits: dict[str, int]) -> None:
        if global_limit < 1:
            raise ValueError("global_limit must be positive")
        self.global_limit = global_limit
        self.service_limits = dict(sorted(service_limits.items()))
        self._active_service = {
            service_id: 0 for service_id in self.service_limits
        }
        self.stats = SchedulingStats(
            max_service_active={service_id: 0 for service_id in self.service_limits}
        )

    def run(
        self,
        items: Iterable[T],
        service_for: Callable[[T], str | None],
        worker: Callable[[T], R],
        quota_changed: Callable[[str, int, int], None] | None = None,
        global_changed: Callable[[int, int], None] | None = None,
    ) -> tuple[R, ...]:
        item_list = list(items)
        if not item_list:
            return ()
        pending = [(index, item) for index, item in enumerate(item_list)]
        pending.sort(key=lambda pair: pair[0])
        ordered: list[R | None] = [None] * len(item_list)
        errors: list[BaseException] = []
        condition = threading.Condition()
        active_global = 0
        remaining = len(item_list)
        queued_at: dict[int, int] = {
            index: time.monotonic_ns() for index, _ in pending
        }

        def run_worker() -> None:
            nonlocal active_global, remaining
            while True:
                with condition:
                    selected = None
                    while selected is None and pending:
                        for position, (index, item) in enumerate(pending):
                            service_id = service_for(item)
                            if (
                                service_id is None
                                or self._active_service.get(service_id, 0)
                                < self.service_limits.get(service_id, 1)
                            ):
                                selected = pending.pop(position)
                                break
                        if selected is None:
                            condition.wait(timeout=0.1)
                    if selected is None:
                        return
                    index, item = selected
                    service_id = service_for(item)
                    active_global += 1
                    self.stats.max_global_active = max(
                        self.stats.max_global_active, active_global
                    )
                    if service_id is not None:
                        self._active_service[service_id] = (
                            self._active_service.get(service_id, 0) + 1
                        )
                        self.stats.max_service_active[service_id] = max(
                            self.stats.max_service_active.get(service_id, 0),
                            self._active_service[service_id],
                        )
                        wait_ms = (
                            time.monotonic_ns() - queued_at.get(index, time.monotonic_ns())
                        ) // 1_000_000
                        self.stats.queue_wait_ms[service_id] = (
                            self.stats.queue_wait_ms.get(service_id, 0) + wait_ms
                        )
                if global_changed is not None:
                    global_changed(active_global, self.global_limit)
                if service_id is not None and quota_changed is not None:
                    quota_changed(
                        service_id,
                        self._active_service[service_id],
                        self.service_limits[service_id],
                    )
                try:
                    ordered[index] = worker(item)
                except BaseException as exc:
                    with condition:
                        errors.append(exc)
                finally:
                    with condition:
                        active_global -= 1
                        if service_id is not None:
                            self._active_service[service_id] = max(
                                0, self._active_service[service_id] - 1
                            )
                        remaining -= 1
                        condition.notify_all()
                    if service_id is not None and quota_changed is not None:
                        quota_changed(
                            service_id,
                            self._active_service[service_id],
                            self.service_limits[service_id],
                        )
                    if global_changed is not None:
                        global_changed(active_global, self.global_limit)

        worker_count = min(self.global_limit, len(item_list))
        threads = [
            threading.Thread(
                target=run_worker,
                name=f"meshdiag-worker-{index}",
                daemon=True,
            )
            for index in range(worker_count)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise errors[0]
        if remaining != 0 or any(item is None for item in ordered):
            raise RuntimeError("scheduler did not complete all requests")
        return tuple(ordered)  # type: ignore[arg-type]
