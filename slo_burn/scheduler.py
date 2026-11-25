"""窗口评估调度器：真实工作线程池 + 租约配额 + 排队计时。"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from .arbitration import OwnershipArbiter, OwnershipEvent, RecycleReason
from .evaluator import now_ms
from .lease_client import LeaseClient


@dataclass
class ScheduledTask:
    task_id: str
    service_id: str
    minute: int
    work: Callable[[dict], object]
    ttl_ms: int = 400


@dataclass
class TaskRecord:
    task_id: str
    service_id: str
    minute: int
    queued_ms: int
    started_ms: int | None = None
    finished_ms: int | None = None
    lease_id: str | None = None
    slot_id: int | None = None
    version: int | None = None
    status: str = "queued"
    result: object = None
    error: str | None = None

    @property
    def wait_ms(self) -> int | None:
        if self.started_ms is None:
            return None
        return self.started_ms - self.queued_ms


@dataclass
class SchedulerReport:
    records: list[TaskRecord] = field(default_factory=list)
    adjudications: list = field(default_factory=list)
    max_concurrent_observed: int = 0
    global_limit: int = 0
    per_service_limit: int = 0


class WindowScheduler:
    """在 ThreadPoolExecutor 上真实投射窗口评估任务。

    每个任务先向评估器子进程申请租约（双重上限由权威强制），拿到槽位后
    才在线程中执行；排队等待起止毫秒写入 TaskRecord。
    """

    def __init__(
        self,
        client: LeaseClient,
        global_limit: int,
        per_service_limit: int,
    ) -> None:
        self.client = client
        self.global_limit = global_limit
        self.per_service_limit = per_service_limit
        self._arbiter = OwnershipArbiter()
        self._events_lock = threading.Lock()
        self._pending_events: list[OwnershipEvent] = []
        self._source_counter = 0
        self._active = 0
        self._active_lock = threading.Lock()
        self._max_concurrent = 0

    def _record_event(self, event: OwnershipEvent) -> None:
        with self._events_lock:
            self._pending_events.append(event)

    def _next_source(self) -> int:
        with self._events_lock:
            self._source_counter += 1
            return self._source_counter

    def run(self, tasks: list[ScheduledTask]) -> SchedulerReport:
        records: dict[str, TaskRecord] = {
            task.task_id: TaskRecord(
                task_id=task.task_id,
                service_id=task.service_id,
                minute=task.minute,
                queued_ms=now_ms(),
            )
            for task in tasks
        }
        pool = ThreadPoolExecutor(max_workers=self.global_limit)
        try:
            futures = []
            for task in tasks:
                futures.append(pool.submit(self._run_one, task, records[task.task_id]))
            for future in futures:
                future.result()
        finally:
            pool.shutdown(wait=True)
        decisions = self._arbiter.adjudicate_all(self._pending_events)
        return SchedulerReport(
            records=[records[task.task_id] for task in tasks],
            adjudications=decisions,
            max_concurrent_observed=self._max_concurrent,
            global_limit=self.global_limit,
            per_service_limit=self.per_service_limit,
        )

    def _grant_with_wait(self, task: ScheduledTask) -> dict:
        """在全局/每服务上限下排队：被权威拒绝则退避重试，直到拿到租约。"""
        attempt = 0
        while True:
            try:
                return self.client.grant(task.service_id, task.ttl_ms)
            except Exception as exc:  # noqa: BLE001 - 配额类失败即排队等待
                if getattr(exc, "code", None) != "E_LEASE":
                    raise
                attempt += 1
                if attempt > 200:
                    raise
                # 先触发到期回收，尽快释放僵尸槽位
                try:
                    self.client.reap()
                except Exception:  # noqa: BLE001
                    pass
                time.sleep(0.02)

    def _run_one(self, task: ScheduledTask, record: TaskRecord) -> None:
        grant = self._grant_with_wait(task)
        lease = grant["lease"]
        record.started_ms = now_ms()
        record.lease_id = lease["lease_id"]
        record.slot_id = lease["slot_id"]
        record.version = lease["version"]
        record.status = "running"
        self._record_event(
            OwnershipEvent(
                observed_ms=lease["granted_ms"],
                service_id=task.service_id,
                slot_id=lease["slot_id"],
                lease_id=lease["lease_id"],
                version=lease["version"],
                reason=RecycleReason.GRANT,
                source_order=self._next_source(),
            )
        )
        with self._active_lock:
            self._active += 1
            self._max_concurrent = max(self._max_concurrent, self._active)
        try:
            record.result = task.work(lease)
            record.status = "completed"
            response = self.client.revoke(
                lease["lease_id"], lease["version"]
            )
            self._record_event(
                OwnershipEvent(
                    observed_ms=response["at_ms"],
                    service_id=task.service_id,
                    slot_id=lease["slot_id"],
                    lease_id=lease["lease_id"],
                    version=lease["version"],
                    reason=RecycleReason.REVOKE,
                    source_order=self._next_source(),
                )
            )
        except Exception as exc:  # noqa: BLE001 - 记录后向上由future抛出策略决定
            record.status = "failed"
            record.error = str(exc)
            raise
        finally:
            record.finished_ms = now_ms()
            with self._active_lock:
                self._active -= 1
