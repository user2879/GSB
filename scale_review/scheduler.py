from __future__ import annotations

import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any

from .canonical import monotonic_ms
from .errors import ScaleReviewError
from .leases import Lease, LeaseManager
from .spec import LeasePolicy


@dataclass(frozen=True)
class ScaleTask:
    task_id: str
    round_id: str
    batch_id: str
    minute: int
    service_id: str
    chunk_index: int
    replica_count: int

    @property
    def priority_key(self) -> tuple[int, str, int, str]:
        return (self.minute, self.service_id, self.chunk_index, self.task_id)


def chunks_for_allocation(
    *,
    round_id: str,
    batch_id: str,
    minute: int,
    service_id: str,
    allocated_replicas: int,
    max_task_replicas: int,
) -> list[ScaleTask]:
    tasks: list[ScaleTask] = []
    start = 0
    chunk_index = 0
    while start < allocated_replicas:
        count = min(max_task_replicas, allocated_replicas - start)
        tasks.append(
            ScaleTask(
                task_id=f"{round_id}:m{minute}:{service_id}:chunk{chunk_index}",
                round_id=round_id,
                batch_id=batch_id,
                minute=minute,
                service_id=service_id,
                chunk_index=chunk_index,
                replica_count=count,
            )
        )
        start += count
        chunk_index += 1
    return tasks


class ParallelScalingScheduler:
    def __init__(self, lease_manager: LeaseManager, policy: LeasePolicy) -> None:
        self.lease_manager = lease_manager
        self.policy = policy
        self.wait_spans: list[dict[str, Any]] = []
        self.execution_results: list[dict[str, Any]] = []
        self.max_observed_inflight = 0

    def run_tasks(self, tasks: list[ScaleTask]) -> list[dict[str, Any]]:
        for task in tasks:
            if task.replica_count > self.policy.max_task_replicas:
                raise ScaleReviewError(
                    "task replica quota exceeded",
                    field="replica_count",
                    details={
                        "task_id": task.task_id,
                        "replica_count": task.replica_count,
                        "max_task_replicas": self.policy.max_task_replicas,
                    },
                )
            if task.replica_count > self.lease_manager.service_quotas[task.service_id]:
                raise ScaleReviewError(
                    "task exceeds service concurrency quota and can never be scheduled",
                    field="replica_count",
                    details={"task_id": task.task_id, "service_id": task.service_id},
                )
            if task.replica_count > self.lease_manager.global_replica_budget:
                raise ScaleReviewError(
                    "task exceeds global replica budget and can never be scheduled",
                    field="replica_count",
                    details={"task_id": task.task_id},
                )

        ordered = sorted(tasks, key=lambda task: task.priority_key)
        pending = list(ordered)
        queued_at: dict[str, int] = {
            task.task_id: monotonic_ms() for task in ordered
        }
        block_reason: dict[str, str] = {task.task_id: "immediate" for task in ordered}
        spans: dict[str, dict[str, Any]] = {}
        results: list[dict[str, Any]] = []

        with ThreadPoolExecutor(
            max_workers=self.policy.max_inflight_tasks,
            thread_name_prefix="scale-executor",
        ) as executor:
            active: dict[Any, ScaleTask] = {}
            active_leases: dict[Any, Lease] = {}
            while pending or active:
                while pending and len(active) < self.policy.max_inflight_tasks:
                    task = pending[0]
                    if len(active) >= self.policy.max_inflight_tasks:
                        block_reason[task.task_id] = "max_inflight_tasks"
                        break
                    lease_id = f"{task.batch_id}:{task.task_id}"
                    lease = self.lease_manager.grant(
                        lease_id,
                        task.service_id,
                        batch_id=task.batch_id,
                        task_id=task.task_id,
                        replica_count=task.replica_count,
                    )
                    if lease is None:
                        block_reason[task.task_id] = (
                            self.lease_manager.capacity_reason(
                                task.service_id, task.replica_count
                            )
                            or "quota_exhausted"
                        )
                        break

                    pending.pop(0)
                    started = queued_at[task.task_id]
                    dispatched = monotonic_ms()
                    span = {
                        "task_id": task.task_id,
                        "round_id": task.round_id,
                        "batch_id": task.batch_id,
                        "minute": task.minute,
                        "service_id": task.service_id,
                        "replica_count": task.replica_count,
                        "queued_at_monotonic_ms": started,
                        "dispatched_at_monotonic_ms": dispatched,
                        "waited_ms": max(0, dispatched - started),
                        "wait_reason": block_reason[task.task_id],
                        "lease_id": lease.lease_id,
                    }
                    spans[task.task_id] = span
                    self.wait_spans.append(span)
                    future = executor.submit(self._execute, task, lease)
                    active[future] = task
                    active_leases[future] = lease

                self.max_observed_inflight = max(
                    self.max_observed_inflight, len(active)
                )
                if active:
                    done, _ = wait(active, timeout=0.02, return_when=FIRST_COMPLETED)
                    for future in done:
                        task = active.pop(future)
                        active_leases.pop(future, None)
                        results.append(future.result())
                elif pending:
                    self.lease_manager.reap_expired()
                    time.sleep(0.01)

        results.sort(key=lambda item: item["task_id"])
        self.execution_results = results
        return results

    def _execute(self, task: ScaleTask, lease: Lease) -> dict[str, Any]:
        if self.policy.simulated_task_ms:
            time.sleep(self.policy.simulated_task_ms / 1000.0)
        return {
            "task_id": task.task_id,
            "round_id": task.round_id,
            "batch_id": task.batch_id,
            "service_id": task.service_id,
            "minute": task.minute,
            "replica_count": task.replica_count,
            "lease_id": lease.lease_id,
            "lease_version_at_start": lease.version,
            "worker_thread": threading_get_ident(),
            "status": "completed",
        }


def threading_get_ident() -> int:
    import threading

    return threading.get_ident()
