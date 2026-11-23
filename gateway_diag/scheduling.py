"""Quota-aware batch scheduler projected onto a real worker thread pool."""

from __future__ import annotations

import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Protocol

from .backends import InferenceReceipt
from .errors import SchedulerError
from .routing import BatchPlan


class BatchRunner(Protocol):
    def infer(
        self,
        backend_id: str,
        batch_id: str,
        items: Iterable[tuple[int, Iterable[float]]],
    ) -> InferenceReceipt: ...


def _monotonic_ms() -> int:
    return time.monotonic_ns() // 1_000_000


@dataclass(frozen=True)
class BatchExecution:
    plan: BatchPlan
    wait_start_ms: int
    wait_end_ms: int
    exec_start_ms: int
    exec_end_ms: int
    elapsed_ms: int
    receipt: InferenceReceipt
    global_limit: int
    backend_quota: int
    thread_name: str

    @property
    def wait_ms(self) -> int:
        return max(0, self.wait_end_ms - self.wait_start_ms)

    @property
    def output_digests(self) -> dict[int, str]:
        return self.receipt.ordered_output_digests


@dataclass(frozen=True)
class SchedulerSummary:
    max_global_observed: int
    max_backend_observed: Mapping[str, int]
    dispatched_batches: int
    worker_threads: tuple[str, ...]


@dataclass
class _Running:
    plan: BatchPlan
    exec_start_ms: int
    thread_name: str | None = None


class QuotaScheduler:
    def __init__(
        self,
        runner: BatchRunner,
        backend_quotas: Mapping[str, int],
        global_limit: int,
        *,
        on_complete: Callable[[BatchExecution], None] | None = None,
        now_ms: Callable[[], int] = _monotonic_ms,
    ) -> None:
        if global_limit < 1:
            raise SchedulerError("global concurrency limit must be positive", field="global_concurrency")
        for backend_id, quota in backend_quotas.items():
            if quota < 1:
                raise SchedulerError(
                    f"backend quota for {backend_id!r} must be positive",
                    backend_id=backend_id,
                    field="backend_concurrency",
                )
        self.runner = runner
        self.backend_quotas = dict(backend_quotas)
        self.global_limit = global_limit
        self.on_complete = on_complete
        self.now_ms = now_ms
        self.max_global_observed = 0
        self.max_backend_observed: dict[str, int] = {key: 0 for key in backend_quotas}
        self.worker_threads: set[str] = set()
        self.dispatched_batches = 0

    def _execute(self, plan: BatchPlan) -> tuple[InferenceReceipt, str]:
        thread_name = threading.current_thread().name
        receipt = self.runner.infer(
            plan.backend_id,
            plan.batch_id,
            [
                (item.sample.request_id, item.sample.input_vector)
                for item in plan.items
            ],
        )
        return receipt, thread_name

    def run(self, batches: Iterable[BatchPlan]) -> tuple[list[BatchExecution], SchedulerSummary]:
        pending = sorted(
            batches, key=lambda batch: (batch.first_arrival_ms, batch.batch_id)
        )
        if not pending:
            return [], SchedulerSummary(0, {}, 0, ())
        unknown = {
            batch.backend_id
            for batch in pending
            if batch.backend_id not in self.backend_quotas
        }
        if unknown:
            backend_id = sorted(unknown)[0]
            raise SchedulerError(
                f"no concurrency quota declared for backend {backend_id!r}",
                backend_id=backend_id,
                field="backend_concurrency",
            )

        now = self.now_ms()
        wait_started = {batch.batch_id: now for batch in pending}
        running: dict[Future[tuple[InferenceReceipt, str]], _Running] = {}
        backend_running = {backend_id: 0 for backend_id in self.backend_quotas}
        results: list[BatchExecution] = []

        with ThreadPoolExecutor(
            max_workers=self.global_limit,
            thread_name_prefix="gateway-batch",
        ) as pool:
            while pending or running:
                dispatch_ready = True
                while dispatch_ready and pending and len(running) < self.global_limit:
                    candidate_index: int | None = None
                    for index, batch in enumerate(pending):
                        if backend_running[batch.backend_id] < self.backend_quotas[batch.backend_id]:
                            candidate_index = index
                            break
                    if candidate_index is None:
                        dispatch_ready = False
                        break
                    plan = pending.pop(candidate_index)
                    dispatch_ms = self.now_ms()
                    future = pool.submit(self._execute, plan)
                    running[future] = _Running(plan=plan, exec_start_ms=dispatch_ms)
                    backend_running[plan.backend_id] += 1
                    current_global = len(running)
                    self.max_global_observed = max(self.max_global_observed, current_global)
                    self.max_backend_observed[plan.backend_id] = max(
                        self.max_backend_observed.get(plan.backend_id, 0),
                        backend_running[plan.backend_id],
                    )
                    self.dispatched_batches += 1

                if not running:
                    if pending:
                        raise SchedulerError(
                            "scheduler cannot dispatch queued batches without exceeding quotas"
                        )
                    break

                done, _ = wait(running.keys(), timeout=0.05, return_when=FIRST_COMPLETED)
                for future in done:
                    metadata = running.pop(future)
                    backend_running[metadata.plan.backend_id] -= 1
                    exec_end_ms = self.now_ms()
                    try:
                        receipt, thread_name = future.result()
                    except Exception:
                        raise
                    self.worker_threads.add(thread_name)
                    execution = BatchExecution(
                        plan=metadata.plan,
                        wait_start_ms=wait_started[metadata.plan.batch_id],
                        wait_end_ms=metadata.exec_start_ms,
                        exec_start_ms=metadata.exec_start_ms,
                        exec_end_ms=exec_end_ms,
                        elapsed_ms=max(0, exec_end_ms - metadata.exec_start_ms),
                        receipt=receipt,
                        global_limit=self.global_limit,
                        backend_quota=self.backend_quotas[metadata.plan.backend_id],
                        thread_name=thread_name,
                    )
                    if self.on_complete is not None:
                        self.on_complete(execution)
                    results.append(execution)

        summary = SchedulerSummary(
            max_global_observed=self.max_global_observed,
            max_backend_observed=dict(sorted(self.max_backend_observed.items())),
            dispatched_batches=self.dispatched_batches,
            worker_threads=tuple(sorted(self.worker_threads)),
        )
        return results, summary
