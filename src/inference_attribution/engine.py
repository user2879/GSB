"""Run planned batches through loopback workers and commit receipts."""

from __future__ import annotations

import os
import signal
import threading
from dataclasses import dataclass
from typing import Any

from .invalidation import InvalidationReport, trace_for_batch
from .ledger import Ledger
from .planner import BatchPlan, DispatchPlan
from .scheduler import ScheduleReport, QuotaScheduler
from .spec import MergeRules, RequestSet, WorkerSet
from .workers import WorkerPool, WorkerResponse


@dataclass(frozen=True)
class ExecutionOutcome:
    completed_batch_ids: tuple[str, ...]
    skipped_batch_ids: tuple[str, ...]
    progress_cursor: int
    ledger_digest: str
    schedule_events: tuple[dict[str, Any], ...]
    max_global_running: int
    max_worker_running: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "completed_batch_ids": list(self.completed_batch_ids),
            "skipped_batch_ids": list(self.skipped_batch_ids),
            "progress_cursor": self.progress_cursor,
            "ledger_digest": self.ledger_digest,
            "schedule_events": list(self.schedule_events),
            "max_global_running": self.max_global_running,
            "max_worker_running": dict(self.max_worker_running),
        }


def _runtime_receipts(response: WorkerResponse) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for receipt in response.receipts:
        receipts.append(
            {
                "source_format": "r3",
                "source_line": 0,
                "batch_id": receipt["batch_id"],
                "request_id": receipt["request_id"],
                "output_digest": receipt["output_digest"],
                "worker_id": receipt["worker_id"],
                "param_version_seq": receipt["param_version_seq"],
                "latency_ms": receipt["latency_ms"],
                "rank_in_batch": receipt["rank_in_batch"],
                "superseded_request_ids": list(receipt["superseded_request_ids"]),
                "param_digest": receipt["param_digest"],
                "filled_fields": [],
                "fill_basis": {},
            }
        )
    return receipts


def execute_plan(
    *,
    plan: DispatchPlan,
    request_set: RequestSet,
    worker_set: WorkerSet,
    rules: MergeRules,
    ledger: Ledger,
    invalid_batch_ids: set[str] | None = None,
    invalidation_report: InvalidationReport | None = None,
    crash_after_commits: int | None = None,
) -> ExecutionOutcome:
    invalid = set(invalid_batch_ids or ())
    if crash_after_commits is not None and crash_after_commits < 1:
        raise ValueError("crash_after_commits must be positive")
    ledger.apply_plan(plan, invalid_batch_ids=invalid)
    rows = ledger.row_by_id()
    target_batches = [
        batch
        for batch in plan.sorted_batches()
        if rows[batch.batch_id]["status"] != "completed"
    ]
    skipped = [
        batch.batch_id
        for batch in plan.sorted_batches()
        if rows[batch.batch_id]["status"] == "completed"
    ]
    if not target_batches:
        return ExecutionOutcome(
            completed_batch_ids=(),
            skipped_batch_ids=tuple(skipped),
            progress_cursor=ledger.progress_cursor,
            ledger_digest=ledger.content_digest,
            schedule_events=(),
            max_global_running=0,
            max_worker_running={},
        )

    worker_map = worker_set.by_id()
    request_map = request_set.by_id()
    needed_workers = sorted({batch.worker_id for batch in target_batches})
    selected_workers = [worker_map[worker_id] for worker_id in needed_workers]
    worker_quotas = {
        worker.worker_id: worker.concurrency_quota for worker in selected_workers
    }
    scheduler = QuotaScheduler(
        global_limit=rules.global_concurrency,
        worker_quotas=worker_quotas,
    )
    completion_lock = threading.Lock()
    committed_here = 0

    with WorkerPool(selected_workers) as pool:
        def process(batch: BatchPlan) -> WorkerResponse:
            nonlocal committed_here
            response = pool.client(batch.worker_id).dispatch_batch(batch, request_map)
            if invalidation_report is not None and batch.batch_id in invalidation_report.invalid_batch_ids():
                trace = trace_for_batch(invalidation_report, batch.batch_id)
            else:
                trace = {
                    "mode": "initial",
                    "direct": False,
                    "indirect": False,
                    "worker_changes": [],
                }
            with completion_lock:
                ledger.complete_batch(
                    batch,
                    receipt_digest=response.batch_output_digest,
                    normalized_receipts=_runtime_receipts(response),
                    invalidation_trace=trace,
                )
                committed_here += 1
                if (
                    crash_after_commits is not None
                    and committed_here >= crash_after_commits
                ):
                    if hasattr(signal, "SIGKILL"):
                        os.kill(os.getpid(), signal.SIGKILL)
                    os._exit(137)
            return response

        report: ScheduleReport[BatchPlan, WorkerResponse] = scheduler.run(
            target_batches,
            process,
            batch_id=lambda batch: batch.batch_id,
            worker_id=lambda batch: batch.worker_id,
        )

    events = tuple(
        result.event.to_dict() for result in report.results
    )
    return ExecutionOutcome(
        completed_batch_ids=tuple(result.item.batch_id for result in report.results),
        skipped_batch_ids=tuple(skipped),
        progress_cursor=ledger.progress_cursor,
        ledger_digest=ledger.content_digest,
        schedule_events=events,
        max_global_running=report.max_global_running,
        max_worker_running=dict(report.max_worker_running),
    )
