from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable


WorkItem = dict[str, Any]
Worker = Callable[[WorkItem], dict[str, Any]]


def work_sort_key(item: WorkItem) -> tuple[str, int, str]:
    return (str(item["ticket_id"]), int(item["start"]), str(item["scope_id"]))


def plan_batch(
    work_items: list[WorkItem],
    budget_points: int,
    pending: list[WorkItem] | None = None,
) -> tuple[list[WorkItem], list[WorkItem], int]:
    """Consume head-of-line work in stable order and retain the unspent tail."""
    queue = sorted((pending or []) + list(work_items), key=work_sort_key)
    eligible: list[WorkItem] = []
    deferred: list[WorkItem] = []
    used = 0
    stopped = False
    for item in queue:
        cost = int(item["point_cost"])
        if stopped:
            deferred.append({**item, "status": "pending", "error_code": "E_SCOPE_BUDGET_PENDING"})
            continue
        if used + cost <= int(budget_points):
            eligible.append(item)
            used += cost
        else:
            stopped = True
            deferred.append({**item, "status": "pending", "error_code": "E_SCOPE_BUDGET_PENDING"})
    return eligible, deferred, used


class BudgetScheduler:
    def __init__(self, budget_points: int, max_workers: int):
        if budget_points < 0:
            raise ValueError("budget_points must be non-negative")
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        self.budget_points = int(budget_points)
        self.max_workers = int(max_workers)
        self.pending: list[WorkItem] = []
        self.consumed_budget_points = 0
        self.batch_count = 0

    def submit_batch(
        self,
        work_items: list[WorkItem],
        worker: Worker,
        *,
        budget_points: int | None = None,
        max_workers: int | None = None,
    ) -> dict[str, Any]:
        budget = self.budget_points if budget_points is None else int(budget_points)
        workers_limit = self.max_workers if max_workers is None else int(max_workers)
        if budget < 0:
            raise ValueError("budget must be non-negative")
        if workers_limit <= 0:
            raise ValueError("max_workers must be positive")

        eligible, deferred, used = plan_batch(work_items, budget, self.pending)
        active = 0
        peak = 0
        lock = threading.Lock()

        def wrapped(item: WorkItem) -> dict[str, Any]:
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                result = worker(item)
            finally:
                with lock:
                    active -= 1
            result.setdefault("scope_id", item["scope_id"])
            result.setdefault("ticket_id", item["ticket_id"])
            result.setdefault("start", item["start"])
            return result

        if eligible:
            with ThreadPoolExecutor(max_workers=min(workers_limit, len(eligible))) as pool:
                completion_order = list(pool.map(wrapped, eligible))
        else:
            completion_order = []

        # pool.map preserves input order; resort defensively so completion timing cannot matter.
        results = sorted(completion_order, key=lambda result: work_sort_key(result))
        self.pending = sorted(deferred, key=work_sort_key)
        self.consumed_budget_points += used
        self.batch_count += 1
        return {
            "batch_index": self.batch_count,
            "budget_points": budget,
            "consumed_batch_points": used,
            "consumed_budget_points": self.consumed_budget_points,
            "processed_count": len(results),
            "pending_count": len(self.pending),
            "peak_in_flight": peak,
            "max_workers_limit": workers_limit,
            "results": results,
            "pending": self.pending,
            "status": "E_SCOPE_BUDGET_PENDING" if self.pending else "OK",
        }


def drain_all(
    work_items: list[WorkItem],
    worker: Worker,
    *,
    budget_points: int,
    max_workers: int,
    larger_budget_on_retry: int | None = None,
) -> dict[str, Any]:
    """Drain work, enlarging the next budget when a head item exceeds one batch."""
    scheduler = BudgetScheduler(budget_points, max_workers)
    all_results: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    current_budget = budget_points
    while True:
        batch = scheduler.submit_batch(work_items if not batches else [], worker, budget_points=current_budget)
        batches.append({key: value for key, value in batch.items() if key not in {"results", "pending"}})
        all_results.extend(batch["results"])
        if not scheduler.pending:
            break
        if larger_budget_on_retry is not None:
            current_budget = larger_budget_on_retry
        elif batch["processed_count"] == 0:
            # A single oversized item cannot ever fit; callers that want eventual completion
            # must supply a larger retry budget. Keep the stable pending marker in that case.
            break
    all_results.sort(key=lambda result: work_sort_key(result))
    return {
        "results": all_results,
        "batches": batches,
        "pending": scheduler.pending,
        "consumed_budget_points": scheduler.consumed_budget_points,
        "complete": not scheduler.pending,
    }
