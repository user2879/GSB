"""Stable budget/quota scheduling and bounded thread-pool execution."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable

from .errors import E_DOC_QUOTA_EXCEEDED, E_MATCH_BUDGET_PENDING
from .hashing import stable_key
from .segmentation import estimate_match_points


@dataclass
class BatchPlan:
    batch_id: int
    tasks: list[dict[str, Any]]
    pending: list[dict[str, Any]]
    failures: list[dict[str, Any]]
    allocations: list[dict[str, Any]]
    budget: int
    consumed: int
    remaining: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "tasks": self.tasks,
            "pending": self.pending,
            "failures": self.failures,
            "allocations": self.allocations,
            "budget": self.budget,
            "consumed": self.consumed,
            "remaining": self.remaining,
        }


def plan_batch(
    documents: list[dict[str, Any]],
    index: dict[str, Any],
    budget: int,
    doc_quota: int,
    batch_id: int = 1,
    pending_ids: list[str] | None = None,
    known_failures: list[dict[str, Any]] | None = None,
) -> BatchPlan:
    by_id = {doc["doc_id"]: doc for doc in documents}
    pending_ids = list(pending_ids or [])
    ordered_ids: list[str] = []
    for doc_id in pending_ids:
        if doc_id in by_id and doc_id not in ordered_ids:
            ordered_ids.append(doc_id)
    for doc in sorted(documents, key=lambda item: item["doc_id"]):
        if doc["doc_id"] not in ordered_ids:
            ordered_ids.append(doc["doc_id"])

    tasks: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    failures = list(known_failures or [])
    allocations: list[dict[str, Any]] = []
    consumed = 0
    budget_closed = False

    for doc_id in ordered_ids:
        doc = by_id[doc_id]
        points = estimate_match_points(doc, index)
        sort_key = stable_key(doc_id, "document-match-tag")
        base = {
            "doc_id": doc_id,
            "doc_seq": doc["doc_seq"],
            "match_points": points,
            "stable_sort_key": sort_key,
            "stage_keys": {
                "load": stable_key(doc_id, "load"),
                "segment": stable_key(doc_id, "segment"),
                "tag": stable_key(doc_id, "tag"),
                "receipts": stable_key(doc_id, "receipts"),
            },
        }

        if points > doc_quota:
            failure = {
                "code": E_DOC_QUOTA_EXCEEDED,
                "doc_id": doc_id,
                "doc_seq": doc["doc_seq"],
                "match_points": points,
                "doc_quota": doc_quota,
                "stable_sort_key": sort_key,
                "batch_id": batch_id,
            }
            if not any(item.get("doc_id") == doc_id and item.get("code") == E_DOC_QUOTA_EXCEEDED for item in failures):
                failures.append(failure)
            allocations.append({**base, "decision": "doc_quota_exceeded"})
            continue

        if budget_closed or points > budget - consumed:
            budget_closed = True
            marker = {
                "code": E_MATCH_BUDGET_PENDING,
                "doc_id": doc_id,
                "doc_seq": doc["doc_seq"],
                "match_points": points,
                "remaining_budget": budget - consumed,
                "stable_sort_key": sort_key,
                "batch_id": batch_id,
            }
            pending.append(marker)
            allocations.append({**base, "decision": "pending_budget"})
            continue

        consumed += points
        task = {**base, "allocated_points": points, "decision": "run"}
        tasks.append(task)
        allocations.append(task)

    pending.sort(key=lambda item: item["doc_id"])
    failures.sort(key=lambda item: (item.get("doc_seq", 0), item.get("code", "")))
    return BatchPlan(
        batch_id=batch_id,
        tasks=tasks,
        pending=pending,
        failures=failures,
        allocations=allocations,
        budget=budget,
        consumed=consumed,
        remaining=budget - consumed,
    )


def execute_pool(
    tasks: list[dict[str, Any]],
    worker: Callable[[dict[str, Any]], dict[str, Any]],
    concurrency: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run tasks in a bounded pool and return results in stable doc_seq order."""

    max_workers = max(1, min(int(concurrency), len(tasks) or 1))
    lock = threading.Lock()
    active = 0
    max_observed = 0

    def wrapped(task: dict[str, Any]) -> dict[str, Any]:
        nonlocal active, max_observed
        with lock:
            active += 1
            max_observed = max(max_observed, active)
        try:
            return worker(task)
        finally:
            with lock:
                active -= 1

    results: list[dict[str, Any]] = []
    if max_workers == 1:
        for task in tasks:
            results.append(wrapped(task))
    else:
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="lexicon-doc") as executor:
            future_map = {executor.submit(wrapped, task): task for task in tasks}
            for future in as_completed(future_map):
                results.append(future.result())

    results.sort(key=lambda result: (result.get("doc_seq", 0), result.get("doc_id", "")))
    metrics = {"configured_concurrency": int(concurrency), "pool_size": max_workers, "max_observed_concurrency": max_observed}
    return results, metrics


def run_batches_until_complete(
    documents: list[dict[str, Any]],
    index: dict[str, Any],
    worker: Callable[[dict[str, Any]], dict[str, Any]],
    config: dict[str, Any],
    max_batches: int = 10,
) -> dict[str, Any]:
    pending_ids: list[str] = []
    failures: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    pool_metrics: list[dict[str, Any]] = []
    cumulative_consumed = 0
    pending: list[dict[str, Any]] = []
    completed_ids: set[str] = set()

    for batch_id in range(1, max_batches + 1):
        remaining_documents = [doc for doc in documents if doc["doc_id"] not in completed_ids]
        plan = plan_batch(
            remaining_documents,
            index,
            int(config["match_budget_points"]),
            int(config["doc_quota_points"]),
            batch_id=batch_id,
            pending_ids=[item["doc_id"] for item in pending],
            known_failures=failures,
        )
        failures = plan.failures
        pending = plan.pending
        cumulative_consumed += plan.consumed
        task_by_id = {task["doc_id"]: task for task in plan.tasks}
        results, metrics = execute_pool(plan.tasks, worker, int(config["concurrency"]))
        all_results.extend(results)
        completed_ids.update(result["doc_id"] for result in results)
        pool_metrics.append(metrics)
        batches.append(plan.as_dict())
        if not pending:
            break
        pending_ids = [item["doc_id"] for item in pending]

    all_results.sort(key=lambda item: (item.get("doc_seq", 0), item.get("doc_id", "")))
    return {
        "results": all_results,
        "batches": batches,
        "pending": pending,
        "failures": failures,
        "pool_metrics": pool_metrics,
        "cumulative_consumed_budget": cumulative_consumed,
    }
