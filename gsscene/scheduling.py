"""Deterministic budget allocation and bounded thread-pool execution."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, TypeVar

from . import codes
from .config import SceneConfig
from .kernels import KernelPlan

T = TypeVar("T")
R = TypeVar("R")


class ConcurrencyTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.started = 0
        self.completed = 0

    def enter(self) -> None:
        with self._lock:
            self.active += 1
            self.started += 1
            self.max_active = max(self.max_active, self.active)

    def leave(self) -> None:
        with self._lock:
            self.active = max(0, self.active - 1)
            self.completed += 1

    def as_dict(self) -> dict[str, int]:
        with self._lock:
            return {
                "completed": self.completed,
                "max_in_flight": self.max_active,
                "started": self.started,
            }


def parallel_map(
    function: Callable[[T], R],
    items: Iterable[T],
    max_workers: int,
    tracker: ConcurrencyTracker,
) -> list[R]:
    values = list(items)
    if not values:
        return []
    if max_workers <= 1 or len(values) == 1:
        results: list[R] = []
        for item in values:
            tracker.enter()
            try:
                results.append(function(item))
            finally:
                tracker.leave()
        return results
    indexed: dict[int, R] = {}

    def tracked(item: T) -> R:
        tracker.enter()
        try:
            return function(item)
        finally:
            tracker.leave()

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="gsscene") as executor:
        futures = {}
        for index, item in enumerate(values):
            future = executor.submit(tracked, item)
            futures[future] = index
        for future in as_completed(futures):
            index = futures[future]
            indexed[index] = future.result()
    return [indexed[index] for index in range(len(values))]


@dataclass
class BudgetEntry:
    view_index: int
    level: int
    requested: int
    kernel_budget: int
    allowed_kernel: int
    allowed_point: int
    status: str
    pending_ranges: list[list[int]]
    pending_count: int
    point_budget: int
    remaining_kernel_budget: int
    remaining_point_budget: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed_kernel": self.allowed_kernel,
            "allowed_point": self.allowed_point,
            "kernel_budget": self.kernel_budget,
            "kind": "kernel",
            "level": self.level,
            "pending_count": self.pending_count,
            "pending_ranges": self.pending_ranges,
            "point_budget": self.point_budget,
            "remaining_kernel_budget": self.remaining_kernel_budget,
            "remaining_point_budget": self.remaining_point_budget,
            "requested": self.requested,
            "status": self.status,
            "view_index": self.view_index,
        }


def allocate_kernel_budgets(plans: Iterable[KernelPlan], config: SceneConfig) -> tuple[list[KernelPlan], dict[str, Any]]:
    ordered = sorted(plans, key=lambda plan: (plan.view_index, plan.level))
    adjusted: list[KernelPlan] = []
    entries: list[BudgetEntry] = []
    pending: list[dict[str, Any]] = []
    for plan in ordered:
        budget = int(config.kernel_budgets[plan.level])
        point_budget = int(config.point_budget)
        requested = int(plan.requested_count)
        kernel_allowed = min(requested, budget)
        point_allowed = min(kernel_allowed, point_budget)
        allowed = min(kernel_allowed, point_allowed)
        if allowed < requested:
            if kernel_allowed <= point_allowed:
                status = codes.E_KERNEL_BUDGET_EXHAUSTED
            else:
                status = codes.E_POINT_BUDGET_PENDING
        else:
            status = "COMPLETED"
        pending_count = max(0, requested - allowed)
        pending_ranges = [[allowed, requested]] if pending_count else []
        if pending_count:
            pending.append(
                {
                    "code": status,
                    "count": pending_count,
                    "kernel_index_end": requested,
                    "kernel_index_start": allowed,
                    "level": plan.level,
                    "view_index": plan.view_index,
                }
            )
        trim_plan = replace(
            plan,
            allowed_count=allowed,
            kernel_keys=plan.kernel_keys[:allowed],
        )
        adjusted.append(trim_plan)
        entries.append(
            BudgetEntry(
                view_index=plan.view_index,
                level=plan.level,
                requested=requested,
                kernel_budget=budget,
                allowed_kernel=kernel_allowed,
                allowed_point=point_allowed,
                status=status,
                pending_ranges=pending_ranges,
                pending_count=pending_count,
                point_budget=point_budget,
                remaining_kernel_budget=max(0, budget - allowed),
                remaining_point_budget=max(0, point_budget - allowed),
            )
        )
    ledger = {
        "kernel_scheduler": {
            "budget_scope": "per_view_level",
            "entries": [entry.as_dict() for entry in entries],
            "max_concurrency": int(config.concurrency),
        },
        "pending": pending,
        "point_scheduler": {
            "budget_scope": "per_view_level_batch",
            "entries": [
                {
                    "level": entry.level,
                    "point_budget": entry.point_budget,
                    "requested": entry.requested,
                    "status": entry.status,
                    "view_index": entry.view_index,
                }
                for entry in entries
            ],
            "max_concurrency": int(config.concurrency),
        },
    }
    return adjusted, ledger
