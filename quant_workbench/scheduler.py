"""Stable point-budget planning and bounded thread-pool execution."""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar

from .errors import E_POINT_BUDGET_PENDING, Issue
from .quantization import DeadlineExceeded

T = TypeVar("T")


@dataclass(slots=True)
class BudgetPlan:
    batches: list[list[str]]
    ledger: list[dict[str, Any]]
    pending_layers: list[str] = field(default_factory=list)
    pending_points: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "batches": self.batches,
            "ledger": self.ledger,
            "pending_layers": self.pending_layers,
            "pending_points": self.pending_points,
        }


def build_budget_plan(point_counts: dict[str, int], budget: int) -> BudgetPlan:
    """Partition layers by lexicographic name using a per-batch point budget.

    A layer that cannot fit into a fresh batch remains queued (and all later
    layers remain behind it) with E_POINT_BUDGET_PENDING semantics.
    """

    if budget <= 0:
        raise ValueError("budget must be positive")
    batches: list[list[str]] = []
    current: list[str] = []
    remaining = budget
    ordered = sorted(point_counts)
    oversized_index: int | None = None
    for index, layer in enumerate(ordered):
        points = int(point_counts[layer])
        if points > budget:
            oversized_index = index
            break
        if current and points > remaining:
            batches.append(current)
            current = []
            remaining = budget
        current.append(layer)
        remaining -= points
    if current:
        batches.append(current)

    ledger: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(batches):
        running = budget
        for position, name in enumerate(batch):
            points = int(point_counts[name])
            ledger.append(
                {
                    "batch": batch_index,
                    "order": position,
                    "layer": name,
                    "points": points,
                    "remaining_before": running,
                    "remaining_after": running - points,
                }
            )
            running -= points
    pending = ordered[oversized_index:] if oversized_index is not None else []
    return BudgetPlan(
        batches=batches,
        ledger=ledger,
        pending_layers=pending,
        pending_points=sum(int(point_counts[name]) for name in pending),
    )


@dataclass(slots=True)
class BatchExecution(Generic[T]):
    completed: dict[str, T]
    failures: list[Issue]
    pending_layers: list[str]
    timed_out: bool
    started_layers: list[str]
    completed_layers: list[str]
    cancelled_layers: list[str]
    elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "failures": [issue.to_dict() for issue in self.failures],
            "pending_layers": self.pending_layers,
            "timed_out": self.timed_out,
            "started_layers": self.started_layers,
            "completed_layers": self.completed_layers,
            "cancelled_layers": self.cancelled_layers,
            "elapsed_ms": self.elapsed_ms,
        }


def _sleep_with_cancel(
    delay_ms: float, deadline: float | None, cancel_event: threading.Event
) -> None:
    remaining = max(0.0, float(delay_ms)) / 1000.0
    while remaining > 0.0:
        if cancel_event.is_set():
            raise DeadlineExceeded("cancelled")
        if deadline is not None and time.perf_counter() >= deadline:
            raise DeadlineExceeded("interactive deadline reached")
        slice_seconds = min(remaining, 0.001)
        time.sleep(slice_seconds)
        remaining -= slice_seconds


def execute_batch(
    *,
    layers: list[str],
    worker: Callable[[str, float | None, threading.Event], T],
    executor: ThreadPoolExecutor,
    deadline: float | None,
    worker_delay_ms: float = 0.0,
) -> BatchExecution[T]:
    """Execute one stable batch concurrently and return results sorted by layer."""

    start = time.perf_counter()
    cancel_event = threading.Event()
    futures: dict[str, Future[T]] = {}
    started: list[str] = []

    def invoke(layer: str) -> T:
        _sleep_with_cancel(worker_delay_ms, deadline, cancel_event)
        if deadline is not None and time.perf_counter() >= deadline:
            raise DeadlineExceeded("interactive deadline reached")
        return worker(layer, deadline, cancel_event)

    for layer in layers:
        if deadline is not None and time.perf_counter() >= deadline:
            break
        started.append(layer)
        futures[layer] = executor.submit(invoke, layer)

    if not futures:
        return BatchExecution(
            completed={},
            failures=[],
            pending_layers=list(layers),
            timed_out=bool(layers),
            started_layers=[],
            completed_layers=[],
            cancelled_layers=[],
            elapsed_ms=(time.perf_counter() - start) * 1000.0,
        )

    timeout: float | None = None
    if deadline is not None:
        timeout = max(0.0, deadline - time.perf_counter())
    done, not_done = wait(futures.values(), timeout=timeout)
    timed_out = bool(not_done)
    if not_done:
        cancel_event.set()
        for future in not_done:
            future.cancel()
    completed: dict[str, T] = {}
    failures: list[Issue] = []
    cancelled: list[str] = []
    for layer in sorted(futures):
        future = futures[layer]
        if future.cancelled() or layer not in {name for name, f in futures.items() if f in done}:
            if not future.done():
                cancelled.append(layer)
            continue
        try:
            completed[layer] = future.result()
        except DeadlineExceeded:
            cancelled.append(layer)
        except BaseException as exc:  # task boundary: turn unexpected errors into diagnostics
            failures.append(
                Issue(
                    "E_INTERNAL",
                    f"layer {layer!r} task failed: {type(exc).__name__}: {exc}",
                    layer=layer,
                )
            )
    pending = [layer for layer in layers if layer not in completed and layer not in {
        issue.layer for issue in failures if issue.layer is not None
    }]
    completed_names = [layer for layer in layers if layer in completed]
    failures.sort(key=lambda issue: (issue.layer or "", issue.code))
    return BatchExecution(
        completed=completed,
        failures=failures,
        pending_layers=pending,
        timed_out=timed_out or bool(cancelled),
        started_layers=started,
        completed_layers=completed_names,
        cancelled_layers=sorted(set(cancelled)),
        elapsed_ms=(time.perf_counter() - start) * 1000.0,
    )


def point_budget_pending_issues(plan: BudgetPlan, *, command_index: int | None = None) -> list[Issue]:
    issues: list[Issue] = []
    for layer in plan.pending_layers:
        issues.append(
            Issue(
                E_POINT_BUDGET_PENDING,
                f"layer {layer!r} remains queued because one layer exceeds the per-batch point budget",
                command_index=command_index,
                layer=layer,
            )
        )
    return issues
