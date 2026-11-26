from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable

from .errors import E_OP_QUOTA_EXCEEDED, E_POINT_BUDGET_PENDING


@dataclass
class Task:
    task_id: str
    ordinal: int
    op_seq: int
    token_seq: int
    work: Callable[[], Any]
    cost: int = 1
    kind: str = "stage"


@dataclass
class SchedulerReport:
    status: str = "ok"
    max_workers: int = 1
    max_observed_workers: int = 0
    point_budget: int = 0
    operator_quota: int = 0
    consumed_budget: int = 0
    remaining_budget: int = 0
    consumed_by_op: dict[str, int] = field(default_factory=dict)
    admitted: list[str] = field(default_factory=list)
    pending: list[dict[str, Any]] = field(default_factory=list)
    quota_aborted: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: list[dict[str, Any]] = field(default_factory=list)
    execution_order: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "max_workers": self.max_workers,
            "max_observed_workers": self.max_observed_workers,
            "point_budget": self.point_budget,
            "operator_quota": self.operator_quota,
            "consumed_budget": self.consumed_budget,
            "remaining_budget": self.remaining_budget,
            "consumed_by_op": self.consumed_by_op,
            "admitted": self.admitted,
            "pending": self.pending,
            "quota_aborted": self.quota_aborted,
            "diagnostics": self.diagnostics,
            "execution_order": self.execution_order,
        }


class BudgetScheduler:
    def __init__(self, *, point_budget: int, operator_quota: int, max_workers: int = 1):
        self.point_budget = int(point_budget)
        self.operator_quota = int(operator_quota)
        self.max_workers = max(1, int(max_workers))

    def run(self, tasks: list[Task], *, serial: bool = False,
            report: SchedulerReport | None = None) -> tuple[dict[str, Any], SchedulerReport]:
        ordered = sorted(tasks, key=lambda t: (t.op_seq, t.token_seq, t.ordinal, t.task_id))
        if report is None:
            report = SchedulerReport(
                max_workers=1 if serial else self.max_workers,
                point_budget=self.point_budget,
                operator_quota=self.operator_quota,
                remaining_budget=self.point_budget,
            )
        else:
            report.max_workers = 1 if serial else max(report.max_workers, self.max_workers)
        admitted: list[Task] = []
        quota_failed_ops: set[int] = set()
        budget_stopped = False

        for task in ordered:
            if budget_stopped:
                self._add_pending(report, task, "previous-task-exhausted-budget")
                continue

            op_key = str(task.op_seq)
            op_consumed = report.consumed_by_op.get(op_key, 0)
            if task.op_seq in quota_failed_ops:
                report.quota_aborted.append({"task_id": task.task_id, "op_seq": task.op_seq, "reason": "operator-quota-aborted"})
                continue

            if op_consumed + task.cost > self.operator_quota:
                quota_failed_ops.add(task.op_seq)
                diagnostic = {
                    "code": E_OP_QUOTA_EXCEEDED,
                    "op_seq": task.op_seq,
                    "task_id": task.task_id,
                    "consumed": op_consumed,
                    "cost": task.cost,
                    "quota": self.operator_quota,
                }
                report.diagnostics.append(diagnostic)
                report.quota_aborted.append(
                    {"task_id": task.task_id, "op_seq": task.op_seq, "code": E_OP_QUOTA_EXCEEDED, "reason": "cost-exceeds-remaining-operator-quota"}
                )
                report.status = E_OP_QUOTA_EXCEEDED
                continue

            if report.consumed_budget + task.cost > self.point_budget:
                self._add_pending(report, task, "point-budget-insufficient")
                budget_stopped = True
                report.status = E_POINT_BUDGET_PENDING
                continue

            admitted.append(task)
            report.admitted.append(task.task_id)
            report.consumed_budget += task.cost
            report.consumed_by_op[op_key] = op_consumed + task.cost

        report.remaining_budget = self.point_budget - report.consumed_budget

        results: dict[str, Any] = {}
        if serial or self.max_workers <= 1:
            for task in admitted:
                results[task.task_id] = task.work()
                report.max_observed_workers = 1
        else:
            active = 0
            observed = 0
            lock = threading.Lock()

            def wrapped(task: Task) -> tuple[str, Any]:
                nonlocal active, observed
                with lock:
                    active += 1
                    observed = max(observed, active)
                try:
                    return task.task_id, task.work()
                finally:
                    with lock:
                        active -= 1

            with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="chaindiag") as pool:
                futures = [pool.submit(wrapped, task) for task in admitted]
                for future in as_completed(futures):
                    task_id, value = future.result()
                    results[task_id] = value
            report.max_observed_workers = observed

        start_ordinal = len(report.execution_order)
        for index, task in enumerate(admitted):
            report.execution_order.append(
                {
                    "ordinal": start_ordinal + index,
                    "task_id": task.task_id,
                    "op_seq": task.op_seq,
                    "token_seq": task.token_seq,
                    "kind": task.kind,
                }
            )

        if report.diagnostics and report.status == "ok":
            report.status = report.diagnostics[0]["code"]
        return results, report

    @staticmethod
    def _add_pending(report: SchedulerReport, task: Task, reason: str) -> None:
        item = {
            "code": E_POINT_BUDGET_PENDING,
            "task_id": task.task_id,
            "op_seq": task.op_seq,
            "token_seq": task.token_seq,
            "cost": task.cost,
            "reason": reason,
        }
        report.pending.append(item)
        if not any(d.get("task_id") == task.task_id and d.get("code") == E_POINT_BUDGET_PENDING for d in report.diagnostics):
            report.diagnostics.append(dict(item))
