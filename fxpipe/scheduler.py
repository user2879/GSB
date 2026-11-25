"""Bounded thread-pool scheduler with point budget and per-operator quota.

Admission (point consumption) happens on the scheduler thread in the stable
order (chain_id, op_seq, token_index), so accounting is identical at every
concurrency level. Only admitted, side-effect-free tasks are fanned out to a
``ThreadPoolExecutor`` whose worker count never exceeds ``max_workers``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from .errors import Code, IssueLog


@dataclass(frozen=True)
class Task:
    chain_id: int
    seq: int
    token_index: int
    cost: int
    payload: dict

    def key(self) -> str:
        return f"c{self.chain_id}op{self.seq}#t{self.token_index}"


@dataclass
class ScheduleResult:
    admitted: list[Task] = field(default_factory=list)
    pending: list[Task] = field(default_factory=list)
    results: list[tuple[Task, object]] = field(default_factory=list)
    consumed_points: int = 0
    quota_aborted: set[str] = field(default_factory=set)


class Scheduler:
    def __init__(self, max_workers: int, point_budget: int, operator_quota: int, log: IssueLog):
        self.max_workers = max(1, max_workers)
        self.point_budget = point_budget
        self.operator_quota = operator_quota
        self.log = log

    def run_batch(
        self,
        tasks: list[Task],
        worker: Callable[[Task], object],
        consumed_already: int = 0,
        quota_already: dict[str, int] | None = None,
        ready: Callable[[Task], bool] | None = None,
    ) -> ScheduleResult:
        ordered = sorted(tasks, key=lambda task: (task.chain_id, task.seq, task.token_index))
        quota_spent = dict(quota_already or {})
        remaining_budget = self.point_budget - consumed_already
        if remaining_budget < 0:
            remaining_budget = 0
        result = ScheduleResult()
        admitted: list[Task] = []
        pending: list[Task] = []

        for task in ordered:
            op_key = f"c{task.chain_id}op{task.seq}"
            if ready is not None and not ready(task):
                pending.append(task)
                continue
            if op_key in result.quota_aborted:
                pending.append(task)
                continue
            spent = quota_spent.get(op_key, 0)
            if spent + task.cost > self.operator_quota:
                result.quota_aborted.add(op_key)
                self.log.add(
                    Code.E_QUOTA,
                    op_key,
                    f"operator quota {self.operator_quota} exceeded at cost {task.cost}",
                )
                pending.append(task)
                continue
            if task.cost > remaining_budget:
                self.log.add(
                    Code.E_BUDGET,
                    "batch",
                    f"point budget exhausted; {task.key()} retained as pending",
                )
                pending.append(task)
                continue
            quota_spent[op_key] = spent + task.cost
            remaining_budget -= task.cost
            result.consumed_points += task.cost
            admitted.append(task)

        # Deterministic fan-out: pure tasks, reassembled in admission order.
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [pool.submit(worker, task) for task in admitted]
            for task, future in zip(admitted, futures):
                result.results.append((task, future.result()))

        result.admitted = admitted
        result.pending = pending
        return result
