"""Deterministic budget constrained scheduler.

The planner makes every ordering/budget decision on a stable task list. Worker
threads therefore cannot influence artifact order, ledger order or the budget
deduction sequence even when they finish out of order.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .errors import E_ARTIFACT_BUDGET_PENDING, E_STAGE_BUDGET_EXCEEDED
from .models import LedgerEntry


@dataclass(frozen=True)
class Task:
    stage_index: int
    stage_name: str
    scope: str
    artifact_id: str
    points: int
    sort_key: str
    params: dict
    upstream: Tuple[str, ...]
    expected_param_version: int
    candidate: Any = None
    decision: Any = None
    plan: Any = None


@dataclass
class TaskOutcome:
    task: Task
    status: str
    artifact: Optional[Any]
    points_after: int
    stage_points_after: int
    reason: str
    receipt_version: Optional[int]
    digest_match: Optional[bool]
    degraded: bool
    degradation_basis: Optional[str]


@dataclass
class BatchResult:
    outcomes: Tuple[TaskOutcome, ...]
    events: Tuple[dict, ...]
    pending: Tuple[Task, ...]
    aborted_stages: Tuple[str, ...]
    batch_points_used: int
    stage_points_used: Dict[str, int]


def stable_sort_key(stage_index: int, stage_name: str, scope: str, sample_order: Dict[str, int]) -> str:
    if scope.startswith("sample:"):
        sample_id = scope.split(":", 1)[1]
        sample_rank = sample_order[sample_id]
        kind = 1
    else:
        sample_rank = -1
        kind = 0
    return f"{stage_index:04d}:{kind}:{sample_rank:08d}:{stage_name}:{scope}"


class Scheduler:
    def __init__(self, batch_points: int, stage_quotas: Dict[str, int], max_workers: int = 1):
        self.batch_points = batch_points
        self.stage_quotas = dict(stage_quotas)
        self.max_workers = max(1, max_workers)
        self.stage_used: Dict[str, int] = {stage: 0 for stage in stage_quotas}

    def run_batch(
        self,
        tasks: Tuple[Task, ...],
        carry_pending: Tuple[Task, ...] = (),
        execute: Optional[Callable[[Task], Any]] = None,
    ) -> BatchResult:
        queued = list(carry_pending) + [task for task in tasks if task not in carry_pending]
        queued.sort(key=lambda task: task.sort_key)
        outcomes: List[TaskOutcome] = []
        events: List[dict] = []
        pending: List[Task] = []
        aborted: List[str] = []
        batch_used = 0
        runnable: List[Task] = []
        for task in queued:
            if task.stage_name in aborted:
                pending.append(task)
                continue
            quota = self.stage_quotas.get(task.stage_name)
            projected_stage = self.stage_used.get(task.stage_name, 0) + task.points
            if quota is not None and projected_stage > quota:
                aborted.append(task.stage_name)
                events.append(
                    {
                        "code": E_STAGE_BUDGET_EXCEEDED,
                        "artifact_id": task.artifact_id,
                        "stage": task.stage_name,
                        "sort_key": task.sort_key,
                        "used": self.stage_used.get(task.stage_name, 0),
                        "quota": quota,
                    }
                )
                outcomes.append(
                    TaskOutcome(
                        task,
                        "STAGE_ABORTED",
                        None,
                        batch_used,
                        self.stage_used.get(task.stage_name, 0),
                        f"stage quota {quota} would be exceeded",
                        None,
                        None,
                        False,
                        None,
                    )
                )
                pending.append(task)
                continue
            if batch_used + task.points > self.batch_points:
                events.append(
                    {
                        "code": E_ARTIFACT_BUDGET_PENDING,
                        "artifact_id": task.artifact_id,
                        "stage": task.stage_name,
                        "sort_key": task.sort_key,
                        "batch_used": batch_used,
                        "batch_points": self.batch_points,
                    }
                )
                outcomes.append(
                    TaskOutcome(
                        task,
                        "PENDING",
                        None,
                        batch_used,
                        self.stage_used.get(task.stage_name, 0),
                        f"batch budget {self.batch_points} insufficient",
                        None,
                        None,
                        False,
                        None,
                    )
                )
                pending.append(task)
                continue
            batch_used += task.points
            self.stage_used[task.stage_name] = projected_stage
            runnable.append(task)
        artifacts: Dict[str, Any] = {}
        if execute is not None and runnable:
            if self.max_workers == 1:
                for task in runnable:
                    artifacts[task.artifact_id] = execute(task)
            else:
                with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                    futures = {pool.submit(execute, task): task for task in runnable}
                    for future in futures:
                        task = futures[future]
                        artifacts[task.artifact_id] = future.result()
        for task in runnable:
            artifact = artifacts.get(task.artifact_id)
            status = "REUSED" if artifact is None else "DONE"
            outcomes.append(
                TaskOutcome(
                    task,
                    status,
                    artifact,
                    batch_used,
                    self.stage_used.get(task.stage_name, 0),
                    task.decision.reason,
                    task.decision.receipt_version,
                    task.decision.digest_match,
                    task.decision.degraded,
                    task.decision.degradation_basis,
                )
            )
        outcomes.sort(key=lambda item: item.task.sort_key)
        events.sort(key=lambda event: (event["sort_key"], event["code"]))
        return BatchResult(
            tuple(outcomes),
            tuple(events),
            tuple(sorted(pending, key=lambda task: task.sort_key)),
            tuple(sorted(aborted)),
            batch_used,
            dict(self.stage_used),
        )


def outcomes_to_ledger(outcomes: Tuple[TaskOutcome, ...], start_order: int = 0) -> Tuple[LedgerEntry, ...]:
    entries = []
    for index, outcome in enumerate(outcomes):
        status_map = {
            "DONE": "RECOMPUTE",
            "REUSED": "REUSE",
            "PENDING": "PENDING",
            "STAGE_ABORTED": "STAGE_ABORTED",
        }
        entries.append(
            LedgerEntry(
                order=start_order + index,
                sort_key=outcome.task.sort_key,
                artifact_id=outcome.task.artifact_id,
                stage=outcome.task.stage_name,
                scope=outcome.task.scope,
                decision=status_map[outcome.status],
                reason=outcome.reason,
                receipt_version=outcome.receipt_version,
                digest_match=outcome.digest_match,
                degraded=outcome.degraded,
                degradation_basis=outcome.degradation_basis,
                upstream=outcome.task.upstream,
                params=outcome.task.params,
                points=outcome.task.points,
                points_after=outcome.points_after,
                stage_points_after=outcome.stage_points_after,
            )
        )
    return tuple(entries)
