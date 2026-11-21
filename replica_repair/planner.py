"""Budget-constrained deterministic repair planning and execution."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping

from .errors import BudgetError, IntegrityError, StateError
from .node_service import NodeClient
from .truth import BlockTruth, TruthReport
from .util import ceil_div, ensure_int


@dataclass(frozen=True)
class RepairBudget:
    max_parallel: int
    per_node_kb: int
    cost_per_kb: int = 1

    def validate(self) -> "RepairBudget":
        if not ensure_int(self.max_parallel) or self.max_parallel <= 0:
            raise BudgetError(
                "max_parallel must be a positive integer",
                path="$.budget.max_parallel",
                details={"actual": self.max_parallel},
            )
        if not ensure_int(self.per_node_kb) or self.per_node_kb <= 0:
            raise BudgetError(
                "per_node_kb must be a positive integer",
                path="$.budget.per_node_kb",
                details={"actual": self.per_node_kb},
            )
        if not ensure_int(self.cost_per_kb) or self.cost_per_kb <= 0:
            raise BudgetError(
                "cost_per_kb must be a positive integer",
                path="$.budget.cost_per_kb",
                details={"actual": self.cost_per_kb},
            )
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_parallel": self.max_parallel,
            "per_node_kb": self.per_node_kb,
            "cost_per_kb": self.cost_per_kb,
            "quota_scope": "per_wave",
        }


@dataclass(frozen=True)
class RepairTask:
    task_id: str
    wave: int
    block_id: str
    source_node: str
    target_node: str
    size_bytes: int
    repair_kb: int
    repair_weight: int
    source_priority: int
    target_priority: int
    cost_units: int
    status: str = "planned"
    error_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "wave": self.wave,
            "block_id": self.block_id,
            "source_node": self.source_node,
            "target_node": self.target_node,
            "size_bytes": self.size_bytes,
            "repair_kb": self.repair_kb,
            "repair_weight": self.repair_weight,
            "source_priority": self.source_priority,
            "target_priority": self.target_priority,
            "cost_units": self.cost_units,
            "status": self.status,
            "error_code": self.error_code,
        }


@dataclass(frozen=True)
class QueueEvent:
    task_id: str
    wave: int
    target_node: str
    repair_kb: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "wave": self.wave,
            "target_node": self.target_node,
            "repair_kb": self.repair_kb,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RepairPlan:
    group_id: str
    manifest_version: int
    budget: RepairBudget
    tasks: tuple[RepairTask, ...]
    queue_events: tuple[QueueEvent, ...]
    waves: tuple[tuple[str, ...], ...]
    total_repair_kb: int
    total_cost_units: int
    per_node_repair_kb: tuple[tuple[str, int], ...]
    max_parallel_observed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "manifest_version": self.manifest_version,
            "budget": self.budget.to_dict(),
            "tasks": [task.to_dict() for task in self.tasks],
            "queue_events": [event.to_dict() for event in self.queue_events],
            "waves": [list(wave) for wave in self.waves],
            "total_repair_kb": self.total_repair_kb,
            "total_cost_units": self.total_cost_units,
            "per_node_repair_kb": {node_id: kb for node_id, kb in self.per_node_repair_kb},
            "max_parallel_observed": self.max_parallel_observed,
        }

    @property
    def task_count(self) -> int:
        return len(self.tasks)


def plan_repairs(
    manifest,
    truth: TruthReport,
    budget: RepairBudget,
) -> RepairPlan:
    budget.validate()
    priorities = {
        node.node_id: (node.priority if node.priority is not None else 1)
        for node in manifest.nodes
    }
    candidates: list[dict[str, Any]] = []
    manifest_nodes = manifest.node_map()
    for block in truth.blocks:
        source = _select_source(block, priorities)
        for mismatch in block.mismatches:
            if mismatch.node_id == source:
                raise StateError(
                    "truth source cannot also be a mismatch target",
                    details={"block_id": block.block_id, "node_id": source},
                )
            repair_kb = ceil_div(block.size_bytes, 1024)
            if repair_kb > budget.per_node_kb:
                raise BudgetError(
                    "a single repair task exceeds the per-node quota and can never be scheduled",
                    path="$.budget.per_node_kb",
                    details={
                        "block_id": block.block_id,
                        "target_node": mismatch.node_id,
                        "required_kb": repair_kb,
                        "quota_kb": budget.per_node_kb,
                    },
                )
            candidates.append(
                {
                    "block_id": block.block_id,
                    "source_node": source,
                    "target_node": mismatch.node_id,
                    "size_bytes": block.size_bytes,
                    "repair_kb": repair_kb,
                    "repair_weight": manifest_nodes[mismatch.node_id]
                    .block_map()[block.block_id]
                    .repair_weight,
                    "source_priority": priorities[source],
                    "target_priority": priorities[mismatch.node_id],
                }
            )

    candidates.sort(
        key=lambda item: (
            -int(item["repair_weight"]),
            str(item["block_id"]),
            str(item["target_node"]),
            str(item["source_node"]),
        )
    )
    base_tasks = [
        RepairTask(
            task_id=f"rep-{index:04d}",
            wave=0,
            block_id=item["block_id"],
            source_node=item["source_node"],
            target_node=item["target_node"],
            size_bytes=item["size_bytes"],
            repair_kb=item["repair_kb"],
            repair_weight=item["repair_weight"],
            source_priority=item["source_priority"],
            target_priority=item["target_priority"],
            cost_units=item["repair_kb"] * budget.cost_per_kb,
        )
        for index, item in enumerate(candidates, start=1)
    ]

    pending = list(base_tasks)
    scheduled: list[RepairTask] = []
    queue_events: list[QueueEvent] = []
    wave_task_ids: list[tuple[str, ...]] = []
    wave_number = 1
    while pending:
        wave_tasks: list[RepairTask] = []
        deferred: list[RepairTask] = []
        node_used: dict[str, int] = {}
        for task in pending:
            quota_used = node_used.get(task.target_node, 0)
            quota_fits = quota_used + task.repair_kb <= budget.per_node_kb
            concurrency_fits = len(wave_tasks) < budget.max_parallel
            if quota_fits and concurrency_fits:
                scheduled_task = replace(task, wave=wave_number)
                wave_tasks.append(scheduled_task)
                node_used[task.target_node] = quota_used + task.repair_kb
            else:
                deferred.append(task)
                reason = "per_node_quota" if not quota_fits else "max_parallel"
                queue_events.append(
                    QueueEvent(
                        task_id=task.task_id,
                        wave=wave_number,
                        target_node=task.target_node,
                        repair_kb=task.repair_kb,
                        reason=reason,
                    )
                )
        if not wave_tasks:
            raise BudgetError(
                "repair scheduler cannot make progress under the supplied budget",
                path="$.budget",
                details={"pending_task_ids": [task.task_id for task in pending]},
            )
        scheduled.extend(wave_tasks)
        wave_task_ids.append(tuple(task.task_id for task in wave_tasks))
        pending = deferred
        wave_number += 1

    per_node: dict[str, int] = {}
    for task in scheduled:
        per_node[task.target_node] = per_node.get(task.target_node, 0) + task.repair_kb
    total_repair_kb = sum(task.repair_kb for task in scheduled)
    total_cost = sum(task.cost_units for task in scheduled)
    max_parallel = max((len(wave) for wave in wave_task_ids), default=0)
    if max_parallel > budget.max_parallel:
        raise BudgetError("scheduler exceeded global concurrency budget", path="$.budget.max_parallel")
    for wave_ids in wave_task_ids:
        wave_tasks = [task for task in scheduled if task.task_id in set(wave_ids)]
        per_target: dict[str, int] = {}
        for task in wave_tasks:
            per_target[task.target_node] = per_target.get(task.target_node, 0) + task.repair_kb
        for target, kb in per_target.items():
            if kb > budget.per_node_kb:
                raise BudgetError(
                    "scheduler exceeded per-node bandwidth quota",
                    path="$.budget.per_node_kb",
                    details={"wave": wave_task_ids.index(wave_ids) + 1, "node_id": target, "kb": kb},
                )

    return RepairPlan(
        group_id=truth.group_id,
        manifest_version=truth.manifest_version,
        budget=budget,
        tasks=tuple(scheduled),
        queue_events=tuple(
            sorted(queue_events, key=lambda item: (item.wave, item.task_id, item.reason))
        ),
        waves=tuple(wave_task_ids),
        total_repair_kb=total_repair_kb,
        total_cost_units=total_cost,
        per_node_repair_kb=tuple(sorted(per_node.items())),
        max_parallel_observed=max_parallel,
    )


def _select_source(block: BlockTruth, priorities: Mapping[str, int]) -> str:
    if not block.truth_nodes:
        raise IntegrityError(
            "truth block has no source node",
            details={"block_id": block.block_id},
        )
    return sorted(block.truth_nodes, key=lambda node_id: (-priorities[node_id], node_id))[0]


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    block_id: str
    source_node: str
    target_node: str
    repair_kb: int
    digest: str
    bytes_written: int
    status: str = "completed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "block_id": self.block_id,
            "source_node": self.source_node,
            "target_node": self.target_node,
            "repair_kb": self.repair_kb,
            "digest": self.digest,
            "bytes_written": self.bytes_written,
            "status": self.status,
        }


@dataclass(frozen=True)
class RepairExecutionResult:
    plan: RepairPlan
    results: tuple[TaskResult, ...]
    completed_tasks: int
    total_repair_kb: int
    total_cost_units: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "results": [result.to_dict() for result in self.results],
            "completed_tasks": self.completed_tasks,
            "total_repair_kb": self.total_repair_kb,
            "total_cost_units": self.total_cost_units,
        }


ProgressHook = Callable[[TaskResult, int], None]


class RepairExecutor:
    def __init__(
        self,
        manifest,
        clients: Mapping[str, NodeClient],
        *,
        progress_hook: ProgressHook | None = None,
        after_progress: Callable[[int], None] | None = None,
    ) -> None:
        self.manifest = manifest
        self.clients = clients
        self.progress_hook = progress_hook
        self.after_progress = after_progress

    def execute(self, plan: RepairPlan) -> RepairExecutionResult:
        if self.progress_hook is None:
            raise StateError("repair execution requires a progress ledger callback")
        task_map = {task.task_id: task for task in plan.tasks}
        if len(task_map) != len(plan.tasks):
            raise StateError("repair plan contains duplicate task ids")
        results: list[TaskResult] = []
        completed = 0
        for wave in plan.waves:
            wave_tasks = [task_map[task_id] for task_id in wave]
            if len(wave_tasks) > plan.budget.max_parallel:
                raise BudgetError(
                    "wave exceeds global concurrency budget",
                    path="$.budget.max_parallel",
                    details={"wave_task_count": len(wave_tasks)},
                )
            with ThreadPoolExecutor(max_workers=len(wave_tasks)) as pool:
                wave_results = list(pool.map(self._execute_task, wave_tasks))
            for result in sorted(wave_results, key=lambda item: item.task_id):
                completed += 1
                self.progress_hook(result, completed)
                if self.after_progress is not None:
                    self.after_progress(completed)
                results.append(result)
        results.sort(key=lambda item: item.task_id)
        return RepairExecutionResult(
            plan=plan,
            results=tuple(results),
            completed_tasks=completed,
            total_repair_kb=sum(result.repair_kb for result in results),
            total_cost_units=plan.total_cost_units,
        )

    def _execute_task(self, task: RepairTask) -> TaskResult:
        if task.source_node not in self.clients or task.target_node not in self.clients:
            raise StateError(
                "repair task references a node without a client",
                details={"task_id": task.task_id},
            )
        source_truth = None
        for node in self.manifest.nodes:
            if node.node_id == task.source_node:
                source_truth = node.block_map().get(task.block_id)
        if source_truth is None or source_truth.digest is None:
            raise IntegrityError(
                "repair source block metadata is unavailable",
                details={"task_id": task.task_id},
            )
        data = self.clients[task.source_node].fetch_block_bytes(
            task.block_id,
            task.size_bytes,
            source_truth.digest,
        )
        response = self.clients[task.target_node].put_block(
            task.block_id,
            data,
            source_truth.digest,
        )
        if response.get("digest") != source_truth.digest:
            raise IntegrityError(
                "target did not confirm repaired digest",
                details={"task_id": task.task_id},
            )
        return TaskResult(
            task_id=task.task_id,
            block_id=task.block_id,
            source_node=task.source_node,
            target_node=task.target_node,
            repair_kb=task.repair_kb,
            digest=source_truth.digest,
            bytes_written=len(data),
        )
