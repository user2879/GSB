"""采集任务生成、分批调度与两版观测配对（需求 4、6）。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .fixedpoint import real_values
from .manifest import Manifest, Registry
from .nodepool import NodePool
from .scheduler import CollectTask, Scheduler, TaskResult

BASELINE_SIDE = "baseline"
CANDIDATE_SIDE = "candidate"


@dataclass(frozen=True)
class SessionParams:
    sample_seed: int
    precision: int
    global_concurrency: int = 3
    per_node_concurrency: int = 2
    socket_timeout_ms: int = 1500
    max_attempts: int = 3
    retry_backoff_ms: int = 25
    batch_size: int = 8
    max_elapsed_delta_ms: int = 50
    max_mismatch_nodes: int = 0
    strict_timing: bool = False


@dataclass(frozen=True)
class Observation:
    version: int
    item: str
    node_id: str
    summary: dict[str, Any]
    elapsed_ms: int
    attempts: int
    retry_count: int
    global_wait_ms: int
    node_wait_ms: int


@dataclass
class CollectionOutcome:
    observations: dict[tuple[int, str, str], Observation] = field(default_factory=dict)
    failed_tasks: list[TaskResult] = field(default_factory=list)
    scheduler_records: list[dict[str, Any]] = field(default_factory=list)


def make_tasks(manifest: Manifest, registry: Registry, side: str, sample_seed: int) -> list[CollectTask]:
    values = real_values(manifest)
    tasks: list[CollectTask] = []
    for key in manifest.keys():
        item = manifest.items[key]
        for node_id in sorted(item.targets):
            task_key = f"{side}:v{manifest.version}:{key}:{node_id}"
            payload = {
                "type": "slice",
                "node_id": node_id,
                "version": manifest.version,
                "side": side,
                "item": key,
                "value": values[key],
                "targets": list(item.targets),
                "sample_seed": sample_seed,
                "work_repeat": item.work_repeat,
                "extra_sleep_ms": item.extra_sleep_ms,
            }
            tasks.append(CollectTask(task_key=task_key, node=registry.nodes[node_id], payload=payload))
    return tasks


def build_task_plan(
    baseline: Manifest,
    candidate: Manifest,
    registry: Registry,
    params: SessionParams,
) -> list[CollectTask]:
    tasks = make_tasks(baseline, registry, BASELINE_SIDE, params.sample_seed)
    tasks.extend(make_tasks(candidate, registry, CANDIDATE_SIDE, params.sample_seed))
    return sorted(tasks, key=lambda task: task.task_key)


def split_batches(tasks: list[CollectTask], batch_size: int) -> list[list[CollectTask]]:
    if batch_size < 1:
        raise ValueError("batch_size 必须为正整数")
    return [tasks[index : index + batch_size] for index in range(0, len(tasks), batch_size)]


def split_item_atomic_batches(tasks: list[CollectTask], batch_size: int) -> list[list[CollectTask]]:
    """按配置项分组切批，保证同一配置项的两版、所有节点观测同批提交。

    账本只能按完整批次恢复，因此一个配置项的基线与候选观测绝不能跨批；
    批大小约束的是“配置项数量”，批内任务按任务键稳定排序。
    """
    if batch_size < 1:
        raise ValueError("batch_size 必须为正整数")
    grouped: dict[str, list[CollectTask]] = {}
    for task in tasks:
        item = task.payload["item"]
        grouped.setdefault(item, []).append(task)
    batches: list[list[CollectTask]] = []
    current: list[CollectTask] = []
    current_items = 0
    for item in sorted(grouped):
        item_tasks = sorted(grouped[item], key=lambda task: task.task_key)
        if current and current_items >= batch_size:
            batches.append(current)
            current = []
            current_items = 0
        current.extend(item_tasks)
        current_items += 1
    if current:
        batches.append(current)
    return batches


def run_batch(
    scheduler: Scheduler,
    tasks: list[CollectTask],
    progress: Any | None = None,
) -> CollectionOutcome:
    outcome = CollectionOutcome()
    for result in scheduler.run(tasks, progress=progress):
        record = {
            "task_key": result.task_key,
            "node_id": result.node_id,
            "enqueued_monotonic_ms": result.enqueued_monotonic_ms,
            "wait_started_ms": result.wait_started_ms,
            "wait_ended_ms": result.wait_ended_ms,
            "global_wait_ms": result.global_wait_ms,
            "node_wait_ms": result.node_wait_ms,
            "attempts": [
                {
                    "attempt_no": attempt.attempt_no,
                    "request_id": attempt.request_id,
                    "started_monotonic_ms": attempt.started_monotonic_ms,
                    "ended_monotonic_ms": attempt.ended_monotonic_ms,
                    "socket_timeout_ms": attempt.socket_timeout_ms,
                    "outcome": attempt.outcome,
                    "error": attempt.error,
                }
                for attempt in result.attempts
            ],
        }
        outcome.scheduler_records.append(record)
        if not result.ok or result.response is None:
            outcome.failed_tasks.append(result)
            continue
        response = result.response
        observation = Observation(
            version=int(response["version"]),
            item=str(response["item"]),
            node_id=str(response["node_id"]),
            summary=dict(response["summary"]),
            elapsed_ms=int(response["elapsed_ms"]),
            attempts=len(result.attempts),
            retry_count=len(result.attempts) - 1,
            global_wait_ms=result.global_wait_ms,
            node_wait_ms=result.node_wait_ms,
        )
        outcome.observations[(observation.version, observation.item, observation.node_id)] = observation
    return outcome


def collect_all(
    baseline: Manifest,
    candidate: Manifest,
    registry: Registry,
    pool: NodePool,
    params: SessionParams,
    progress: Any | None = None,
) -> tuple[list[CollectionOutcome], list[list[CollectTask]]]:
    del pool
    plan = build_task_plan(baseline, candidate, registry, params)
    batches = split_item_atomic_batches(plan, params.batch_size)
    scheduler = Scheduler(
        global_concurrency=params.global_concurrency,
        per_node_concurrency=params.per_node_concurrency,
        socket_timeout_ms=params.socket_timeout_ms,
        max_attempts=params.max_attempts,
        retry_backoff_ms=params.retry_backoff_ms,
    )
    outcomes = [run_batch(scheduler, batch, progress) for batch in batches]
    return outcomes, batches
