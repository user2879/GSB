from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import queue
import threading
import time
from typing import Any, Callable, Iterable

from .errors import LabError


@dataclass(frozen=True)
class TaskRunResult:
    value: Any
    completed_rows: int
    peak_memory_bytes: int


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    kind: str
    operator_id: str
    tables: tuple[str, ...]
    memory_reservation_bytes: int
    dependencies: tuple[str, ...]
    runner: Callable[[dict[str, Any]], TaskRunResult] = field(compare=False, repr=False)


@dataclass(frozen=True)
class TaskTrace:
    task_id: str
    kind: str
    operator_id: str
    tables: tuple[str, ...]
    dependencies: tuple[str, ...]
    memory_reservation_bytes: int
    created_ns: int
    wait_started_ns: int
    start_ns: int
    end_ns: int
    wait_ms: float
    duration_ms: float
    completed_rows: int
    peak_memory_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "kind": self.kind,
            "operator_id": self.operator_id,
            "tables": list(self.tables),
            "dependencies": list(self.dependencies),
            "memory_reservation_bytes": self.memory_reservation_bytes,
            "created_ns": self.created_ns,
            "wait_started_ns": self.wait_started_ns,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "wait_ms": self.wait_ms,
            "duration_ms": self.duration_ms,
            "completed_rows": self.completed_rows,
            "peak_memory_bytes": self.peak_memory_bytes,
        }


@dataclass(frozen=True)
class ScheduleResult:
    values: dict[str, Any]
    traces: tuple[TaskTrace, ...]
    max_global_running: int
    max_memory_in_use_bytes: int
    max_per_table_running: dict[str, int]
    constraints: dict[str, Any]

    def trace_dicts(self) -> list[dict[str, Any]]:
        return [trace.as_dict() for trace in self.traces]


class ConstrainedScheduler:
    """DAG scheduler with global, per-table and memory admission control."""

    def __init__(
        self,
        *,
        global_concurrency: int,
        per_table_quota: int,
        memory_budget_bytes: int,
    ) -> None:
        if global_concurrency < 1:
            raise LabError("E_RESOURCE", "global_concurrency 必须至少为 1", field="tuning.global_concurrency")
        if per_table_quota < 1:
            raise LabError("E_RESOURCE", "per_table_scan_quota 必须至少为 1", field="tuning.per_table_scan_quota")
        if memory_budget_bytes < 1:
            raise LabError("E_RESOURCE", "memory_budget_bytes 必须至少为 1", field="tuning.memory_budget_bytes")
        self.global_concurrency = global_concurrency
        self.per_table_quota = per_table_quota
        self.memory_budget_bytes = memory_budget_bytes

    def run(self, tasks: Iterable[TaskSpec]) -> ScheduleResult:
        task_list = list(tasks)
        task_map = {task.task_id: task for task in task_list}
        if len(task_map) != len(task_list):
            raise LabError("E_SCHED", "任务 ID 重复", field="tasks.task_id")
        for task in task_list:
            unknown = [dependency for dependency in task.dependencies if dependency not in task_map]
            if unknown:
                raise LabError(
                    "E_SCHED",
                    f"任务 {task.task_id} 依赖未知任务 {unknown[0]}",
                    field="tasks.dependencies",
                    context={"task_id": task.task_id},
                )
            if task.memory_reservation_bytes > self.memory_budget_bytes:
                raise LabError(
                    "E_RESOURCE",
                    "单任务内存预留超过全局内存预算，任务永远无法开始",
                    field="tasks.memory_reservation_bytes",
                    context={
                        "task_id": task.task_id,
                        "required": task.memory_reservation_bytes,
                        "budget": self.memory_budget_bytes,
                    },
                )

        lock = threading.Lock()
        done_queue: queue.Queue[tuple[str, BaseException | None]] = queue.Queue()
        values: dict[str, Any] = {}
        failures: dict[str, BaseException] = {}
        active_global = 0
        active_memory = 0
        active_per_table: dict[str, int] = {table: 0 for task in task_list for table in task.tables}
        max_global = 0
        max_memory = 0
        max_per_table = {table: 0 for table in active_per_table}
        created = {task.task_id: time.monotonic_ns() for task in task_list}
        wait_started: dict[str, int] = {}
        start_times: dict[str, int] = {}
        end_times: dict[str, int] = {}
        run_results: dict[str, TaskRunResult] = {}
        submitted: set[str] = set()
        completed: set[str] = set()
        running: set[str] = set()
        pending = set(task_map)

        def record_completion(task_id: str, error: BaseException | None) -> None:
            running.discard(task_id)
            if error is None:
                completed.add(task_id)
                values[task_id] = run_results[task_id].value
            else:
                failures[task_id] = error

        def consume_available(block: bool) -> None:
            received = False
            while True:
                try:
                    task_id, error = done_queue.get(timeout=30.0) if block and not received else done_queue.get_nowait()
                except queue.Empty:
                    return
                received = True
                record_completion(task_id, error)

        def worker(task: TaskSpec, dependency_values: dict[str, Any]) -> None:
            started = time.monotonic_ns()
            with lock:
                start_times[task.task_id] = started
            error: BaseException | None = None
            result: TaskRunResult | None = None
            try:
                result = task.runner(dependency_values)
            except BaseException as exc:  # propagate after accounting release
                error = exc
            finally:
                ended = time.monotonic_ns()
                accounting_error: BaseException | None = None
                try:
                    with lock:
                        nonlocal active_global, active_memory
                        active_global -= 1
                        active_memory -= task.memory_reservation_bytes
                        for table in task.tables:
                            active_per_table[table] -= 1
                        end_times[task.task_id] = ended
                        if result is not None:
                            run_results[task.task_id] = result
                        if error is not None:
                            failures[task.task_id] = error
                except BaseException as exc:
                    accounting_error = exc
                    failures[task.task_id] = exc
                done_queue.put((task.task_id, error or accounting_error))

        executor = ThreadPoolExecutor(
            max_workers=self.global_concurrency,
            thread_name_prefix="joinlab",
        )
        try:
            while pending or running:
                made_progress = False
                for task_id in sorted(tuple(pending)):
                    task = task_map[task_id]
                    if not set(task.dependencies).issubset(completed):
                        continue
                    wait_started.setdefault(task_id, time.monotonic_ns())
                    if any(active_per_table.get(table, 0) >= self.per_table_quota for table in task.tables):
                        continue
                    if active_global >= self.global_concurrency:
                        continue
                    if active_memory + task.memory_reservation_bytes > self.memory_budget_bytes:
                        continue
                    pending.remove(task_id)
                    submitted.add(task_id)
                    running.add(task_id)
                    active_global += 1
                    active_memory += task.memory_reservation_bytes
                    for table in task.tables:
                        active_per_table[table] = active_per_table.get(table, 0) + 1
                    max_global = max(max_global, active_global)
                    max_memory = max(max_memory, active_memory)
                    for table in task.tables:
                        max_per_table[table] = max(max_per_table.get(table, 0), active_per_table[table])
                    dependency_values = {
                        dependency: values[dependency] for dependency in task.dependencies
                    }
                    executor.submit(worker, task, dependency_values)
                    made_progress = True
                if not made_progress:
                    if not running:
                        blocked = sorted(pending)
                        raise LabError(
                            "E_SCHED",
                            "任务 DAG 无法继续，可能存在循环依赖或不可满足约束",
                            field="tasks",
                            context={"blocked_tasks": blocked},
                        )
                    consume_available(block=True)
                else:
                    consume_available(block=False)
                # Consume completions that arrived while submitting.
                consume_available(block=False)
                if failures:
                    first_task = sorted(failures)[0]
                    failure = failures[first_task]
                    if isinstance(failure, LabError):
                        raise failure
                    raise LabError(
                        "E_EXEC",
                        f"任务 {first_task} 执行失败: {failure}",
                        field="tasks.runner",
                        context={"task_id": first_task},
                    ) from failure
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

        if failures:
            first_task = sorted(failures)[0]
            failure = failures[first_task]
            if isinstance(failure, LabError):
                raise failure
            raise LabError(
                "E_EXEC",
                f"任务 {first_task} 执行失败: {failure}",
                field="tasks.runner",
                context={"task_id": first_task},
            ) from failure
        if completed != set(task_map):
            raise LabError("E_SCHED", "调度未完成全部任务", field="tasks")

        traces: list[TaskTrace] = []
        for task in task_list:
            start_ns = start_times[task.task_id]
            end_ns = end_times[task.task_id]
            wait_ns = max(0, start_ns - wait_started[task.task_id])
            result = run_results[task.task_id]
            traces.append(
                TaskTrace(
                    task_id=task.task_id,
                    kind=task.kind,
                    operator_id=task.operator_id,
                    tables=task.tables,
                    dependencies=task.dependencies,
                    memory_reservation_bytes=task.memory_reservation_bytes,
                    created_ns=created[task.task_id],
                    wait_started_ns=wait_started[task.task_id],
                    start_ns=start_ns,
                    end_ns=end_ns,
                    wait_ms=wait_ns / 1_000_000.0,
                    duration_ms=(end_ns - start_ns) / 1_000_000.0,
                    completed_rows=result.completed_rows,
                    peak_memory_bytes=result.peak_memory_bytes,
                )
            )
        traces.sort(key=lambda trace: (trace.start_ns, trace.task_id))
        constraints = {
            "global_concurrency_limit": self.global_concurrency,
            "per_table_quota_limit": self.per_table_quota,
            "memory_budget_bytes": self.memory_budget_bytes,
            "observed_max_global_running": max_global,
            "observed_max_memory_in_use_bytes": max_memory,
            "observed_max_per_table_running": dict(sorted(max_per_table.items())),
            "global_limit_respected": max_global <= self.global_concurrency,
            "table_quota_respected": all(
                value <= self.per_table_quota for value in max_per_table.values()
            ),
            "memory_budget_respected": max_memory <= self.memory_budget_bytes,
        }
        return ScheduleResult(
            values=values,
            traces=tuple(traces),
            max_global_running=max_global,
            max_memory_in_use_bytes=max_memory,
            max_per_table_running=dict(sorted(max_per_table.items())),
            constraints=constraints,
        )
