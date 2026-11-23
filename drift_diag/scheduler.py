"""Real ThreadPoolExecutor scheduling with global and per-workload quotas."""

from __future__ import annotations

import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .config import EvaluationConfig
from .drift import compare_field
from .models import FIELD_INDEX, FIELD_ORDER, TargetSet


@dataclass(frozen=True)
class FieldTask:
    task_id: str
    workload: str
    field: str


def _round_ms(relative_ns: int, origin_ns: int) -> float:
    return round((relative_ns - origin_ns) / 1_000_000.0, 3)


def schedule_decisions(
    target: TargetSet,
    actual_state: dict[str, dict[str, Any]],
    config: EvaluationConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate all fields on worker threads, then return stable sorted results.

    A round submits at most ``global_limit`` tasks. The pool also has that many
    workers. A task only counts as running after it acquires the per-workload
    semaphore, so quota waiters are explicitly visible in the trace.
    """

    config.validate()
    # Field-major ordering keeps rounds diverse across workloads while remaining
    # deterministic. Final decisions are sorted back into workload-major order.
    tasks = [
        FieldTask(
            task_id=f"{workload.name}:{field}",
            workload=workload.name,
            field=field,
        )
        for field in FIELD_ORDER
        for workload in target.ordered_workloads()
    ]

    origin_ns = time.monotonic_ns()
    lock = threading.Lock()
    global_active = 0
    global_peak = 0
    workload_active: Counter[str] = Counter()
    workload_peak: Counter[str] = Counter()
    trajectory: list[dict[str, Any]] = []

    semaphores = {
        name: threading.Semaphore(config.per_workload_quota)
        for name in target.workloads
    }
    target_by_name = target.workloads
    main_thread_ident = threading.get_ident()

    def execute(task: FieldTask, submitted_ns: int, round_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
        nonlocal global_active, global_peak
        worker_start_ns = time.monotonic_ns()
        quota_wait_start_ns = worker_start_ns
        semaphore = semaphores[task.workload]
        semaphore.acquire()
        quota_acquired_ns = time.monotonic_ns()

        with lock:
            global_active += 1
            workload_active[task.workload] += 1
            global_peak = max(global_peak, global_active)
            workload_peak[task.workload] = max(
                workload_peak[task.workload], workload_active[task.workload]
            )
            trajectory.append(
                {
                    "at_ms": _round_ms(quota_acquired_ns, origin_ns),
                    "type": "running",
                    "task_id": task.task_id,
                    "workload": task.workload,
                    "global_active": global_active,
                    "workload_active": workload_active[task.workload],
                }
            )

        execution_start_ns = time.monotonic_ns()
        try:
            if config.task_delay_ms:
                time.sleep(config.task_delay_ms / 1000.0)
            decision = compare_field(
                target_by_name[task.workload],
                task.field,
                actual_state,
                scale_exp=config.scale_exp,
            )
        finally:
            execution_end_ns = time.monotonic_ns()
            with lock:
                trajectory.append(
                    {
                        "at_ms": _round_ms(execution_end_ns, origin_ns),
                        "type": "finished",
                        "task_id": task.task_id,
                        "workload": task.workload,
                        "global_active": global_active - 1,
                        "workload_active": workload_active[task.workload] - 1,
                    }
                )
                global_active -= 1
                workload_active[task.workload] -= 1
            semaphore.release()

        assignment = {
            "task_id": task.task_id,
            "workload": task.workload,
            "field": task.field,
            "round": round_index,
            "worker_thread": threading.current_thread().name,
            "worker_ident": threading.get_ident(),
            "main_thread_ident": main_thread_ident,
            "queue_wait_start_ms": _round_ms(submitted_ns, origin_ns),
            "queue_wait_end_ms": _round_ms(worker_start_ns, origin_ns),
            "queue_wait_ms": round((worker_start_ns - submitted_ns) / 1_000_000.0, 3),
            "quota_wait_start_ms": _round_ms(quota_wait_start_ns, origin_ns),
            "quota_wait_end_ms": _round_ms(quota_acquired_ns, origin_ns),
            "quota_wait_ms": round((quota_acquired_ns - quota_wait_start_ns) / 1_000_000.0, 3),
            "execution_start_ms": _round_ms(execution_start_ns, origin_ns),
            "execution_end_ms": _round_ms(execution_end_ns, origin_ns),
            "execution_ms": round((execution_end_ns - execution_start_ns) / 1_000_000.0, 3),
        }
        return decision, assignment

    rounds: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=config.global_limit, thread_name_prefix="drift-judge") as pool:
        for round_index, start in enumerate(range(0, len(tasks), config.global_limit), start=1):
            round_tasks = tasks[start : start + config.global_limit]
            submitted: list[tuple[FieldTask, int]] = []
            futures: list[Any] = []
            for task in round_tasks:
                submitted_ns = time.monotonic_ns()
                submitted.append((task, submitted_ns))
                futures.append(pool.submit(execute, task, submitted_ns, round_index))

            assignments: list[dict[str, Any]] = []
            first_error: BaseException | None = None
            for future in futures:
                try:
                    decision, assignment = future.result()
                except BaseException as exc:  # pragma: no cover - exercised through wrappers
                    if first_error is None:
                        first_error = exc
                    continue
                results.append(decision)
                assignments.append(assignment)
            if first_error is not None:
                raise first_error

            rounds.append(
                {
                    "round": round_index,
                    "assignments": assignments,
                    "global_active_peak_so_far": global_peak,
                    "per_workload_active_peak_so_far": {
                        name: workload_peak[name] for name in sorted(target.workloads)
                    },
                }
            )

    results.sort(key=lambda decision: (decision["workload"], FIELD_INDEX[decision["field"]]))
    trace = {
        "main_thread_ident": main_thread_ident,
        "global_limit": config.global_limit,
        "per_workload_quota": config.per_workload_quota,
        "task_delay_ms": config.task_delay_ms,
        "rounds": rounds,
        "concurrency_trajectory": trajectory,
        "global_active_peak": global_peak,
        "per_workload_active_peak": {
            name: workload_peak[name] for name in sorted(target.workloads)
        },
        "assignments_all_on_workers": all(
            assignment["worker_ident"] != main_thread_ident
            for round_info in rounds
            for assignment in round_info["assignments"]
        ),
        "quota_respected": all(
            peak <= config.per_workload_quota
            for peak in (workload_peak[name] for name in sorted(target.workloads))
        ),
        "global_limit_respected": global_peak <= config.global_limit,
    }
    return deepcopy(results), trace
