"""盖章/验章任务的并行调度与预算控制。

提供两套共享同一份轨迹数据结构的实现：

* plan_virtual_schedule：纯确定性虚拟时钟调度。任务时长由整数种子
  派生，全局并发上限与每后端配额同时生效，等待开始/结束均以单调
  虚拟毫秒记录，轨迹完全可复算，与线程完成顺序无关；
* run_threaded：真实线程池执行。单一协调线程按同一准入规则派发，
  实际执行盖章/验章并以 time.monotonic_ns 记录等待区间，用于演示
  真实并行下的预算控制。
"""

from __future__ import annotations

import heapq
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

from .prng import mix_seed


@dataclass(frozen=True)
class Task:
    task_id: str
    backend_id: str
    kind: str  # "stamp" 或 "verify"
    duration_ms: int
    order: int  # 提交顺序，用于稳定平局判定


@dataclass(frozen=True)
class TraceEntry:
    task_id: str
    backend_id: str
    kind: str
    wait_start_ms: int
    wait_end_ms: int
    run_start_ms: int
    run_end_ms: int
    wait_ms: int
    order: int


def deterministic_duration(task_id: str, backend_id: str, kind: str,
                           master_seed: int, min_ms: int = 5,
                           max_ms: int = 40) -> int:
    """由任务标识派生确定性时长，保证调度轨迹稳定。"""
    kind_code = 1 if kind == "stamp" else 2
    seed = mix_seed(master_seed, kind_code,
                    sum(task_id.encode("utf-8")),
                    sum(backend_id.encode("utf-8")))
    span = max_ms - min_ms + 1
    return min_ms + seed % span


def plan_virtual_schedule(
    tasks: list[Task],
    global_limit: int,
    per_backend_limit: int,
) -> list[TraceEntry]:
    """虚拟时钟调度，返回按提交顺序排列的稳定轨迹。"""
    if global_limit <= 0 or per_backend_limit <= 0:
        raise ValueError("并发上限必须为正整数")

    queues: dict[str, deque[Task]] = {}
    for task in sorted(tasks, key=lambda t: t.order):
        queues.setdefault(task.backend_id, deque()).append(task)

    running: list[tuple[int, int, Task]] = []  # (结束时间, 准入序号, 任务)
    in_flight = 0
    per_backend_running: dict[str, int] = {}
    clock_ms = 0
    admission_counter = 0
    traces: dict[str, TraceEntry] = {}

    def admit() -> bool:
        nonlocal in_flight, admission_counter
        for backend_id in sorted(queues):
            queue = queues[backend_id]
            if not queue:
                continue
            if in_flight >= global_limit:
                return False
            if per_backend_running.get(backend_id, 0) >= per_backend_limit:
                continue
            task = queue.popleft()
            run_start = clock_ms
            run_end = run_start + task.duration_ms
            traces[task.task_id] = TraceEntry(
                task_id=task.task_id,
                backend_id=task.backend_id,
                kind=task.kind,
                wait_start_ms=0,
                wait_end_ms=run_start,
                run_start_ms=run_start,
                run_end_ms=run_end,
                wait_ms=run_start,
                order=task.order,
            )
            heapq.heappush(running, (run_end, admission_counter, task))
            admission_counter += 1
            in_flight += 1
            per_backend_running[backend_id] = \
                per_backend_running.get(backend_id, 0) + 1
            return True
        return False

    while any(queues.values()) or running:
        admitted = True
        while admitted:
            admitted = admit()
        if not running:
            break
        end_ms, _, finished = heapq.heappop(running)
        clock_ms = end_ms
        in_flight -= 1
        per_backend_running[finished.backend_id] -= 1

    return [traces[task.task_id]
            for task in sorted(tasks, key=lambda t: t.order)]


@dataclass
class _Budget:
    global_limit: int
    per_backend_limit: int
    in_flight: int = 0
    per_backend: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.per_backend is None:
            self.per_backend = {}


def run_threaded(
    tasks: list[Task],
    worker: Callable[[Task], None],
    global_limit: int,
    per_backend_limit: int,
    *,
    max_workers: int | None = None,
) -> list[TraceEntry]:
    """真实线程执行，单协调线程保证准入顺序确定、预算不超限。"""
    if global_limit <= 0 or per_backend_limit <= 0:
        raise ValueError("并发上限必须为正整数")

    ordered = sorted(tasks, key=lambda t: t.order)
    queue: deque[Task] = deque(ordered)
    budget = _Budget(global_limit, per_backend_limit)
    condition = threading.Condition()
    traces: dict[str, TraceEntry] = {}
    pool_size = max_workers or global_limit
    semaphore = threading.Semaphore(pool_size)

    def execute(task: Task, wait_start_ms: int) -> None:
        run_start = time.monotonic_ns() // 1_000_000
        try:
            worker(task)
        finally:
            run_end = time.monotonic_ns() // 1_000_000
            entry = TraceEntry(
                task_id=task.task_id,
                backend_id=task.backend_id,
                kind=task.kind,
                wait_start_ms=wait_start_ms,
                wait_end_ms=run_start,
                run_start_ms=run_start,
                run_end_ms=run_end,
                wait_ms=run_start - wait_start_ms,
                order=task.order,
            )
            with condition:
                traces[task.task_id] = entry
                budget.in_flight -= 1
                budget.per_backend[task.backend_id] -= 1
                condition.notify_all()
            semaphore.release()

    def coordinator() -> None:
        threads: list[threading.Thread] = []
        while queue:
            with condition:
                while True:
                    candidate_index = -1
                    for index, task in enumerate(queue):
                        backend_count = budget.per_backend.get(
                            task.backend_id, 0)
                        if (budget.in_flight < global_limit
                                and backend_count < per_backend_limit):
                            candidate_index = index
                            break
                    if candidate_index >= 0:
                        break
                    if not queue:
                        break
                    condition.wait(timeout=0.01)
                if not queue:
                    break
                task = queue[candidate_index]
                del queue[candidate_index]
                budget.in_flight += 1
                budget.per_backend[task.backend_id] = \
                    budget.per_backend.get(task.backend_id, 0) + 1
            semaphore.acquire()
            wait_start = time.monotonic_ns() // 1_000_000
            thread = threading.Thread(
                target=execute, args=(task, wait_start),
                name=f"worker-{task.task_id}", daemon=True,
            )
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()

    coordinator()
    return [traces[task.task_id] for task in ordered]
