"""三重预算约束下的块级并行扫描调度。

约束同时生效，且所有准入判断都在同一把调度锁内完成，杜绝“先运行后记账”
绕过预算：

1. 全局并行度 ``global_parallelism``：任一时刻运行任务数不得超限；
2. 每段并发配额 ``per_segment_concurrency``：同一段运行任务数不得超限；
3. 每列解码内存预算 ``per_column_memory``：同一列正在解码的内存占用之和
   不得超限；预算为 0 时直接 ``E_BUDGET``。

任务按 (行组, 列, 块序号) 稳定排序后依次尝试准入；无法满足时排队等待，
并以 :class:`time.monotonic_ns` 记录排队开始/结束与运行开始/结束的毫秒数。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from .codecs import block_summary
from .errors import BudgetError
from .segment import Segment
from .versioning import BudgetParams

TASK_ID_PREFIX = "task"


@dataclass
class BlockTask:
    task_id: str
    segment_id: str
    row_group: int
    column: str
    block_index: int
    memory_cost: int          # 解码内存占用（字节），取块 pre_bytes
    encoding: str
    row_count: int
    status: str = "QUEUED"
    queue_start_ms: int = 0
    wait_end_ms: int = 0
    run_start_ms: int = 0
    run_end_ms: int = 0
    wait_ms: int = 0
    run_ms: int = 0
    rows_completed: int = 0
    result_digest: str = ""
    error_code: str | None = None

    def key(self) -> tuple:
        return (self.segment_id, self.row_group, self.column,
                self.block_index)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id, "segment_id": self.segment_id,
            "row_group": self.row_group, "column": self.column,
            "block_index": self.block_index, "memory_cost": self.memory_cost,
            "encoding": self.encoding, "row_count": self.row_count,
            "status": self.status, "queue_start_ms": self.queue_start_ms,
            "wait_end_ms": self.wait_end_ms,
            "run_start_ms": self.run_start_ms,
            "run_end_ms": self.run_end_ms, "wait_ms": self.wait_ms,
            "run_ms": self.run_ms, "rows_completed": self.rows_completed,
            "result_digest": self.result_digest,
            "error_code": self.error_code,
        }


@dataclass
class ScheduleResult:
    tasks: list[BlockTask] = field(default_factory=list)
    max_running_observed: int = 0
    max_segment_running: dict[str, int] = field(default_factory=dict)
    max_column_memory: dict[str, int] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)

    def statistics(self) -> dict:
        return {
            "task_count": len(self.tasks),
            "completed": sum(t.status == "COMPLETED" for t in self.tasks),
            "failed": sum(t.status == "FAILED" for t in self.tasks),
            "max_running_observed": self.max_running_observed,
            "max_segment_running": dict(
                sorted(self.max_segment_running.items())),
            "max_column_memory": dict(sorted(self.max_column_memory.items())),
            "total_wait_ms": sum(t.wait_ms for t in self.tasks),
            "total_run_ms": sum(t.run_ms for t in self.tasks),
            "violations": list(self.violations),
        }


DecodeFn = Callable[[BlockTask, Segment], tuple[list, int]]


def _default_decode(task: BlockTask,
                    segment: Segment) -> tuple[list, int]:
    values = segment.read_block(task.row_group, task.column,
                                task.block_index)
    return values, len(values)


class Scheduler:
    def __init__(self, params: BudgetParams,
                 segments: dict[str, Segment], *,
                 decode_fn: DecodeFn | None = None,
                 work_ms: int = 2):
        params.validate()
        self.params = params
        self.segments = segments
        self.decode_fn = decode_fn or _default_decode
        self.work_ms = work_ms

        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._running = 0
        self._segment_running: dict[str, int] = {}
        self._column_memory: dict[str, int] = {}
        self._max_running = 0
        self._max_segment: dict[str, int] = {}
        self._max_column: dict[str, int] = {}
        self._violations: list[str] = []
        self._stop = False

    # ------------------------------------------------------------------
    def build_tasks(self, candidate_blocks: list[dict]) -> list[BlockTask]:
        tasks: list[BlockTask] = []
        ordered = sorted(candidate_blocks,
                         key=lambda item: (item["segment_id"],
                                           item["row_group"], item["column"],
                                           item["block_index"]))
        for ordinal, item in enumerate(ordered):
            segment = self.segments[item["segment_id"]]
            header = segment.header(item["row_group"], item["column"],
                                    item["block_index"])
            tasks.append(BlockTask(
                task_id=f"{TASK_ID_PREFIX}-{ordinal:04d}",
                segment_id=item["segment_id"],
                row_group=item["row_group"], column=item["column"],
                block_index=item["block_index"],
                memory_cost=header.pre_bytes,
                encoding=header.encoding, row_count=header.row_count))
        return tasks

    def run(self, candidate_blocks: list[dict]) -> ScheduleResult:
        if self.params.per_column_memory <= 0:
            raise BudgetError("解码内存预算为零，调度被拒绝",
                              {"per_column_memory": 0})
        tasks = self.build_tasks(candidate_blocks)
        for task in tasks:
            if task.memory_cost > self.params.per_column_memory:
                raise BudgetError(
                    "块解码内存超过每列预算，无法调度",
                    {"column": task.column, "block_index": task.block_index,
                     "memory_cost": task.memory_cost,
                     "budget": self.params.per_column_memory})
        now = lambda: time.monotonic_ns() // 1_000_000
        for task in tasks:
            task.queue_start_ms = now()

        pool_size = min(self.params.global_parallelism, max(1, len(tasks)))
        threads = []
        result = ScheduleResult(tasks=tasks)
        index = {"next": 0}

        def worker() -> None:
            while True:
                with self._condition:
                    task = self._admit(index, tasks, now)
                    if task is None:
                        return
                self._execute(task, now)

        for _ in range(pool_size):
            thread = threading.Thread(target=worker, daemon=True)
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()

        result.max_running_observed = self._max_running
        result.max_segment_running = dict(sorted(self._max_segment.items()))
        result.max_column_memory = dict(sorted(self._max_column.items()))
        result.violations = list(self._violations)
        self._assert_invariants(result)
        return result

    # ------------------------------------------------------------------
    def _admit(self, index: dict, tasks: list[BlockTask],
               now) -> BlockTask | None:
        while True:
            if index["next"] >= len(tasks):
                return None
            task = tasks[index["next"]]
            segment_load = self._segment_running.get(task.segment_id, 0)
            column_load = self._column_memory.get(task.column, 0)
            if (self._running < self.params.global_parallelism
                    and segment_load < self.params.per_segment_concurrency
                    and column_load + task.memory_cost
                    <= self.params.per_column_memory):
                index["next"] += 1
                self._running += 1
                self._segment_running[task.segment_id] = segment_load + 1
                self._column_memory[task.column] = column_load + \
                    task.memory_cost
                self._max_running = max(self._max_running, self._running)
                self._max_segment[task.segment_id] = max(
                    self._max_segment.get(task.segment_id, 0),
                    segment_load + 1)
                self._max_column[task.column] = max(
                    self._max_column.get(task.column, 0),
                    column_load + task.memory_cost)
                task.wait_end_ms = now()
                task.run_start_ms = task.wait_end_ms
                task.wait_ms = task.wait_end_ms - task.queue_start_ms
                task.status = "RUNNING"
                return task
            self._condition.wait(timeout=0.05)

    def _execute(self, task: BlockTask, now) -> None:
        try:
            segment = self.segments[task.segment_id]
            values, row_count = self.decode_fn(task, segment)
            if self.work_ms:
                time.sleep(self.work_ms / 1000.0)
            task.rows_completed = row_count
            task.result_digest = block_summary(values)
            task.status = "COMPLETED"
        except BudgetError:
            task.status = "FAILED"
            task.error_code = "E_BUDGET"
            with self._lock:
                self._violations.append(task.task_id)
        except Exception as exc:  # 业务错误透传编码
            task.status = "FAILED"
            task.error_code = getattr(exc, "code", "E_UNKNOWN")
            task.error_message = str(exc)
        finally:
            task.run_end_ms = now()
            task.run_ms = task.run_end_ms - task.run_start_ms
            with self._condition:
                self._running -= 1
                self._segment_running[task.segment_id] = \
                    self._segment_running.get(task.segment_id, 1) - 1
                self._column_memory[task.column] = \
                    self._column_memory.get(task.column,
                                            task.memory_cost) - \
                    task.memory_cost
                if self._column_memory[task.column] == 0:
                    del self._column_memory[task.column]
                self._condition.notify_all()

    def _assert_invariants(self, result: ScheduleResult) -> None:
        if result.max_running_observed > self.params.global_parallelism:
            raise BudgetError("全局并行度上限被绕过",
                              {"observed": result.max_running_observed,
                               "limit": self.params.global_parallelism})
        for segment_id, load in result.max_segment_running.items():
            if load > self.params.per_segment_concurrency:
                raise BudgetError(f"段 {segment_id} 并发配额被绕过",
                                  {"observed": load,
                                   "limit": self.params.per_segment_concurrency})
        for column, load in result.max_column_memory.items():
            if load > self.params.per_column_memory:
                raise BudgetError(f"列 {column} 解码内存预算被绕过",
                                  {"observed": load,
                                   "limit": self.params.per_column_memory})
