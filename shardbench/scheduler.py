"""受全局并发预算与每库配额双重约束的并行路由探查调度器。

约束（调度报告可独立核验从未超限）：
* 任一时刻运行任务数 <= global_concurrency；
* 任一时刻每个物理库运行任务数 <= per_db_quota；
* 预算或配额不足时任务排队，记录单调时钟毫秒的等待开始/结束；
* 结局区分成功 / 预算等待超时 / 依赖未就绪 / 配置缺失四类业务码。

工作线程数固定为 global_concurrency；通过单一条件变量串行化准入决策，
准入检查与占用计数在锁内完成，杜绝越过预算的窗口。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

from .config import ShardConfig
from .determinism import split_target
from .errors import (
    OUTCOME_BUDGET_TIMEOUT,
    OUTCOME_CONFIG_MISSING,
    OUTCOME_DEPENDENCY_NOT_READY,
    OUTCOME_SUCCESS,
)


def mono_ms() -> int:
    return time.monotonic_ns() // 1_000_000


@dataclass
class ProbeTask:
    task_id: str
    unit_id: str
    dbs: tuple[str, ...]
    ready: bool
    config_present: bool
    order_index: int


@dataclass
class TaskTrace:
    task_id: str
    unit_id: str
    dbs: list[str]
    enqueue_ms: int
    wait_start_ms: int
    wait_end_ms: int | None
    run_start_ms: int | None
    run_end_ms: int | None
    outcome_code: str
    detail: str
    max_global_running: int
    max_per_db_running: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "unit_id": self.unit_id,
            "dbs": list(self.dbs),
            "enqueue_ms": self.enqueue_ms,
            "wait_start_ms": self.wait_start_ms,
            "wait_end_ms": self.wait_end_ms,
            "run_start_ms": self.run_start_ms,
            "run_end_ms": self.run_end_ms,
            "wait_elapsed_ms": (
                self.wait_end_ms - self.wait_start_ms
                if self.wait_end_ms is not None
                else None
            ),
            "run_elapsed_ms": (
                self.run_end_ms - self.run_start_ms
                if self.run_end_ms is not None and self.run_start_ms is not None
                else None
            ),
            "outcome_code": self.outcome_code,
            "detail": self.detail,
            "max_global_running": self.max_global_running,
            "max_per_db_running": dict(self.max_per_db_running),
        }


ProbeFn = Callable[[ProbeTask], str]


def _default_probe(task: ProbeTask) -> str:
    # 本地确定性探查：仅做极短休眠模拟调度，不访问任何真实数据库。
    time.sleep(0.002)
    return f"探查完成：库 {','.join(task.dbs)}"


class Scheduler:
    def __init__(
        self,
        config: ShardConfig,
        probe_fn: ProbeFn | None = None,
    ) -> None:
        self.config = config
        self.budget = config.budget
        self.probe_fn = probe_fn or _default_probe
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._pending: list[ProbeTask] = []
        self._global_running = 0
        self._per_db_running: dict[str, int] = {}
        self._peak_global = 0
        self._peak_per_db: dict[str, int] = {}
        self._traces: list[TaskTrace] = []
        self._shutdown = False

    def submit(self, tasks: list[ProbeTask]) -> None:
        with self._cond:
            now = mono_ms()
            for task in sorted(tasks, key=lambda item: item.order_index):
                self._pending.append(task)
                self._traces.append(
                    TaskTrace(
                        task_id=task.task_id,
                        unit_id=task.unit_id,
                        dbs=list(task.dbs),
                        enqueue_ms=now,
                        wait_start_ms=now,
                        wait_end_ms=None,
                        run_start_ms=None,
                        run_end_ms=None,
                        outcome_code="",
                        detail="",
                        max_global_running=0,
                        max_per_db_running={},
                    )
                )
            self._cond.notify_all()

    def _trace_of(self, task_id: str) -> TaskTrace:
        return next(trace for trace in self._traces if trace.task_id == task_id)

    def _admittable(self, task: ProbeTask) -> bool:
        if self._global_running >= self.budget.global_concurrency:
            return False
        for db_name in task.dbs:
            if self._per_db_running.get(db_name, 0) >= self.budget.per_db_quota:
                return False
        return True

    def _acquire(self, task: ProbeTask) -> None:
        self._global_running += 1
        self._peak_global = max(self._peak_global, self._global_running)
        peaks: dict[str, int] = {}
        for db_name in task.dbs:
            current = self._per_db_running.get(db_name, 0) + 1
            self._per_db_running[db_name] = current
            self._peak_per_db[db_name] = max(
                self._peak_per_db.get(db_name, 0), current
            )
            peaks[db_name] = self._peak_per_db[db_name]
        trace = self._trace_of(task.task_id)
        trace.max_global_running = self._peak_global
        trace.max_per_db_running = dict(peaks)

    def _release(self, task: ProbeTask) -> None:
        self._global_running -= 1
        for db_name in task.dbs:
            self._per_db_running[db_name] = self._per_db_running.get(db_name, 1) - 1

    def _worker_loop(self) -> None:
        while True:
            with self._cond:
                task = self._wait_for_task()
                if task is None:
                    return
                trace = self._trace_of(task.task_id)
                if not task.config_present:
                    self._finish_locked(
                        task, trace, OUTCOME_CONFIG_MISSING, "路由单元配置缺失，无法探查"
                    )
                    self._cond.notify_all()
                    continue
                if not task.ready:
                    self._finish_locked(
                        task,
                        trace,
                        OUTCOME_DEPENDENCY_NOT_READY,
                        "上游路由单元尚未就绪，拒绝并行探查",
                    )
                    self._cond.notify_all()
                    continue
                # 等待基准为任务真正开始申请并发额度的时刻；排队时间也计入预算。
                waiting_since = mono_ms()
                trace.wait_start_ms = waiting_since
                deadline = waiting_since + self.budget.wait_timeout_ms
                while not self._admittable(task):
                    remaining = deadline - mono_ms()
                    if remaining <= 0:
                        self._finish_locked(
                            task,
                            trace,
                            OUTCOME_BUDGET_TIMEOUT,
                            "预算或配额等待超时，任务未获得并发额度",
                        )
                        self._cond.notify_all()
                        break
                    self._cond.wait(timeout=min(remaining, 50) / 1000.0)
                if trace.outcome_code:
                    continue
                now = mono_ms()
                trace.wait_end_ms = now
                trace.run_start_ms = now
                self._acquire(task)

            detail = ""
            failure: str | None = None
            try:
                detail = self.probe_fn(task)
            except Exception as exc:  # pragma: no cover - 防御性兜底
                failure = f"探查异常: {exc}"

            with self._cond:
                end = mono_ms()
                trace.run_end_ms = end
                self._release(task)
                if failure is None:
                    trace.outcome_code = OUTCOME_SUCCESS
                    trace.detail = detail
                else:
                    trace.outcome_code = OUTCOME_CONFIG_MISSING
                    trace.detail = failure
                self._cond.notify_all()

    def _wait_for_task(self) -> ProbeTask | None:
        while not self._pending and not self._shutdown:
            self._cond.wait()
        if self._shutdown and not self._pending:
            return None
        # 选择优先级（全程持锁，绝不绕过预算）：
        # 1) 可立即终结的任务（依赖未就绪/配置缺失）；
        # 2) 当前可准入的最早任务；
        # 3) 队首任务——它因预算/配额阻塞，进入带 deadline 的等待窗口，
        #    超时则记录 T_BUDGET_TIMEOUT，额度释放则被唤醒执行。
        for index, task in enumerate(self._pending):
            if (
                not task.config_present
                or not task.ready
            ):
                return self._pending.pop(index)
        for index, task in enumerate(self._pending):
            if self._admittable(task):
                return self._pending.pop(index)
        return self._pending.pop(0)

    def _finish_locked(
        self, task: ProbeTask, trace: TaskTrace, code: str, detail: str
    ) -> None:
        now = mono_ms()
        trace.wait_end_ms = now
        trace.run_start_ms = None
        trace.run_end_ms = None
        trace.outcome_code = code
        trace.detail = detail
        trace.max_global_running = self._peak_global
        trace.max_per_db_running = dict(self._peak_per_db)

    def run(self, tasks: list[ProbeTask]) -> list[TaskTrace]:
        workers = [
            threading.Thread(
                target=self._worker_loop,
                name=f"shardbench-worker-{index}",
                daemon=True,
            )
            for index in range(self.budget.global_concurrency)
        ]
        for worker in workers:
            worker.start()
        self.submit(tasks)
        with self._cond:
            while len([trace for trace in self._traces if not trace.outcome_code]) > 0:
                self._cond.wait(timeout=0.05)
            self._shutdown = True
            self._cond.notify_all()
        for worker in workers:
            worker.join()
        return self.report()

    def report(self) -> list[TaskTrace]:
        order = {
            task.task_id: task.order_index
            for task in self._pending
        }
        traces = sorted(
            self._traces,
            key=lambda trace: (
                self._order_of(trace.task_id, order),
                trace.task_id,
            ),
        )
        return traces

    def _order_of(self, task_id: str, pending_order: dict[str, int]) -> int:
        if task_id in pending_order:
            return pending_order[task_id]
        for trace in self._traces:
            if trace.task_id == task_id:
                return self._traces.index(trace)
        return 0

    def verify_never_exceeded(self) -> dict[str, object]:
        """独立核验入口：峰值占用对照预算，返回是否从未超限及峰值数据。"""
        return {
            "global_concurrency": self.budget.global_concurrency,
            "per_db_quota": self.budget.per_db_quota,
            "peak_global_running": self._peak_global,
            "peak_per_db_running": dict(sorted(self._peak_per_db.items())),
            "global_ok": self._peak_global <= self.budget.global_concurrency,
            "per_db_ok": all(
                peak <= self.budget.per_db_quota
                for peak in self._peak_per_db.values()
            ),
        }


def dbs_for_targets(targets: list[str]) -> list[str]:
    """从物理库表标识提取去重后的物理库，稳定排序，用于每库配额统计。"""
    return sorted({f"db{split_target(target)[0]}" for target in targets})
