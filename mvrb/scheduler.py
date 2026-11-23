"""全局点数预算 + 单视图配额的并行调度器（并发度受配置约束）。

预算切分在提交前按稳定顺序完成，因此无论线程调度顺序、并发度或预算耗尽时机如何，
最终视图输出顺序、失败清单顺序都与单线程按视图序号执行一致。
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from .errors import E_POINT_BUDGET_EXHAUSTED, E_VIEW_BUDGET_EXHAUSTED, FailureList


@dataclass
class ViewTask:
    view: int
    sort_key: str
    work: object            # 无参可调用对象，返回视图产物；可为 None（仅做预算登记）
    point_count: int = 0
    result: object = None


@dataclass
class ScheduleReport:
    order: list = field(default_factory=list)
    exhausted_views: list = field(default_factory=list)
    todo_queue: list = field(default_factory=list)
    global_exhausted: bool = False
    peak_concurrency: int = 0
    budget_remaining: int = 0
    results: dict = field(default_factory=dict)


class BudgetScheduler:
    def __init__(self, max_workers: int, point_budget: int, view_quota: int,
                 force_serial: bool = False):
        self.max_workers = max(1, int(max_workers))
        self.point_budget = int(point_budget)
        self.view_quota = int(view_quota)
        self.force_serial = force_serial

    def run(self, tasks: list, failures: FailureList) -> ScheduleReport:
        tasks = sorted(tasks, key=lambda t: t.sort_key)
        report = ScheduleReport(budget_remaining=self.point_budget)
        runnable = []
        for task in tasks:
            if task.point_count > self.view_quota:
                failures.fail(
                    E_VIEW_BUDGET_EXHAUSTED,
                    f"view {task.view} points {task.point_count} > quota {self.view_quota}",
                    view=task.view,
                    stage="schedule",
                    sort_key=f"schedule|{task.view:04d}|{E_VIEW_BUDGET_EXHAUSTED}",
                )
                report.exhausted_views.append(task.view)
                continue
            runnable.append(task)

        admitted, todo, remaining = [], [], self.point_budget
        for task in runnable:
            if remaining <= 0:
                todo.append(task)
                continue
            admitted.append(task)
            remaining -= task.point_count
        report.budget_remaining = remaining
        report.todo_queue = [t.view for t in todo]
        report.global_exhausted = bool(todo)
        for task in todo:
            failures.fail(
                E_POINT_BUDGET_EXHAUSTED,
                f"global budget exhausted; view {task.view} kept in todo queue",
                view=task.view,
                stage="schedule",
                sort_key=f"schedule|{task.view:04d}|{E_POINT_BUDGET_EXHAUSTED}",
            )

        workers = 1 if self.force_serial else min(self.max_workers, max(1, len(admitted)))
        if workers <= 1 or all(t.work is None for t in admitted):
            for task in admitted:
                task.result = task.work() if task.work is not None else None
                report.results[task.view] = task.result
            report.peak_concurrency = 1 if admitted else 0
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(t.work): t for t in admitted}
                report.peak_concurrency = max(1, len(futures))
                for fut in as_completed(futures):
                    task = futures[fut]
                    task.result = fut.result()
                    report.results[task.view] = task.result

        report.order = [t.view for t in admitted]
        return report

    def resume(self, tasks: list, failures: FailureList, extra_budget: int) -> ScheduleReport:
        """下一批：待办队列优先续跑。"""
        self.point_budget = extra_budget
        return self.run(tasks, failures)
