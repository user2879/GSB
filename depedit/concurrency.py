"""线程池并发执行，结果按稳定排序键重排，与单线程升序结果逐字段一致。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

from .config import AppConfig
from .fingerprints import stable_task_key


@dataclass
class Task:
    sent_id: str
    stage: str
    payload: Any = None

    @property
    def key(self) -> str:
        return stable_task_key(self.sent_id, self.stage)


@dataclass
class TaskOutcome:
    key: str
    sent_id: str
    stage: str
    value: Any = None
    failure: dict[str, Any] | None = None


def run_tasks(
    tasks: list[Task],
    worker: Callable[[Task], Any],
    failure_builder: Callable[[Task, Exception], dict[str, Any]],
    cfg: AppConfig,
) -> list[TaskOutcome]:
    """并发执行；异常被隔离为 failure 记录，不中断其余任务。

    最终顺序按 (sent_id, stage) 稳定排序，与并发度无关。
    """
    ordered = sorted(tasks, key=lambda t: t.key)
    outcomes: list[TaskOutcome] = [None] * len(ordered)  # type: ignore[list-item]
    index = {task.key: i for i, task in enumerate(ordered)}

    def _run(task: Task) -> tuple[str, Any, dict[str, Any] | None]:
        try:
            return task.key, worker(task), None
        except Exception as exc:  # 真实隔离：单句失败不影响其他句
            return task.key, None, failure_builder(task, exc)

    max_workers = max(1, min(int(cfg.max_workers), int(cfg.workers_hard_limit)))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for key, value, failure in pool.map(_run, ordered):
            i = index[key]
            task = ordered[i]
            outcomes[i] = TaskOutcome(
                key=key, sent_id=task.sent_id, stage=task.stage, value=value, failure=failure
            )
    return outcomes
