"""三类预算联合调度器：独立阻塞、待办优先、稳定顺序、线程池并发上限。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .config import Config


@dataclass
class TodoUnit:
    unit_key: str
    sample_id: str
    stage: str
    index: str
    consumed: dict[str, int]

    def to_dict(self) -> dict:
        return {"unit_key": self.unit_key, "sample_id": self.sample_id,
                "stage": self.stage, "index": self.index, "consumed": self.consumed}

    @property
    def sort_key(self) -> str:
        return self.unit_key


@dataclass
class BudgetState:
    scan_rows_remaining: int
    compare_points_remaining: int
    adjudication_points_remaining: int
    scan_rows_total: int
    compare_points_total: int
    adjudication_points_total: int

    @classmethod
    def from_config(cls, cfg: Config) -> "BudgetState":
        return cls(cfg.scan_row_budget, cfg.compare_point_budget,
                   cfg.adjudication_budget, cfg.scan_row_budget,
                   cfg.compare_point_budget, cfg.adjudication_budget)

    def to_dict(self) -> dict:
        return {
            "scan_rows": {"remaining": self.scan_rows_remaining,
                          "total": self.scan_rows_total},
            "compare_points": {"remaining": self.compare_points_remaining,
                               "total": self.compare_points_total},
            "adjudication_points": {
                "remaining": self.adjudication_points_remaining,
                "total": self.adjudication_points_total},
        }


class Scheduler:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.budget = BudgetState.from_config(cfg)
        self.todos: list[TodoUnit] = []
        self.failures: list[dict] = []
        self.preview: list[dict] = []

    def todo_keys(self) -> list[str]:
        return [item.unit_key for item in sorted(self.todos, key=lambda item: item.unit_key)]

    def add_todo(self, unit: TodoUnit) -> None:
        if not any(item.unit_key == unit.unit_key for item in self.todos):
            self.todos.append(unit)
            self.todos.sort(key=lambda item: item.unit_key)

    def pop_todos_for(self, stage: str) -> list[TodoUnit]:
        picked = [item for item in self.todos if item.stage == stage]
        self.todos = [item for item in self.todos if item.stage != stage]
        return picked

    def run_pool(self, tasks: list, worker):
        """线程池并发，max_workers 严格取配置；结果按稳定排序键回排，保证确定性。"""
        ordered = sorted(tasks, key=lambda item: item.sort_key)
        if self.cfg.max_workers == 1 or len(ordered) <= 1:
            return [worker(item) for item in ordered]
        with ThreadPoolExecutor(max_workers=self.cfg.max_workers) as pool:
            results = list(pool.map(worker, ordered))
        return results

    def record_failure(self, sample_id: str, stage: str, code: str, message: str) -> None:
        entry = {"sample_id": sample_id, "stage": stage, "code": code, "message": message}
        if entry not in self.failures:
            self.failures.append(entry)
            self.failures.sort(key=lambda item: (item["sample_id"], item["stage"], item["code"]))

    def update_preview(self, sample_id: str, stage: str, summary: dict) -> None:
        self.preview = [item for item in self.preview
                        if not (item["sample_id"] == sample_id and item["stage"] == stage)]
        self.preview.append({"sample_id": sample_id, "stage": stage, "summary": summary})
        self.preview.sort(key=lambda item: (item["sample_id"], item["stage"]))

    def to_dict(self) -> dict:
        return {
            "budget": self.budget.to_dict(),
            "todos": [item.to_dict() for item in sorted(self.todos, key=lambda x: x.unit_key)],
            "failures": self.failures,
            "preview": self.preview,
        }
