"""任务网格展开与确定性调度规划。

关键设计：预算/配额的裁决完全由一个纯函数 build_schedule_plan 完成，输入
仅为配置与数据指纹，与线程池执行顺序无关，因此并发度变化不会改变：
- 任务启动/中止集合；
- E_BUDGET_EXHAUSTED / E_TASK_QUOTA_EXCEEDED 标记集合；
- 预算消耗汇总与稳定顺序的决策记录。
执行层只按计划的 epoch_limit 运行确定性训练。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import ExperimentConfig
from .data import PreparedData
from .models import (
    TaskSpec,
    make_cache_key,
    make_task_key,
    make_task_spec,
)

READY = "ready"
RUNNING = "running"
COMPLETED = "completed"
CONVERGED = "converged"
QUOTA_EXCEEDED = "quota_exceeded"
BUDGET_TRUNCATED = "budget_truncated"
BUDGET_EXHAUSTED = "budget_exhausted"
FAILED = "failed"


@dataclass(frozen=True)
class GridTask:
    task_key: str
    cache_key: str
    scheme: str
    penalty: float
    l1_ratio: float
    seed: int
    spec: TaskSpec

    def stable_order(self) -> tuple[str, float, float, float, int]:
        return (
            self.scheme,
            self.penalty,
            self.l1_ratio,
            self.spec.learning_rate,
            self.seed,
        )


@dataclass
class PlannedTask:
    task: GridTask
    planned_epochs: int          # 计划执行轮次上限
    starts: bool                 # 是否允许启动
    quota_capped: bool           # 是否被单方案配额截断
    budget_limited: bool         # 是否被全局预算部分截断
    decision: str                # 调度决策业务语义
    ready_position: int          # 就绪队列中的稳定位置

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_key": self.task.task_key,
            "cache_key": self.task.cache_key,
            "scheme": self.task.scheme,
            "penalty": self.task.penalty,
            "l1_ratio": self.task.l1_ratio,
            "seed": self.task.seed,
            "planned_epochs": self.planned_epochs,
            "starts": self.starts,
            "quota_capped": self.quota_capped,
            "budget_limited": self.budget_limited,
            "decision": self.decision,
            "ready_position": self.ready_position,
        }


@dataclass
class SchedulePlan:
    ordered_tasks: list[PlannedTask]
    consumed_budget: int
    remaining_budget: int
    exhausted_tasks: list[str]
    quota_tasks: list[str]
    decision_log: list[dict[str, Any]]

    def by_key(self) -> dict[str, PlannedTask]:
        return {item.task.task_key: item for item in self.ordered_tasks}


def expand_grid(config: ExperimentConfig, data: PreparedData) -> list[GridTask]:
    seed = config.seeds[0]
    tasks: list[GridTask] = []
    for scheme in config.schemes:
        for penalty in scheme.penalties:
            ratio_candidates = (
                scheme.l1_ratios if scheme.name == "elasticnet" else (0.0,)
            )
            for ratio in ratio_candidates:
                spec = make_task_spec(
                    scheme.name, penalty, ratio, config
                )
                task_key = make_task_key(spec, seed)
                cache_key = make_cache_key(spec, data.data_fingerprint)
                tasks.append(
                    GridTask(
                        task_key=task_key,
                        cache_key=cache_key,
                        scheme=scheme.name,
                        penalty=float(penalty),
                        l1_ratio=float(ratio),
                        seed=seed,
                        spec=spec,
                    )
                )
    tasks.sort(key=lambda item: item.stable_order())
    return tasks


def build_schedule_plan(
    config: ExperimentConfig, tasks: list[GridTask]
) -> SchedulePlan:
    """按稳定顺序裁决全局预算与单方案配额。"""

    quota = config.scheduler.per_scheme_epoch_quota
    budget = config.scheduler.global_epoch_budget
    remaining = budget
    planned: list[PlannedTask] = []
    decision_log: list[dict[str, Any]] = []
    exhausted: list[str] = []
    quota_hits: list[str] = []
    consumed = 0

    for position, task in enumerate(tasks):
        cap = min(task.spec.max_epochs, quota)
        quota_capped = cap < task.spec.max_epochs
        if remaining <= 0:
            planned.append(
                PlannedTask(
                    task=task,
                    planned_epochs=0,
                    starts=False,
                    quota_capped=False,
                    budget_limited=False,
                    decision="E_BUDGET_EXHAUSTED",
                    ready_position=position,
                )
            )
            exhausted.append(task.task_key)
            decision_log.append(
                {
                    "order": len(decision_log),
                    "task_key": task.task_key,
                    "ready_position": position,
                    "decision": "E_BUDGET_EXHAUSTED",
                    "planned_epochs": 0,
                    "remaining_before": remaining,
                    "remaining_after": remaining,
                }
            )
            continue

        allowed = min(cap, remaining)
        budget_limited = allowed < cap
        remaining -= allowed
        consumed += allowed
        if quota_capped:
            decision = "E_TASK_QUOTA_EXCEEDED"
            quota_hits.append(task.task_key)
        elif budget_limited:
            decision = "BUDGET_PARTIAL"
        else:
            decision = "START"
        planned.append(
            PlannedTask(
                task=task,
                planned_epochs=allowed,
                starts=True,
                quota_capped=quota_capped,
                budget_limited=budget_limited,
                decision=decision,
                ready_position=position,
            )
        )
        decision_log.append(
            {
                "order": len(decision_log),
                "task_key": task.task_key,
                "ready_position": position,
                "decision": decision,
                "planned_epochs": allowed,
                "remaining_before": remaining + allowed,
                "remaining_after": remaining,
            }
        )

    return SchedulePlan(
        ordered_tasks=planned,
        consumed_budget=consumed,
        remaining_budget=remaining,
        exhausted_tasks=exhausted,
        quota_tasks=quota_hits,
        decision_log=decision_log,
    )


def canonical_completion_order(tasks: list[GridTask]) -> list[str]:
    """任务完成顺序清单的规范顺序：稳定键顺序（与并发度无关）。"""

    return [task.task_key for task in sorted(tasks, key=lambda item: item.stable_order())]


def plan_to_dict(plan: SchedulePlan) -> dict[str, Any]:
    return {
        "tasks": [item.to_dict() for item in plan.ordered_tasks],
        "consumed_budget": plan.consumed_budget,
        "remaining_budget": plan.remaining_budget,
        "exhausted_tasks": list(plan.exhausted_tasks),
        "quota_tasks": list(plan.quota_tasks),
        "decision_log": list(plan.decision_log),
    }
