"""单批元素预算与单层配额约束下的稳定调度器。"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import ErrorCode, FailureList


@dataclass
class Job:
    layer: str
    row_index: int
    width: int
    upstream_rows: list[tuple[str, int]]


@dataclass
class BatchPlan:
    ready: list[Job]
    pending: list[Job]


def stable_jobs(
    layer_order: list[str],
    rows_per_layer: dict[str, int],
    width_per_layer: dict[str, int],
    upstream_per_layer: dict[str, list[tuple[str, int]]],
) -> list[Job]:
    """层名字典序 + 行序号升序的稳定作业序列。"""

    jobs: list[Job] = []
    # layer_order 已是拓扑序；要求字典序稳定时由调用方保证其同层排序
    for layer in layer_order:
        for row_index in range(rows_per_layer[layer]):
            jobs.append(
                Job(
                    layer=layer,
                    row_index=row_index,
                    width=width_per_layer[layer],
                    upstream_rows=list(upstream_per_layer.get(layer, [])),
                )
            )
    return jobs


def plan_batch(
    jobs: list[Job],
    element_budget: int,
    layer_row_quota: int,
    failures: FailureList,
    source: str = "scheduler",
) -> BatchPlan:
    """按稳定顺序消耗预算；超预算行保留待办并打 E_OP_BUDGET_PENDING。"""

    ready: list[Job] = []
    pending: list[Job] = []
    elements_used = 0
    layer_rows_used: dict[str, int] = {}
    for job in jobs:
        layer_used = layer_rows_used.get(job.layer, 0)
        if layer_used >= layer_row_quota:
            pending.append(job)
            failures.add(
                ErrorCode.E_BUDGET_PENDING,
                source,
                f"层 {job.layer} 达到单层配额 {layer_row_quota}，行 {job.row_index} 挂起",
                {"layer": job.layer, "row": job.row_index, "reason": "layer_quota"},
            )
            continue
        if elements_used + job.width > element_budget:
            pending.append(job)
            failures.add(
                ErrorCode.E_BUDGET_PENDING,
                source,
                f"单批元素预算 {element_budget} 不足，行 {job.layer}/{job.row_index} 挂起",
                {
                    "layer": job.layer,
                    "row": job.row_index,
                    "reason": "element_budget",
                    "elements_used": elements_used,
                    "need": job.width,
                },
            )
            continue
        ready.append(job)
        elements_used += job.width
        layer_rows_used[job.layer] = layer_used + 1
    return BatchPlan(ready=ready, pending=pending)
