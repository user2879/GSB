"""单批乘加点预算调度与并发/串行一致执行。

调度按批次号与位置索引稳定排序消耗预算；单位置计算量超出当前批次
剩余预算时记 E_BATCH_BUDGET_EXHAUSTED 并保留待算，超出整批预算时
同样记录并直接跳过（避免无限重排）。线程池最大并发数严格取配置，
任务携带 (批次, 位置, 阶段名) 稳定排序键，所有产物在提交后按该键
重排，因此与单线程执行结果完全一致。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .errors import E_BATCH_BUDGET_EXHAUSTED, FailureLedger

STAGE_ORDER = {"kernel": 0, "truncate": 1, "dequant": 2}


@dataclass(frozen=True)
class ScheduledTask:
    batch: int
    position: int
    stage: str
    mac_cost: int

    @property
    def sort_key(self) -> tuple:
        return (self.batch, self.position, STAGE_ORDER[self.stage])


@dataclass
class BatchPlan:
    batches: list[list[ScheduledTask]] = field(default_factory=list)
    budget_ledger: list[dict] = field(default_factory=list)


def plan_batches(total_positions: int, macs_per_position: int, mac_budget: int,
                 ledger: FailureLedger) -> BatchPlan:
    """只做预算切分与账本记录，与执行模式无关。"""
    plan = BatchPlan()
    batch_no = 0
    cursor = 0
    while cursor < total_positions:
        remaining_budget = mac_budget
        current: list[ScheduledTask] = []
        plan.budget_ledger.append(
            {"batch": batch_no, "start_budget": mac_budget, "allocated": 0, "reserved": []}
        )
        while cursor < total_positions:
            cost = macs_per_position
            sort_key = (batch_no, cursor, STAGE_ORDER["kernel"])
            if cost > mac_budget:
                ledger.add(
                    E_BATCH_BUDGET_EXHAUSTED,
                    f"位置 {cursor} 计算量 {cost} 超出单批预算 {mac_budget}，跳过待算位置",
                    "schedule",
                    sort_key,
                    position=cursor,
                    batch=batch_no,
                )
                plan.budget_ledger[batch_no]["reserved"].append(cursor)
                cursor += 1
                continue
            if cost > remaining_budget:
                ledger.add(
                    E_BATCH_BUDGET_EXHAUSTED,
                    f"批次 {batch_no} 剩余预算 {remaining_budget} 不足以计算位置 {cursor}"
                    f"（需要 {cost}），该位置保留待算",
                    "schedule",
                    sort_key,
                    position=cursor,
                    batch=batch_no,
                )
                plan.budget_ledger[batch_no]["reserved"].append(cursor)
                break
            for stage in ("kernel", "truncate", "dequant"):
                current.append(
                    ScheduledTask(batch=batch_no, position=cursor, stage=stage, mac_cost=cost)
                )
            remaining_budget -= cost
            plan.budget_ledger[batch_no]["allocated"] += cost
            cursor += 1
        plan.batches.append(current)
        plan.budget_ledger[batch_no]["remaining_budget"] = remaining_budget
        batch_no += 1
    return plan


def execute_tasks(tasks: list[ScheduledTask], worker, concurrency: int,
                  threaded: bool) -> list:
    """执行任务并按稳定排序键返回结果列表。

    worker 接收 ScheduledTask，返回可 JSON 序列化的结果字典。
    """
    ordered = sorted(tasks, key=lambda task: task.sort_key)
    if not threaded or concurrency <= 1:
        return [worker(task) for task in ordered]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(worker, task): task for task in ordered}
        results: dict[tuple, object] = {}
        for future, task in futures.items():
            results[task.sort_key] = future.result()
    return [results[task.sort_key] for task in ordered]


def max_in_flight(concurrency: int) -> int:
    return max(1, concurrency)
