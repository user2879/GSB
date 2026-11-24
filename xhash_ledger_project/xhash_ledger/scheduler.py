"""全局签名计算点预算与单样本配额调度。

处理顺序固定为：上轮遗留待办（入队顺序）优先，其后按样本标识升序。
点扣减在主线程按该稳定顺序进行，因此待办集合与预算扣减序列与线程池并发度无关。
每个并发任务携带由 (样本标识, 位序号, 阶段名) 派生的稳定排序键。
"""
from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
from typing import Any, Callable

from .errors import (
    E_SIGNATURE_BUDGET_PENDING,
    E_SIGNATURE_QUOTA_EXCEEDED,
    Failure,
    make_failure,
)


def feature_cost(feature_dim: int) -> int:
    return feature_dim


def signature_cost(signature_bits: int, feature_dim: int) -> int:
    return signature_bits * feature_dim


def bucket_cost(signature_bits: int, bucket_bits: int) -> int:
    chunks = (signature_bits + bucket_bits - 1) // bucket_bits
    return chunks * bucket_bits


def total_cost(feature_dim: int, signature_bits: int, bucket_bits: int) -> int:
    return feature_cost(feature_dim) + signature_cost(signature_bits, feature_dim) + bucket_cost(
        signature_bits, bucket_bits
    )


@dataclass
class SchedulePlan:
    ready: list[str] = field(default_factory=list)
    quota_failed: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    point_deductions: list[dict[str, Any]] = field(default_factory=list)
    consumed: int = 0
    remaining_budget: int = 0
    failures: list[Failure] = field(default_factory=list)


def stable_task_key(sample_id: str, bit_index: int, stage: str) -> str:
    return f"{sample_id}|{stage}|{bit_index:08d}"


def plan_execution(
    candidate_ids: list[str],
    pending_carry: list[str],
    quota: int,
    budget_remaining: int,
    feature_dim: int,
    signature_bits: int,
    bucket_bits: int,
) -> SchedulePlan:
    """按确定性规则划分就绪 / 配额失败 / 预算待办，并记录点扣减序列。"""
    plan = SchedulePlan(remaining_budget=budget_remaining)
    ordered = list(pending_carry) + [sid for sid in sorted(candidate_ids) if sid not in pending_carry]
    per_sample = total_cost(feature_dim, signature_bits, bucket_bits)
    stages = [
        ("feature", feature_cost(feature_dim)),
        ("signature", signature_cost(signature_bits, feature_dim)),
        ("bucket", bucket_cost(signature_bits, bucket_bits)),
    ]
    for sid in ordered:
        if per_sample > quota:
            plan.quota_failed.append(sid)
            plan.failures.append(
                make_failure(
                    E_SIGNATURE_QUOTA_EXCEEDED, "scheduler", sid, "point_quota",
                    f"单样本需要 {per_sample} 点，超出配额 {quota}",
                )
            )
            continue
        if per_sample > plan.remaining_budget:
            plan.pending.append(sid)
            plan.failures.append(
                make_failure(
                    E_SIGNATURE_BUDGET_PENDING, "scheduler", sid, "global_budget",
                    f"单样本需要 {per_sample} 点，全局剩余预算 {plan.remaining_budget} 不足，转入待办",
                )
            )
            continue
        plan.ready.append(sid)
        for stage_name, stage_cost in stages:
            plan.point_deductions.append(
                {
                    "sample_id": sid,
                    "stage": stage_name,
                    "cost": stage_cost,
                    "task_key": stable_task_key(sid, 0, stage_name),
                }
            )
        plan.consumed += per_sample
        plan.remaining_budget -= per_sample
    return plan


def run_thread_pool(
    ready_ids: list[str],
    worker: Callable[[str], Any],
    max_workers: int,
) -> list[Any]:
    """线程池并发执行，结果按样本标识稳定顺序归并，保证并发/串行逐字段一致。"""
    if not ready_ids:
        return []
    workers = max(1, min(max_workers, len(ready_ids)))
    if workers == 1:
        return [worker(sid) for sid in ready_ids]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {pool.submit(worker, sid): sid for sid in ready_ids}
        results: dict[str, Any] = {}
        for future in concurrent.futures.as_completed(future_map):
            results[future_map[future]] = future.result()
    return [results[sid] for sid in ready_ids]
