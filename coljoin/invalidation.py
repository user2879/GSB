"""依赖链指纹、版本推进与失效传播。

依赖链：代价权重与直方图 → 计划版本 → 分片批次 → 回执采纳结论。

分层指纹：
- data_digest：块组行数据（空值位图 + 列编码载荷）摘要；
- stats_digest：直方图/NDV 统计口径摘要；
- cost_input_digest：统计口径 + 权重 + 会话缩放/粒度摘要；
- plan_digest：规范化计划体摘要；
- shard_digest：计划 + 分片批次切分摘要；
- receipt_digest：采纳回执摘要。

版本号只单调递增；每次变更输出按 bg_id/plan_id 稳定排序的直接、间接失效集合与依据。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .canonical import digest
from .cost import CostEstimator
from .model import BlockGroupSpec, CostWeights, QuerySpec, SessionSpec
from .planner import enumerate_plans
from .statistics import stats_digest
from .storage import read_column


def data_digest_for_group(data_dir: str, group: BlockGroupSpec) -> str:
    payload = {"bg_id": group.bg_id, "rows": group.rows, "columns": []}
    for column in sorted(group.columns, key=lambda item: item.name):
        values, bitmap = read_column(data_dir, group.bg_id, column.name)
        payload["columns"].append(
            {
                "name": column.name,
                "encoding": column.encoding,
                "values": values.tolist(),
                "null_bitmap": bitmap.tolist(),
            }
        )
    return digest(payload)


def data_digests(data_dir: str, groups: List[BlockGroupSpec]) -> Dict[str, str]:
    ordered = sorted(groups, key=lambda item: item.bg_id)
    return {
        group.bg_id: data_digest_for_group(data_dir, group)
        for group in ordered
    }


def cost_input_digest(
    groups: List[BlockGroupSpec], weights: CostWeights, session: SessionSpec
) -> str:
    return digest(
        {
            "stats": stats_digest(groups),
            "weights": {
                "scan": weights.scan,
                "compare": weights.compare,
                "build": weights.build,
                "probe": weights.probe,
                "memory": weights.memory,
            },
            "session": {
                "scale": session.scale,
                "block_granularity": session.block_granularity,
                "stats_version": session.stats_version,
            },
        }
    )


def plan_body_digest(plan: Dict[str, Any]) -> str:
    return digest(plan["body"])


def shard_batch_digest(plan: Dict[str, Any], batch: Dict[str, Any]) -> str:
    return digest({"plan": plan["body"], "batch": batch})


@dataclass
class PlanVersionState:
    version: int
    cost_input_digest: str
    plan_digests: Dict[str, str] = field(default_factory=dict)
    shard_digests: Dict[str, str] = field(default_factory=dict)
    receipt_digests: Dict[str, str] = field(default_factory=dict)
    data_digests: Dict[str, str] = field(default_factory=dict)


def compute_change_set(
    old_data: Dict[str, str], new_data: Dict[str, str]
) -> List[str]:
    changed: List[str] = []
    for bg_id in sorted(set(old_data) | set(new_data)):
        if old_data.get(bg_id) != new_data.get(bg_id):
            changed.append(bg_id)
    return changed


def recompute_invalidated_only(
    query: QuerySpec,
    groups: List[BlockGroupSpec],
    weights: CostWeights,
    session: SessionSpec,
    previous_evaluations: Dict[str, Dict[str, Any]],
    invalidated_plan_ids: List[str],
) -> Dict[str, Dict[str, Any]]:
    """仅重算失效计划；未失效计划的估算摘要原样保留，逐字节可复核。"""
    estimator = CostEstimator(groups, weights, session)
    plans_by_id = {plan["plan_id"]: plan for plan in enumerate_plans(query)}
    result = {plan_id: dict(summary) for plan_id, summary in previous_evaluations.items()}
    for plan_id in invalidated_plan_ids:
        result[plan_id] = estimator.evaluate(plans_by_id[plan_id])
    return result


def trace_invalidation(
    query: QuerySpec,
    groups: List[BlockGroupSpec],
    weights: CostWeights,
    session: SessionSpec,
    changed_block_groups: List[str],
    old_state: Optional[PlanVersionState],
    new_cost_digest: str,
) -> Dict[str, Any]:
    """计算直接/间接失效集合并重算全部候选的定点估算（确定性输出）。"""
    plans = enumerate_plans(query)
    estimator = CostEstimator(groups, weights, session)
    evaluations = {plan["plan_id"]: estimator.evaluate(plan) for plan in plans}
    new_plan_digests = {
        plan["plan_id"]: plan_body_digest(plan) for plan in plans
    }

    direct_plans: Set[str] = set()
    reasons: List[Dict[str, Any]] = []
    changed_set = set(changed_block_groups)
    for plan in plans:
        involved = set(plan["body"]["order"])
        touched = sorted(involved & changed_set)
        if touched:
            direct_plans.add(plan["plan_id"])
            reasons.append(
                {
                    "plan_id": plan["plan_id"],
                    "kind": "data_or_stats_change",
                    "block_groups": touched,
                    "basis": "计划引用的块组数据/直方图指纹变化",
                }
            )

    old_cost_digest = old_state.cost_input_digest if old_state else None
    cost_changed = old_cost_digest != new_cost_digest
    if cost_changed:
        for plan in plans:
            if plan["plan_id"] not in direct_plans:
                reasons.append(
                    {
                        "plan_id": plan["plan_id"],
                        "kind": "cost_input_change",
                        "block_groups": [],
                        "basis": "权重/缩放/统计口径指纹变化导致代价重算",
                    }
                )
            direct_plans.add(plan["plan_id"])

    indirect_shards: List[str] = []
    indirect_receipts: List[str] = []
    if old_state is not None:
        for plan_id in sorted(direct_plans):
            for shard_id in sorted(old_state.shard_digests):
                if shard_id.startswith(plan_id + ":"):
                    indirect_shards.append(shard_id)
            for receipt_id in sorted(old_state.receipt_digests):
                if receipt_id.startswith(plan_id + ":"):
                    indirect_receipts.append(receipt_id)

    next_version = (old_state.version + 1) if old_state else 1
    return {
        "new_version": next_version,
        "cost_input_changed": cost_changed,
        "changed_block_groups": sorted(changed_block_groups),
        "invalidated_plans": sorted(direct_plans),
        "invalidated_shards": sorted(set(indirect_shards)),
        "invalidated_receipts": sorted(set(indirect_receipts)),
        "reasons": sorted(reasons, key=lambda item: (item["plan_id"], item["kind"])),
        "plans": plans,
        "evaluations": evaluations,
        "plan_digests": new_plan_digests,
    }
