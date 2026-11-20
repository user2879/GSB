"""依赖图上的失效传播与仅失效子集重算。

从种子集合（规则变更/环节点调整/上游结果变化直接影响的单元）出发，沿
``downstream`` 关系 BFS 推导全部直接与间接下游；输出稳定的失效集合、
拓扑顺序以及每一步传播依据。未失效单元保持原结果不动。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import ShardConfig


@dataclass
class PropagationStep:
    source: str
    affected: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"source": self.source, "affected": self.affected, "reason": self.reason}


@dataclass
class InvalidationResult:
    seeds: list[str]
    invalidated: list[str]
    topo_order: list[str]
    steps: list[PropagationStep] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "seeds": list(self.seeds),
            "invalidated": list(self.invalidated),
            "topo_order": list(self.topo_order),
            "steps": [step.to_dict() for step in self.steps],
        }


def propagate(
    config: ShardConfig,
    seeds: list[str],
    *,
    seed_reason: str = "规则或哈希环版本变更",
) -> InvalidationResult:
    for seed in seeds:
        if seed not in config.units:
            raise KeyError(f"失效种子单元不存在: {seed}")
    steps: list[PropagationStep] = [
        PropagationStep(source="<seed>", affected=seed, reason=seed_reason)
        for seed in sorted(set(seeds))
    ]
    invalidated: set[str] = set(seeds)
    frontier = sorted(set(seeds))
    while frontier:
        current = frontier.pop(0)
        for child in config.downstream(current):
            reason = f"上游单元 {current} 结果可能变化，派生结果需重算"
            steps.append(
                PropagationStep(source=current, affected=child, reason=reason)
            )
            if child not in invalidated:
                invalidated.add(child)
                frontier.append(child)
                frontier.sort()
    topo_order = [unit_id for unit_id in config.topo_order if unit_id in invalidated]
    return InvalidationResult(
        seeds=sorted(set(seeds)),
        invalidated=sorted(invalidated),
        topo_order=topo_order,
        steps=steps,
    )


def seeds_from_rule_change(
    config: ShardConfig, changed_tables: list[str]
) -> list[str]:
    changed = set(changed_tables)
    return sorted(
        unit_id
        for unit_id, unit in config.units.items()
        if unit.table in changed
    )


def compare_recomputations(
    incremental: dict[str, dict], full: dict[str, dict]
) -> list[dict[str, object]]:
    """逐项比较增量与全量重算结果，返回所有不一致项（空列表表示完全一致）。"""
    differences: list[dict[str, object]] = []
    for unit_id in sorted(set(incremental) | set(full)):
        left = incremental.get(unit_id)
        right = full.get(unit_id)
        if left is None or right is None:
            differences.append(
                {
                    "unit_id": unit_id,
                    "field": "existence",
                    "incremental": left,
                    "full": right,
                }
            )
            continue
        for field_name in sorted(set(left) | set(right)):
            if left.get(field_name) != right.get(field_name):
                differences.append(
                    {
                        "unit_id": unit_id,
                        "field": field_name,
                        "incremental": left.get(field_name),
                        "full": right.get(field_name),
                    }
                )
    return differences
