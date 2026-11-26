"""参数修订驱动的失效传播与仅失效子集重算。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .spec import Revision, ShardSpec, Spec


@dataclass
class PropagationStep:
    invalidated_shard: str
    caused_by_shard: str | None
    reason: str
    depth: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "invalidated_shard": self.invalidated_shard,
            "caused_by_shard": self.caused_by_shard,
            "reason": self.reason,
            "depth": self.depth,
        }


@dataclass
class InvalidationResult:
    revision_version: int
    directly_invalidated: tuple[str, ...]
    invalidated: tuple[str, ...]
    steps: tuple[PropagationStep, ...]
    topo_order: tuple[str, ...]
    reasons: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision_version": self.revision_version,
            "directly_invalidated": list(self.directly_invalidated),
            "invalidated": list(self.invalidated),
            "steps": [step.to_dict() for step in self.steps],
            "topo_order": list(self.topo_order),
            "reasons": dict(sorted(self.reasons.items())),
        }


def stable_topo_order(shards: list[ShardSpec]
                      | tuple[ShardSpec, ...]) -> list[str]:
    """稳定拓扑序：先按上游深度，再按分片标识字典序。"""
    deps = {s.shard_id: set(s.upstream) for s in shards}
    depth: dict[str, int] = {}

    def resolve(shard_id: str) -> int:
        if shard_id in depth:
            return depth[shard_id]
        if not deps[shard_id]:
            depth[shard_id] = 0
        else:
            depth[shard_id] = 1 + max(resolve(dep) for dep in deps[shard_id])
        return depth[shard_id]

    for shard_id in deps:
        resolve(shard_id)
    return sorted(deps, key=lambda sid: (depth[sid], sid))


def effective_dependencies(spec: Spec, revision: Revision) -> dict[str, tuple[str, ...]]:
    """叠加修订后的有效分片依赖（修订只覆盖显式声明的分片）。"""
    deps = {s.shard_id: tuple(s.upstream) for s in spec.shards}
    deps.update(revision.shard_deps)
    return deps


def _downstream_map(deps: dict[str, tuple[str, ...]]) -> dict[str, set[str]]:
    downstream: dict[str, set[str]] = {sid: set() for sid in deps}
    for shard_id, upstream in deps.items():
        for dep in upstream:
            downstream[dep].add(shard_id)
    return downstream


def diff_revision(previous: Revision | None, current: Revision) -> dict[str, list[str]]:
    """列出修订实际改动的维度。"""
    changes: dict[str, list[str]] = {
        "global_rate": [], "global_burst": [], "node_rates": [],
        "node_bursts": [], "shard_deps": [], "scale": [],
    }
    if previous is None:
        if current.global_rate is not None:
            changes["global_rate"].append("<root>")
        if current.global_burst is not None:
            changes["global_burst"].append("<root>")
        if current.scale is not None:
            changes["scale"].append(str(current.scale))
        changes["node_rates"].extend(sorted(current.node_rates))
        changes["node_bursts"].extend(sorted(current.node_bursts))
        changes["shard_deps"].extend(sorted(current.shard_deps))
        return changes
    if current.global_rate is not None and current.global_rate != previous.global_rate:
        changes["global_rate"].append(str(current.global_rate))
    if current.global_burst is not None and current.global_burst != previous.global_burst:
        changes["global_burst"].append(str(current.global_burst))
    if current.scale is not None and current.scale != previous.scale:
        changes["scale"].append(str(current.scale))
    for node_id, value in sorted(current.node_rates.items()):
        if previous.node_rates.get(node_id) != value:
            changes["node_rates"].append(node_id)
    for node_id, value in sorted(current.node_bursts.items()):
        if previous.node_bursts.get(node_id) != value:
            changes["node_bursts"].append(node_id)
    for shard_id, deps in sorted(current.shard_deps.items()):
        if previous.shard_deps.get(shard_id) != deps:
            changes["shard_deps"].append(shard_id)
    return changes


def propagate_invalidation(spec: Spec, previous: Revision | None,
                           current: Revision,
                           shard_nodes: dict[str, str]) -> InvalidationResult:
    """推导直接 + 间接失效分片，输出稳定序与每步依据。"""
    changes = diff_revision(previous, current)
    direct: set[str] = set()
    direct_reasons: dict[str, str] = {}

    def mark_direct(shard_id: str, reason: str) -> None:
        if shard_id not in direct:
            direct.add(shard_id)
            direct_reasons[shard_id] = reason

    # 全局速率/突发/缩放因子变化影响所有分片。
    global_changed = bool(changes["global_rate"] or changes["global_burst"]
                          or changes["scale"])
    if global_changed:
        for shard in spec.shards:
            mark_direct(shard.shard_id, "global_or_scale_parameter_changed")
    # 节点额度变化影响该节点上持有的分片。
    for node_id in sorted(set(changes["node_rates"]) | set(changes["node_bursts"])):
        for shard_id, owner in sorted(shard_nodes.items()):
            if owner == node_id:
                mark_direct(shard_id, f"node_quota_changed:{node_id}")
    # 分片依赖变化：被改分片与受其下游影响者分开记录，直接失效的是被改分片。
    for shard_id in changes["shard_deps"]:
        mark_direct(shard_id, "shard_dependency_edited")

    deps = effective_dependencies(spec, current)
    downstream = _downstream_map(deps)
    topo = stable_topo_order(spec.shards)
    steps: list[PropagationStep] = []
    reasons: dict[str, str] = dict(direct_reasons)
    invalidated: set[str] = set(direct)
    for shard_id in sorted(direct):
        steps.append(PropagationStep(
            invalidated_shard=shard_id, caused_by_shard=None,
            reason=direct_reasons[shard_id], depth=0))

    # 沿下游边做 BFS，同层按分片标识序，保证稳定。
    frontier = sorted(direct)
    depth = 1
    while frontier:
        next_frontier: set[str] = set()
        for shard_id in frontier:
            for child in sorted(downstream.get(shard_id, ())):
                if child not in invalidated:
                    invalidated.add(child)
                    reason = f"downstream_of:{shard_id}"
                    reasons[child] = reason
                    steps.append(PropagationStep(
                        invalidated_shard=child, caused_by_shard=shard_id,
                        reason=reason, depth=depth))
                    next_frontier.add(child)
        frontier = sorted(next_frontier)
        depth += 1

    ordered_invalidated = tuple(sid for sid in topo if sid in invalidated)
    return InvalidationResult(
        revision_version=current.version,
        directly_invalidated=tuple(sorted(direct)),
        invalidated=ordered_invalidated,
        steps=tuple(steps), topo_order=tuple(topo), reasons=reasons,
    )
