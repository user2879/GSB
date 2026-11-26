from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .spec import SessionSpec


def dependent_map(spec: SessionSpec) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {shard.shard_id: [] for shard in spec.shards}
    for shard in spec.shards:
        for dependency in shard.depends_on:
            if dependency in result:
                result[dependency].append(shard.shard_id)
    for key in result:
        result[key].sort()
    return result


def transitive_descendants(spec: SessionSpec, root_ids: set[str]) -> tuple[set[str], list[dict[str, str]]]:
    dependents = dependent_map(spec)
    invalid = set(root_ids)
    steps: list[dict[str, str]] = []
    queue = [shard_id for shard_id in spec.topo_order if shard_id in root_ids]
    processed: set[str] = set()
    while queue:
        current = queue.pop(0)
        if current in processed:
            continue
        processed.add(current)
        for dependent in dependents.get(current, []):
            steps.append(
                {
                    "from_shard_id": current,
                    "to_shard_id": dependent,
                    "basis": f"dependency_edge:{current}:depends_on?reverse->{dependent}",
                }
            )
            if dependent not in invalid:
                invalid.add(dependent)
                queue.append(dependent)
        queue.sort(key=lambda sid: spec.topo_order.index(sid))
    return invalid, steps


@dataclass(frozen=True)
class InvalidationPlan:
    direct_shard_ids: tuple[str, ...]
    invalid_shard_ids: tuple[str, ...]
    unchanged_shard_ids: tuple[str, ...]
    propagation_steps: tuple[dict[str, Any], ...]
    direct_reasons: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "direct_shard_ids": list(self.direct_shard_ids),
            "invalid_shard_ids": list(self.invalid_shard_ids),
            "unchanged_shard_ids": list(self.unchanged_shard_ids),
            "propagation_steps": list(self.propagation_steps),
            "direct_reasons": list(self.direct_reasons),
            "stable_order": "topological_rank_then_shard_id",
        }


def compare_specs(old: SessionSpec, new: SessionSpec) -> InvalidationPlan:
    old_nodes = {node.node_id: node for node in old.nodes}
    new_nodes = {node.node_id: node for node in new.nodes}
    old_shards = {shard.shard_id: shard for shard in old.shards}
    new_shards = {shard.shard_id: shard for shard in new.shards}

    direct: dict[str, dict[str, Any]] = {}

    def add_direct(shard_id: str, reason: str, field: str, before: Any, after: Any) -> None:
        if shard_id not in new_shards:
            return
        direct[shard_id] = {
            "shard_id": shard_id,
            "reason": reason,
            "field": field,
            "before": before,
            "after": after,
        }

    for node_id, new_node in new_nodes.items():
        old_node = old_nodes.get(node_id)
        if old_node is None:
            for shard in new.shards_for_node(node_id):
                add_direct(shard.shard_id, "node_added", "nodes", None, node_id)
            continue
        if old_node.capacity_version != new_node.capacity_version:
            for shard in new.shards_for_node(node_id):
                add_direct(
                    shard.shard_id,
                    "capacity_version_changed",
                    f"nodes.{node_id}.capacity_version",
                    old_node.capacity_version,
                    new_node.capacity_version,
                )
        for field_name in ("concurrency_quota", "lease_limit"):
            before = getattr(old_node, field_name)
            after = getattr(new_node, field_name)
            if before != after:
                for shard in new.shards_for_node(node_id):
                    add_direct(
                        shard.shard_id,
                        "node_quota_parameter_changed",
                        f"nodes.{node_id}.{field_name}",
                        before,
                        after,
                    )

    if old.scale_power != new.scale_power or old.accounting != new.accounting:
        for shard in new.shards:
            add_direct(
                shard.shard_id,
                "fixed_point_accounting_changed",
                "scale_power_or_accounting",
                {"scale_power": old.scale_power, "accounting": old.accounting.__dict__},
                {"scale_power": new.scale_power, "accounting": new.accounting.__dict__},
            )

    for shard_id, new_shard in new_shards.items():
        old_shard = old_shards.get(shard_id)
        if old_shard is None:
            add_direct(shard_id, "shard_added", "shards", None, shard_id)
            continue
        semantic_fields = (
            "node_id",
            "depends_on",
            "sample_rows",
            "resource_units",
            "planned_renewals",
        )
        for field_name in semantic_fields:
            before = getattr(old_shard, field_name)
            after = getattr(new_shard, field_name)
            if before != after:
                add_direct(
                    shard_id,
                    "shard_semantic_field_changed",
                    f"shards.{shard_id}.{field_name}",
                    list(before) if isinstance(before, tuple) else before,
                    list(after) if isinstance(after, tuple) else after,
                )
                break

    root_ids = set(direct)
    invalid, edge_steps = transitive_descendants(new, root_ids)
    topo_index = {shard_id: index for index, shard_id in enumerate(new.topo_order)}
    invalid_ordered = tuple(sorted(invalid, key=lambda sid: (topo_index[sid], sid)))
    direct_ordered = tuple(sorted(root_ids, key=lambda sid: (topo_index[sid], sid)))
    unchanged = tuple(sid for sid in new.topo_order if sid not in invalid)

    direct_reasons = tuple(direct[sid] for sid in direct_ordered)
    propagation_steps: list[dict[str, Any]] = []
    for reason in direct_reasons:
        propagation_steps.append(
            {
                "shard_id": reason["shard_id"],
                "step": "direct_invalidation",
                "basis": f"{reason['reason']}:{reason['field']}",
            }
        )
    propagation_steps.extend(edge_steps)
    propagation_steps.sort(key=lambda step: (topo_index.get(step.get("to_shard_id", step.get("shard_id")), 10**9), step["basis"]))

    return InvalidationPlan(
        direct_shard_ids=direct_ordered,
        invalid_shard_ids=invalid_ordered,
        unchanged_shard_ids=unchanged,
        propagation_steps=tuple(propagation_steps),
        direct_reasons=direct_reasons,
    )
