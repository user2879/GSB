"""Dependency-aware invalidation and stable evidence propagation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .model import Spec


@dataclass(frozen=True)
class InvalidationStep:
    shard_id: str
    node_id: str
    kind: str
    direct_reasons: tuple[str, ...]
    propagation_basis: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "node_id": self.node_id,
            "kind": self.kind,
            "direct_reasons": list(self.direct_reasons),
            "propagation_basis": list(self.propagation_basis),
        }


@dataclass(frozen=True)
class InvalidationPlan:
    invalid_shards: tuple[str, ...]
    preserved_shards: tuple[str, ...]
    removed_shards: tuple[str, ...]
    steps: tuple[InvalidationStep, ...]
    global_changes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "invalid_shards": list(self.invalid_shards),
            "preserved_shards": list(self.preserved_shards),
            "removed_shards": list(self.removed_shards),
            "steps": [step.as_dict() for step in self.steps],
            "global_changes": list(self.global_changes),
        }


def plan_invalidation(old_spec: Spec, new_spec: Spec) -> InvalidationPlan:
    """Compute true result invalidation with direct causes and propagation paths."""

    new_shards = new_spec.shard_map()
    old_shards = old_spec.shard_map()
    all_new_ids = set(new_shards)
    removed = tuple(sorted(set(old_shards) - all_new_ids))
    direct: dict[str, list[str]] = {shard_id: [] for shard_id in all_new_ids}
    global_changes: list[str] = []

    if old_spec.capacity_version != new_spec.capacity_version:
        global_changes.append("capacity_version")
    if old_spec.fixed_scale != new_spec.fixed_scale:
        global_changes.append("fixed_scale")
    if old_spec.global_budget != new_spec.global_budget:
        global_changes.append("global_budget")
    for field_name in (
        "max_workers",
        "lease_ttl_ms",
        "handshake_timeout_ms",
        "io_timeout_ms",
        "poll_interval_ms",
    ):
        if getattr(old_spec, field_name) != getattr(new_spec, field_name):
            global_changes.append(field_name)

    if global_changes:
        reason = "global_declaration_changed:" + ",".join(global_changes)
        for shard_id in all_new_ids:
            direct[shard_id].append(reason)

    old_nodes = old_spec.node_map()
    new_nodes = new_spec.node_map()
    for node_id in sorted(set(old_nodes) | set(new_nodes)):
        old_node = old_nodes.get(node_id)
        new_node = new_nodes.get(node_id)
        if old_node is None or new_node is None:
            reason = "node_declaration_added_or_removed"
        else:
            sample_changed = (
                old_node.seed != new_node.seed or old_node.fixed_scale != new_node.fixed_scale
            )
            quota_changed = (
                old_node.quota != new_node.quota
                or old_node.used != new_node.used
                or old_node.lease_limit != new_node.lease_limit
                or old_node.concurrency != new_node.concurrency
            )
            if sample_changed:
                reason = "node_sample_or_scale_changed"
            elif quota_changed:
                reason = "node_quota_changed"
            else:
                continue
        for shard in new_spec.shards:
            if shard.node_id == node_id:
                direct[shard.id].append(reason)

    for shard_id in sorted(all_new_ids):
        old_shard = old_shards.get(shard_id)
        new_shard = new_shards[shard_id]
        if old_shard is None:
            direct[shard_id].append("shard_added")
            continue
        if old_shard.node_id != new_shard.node_id:
            direct[shard_id].append("shard_node_reference_changed")
        if old_shard.dependencies != new_shard.dependencies:
            direct[shard_id].append("shard_dependencies_changed")
        if old_shard.demand != new_shard.demand:
            direct[shard_id].append("shard_resource_demand_changed")
        if (
            old_shard.sample_count != new_shard.sample_count
            or old_shard.group_count != new_shard.group_count
            or old_shard.batch_size != new_shard.batch_size
        ):
            direct[shard_id].append("shard_sample_declaration_changed")

    invalid: set[str] = {shard_id for shard_id, reasons in direct.items() if reasons}
    # One pass in stable topological order is sufficient because every
    # propagated cause follows the dependency edge.
    for shard_id in new_spec.topological_order():
        if shard_id in invalid:
            continue
        invalid_dependencies = sorted(
            dependency
            for dependency in new_shards[shard_id].dependencies
            if dependency in invalid
        )
        if invalid_dependencies:
            invalid.add(shard_id)

    ordered_invalid = tuple(
        shard_id for shard_id in new_spec.topological_order() if shard_id in invalid
    )
    steps: list[InvalidationStep] = []
    for shard_id in ordered_invalid:
        shard = new_shards[shard_id]
        direct_reasons = tuple(sorted(set(direct[shard_id])))
        invalid_deps = tuple(
            sorted(dependency for dependency in shard.dependencies if dependency in invalid)
        )
        if direct_reasons:
            kind = "direct"
        else:
            kind = "propagated"
        steps.append(
            InvalidationStep(
                shard_id=shard_id,
                node_id=shard.node_id,
                kind=kind,
                direct_reasons=direct_reasons,
                propagation_basis=tuple(f"upstream_invalid:{dep}" for dep in invalid_deps),
            )
        )
    preserved = tuple(
        shard_id for shard_id in new_spec.topological_order() if shard_id not in invalid
    )
    return InvalidationPlan(
        invalid_shards=ordered_invalid,
        preserved_shards=preserved,
        removed_shards=removed,
        steps=tuple(steps),
        global_changes=tuple(sorted(set(global_changes))),
    )
