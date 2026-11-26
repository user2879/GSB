"""Failure propagation along the dependency graph with full-rebuild parity."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import DivergeError
from .spec import Spec, topological_order


@dataclass
class InvalidationStep:
    shard_id: str
    reason: str
    caused_by: str | None
    depth: int

    def to_dict(self) -> dict:
        return {
            "shard_id": self.shard_id,
            "reason": self.reason,
            "caused_by": self.caused_by,
            "depth": self.depth,
        }


def direct_invalidations(old_spec: Spec, new_spec: Spec) -> list[InvalidationStep]:
    """Shards directly invalidated by spec/parameter/resource changes."""
    steps: list[InvalidationStep] = []
    old_by_id = {shard.shard_id: shard for shard in old_spec.shards}
    new_by_id = {shard.shard_id: shard for shard in new_spec.shards}
    for shard_id in sorted(set(old_by_id) | set(new_by_id)):
        before = old_by_id.get(shard_id)
        after = new_by_id.get(shard_id)
        if before is None and after is not None:
            steps.append(InvalidationStep(shard_id, "shard_added", None, 0))
        elif before is not None and after is None:
            steps.append(InvalidationStep(shard_id, "shard_removed", None, 0))
        elif before is not None and after is not None:
            if before.sample_rows != after.sample_rows:
                steps.append(
                    InvalidationStep(shard_id, "sample_rows_changed", None, 0)
                )
            if before.resource_units != after.resource_units:
                steps.append(
                    InvalidationStep(shard_id, "resource_units_changed", None, 0)
                )
            if before.depends_on != after.depends_on:
                steps.append(
                    InvalidationStep(shard_id, "dependency_edges_changed", None, 0)
                )
            if before.priority != after.priority:
                steps.append(
                    InvalidationStep(shard_id, "priority_changed", None, 0)
                )
    if old_spec.capacity_versions() != new_spec.capacity_versions():
        for shard_id in sorted(new_by_id):
            steps.append(InvalidationStep(shard_id, "capacity_versions_changed", None, 0))
    else:
        changed_nodes = _quota_changed_nodes(old_spec, new_spec)
        if changed_nodes:
            for shard_id in sorted(new_by_id):
                steps.append(
                    InvalidationStep(
                        shard_id,
                        "quota_parameters_changed",
                        ",".join(changed_nodes),
                        0,
                    )
                )
    return _dedupe(steps)


def _quota_changed_nodes(old_spec: Spec, new_spec: Spec) -> list[str]:
    changed: list[str] = []
    old_nodes = {node.node_id: node for node in old_spec.nodes}
    new_nodes = {node.node_id: node for node in new_spec.nodes}
    for node_id in sorted(set(old_nodes) | set(new_nodes)):
        before = old_nodes.get(node_id)
        after = new_nodes.get(node_id)
        if before is None or after is None:
            changed.append(node_id)
            continue
        old_levels = {(level.version, level.concurrency, level.lease_limit) for level in before.capacities}
        new_levels = {(level.version, level.concurrency, level.lease_limit) for level in after.capacities}
        if old_levels != new_levels or before.seed != after.seed or before.port != after.port:
            changed.append(node_id)
    return changed


def _dedupe(steps: list[InvalidationStep]) -> list[InvalidationStep]:
    seen: set[str] = set()
    unique: list[InvalidationStep] = []
    for step in steps:
        if step.shard_id in seen:
            continue
        seen.add(step.shard_id)
        unique.append(step)
    return unique


def propagate(
    spec: Spec, direct: list[InvalidationStep]
) -> list[InvalidationStep]:
    """Expand direct invalidation through dependents in stable topo order."""
    invalid: dict[str, InvalidationStep] = {
        step.shard_id: step for step in _dedupe(direct)
    }
    dependents: dict[str, list[str]] = {shard.shard_id: [] for shard in spec.shards}
    for shard in spec.shards:
        for upstream in shard.depends_on:
            dependents.setdefault(upstream, []).append(shard.shard_id)

    order = topological_order(spec)
    changed = True
    while changed:
        changed = False
        for shard_id in order:
            if shard_id in invalid:
                continue
            shard = spec.shard(shard_id)
            for upstream in sorted(shard.depends_on):
                if upstream in invalid:
                    depth = invalid[upstream].depth + 1
                    invalid[shard_id] = InvalidationStep(
                        shard_id,
                        "upstream_invalid",
                        upstream,
                        depth,
                    )
                    changed = True
                    break
    result: list[InvalidationStep] = []
    depth_groups: dict[int, list[str]] = {}
    for shard_id, step in invalid.items():
        depth_groups.setdefault(step.depth, []).append(shard_id)
    for depth in sorted(depth_groups):
        for shard_id in sorted(depth_groups[depth]):
            result.append(invalid[shard_id])
    return result


def plan_invalidation(old_spec: Spec, new_spec: Spec) -> list[dict]:
    direct = direct_invalidations(old_spec, new_spec)
    steps = propagate(new_spec, direct)
    return [step.to_dict() for step in steps]


def compare_field_by_field(
    incremental: dict, full_rebuild: dict, *, context: str
) -> None:
    """Raise E_DIVERGE at the first differing field in deterministic key order."""
    keys = sorted(set(incremental) | set(full_rebuild))
    for key in keys:
        path = f"{context}.{key}"
        if key not in incremental:
            raise DivergeError(f"missing field in incremental result: {key}", path=path)
        if key not in full_rebuild:
            raise DivergeError(f"missing field in full rebuild: {key}", path=path)
        left = incremental[key]
        right = full_rebuild[key]
        if isinstance(left, dict) and isinstance(right, dict):
            compare_field_by_field(left, right, context=path)
        elif isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                raise DivergeError(
                    f"list length differs: {len(left)} != {len(right)}", path=path
                )
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                if isinstance(left_item, dict) and isinstance(right_item, dict):
                    compare_field_by_field(
                        left_item, right_item, context=f"{path}[{index}]"
                    )
                elif left_item != right_item:
                    raise DivergeError(
                        f"list item differs: {left_item!r} != {right_item!r}",
                        path=f"{path}[{index}]",
                    )
        elif left != right:
            raise DivergeError(
                f"value differs: {left!r} != {right!r}", path=path
            )


def assert_rebuild_parity(
    incremental: dict, full_rebuild: dict, *, invalidated: list[str]
) -> None:
    compare_field_by_field(
        {"summaries": incremental["summaries"], "ownership": incremental["ownership"]},
        {"summaries": full_rebuild["summaries"], "ownership": full_rebuild["ownership"]},
        context="rebuild",
    )
    compare_field_by_field(
        {"water_marks": incremental["water_marks"]},
        {"water_marks": full_rebuild["water_marks"]},
        context="rebuild",
    )
    compare_field_by_field(
        {"invalidated": invalidated, "basis": incremental["propagation_basis"]},
        {
            "invalidated": full_rebuild["invalidated"],
            "basis": full_rebuild["propagation_basis"],
        },
        context="rebuild",
    )
