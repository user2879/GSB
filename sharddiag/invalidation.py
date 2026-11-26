"""Precise invalidation propagation through node references and dependencies."""

from __future__ import annotations

from typing import Any, Iterable

from .canonical import canonical_digest
from .spec import Spec


def changed_nodes_between_versions(spec: Spec, old_version: int, new_version: int) -> list[str]:
    old = spec.capacity_versions[old_version]["node_calibers"]
    new = spec.capacity_versions[new_version]["node_calibers"]
    changed: list[str] = []
    for node_id in sorted(spec.nodes):
        if canonical_digest(old[node_id]) != canonical_digest(new[node_id]):
            changed.append(node_id)
    return changed


def derive_invalidation(
    spec: Spec,
    *,
    changed_node_ids: Iterable[str] | None = None,
    direct_shard_ids: Iterable[str] | None = None,
    reason: str = "capacity_or_quota_generation_changed",
) -> dict[str, Any]:
    changed_nodes = set(changed_node_ids or [])
    direct: set[str] = set(direct_shard_ids or [])
    steps: list[dict[str, Any]] = []

    for node_id in sorted(changed_nodes):
        for shard_id in spec.topo_order:
            if spec.shards[shard_id]["node_id"] == node_id:
                direct.add(shard_id)
                steps.append(
                    {
                        "step": len(steps) + 1,
                        "kind": "direct_node_reference",
                        "node_id": node_id,
                        "shard_id": shard_id,
                        "basis": f"shard {shard_id!r} references changed node {node_id!r}",
                    }
                )

    for shard_id in sorted(direct, key=lambda item: (spec.topo_rank[item], item)):
        steps.append(
            {
                "step": len(steps) + 1,
                "kind": "direct_change",
                "shard_id": shard_id,
                "basis": f"shard {shard_id!r} is explicitly in the changed set",
            }
        )

    invalid = set(direct)
    children: dict[str, list[str]] = {shard_id: [] for shard_id in spec.shards}
    for shard_id, shard in spec.shards.items():
        for upstream in shard["depends_on"]:
            children[upstream].append(shard_id)

    for upstream in spec.topo_order:
        if upstream not in invalid:
            continue
        for child in sorted(children[upstream], key=lambda item: (spec.topo_rank[item], item)):
            if child not in invalid:
                invalid.add(child)
                steps.append(
                    {
                        "step": len(steps) + 1,
                        "kind": "dependency_propagation",
                        "from_shard": upstream,
                        "to_shard": child,
                        "edge": f"{child}.depends_on includes {upstream}",
                        "basis": f"downstream {child!r} consumes changed upstream {upstream!r}",
                    }
                )

    invalid_ordered = sorted(invalid, key=lambda item: (spec.topo_rank[item], item))
    unaffected = sorted(set(spec.shards) - invalid, key=lambda item: (spec.topo_rank[item], item))
    return {
        "reason": reason,
        "changed_node_ids": sorted(changed_nodes),
        "direct_shard_ids": sorted(direct, key=lambda item: (spec.topo_rank[item], item)),
        "invalid_shard_ids": invalid_ordered,
        "unaffected_shard_ids": unaffected,
        "propagation_steps": steps,
        "ordering_rule": "stable_kahn_topological_rank_then_lexicographic_shard_id",
    }


def record_hash(record: dict[str, Any]) -> str:
    return canonical_digest(record)


def hash_records(records: dict[str, dict[str, Any]]) -> dict[str, str]:
    return {shard_id: record_hash(records[shard_id]) for shard_id in sorted(records)}


def verify_byte_unchanged(
    before: dict[str, str], after: dict[str, str], *, allowed_changed: Iterable[str]
) -> dict[str, Any]:
    allowed = set(allowed_changed)
    changed = sorted(shard_id for shard_id in before if before.get(shard_id) != after.get(shard_id))
    missing_after = sorted(shard_id for shard_id in before if shard_id not in after)
    unexpected = sorted(shard_id for shard_id in changed if shard_id not in allowed) + missing_after
    return {
        "ok": not unexpected,
        "changed_shard_ids": changed,
        "allowed_changed_shard_ids": sorted(allowed),
        "unexpected_changed_or_missing": sorted(set(unexpected)),
        "verification": "canonical_record_sha256_equality_for_each_unaffected_shard",
    }
