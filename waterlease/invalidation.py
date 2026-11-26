from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .spec import SpecBundle
from .topology import reverse_dependency_map, topological_order


def _index(items: Any, key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        data = item if isinstance(item, dict) else asdict(item)
        result[data[key]] = data
    return result


def _changed_fields(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    fields = sorted(set(old) | set(new))
    return [field for field in fields if old.get(field) != new.get(field)]


def plan_invalidation(old: SpecBundle, new: SpecBundle) -> dict[str, Any]:
    """Derive direct and transitive invalid shards between two specs."""

    order = topological_order(new.shards)
    rank = {shard_id: index for index, shard_id in enumerate(order)}
    reverse_deps = reverse_dependency_map(new.shards)
    new_shards = _index(new.shards, "shard_id")
    old_shards = _index(old.shards, "shard_id")
    new_nodes = _index(new.nodes, "node_id")
    old_nodes = _index(old.nodes, "node_id")
    new_caps = _index(new.capacity_versions, "version")
    old_caps = _index(old.capacity_versions, "version")

    reasons: dict[str, list[dict[str, Any]]] = {shard_id: [] for shard_id in new_shards}

    def add_reason(shard_id: str, reason: dict[str, Any]) -> None:
        if shard_id in reasons and reason not in reasons[shard_id]:
            reasons[shard_id].append(reason)

    removed_shards = sorted(set(old_shards) - set(new_shards))
    added_shards = sorted(set(new_shards) - set(old_shards))
    for shard_id in added_shards:
        add_reason(
            shard_id,
            {
                "basis": "shard_added",
                "path": f"$.shards[{shard_id}]",
                "detail": "new shard has no committed conclusion",
            },
        )

    for shard_id in sorted(set(old_shards) & set(new_shards)):
        changed = _changed_fields(old_shards[shard_id], new_shards[shard_id])
        if changed:
            add_reason(
                shard_id,
                {
                    "basis": "shard_spec_changed",
                    "path": f"$.shards[{shard_id}]",
                    "changed_fields": changed,
                },
            )

    changed_nodes: dict[str, list[str]] = {}
    for node_id in sorted(set(old_nodes) & set(new_nodes)):
        changed = _changed_fields(old_nodes[node_id], new_nodes[node_id])
        if changed:
            changed_nodes[node_id] = changed
    for shard_id, shard in new_shards.items():
        node_id = shard["node_id"]
        if node_id in changed_nodes:
            add_reason(
                shard_id,
                {
                    "basis": "node_reference_changed",
                    "path": f"$.nodes[{node_id}]",
                    "source_node_id": node_id,
                    "changed_fields": changed_nodes[node_id],
                },
            )

    changed_cap_versions: dict[str, list[str]] = {}
    for version in sorted(set(old_caps) & set(new_caps)):
        changed = _changed_fields(old_caps[version], new_caps[version])
        if changed:
            changed_cap_versions[version] = changed
    added_cap_versions = sorted(set(new_caps) - set(old_caps))
    for shard_id, shard in new_shards.items():
        version = shard["capacity_version"]
        if version in changed_cap_versions:
            add_reason(
                shard_id,
                {
                    "basis": "capacity_version_changed",
                    "path": f"$.capacity_versions[{version}]",
                    "capacity_version": version,
                    "changed_fields": changed_cap_versions[version],
                },
            )
        if version in added_cap_versions:
            add_reason(
                shard_id,
                {
                    "basis": "capacity_reference_changed",
                    "path": f"$.shards[{shard_id}].capacity_version",
                    "capacity_version": version,
                },
            )

    session_changed = [
        field_name
        for field_name in _changed_fields(asdict(old.session), asdict(new.session))
        if field_name != "spec_version"
    ]
    if session_changed:
        for shard_id in new_shards:
            add_reason(
                shard_id,
                {
                    "basis": "session_water_config_changed",
                    "path": "$.session",
                    "changed_fields": session_changed,
                },
            )

    invalid = {shard_id for shard_id, items in reasons.items() if items}

    propagation_steps: list[dict[str, Any]] = []
    for shard_id in order:
        shard = new_shards[shard_id]
        if shard_id not in invalid:
            for upstream in sorted(shard["upstreams"], key=lambda item: rank.get(item, 10**9)):
                if upstream in invalid:
                    invalid.add(shard_id)
                    reason = {
                        "basis": "dependency_upstream_invalid",
                        "path": f"$.shards[{shard_id}].upstreams",
                        "source_shard_id": upstream,
                        "edge": f"{upstream}->{shard_id}",
                    }
                    add_reason(shard_id, reason)
                    break

    invalid_ordered = sorted(invalid, key=lambda shard_id: (rank[shard_id], shard_id))
    for sequence, shard_id in enumerate(invalid_ordered):
        reason_list = reasons[shard_id]
        primary = reason_list[0]["basis"] if reason_list else "dependency_upstream_invalid"
        propagation_steps.append(
            {
                "sequence": sequence,
                "shard_id": shard_id,
                "topo_rank": rank[shard_id],
                "primary_basis": primary,
                "reasons": reason_list,
            }
        )

    unaffected = [shard_id for shard_id in order if shard_id not in invalid]
    return {
        "old_spec_digest": old.spec_digest,
        "new_spec_digest": new.spec_digest,
        "topological_order": order,
        "invalid_set": invalid_ordered,
        "unaffected_set": unaffected,
        "removed_shards": removed_shards,
        "added_shards": added_shards,
        "propagation_steps": propagation_steps,
        "reverse_dependencies": {key: list(value) for key, value in reverse_deps.items()},
        "recompute_policy": "replace current conclusion as a whole; append audit revision",
    }
