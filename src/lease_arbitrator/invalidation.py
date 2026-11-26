from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Iterable


def _lease_node(spec: dict[str, Any]) -> dict[str, str]:
    return {lease["lease_id"]: lease["node_id"] for lease in spec["leases"]}


def build_propagation_graph(
    spec: dict[str, Any],
    events: list[dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, str]]]:
    """Edges mean: if source is invalid, target is invalid."""
    adjacency: dict[str, list[dict[str, str]]] = defaultdict(list)
    lease_node = _lease_node(spec)

    for node in spec["nodes"]:
        node_key = f"node:{node['node_id']}"
        for lease in spec["leases"]:
            if lease["node_id"] == node["node_id"]:
                adjacency[node_key].append(
                    {
                        "target": f"lease:{lease['lease_id']}",
                        "lease_id": lease["lease_id"],
                        "node_id": node["node_id"],
                        "reason": "node_membership",
                        "basis": f"lease.target_node == {node['node_id']}",
                    }
                )

    for version in spec.get("quota_versions", []):
        version_key = f"version:{version['quota_version']}"
        for lease in spec["leases"]:
            if int(lease.get("quota_version", 1)) <= int(version["quota_version"]):
                adjacency[version_key].append(
                    {
                        "target": f"lease:{lease['lease_id']}",
                        "lease_id": lease["lease_id"],
                        "node_id": lease["node_id"],
                        "reason": "quota_version_binding",
                        "basis": (
                            f"lease.quota_version={lease.get('quota_version', 1)} "
                            f"bound_to_policy<={version['quota_version']}"
                        ),
                    }
                )

    by_id = {lease["lease_id"]: lease for lease in spec["leases"]}
    for lease in spec["leases"]:
        source_key = f"lease:{lease['lease_id']}"
        for ref in lease.get("references", []):
            if ref in by_id:
                # If referenced lease is deleted, the depending lease is invalid.
                adjacency[f"lease:{ref}"].append(
                    {
                        "target": source_key,
                        "lease_id": lease["lease_id"],
                        "node_id": lease["node_id"],
                        "reason": "lease_reference",
                        "basis": f"{lease['lease_id']}.references includes {ref}",
                    }
                )
                # Generation/reference chain also exposes the reverse audit edge.
                adjacency[source_key].append(
                    {
                        "target": f"lease:{ref}",
                        "lease_id": ref,
                        "node_id": by_id[ref]["node_id"],
                        "reason": "generation_reference_reverse",
                        "basis": f"{lease['lease_id']} declares generation dependency on {ref}",
                    }
                )

    for event in events or []:
        seq = event["seq"]
        event_key = f"event:{seq}"
        if event.get("lease_id"):
            lease_id = event["lease_id"]
            adjacency[f"lease:{lease_id}"].append(
                {
                    "target": event_key,
                    "lease_id": lease_id,
                    "reason": "event_reference",
                    "basis": f"event {seq} references lease {lease_id}",
                }
            )
            adjacency[event_key].append(
                {
                    "target": f"lease:{lease_id}",
                    "lease_id": lease_id,
                    "reason": "event_lease_binding",
                    "basis": f"event {seq} acts on lease {lease_id}",
                }
            )
        if event.get("type") == "upgrade_quota":
            version_key = f"version:{event['quota_version']}"
            adjacency[version_key].append(
                {
                    "target": event_key,
                    "reason": "version_upgrade_event",
                    "basis": f"event {seq} upgrades quota_version to {event['quota_version']}",
                }
            )

    return dict(adjacency)


def propagate_invalidation(
    spec: dict[str, Any],
    seeds: Iterable[str],
    *,
    events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    graph = build_propagation_graph(spec, events)
    lease_node = _lease_node(spec)
    seed_list = list(seeds)
    queue: deque[str] = deque(seed_list)
    visited: set[str] = set(seed_list)
    steps: list[dict[str, Any]] = []
    invalid_leases: set[str] = set()

    for seed in seed_list:
        if seed.startswith("lease:"):
            invalid_leases.add(seed.split(":", 1)[1])

    while queue:
        source = queue.popleft()
        edges = graph.get(source, [])
        edges = sorted(
            edges,
            key=lambda edge: (
                edge.get("node_id", "~"),
                edge.get("lease_id", "~"),
                edge.get("reason", ""),
                edge.get("target", ""),
            ),
        )
        for edge in edges:
            target = edge["target"]
            steps.append({"from": source, **edge})
            if target not in visited:
                visited.add(target)
                queue.append(target)
            if target.startswith("lease:"):
                invalid_leases.add(target.split(":", 1)[1])

    ordered_leases = sorted(invalid_leases, key=lambda lease_id: (lease_node.get(lease_id, "~"), lease_id))
    ordered_steps = sorted(
        steps,
        key=lambda step: (
            step.get("node_id", "~"),
            step.get("lease_id", "~"),
            step.get("from", ""),
            step.get("reason", ""),
        ),
    )
    return {
        "seeds": sorted(seed_list),
        "invalid_lease_ids": ordered_leases,
        "invalid_set": [
            {
                "node_id": lease_node.get(lease_id),
                "lease_id": lease_id,
            }
            for lease_id in ordered_leases
        ],
        "propagation_steps": ordered_steps,
        "ordering": "node_id_asc,then_lease_id_asc",
        "recompute_scope": {
            "only_subset": ordered_leases,
            "stable_output": True,
            "current_view": "whole_replacement_not_merge",
        },
    }


def plan_edit_invalidation(
    spec: dict[str, Any],
    edit: dict[str, Any],
    *,
    events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    edit_type = edit.get("edit_type")
    if edit_type == "remove_node":
        node_id = edit["node_id"]
        result = propagate_invalidation(spec, [f"node:{node_id}"], events=events)
        result["edit"] = {"edit_type": edit_type, "node_id": node_id}
    elif edit_type == "delete_lease":
        lease_id = edit["lease_id"]
        result = propagate_invalidation(spec, [f"lease:{lease_id}"], events=events)
        result["edit"] = {"edit_type": edit_type, "lease_id": lease_id}
    elif edit_type == "upgrade_quota":
        version = int(edit["quota_version"])
        result = propagate_invalidation(spec, [f"version:{version}"], events=events)
        result["edit"] = {"edit_type": edit_type, "quota_version": version}
        # A quota replacement revokes all still-active old-version leases, even if
        # graph closure includes historical/dependent leases too.
        result["whole_replacement_revoke_rule"] = (
            "all unexpired active leases with quota_version < new version"
        )
    elif edit_type == "upsert_node":
        result = {
            "edit": {"edit_type": edit_type, "node_id": edit["node"]["node_id"]},
            "seeds": [],
            "invalid_lease_ids": [],
            "invalid_set": [],
            "propagation_steps": [],
            "ordering": "node_id_asc,then_lease_id_asc",
            "recompute_scope": {
                "only_subset": [],
                "capacity_recompute": True,
                "current_view": "whole_replacement_not_merge",
            },
        }
    else:
        raise ValueError(f"unsupported edit_type {edit_type!r}")
    return result
