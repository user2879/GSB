"""Stable invalidation propagation for topology and quota versions."""

from __future__ import annotations

from typing import Any

from .statemachine import ACTIVE_STATES


def changed_node_ids(event: dict[str, Any], known_node_ids: list[str]) -> list[str]:
    nodes = set(event.get("changed_node_ids", []))
    nodes.update(event.get("removed_node_ids", []))
    if event.get("scope") == "global":
        nodes.update(known_node_ids)
    return sorted(nodes)


def build_propagation_report(
    *,
    event: dict[str, Any],
    leases: dict[str, dict[str, Any]],
    known_node_ids: list[str],
    topology_version: int,
    quota_version: int,
) -> dict[str, Any]:
    changed_nodes = changed_node_ids(event, known_node_ids)
    direct_lease_ids = sorted(
        lease_id
        for lease_id, lease in leases.items()
        if lease["node_id"] in changed_nodes and lease["state"] in ACTIVE_STATES
    )

    steps: list[dict[str, Any]] = []
    order = 1
    for node_id in changed_nodes:
        node_leases = sorted(
            lease_id for lease_id, lease in leases.items() if lease["node_id"] == node_id
        )
        current_refs = [
            f"lease:{lease_id}#generation={leases[lease_id]['generation']}"
            for lease_id in node_leases
        ]
        steps.append(
            {
                "order": order,
                "kind": "changed_node",
                "ref": f"node:{node_id}",
                "basis": "node declared in changed_node_ids/removed_node_ids or global scope",
                "lease_refs": current_refs,
            }
        )
        order += 1

    for lease_id in direct_lease_ids:
        lease = leases[lease_id]
        generation_refs: list[str] = []
        event_refs: list[str] = []
        for item in lease["history"]:
            before = item["generation_before"]
            after = item["generation_after"]
            if before is not None:
                generation_refs.append(f"lease:{lease_id}#generation={before}")
            generation_refs.append(f"lease:{lease_id}#generation={after}")
            event_refs.append(f"event:{item['event_seq']}")
        steps.append(
            {
                "order": order,
                "kind": "lease_generation_chain",
                "ref": f"lease:{lease_id}",
                "basis": "monotonic generation labels referenced by transition records",
                "generation_refs": sorted(set(generation_refs)),
                "event_refs": event_refs,
            }
        )
        order += 1

        equity = lease["equity_id"]
        peer_ids = sorted(
            other_id
            for other_id, other in leases.items()
            if other["equity_id"] == equity and other_id != lease_id
        )
        if peer_ids:
            steps.append(
                {
                    "order": order,
                    "kind": "equity_event_reference",
                    "ref": f"equity:{equity}",
                    "basis": "ownership and event references for the same capacity equity",
                    "peer_leases": peer_ids,
                }
            )
            order += 1

    unchanged = sorted(set(leases) - set(direct_lease_ids))
    return {
        "entry_kind": "propagation",
        "event_seq": event["seq"],
        "trigger": event["type"],
        "requested_topology_version": event.get("topology_version", topology_version),
        "requested_quota_version": event.get("quota_version", quota_version),
        "topology_version": topology_version,
        "quota_version": quota_version,
        "changed_node_ids": changed_nodes,
        "affected_leases": direct_lease_ids,
        "invalidation_set": direct_lease_ids,
        "unchanged_leases": unchanged,
        "steps": steps,
    }
