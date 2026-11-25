from __future__ import annotations

from typing import Any, Optional

from .canonical import canonical_bytes, sha256_hex
from .ownership import adjudicate_events
from .state_machine import ACTIVE_STATES


def _node_map(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {node["node_id"]: node for node in spec.get("nodes", [])}


def evolution_diff(old_spec: dict[str, Any], new_spec: dict[str, Any]) -> dict[str, Any]:
    old_nodes = _node_map(old_spec)
    new_nodes = _node_map(new_spec)
    removed = sorted(set(old_nodes) - set(new_nodes))
    added = sorted(set(new_nodes) - set(old_nodes))
    changed: list[str] = []
    for node_id in sorted(set(old_nodes) & set(new_nodes)):
        comparable = ("host", "port", "checksum")
        if any(old_nodes[node_id].get(key) != new_nodes[node_id].get(key) for key in comparable):
            changed.append(node_id)
    return {
        "quota_version_changed": old_spec.get("quota_version") != new_spec.get("quota_version"),
        "old_quota_version": old_spec.get("quota_version"),
        "new_quota_version": new_spec.get("quota_version"),
        "nodes_added": added,
        "nodes_removed": removed,
        "nodes_changed": changed,
        "lease_duration_changed": old_spec.get("lease_duration_ms") != new_spec.get("lease_duration_ms"),
    }


def propagate_invalidation(
    old_spec: dict[str, Any],
    new_spec: dict[str, Any],
    events: list[dict[str, Any]],
    before_projection: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    diff = evolution_diff(old_spec, new_spec)
    before = before_projection or adjudicate_events(events, old_spec)
    leases_by_id = {lease["lease_id"]: lease for lease in before["leases"]}
    events_by_lease: dict[str, list[dict[str, Any]]] = {}
    events_by_node: dict[str, list[dict[str, Any]]] = {}
    for event in sorted(events, key=lambda item: int(item["seq"])):
        if event.get("lease_id"):
            events_by_lease.setdefault(event["lease_id"], []).append(event)
        if event.get("node_id"):
            events_by_node.setdefault(event["node_id"], []).append(event)

    affected: dict[str, dict[str, Any]] = {}
    trace: list[dict[str, Any]] = []

    def add_trace(node_id: str, event_seq: Optional[int], lease_id: Optional[str], step: str, basis: Any) -> None:
        trace.append(
            {
                "node_id": node_id or "",
                "event_seq": event_seq,
                "lease_id": lease_id,
                "step": step,
                "basis": basis,
            }
        )

    if diff["quota_version_changed"]:
        add_trace(
            "",
            None,
            None,
            "quota_epoch_changed",
            {"from_version": diff["old_quota_version"], "to_version": diff["new_quota_version"]},
        )
    for node_id in diff["nodes_removed"]:
        add_trace(node_id, None, None, "node_removed", {"node_id": node_id})
    for node_id in diff["nodes_changed"]:
        add_trace(node_id, None, None, "node_declaration_changed", {"node_id": node_id})

    for lease in before["leases"]:
        if lease["state"] not in ACTIVE_STATES:
            continue
        lease_id = lease["lease_id"]
        node_id = lease["node_id"]
        reasons: list[str] = []
        if diff["quota_version_changed"]:
            lease_quota = lease.get("quota_version")
            if lease_quota is None or int(lease_quota) < int(diff["new_quota_version"]):
                reasons.append("old_quota_generation")
        if node_id in diff["nodes_removed"]:
            reasons.append("node_removed")
        if node_id in diff["nodes_changed"]:
            reasons.append("node_declaration_changed")
        if not reasons:
            continue

        first_seq = lease.get("last_event_seq")
        affected[lease_id] = {
            "lease_id": lease_id,
            "node_id": node_id,
            "slot_id": lease["slot_id"],
            "generation": int(lease["generation"]),
            "quota_version": lease.get("quota_version"),
            "first_event_seq": first_seq,
            "reasons": reasons,
        }
        add_trace(node_id, first_seq, lease_id, "node_to_lease", {"reasons": reasons})
        add_trace(
            node_id,
            first_seq,
            lease_id,
            "lease_generation_current",
            {"generation": int(lease["generation"]), "state": lease["state"]},
        )
        for record in lease.get("history", []):
            add_trace(
                node_id,
                record.get("event_seq"),
                lease_id,
                "generation_transition",
                {
                    "before_state": record.get("before_state"),
                    "after_state": record.get("after_state"),
                    "reason": record.get("reason"),
                    "generation_before": record.get("generation_before"),
                    "generation_after": record.get("generation_after"),
                },
            )
        for event in events_by_lease.get(lease_id, []):
            add_trace(
                node_id,
                int(event["seq"]),
                lease_id,
                "event_reference",
                {"kind": event["kind"], "slot_id": event.get("slot_id"), "claim_id": event.get("claim_id")},
            )

    ordered_affected = sorted(
        affected.values(),
        key=lambda item: (item["node_id"], item["first_event_seq"] or 0, item["lease_id"]),
    )
    trace = sorted(
        trace,
        key=lambda item: (
            item["node_id"],
            item["event_seq"] if item["event_seq"] is not None else -1,
            item["lease_id"] or "",
            item["step"],
        ),
    )

    max_seq = max((int(event["seq"]) for event in events), default=0)
    fence_events: list[dict[str, Any]] = []
    for offset, item in enumerate(ordered_affected, start=1):
        fence_events.append(
            {
                "seq": max_seq + offset,
                "kind": "lease_expired",
                "lease_id": item["lease_id"],
                "node_id": item["node_id"],
                "slot_id": item["slot_id"],
                "generation": item["generation"],
                "reason": "quota_fenced",
                "evolution_fence": True,
            }
        )

    after = adjudicate_events(events + fence_events, new_spec)
    after_leases = {lease["lease_id"]: lease for lease in after["leases"]}
    affected_ids = set(affected)
    proofs: list[dict[str, Any]] = []
    for lease_id in sorted(set(leases_by_id) | set(after_leases)):
        if lease_id in affected_ids:
            continue
        before_bytes = canonical_bytes(leases_by_id[lease_id])
        after_bytes = canonical_bytes(after_leases[lease_id])
        proofs.append(
            {
                "item": f"lease:{lease_id}",
                "identical": before_bytes == after_bytes,
                "byte_length": len(before_bytes),
                "sha256_before": sha256_hex(before_bytes),
                "sha256_after": sha256_hex(after_bytes),
            }
        )

    affected_nodes = {item["node_id"] for item in ordered_affected}
    before_nodes = before["capacity"]["by_node"]
    after_nodes = after["capacity"]["by_node"]
    for node_id in sorted(set(before_nodes) | set(after_nodes)):
        if node_id in affected_nodes:
            continue
        before_bytes = canonical_bytes(before_nodes.get(node_id))
        after_bytes = canonical_bytes(after_nodes.get(node_id))
        proofs.append(
            {
                "item": f"capacity_node:{node_id}",
                "identical": before_bytes == after_bytes,
                "byte_length": len(before_bytes),
                "sha256_before": sha256_hex(before_bytes),
                "sha256_after": sha256_hex(after_bytes),
            }
        )

    before_owners = {owner["slot_id"]: owner for owner in before["owners"]}
    after_owners = {owner["slot_id"]: owner for owner in after["owners"]}
    affected_slots = {item["slot_id"] for item in ordered_affected}
    for slot_id in sorted(set(before_owners) | set(after_owners)):
        if slot_id in affected_slots:
            continue
        before_bytes = canonical_bytes(before_owners.get(slot_id))
        after_bytes = canonical_bytes(after_owners.get(slot_id))
        proofs.append(
            {
                "item": f"ownership:{slot_id}",
                "identical": before_bytes == after_bytes,
                "byte_length": len(before_bytes),
                "sha256_before": sha256_hex(before_bytes),
                "sha256_after": sha256_hex(after_bytes),
            }
        )

    return {
        "evolution": diff,
        "invalid_set": ordered_affected,
        "trace": trace,
        "fence_events": fence_events,
        "subset_recomputed": [item["lease_id"] for item in ordered_affected],
        "unaffected_proofs": proofs,
        "all_unaffected_bytes_identical": all(item["identical"] for item in proofs),
        "before_capacity": before["capacity"],
        "after_capacity": after["capacity"],
        "after_projection": after,
    }
