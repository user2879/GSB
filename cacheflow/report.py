from __future__ import annotations

from collections import defaultdict
from typing import Any

from . import FORMAT_VERSION
from .canonical import canonical_bytes, sha256_hex
from .engine import Engine
from .spec import Bundle


def _sorted_transitions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (int(row["event_seq"]), int(row.get("ordinal", 0)), str(row["lease_id"])))


def _sorted_decisions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (int(row["event_seq"]), int(row.get("ordinal", 0)), str(row["decision_id"])))


def _sorted_reclaims(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (int(row["event_seq"]), int(row.get("ordinal", 0)), str(row["reclaim_id"])))


def build_conclusion(
    *,
    spec_digest: str,
    quota_version: int,
    nodes: list[dict[str, Any]],
    leases: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
    holds: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    reclaims: list[dict[str, Any]],
    cutoff_seq: int | None = None,
) -> dict[str, Any]:
    node_rows = sorted(nodes, key=lambda row: row["node_id"])
    active_holds = sorted(
        [hold for hold in holds if hold["status"] == "active"],
        key=lambda hold: (hold["node_id"], int(hold["slot_id"]), hold["artifact_key"], int(hold["shard_index"])),
    )
    provisional_holds = sorted(
        [hold for hold in holds if hold["status"] == "provisional"],
        key=lambda hold: (hold["node_id"], int(hold["slot_id"]), hold["lease_id"]),
    )

    holds_by_node: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for hold in active_holds:
        holds_by_node[hold["node_id"]].append(hold)

    capacity: dict[str, dict[str, Any]] = {}
    for node in node_rows:
        node_id = node["node_id"]
        used = len(holds_by_node.get(node_id, []))
        capacity_slots = int(node["capacity_slots"])
        active_leases = sum(
            1
            for lease in leases
            if lease["node_id"] == node_id and lease["state"] in {"granted", "renewed"}
        )
        capacity[node_id] = {
            "host": node["host"],
            "port": int(node["port"]),
            "capacity_slots": capacity_slots,
            "used_slots": used,
            "free_slots": capacity_slots - used,
            "watermark_ratio": used / capacity_slots if capacity_slots else 1.0,
            "active_leases": active_leases,
            "max_leases": int(node["max_leases"]),
            "provisional_unknown_slots": sum(1 for hold in provisional_holds if hold["node_id"] == node_id),
        }

    slot_map: dict[str, str] = {}
    slot_conflicts: list[dict[str, Any]] = []
    artifact_shard_map: dict[str, str] = {}
    shard_conflicts: list[dict[str, Any]] = []
    ownership_map: dict[str, dict[str, Any]] = defaultdict(lambda: {"owners": set(), "shards": []})

    for hold in active_holds:
        slot_key = f"{hold['node_id']}:{int(hold['slot_id'])}"
        slot_owner = f"{hold['artifact_key']}#{hold['shard_index']}:{hold['lease_id']}"
        if slot_key in slot_map and slot_map[slot_key] != slot_owner:
            slot_conflicts.append({"slot": slot_key, "holders": [slot_map[slot_key], slot_owner]})
        slot_map[slot_key] = slot_owner

        shard_key = f"{hold['artifact_key']}#{int(hold['shard_index'])}"
        shard_owner = f"{hold['node_id']}:{hold['lease_id']}:slot{hold['slot_id']}"
        if shard_key in artifact_shard_map and artifact_shard_map[shard_key] != shard_owner:
            shard_conflicts.append({"shard": shard_key, "holders": [artifact_shard_map[shard_key], shard_owner]})
        artifact_shard_map[shard_key] = shard_owner

        owner = ownership_map[hold["artifact_key"]]
        owner["owners"].add(hold["node_id"])
        owner["shards"].append(
            {"shard_index": int(hold["shard_index"]), "node_id": hold["node_id"], "slot_id": int(hold["slot_id"])}
        )

    current_ownership: list[dict[str, Any]] = []
    for artifact_key in sorted(ownership_map):
        value = ownership_map[artifact_key]
        owners = sorted(value["owners"])
        current_ownership.append(
            {
                "artifact_key": artifact_key,
                "owner_node_id": owners[0] if len(owners) == 1 else None,
                "conflicting_owner_node_ids": owners if len(owners) != 1 else [],
                "shards": sorted(value["shards"], key=lambda row: row["shard_index"]),
            }
        )

    reclaim_slot_sets: dict[str, set[str]] = defaultdict(set)
    reclaim_overlaps: list[dict[str, Any]] = []
    business_reclaims = [row for row in reclaims if row.get("reason_kind") != "unknown_request"]
    for reclaim in _sorted_reclaims(business_reclaims):
        reason = reclaim["reason_kind"]
        for hold_id in reclaim.get("hold_ids", []):
            for prior_reason, prior_slots in reclaim_slot_sets.items():
                if hold_id in prior_slots and prior_reason != reason:
                    reclaim_overlaps.append(
                        {"hold_id": hold_id, "reasons": sorted({prior_reason, reason})}
                    )
            reclaim_slot_sets[reason].add(hold_id)

    conclusion: dict[str, Any] = {
        "format_version": FORMAT_VERSION,
        "spec_digest": spec_digest,
        "cutoff_seq": cutoff_seq,
        "quota_version": int(quota_version),
        "capacity": capacity,
        "cluster_watermark": {
            "capacity_slots": sum(int(node["capacity_slots"]) for node in node_rows),
            "used_slots": len(active_holds),
            "free_slots": sum(capacity[node["node_id"]]["free_slots"] for node in node_rows),
            "active_leases": sum(1 for lease in leases if lease["state"] in {"granted", "renewed"}),
        },
        "leases": sorted(leases, key=lambda lease: lease["lease_id"]),
        "transitions": _sorted_transitions(transitions),
        "active_holds": active_holds,
        "provisional_holds": provisional_holds,
        "current_ownership": current_ownership,
        "decisions": _sorted_decisions(decisions),
        "reclaims": _sorted_reclaims(business_reclaims),
        "uniqueness": {
            "slot_conflict_count": len(slot_conflicts),
            "shard_conflict_count": len(shard_conflicts),
            "reclaim_overlap_count": len(reclaim_overlaps),
            "slot_conflicts": slot_conflicts,
            "shard_conflicts": shard_conflicts,
            "reclaim_overlaps": reclaim_overlaps,
            "unique_holding_ok": not slot_conflicts and not shard_conflicts and not reclaim_overlaps,
        },
    }
    conclusion["content_digest"] = sha256_hex(canonical_bytes({k: v for k, v in conclusion.items() if k != "content_digest"}))
    return conclusion


def conclusion_from_engine(bundle: Bundle, engine: Engine, *, cutoff_seq: int | None = None) -> dict[str, Any]:
    nodes = [
        {
            "node_id": node.node_id,
            "host": node.host,
            "port": node.port,
            "capacity_slots": node.capacity_slots,
            "max_leases": node.max_leases,
            "seed": node.seed,
        }
        for node in bundle.nodes
    ]
    cutoff = cutoff_seq
    if cutoff is None and bundle.events:
        cutoff = max(int(event["seq"]) for event in bundle.events)
    return build_conclusion(
        spec_digest=bundle.spec_digest(),
        quota_version=engine.quota_version,
        nodes=nodes,
        leases=list(engine.leases.values()),
        transitions=engine.transitions,
        holds=list(engine.holds.values()),
        decisions=engine.decisions,
        reclaims=engine.reclaims,
        cutoff_seq=cutoff,
    )


def write_conclusion(path: str, conclusion: dict[str, Any]) -> str:
    raw = canonical_bytes(conclusion)
    from pathlib import Path

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(raw)
    return sha256_hex(raw)
