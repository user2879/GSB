from __future__ import annotations

from typing import Any


def authority_sort_key(candidate: dict[str, Any]) -> tuple[int, int, int, str, str]:
    digest = candidate.get("content_digest")
    digest_present = 0 if isinstance(digest, str) and digest else 1
    return (
        -int(candidate["capacity_version"]),
        int(candidate["elapsed_ms"]),
        digest_present,
        digest if digest_present == 0 else "",
        str(candidate["node_id"]),
    )


def choose_authority(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    ordered = sorted(candidates, key=authority_sort_key)
    winner = ordered[0]
    return {
        "artifact_key": winner["key"],
        "winner_node_id": winner["node_id"],
        "winner_request_seq": winner.get("request_seq"),
        "capacity_version": winner["capacity_version"],
        "elapsed_ms": winner["elapsed_ms"],
        "content_digest": winner.get("content_digest"),
        "decision_rule": [
            "capacity_version_desc",
            "elapsed_ms_asc",
            "content_digest_asc_present_first",
            "node_id_asc_when_digest_absent_or_final_tie",
        ],
        "ordered_candidates": [
            {
                "node_id": item["node_id"],
                "request_seq": item.get("request_seq"),
                "capacity_version": item["capacity_version"],
                "elapsed_ms": item["elapsed_ms"],
                "content_digest": item.get("content_digest"),
                "rank_tuple": list(authority_sort_key(item)),
            }
            for item in ordered
        ],
    }


def victim_sort_key(record: dict[str, Any]) -> tuple[int, int, str]:
    return (
        -int(record.get("priority", 0)),
        int(record["expires_ms"]),
        str(record["key"]),
    )


def sorted_victims(residents: dict[str, dict[str, Any]] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    values = list(residents.values()) if isinstance(residents, dict) else list(residents)
    return sorted(values, key=victim_sort_key)


def plan_mandatory_evictions(
    residents: dict[str, dict[str, Any]],
    *,
    lease_limit: int,
    hard_limit: int,
) -> dict[str, Any]:
    occupancy = sum(int(item["slots"]) for item in residents.values())
    active_leases = len(residents)
    ordered = sorted_victims(residents)
    selected: list[dict[str, Any]] = []
    remaining_occupancy = occupancy
    remaining_leases = active_leases

    for record in ordered:
        lease_violation = remaining_leases > lease_limit
        hard_violation = remaining_occupancy >= hard_limit
        if not lease_violation and not hard_violation:
            break
        triggers = []
        if lease_violation:
            triggers.append("lease_limit")
        if hard_violation:
            triggers.append("hard_watermark")
        selected.append({
            "node_id": record["node_id"],
            "key": record["key"],
            "lease_id": record["lease_id"],
            "priority": int(record.get("priority", 0)),
            "expires_ms": int(record["expires_ms"]),
            "slots": int(record["slots"]),
            "content_digest": record.get("content_digest"),
            "victim_rank": list(victim_sort_key(record)),
            "selected_for": triggers,
            "occupancy_before_selection": remaining_occupancy,
            "active_leases_before_selection": remaining_leases,
        })
        remaining_occupancy -= int(record["slots"])
        remaining_leases -= 1

    return {
        "initial_occupancy": occupancy,
        "initial_active_leases": active_leases,
        "lease_limit": int(lease_limit),
        "hard_limit": int(hard_limit),
        "required": len(selected),
        "victims": selected,
        "planned_victim_keys": [item["key"] for item in selected],
        "post_eviction_occupancy": remaining_occupancy,
        "post_eviction_active_leases": remaining_leases,
        "conditions_satisfied": remaining_leases <= lease_limit and remaining_occupancy < hard_limit,
    }
