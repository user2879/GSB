from __future__ import annotations

from typing import Any

from .errors import E_GLOBAL_BUDGET_PENDING
from .util import stable_hash


def requested_iterations(method: str, settings: dict[str, Any]) -> int:
    if method == "dbscan":
        return 1
    return max(1, int(settings.get("max_iter", 1)))


def plan_budget(
    candidates: list[dict[str, Any]],
    *,
    global_budget: int,
    per_candidate_quota: int,
) -> dict[str, Any]:
    """Create a deterministic reservation ledger before parallel execution.

    A full request is reserved synchronously in candidate-key order.  Candidates
    whose complete reservation does not fit are put in the pending queue without
    consuming budget.  A candidate whose request exceeds the per-candidate quota
    is eligible to run at the quota; failure is only declared if it does not
    converge within that cap.
    """

    remaining = max(0, int(global_budget))
    entries: list[dict[str, Any]] = []
    sequence = 0
    for candidate in sorted(candidates, key=lambda item: item["stable_key"]):
        sequence += 1
        method = str(candidate.get("method", ""))
        settings = candidate.get("runtime_settings", {})
        request = requested_iterations(method, settings)
        entry: dict[str, Any] = {
            "sequence": sequence,
            "candidate": candidate["stable_key"],
            "method": method,
            "params": candidate.get("params", {}),
            "requested": request,
            "quota": int(per_candidate_quota),
            "reserved": 0,
            "remaining_before_reservation": remaining,
            "remaining_after_reservation": remaining,
            "consumed": 0,
            "outcome": "invalid",
        }
        if not candidate.get("valid", False):
            entries.append(entry)
            continue
        allocation = min(request, max(0, int(per_candidate_quota)))
        entry["allocation_limit"] = allocation
        if allocation <= 0 or remaining < allocation:
            entry["outcome"] = "global_budget_pending"
            entry["pending_reason"] = E_GLOBAL_BUDGET_PENDING
            entries.append(entry)
            continue
        remaining -= allocation
        entry["reserved"] = allocation
        entry["remaining_after_reservation"] = remaining
        entry["outcome"] = "allocated"
        entries.append(entry)

    by_candidate = {entry["candidate"]: entry for entry in entries}
    return {
        "global_budget": max(0, int(global_budget)),
        "per_candidate_quota": int(per_candidate_quota),
        "reserved_total": sum(entry["reserved"] for entry in entries),
        "remaining_after_reservations": remaining,
        "entries": entries,
        "by_candidate": by_candidate,
        "reservation_order": [entry["candidate"] for entry in entries],
        "policy": "stable_candidate_key_ascending_full_request_reservation",
    }


def finalize_budget(plan: dict[str, Any], consumed_by_candidate: dict[str, int], outcomes: dict[str, str]) -> dict[str, Any]:
    budget = int(plan["global_budget"])
    consumed_total = 0
    entries: list[dict[str, Any]] = []
    for original in plan["entries"]:
        entry = dict(original)
        key = entry["candidate"]
        consumed = max(0, int(consumed_by_candidate.get(key, 0)))
        if entry["outcome"] == "allocated" and consumed > entry["reserved"]:
            consumed = entry["reserved"]
        entry["consumed"] = consumed
        entry["outcome"] = outcomes.get(key, entry["outcome"])
        consumed_total += consumed
        entries.append(entry)
    final = dict(plan)
    final["by_candidate"] = {entry["candidate"]: entry for entry in entries}
    final["entries"] = entries
    final["consumed_total"] = consumed_total
    final["remaining"] = max(0, budget - consumed_total)
    final["canonical_order"] = sorted(entries, key=lambda item: item["candidate"])
    final["content_fingerprint"] = stable_hash(entries)
    return final
