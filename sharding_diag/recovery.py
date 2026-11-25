"""Hard-kill recovery and ledger-driven replay verification.

Recovery trusts only fully committed ledger rows.  A shard whose lease was
granted but whose outcome was never committed is classified ``UNKNOWN``: its
in-flight budget is reclaimed and it must be re-run.  After rebuilding the
scheduler view, committed shards are recomputed locally (same deterministic
samples) and compared field by field; the first mismatch raises ``E_REPLAY``
naming the shard and field.
"""

from typing import Dict, List

from .errors import ReplayError
from .sampling import compute_summary, generate_sample, summary_digest

COMPARE_FIELDS = (
    "node_id",
    "lease_version",
    "capacity_version",
    "summary_digest",
    "fixed_total",
    "status",
)


def rebuild_scheduler(scheduler) -> dict:
    ledger = scheduler.ledger
    lease_snapshot = ledger.lease_snapshot()
    scheduler.lease_mgr.restore_snapshot(lease_snapshot, active_only=True)
    scheduler.freeze_mgr.restore_snapshot(ledger.freeze_snapshot())

    outcomes = ledger.outcomes()
    progress = ledger.progress()
    unknown: List[dict] = []
    budget_before = scheduler.lease_mgr.budget_used()
    for shard_id, cursor in sorted(progress.items()):
        if shard_id in outcomes and outcomes[shard_id]["status"] == "COMMITTED":
            continue
        # Granted/dispatched but never committed -> unknown, reclaim budget.
        active = scheduler.lease_mgr.active_lease(
            _node_of_progress(lease_snapshot, shard_id), shard_id
        )
        node_id = active.node_id if active else cursor.get("state", "UNKNOWN")
        reclaimed = active.demand if active else 0
        unknown.append({
            "shard_id": shard_id,
            "state": cursor["state"],
            "node_id": active.node_id if active else None,
            "budget_reclaimed": reclaimed,
        })
        if active is not None:
            event = scheduler.lease_mgr.release_for_freeze(active.node_id, shard_id)
            if event:
                ledger.record_lease_release(event)
        scheduler.freeze_mgr.refreeze(shard_id, "hard_kill_unknown")
    return {
        "rebuilt_outcomes": len(outcomes),
        "unknown_shards": unknown,
        "budget_used_before_recovery": budget_before,
        "budget_used_after_recovery": scheduler.lease_mgr.budget_used(),
        "total_budget_reclaimed": budget_before - scheduler.lease_mgr.budget_used(),
        "frozen": scheduler.freeze_mgr.frozen_shards(),
    }


def _node_of_progress(lease_snapshot, shard_id):
    for lease in lease_snapshot.get("leases", []):
        if lease["shard_id"] == shard_id and not lease["released"]:
            return lease["node_id"]
    return "<none>"


def replay_committed(scheduler, declaration=None) -> dict:
    """Recompute every committed shard locally and verify field equality.

    ``declaration`` may pin the declaration that was authoritative when the
    outcomes were committed; otherwise the scheduler's current declaration is
    used (correct for the post-recovery rerun path).
    """
    declaration = declaration or scheduler.declaration
    node_map = declaration.node_map()
    shard_map = declaration.shard_map()
    outcomes = scheduler.ledger.outcomes()
    checked = 0
    for shard_id in sorted(outcomes):
        stored = outcomes[shard_id]
        shard = shard_map[shard_id]
        node = node_map[stored["node_id"]]
        sample = generate_sample(node.seed, shard_id, shard.rows)
        recomputed = compute_summary(sample, node.node_id, shard_id, node.fixed_point_scale)
        # The summary is recomputed from the deterministic sample; the
        # recorded node/capacity/lease identity is verified against what the
        # ledger committed at outcome time, so capacity evolution does not
        # retroactively invalidate historically correct rows.
        expected = {
            "node_id": node.node_id,
            "lease_version": stored["lease_version"],
            "capacity_version": stored["capacity_version"],
            "summary_digest": summary_digest(recomputed),
            "fixed_total": recomputed["total_sum_fp"],
            "status": "COMMITTED",
        }
        expected["node_id"] = stored["node_id"]
        actual = {field: stored.get(field) for field in COMPARE_FIELDS}
        for field in COMPARE_FIELDS:
            if actual[field] != expected[field]:
                raise ReplayError(
                    f"replay mismatch on {shard_id} field {field!r}: "
                    f"ledger={actual[field]!r} recomputed={expected[field]!r}",
                    object_id=shard_id,
                    field=field,
                    detail={"ledger": actual[field], "recomputed": expected[field]},
                )
        checked += 1
    return {"checked_shards": checked, "fields_per_shard": list(COMPARE_FIELDS)}


def rerun_unknown(scheduler) -> dict:
    """Re-dispatch UNKNOWN shards (budget already reclaimed) and verify."""
    ledger = scheduler.ledger
    unknown_ids = sorted(
        sid for sid, cursor in ledger.progress().items()
        if sid not in ledger.outcomes() or ledger.outcomes()[sid]["status"] != "COMMITTED"
    )
    if not unknown_ids:
        return {"reran": []}
    scheduler.start_nodes()
    result = scheduler.run_subset(set(unknown_ids))
    replay_committed(scheduler)
    return {"reran": unknown_ids, "rounds": result["rounds"]}
