from __future__ import annotations

from typing import Any

from .ledger import Ledger
from .ledger import monotonic_ms


def recover_inflight(ledger: Ledger) -> list[dict[str, Any]]:
    """Mark committed-dispatch/uncommitted-outcome attempts unknown and reclaim them.

    SQLite itself rolls back an outcome transaction interrupted by SIGKILL. The
    committed attempt/occupancy row is therefore the recovery signal.
    """

    rows = ledger.conn.execute(
        """
        SELECT a.shard_id, a.attempt, a.lease_id, a.revision, a.node_id,
               l.quota_version AS quota_version
        FROM attempts a
        LEFT JOIN leases l ON l.lease_id = a.lease_id
        WHERE a.status = 'DISPATCHED'
        ORDER BY a.shard_id, a.attempt
        """
    ).fetchall()
    event_row = ledger.conn.execute(
        "SELECT COALESCE(MAX(event_seq), 0) AS s FROM event_log WHERE stream='runtime'"
    ).fetchone()
    next_event_seq = int(event_row["s"])
    ownership_row = ledger.conn.execute(
        "SELECT COALESCE(MAX(sequence_no), 0) AS s FROM ownership WHERE stream='runtime'"
    ).fetchone()
    next_ownership_seq = int(ownership_row["s"])
    marks: list[dict[str, Any]] = []
    for row in rows:
        next_event_seq += 1
        next_ownership_seq += 1
        quota_version = int(row["quota_version"] or 0)
        event = {
            "seq": next_event_seq,
            "type": "lease_release",
            "lease_id": row["lease_id"],
            "at_ms": monotonic_ms(),
            "quota_version": quota_version,
            "recovery": True,
            "reason": "crash_recovery",
        }
        marks.append(
            {
                "shard_id": row["shard_id"],
                "attempt": int(row["attempt"]),
                "lease_id": row["lease_id"],
                "revision": int(row["revision"]),
                "node_id": row["node_id"],
                "status": "UNKNOWN_RERUN",
                "reason": "dispatch/lease was committed but no committed outcome/progress transaction exists",
                "rule": "reclaim occupancy first, then rerun shard; no crash-time exit report can be reconstructed",
                "event": event,
                "ownership": {
                    "sequence": next_ownership_seq,
                    "event_seq": next_event_seq,
                    "quota_version": quota_version,
                    "at_ms": event["at_ms"],
                    "entitlement_id": f"lease:{row['lease_id']}",
                    "lease_id": row["lease_id"],
                    "shard_id": row["shard_id"],
                    "node_id": row["node_id"],
                    "previous_holder_key": f"{row['node_id']}::{row['shard_id']}",
                    "holder_key": None,
                    "change": "reclaimed",
                    "reason": "crash_recovery",
                    "basis": {"event_type": "lease_release", "recovery": True},
                },
            }
        )
    if marks:
        ledger.record_recovery(marks)
    return marks


def committed_view(ledger: Ledger) -> dict[str, dict[str, Any]]:
    rows = ledger.conn.execute(
        "SELECT shard_id, revision, attempt, summary_digest, outcome_canonical FROM current_view ORDER BY shard_id"
    ).fetchall()
    return {
        row["shard_id"]: {
            "shard_id": row["shard_id"],
            "revision": int(row["revision"]),
            "attempt": int(row["attempt"]),
            "summary_digest": row["summary_digest"],
            "outcome_canonical": row["outcome_canonical"],
        }
        for row in rows
    }
