"""Whole current-view construction.

Views are immutable dictionaries replaced via snapshot files; callers never
patch a previously exposed view in place.
"""

from __future__ import annotations

from typing import Any

from .ledger import Ledger


def build_ledger_view(ledger: Ledger) -> dict[str, Any]:
    records = []
    stale = []
    unknown = []
    for row in ledger.list_rows():
        record = ledger.get_record(row["shard_id"]) or {}
        state = row["state"]
        if state in {"unknown", "failed"}:
            unknown.append({"shard_id": row["shard_id"], "state": state, "marker": record.get("marker")})
        if state == "stale" or record.get("stale") is True or record.get("current") is False:
            stale.append({"shard_id": row["shard_id"], "state": state})
            continue
        if record.get("outcome") and state in {"committed", "current"}:
            records.append(record)
    records.sort(key=lambda item: (item.get("topology_index", 10**9), item["shard_id"]))
    meta = ledger.all_meta()
    return {
        "view": "shardmesh-current",
        "snapshot_version": int(meta.get("snapshot_version", "0")),
        "ledger_format_version": int(meta.get("format_version", "1")),
        "cursor_seq": int(meta.get("cursor_seq", "0")),
        "current_count": len(records),
        "stale_count": len(stale),
        "unknown": unknown,
        "stale": stale,
        "records": records,
    }


def build_engine_view(engine: Any) -> dict[str, Any]:
    ledger_view = build_ledger_view(engine.ledger)
    ledger_view.update(
        {
            "budget": engine.budget,
            "scale": engine.scale,
            "capacity": engine.capacity.snapshot(),
            "active_leases": engine.leases.snapshot(),
            "invalidation_plans": engine.invalidation_plans,
            "capacity_revocations": engine.capacity_revocations,
            "topology": engine.topo,
        }
    )
    return ledger_view
