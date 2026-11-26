from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .canonical import digest_json
from .determinism import expected_summary
from .impact import event_edit_impact
from .ledger import Ledger
from .spec import EventStream, Spec
from .watermark import global_budget_attribution, node_water_report


def recompute_edited_event_stream(
    spec: Spec,
    new_events: EventStream,
    ledger_path: str | Path,
    *,
    old_events: EventStream | None = None,
    impact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    plan = impact or event_edit_impact(spec, old_events, new_events)
    ledger = Ledger.open_session(ledger_path, spec.digest())
    try:
        before_rows = {
            row["shard_id"]: row["content_digest"]
            for row in ledger.conn.execute("SELECT shard_id,content_digest FROM shard_outcomes").fetchall()
        }
        dispatch_node = {
            event.data["shard_id"]: event.data["node_id"]
            for event in new_events.events
            if event.type == "dispatch"
        }
        outcomes = {row["shard_id"]: dict(row) for row in ledger.outcomes()}
        recomputed: list[dict[str, Any]] = []
        cache_hits: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []

        targets = set(plan["aggregate_recompute_shards"])
        for shard_id in sorted(targets):
            shard = spec.shard_map.get(shard_id)
            if shard is None:
                continue
            row = outcomes.get(shard_id)
            node_id = dispatch_node.get(shard_id) or (row or {}).get("node_id")
            if node_id not in spec.node_map:
                skipped.append({"shard_id": shard_id, "reason": "no_node_assignment"})
                continue
            node = spec.node_map[node_id]
            quota_version = int((row or {}).get("quota_version") or node.quota_version)
            summary = expected_summary(
                node,
                shard,
                quota_version,
                batch_size=spec.runtime.batch_size,
                fixed_point_scale=spec.runtime.fixed_point_scale,
            )
            matches_stored = bool(row and row.get("aggregate_digest") == summary["aggregate_digest"])
            recomputed.append({
                "shard_id": shard_id,
                "node_id": node_id,
                "quota_version": quota_version,
                "aggregate_digest": summary["aggregate_digest"],
                "matches_stored_digest": matches_stored,
                "numpy_recomputed": True,
            })

        for shard_id, row in outcomes.items():
            if shard_id in targets:
                continue
            cache_hits.append({
                "shard_id": shard_id,
                "aggregate_digest": row.get("aggregate_digest"),
                "content_digest": row.get("content_digest"),
                "numpy_recomputed": False,
            })

        water = _recompute_dirty_water(spec, ledger, plan)
        progress = ledger.progress() or {}
        marker = digest_json({"plan": plan, "recomputed": recomputed, "water": water})
        with ledger.transaction():
            ledger.set_meta_locked("last_incremental_digest", marker)
            ledger.set_progress_locked(int(progress.get("last_event_seq") or len(new_events.events)), "incremental_recomputed")

        after_rows = {
            row["shard_id"]: row["content_digest"]
            for row in ledger.conn.execute("SELECT shard_id,content_digest FROM shard_outcomes").fetchall()
        }
        unchanged = before_rows == after_rows
        return {
            "ok": True,
            "ledger_replaced": ledger.replaced,
            "impact": plan,
            "recomputed_aggregates": recomputed,
            "aggregate_cache_hits": cache_hits,
            "skipped": skipped,
            "dirty_water": water,
            "outcome_rows_unchanged_by_incremental_pass": unchanged,
            "untouched_outcome_count": len(cache_hits),
            "marker_digest": marker,
        }
    finally:
        ledger.close()


def _recompute_dirty_water(spec: Spec, ledger: Ledger, plan: dict[str, Any]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for scope in plan.get("water_recompute_scopes", []):
        if scope == "global":
            row = ledger.conn.execute(
                "SELECT * FROM watermarks WHERE scope_type='global' ORDER BY id DESC LIMIT 1"
            ).fetchone()
            participants: list[tuple[str, int]] = []
            used = None
            if row is not None:
                report = json.loads(row["report_json"] or "{}")
                used = row["used_units"]
                participants = [(item["object_id"], int(item["raw_units"])) for item in report.get("participants", [])]
            attribution = global_budget_attribution(participants, spec.global_budget, spec.runtime.fixed_point_scale)
            reports.append({
                "scope": "global",
                "last_recorded_used_units": used if used is not None else attribution.used_units,
                "report": attribution.to_dict(),
            })
        elif scope.startswith("node:"):
            node_id = scope.split(":", 1)[1]
            node = spec.node_map.get(node_id)
            if node is None:
                continue
            row = ledger.conn.execute(
                "SELECT * FROM watermarks WHERE scope_type='node' AND scope_id=? ORDER BY id DESC LIMIT 1",
                (node_id,),
            ).fetchone()
            active_shards = int(row["active_shards"]) if row is not None else 0
            active_leases = int(row["active_leases"]) if row is not None else 0
            participant_ids: list[str] = []
            if row is not None:
                report = json.loads(row["report_json"] or "{}")
                participant_ids = report.get("participant_ids", [])
            reports.append({
                "scope": scope,
                "report": node_water_report(
                    node_id=node_id,
                    active_shards=active_shards,
                    active_leases=active_leases,
                    concurrency_quota=node.concurrency_quota,
                    max_leases=node.max_leases_per_node,
                    scale=spec.runtime.fixed_point_scale,
                    participant_ids=participant_ids,
                ),
            })
    return reports
