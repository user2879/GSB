from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import LEDGER_FORMAT_VERSION
from .canonical import canonical_bytes, sha256_hex
from .determinism import expected_summary, generate_samples, samples_raw_bytes
from .errors import VerifyError
from .impact import event_hash, invalidation_plan
from .ledger import Ledger
from .spec import EventStream, Spec, load_events, load_spec, stable_topological_order

TERMINAL_KINDS = {
    "release_complete", "cancel", "rollback_revoke",
    "expire_reclaim", "fail", "dependency_block",
}


def _loads(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def verify_files(spec_path: str | Path, events_path: str | Path, ledger_path: str | Path) -> dict[str, Any]:
    spec = load_spec(spec_path)
    events = load_events(events_path, spec)
    return verify_ledger(spec, events, ledger_path)


def verify_ledger(spec: Spec, events: EventStream, ledger_path: str | Path) -> dict[str, Any]:
    path = Path(ledger_path)
    if not path.exists():
        raise VerifyError(f"ledger does not exist: {path}", field="ledger_path")
    ledger = Ledger.open_read_only(path)
    try:
        meta = ledger.meta()
        if meta.get("format_version") != str(LEDGER_FORMAT_VERSION):
            raise VerifyError(
                "ledger format_version mismatch",
                field="meta.format_version",
                object_id=str(path),
                details={"expected": LEDGER_FORMAT_VERSION, "actual": meta.get("format_version")},
            )
        if meta.get("spec_digest") != spec.digest():
            raise VerifyError(
                "ledger spec_digest does not match current declaration; whole snapshot replacement required",
                field="meta.spec_digest",
                object_id=str(path),
                details={"expected": spec.digest(), "actual": meta.get("spec_digest")},
            )

        outcomes = {row["shard_id"]: row for row in ledger.conn.execute("SELECT * FROM shard_outcomes").fetchall()}
        dispatch_node = _dispatch_nodes(events)
        topo = stable_topological_order(spec.shards)
        ordered_ids = [sid for sid in topo if sid in outcomes] + sorted(sid for sid in outcomes if sid not in topo)

        for shard_id in ordered_ids:
            row = outcomes[shard_id]
            _verify_outcome(spec, shard_id, dict(row), dispatch_node)

        lease_report = _replay_lease_capacity(spec, ledger.lease_events())
        event_report = _verify_event_journal(events, ledger.journal())
        rollback_report = _verify_rollbacks(spec, events, ledger.journal(), ledger.lease_events())
        progress = ledger.progress()
        if progress is None:
            raise VerifyError("ledger is missing progress row", field="progress")
        if int(progress["last_event_seq"]) != len(events.events):
            raise VerifyError(
                "progress cursor does not match event stream length",
                field="progress.last_event_seq",
                object_id="progress",
                details={"expected": len(events.events), "actual": progress["last_event_seq"]},
            )
        expected_root = ledger.compute_root_digest()
        if meta.get("content_digest") != expected_root:
            raise VerifyError(
                "ledger content digest mismatch after independent recomputation",
                field="meta.content_digest",
                object_id=str(path),
                details={"expected": expected_root, "actual": meta.get("content_digest")},
            )
        return {
            "ok": True,
            "verified_shards": ordered_ids,
            "lease_replay": lease_report,
            "events": event_report,
            "rollbacks": rollback_report,
            "progress": dict(progress),
            "content_digest": expected_root,
        }
    finally:
        ledger.close()


def _dispatch_nodes(events: EventStream) -> dict[str, str]:
    result: dict[str, str] = {}
    for event in events.events:
        if event.type == "dispatch":
            result[event.data["shard_id"]] = event.data["node_id"]
    return result


def _verify_outcome(spec: Spec, shard_id: str, row: dict[str, Any], dispatch_node: dict[str, str]) -> None:
    shard = spec.shard_map.get(shard_id)
    if shard is None:
        raise VerifyError("ledger contains outcome for undeclared shard", field="shard_id", object_id=shard_id)
    expected_node = dispatch_node.get(shard_id)
    if expected_node and row.get("node_id") != expected_node:
        raise VerifyError(
            "node attribution mismatch",
            field="node_id",
            object_id=shard_id,
            details={"expected": expected_node, "actual": row.get("node_id")},
        )
    if row.get("resource_units") != shard.resource_units:
        raise VerifyError(
            "resource_units mismatch",
            field="resource_units",
            object_id=shard_id,
            details={"expected": shard.resource_units, "actual": row.get("resource_units")},
        )
    if row.get("sample_rows") != shard.sample_rows:
        raise VerifyError(
            "sample_rows mismatch",
            field="sample_rows",
            object_id=shard_id,
            details={"expected": shard.sample_rows, "actual": row.get("sample_rows")},
        )
    node_id = row.get("node_id") or expected_node
    if node_id not in spec.node_map:
        if row.get("status") == "completed":
            raise VerifyError("completed outcome has unknown node", field="node_id", object_id=shard_id)
        return
    node = spec.node_map[node_id]
    samples = generate_samples(node.seed, shard_id, shard.sample_rows)
    expected_sample_digest = sha256_hex(samples_raw_bytes(samples))
    if row.get("sample_digest") and row["sample_digest"] != expected_sample_digest:
        raise VerifyError(
            "deterministic sample bytes mismatch",
            field="sample_digest",
            object_id=shard_id,
            details={"expected": expected_sample_digest, "actual": row["sample_digest"]},
        )
    if row.get("status") != "completed":
        return
    quota_version = int(row.get("quota_version") or node.quota_version)
    expected = expected_summary(
        node,
        shard,
        quota_version,
        batch_size=spec.runtime.batch_size,
        fixed_point_scale=spec.runtime.fixed_point_scale,
    )
    if row.get("participated_rows") != shard.sample_rows or expected["participated_rows"] != shard.sample_rows:
        raise VerifyError(
            "participated_rows mismatch",
            field="participated_rows",
            object_id=shard_id,
            details={"expected": shard.sample_rows, "actual": row.get("participated_rows")},
        )
    aggregate_json = row.get("aggregate_json")
    try:
        stored_summary = json.loads(aggregate_json) if isinstance(aggregate_json, str) else aggregate_json
    except (TypeError, json.JSONDecodeError) as exc:
        raise VerifyError("aggregate_json is not valid JSON", field="aggregate_json", object_id=shard_id) from exc
    if not isinstance(stored_summary, dict):
        raise VerifyError("aggregate_json must be an object", field="aggregate_json", object_id=shard_id)
    for field_name in (
        "caliber_version", "fixed_point_scale", "node_id", "shard_id", "sample_rows",
        "participated_rows", "quota_version", "aggregate_digest",
    ):
        if stored_summary.get(field_name) != expected.get(field_name):
            raise VerifyError(
                f"aggregate field {field_name} mismatch",
                field=f"aggregate_json.{field_name}",
                object_id=shard_id,
                details={"expected": expected.get(field_name), "actual": stored_summary.get(field_name)},
            )
    if canonical_bytes(stored_summary.get("groups")) != canonical_bytes(expected["groups"]):
        raise VerifyError(
            "aggregate groups mismatch",
            field="aggregate_json.groups",
            object_id=shard_id,
            details={"expected_digest": expected["aggregate_digest"], "actual_digest": stored_summary.get("aggregate_digest")},
        )
    if row.get("aggregate_digest") != expected["aggregate_digest"]:
        raise VerifyError(
            "aggregate_digest mismatch",
            field="aggregate_digest",
            object_id=shard_id,
            details={"expected": expected["aggregate_digest"], "actual": row.get("aggregate_digest")},
        )
    if not row.get("lease_id") or not row.get("lease_version"):
        raise VerifyError("completed outcome lacks lease attribution", field="lease_id", object_id=shard_id)


def _replay_lease_capacity(spec: Spec, lease_events: list[dict[str, Any]]) -> dict[str, Any]:
    active: dict[str, dict[str, Any]] = {}
    active_shard_lease: dict[str, str] = {}
    terminated: set[str] = set()
    max_used = 0
    for row in lease_events:
        kind = row["kind"]
        lease_id = row["lease_id"]
        shard_id = row["shard_id"]
        node_id = row["node_id"]
        if kind == "grant":
            if lease_id in active:
                raise VerifyError("lease was granted twice", field="lease_id", object_id=lease_id)
            if lease_id in terminated:
                raise VerifyError("terminal lease generation was granted again", field="lease_id", object_id=lease_id)
            if shard_id in active_shard_lease:
                raise VerifyError("shard holds two active leases", field="lease_id", object_id=shard_id)
            active[lease_id] = {"shard_id": shard_id, "node_id": node_id, "version": row["to_version"]}
            active_shard_lease[shard_id] = lease_id
            shard = spec.shard_map.get(shard_id)
            node = spec.node_map.get(node_id)
            if shard is None or node is None:
                raise VerifyError("grant references unknown shard/node", field="lease_id", object_id=lease_id)
            used = sum(spec.shard_map[item["shard_id"]].resource_units for item in active.values())
            max_used = max(max_used, used)
            node_leases = sum(1 for item in active.values() if item["node_id"] == node_id)
            if used > spec.global_budget:
                raise VerifyError("global budget exceeded during replay", field="resource_units", object_id=shard_id, details={"used": used})
            if node_leases > node.max_leases_per_node:
                raise VerifyError("node lease cap exceeded during replay", field="max_leases_per_node", object_id=node_id)
            if node_leases > node.concurrency_quota:
                raise VerifyError("node concurrency quota exceeded during replay", field="concurrency_quota", object_id=node_id)
        elif kind == "renew":
            lease = active.get(lease_id)
            if lease is None:
                raise VerifyError("renew for non-active lease", field="lease_id", object_id=lease_id)
            if row["from_version"] != lease["version"] or row["to_version"] != lease["version"] + 1:
                raise VerifyError(
                    "renewal version transition mismatch",
                    field="lease_version",
                    object_id=lease_id,
                    details={"stored": lease["version"], "from": row["from_version"], "to": row["to_version"]},
                )
            lease["version"] = row["to_version"]
        elif kind in TERMINAL_KINDS:
            if lease_id not in active:
                raise VerifyError(
                    "lease has duplicate or terminal adjudication",
                    field="lease_events.kind",
                    object_id=lease_id or f"event-row-{row.get('id')}",
                    details={"kind": kind},
                )
            lease = active.pop(lease_id)
            terminated.add(lease_id)
            active_shard_lease.pop(lease["shard_id"], None)
        else:
            raise VerifyError(f"unknown lease event kind {kind}", field="lease_events.kind", object_id=str(row["id"]))
    if active:
        raise VerifyError(
            "ledger replay ends with active leases",
            field="lease_events",
            object_id=next(iter(active)),
            details={"active": sorted(active)},
        )
    return {"lease_event_count": len(lease_events), "peak_resource_units": max_used, "unique_active_ownership": True}


def _verify_event_journal(events: EventStream, journal: list[dict[str, Any]]) -> dict[str, Any]:
    by_seq = {row["event_seq"]: row for row in journal}
    for event in events.events:
        row = by_seq.get(event.seq)
        if row is None:
            raise VerifyError("event is missing from journal", field="event_journal.event_seq", object_id=f"event:{event.seq}")
        expected_hash = event_hash(event)
        if row["event_hash"] != expected_hash:
            raise VerifyError("event hash mismatch", field="event_hash", object_id=f"event:{event.seq}")
        stored = json.loads(row["event_json"])
        if stored.get("type") != event.type or stored.get("seq") != event.seq:
            raise VerifyError("journal event payload mismatch", field="event_json", object_id=f"event:{event.seq}")
    return {"journaled_events": len(journal), "expected_events": len(events.events)}


def _verify_rollbacks(spec: Spec, events: EventStream, journal: list[dict[str, Any]], lease_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    journal_by_seq = {row["event_seq"]: row for row in journal}
    grants = {row["lease_id"]: row for row in lease_events if row["kind"] == "grant"}
    for event in events.events:
        if event.type != "rollback":
            continue
        row = journal_by_seq.get(event.seq)
        if row is None:
            raise VerifyError("rollback event missing decision journal", field="event_journal", object_id=f"event:{event.seq}")
        decision = json.loads(row["decision_json"])
        impact = decision.get("impact") or {}
        steps = impact.get("steps") or []
        direct = {step["shard_id"] for step in steps if step.get("kind") == "direct"}
        expected = invalidation_plan(
            spec,
            direct,
            node_id=event.data.get("node_id"),
            target_version=event.data["target_version"],
        )
        if impact.get("invalid_shards") != expected["invalid_shards"]:
            first_actual = (impact.get("invalid_shards") or [None])[0]
            raise VerifyError(
                "rollback invalidation set/order mismatch",
                field="impact.invalid_shards",
                object_id=first_actual or f"event:{event.seq}",
                details={"expected": expected["invalid_shards"], "actual": impact.get("invalid_shards")},
            )
        for lease_event in lease_events:
            if lease_event["kind"] != "rollback_revoke":
                continue
            grant = grants.get(lease_event["lease_id"])
            if grant is None:
                raise VerifyError("rollback revocation lacks grant", field="lease_id", object_id=lease_event["lease_id"])
            grant_detail = _loads(grant.get("detail_json"))
            quota_version = grant_detail.get("quota_version")
            if quota_version is None:
                # Read-only fallback for ledgers written before detail carried quota_version.
                row = ledger.conn.execute(
                    "SELECT quota_version FROM shard_outcomes WHERE shard_id=?",
                    (grant["shard_id"],),
                ).fetchone()
                quota_version = row["quota_version"] if row else None
            if quota_version is None or int(quota_version) <= int(event.data["target_version"]):
                raise VerifyError("rollback revoked a lease that was not on a newer quota version", field="quota_version", object_id=lease_event["lease_id"])
        reports.append({"seq": event.seq, "invalid_shards": expected["invalid_shards"]})
    return reports
