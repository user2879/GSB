from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .arbitration import ClusterCapacity, arbitrate_candidates
from .content import build_shard, chain_digest
from .errors import VerificationError
from .specs import load_bundle


def _connect_read_only(db_path: Path) -> sqlite3.Connection:
    if not db_path.is_file():
        raise VerificationError(f"ledger does not exist: {db_path}", entry_id="*", field="ledger")
    uri = db_path.resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def _mismatch(
    mismatches: list[dict[str, Any]],
    *,
    artifact_key: str | None,
    node_id: str | None,
    field_name: str,
    expected: Any,
    actual: Any,
    message: str,
) -> None:
    mismatches.append(
        {
            "artifact_key": artifact_key or "*",
            "node_id": node_id or "*",
            "field": field_name,
            "expected": expected,
            "actual": actual,
            "message": message,
        }
    )


def _raise_first(mismatches: list[dict[str, Any]]) -> None:
    if not mismatches:
        return
    first = sorted(
        mismatches,
        key=lambda item: (str(item["artifact_key"]), str(item["node_id"]), str(item["field"]), item["message"]),
    )[0]
    raise VerificationError(
        "ledger verification failed at "
        f"artifact={first['artifact_key']} node={first['node_id']} field={first['field']}: {first['message']}",
        entry_id=str(first["artifact_key"]),
        field=str(first["field"]),
        details={"node_id": first["node_id"], **first},
    )


def verify_ledger(sample_dir: str | Path, db_path: str | Path) -> dict[str, Any]:
    bundle = load_bundle(sample_dir)
    path = Path(db_path)
    conn = _connect_read_only(path)
    mismatches: list[dict[str, Any]] = []
    try:
        meta = {row["key"]: row["value"] for row in conn.execute("SELECT key,value FROM meta")}
        if int(meta.get("ledger_format_version", "0")) != 1:
            _mismatch(
                mismatches,
                artifact_key="*",
                node_id="*",
                field_name="ledger_format_version",
                expected=1,
                actual=meta.get("ledger_format_version"),
                message="unsupported ledger format version",
            )
        if meta.get("spec_digest") != bundle.spec_digest:
            _mismatch(
                mismatches,
                artifact_key="*",
                node_id="*",
                field_name="spec_digest",
                expected=bundle.spec_digest,
                actual=meta.get("spec_digest"),
                message="ledger declarations do not match sample files",
            )

        genesis_payload = {
            "ledger_format_version": 1,
            "spec_digest": bundle.spec_digest,
            "source_digests": bundle.raw_digests,
        }
        genesis_digest = meta.get("genesis_digest") or chain_digest("GENESIS", genesis_payload)
        chain_errors = _verify_hash_chain(conn, meta.get("head_digest", ""), genesis_digest)
        for entry_id in chain_errors:
            _mismatch(mismatches, artifact_key="*", node_id="*", field_name="ledger_digest", expected="valid hash chain", actual=entry_id, message="ledger hash chain mismatch")

        unknown = [dict(row) for row in conn.execute("SELECT batch_id,event_seq FROM unknown_batches WHERE status='open' ORDER BY event_seq")]
        if unknown:
            raise VerificationError(
                "ledger contains an unknown uncommitted batch; rerun that event before verification",
                code="E_UNKNOWN",
                entry_id=f"seq={unknown[0]['event_seq']}",
                field="unknown_batch",
                details={"unknown_batches": unknown},
            )

        records = [dict(row) for row in conn.execute("SELECT * FROM records ORDER BY id")]
        for record in records:
            record["payload"] = json.loads(record["payload_json"])

        capacity = ClusterCapacity.from_specs(bundle.nodes)
        lease_status: dict[str, str] = {}
        lease_rows_expected: dict[str, dict[str, Any]] = {}
        active_nodes = list(bundle.node_ids)
        watermark_scale = bundle.pipeline.watermark_scale

        for record in records:
            rtype = record["record_type"]
            node_id = record["node_id"]
            artifact_key = record["artifact_key"]
            payload = record["payload"]

            if rtype == "quota_replacement":
                active_nodes = list(payload.get("active_nodes") or active_nodes)

            if rtype == "arbitration":
                _verify_arbitration(
                    record=record,
                    payload=payload,
                    bundle=bundle,
                    capacity=capacity,
                    active_nodes=active_nodes,
                    mismatches=mismatches,
                )

            if rtype == "reserve":
                reservation_id = record["reservation_id"] or payload.get("reservation_id")
                try:
                    capacity.reserve(reservation_id, node_id)  # type: ignore[arg-type]
                except Exception as exc:
                    _mismatch(mismatches, artifact_key=artifact_key, node_id=node_id, field_name="reserve", expected="valid reservation", actual=str(exc), message="reservation replay failed")

            elif rtype == "rollback":
                reservation_id = record["reservation_id"] or payload.get("reservation_id")
                try:
                    capacity.rollback_reservation(reservation_id)
                except Exception as exc:
                    _mismatch(mismatches, artifact_key=artifact_key, node_id=node_id, field_name="rollback", expected="exact rollback", actual=str(exc), message="rollback replay failed")

            elif rtype == "lease_grant":
                lease_id = record["lease_id"]
                reservation_id = record["reservation_id"] or payload.get("reservation_id")
                try:
                    owner = capacity.grant(reservation_id, lease_id)
                except Exception as exc:
                    _mismatch(mismatches, artifact_key=artifact_key, node_id=node_id, field_name="lease_grant", expected="valid grant", actual=str(exc), message="grant replay failed")
                    owner = node_id
                lease_status[lease_id] = "active"
                lease_rows_expected[lease_id] = payload
                if owner != node_id:
                    _mismatch(mismatches, artifact_key=artifact_key, node_id=node_id, field_name="node_id", expected=node_id, actual=owner, message="lease granted to unexpected node")

                shard_index = int(payload.get("shard_index", 0))
                expected_shard = build_shard(
                    node_id=node_id,
                    node_seed=bundle.node(node_id).seed,
                    artifact_key=artifact_key,
                    shard_index=shard_index,
                    byte_count=bundle.shard_bytes,
                )
                if payload.get("content_digest") != expected_shard["digest"]:
                    _mismatch(mismatches, artifact_key=artifact_key, node_id=node_id, field_name="content_digest", expected=expected_shard["digest"], actual=payload.get("content_digest"), message="lease content digest mismatch")
                if int(payload.get("shard_bytes", -1)) != expected_shard["shard_bytes"]:
                    _mismatch(mismatches, artifact_key=artifact_key, node_id=node_id, field_name="shard_bytes", expected=expected_shard["shard_bytes"], actual=payload.get("shard_bytes"), message="shard byte count mismatch")
                if int(payload.get("source_version", -1)) != expected_shard["source_version"]:
                    _mismatch(mismatches, artifact_key=artifact_key, node_id=node_id, field_name="source_version", expected=expected_shard["source_version"], actual=payload.get("source_version"), message="content source version mismatch")

            elif rtype == "renew":
                lease_id = record["lease_id"]
                if lease_id in lease_rows_expected:
                    lease_rows_expected[lease_id]["lease_version"] = payload.get("new_version")
                    lease_status[lease_id] = "active"

            elif rtype in {"release", "reclaim"}:
                lease_id = record["lease_id"]
                reason = payload.get("reason") or ("user_release" if rtype == "release" else None)
                if lease_status.get(lease_id) != "active":
                    _mismatch(
                        mismatches,
                        artifact_key=artifact_key,
                        node_id=node_id,
                        field_name="single_reclaim",
                        expected="lease active before one close",
                        actual=lease_status.get(lease_id, "missing"),
                        message=f"lease was closed more than once or before {reason}",
                    )
                else:
                    released_node = capacity.release_lease(lease_id)
                    lease_status[lease_id] = "closed"
                    if released_node != node_id:
                        _mismatch(mismatches, artifact_key=artifact_key, node_id=node_id, field_name="lease_node", expected=node_id, actual=released_node, message="reclaim changed node ownership")

            elif rtype == "read_receipt":
                shard_index = int(payload.get("shard_index", 0))
                expected_shard = build_shard(
                    node_id=node_id,
                    node_seed=bundle.node(node_id).seed,
                    artifact_key=artifact_key,
                    shard_index=shard_index,
                    byte_count=bundle.shard_bytes,
                )
                if payload.get("digest") != expected_shard["digest"]:
                    _mismatch(mismatches, artifact_key=artifact_key, node_id=node_id, field_name="content_digest", expected=expected_shard["digest"], actual=payload.get("digest"), message="read digest mismatch")

            elif rtype == "watermark":
                node_capacity = capacity.nodes[node_id]
                expected_wm = node_capacity.watermark(watermark_scale)
                for field_name in ("used_slots", "remaining_slots", "active_leases", "lease_remaining", "watermark_fixed"):
                    if payload.get(field_name) != expected_wm[field_name]:
                        _mismatch(mismatches, artifact_key="*", node_id=node_id, field_name=field_name, expected=expected_wm[field_name], actual=payload.get(field_name), message="watermark/capacity field mismatch during replay")

        _compare_materialized_nodes(conn, capacity, mismatches)
        _compare_leases(conn, lease_status, mismatches)
        _raise_first(mismatches)

        return {
            "ok": True,
            "code": "E_OK",
            "ledger": str(path),
            "ledger_format_version": int(meta.get("ledger_format_version", "0")),
            "ledger_digest": meta.get("head_digest"),
            "records_checked": len(records),
            "active_leases": len(conn.execute("SELECT 1 FROM leases WHERE status='active'").fetchall()),
            "unknown_batches": 0,
        }
    finally:
        conn.close()


def _verify_hash_chain(conn: sqlite3.Connection, head: str, genesis_digest: str) -> list[str]:
    rows = conn.execute("SELECT entry_id,prev_digest,digest,payload_json FROM ledger_entries ORDER BY entry_seq").fetchall()
    previous = genesis_digest
    errors: list[str] = []
    for row in rows:
        payload = json.loads(row["payload_json"])
        expected = chain_digest(previous, payload)
        if row["prev_digest"] != previous or row["digest"] != expected:
            errors.append(row["entry_id"])
        previous = row["digest"]
    if head != previous:
        errors.append("meta:head_digest")
    return errors


def _verify_arbitration(
    *,
    record: dict[str, Any],
    payload: dict[str, Any],
    bundle: Any,
    capacity: ClusterCapacity,
    active_nodes: list[str],
    mismatches: list[dict[str, Any]],
) -> None:
    event_seq = int(record["event_seq"])
    event = next((item for item in bundle.events if item.seq == event_seq), None)
    artifact_key = record["artifact_key"]
    node_id = record["node_id"]
    if event is None or event.type != "write":
        _mismatch(mismatches, artifact_key=artifact_key, node_id=node_id, field_name="event_seq", expected="write event", actual=event_seq, message="arbitration has no matching write event")
        return

    candidate_ids = [str(value) for value in event.get("candidate_nodes") or active_nodes]
    candidates: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        node_capacity = capacity.nodes[candidate_id]
        candidates.append(
            {
                "node_id": candidate_id,
                "digest_match": True,
                "remaining_slots": node_capacity.remaining_slots,
                "eligible": node_capacity.remaining_slots > 0 and node_capacity.lease_remaining > 0,
            }
        )
    expected_decision = arbitrate_candidates(candidates)
    actual_winner = payload.get("winner_node_id")
    if actual_winner != expected_decision["winner_node_id"]:
        _mismatch(
            mismatches,
            artifact_key=artifact_key,
            node_id=expected_decision["winner_node_id"],
            field_name="arbitration_winner",
            expected=expected_decision["winner_node_id"],
            actual=actual_winner,
            message="stable arbitration winner mismatch",
        )
    if payload.get("basis") != expected_decision["basis"]:
        _mismatch(
            mismatches,
            artifact_key=artifact_key,
            node_id=actual_winner,
            field_name="basis",
            expected=expected_decision["basis"],
            actual=payload.get("basis"),
            message="arbitration basis mismatch",
        )
    if payload.get("indistinguishable_node_ids") != expected_decision["indistinguishable_node_ids"]:
        _mismatch(
            mismatches,
            artifact_key=artifact_key,
            node_id=actual_winner,
            field_name="indistinguishable_node_ids",
            expected=expected_decision["indistinguishable_node_ids"],
            actual=payload.get("indistinguishable_node_ids"),
            message="indistinguishable tie group mismatch",
        )


def _compare_materialized_nodes(conn: sqlite3.Connection, capacity: ClusterCapacity, mismatches: list[dict[str, Any]]) -> None:
    rows = {row["node_id"]: dict(row) for row in conn.execute("SELECT * FROM node_state ORDER BY node_id")}
    for node_id in sorted(capacity.nodes):
        expected = capacity.nodes[node_id]
        actual = rows.get(node_id)
        if actual is None:
            _mismatch(mismatches, artifact_key="*", node_id=node_id, field_name="node_state", expected="row", actual=None, message="missing node state")
            continue
        comparisons = {
            "used_slots": expected.used_slots,
            "remaining_slots": expected.remaining_slots if "remaining_slots" in actual else actual["capacity_slots"] - actual["used_slots"],
            "active_leases": expected.active_leases,
            "lease_remaining": expected.lease_remaining,
        }
        for field_name, expected_value in comparisons.items():
            actual_value = actual.get(field_name)
            if field_name == "remaining_slots":
                actual_value = actual["capacity_slots"] - actual["used_slots"]
            elif field_name == "lease_remaining":
                actual_value = actual["lease_limit"] - actual["active_leases"]
            if actual_value != expected_value:
                _mismatch(mismatches, artifact_key="*", node_id=node_id, field_name=field_name, expected=expected_value, actual=actual_value, message="materialized capacity differs from replay")


def _compare_leases(conn: sqlite3.Connection, lease_status: dict[str, str], mismatches: list[dict[str, Any]]) -> None:
    rows = {row["lease_id"]: dict(row) for row in conn.execute("SELECT * FROM leases")}
    expected_active_by_node: dict[str, set[str]] = {}
    actual_active_by_node: dict[str, set[str]] = {}
    for lease_id, status in lease_status.items():
        if status == "active":
            row = rows.get(lease_id)
            node_id = row["node_id"] if row else "*"
            expected_active_by_node.setdefault(node_id, set()).add(lease_id)
    for lease_id, row in rows.items():
        if row["status"] == "active":
            actual_active_by_node.setdefault(row["node_id"], set()).add(lease_id)
        if lease_id not in lease_status:
            _mismatch(mismatches, artifact_key=row["artifact_key"], node_id=row["node_id"], field_name="lease_id", expected="ledger grant record", actual=lease_id, message="lease row absent from record replay")
    for node_id in sorted(set(expected_active_by_node) | set(actual_active_by_node)):
        expected_set = sorted(expected_active_by_node.get(node_id, set()))
        actual_set = sorted(actual_active_by_node.get(node_id, set()))
        if expected_set != actual_set:
            artifact_key = "*"
            if expected_set:
                row = rows.get(expected_set[0])
                artifact_key = row["artifact_key"] if row else "*"
            elif actual_set:
                artifact_key = rows[actual_set[0]]["artifact_key"]
            _mismatch(mismatches, artifact_key=artifact_key, node_id=node_id, field_name="active_lease_set", expected=expected_set, actual=actual_set, message="active lease set differs")
