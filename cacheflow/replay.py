from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .canonical import assert_consistent, canonical_bytes, sha256_hex
from .engine import Engine
from .errors import VerifyError
from .ledger import Ledger
from .report import build_conclusion, conclusion_from_engine
from .spec import Bundle


def _rows(conn: sqlite3.Connection, sql: str, parameters: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    return [dict(row) for row in conn.execute(sql, parameters).fetchall()]


def conclusion_from_readonly_ledger(conn: sqlite3.Connection, bundle: Bundle) -> dict[str, Any]:
    meta = {row["key"]: row["value"] for row in _rows(conn, "SELECT key,value FROM schema_meta")}
    spec_digest = bundle.spec_digest()
    if meta.get("spec_digest") != spec_digest:
        raise VerifyError(
            "original declarations digest does not match ledger spec_digest",
            entry="spec_digest",
            field="$.spec_digest",
            details={"ledger": meta.get("spec_digest"), "declarations": spec_digest},
        )

    quota_version = bundle.quota_version
    for event_row in _rows(conn, "SELECT event_json FROM events WHERE event_type='quota_bump' ORDER BY seq"):
        event = json.loads(event_row["event_json"])
        quota_version = max(quota_version, int(event["quota_version"]))

    lease_rows = _rows(conn, "SELECT lease_id,node_id,artifact_key,quota_version,state,generation,last_event_seq FROM lease_current")
    leases = [
        {
            "lease_id": row["lease_id"],
            "node_id": row["node_id"],
            "artifact_key": row["artifact_key"],
            "quota_version": int(row["quota_version"]),
            "state": row["state"],
            "generation": int(row["generation"]),
            "last_event_seq": int(row["last_event_seq"]),
        }
        for row in lease_rows
    ]
    transitions = [
        {
            "event_seq": int(row["event_seq"]),
            "ordinal": int(row["ordinal"]),
            "lease_id": row["lease_id"],
            "node_id": row["node_id"],
            "trigger": row["trigger"],
            "reason": row["reason"],
            "state_before": row["state_before"],
            "state_after": row["state_after"],
            "generation_before": int(row["generation_before"]),
            "generation_after": int(row["generation_after"]),
            "carried_label": None if row["carried_label"] is None else int(row["carried_label"]),
            "quota_version": int(row["quota_version"]),
        }
        for row in _rows(conn, "SELECT * FROM transitions")
    ]
    holds = [
        {
            "hold_id": row["hold_id"],
            "node_id": row["node_id"],
            "slot_id": int(row["slot_id"]),
            "lease_id": row["lease_id"],
            "artifact_key": row["artifact_key"],
            "shard_index": int(row["shard_index"]),
            "status": row["status"],
            "opened_seq": int(row["opened_seq"]),
            "closed_seq": None if row["closed_seq"] is None else int(row["closed_seq"]),
            "reclaim_reason": row["reclaim_reason"],
        }
        for row in _rows(conn, "SELECT * FROM holds")
    ]
    decisions: list[dict[str, Any]] = []
    for row in _rows(conn, "SELECT * FROM decisions"):
        body = json.loads(row["decision_json"])
        decisions.append(
            {
                "decision_id": row["decision_id"],
                "event_seq": int(row["event_seq"]),
                "ordinal": int(row["ordinal"]),
                "artifact_key": row["artifact_key"],
                "owner_node_id": row["owner_node_id"],
                "basis": row["basis"],
                "generation": None if row["generation"] is None else int(row["generation"]),
                "quota_version": int(row["quota_version"]),
                "decision": body,
            }
        )
    reclaims: list[dict[str, Any]] = []
    for row in _rows(conn, "SELECT * FROM reclaims"):
        reclaims.append(
            {
                "reclaim_id": row["reclaim_id"],
                "event_seq": int(row["event_seq"]),
                "ordinal": int(row["ordinal"]),
                "lease_id": row["lease_id"],
                "node_id": row["node_id"],
                "reason_kind": row["reason_kind"],
                "generation_before": int(row["generation_before"]),
                "generation_after": int(row["generation_after"]),
                "hold_ids": json.loads(row["hold_ids_json"]),
            }
        )

    cutoff_row = conn.execute("SELECT MAX(seq) AS seq FROM events").fetchone()
    cutoff = int(cutoff_row["seq"] or 0)
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
    return build_conclusion(
        spec_digest=spec_digest,
        quota_version=quota_version,
        nodes=nodes,
        leases=leases,
        transitions=transitions,
        holds=holds,
        decisions=decisions,
        reclaims=reclaims,
        cutoff_seq=cutoff,
    )


def verify_ledger(bundle: Bundle, ledger_path: str | Path) -> dict[str, Any]:
    path = Path(ledger_path)
    if not path.exists():
        raise VerifyError("ledger file does not exist", entry=str(path), field="ledger")
    conn = Ledger.read_only_connection(path)
    try:
        replayed = conclusion_from_readonly_ledger(conn, bundle)

        rebuilt_engine = Engine(bundle)
        event_rows = conn.execute("SELECT event_json FROM events ORDER BY seq").fetchall()
        for event_row in event_rows:
            rebuilt_engine.apply_spec_event(json.loads(event_row["event_json"]))
        rebuilt = conclusion_from_engine(bundle, rebuilt_engine, cutoff_seq=replayed["cutoff_seq"])
        assert_consistent(rebuilt, replayed, VerifyError)

        snapshot_row = conn.execute(
            "SELECT * FROM snapshots WHERE snapshot_key='latest'"
        ).fetchone()
        if snapshot_row is None:
            raise VerifyError("latest snapshot is missing", entry="snapshots", field="latest")
        stored = json.loads(snapshot_row["canonical_json"])
        assert_consistent(replayed, stored, VerifyError)
        raw = canonical_bytes(stored)
        if sha256_hex(raw) != snapshot_row["digest"]:
            raise VerifyError(
                "stored snapshot digest mismatch",
                entry="snapshots.latest",
                field="digest",
                details={"column": snapshot_row["digest"], "canonical": sha256_hex(raw)},
            )
        if sha256_hex(canonical_bytes(replayed)) != snapshot_row["digest"]:
            raise VerifyError(
                "replayed conclusion digest does not match ledger snapshot",
                entry="snapshots.latest",
                field="digest",
            )
        return {
            "ok": True,
            "ledger_path": str(path),
            "cutoff_seq": replayed["cutoff_seq"],
            "digest": replayed["content_digest"],
            "unique_holding_ok": replayed["uniqueness"]["unique_holding_ok"],
            "canonical_bytes": canonical_bytes(replayed),
        }
    finally:
        conn.close()
