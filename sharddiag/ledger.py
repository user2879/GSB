"""Atomic SQLite batch ledger, crash recovery, and whole-snapshot replacement."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from .canonical import canonical_bytes, canonical_digest
from .errors import LedgerError


LEDGER_FORMAT_VERSION = 1


def _now_ms() -> int:
    return time.monotonic_ns() // 1_000_000


def verify_snapshot_file(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    try:
        envelope = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LedgerError(f"snapshot is unreadable or invalid JSON: {exc}", path=str(target)) from exc
    if not isinstance(envelope, dict):
        raise LedgerError("snapshot must be a JSON object", path=str(target))
    if not isinstance(envelope.get("format_version"), int) or isinstance(envelope.get("format_version"), bool):
        raise LedgerError("snapshot format_version must be an integer", path=str(target))
    if envelope["format_version"] != LEDGER_FORMAT_VERSION:
        raise LedgerError(
            "snapshot format_version mismatch",
            path=str(target),
            expected=LEDGER_FORMAT_VERSION,
            actual=envelope["format_version"],
        )
    claimed = envelope.get("content_digest")
    content = {key: value for key, value in envelope.items() if key != "content_digest"}
    actual = canonical_digest(content)
    if claimed != actual:
        raise LedgerError(
            "snapshot content_digest mismatch; whole replacement is required",
            path=str(target),
            expected_digest=actual,
            claimed_digest=claimed,
        )
    return envelope


def write_full_snapshot(path: str | Path, content: dict[str, Any]) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    envelope = {
        "format_version": LEDGER_FORMAT_VERSION,
        "saved_monotonic_ms": _now_ms(),
        **content,
    }
    envelope["content_digest"] = canonical_digest(
        {key: value for key, value in envelope.items() if key != "content_digest"}
    )
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, target)
    return envelope


def replace_snapshot_if_invalid(path: str | Path, replacement_content: dict[str, Any]) -> dict[str, Any]:
    """Replace an unreadable/version-mismatched snapshot wholesale."""

    try:
        return verify_snapshot_file(path)
    except LedgerError:
        return write_full_snapshot(path, replacement_content)


class Ledger:
    def __init__(self, path: str | Path, conn: sqlite3.Connection) -> None:
        self.path = Path(path)
        self.conn = conn

    @classmethod
    def open(cls, path: str | Path, *, expected_spec_digest: str | None = None) -> "Ledger":
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(target), timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA foreign_keys=ON")
        ledger = cls(target, conn)
        ledger._create_schema()
        ledger._initialize_meta(expected_spec_digest)
        ledger.verify_chain(expected_spec_digest=expected_spec_digest)
        return ledger

    def _create_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS batches (
                batch_seq INTEGER PRIMARY KEY,
                format_version INTEGER NOT NULL,
                kind TEXT NOT NULL,
                payload_digest TEXT NOT NULL,
                chain_digest TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                monotonic_ms INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS shard_state (
                shard_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                lease_id TEXT,
                node_id TEXT,
                capacity_version INTEGER,
                resource_units INTEGER NOT NULL DEFAULT 0,
                summary_digest TEXT,
                summary_json TEXT,
                unknown_reason TEXT,
                updated_seq INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS lease_state (
                lease_id TEXT PRIMARY KEY,
                shard_id TEXT NOT NULL,
                node_id TEXT NOT NULL,
                status TEXT NOT NULL,
                capacity_version INTEGER NOT NULL,
                lease_version INTEGER NOT NULL,
                resource_units INTEGER NOT NULL,
                lease_json TEXT NOT NULL,
                updated_seq INTEGER NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS one_active_lease_per_shard
                ON lease_state(shard_id) WHERE status = 'active';
            CREATE UNIQUE INDEX IF NOT EXISTS one_active_equity_per_node_shard
                ON lease_state(node_id, shard_id) WHERE status = 'active';
            """
        )
        self.conn.commit()

    def _meta(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return str(row["value"]) if row else default

    def _set_meta(self, key: str, value: Any) -> None:
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(value)),
        )

    def _initialize_meta(self, expected_spec_digest: str | None) -> None:
        existing_version = self._meta("format_version")
        if existing_version is None:
            with self.conn:
                self._set_meta("format_version", LEDGER_FORMAT_VERSION)
                self._set_meta("head_digest", "GENESIS")
                self._set_meta("incarnation", 0)
                if expected_spec_digest:
                    self._set_meta("spec_digest", expected_spec_digest)
        else:
            try:
                version = int(existing_version)
            except ValueError as exc:
                raise LedgerError("ledger format_version is not an integer", actual=existing_version) from exc
            if version != LEDGER_FORMAT_VERSION:
                raise LedgerError(
                    "ledger format_version mismatch; replace the whole ledger/snapshot",
                    expected=LEDGER_FORMAT_VERSION,
                    actual=version,
                )
            if expected_spec_digest:
                actual_spec = self._meta("spec_digest")
                if actual_spec is not None and actual_spec != expected_spec_digest:
                    raise LedgerError(
                        "spec digest mismatch; refusing incremental merge",
                        expected=expected_spec_digest,
                        actual=actual_spec,
                    )

    def verify_chain(self, *, expected_spec_digest: str | None = None) -> None:
        rows = self.conn.execute(
            "SELECT * FROM batches ORDER BY batch_seq ASC"
        ).fetchall()
        previous = "GENESIS"
        expected_seq = 1
        for row in rows:
            if int(row["batch_seq"]) != expected_seq:
                raise LedgerError("batch sequence is not contiguous", expected_seq=expected_seq, actual=row["batch_seq"])
            if int(row["format_version"]) != LEDGER_FORMAT_VERSION:
                raise LedgerError("batch format_version mismatch", batch_seq=expected_seq)
            payload = json.loads(row["payload_json"])
            payload_digest = canonical_digest(payload)
            if payload_digest != row["payload_digest"]:
                raise LedgerError("batch payload digest mismatch", batch_seq=expected_seq)
            chain = hashlib_chain(expected_seq, previous, payload_digest)
            if chain != row["chain_digest"]:
                raise LedgerError("batch chain digest mismatch", batch_seq=expected_seq)
            previous = chain
            expected_seq += 1
        head = self._meta("head_digest")
        if head != previous:
            raise LedgerError("ledger head_digest does not match replayed batch chain", expected=previous, actual=head)
        if expected_spec_digest:
            spec_digest = self._meta("spec_digest")
            if spec_digest is not None and spec_digest != expected_spec_digest:
                raise LedgerError("ledger spec_digest mismatch")

    def batch_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) AS count FROM batches").fetchone()
        return int(row["count"])

    def commit_batch(
        self,
        kind: str,
        payload: dict[str, Any],
        *,
        shard_states: list[dict[str, Any]] | None = None,
        lease_states: list[dict[str, Any]] | None = None,
        meta_updates: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self.conn:
            seq_row = self.conn.execute("SELECT COALESCE(MAX(batch_seq),0)+1 AS seq FROM batches").fetchone()
            seq = int(seq_row["seq"])
            previous = self._meta("head_digest", "GENESIS") or "GENESIS"
            payload_digest = canonical_digest(payload)
            chain_digest = hashlib_chain(seq, previous, payload_digest)
            self.conn.execute(
                """
                INSERT INTO batches(
                    batch_seq, format_version, kind, payload_digest, chain_digest,
                    payload_json, monotonic_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    seq,
                    LEDGER_FORMAT_VERSION,
                    kind,
                    payload_digest,
                    chain_digest,
                    canonical_bytes(payload).decode("utf-8"),
                    _now_ms(),
                ),
            )
            for shard in shard_states or []:
                summary = shard.get("summary")
                if summary is not None:
                    summary_json = canonical_bytes(summary).decode("utf-8")
                    summary_digest = shard.get("summary_digest") or summary.get("summary_digest")
                elif shard.get("summary_json") is not None:
                    summary_json = shard.get("summary_json")
                    summary_digest = shard.get("summary_digest")
                elif shard.get("clear_summary"):
                    summary_json = None
                    summary_digest = None
                else:
                    existing = self.conn.execute(
                        "SELECT summary_json, summary_digest FROM shard_state WHERE shard_id=?",
                        (shard["shard_id"],),
                    ).fetchone()
                    summary_json = existing["summary_json"] if existing else None
                    summary_digest = existing["summary_digest"] if existing else None
                self.conn.execute(
                    """
                    INSERT INTO shard_state(
                        shard_id, status, lease_id, node_id, capacity_version,
                        resource_units, summary_digest, summary_json, unknown_reason, updated_seq
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(shard_id) DO UPDATE SET
                        status=excluded.status,
                        lease_id=excluded.lease_id,
                        node_id=excluded.node_id,
                        capacity_version=excluded.capacity_version,
                        resource_units=excluded.resource_units,
                        summary_digest=excluded.summary_digest,
                        summary_json=excluded.summary_json,
                        unknown_reason=excluded.unknown_reason,
                        updated_seq=excluded.updated_seq
                    """,
                    (
                        shard["shard_id"],
                        shard["status"],
                        shard.get("lease_id"),
                        shard.get("node_id"),
                        shard.get("capacity_version"),
                        int(shard.get("resource_units", 0)),
                        summary_digest,
                        summary_json,
                        shard.get("unknown_reason"),
                        seq,
                    ),
                )
            for lease in lease_states or []:
                self.conn.execute(
                    """
                    INSERT INTO lease_state(
                        lease_id, shard_id, node_id, status, capacity_version,
                        lease_version, resource_units, lease_json, updated_seq
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(lease_id) DO UPDATE SET
                        shard_id=excluded.shard_id,
                        node_id=excluded.node_id,
                        status=excluded.status,
                        capacity_version=excluded.capacity_version,
                        lease_version=excluded.lease_version,
                        resource_units=excluded.resource_units,
                        lease_json=excluded.lease_json,
                        updated_seq=excluded.updated_seq
                    """,
                    (
                        lease["lease_id"],
                        lease["shard_id"],
                        lease["node_id"],
                        lease["status"],
                        int(lease.get("capacity_version", 1)),
                        int(lease.get("lease_version", 1)),
                        int(lease.get("resource_units", 0)),
                        canonical_bytes(lease).decode("utf-8"),
                        seq,
                    ),
                )
            self._set_meta("head_digest", chain_digest)
            for key, value in (meta_updates or {}).items():
                self._set_meta(key, value)
        return {"batch_seq": seq, "payload_digest": payload_digest, "chain_digest": chain_digest}

    def recover_interrupted(self) -> dict[str, Any]:
        incarnation = int(self._meta("incarnation", "0") or "0") + 1
        shard_rows = self.conn.execute("SELECT * FROM shard_state").fetchall()
        lease_rows = self.conn.execute("SELECT * FROM lease_state WHERE status='active'").fetchall()
        active_lease_ids = {row["lease_id"] for row in lease_rows}
        unknown_shards: list[str] = []
        shard_upserts: list[dict[str, Any]] = []
        terminal = {"succeeded", "failed", "cancelled", "frozen"}
        for row in shard_rows:
            if row["status"] not in terminal:
                unknown_shards.append(row["shard_id"])
                shard_upserts.append(
                    {
                        "shard_id": row["shard_id"],
                        "status": "unknown",
                        "lease_id": None,
                        "node_id": row["node_id"],
                        "capacity_version": row["capacity_version"],
                        "resource_units": 0,
                        "unknown_reason": "process_interrupted_before_terminal_batch",
                        "clear_summary": True,
                    }
                )
        lease_upserts: list[dict[str, Any]] = []
        reclaimed: list[str] = []
        for row in lease_rows:
            lease = json.loads(row["lease_json"])
            lease["status"] = "reclaimed_interrupted"
            lease["reclaim_reason"] = "crash_recovery_first_reclaim_occupancy"
            reclaimed.append(row["lease_id"])
            lease_upserts.append(lease)
        result = self.commit_batch(
            "recovery_reconcile",
            {
                "incarnation": incarnation,
                "unknown_shard_ids": sorted(unknown_shards),
                "reclaimed_lease_ids": sorted(reclaimed),
                "rule": "only_committed_batches_are_replayed;_uncommitted_work_is_unknown_and_rerun",
            },
            shard_states=shard_upserts,
            lease_states=lease_upserts,
            meta_updates={"incarnation": incarnation},
        )
        return {
            "incarnation": incarnation,
            "unknown_shard_ids": sorted(unknown_shards),
            "reclaimed_lease_ids": sorted(reclaimed),
            **result,
        }

    def shard_states(self) -> dict[str, dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM shard_state").fetchall()
        result = {}
        for row in rows:
            state = dict(row)
            if row["summary_json"]:
                state["summary"] = json.loads(row["summary_json"])
            result[row["shard_id"]] = state
        return result

    def lease_states(self) -> dict[str, dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM lease_state").fetchall()
        result = {}
        for row in rows:
            lease = json.loads(row["lease_json"])
            result[row["lease_id"]] = lease
        return result

    def active_leases(self) -> list[dict[str, Any]]:
        return [lease for lease in self.lease_states().values() if lease.get("status") == "active"]

    def batches(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM batches ORDER BY batch_seq").fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json"))
            result.append(item)
        return result

    def canonical_accounting_summary(self) -> dict[str, Any]:
        shards = []
        for shard_id in sorted(self.shard_states()):
            row = self.shard_states()[shard_id]
            shards.append(
                {
                    "shard_id": shard_id,
                    "status": row["status"],
                    "node_id": row["node_id"],
                    "capacity_version": row["capacity_version"],
                    "resource_units": row["resource_units"],
                    "summary_digest": row["summary_digest"],
                }
            )
        leases = []
        for lease_id in sorted(self.lease_states()):
            lease = self.lease_states()[lease_id]
            leases.append(
                {
                    "lease_id": lease_id,
                    "shard_id": lease.get("shard_id"),
                    "node_id": lease.get("node_id"),
                    "status": lease.get("status"),
                    "capacity_version": lease.get("capacity_version"),
                    "lease_version": lease.get("lease_version"),
                    "renew_count": lease.get("renew_count"),
                    "resource_units": lease.get("resource_units"),
                }
            )
        content = {
            "format_version": LEDGER_FORMAT_VERSION,
            "spec_digest": self._meta("spec_digest"),
            "shards": shards,
            "leases": leases,
        }
        return {**content, "ledger_summary_digest": canonical_digest(content)}

    def export_snapshot(self, path: str | Path) -> dict[str, Any]:
        state = {"shards": list(self.shard_states().values()), "leases": list(self.lease_states().values())}
        content = {
            "spec_digest": self._meta("spec_digest"),
            "head_digest": self._meta("head_digest"),
            "incarnation": int(self._meta("incarnation", "0") or "0"),
            "batch_count": self.batch_count(),
            "accounting_summary": self.canonical_accounting_summary(),
            "state": state,
            "batches": self.batches(),
        }
        return write_full_snapshot(path, content)

    def close(self) -> None:
        self.conn.close()


def hashlib_chain(seq: int, previous: str, payload_digest: str) -> str:
    import hashlib

    material = f"sharddiag-ledger-chain-v1\n{LEDGER_FORMAT_VERSION}\n{seq}\n{previous}\n{payload_digest}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def replace_sqlite_with_snapshot(
    db_path: str | Path, snapshot: dict[str, Any], *, expected_spec_digest: str | None = None
) -> Ledger:
    """Wholesale replacement; never merges row-by-row with a corrupt DB."""

    target = Path(db_path)
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(target) + suffix)
        if candidate.exists():
            candidate.unlink()
    ledger = Ledger.open(target, expected_spec_digest=expected_spec_digest)
    shards = snapshot.get("state", {}).get("shards", [])
    leases = snapshot.get("state", {}).get("leases", [])
    ledger.commit_batch(
        "snapshot_wholesale_replace",
        {
            "source_digest": snapshot.get("content_digest"),
            "source_head_digest": snapshot.get("head_digest"),
            "shard_count": len(shards),
            "lease_count": len(leases),
            "merge_policy": "whole_replacement_not_incremental_merge",
        },
        shard_states=shards,
        lease_states=leases,
    )
    return ledger


def open_or_replace_from_snapshot(
    db_path: str | Path,
    snapshot_path: str | Path,
    *,
    expected_spec_digest: str | None = None,
) -> tuple[Ledger, str]:
    try:
        return Ledger.open(db_path, expected_spec_digest=expected_spec_digest), "opened"
    except LedgerError as original:
        try:
            snapshot = verify_snapshot_file(snapshot_path)
        except LedgerError:
            raise original
        quarantined = Path(str(db_path) + f".corrupt-{int(time.time())}")
        if Path(db_path).exists():
            Path(db_path).replace(quarantined)
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(db_path) + suffix)
            if sidecar.exists():
                sidecar.unlink()
        ledger = replace_sqlite_with_snapshot(
            db_path, snapshot, expected_spec_digest=expected_spec_digest
        )
        return ledger, f"replaced_from_snapshot;quarantine={quarantined}"
