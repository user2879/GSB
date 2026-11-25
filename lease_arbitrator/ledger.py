from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from . import LEDGER_FORMAT_VERSION
from .content import canonical_digest, chain_digest
from .errors import ArbitrationError


SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ledger_entries (
        entry_seq INTEGER PRIMARY KEY AUTOINCREMENT,
        entry_id TEXT NOT NULL UNIQUE,
        entry_type TEXT NOT NULL,
        event_seq INTEGER,
        status TEXT NOT NULL,
        prev_digest TEXT NOT NULL,
        digest TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_wall_ms INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS event_batches (
        event_seq INTEGER PRIMARY KEY,
        entry_id TEXT NOT NULL,
        batch_id TEXT NOT NULL UNIQUE,
        status TEXT NOT NULL,
        attempt INTEGER NOT NULL,
        rerun_of TEXT,
        cursor_seq INTEGER NOT NULL,
        digest TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        entry_id TEXT NOT NULL,
        event_seq INTEGER NOT NULL,
        sort_index INTEGER NOT NULL,
        record_type TEXT NOT NULL,
        artifact_key TEXT,
        node_id TEXT,
        lease_id TEXT,
        lease_ref TEXT,
        reservation_id TEXT,
        quota_version INTEGER,
        payload_json TEXT NOT NULL,
        record_digest TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS cursor_state (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        last_event_seq INTEGER NOT NULL,
        last_entry_id TEXT,
        updated_wall_ms INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS node_state (
        node_id TEXT PRIMARY KEY,
        quota_version INTEGER NOT NULL,
        capacity_slots INTEGER NOT NULL,
        lease_limit INTEGER NOT NULL,
        used_slots INTEGER NOT NULL,
        active_leases INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS leases (
        lease_id TEXT PRIMARY KEY,
        lease_ref TEXT,
        node_id TEXT NOT NULL,
        artifact_key TEXT NOT NULL,
        shard_index INTEGER NOT NULL,
        reservation_id TEXT,
        quota_version INTEGER NOT NULL,
        lease_version INTEGER NOT NULL,
        status TEXT NOT NULL,
        grant_wall_ms INTEGER NOT NULL,
        expiry_wall_ms INTEGER NOT NULL,
        grant_monotonic_ms INTEGER NOT NULL,
        expiry_monotonic_ms INTEGER NOT NULL,
        ttl_ms INTEGER NOT NULL,
        close_reason TEXT,
        closed_event_seq INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS conclusions (
        conclusion_id TEXT PRIMARY KEY,
        entry_id TEXT NOT NULL,
        event_seq INTEGER NOT NULL,
        kind TEXT NOT NULL,
        artifact_key TEXT,
        shard_index INTEGER,
        node_id TEXT,
        quota_version INTEGER NOT NULL,
        status TEXT NOT NULL,
        stale_reason TEXT,
        payload_json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS unknown_batches (
        batch_id TEXT PRIMARY KEY,
        event_seq INTEGER NOT NULL,
        entry_id TEXT NOT NULL UNIQUE,
        status TEXT NOT NULL,
        marker_json TEXT NOT NULL,
        replacement_batch_id TEXT,
        detected_wall_ms INTEGER NOT NULL
    )
    """,
)


def wall_ms() -> int:
    return time.time_ns() // 1_000_000


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA journal_mode=WAL")
        for statement in SCHEMA_STATEMENTS:
            self.conn.execute(statement)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is None:
            self.conn.commit()
        else:
            self.conn.rollback()
        self.close()

    def meta(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return str(row["value"]) if row else default

    def set_meta(self, key: str, value: Any) -> None:
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(value)),
        )

    @property
    def format_version(self) -> int:
        return int(self.meta("ledger_format_version", "0") or 0)

    @property
    def head_digest(self) -> str:
        return self.meta("head_digest", "") or ""

    @property
    def last_event_seq(self) -> int:
        row = self.conn.execute("SELECT last_event_seq FROM cursor_state WHERE id=1").fetchone()
        return int(row["last_event_seq"]) if row else 0

    def initialize_bundle(self, bundle: Any) -> None:
        existing = self.meta("spec_digest")
        if existing and existing != bundle.spec_digest:
            raise ArbitrationError(
                "ledger was initialized from different declaration contents",
                code="E_LEDGER_SPEC",
                field="spec_digest",
                details={"expected": existing, "received": bundle.spec_digest},
            )
        if existing:
            return
        with self.conn:
            self.set_meta("ledger_format_version", LEDGER_FORMAT_VERSION)
            self.set_meta("spec_digest", bundle.spec_digest)
            self.set_meta("source_digests", json.dumps(bundle.raw_digests, sort_keys=True))
            self.set_meta("current_quota_version", bundle.initial_quota_version)
            self.set_meta("active_nodes", json.dumps(bundle.node_ids))
            self.set_meta("watermark_scale", bundle.pipeline.watermark_scale)
            self.set_meta("high_watermark_fixed", bundle.pipeline.high_watermark_fixed)
            self.conn.execute(
                "INSERT INTO cursor_state(id,last_event_seq,last_entry_id,updated_wall_ms) VALUES(1,0,NULL,?)",
                (wall_ms(),),
            )
            for node in bundle.nodes:
                self.conn.execute(
                    """
                    INSERT INTO node_state(node_id,quota_version,capacity_slots,lease_limit,used_slots,active_leases)
                    VALUES(?,?,?,?,0,0)
                    """,
                    (
                        node.node_id,
                        bundle.initial_quota_version,
                        node.capacity_slots,
                        node.lease_limit,
                    ),
                )
            genesis_payload = {
                "ledger_format_version": LEDGER_FORMAT_VERSION,
                "spec_digest": bundle.spec_digest,
                "source_digests": bundle.raw_digests,
            }
            digest = chain_digest("GENESIS", genesis_payload)
            self.set_meta("head_digest", digest)
            self.set_meta("genesis_digest", digest)

    def _append_entry(
        self,
        *,
        entry_id: str,
        entry_type: str,
        payload: dict[str, Any],
        event_seq: int | None,
        status: str,
    ) -> str:
        previous = self.head_digest or "GENESIS"
        digest = chain_digest(previous, payload)
        self.conn.execute(
            """
            INSERT INTO ledger_entries(entry_id,entry_type,event_seq,status,prev_digest,digest,payload_json,created_wall_ms)
            VALUES(?,?,?,?,?,?,?,?)
            """,
            (
                entry_id,
                entry_type,
                event_seq,
                status,
                previous,
                digest,
                json.dumps(payload, ensure_ascii=False, sort_keys=True),
                wall_ms(),
            ),
        )
        self.set_meta("head_digest", digest)
        return digest

    def mark_unknown_batch(self, marker: dict[str, Any]) -> str:
        batch_id = marker["batch_id"]
        event_seq = int(marker["event_seq"])
        entry_id = f"unknown:{batch_id}"
        if self.conn.execute("SELECT 1 FROM unknown_batches WHERE batch_id=?", (batch_id,)).fetchone():
            return entry_id
        payload = {"kind": "unknown_batch", "marker": marker}
        with self.conn:
            digest = self._append_entry(
                entry_id=entry_id,
                entry_type="unknown_batch",
                payload=payload,
                event_seq=event_seq,
                status="unknown",
            )
            self.conn.execute(
                """
                INSERT INTO unknown_batches(batch_id,event_seq,entry_id,status,marker_json,detected_wall_ms)
                VALUES(?,?,?,'open',?,?)
                """,
                (batch_id, event_seq, entry_id, json.dumps(marker, sort_keys=True), wall_ms()),
            )
        return digest

    def open_unknown_batches(self) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM unknown_batches WHERE status='open' ORDER BY event_seq,batch_id"))

    def commit_event_batch(
        self,
        *,
        batch_id: str,
        attempt: int,
        event: Any,
        records: list[dict[str, Any]],
        node_states: dict[str, dict[str, Any]],
        leases: list[dict[str, Any]] | None = None,
        lease_updates: list[dict[str, Any]] | None = None,
        conclusions: list[dict[str, Any]] | None = None,
        current_quota_version: int | None = None,
        active_nodes: list[str] | None = None,
        quota_replacement: bool = False,
        replacement_for: str | None = None,
        meta_updates: dict[str, str] | None = None,
    ) -> str:
        entry_id = f"event:{event.seq}:a{attempt}"
        if self.conn.execute("SELECT 1 FROM ledger_entries WHERE entry_id=?", (entry_id,)).fetchone():
            row = self.conn.execute("SELECT digest FROM ledger_entries WHERE entry_id=?", (entry_id,)).fetchone()
            return str(row["digest"])

        payload = {
            "kind": "event_batch",
            "batch_id": batch_id,
            "attempt": attempt,
            "event": event.data,
            "records": records,
            "cursor_seq": event.seq,
            "replacement_for": replacement_for,
        }

        with self.conn:
            digest = self._append_entry(
                entry_id=entry_id,
                entry_type="event_batch",
                payload=payload,
                event_seq=event.seq,
                status="committed",
            )
            self.conn.execute(
                """
                INSERT INTO event_batches(event_seq,entry_id,batch_id,status,attempt,rerun_of,cursor_seq,digest)
                VALUES(?,?,?,'committed',?,?,?,?)
                """,
                (event.seq, entry_id, batch_id, attempt, replacement_for, event.seq, digest),
            )
            for sort_index, record in enumerate(records):
                record_payload = record.get("payload", record)
                self.conn.execute(
                    """
                    INSERT INTO records(entry_id,event_seq,sort_index,record_type,artifact_key,node_id,lease_id,
                                        lease_ref,reservation_id,quota_version,payload_json,record_digest)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        entry_id,
                        event.seq,
                        sort_index,
                        record["record_type"],
                        record.get("artifact_key"),
                        record.get("node_id"),
                        record.get("lease_id"),
                        record.get("lease_ref"),
                        record.get("reservation_id"),
                        record.get("quota_version", current_quota_version or int(self.meta("current_quota_version", "1"))),
                        json.dumps(record_payload, ensure_ascii=False, sort_keys=True),
                        canonical_digest(record_payload),
                    ),
                )

            for node_id, state in node_states.items():
                self.conn.execute(
                    """
                    UPDATE node_state
                       SET quota_version=?, capacity_slots=?, lease_limit=?, used_slots=?, active_leases=?
                     WHERE node_id=?
                    """,
                    (
                        int(state["quota_version"]),
                        int(state["capacity_slots"]),
                        int(state["lease_limit"]),
                        int(state["used_slots"]),
                        int(state["active_leases"]),
                        node_id,
                    ),
                )

            for lease in leases or []:
                columns = (
                    "lease_id", "lease_ref", "node_id", "artifact_key", "shard_index", "reservation_id",
                    "quota_version", "lease_version", "status", "grant_wall_ms", "expiry_wall_ms",
                    "grant_monotonic_ms", "expiry_monotonic_ms", "ttl_ms", "close_reason", "closed_event_seq",
                )
                placeholders = ",".join("?" for _ in columns)
                update = ",".join(f"{name}=excluded.{name}" for name in columns if name != "lease_id")
                values = tuple(lease.get(name) for name in columns)
                self.conn.execute(
                    f"INSERT INTO leases({','.join(columns)}) VALUES({placeholders}) "
                    f"ON CONFLICT(lease_id) DO UPDATE SET {update}",
                    values,
                )

            for update in lease_updates or []:
                self.conn.execute(
                    """
                    UPDATE leases
                       SET status=?, lease_version=COALESCE(?, lease_version), close_reason=?, closed_event_seq=?
                     WHERE lease_id=?
                    """,
                    (
                        update["status"],
                        update.get("lease_version"),
                        update.get("close_reason"),
                        update.get("closed_event_seq"),
                        update["lease_id"],
                    ),
                )

            if quota_replacement:
                version = current_quota_version or int(self.meta("current_quota_version", "1"))
                self.conn.execute(
                    """
                    UPDATE conclusions
                       SET status='stale', stale_reason=COALESCE(stale_reason, 'quota_or_topology_replacement')
                     WHERE status='current' AND quota_version < ?
                    """,
                    (version,),
                )
            else:
                for conclusion in conclusions or []:
                    if conclusion["kind"] == "arbitration":
                        self.conn.execute(
                            """
                            UPDATE conclusions SET status='superseded'
                             WHERE status='current' AND kind='arbitration'
                               AND COALESCE(artifact_key,'')=COALESCE(?,'')
                               AND COALESCE(shard_index,-1)=COALESCE(?,-1)
                            """,
                            (conclusion.get("artifact_key"), conclusion.get("shard_index")),
                        )
                    elif conclusion["kind"] == "watermark":
                        self.conn.execute(
                            "UPDATE conclusions SET status='superseded' WHERE status='current' AND kind='watermark' AND node_id=?",
                            (conclusion.get("node_id"),),
                        )

            for conclusion in conclusions or []:
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO conclusions(conclusion_id,entry_id,event_seq,kind,artifact_key,shard_index,
                                                       node_id,quota_version,status,stale_reason,payload_json)
                    VALUES(?,?,?,?,?,?,?,?, 'current', NULL, ?)
                    """,
                    (
                        conclusion["conclusion_id"],
                        entry_id,
                        event.seq,
                        conclusion["kind"],
                        conclusion.get("artifact_key"),
                        conclusion.get("shard_index"),
                        conclusion.get("node_id"),
                        conclusion.get("quota_version", current_quota_version or int(self.meta("current_quota_version", "1"))),
                        json.dumps(conclusion["payload"], ensure_ascii=False, sort_keys=True),
                    ),
                )

            self.conn.execute(
                "UPDATE cursor_state SET last_event_seq=?, last_entry_id=?, updated_wall_ms=? WHERE id=1",
                (event.seq, entry_id, wall_ms()),
            )
            if current_quota_version is not None:
                self.set_meta("current_quota_version", current_quota_version)
            if active_nodes is not None:
                self.set_meta("active_nodes", json.dumps(active_nodes, sort_keys=True))
            for key, value in (meta_updates or {}).items():
                self.set_meta(key, value)
            if replacement_for:
                self.conn.execute(
                    """
                    UPDATE unknown_batches
                       SET status='resolved', replacement_batch_id=?
                     WHERE batch_id=? AND status='open'
                    """,
                    (batch_id, replacement_for),
                )
        return digest

    def node_states(self) -> dict[str, dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM node_state ORDER BY node_id").fetchall()
        return {row["node_id"]: dict(row) for row in rows}

    def active_leases(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM leases WHERE status IN ('active','granted') ORDER BY lease_id").fetchall()
        return [dict(row) for row in rows]

    def all_records(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM records ORDER BY id").fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json"))
            result.append(item)
        return result

    def current_conclusions(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM conclusions WHERE status='current' ORDER BY kind, artifact_key, node_id").fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item.pop("payload_json"))
            result.append(item)
        return result

    def verify_hash_chain(self) -> list[str]:
        rows = self.conn.execute("SELECT * FROM ledger_entries ORDER BY entry_seq").fetchall()
        previous = "GENESIS"
        errors: list[str] = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            expected = chain_digest(previous, payload)
            if row["prev_digest"] != previous or row["digest"] != expected:
                errors.append(row["entry_id"])
            previous = row["digest"]
        if self.head_digest != previous:
            errors.append("meta:head_digest")
        return errors
