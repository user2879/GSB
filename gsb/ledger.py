from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from . import LEDGER_FORMAT_VERSION
from .canonical import canonical_dumps, digest
from .errors import VerifyError


SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS ledger_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS lease_records (
        lease_id TEXT PRIMARY KEY,
        shard_id TEXT NOT NULL,
        node_id TEXT NOT NULL,
        capacity_version INTEGER NOT NULL,
        lease_version INTEGER NOT NULL,
        resource_units INTEGER NOT NULL,
        status TEXT NOT NULL,
        granted_monotonic_ms INTEGER NOT NULL,
        expires_monotonic_ms INTEGER NOT NULL,
        record_digest TEXT NOT NULL,
        committed_at_ns INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS outcomes (
        shard_id TEXT PRIMARY KEY,
        cursor INTEGER NOT NULL UNIQUE,
        topo_index INTEGER NOT NULL,
        lease_id TEXT NOT NULL,
        node_id TEXT NOT NULL,
        capacity_version INTEGER NOT NULL,
        lease_version INTEGER NOT NULL,
        status TEXT NOT NULL,
        summary_json TEXT NOT NULL,
        summary_digest TEXT NOT NULL,
        resource_units INTEGER NOT NULL,
        water_q INTEGER NOT NULL,
        scale_factor INTEGER NOT NULL,
        wait_started_ms INTEGER,
        wait_ended_ms INTEGER,
        started_monotonic_ms INTEGER,
        finished_monotonic_ms INTEGER,
        node_elapsed_ms INTEGER,
        request_digest TEXT,
        propagation_basis_json TEXT,
        record_digest TEXT NOT NULL,
        committed_at_ns INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS occupancy_events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        lease_id TEXT NOT NULL,
        shard_id TEXT NOT NULL,
        node_id TEXT NOT NULL,
        delta_units INTEGER NOT NULL,
        active_units_after INTEGER NOT NULL,
        reason TEXT NOT NULL,
        capacity_version INTEGER NOT NULL,
        monotonic_ms INTEGER NOT NULL,
        record_digest TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS recovery_runs (
        recovery_id INTEGER PRIMARY KEY AUTOINCREMENT,
        recovered_at_ns INTEGER NOT NULL,
        reclaimed_leases INTEGER NOT NULL,
        reclaimed_units INTEGER NOT NULL,
        unknown_shards_json TEXT NOT NULL,
        report_digest TEXT NOT NULL
    )
    """,
]


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.parent and str(self.path.parent) not in ("", "."):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.isolation_level = None
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute("PRAGMA journal_mode=DELETE")
        self.conn.execute(f"PRAGMA user_version={LEDGER_FORMAT_VERSION}")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                for statement in SCHEMA_STATEMENTS:
                    self.conn.execute(statement)
                self.conn.execute(
                    "INSERT OR IGNORE INTO ledger_meta(key, value) VALUES('ledger_format_version', ?)",
                    (str(LEDGER_FORMAT_VERSION),),
                )
                self.conn.execute(
                    "INSERT OR IGNORE INTO ledger_meta(key, value) VALUES('next_cursor', '1')"
                )
                self.conn.execute(
                    "INSERT OR IGNORE INTO ledger_meta(key, value) VALUES('content_digest', '')"
                )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def set_spec_digest(self, spec_digest: str) -> None:
        self.set_meta("spec_digest", spec_digest)

    def set_meta(self, key: str, value: Any) -> None:
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                self.conn.execute(
                    "INSERT INTO ledger_meta(key, value) VALUES(?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (key, str(value)),
                )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM ledger_meta WHERE key=?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def format_version(self) -> int:
        value = self.get_meta("ledger_format_version")
        if value is None:
            raise VerifyError("ledger lacks integer format version", field="ledger_meta")
        try:
            version = int(value)
        except ValueError as exc:
            raise VerifyError("ledger format version is not an integer", field="ledger_meta") from exc
        if version != LEDGER_FORMAT_VERSION:
            raise VerifyError(
                f"unsupported ledger format version {version}",
                field="ledger_format_version",
                expected=LEDGER_FORMAT_VERSION,
                actual=version,
            )
        return version

    def record_lease_grant(self, lease: dict[str, Any]) -> int:
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                now_ns = time.time_ns()
                record_digest = digest({"kind": "lease_grant", "lease": lease})
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO lease_records(
                        lease_id, shard_id, node_id, capacity_version, lease_version,
                        resource_units, status, granted_monotonic_ms,
                        expires_monotonic_ms, record_digest, committed_at_ns
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        lease["lease_id"],
                        lease["shard_id"],
                        lease["node_id"],
                        int(lease["capacity_version"]),
                        int(lease["lease_version"]),
                        int(lease["resource_units"]),
                        lease["status"],
                        int(lease["granted_monotonic_ms"]),
                        int(lease["expires_monotonic_ms"]),
                        record_digest,
                        now_ns,
                    ),
                )
                active_after = self._insert_occupancy(
                    lease,
                    delta=int(lease["resource_units"]),
                    reason="grant",
                    monotonic_ms=int(lease["granted_monotonic_ms"]),
                )
                self.conn.commit()
                return active_after
            except Exception:
                self.conn.rollback()
                raise

    def commit_outcome(self, outcome: dict[str, Any], lease: dict[str, Any]) -> int:
        """Atomically commit one shard ending and advance its cursor."""

        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                cursor = int(self.get_meta("next_cursor") or "1")
                outcome = {**outcome, "cursor": cursor}
                record_digest = digest({"kind": "outcome", "outcome": _digestable_outcome(outcome)})
                summary_json = canonical_dumps(outcome["summary"])
                now_ns = time.time_ns()
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO outcomes(
                        shard_id, cursor, topo_index, lease_id, node_id,
                        capacity_version, lease_version, status, summary_json,
                        summary_digest, resource_units, water_q, scale_factor,
                        wait_started_ms, wait_ended_ms, started_monotonic_ms,
                        finished_monotonic_ms, node_elapsed_ms, request_digest,
                        propagation_basis_json, record_digest, committed_at_ns
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        outcome["shard_id"],
                        cursor,
                        int(outcome["topo_index"]),
                        outcome["lease_id"],
                        outcome["node_id"],
                        int(outcome["capacity_version"]),
                        int(outcome["lease_version"]),
                        outcome.get("status", "committed"),
                        summary_json,
                        outcome["summary_digest"],
                        int(outcome["resource_units"]),
                        int(outcome["water_q"]),
                        int(outcome["scale_factor"]),
                        outcome.get("wait_started_ms"),
                        outcome.get("wait_ended_ms"),
                        outcome.get("started_monotonic_ms"),
                        outcome.get("finished_monotonic_ms"),
                        outcome.get("node_elapsed_ms"),
                        outcome.get("request_digest"),
                        canonical_dumps(outcome["propagation_basis"])
                        if outcome.get("propagation_basis") is not None
                        else None,
                        record_digest,
                        now_ns,
                    ),
                )
                self.conn.execute(
                    """
                    UPDATE lease_records
                       SET status=?, lease_version=?, record_digest=?
                     WHERE lease_id=?
                    """,
                    (
                        lease.get("status", "completed"),
                        int(lease["lease_version"]),
                        digest({"kind": "lease_complete", "lease": lease}),
                        lease["lease_id"],
                    ),
                )
                active_after = self._insert_occupancy(
                    lease,
                    delta=-int(outcome["resource_units"]),
                    reason="completed",
                    monotonic_ms=int(outcome.get("finished_monotonic_ms") or 0),
                )
                self.conn.execute(
                    "INSERT INTO ledger_meta(key,value) VALUES('next_cursor', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (str(cursor + 1),),
                )
                content_digest = self._recompute_content_digest_locked()
                self.conn.execute(
                    "INSERT INTO ledger_meta(key,value) VALUES('content_digest', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (content_digest,),
                )
                self.conn.commit()
                return active_after
            except Exception:
                self.conn.rollback()
                raise

    def release_lease(
        self,
        lease: dict[str, Any],
        *,
        reason: str,
        monotonic_ms: int,
    ) -> int:
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                self.conn.execute(
                    "UPDATE lease_records SET status=?, record_digest=? WHERE lease_id=?",
                    (reason, digest({"kind": "lease_release", "lease": lease, "reason": reason}), lease["lease_id"]),
                )
                active_after = self._insert_occupancy(
                    lease,
                    delta=-int(lease["resource_units"]),
                    reason=reason,
                    monotonic_ms=monotonic_ms,
                )
                self.conn.commit()
                return active_after
            except Exception:
                self.conn.rollback()
                raise

    def _insert_occupancy(
        self,
        lease: dict[str, Any],
        *,
        delta: int,
        reason: str,
        monotonic_ms: int,
    ) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(SUM(delta_units), 0) AS active FROM occupancy_events"
        ).fetchone()
        active_after = int(row["active"]) + int(delta)
        event = {
            "lease_id": lease["lease_id"],
            "shard_id": lease["shard_id"],
            "node_id": lease["node_id"],
            "delta_units": int(delta),
            "active_units_after": active_after,
            "reason": reason,
            "capacity_version": int(lease["capacity_version"]),
            "monotonic_ms": int(monotonic_ms),
        }
        self.conn.execute(
            """
            INSERT INTO occupancy_events(
                lease_id, shard_id, node_id, delta_units, active_units_after,
                reason, capacity_version, monotonic_ms, record_digest
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event["lease_id"],
                event["shard_id"],
                event["node_id"],
                event["delta_units"],
                event["active_units_after"],
                event["reason"],
                event["capacity_version"],
                event["monotonic_ms"],
                digest({"kind": "occupancy", "event": event}),
            ),
        )
        return active_after

    def list_outcomes(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM outcomes ORDER BY cursor").fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["summary"] = __import__("json").loads(item.pop("summary_json"))
            result.append(item)
        return result

    def get_outcome(self, shard_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM outcomes WHERE shard_id=?", (shard_id,)).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["summary"] = __import__("json").loads(item.pop("summary_json"))
        return item

    def outcome_record_bytes(self, shard_id: str) -> bytes:
        row = self.conn.execute("SELECT * FROM outcomes WHERE shard_id=?", (shard_id,)).fetchone()
        if row is None:
            raise KeyError(shard_id)
        return canonical_dumps(dict(row)).encode("utf-8")

    def active_lease_records(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM lease_records WHERE status IN ('granted', 'renewed') ORDER BY lease_id"
        ).fetchall()
        return [dict(row) for row in rows]

    def recover(self, spec: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reclaim all complete active-lease records from a dead process."""

        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                active = [dict(row) for row in self.conn.execute(
                    "SELECT * FROM lease_records WHERE status IN ('granted', 'renewed')"
                ).fetchall()]
                reclaimed_units = 0
                now_ns = time.time_ns()
                monotonic = now_ns // 1_000_000
                for record in active:
                    lease = {
                        "lease_id": record["lease_id"],
                        "shard_id": record["shard_id"],
                        "node_id": record["node_id"],
                        "capacity_version": record["capacity_version"],
                        "resource_units": record["resource_units"],
                    }
                    reclaimed_units += int(record["resource_units"])
                    self.conn.execute(
                        "UPDATE lease_records SET status='reclaimed', record_digest=? WHERE lease_id=?",
                        (digest({"kind": "recovery_reclaim", "lease_id": record["lease_id"]}), record["lease_id"]),
                    )
                    self._insert_occupancy(
                        lease,
                        delta=-int(record["resource_units"]),
                        reason="crash_recovery_reclaim",
                        monotonic_ms=monotonic,
                    )
                unknown: list[str] = []
                if spec is not None:
                    committed = {
                        row["shard_id"]
                        for row in self.conn.execute("SELECT shard_id FROM outcomes").fetchall()
                    }
                    unknown = [
                        shard_id
                        for shard_id in spec["topo_order"]
                        if shard_id not in committed
                    ]
                report = {
                    "reclaimed_leases": len(active),
                    "reclaimed_units": reclaimed_units,
                    "unknown_shards": unknown,
                    "recovered_at_ns": now_ns,
                }
                self.conn.execute(
                    """
                    INSERT INTO recovery_runs(
                        recovered_at_ns, reclaimed_leases, reclaimed_units,
                        unknown_shards_json, report_digest
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        now_ns,
                        len(active),
                        reclaimed_units,
                        canonical_dumps(unknown),
                        digest(report),
                    ),
                )
                content_digest = self._recompute_content_digest_locked()
                self.conn.execute(
                    "INSERT INTO ledger_meta(key,value) VALUES('content_digest', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (content_digest,),
                )
                self.conn.commit()
                return report
            except Exception:
                self.conn.rollback()
                raise

    def current_active_units(self) -> int:
        row = self.conn.execute("SELECT COALESCE(SUM(delta_units),0) AS units FROM occupancy_events").fetchone()
        return int(row["units"])

    def stored_content_digest(self) -> str | None:
        return self.get_meta("content_digest")

    def recompute_content_digest(self) -> str:
        with self._lock:
            return self._recompute_content_digest_locked()

    def _recompute_content_digest_locked(self) -> str:
        outcome_rows = self.conn.execute("SELECT * FROM outcomes ORDER BY cursor").fetchall()
        lease_rows = self.conn.execute("SELECT * FROM lease_records ORDER BY lease_id").fetchall()
        payload = {
            "format_version": LEDGER_FORMAT_VERSION,
            "outcomes": [dict(row) for row in outcome_rows],
            "leases": [dict(row) for row in lease_rows],
        }
        return digest(payload)

    def verify_content_digest(self) -> None:
        stored = self.stored_content_digest()
        actual = self.recompute_content_digest()
        if stored != actual:
            raise VerifyError(
                "ledger content digest mismatch",
                field="content_digest",
                expected=stored,
                actual=actual,
            )


def _digestable_outcome(outcome: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in outcome.items() if key not in {"cursor", "record_digest"}}
