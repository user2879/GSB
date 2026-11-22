"""Transactional SQLite event ledger with digest chaining and crash markers."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from .canonical import (
    FORMAT_VERSION,
    canonical_json,
    chain_digest,
    digest_value,
    sha256_bytes,
)
from .engine import EventResult
from .errors import WorkbenchError
from .spec import Declaration


EMPTY_LEDGER_DIGEST = sha256_bytes(b"quota-workbench-ledger-v1")


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ledger_info (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    format_version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS event_attempts (
    event_id INTEGER PRIMARY KEY,
    status TEXT NOT NULL,
    reason TEXT
);
CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY,
    time_ms INTEGER NOT NULL,
    trigger_type TEXT NOT NULL,
    target_tenant TEXT,
    target_workload TEXT,
    pressure_units INTEGER NOT NULL,
    terminal_code TEXT NOT NULL,
    terminal_reason TEXT NOT NULL,
    iteration_count INTEGER NOT NULL,
    preemption_count INTEGER NOT NULL,
    reclaimed_units INTEGER NOT NULL,
    event_json TEXT NOT NULL,
    record_json TEXT NOT NULL,
    record_digest TEXT NOT NULL,
    chain_digest TEXT NOT NULL,
    state_json TEXT NOT NULL,
    final_matrix_json TEXT NOT NULL,
    gap_json TEXT
);
CREATE TABLE IF NOT EXISTS tenant_matrices (
    event_id INTEGER NOT NULL,
    iteration INTEGER NOT NULL,
    tenant_id TEXT NOT NULL,
    priority INTEGER NOT NULL,
    share_bp INTEGER NOT NULL,
    reserve_units INTEGER NOT NULL,
    share_floor_units INTEGER NOT NULL,
    share_remainder_numerator INTEGER NOT NULL,
    share_rounding_bonus_units INTEGER NOT NULL,
    share_entitlement_units INTEGER NOT NULL,
    reserve_uplift_units INTEGER NOT NULL,
    capacity_reduction_units INTEGER NOT NULL,
    entitlement_units INTEGER NOT NULL,
    share_quantization_error_scaled_1e4 INTEGER NOT NULL,
    entitlement_error_scaled_1e4 INTEGER NOT NULL,
    workload_units INTEGER NOT NULL,
    pressure_units INTEGER NOT NULL,
    actual_units INTEGER NOT NULL,
    excess_units INTEGER NOT NULL,
    deviation_units INTEGER NOT NULL,
    headroom_units INTEGER NOT NULL,
    reserve_remaining_units INTEGER NOT NULL,
    matrix_digest TEXT NOT NULL,
    PRIMARY KEY (event_id, iteration, tenant_id),
    FOREIGN KEY (event_id) REFERENCES events(event_id)
);
CREATE TABLE IF NOT EXISTS iterations (
    event_id INTEGER NOT NULL,
    iteration INTEGER NOT NULL,
    terminal_reason TEXT,
    matrix_digest TEXT NOT NULL,
    matrix_json TEXT NOT NULL,
    gap_json TEXT,
    PRIMARY KEY (event_id, iteration),
    FOREIGN KEY (event_id) REFERENCES events(event_id)
);
CREATE TABLE IF NOT EXISTS preemptions (
    event_id INTEGER NOT NULL,
    sequence INTEGER NOT NULL,
    iteration INTEGER NOT NULL,
    tenant_id TEXT NOT NULL,
    workload_id TEXT NOT NULL,
    replicas_preempted INTEGER NOT NULL,
    units_per_replica INTEGER NOT NULL,
    reclaimed_units INTEGER NOT NULL,
    tenant_usage_before_units INTEGER NOT NULL,
    tenant_usage_after_units INTEGER NOT NULL,
    cluster_usage_before_units INTEGER NOT NULL,
    cluster_usage_after_units INTEGER NOT NULL,
    matrix_before_digest TEXT NOT NULL,
    matrix_after_digest TEXT NOT NULL,
    selection_basis_json TEXT NOT NULL,
    PRIMARY KEY (event_id, sequence),
    FOREIGN KEY (event_id) REFERENCES events(event_id)
);
"""


class Ledger:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, isolation_level=None, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = FULL")
        self.conn.executescript(SCHEMA)
        self._initialize_meta()

    def close(self) -> None:
        self.conn.close()

    def _meta_get(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def _meta_set(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def _initialize_meta(self) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO ledger_info(singleton, format_version) VALUES(1, ?)",
            (FORMAT_VERSION,),
        )
        row = self.conn.execute(
            "SELECT format_version FROM ledger_info WHERE singleton = 1"
        ).fetchone()
        version = int(row["format_version"]) if row is not None else None
        if version != FORMAT_VERSION:
            raise WorkbenchError(
                "E_LEDGER",
                f"unsupported ledger format version {version}",
                object_type="ledger",
                object_id=str(self.path),
                field="format_version",
                details={"expected": FORMAT_VERSION, "actual": version},
            )
        defaults = {
            "cursor": "0",
            "head_digest": EMPTY_LEDGER_DIGEST,
        }
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            for key, value in defaults.items():
                if self._meta_get(key) is None:
                    self._meta_set(key, value)
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    @property
    def cursor(self) -> int:
        value = self._meta_get("cursor")
        return int(value or "0")

    @property
    def head_digest(self) -> str:
        value = self._meta_get("head_digest")
        return value or EMPTY_LEDGER_DIGEST

    def ensure_context(self, declaration: Declaration) -> None:
        expected = {
            "tenant_set_version": declaration.tenant_set_version(),
            "workload_declaration_version": declaration.workload_declaration_version(),
            "quota_context_version": declaration.quota_context_version(),
            "spec_version": declaration.spec_version(),
            "cluster_total_units": str(declaration.cluster_total_units),
        }
        actual = {key: self._meta_get(key) for key in expected}
        if all(value is None for value in actual.values()) and self.cursor == 0:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                for key, value in expected.items():
                    self._meta_set(key, value)
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise
            return
        mismatches = {
            key: {"expected": value, "actual": actual[key]}
            for key, value in expected.items()
            if actual[key] != value
        }
        if mismatches:
            raise WorkbenchError(
                "E_CONTEXT",
                "ledger declaration context differs from the supplied declaration",
                object_type="ledger",
                object_id=str(self.path),
                field=sorted(mismatches)[0],
                details={"mismatches": mismatches, "unreproducible": True},
            )

    def context_versions(self) -> dict[str, Any]:
        keys = (
            "tenant_set_version",
            "workload_declaration_version",
            "quota_context_version",
            "spec_version",
        )
        result: dict[str, Any] = {}
        for key in keys:
            result[key] = self._meta_get(key)
        total = self._meta_get("cluster_total_units")
        result["cluster_total_units"] = int(total) if total is not None else None
        result["format_version"] = FORMAT_VERSION
        return result

    def note_startup_recovery(self) -> list[int]:
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            rows = self.conn.execute(
                "SELECT event_id FROM event_attempts WHERE status = 'IN_PROGRESS' ORDER BY event_id"
            ).fetchall()
            unknown = [int(row["event_id"]) for row in rows]
            if unknown:
                self.conn.execute(
                    "UPDATE event_attempts SET status = 'UNKNOWN_REQUIRES_RERUN', "
                    "reason = 'process terminated before committed convergence' "
                    "WHERE status = 'IN_PROGRESS'"
                )
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        return unknown

    def unknown_attempts(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT event_id, status, reason FROM event_attempts "
            "WHERE status IN ('UNKNOWN_REQUIRES_RERUN', 'FAILED') ORDER BY event_id"
        ).fetchall()
        return [dict(row) for row in rows]

    def recovery_report(self) -> dict[str, Any]:
        unknown = self.unknown_attempts()
        return {
            "status": "UNKNOWN_REQUIRES_RERUN" if unknown else "CLEAN",
            "unknown_event_ids": [item["event_id"] for item in unknown],
            "attempts": unknown,
            "statement": (
                "Only fully committed events may be restored. "
                "Uncommitted convergence must be rerun; no exit report is claimable."
                if unknown
                else "All recorded attempts are committed."
            ),
        }

    def prepare_event(self, event_id: int) -> None:
        expected = self.cursor + 1
        if event_id != expected:
            raise WorkbenchError(
                "E_LEDGER",
                f"event cursor mismatch: expected event {expected}, got {event_id}",
                object_type="event",
                object_id=str(event_id),
                field="event_id",
                details={"cursor": self.cursor},
            )
        exists = self.conn.execute(
            "SELECT 1 FROM events WHERE event_id = ?", (event_id,)
        ).fetchone()
        if exists is not None:
            raise WorkbenchError(
                "E_LEDGER",
                f"event {event_id} is already committed",
                object_type="event",
                object_id=str(event_id),
                field="event_id",
            )
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.conn.execute(
                "INSERT INTO event_attempts(event_id, status, reason) VALUES(?, 'IN_PROGRESS', NULL) "
                "ON CONFLICT(event_id) DO UPDATE SET status='IN_PROGRESS', reason=NULL",
                (event_id,),
            )
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def fail_attempt(self, event_id: int, reason: str) -> None:
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.conn.execute(
                "INSERT INTO event_attempts(event_id, status, reason) VALUES(?, 'FAILED', ?) "
                "ON CONFLICT(event_id) DO UPDATE SET status='FAILED', reason=excluded.reason",
                (event_id, reason),
            )
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def _insert_record_parts(self, record: dict[str, Any]) -> None:
        event_id = int(record["event_id"])
        matrix = record["final_matrix"]
        state_json = canonical_json(record["workload_state"])
        final_matrix_json = canonical_json(matrix)
        gap_json = canonical_json(record["gap"]) if record["gap"] is not None else None
        event = record["event"]
        terminal = record["terminal"]
        self.conn.execute(
            "INSERT INTO events("
            "event_id, time_ms, trigger_type, target_tenant, target_workload, pressure_units, "
            "terminal_code, terminal_reason, iteration_count, preemption_count, reclaimed_units, "
            "event_json, record_json, record_digest, chain_digest, state_json, final_matrix_json, gap_json"
            ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                event_id,
                int(event["time_ms"]),
                event["trigger_type"],
                event.get("target_tenant"),
                event.get("target_workload"),
                int(event["pressure_units"]),
                terminal["code"],
                terminal["reason"],
                int(terminal["iteration_count"]),
                int(terminal["preemption_count"]),
                int(terminal["reclaimed_units"]),
                canonical_json(event),
                canonical_json(record),
                digest_value(record),
                "",
                state_json,
                final_matrix_json,
                gap_json,
            ),
        )
        for iteration in record["iterations"]:
            iteration_no = int(iteration["iteration"])
            iteration_matrix = iteration["matrix"]
            matrix_digest = iteration["matrix_digest"]
            gap_value = iteration.get("gap")
            self.conn.execute(
                "INSERT INTO iterations(event_id, iteration, terminal_reason, matrix_digest, "
                "matrix_json, gap_json) VALUES(?,?,?,?,?,?)",
                (
                    event_id,
                    iteration_no,
                    iteration.get("terminal_reason"),
                    matrix_digest,
                    canonical_json(iteration_matrix),
                    canonical_json(gap_value) if gap_value is not None else None,
                ),
            )
            for row in iteration_matrix["tenants"]:
                self.conn.execute(
                    "INSERT INTO tenant_matrices("
                    "event_id, iteration, tenant_id, priority, share_bp, reserve_units, "
                    "share_floor_units, share_remainder_numerator, share_rounding_bonus_units, "
                    "share_entitlement_units, reserve_uplift_units, capacity_reduction_units, "
                    "entitlement_units, share_quantization_error_scaled_1e4, "
                    "entitlement_error_scaled_1e4, workload_units, pressure_units, actual_units, "
                    "excess_units, deviation_units, headroom_units, reserve_remaining_units, matrix_digest"
                    ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        event_id,
                        iteration_no,
                        row["tenant_id"],
                        row["priority"],
                        row["share_bp"],
                        row["reserve_units"],
                        row["share_floor_units"],
                        row["share_remainder_numerator"],
                        row["share_rounding_bonus_units"],
                        row["share_entitlement_units"],
                        row["reserve_uplift_units"],
                        row["capacity_reduction_units"],
                        row["entitlement_units"],
                        row["share_quantization_error_scaled_1e4"],
                        row["entitlement_error_scaled_1e4"],
                        row["workload_units"],
                        row["pressure_units"],
                        row["actual_units"],
                        row["excess_units"],
                        row["deviation_units"],
                        row["headroom_units"],
                        row["reserve_remaining_units"],
                        matrix_digest,
                    ),
                )
        for preemption in record["preemptions"]:
            self.conn.execute(
                "INSERT INTO preemptions("
                "event_id, sequence, iteration, tenant_id, workload_id, replicas_preempted, "
                "units_per_replica, reclaimed_units, tenant_usage_before_units, "
                "tenant_usage_after_units, cluster_usage_before_units, cluster_usage_after_units, "
                "matrix_before_digest, matrix_after_digest, selection_basis_json"
                ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    event_id,
                    preemption["sequence"],
                    preemption["iteration"],
                    preemption["tenant_id"],
                    preemption["workload_id"],
                    preemption["replicas_preempted"],
                    preemption["units_per_replica"],
                    preemption["reclaimed_units"],
                    preemption["tenant_usage_before_units"],
                    preemption["tenant_usage_after_units"],
                    preemption["cluster_usage_before_units"],
                    preemption["cluster_usage_after_units"],
                    preemption["matrix_before_digest"],
                    preemption["matrix_after_digest"],
                    canonical_json(preemption["selection_basis"]),
                ),
            )

    def commit_result(self, result: EventResult) -> None:
        if result.event_id != self.cursor + 1:
            raise WorkbenchError(
                "E_LEDGER",
                f"cannot commit event {result.event_id} at cursor {self.cursor}",
                object_type="event",
                object_id=str(result.event_id),
                field="cursor",
            )
        record = result.record
        record_json = canonical_json(record)
        record_digest = sha256_bytes(record_json.encode("utf-8"))
        if record_json != result.record_json or record_digest != result.record_digest:
            raise WorkbenchError(
                "E_LEDGER",
                "result record is not canonical",
                object_type="event",
                object_id=str(result.event_id),
                field="record_json",
            )
        previous_head = self.head_digest
        next_head = chain_digest(previous_head, record_digest)
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self._insert_record_parts(record)
            self.conn.execute(
                "UPDATE events SET chain_digest = ? WHERE event_id = ?",
                (next_head, result.event_id),
            )
            self._meta_set("cursor", str(result.event_id))
            self._meta_set("head_digest", next_head)
            self.conn.execute(
                "INSERT INTO event_attempts(event_id, status, reason) VALUES(?, 'COMMITTED', NULL) "
                "ON CONFLICT(event_id) DO UPDATE SET status='COMMITTED', reason=NULL",
                (result.event_id,),
            )
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def latest_state(self) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT state_json FROM events ORDER BY event_id DESC LIMIT 1"
        ).fetchone()
        return None if row is None else json.loads(row["state_json"])

    def latest_matrix(self) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT final_matrix_json FROM events ORDER BY event_id DESC LIMIT 1"
        ).fetchone()
        return None if row is None else json.loads(row["final_matrix_json"])

    def event_rows(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT event_id, time_ms, trigger_type, target_tenant, target_workload, "
            "pressure_units, terminal_code, terminal_reason, iteration_count, "
            "preemption_count, reclaimed_units, record_digest, chain_digest "
            "FROM events ORDER BY event_id"
        ).fetchall()
        return [dict(row) for row in rows]

    def event_storage_row(self, event_id: int) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM events WHERE event_id = ?", (event_id,)
        ).fetchone()
        return None if row is None else dict(row)

    def committed_records(self, *, verify_digests: bool = True) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT event_id, record_json, record_digest, chain_digest "
            "FROM events ORDER BY event_id"
        ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            raw = str(row["record_json"])
            recomputed = sha256_bytes(raw.encode("utf-8"))
            if verify_digests and recomputed != row["record_digest"]:
                raise WorkbenchError(
                    "E_LEDGER",
                    "stored record digest does not match bytes",
                    object_type="event",
                    object_id=str(row["event_id"]),
                    field="record_digest",
                    details={"stored": row["record_digest"], "recomputed": recomputed},
                )
            result.append(
                {
                    "event_id": int(row["event_id"]),
                    "record": json.loads(raw),
                    "record_json": raw,
                    "record_digest": str(row["record_digest"]),
                    "chain_digest": str(row["chain_digest"]),
                }
            )
        return result

    def verify_chain(self) -> dict[str, Any]:
        previous = EMPTY_LEDGER_DIGEST
        records = self.committed_records()
        if len(records) != self.cursor:
            raise WorkbenchError(
                "E_LEDGER",
                "ledger cursor does not match committed event count",
                object_type="ledger",
                object_id=str(self.path),
                field="cursor",
                details={"cursor": self.cursor, "event_count": len(records)},
            )
        for item in records:
            expected = chain_digest(previous, item["record_digest"])
            if item["chain_digest"] != expected:
                raise WorkbenchError(
                    "E_LEDGER",
                    "ledger digest chain is broken",
                    object_type="event",
                    object_id=str(item["event_id"]),
                    field="chain_digest",
                    details={"expected": expected, "actual": item["chain_digest"]},
                )
            previous = expected
        if previous != self.head_digest:
            raise WorkbenchError(
                "E_LEDGER",
                "ledger head digest does not match event chain",
                object_type="ledger",
                object_id=str(self.path),
                field="head_digest",
                details={"expected": previous, "actual": self.head_digest},
            )
        return {"event_count": self.cursor, "head_digest": previous}

    def matrices_for_event(self, event_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM tenant_matrices WHERE event_id = ? "
            "ORDER BY iteration, tenant_id",
            (event_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def iterations_for_event(self, event_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM iterations WHERE event_id = ? ORDER BY iteration",
            (event_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def preemptions_for_event(self, event_id: int) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM preemptions WHERE event_id = ? ORDER BY sequence",
            (event_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def all_preemptions(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM preemptions ORDER BY event_id, sequence"
        ).fetchall()
        return [dict(row) for row in rows]

    def export_snapshot_events(self) -> list[dict[str, Any]]:
        return [
            {
                "event_id": item["event_id"],
                "record": item["record"],
                "record_digest": item["record_digest"],
                "chain_digest": item["chain_digest"],
            }
            for item in self.committed_records()
        ]

    def import_snapshot_events(
        self,
        records: Iterable[dict[str, Any]],
        *,
        expected_cursor: int,
        expected_head: str,
        versions: dict[str, Any],
    ) -> None:
        if self.cursor != 0 or self.conn.execute("SELECT 1 FROM events LIMIT 1").fetchone():
            raise WorkbenchError(
                "E_SNAPSHOT",
                "snapshot restore requires an empty destination ledger",
                object_type="ledger",
                object_id=str(self.path),
                field="destination",
            )
        previous = EMPTY_LEDGER_DIGEST
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            for item in records:
                record = item["record"]
                raw = canonical_json(record)
                digest = sha256_bytes(raw.encode("utf-8"))
                if digest != item["record_digest"]:
                    raise WorkbenchError(
                        "E_SNAPSHOT",
                        "snapshot record digest does not match record content",
                        object_type="event",
                        object_id=str(item["event_id"]),
                        field="record_digest",
                    )
                expected_chain = chain_digest(previous, digest)
                if expected_chain != item["chain_digest"]:
                    raise WorkbenchError(
                        "E_SNAPSHOT",
                        "snapshot record chain digest is invalid",
                        object_type="event",
                        object_id=str(item["event_id"]),
                        field="chain_digest",
                    )
                self._insert_record_parts(record)
                self.conn.execute(
                    "UPDATE events SET chain_digest = ? WHERE event_id = ?",
                    (expected_chain, item["event_id"]),
                )
                previous = expected_chain
            if previous != expected_head:
                raise WorkbenchError(
                    "E_SNAPSHOT",
                    "snapshot ledger head digest does not match imported records",
                    object_type="snapshot",
                    field="ledger.head_digest",
                    details={"expected": expected_head, "actual": previous},
                )
            max_event_id = self.conn.execute(
                "SELECT MAX(event_id) AS m FROM events"
            ).fetchone()["m"]
            if (max_event_id or 0) != expected_cursor:
                raise WorkbenchError(
                    "E_SNAPSHOT",
                    "snapshot cursor does not match imported event count",
                    object_type="snapshot",
                    field="ledger.cursor",
                )
            for key in (
                "tenant_set_version",
                "workload_declaration_version",
                "quota_context_version",
                "spec_version",
            ):
                value = versions.get(key)
                if not isinstance(value, str):
                    raise WorkbenchError(
                        "E_SNAPSHOT",
                        f"snapshot version {key} is missing",
                        object_type="snapshot",
                        field=f"versions.{key}",
                    )
                self._meta_set(key, value)
            cluster = versions.get("cluster_total_units")
            if not isinstance(cluster, int) or isinstance(cluster, bool):
                raise WorkbenchError(
                    "E_SNAPSHOT",
                    "snapshot cluster_total_units is missing",
                    object_type="snapshot",
                    field="versions.cluster_total_units",
                )
            self._meta_set("cluster_total_units", str(cluster))
            self._meta_set("cursor", str(expected_cursor))
            self._meta_set("head_digest", expected_head)
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
