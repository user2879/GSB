"""Idempotent SQLite decision ledger with a single minute-level transaction.

Write atomicity contract:
* every completed minute inserts all service rows, the minute row and the
  progress cursor inside one SQLite transaction;
* re-writing the same (run_id, minute) is an idempotent upsert and never
  creates duplicate rows;
* a process killed mid-transaction leaves no partial minute, so recovery only
  trusts minutes whose progress cursor was durably committed.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .common import canonical_dumps
from .errors import VersionConflictError

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS spec_versions (
    spec_version INTEGER PRIMARY KEY,
    digest TEXT NOT NULL UNIQUE,
    label TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('ACTIVE', 'STALE', 'READ_ONLY')),
    created_at_minute INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    spec_version INTEGER NOT NULL,
    horizon INTEGER NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('RUNNING', 'COMPLETE', 'UNKNOWN')),
    created_order INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS minute_progress (
    run_id TEXT NOT NULL,
    minute INTEGER NOT NULL,
    spec_version INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('COMMITTED', 'UNKNOWN')),
    decision_json TEXT NOT NULL,
    PRIMARY KEY (run_id, minute)
);
CREATE TABLE IF NOT EXISTS minute_services (
    run_id TEXT NOT NULL,
    minute INTEGER NOT NULL,
    service_name TEXT NOT NULL,
    spec_version INTEGER NOT NULL,
    allocated_replicas INTEGER NOT NULL,
    servable_requests INTEGER NOT NULL,
    rejected_requests INTEGER NOT NULL,
    trim_reasons TEXT NOT NULL,
    estimate_json TEXT NOT NULL,
    PRIMARY KEY (run_id, minute, service_name)
);
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    spec_version INTEGER NOT NULL,
    target_minute INTEGER NOT NULL,
    created_order INTEGER NOT NULL,
    note TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class MinuteRecord:
    run_id: str
    minute: int
    spec_version: int
    status: str
    decision: dict[str, Any]


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        with self.conn:
            self.conn.executescript(_SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ meta
    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return default if row is None else row["value"]

    def set_meta(self, key: str, value: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    # ----------------------------------------------------------- spec versions
    def latest_spec_version(self) -> int | None:
        row = self.conn.execute(
            "SELECT MAX(spec_version) AS v FROM spec_versions"
        ).fetchone()
        return None if row["v"] is None else int(row["v"])

    def get_spec(self, spec_version: int) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT spec_json FROM spec_versions WHERE spec_version = ?",
            (spec_version,),
        ).fetchone()
        return None if row is None else json.loads(row["spec_json"])

    def get_spec_status(self, spec_version: int) -> str | None:
        row = self.conn.execute(
            "SELECT status FROM spec_versions WHERE spec_version = ?",
            (spec_version,),
        ).fetchone()
        return None if row is None else row["status"]

    def digest_version(self, digest: str) -> int | None:
        row = self.conn.execute(
            "SELECT spec_version FROM spec_versions WHERE digest = ?", (digest,)
        ).fetchone()
        return None if row is None else int(row["spec_version"])

    def register_spec(self, spec: Any, created_at_minute: int) -> tuple[int, bool]:
        """Register a spec; reuse the version if its digest already exists."""
        existing = self.digest_version(spec.digest)
        if existing is not None:
            return existing, False
        previous = self.latest_spec_version()
        new_version = 1 if previous is None else previous + 1
        if previous is not None:
            with self.conn:
                self.conn.execute(
                    "UPDATE spec_versions SET status = 'STALE' WHERE spec_version = ?",
                    (previous,),
                )
        with self.conn:
            self.conn.execute(
                "INSERT INTO spec_versions(spec_version, digest, label, spec_json, "
                "status, created_at_minute) VALUES(?, ?, ?, ?, 'ACTIVE', ?)",
                (
                    new_version,
                    spec.digest,
                    spec.label,
                    canonical_dumps(_spec_payload(spec)),
                    created_at_minute,
                ),
            )
        return new_version, True

    def mark_spec_read_only(self, spec_version: int) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE spec_versions SET status = 'READ_ONLY' WHERE spec_version = ? "
                "AND status = 'STALE'",
                (spec_version,),
            )

    def list_spec_versions(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT spec_version, digest, label, status, created_at_minute "
            "FROM spec_versions ORDER BY spec_version"
        ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------- runs
    def create_run(self, run_id: str, spec_version: int, horizon: int) -> None:
        order = int(self.get_meta("run_counter", "0")) + 1
        with self.conn:
            self.conn.execute(
                "INSERT INTO runs(run_id, spec_version, horizon, state, created_order) "
                "VALUES(?, ?, ?, 'RUNNING', ?)",
                (run_id, spec_version, horizon, order),
            )
            self.set_meta("run_counter", str(order))

    def get_run(self, run_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()

    def list_runs(self) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.conn.execute(
                "SELECT * FROM runs ORDER BY created_order"
            ).fetchall()
        ]

    def latest_run_id(self) -> str | None:
        row = self.conn.execute(
            "SELECT run_id FROM runs ORDER BY created_order DESC LIMIT 1"
        ).fetchone()
        return None if row is None else row["run_id"]

    def committed_minutes(self, run_id: str) -> list[int]:
        rows = self.conn.execute(
            "SELECT minute FROM minute_progress WHERE run_id = ? AND status = 'COMMITTED' "
            "ORDER BY minute",
            (run_id,),
        ).fetchall()
        return [int(row["minute"]) for row in rows]

    def highest_committed_minute(self, run_id: str) -> int | None:
        row = self.conn.execute(
            "SELECT MAX(minute) AS m FROM minute_progress "
            "WHERE run_id = ? AND status = 'COMMITTED'",
            (run_id,),
        ).fetchone()
        return None if row["m"] is None else int(row["m"])

    # ------------------------------------------------------------------ writes
    def commit_minute(self, run_id: str, decision: Any) -> None:
        """Idempotent single-transaction write of one fully decided minute."""
        minute = int(decision.minute)
        spec_version = int(decision.spec_version)
        decision_json = canonical_dumps(decision.to_dict())
        with self.conn:  # one transaction for the whole minute
            self.conn.execute(
                "INSERT INTO minute_progress(run_id, minute, spec_version, status, decision_json) "
                "VALUES(?, ?, ?, 'COMMITTED', ?) "
                "ON CONFLICT(run_id, minute) DO UPDATE SET "
                "spec_version = excluded.spec_version, "
                "status = excluded.status, "
                "decision_json = excluded.decision_json",
                (run_id, minute, spec_version, decision_json),
            )
            for alloc in decision.allocations:
                self.conn.execute(
                    "INSERT INTO minute_services(run_id, minute, service_name, spec_version, "
                    "allocated_replicas, servable_requests, rejected_requests, "
                    "trim_reasons, estimate_json) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(run_id, minute, service_name) DO UPDATE SET "
                    "spec_version = excluded.spec_version, "
                    "allocated_replicas = excluded.allocated_replicas, "
                    "servable_requests = excluded.servable_requests, "
                    "rejected_requests = excluded.rejected_requests, "
                    "trim_reasons = excluded.trim_reasons, "
                    "estimate_json = excluded.estimate_json",
                    (
                        run_id,
                        minute,
                        alloc.service,
                        spec_version,
                        alloc.allocated_replicas,
                        alloc.servable_requests,
                        alloc.rejected_requests,
                        canonical_dumps(list(alloc.trim_reasons)),
                        canonical_dumps(alloc.estimate.to_dict()),
                    ),
                )
            self.conn.execute(
                "UPDATE runs SET state = CASE WHEN ? >= horizon - 1 THEN 'COMPLETE' "
                "ELSE state END WHERE run_id = ?",
                (minute, run_id),
            )
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES('cursor:' || ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (run_id, str(minute)),
            )

    def mark_run_unknown(self, run_id: str, up_to_minute: int) -> None:
        """Mark an unfinished run UNKNOWN; its next minute must be re-run."""
        with self.conn:
            self.conn.execute(
                "UPDATE runs SET state = 'UNKNOWN' WHERE run_id = ? AND state = 'RUNNING'",
                (run_id,),
            )
            self.conn.execute(
                "INSERT INTO minute_progress(run_id, minute, spec_version, status, decision_json) "
                "SELECT run_id, ?, spec_version, 'UNKNOWN', '' FROM runs WHERE run_id = ? "
                "ON CONFLICT(run_id, minute) DO UPDATE SET status = 'UNKNOWN'",
                (up_to_minute, run_id),
            )

    # ------------------------------------------------------------------ reads
    def get_minute(self, run_id: str, minute: int) -> MinuteRecord | None:
        row = self.conn.execute(
            "SELECT * FROM minute_progress WHERE run_id = ? AND minute = ?",
            (run_id, minute),
        ).fetchone()
        if row is None:
            return None
        decision = json.loads(row["decision_json"]) if row["decision_json"] else {}
        return MinuteRecord(
            run_id=run_id,
            minute=minute,
            spec_version=int(row["spec_version"]),
            status=row["status"],
            decision=decision,
        )

    def get_service_row(self, run_id: str, minute: int, service: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM minute_services WHERE run_id = ? AND minute = ? AND service_name = ?",
            (run_id, minute, service),
        ).fetchone()

    def iter_committed(self, run_id: str) -> Iterable[MinuteRecord]:
        for minute in self.committed_minutes(run_id):
            record = self.get_minute(run_id, minute)
            if record is not None:
                yield record

    # -------------------------------------------------------------- snapshots
    def save_snapshot(self, snapshot_id: str, run_id: str, spec_version: int,
                      target_minute: int, note: str = "") -> None:
        order = int(self.get_meta("snapshot_counter", "0")) + 1
        with self.conn:
            self.conn.execute(
                "INSERT INTO snapshots(snapshot_id, run_id, spec_version, target_minute, "
                "created_order, note) VALUES(?, ?, ?, ?, ?, ?)",
                (snapshot_id, run_id, spec_version, target_minute, order, note),
            )
            self.set_meta("snapshot_counter", str(order))

    def get_snapshot(self, snapshot_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM snapshots WHERE snapshot_id = ?", (snapshot_id,)
        ).fetchone()

    def list_snapshots(self) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.conn.execute(
                "SELECT * FROM snapshots ORDER BY created_order"
            ).fetchall()
        ]

    def assert_snapshot_version(self, snapshot_id: str, expected_version: int) -> None:
        row = self.get_snapshot(snapshot_id)
        if row is None:
            raise VersionConflictError(
                f"snapshot {snapshot_id!r} does not exist",
                fields=[f"/snapshots/{snapshot_id}"],
                context={"snapshot_id": snapshot_id},
            )
        if int(row["spec_version"]) != int(expected_version):
            raise VersionConflictError(
                "snapshot references a different declaration version than the current spec",
                fields=[f"/snapshots/{snapshot_id}/spec_version"],
                context={
                    "snapshot_id": snapshot_id,
                    "snapshot_spec_version": int(row["spec_version"]),
                    "current_spec_version": int(expected_version),
                },
            )


def _spec_payload(spec: Any) -> dict[str, Any]:
    return {
        "global_replica_budget": spec.global_replica_budget,
        "services": [
            {
                "name": svc.name,
                "requests": list(svc.requests),
                "concurrency_limit": svc.concurrency_limit,
                "request_ms": svc.request_ms,
                "scale_threshold_pct": svc.scale_threshold_pct,
                "min_replicas": svc.min_replicas,
                "max_replicas": svc.max_replicas,
                "concurrency_quota": svc.concurrency_quota,
            }
            for svc in spec.services
        ],
        "version_label": spec.label,
    }
