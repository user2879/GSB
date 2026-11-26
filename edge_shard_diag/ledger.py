from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import LEDGER_FORMAT_VERSION, SNAPSHOT_FORMAT_VERSION
from .errors import Codes, DiagnosticError


def monotonic_ms() -> int:
    """Real local monotonic clock in truncated integer milliseconds."""

    import time

    return time.monotonic_ns() // 1_000_000


def wall_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Ledger:
    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        with self.lock:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_wall TEXT NOT NULL,
                    format_version INTEGER NOT NULL,
                    spec_format_version INTEGER NOT NULL,
                    node_set_version INTEGER NOT NULL,
                    content_digest TEXT NOT NULL,
                    capacity_version INTEGER NOT NULL,
                    fixed_revision INTEGER NOT NULL,
                    scale_factor INTEGER NOT NULL,
                    replaced_from INTEGER,
                    replace_reason TEXT,
                    cursor_json TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    monotonic_ms INTEGER NOT NULL,
                    wall_iso TEXT NOT NULL,
                    capacity_version INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    shard_id TEXT,
                    node_id TEXT,
                    lease_id TEXT,
                    basis TEXT NOT NULL,
                    detail_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS shard_outcomes (
                    shard_id TEXT NOT NULL,
                    capacity_version INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    stale INTEGER NOT NULL DEFAULT 0,
                    unknown INTEGER NOT NULL DEFAULT 0,
                    rerun_required INTEGER NOT NULL DEFAULT 0,
                    node_id TEXT,
                    lease_id TEXT,
                    lease_version INTEGER,
                    resource_units INTEGER,
                    summary TEXT,
                    summary_prefix TEXT,
                    sample_snapshot TEXT,
                    duration_ms REAL,
                    water_json TEXT,
                    error_json TEXT,
                    semantic_json TEXT,
                    updated_monotonic_ms INTEGER NOT NULL,
                    PRIMARY KEY (shard_id, capacity_version)
                );
                CREATE TABLE IF NOT EXISTS leases (
                    lease_id TEXT PRIMARY KEY,
                    shard_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    capacity_version INTEGER NOT NULL,
                    granted_ms INTEGER NOT NULL,
                    expires_ms INTEGER NOT NULL,
                    ttl_ms INTEGER NOT NULL,
                    renewed_ms INTEGER,
                    renew_count INTEGER NOT NULL DEFAULT 0,
                    lease_version INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    basis TEXT NOT NULL,
                    event_seq INTEGER
                );
                CREATE TABLE IF NOT EXISTS allocations (
                    allocation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lease_id TEXT NOT NULL,
                    shard_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    capacity_version INTEGER NOT NULL,
                    resource_units INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    basis TEXT NOT NULL,
                    granted_ms INTEGER NOT NULL,
                    released_ms INTEGER,
                    release_reason TEXT,
                    event_seq INTEGER
                );
                CREATE UNIQUE INDEX IF NOT EXISTS uq_active_allocation_shard
                    ON allocations(shard_id) WHERE status='HELD';
                CREATE TABLE IF NOT EXISTS resource_owners (
                    capacity_version INTEGER NOT NULL,
                    unit_index INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    shard_id TEXT,
                    lease_id TEXT,
                    event_seq INTEGER,
                    updated_ms INTEGER NOT NULL,
                    PRIMARY KEY (capacity_version, unit_index)
                );
                CREATE TABLE IF NOT EXISTS lease_slots (
                    capacity_version INTEGER NOT NULL,
                    node_id TEXT NOT NULL,
                    slot_index INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    lease_id TEXT,
                    shard_id TEXT,
                    event_seq INTEGER,
                    updated_ms INTEGER NOT NULL,
                    PRIMARY KEY (capacity_version, node_id, slot_index)
                );
                CREATE TABLE IF NOT EXISTS cursors (
                    cursor_key TEXT PRIMARY KEY,
                    capacity_version INTEGER NOT NULL,
                    topology_json TEXT NOT NULL,
                    next_index INTEGER NOT NULL,
                    completed_count INTEGER NOT NULL,
                    last_shard_id TEXT,
                    cursor_json TEXT NOT NULL,
                    updated_ms INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS receipt_conclusions (
                    receipt_id TEXT PRIMARY KEY,
                    shard_id TEXT,
                    capacity_version INTEGER NOT NULL,
                    format_version TEXT NOT NULL,
                    adopted INTEGER NOT NULL,
                    stale INTEGER NOT NULL DEFAULT 0,
                    payload_json TEXT NOT NULL,
                    updated_ms INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS diagnostic_reports (
                    report_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    created_ms INTEGER NOT NULL
                );
                """
            )
            self.conn.execute(
                "INSERT OR IGNORE INTO meta(key,value) VALUES('ledger_format_version', ?)",
                (str(LEDGER_FORMAT_VERSION),),
            )

    def close(self) -> None:
        with self.lock:
            self.conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _event_locked(
        self,
        cur: sqlite3.Cursor,
        *,
        capacity_version: int,
        kind: str,
        basis: str,
        shard_id: str | None = None,
        node_id: str | None = None,
        lease_id: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> int:
        now = monotonic_ms()
        cur.execute(
            """INSERT INTO events(monotonic_ms,wall_iso,capacity_version,kind,shard_id,node_id,lease_id,basis,detail_json)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (now, wall_iso(), capacity_version, kind, shard_id, node_id, lease_id, basis, json.dumps(detail or {}, sort_keys=True)),
        )
        return int(cur.lastrowid)

    def record_event(self, **kwargs: Any) -> int:
        with self.lock:
            cur = self.conn.cursor()
            self.conn.execute("BEGIN IMMEDIATE")
            seq = self._event_locked(cur, **kwargs)
            self.conn.commit()
            return seq

    def save_report(self, kind: str, report: dict[str, Any]) -> int:
        with self.lock:
            cur = self.conn.execute(
                "INSERT INTO diagnostic_reports(kind,content_json,created_ms) VALUES(?,?,?)",
                (kind, json.dumps(report, ensure_ascii=False, sort_keys=True), monotonic_ms()),
            )
            return int(cur.lastrowid)

    def create_snapshot(
        self,
        *,
        spec_format_version: int,
        node_set_version: int,
        content_digest: str,
        capacity_version: int,
        fixed_revision: int,
        scale_factor: int,
        cursor: dict[str, Any],
        payload: dict[str, Any],
        replaced_from: int | None = None,
        replace_reason: str | None = None,
    ) -> dict[str, Any]:
        with self.lock:
            cur = self.conn.execute(
                """INSERT INTO snapshots(created_wall,format_version,spec_format_version,node_set_version,content_digest,
                   capacity_version,fixed_revision,scale_factor,replaced_from,replace_reason,cursor_json,payload_json)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    wall_iso(),
                    SNAPSHOT_FORMAT_VERSION,
                    spec_format_version,
                    node_set_version,
                    content_digest,
                    capacity_version,
                    fixed_revision,
                    scale_factor,
                    replaced_from,
                    replace_reason,
                    json.dumps(cursor, sort_keys=True),
                    json.dumps(payload, ensure_ascii=False, sort_keys=True),
                ),
            )
            snapshot_id = int(cur.lastrowid)
            self.conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('current_snapshot_id',?)", (str(snapshot_id),))
            self.conn.commit()
            return self.get_snapshot(snapshot_id)

    def get_snapshot(self, snapshot_id: int) -> dict[str, Any]:
        row = self.conn.execute("SELECT * FROM snapshots WHERE snapshot_id=?", (snapshot_id,)).fetchone()
        if row is None:
            raise DiagnosticError(Codes.REPLAY, f"snapshot {snapshot_id} not found")
        return self._snapshot_dict(row)

    def latest_snapshot(self) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM snapshots ORDER BY snapshot_id DESC LIMIT 1").fetchone()
        return self._snapshot_dict(row) if row else None

    def list_snapshots(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT snapshot_id,created_wall,format_version,node_set_version,capacity_version,fixed_revision,scale_factor,replaced_from,replace_reason,content_digest FROM snapshots ORDER BY snapshot_id").fetchall()
        return [dict(row) for row in rows]

    def _snapshot_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        value["cursor"] = json.loads(value.pop("cursor_json"))
        value["payload"] = json.loads(value.pop("payload_json"))
        return value

    def open_or_replace_snapshot(self, bundle: Any, scale_factor: int, cursor: dict[str, Any], payload: dict[str, Any]) -> tuple[dict[str, Any], bool, list[str]]:
        latest = self.latest_snapshot()
        reasons: list[str] = []
        if latest is not None:
            if latest["format_version"] != SNAPSHOT_FORMAT_VERSION:
                reasons.append(f"format_version {latest['format_version']} != {SNAPSHOT_FORMAT_VERSION}")
            if latest["node_set_version"] != bundle.node_set_version:
                reasons.append(f"node_set_version {latest['node_set_version']} != {bundle.node_set_version}")
            if latest["content_digest"] != bundle.content_digest:
                reasons.append("content_digest mismatch")
            if latest["spec_format_version"] != bundle.spec_format_version:
                reasons.append("spec_format_version mismatch")
        if latest is None or reasons:
            snapshot = self.create_snapshot(
                spec_format_version=bundle.spec_format_version,
                node_set_version=bundle.node_set_version,
                content_digest=bundle.content_digest,
                capacity_version=bundle.active_capacity_version,
                fixed_revision=bundle.active_fixed_revision,
                scale_factor=scale_factor,
                cursor=cursor,
                payload=payload,
                replaced_from=latest["snapshot_id"] if latest else None,
                replace_reason="; ".join(reasons) if reasons else "no prior snapshot",
            )
            return snapshot, True, reasons or ["no prior snapshot"]
        return latest, False, []

    def initialize_capacity_slots(self, capacity: Any) -> None:
        now = monotonic_ms()
        with self.lock:
            self.conn.execute("BEGIN IMMEDIATE")
            for unit_index in range(capacity.budget_units):
                self.conn.execute(
                    """INSERT OR IGNORE INTO resource_owners(capacity_version,unit_index,status,updated_ms)
                       VALUES(?,?, 'FREE', ?)""",
                    (capacity.version, unit_index, now),
                )
            for node_id, limit in capacity.node_limits.items():
                for slot_index in range(limit.lease_limit):
                    self.conn.execute(
                        """INSERT OR IGNORE INTO lease_slots(capacity_version,node_id,slot_index,status,updated_ms)
                           VALUES(?,?,?, 'FREE', ?)""",
                        (capacity.version, node_id, slot_index, now),
                    )
            self.conn.commit()

    def active_usage(self, capacity_version: int, node_id: str | None = None) -> dict[str, int]:
        params: list[Any] = [capacity_version]
        where = "capacity_version=? AND status='HELD'"
        if node_id is not None:
            where += " AND node_id=?"
            params.append(node_id)
        row = self.conn.execute(f"SELECT COALESCE(SUM(resource_units),0) units, COUNT(*) leases FROM allocations WHERE {where}", params).fetchone()
        shard_count = self.conn.execute(
            f"SELECT COUNT(*) count FROM shard_outcomes WHERE capacity_version=? AND status IN ('LEASED','RUNNING','DISPATCHED')" + (" AND node_id=?" if node_id else ""),
            params if node_id is None else [capacity_version, node_id],
        ).fetchone()["count"]
        return {"resource_units": int(row["units"]), "active_leases": int(row["leases"]), "active_shards": int(shard_count)}

    def grant_lease(
        self,
        *,
        lease_id: str,
        shard_id: str,
        node_id: str,
        capacity_version: int,
        resource_units: int,
        ttl_ms: int,
        budget_units: int,
        concurrency_quota: int,
        lease_limit: int,
    ) -> dict[str, Any]:
        now = monotonic_ms()
        with self.lock:
            cur = self.conn.cursor()
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                global_usage = self.active_usage(capacity_version)
                node_usage = self.active_usage(capacity_version, node_id)
                if global_usage["resource_units"] + resource_units > budget_units:
                    raise DiagnosticError(Codes.BUDGET, "global resource budget would be exceeded", field="resource_units", details={"global": global_usage, "node": node_usage})
                if node_usage["active_leases"] + 1 > lease_limit:
                    raise DiagnosticError(Codes.BUDGET, "node lease limit would be exceeded", field="lease_limit", details=node_usage)
                if node_usage["active_shards"] + 1 > concurrency_quota:
                    raise DiagnosticError(Codes.BUDGET, "node concurrency quota would be exceeded", field="concurrency_quota", details=node_usage)
                unit_rows = cur.execute(
                    "SELECT unit_index FROM resource_owners WHERE capacity_version=? AND status='FREE' ORDER BY unit_index LIMIT ?",
                    (capacity_version, resource_units),
                ).fetchall()
                if len(unit_rows) != resource_units:
                    raise DiagnosticError(Codes.BUDGET, "not enough free resource-unit ownership slots", field="resource_units")
                unit_indices = [int(row["unit_index"]) for row in unit_rows]
                for index in unit_indices:
                    changed = cur.execute(
                        "UPDATE resource_owners SET status='HELD',shard_id=?,lease_id=?,event_seq=?,updated_ms=? WHERE capacity_version=? AND unit_index=? AND status='FREE'",
                        (shard_id, lease_id, None, now, capacity_version, index),
                    ).rowcount
                    if changed != 1:
                        raise DiagnosticError(Codes.BUDGET, f"resource unit {index} was concurrently owned", field="resource_units")
                slot_row = cur.execute(
                    "SELECT slot_index FROM lease_slots WHERE capacity_version=? AND node_id=? AND status='FREE' ORDER BY slot_index LIMIT 1",
                    (capacity_version, node_id),
                ).fetchone()
                if slot_row is None:
                    raise DiagnosticError(Codes.BUDGET, "no free lease slot", field="lease_limit")
                slot_index = int(slot_row["slot_index"])
                changed = cur.execute(
                    "UPDATE lease_slots SET status='HELD',lease_id=?,shard_id=?,event_seq=?,updated_ms=? WHERE capacity_version=? AND node_id=? AND slot_index=? AND status='FREE'",
                    (lease_id, shard_id, None, now, capacity_version, node_id, slot_index),
                ).rowcount
                if changed != 1:
                    raise DiagnosticError(Codes.BUDGET, "lease slot was concurrently owned", field="lease_limit")
                event_seq = self._event_locked(
                    cur,
                    capacity_version=capacity_version,
                    kind="LEASE_GRANTED",
                    basis="budget+concurrency+lease_limit checks passed under BEGIN IMMEDIATE",
                    shard_id=shard_id,
                    node_id=node_id,
                    lease_id=lease_id,
                    detail={"unit_indices": unit_indices, "slot_index": slot_index, "lease_version": 1, "ttl_ms": ttl_ms},
                )
                cur.execute(
                    """INSERT INTO leases(lease_id,shard_id,node_id,capacity_version,granted_ms,expires_ms,ttl_ms,renewed_ms,
                       renew_count,lease_version,status,basis,event_seq) VALUES(?,?,?,?,?,?,?,NULL,0,1,'ACTIVE','initial grant',?)
                       ON CONFLICT(lease_id) DO UPDATE SET shard_id=excluded.shard_id,node_id=excluded.node_id,
                         capacity_version=excluded.capacity_version,granted_ms=excluded.granted_ms,
                         expires_ms=excluded.expires_ms,ttl_ms=excluded.ttl_ms,renewed_ms=NULL,renew_count=0,
                         lease_version=1,status='ACTIVE',basis=excluded.basis,event_seq=excluded.event_seq""",
                    (lease_id, shard_id, node_id, capacity_version, now, now + ttl_ms, ttl_ms, event_seq),
                )
                cur.execute(
                    """INSERT INTO allocations(lease_id,shard_id,node_id,capacity_version,resource_units,status,basis,granted_ms,event_seq)
                       VALUES(?,?,?,?,?,'HELD','scheduler grant',?,?)""",
                    (lease_id, shard_id, node_id, capacity_version, resource_units, now, event_seq),
                )
                cur.execute(
                    """INSERT INTO shard_outcomes(shard_id,capacity_version,status,stale,unknown,rerun_required,node_id,lease_id,
                       lease_version,resource_units,updated_monotonic_ms)
                       VALUES(?,?,'LEASED',0,0,0,?,?,1,?,?)
                       ON CONFLICT(shard_id,capacity_version) DO UPDATE SET status='LEASED',unknown=0,rerun_required=0,
                         node_id=excluded.node_id,lease_id=excluded.lease_id,lease_version=1,resource_units=excluded.resource_units,
                         updated_monotonic_ms=excluded.updated_monotonic_ms""",
                    (shard_id, capacity_version, node_id, lease_id, resource_units, now),
                )
                cur.execute("UPDATE resource_owners SET event_seq=? WHERE lease_id=?", (event_seq, lease_id))
                cur.execute("UPDATE lease_slots SET event_seq=? WHERE lease_id=?", (event_seq, lease_id))
                self.conn.commit()
                return {
                    "lease_id": lease_id,
                    "shard_id": shard_id,
                    "node_id": node_id,
                    "capacity_version": capacity_version,
                    "granted_ms": now,
                    "expires_ms": now + ttl_ms,
                    "ttl_ms": ttl_ms,
                    "renew_count": 0,
                    "lease_version": 1,
                    "unit_indices": unit_indices,
                    "slot_index": slot_index,
                    "event_seq": event_seq,
                    "status": "ACTIVE",
                }
            except Exception:
                self.conn.rollback()
                raise

    def get_lease(self, lease_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM leases WHERE lease_id=?", (lease_id,)).fetchone()
        return dict(row) if row else None

    def renew_lease(self, lease_id: str, expected_version: int, *, ttl_ms: int | None = None) -> dict[str, Any]:
        now = monotonic_ms()
        with self.lock:
            cur = self.conn.cursor()
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                row = cur.execute("SELECT * FROM leases WHERE lease_id=?", (lease_id,)).fetchone()
                if row is None or row["status"] != "ACTIVE":
                    raise DiagnosticError(Codes.LEASE, f"lease is not active: {lease_id}", field="lease_version")
                if int(row["lease_version"]) != int(expected_version):
                    raise DiagnosticError(
                        Codes.LEASE,
                        "optimistic lease version mismatch; expiry was not extended",
                        field="lease_version",
                        details={"expected": expected_version, "actual": row["lease_version"]},
                    )
                new_version = int(row["lease_version"]) + 1
                new_ttl = int(ttl_ms if ttl_ms is not None else row["ttl_ms"])
                event_seq = self._event_locked(
                    cur,
                    capacity_version=int(row["capacity_version"]),
                    kind="LEASE_RENEWED",
                    basis="node optimistic version matched; monotonic deadline extended",
                    shard_id=row["shard_id"],
                    node_id=row["node_id"],
                    lease_id=lease_id,
                    detail={"old_version": expected_version, "new_version": new_version},
                )
                cur.execute(
                    """UPDATE leases SET lease_version=?,renew_count=renew_count+1,renewed_ms=?,expires_ms=?,event_seq=?
                       WHERE lease_id=? AND lease_version=? AND status='ACTIVE'""",
                    (new_version, now, now + new_ttl, event_seq, lease_id, expected_version),
                )
                cur.execute(
                    "UPDATE shard_outcomes SET lease_version=?,updated_monotonic_ms=? WHERE lease_id=? AND capacity_version=?",
                    (new_version, now, lease_id, row["capacity_version"]),
                )
                self.conn.commit()
                result = dict(cur.execute("SELECT * FROM leases WHERE lease_id=?", (lease_id,)).fetchone())
                return result
            except Exception:
                self.conn.rollback()
                raise

    def _release_lease_locked(
        self,
        cur: sqlite3.Cursor,
        lease_id: str,
        *,
        new_status: str,
        reason: str,
        kind: str,
        basis: str,
    ) -> dict[str, Any] | None:
        row = cur.execute("SELECT * FROM leases WHERE lease_id=?", (lease_id,)).fetchone()
        if row is None or row["status"] != "ACTIVE":
            return None
        now = monotonic_ms()
        changed = cur.execute("UPDATE leases SET status=?, basis=? WHERE lease_id=? AND status='ACTIVE'", (new_status, reason, lease_id)).rowcount
        if changed != 1:
            return None
        event_seq = self._event_locked(
            cur,
            capacity_version=int(row["capacity_version"]),
            kind=kind,
            basis=basis,
            shard_id=row["shard_id"],
            node_id=row["node_id"],
            lease_id=lease_id,
            detail={"reason": reason},
        )
        cur.execute(
            "UPDATE resource_owners SET status='FREE',shard_id=NULL,lease_id=NULL,event_seq=?,updated_ms=? WHERE lease_id=? AND status='HELD'",
            (event_seq, now, lease_id),
        )
        cur.execute(
            "UPDATE lease_slots SET status='FREE',shard_id=NULL,lease_id=NULL,event_seq=?,updated_ms=? WHERE lease_id=? AND status='HELD'",
            (event_seq, now, lease_id),
        )
        cur.execute(
            "UPDATE allocations SET status='RELEASED',released_ms=?,release_reason=?,event_seq=? WHERE lease_id=? AND status='HELD'",
            (now, reason, event_seq, lease_id),
        )
        return dict(row)

    def expire_due_leases(self) -> list[dict[str, Any]]:
        now = monotonic_ms()
        expired: list[dict[str, Any]] = []
        with self.lock:
            lease_ids = [
                row["lease_id"]
                for row in self.conn.execute("SELECT lease_id FROM leases WHERE status='ACTIVE' AND expires_ms <= ? ORDER BY lease_id", (now,)).fetchall()
            ]
            for lease_id in lease_ids:
                self.conn.execute("BEGIN IMMEDIATE")
                try:
                    row = self._release_lease_locked(
                        self.conn.cursor(),
                        lease_id,
                        new_status="EXPIRED",
                        reason="real monotonic deadline reached",
                        kind="LEASE_EXPIRED",
                        basis="expiry decided only by time.monotonic_ns, not an injected delay",
                    )
                    if row:
                        expired.append(row)
                    self.conn.commit()
                except Exception:
                    self.conn.rollback()
                    raise
        return expired

    def revoke_capacity_leases(self, new_capacity_version: int) -> dict[str, Any]:
        revoked: list[str] = []
        with self.lock:
            cur = self.conn.cursor()
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                rows = cur.execute(
                    "SELECT lease_id FROM leases WHERE status='ACTIVE' AND capacity_version <> ? ORDER BY lease_id",
                    (new_capacity_version,),
                ).fetchall()
                for row in rows:
                    released = self._release_lease_locked(
                        cur,
                        row["lease_id"],
                        new_status="REVOKED",
                        reason=f"capacity generation replaced by v{new_capacity_version}",
                        kind="CAPACITY_LEASES_REVOKED",
                        basis="capacity version upgrade performs whole-quota replacement and revokes old unexpired leases",
                    )
                    if released:
                        revoked.append(row["lease_id"])
                        cur.execute(
                            "UPDATE shard_outcomes SET status='FROZEN', stale=1, unknown=0, rerun_required=0, updated_monotonic_ms=? WHERE shard_id=? AND capacity_version=?",
                            (monotonic_ms(), released["shard_id"], released["capacity_version"]),
                        )
                cur.execute("UPDATE shard_outcomes SET stale=1 WHERE capacity_version < ?", (new_capacity_version,))
                cur.execute("UPDATE receipt_conclusions SET stale=1 WHERE capacity_version < ?", (new_capacity_version,))
                event_seq = self._event_locked(
                    cur,
                    capacity_version=new_capacity_version,
                    kind="CAPACITY_GENERATION_REPLACED",
                    basis="quotas, lease limits, budget and current view replaced wholesale",
                    detail={"revoked_lease_ids": revoked},
                )
                self.conn.commit()
                return {"revoked_lease_ids": revoked, "event_seq": event_seq}
            except Exception:
                self.conn.rollback()
                raise

    def begin_crash_partial_outcome(self, outcome: dict[str, Any]) -> None:
        """Open and leave uncommitted a partial completion write; caller SIGKILLs immediately."""

        now = monotonic_ms()
        self.lock.acquire()
        self.conn.execute("BEGIN IMMEDIATE")
        self.conn.execute(
            """INSERT INTO shard_outcomes(shard_id,capacity_version,status,summary,summary_prefix,sample_snapshot,
               node_id,lease_id,lease_version,resource_units,duration_ms,updated_monotonic_ms)
               VALUES(?,?,'OUTCOME_UNCOMMITTED',?,?,?,?,?,?,?,?,?)
               ON CONFLICT(shard_id,capacity_version) DO UPDATE SET status='OUTCOME_UNCOMMITTED',summary=excluded.summary""",
            (
                outcome["shard_id"],
                outcome["capacity_version"],
                outcome.get("summary"),
                outcome.get("summary_prefix"),
                outcome.get("sample_snapshot"),
                outcome.get("node_id"),
                outcome.get("lease_id"),
                outcome.get("lease_version"),
                outcome.get("resource_units"),
                outcome.get("duration_ms"),
                now,
            ),
        )
        # Intentionally no commit and no lock release before os.kill(SIGKILL).

    def commit_outcome(self, outcome: dict[str, Any], water: dict[str, Any], error: dict[str, Any], cursor: dict[str, Any]) -> int:
        now = monotonic_ms()
        with self.lock:
            cur = self.conn.cursor()
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                lease_id = outcome["lease_id"]
                lease = cur.execute("SELECT * FROM leases WHERE lease_id=?", (lease_id,)).fetchone()
                if lease is None or lease["status"] != "ACTIVE":
                    raise DiagnosticError(Codes.LEASE, f"cannot commit outcome for non-active lease {lease_id}", field="lease_id")
                event_seq = self._event_locked(
                    cur,
                    capacity_version=int(outcome["capacity_version"]),
                    kind="SHARD_OUTCOME_COMMITTED",
                    basis="summary, lease attribution, occupancy, water, error and cursor commit atomically",
                    shard_id=outcome["shard_id"],
                    node_id=outcome["node_id"],
                    lease_id=lease_id,
                    detail={"summary_prefix": outcome.get("summary_prefix")},
                )
                cur.execute(
                    """INSERT INTO shard_outcomes(shard_id,capacity_version,status,stale,unknown,rerun_required,node_id,lease_id,
                       lease_version,resource_units,summary,summary_prefix,sample_snapshot,duration_ms,water_json,error_json,
                       semantic_json,updated_monotonic_ms)
                       VALUES(?,?,'COMPLETED',0,0,0,?,?,?,?,?,?,?,?,?,?,?,?)
                       ON CONFLICT(shard_id,capacity_version) DO UPDATE SET status='COMPLETED',stale=0,unknown=0,
                         rerun_required=0,node_id=excluded.node_id,lease_id=excluded.lease_id,lease_version=excluded.lease_version,
                         resource_units=excluded.resource_units,summary=excluded.summary,summary_prefix=excluded.summary_prefix,
                         sample_snapshot=excluded.sample_snapshot,duration_ms=excluded.duration_ms,water_json=excluded.water_json,
                         error_json=excluded.error_json,semantic_json=excluded.semantic_json,
                         updated_monotonic_ms=excluded.updated_monotonic_ms""",
                    (
                        outcome["shard_id"],
                        outcome["capacity_version"],
                        outcome["node_id"],
                        lease_id,
                        int(outcome.get("lease_version", lease["lease_version"])),
                        int(outcome["resource_units"]),
                        outcome["summary"],
                        outcome["summary_prefix"],
                        outcome.get("sample_snapshot"),
                        float(outcome.get("duration_ms", 0.0)),
                        json.dumps(water, ensure_ascii=False, sort_keys=True),
                        json.dumps(error, ensure_ascii=False, sort_keys=True),
                        json.dumps(outcome.get("semantic", {}), ensure_ascii=False, sort_keys=True),
                        now,
                    ),
                )
                self._release_lease_locked(
                    cur,
                    lease_id,
                    new_status="COMPLETED",
                    reason="shard outcome committed",
                    kind="LEASE_COMPLETED",
                    basis="same atomic shard outcome transaction",
                )
                topology = cursor.get("topology", [])
                cur.execute(
                    """INSERT INTO cursors(cursor_key,capacity_version,topology_json,next_index,completed_count,last_shard_id,cursor_json,updated_ms)
                       VALUES('current',?,?,?,?,?,?,?)
                       ON CONFLICT(cursor_key) DO UPDATE SET capacity_version=excluded.capacity_version,
                         topology_json=excluded.topology_json,next_index=excluded.next_index,
                         completed_count=excluded.completed_count,last_shard_id=excluded.last_shard_id,
                         cursor_json=excluded.cursor_json,updated_ms=excluded.updated_ms""",
                    (
                        int(outcome["capacity_version"]),
                        json.dumps(topology),
                        int(cursor.get("next_index", 0)),
                        int(cursor.get("completed_count", 0)),
                        outcome["shard_id"],
                        json.dumps(cursor, sort_keys=True),
                        now,
                    ),
                )
                self.conn.commit()
                return event_seq
            except Exception:
                self.conn.rollback()
                raise

    def recover_orphans(self) -> dict[str, Any]:
        now = monotonic_ms()
        unknown: list[str] = []
        reclaimed_leases: list[str] = []
        with self.lock:
            cur = self.conn.cursor()
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                active_leases = cur.execute("SELECT lease_id, shard_id FROM leases WHERE status='ACTIVE'").fetchall()
                for row in active_leases:
                    released = self._release_lease_locked(
                        cur,
                        row["lease_id"],
                        new_status="ORPHAN_RECLAIMED",
                        reason="restart found committed grant without committed shard outcome",
                        kind="RECOVERY_RECLAIM",
                        basis="only fully committed ledger records survive SIGKILL; grant is committed, outcome transaction is not",
                    )
                    if released:
                        reclaimed_leases.append(row["lease_id"])
                        unknown.append(row["shard_id"])
                rows = cur.execute(
                    "SELECT shard_id, capacity_version FROM shard_outcomes WHERE status IN ('LEASED','RUNNING','DISPATCHED','OUTCOME_UNCOMMITTED','UNKNOWN') OR unknown=1"
                ).fetchall()
                for row in rows:
                    shard_id = row["shard_id"]
                    if shard_id not in unknown:
                        unknown.append(shard_id)
                    cur.execute(
                        "UPDATE shard_outcomes SET status='UNKNOWN',unknown=1,rerun_required=1,updated_monotonic_ms=? WHERE shard_id=? AND capacity_version=?",
                        (now, shard_id, row["capacity_version"]),
                    )
                seq = self._event_locked(
                    cur,
                    capacity_version=int(self.latest_snapshot()["capacity_version"]) if self.latest_snapshot() else 0,
                    kind="RECOVERY_COMPLETED",
                    basis="orphan committed grants reclaimed before rerun; uncommitted SQLite transaction is rolled back",
                    detail={"unknown_shards": unknown, "reclaimed_leases": reclaimed_leases},
                )
                self.conn.commit()
                return {"unknown_shards": sorted(set(unknown)), "reclaimed_leases": reclaimed_leases, "event_seq": seq}
            except Exception:
                self.conn.rollback()
                raise

    def mark_unknown_for_test(self, shard_id: str, capacity_version: int) -> None:
        with self.lock:
            self.conn.execute(
                "UPDATE shard_outcomes SET status='UNKNOWN',unknown=1,rerun_required=1 WHERE shard_id=? AND capacity_version=?",
                (shard_id, capacity_version),
            )
            self.conn.commit()

    def mark_rerun(self, shard_ids: set[str] | list[str], capacity_version: int) -> None:
        if not shard_ids:
            return
        now = monotonic_ms()
        with self.lock:
            self.conn.execute("BEGIN IMMEDIATE")
            self.conn.execute("DELETE FROM cursors WHERE cursor_key='current'")
            for shard_id in shard_ids:
                self.conn.execute(
                    """INSERT INTO shard_outcomes(shard_id,capacity_version,status,stale,unknown,rerun_required,updated_monotonic_ms)
                       VALUES(?,?,'PENDING',0,0,1,?)
                       ON CONFLICT(shard_id,capacity_version) DO UPDATE SET status='PENDING',stale=0,unknown=0,
                         rerun_required=1,updated_monotonic_ms=?""",
                    (shard_id, capacity_version, now, now),
                )
            self.conn.commit()

    def get_cursor(self) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM cursors WHERE cursor_key='current'").fetchone()
        if row is None:
            return None
        value = dict(row)
        value["cursor"] = json.loads(value.pop("cursor_json"))
        value["topology"] = json.loads(value.pop("topology_json"))
        return value

    def list_outcomes(self, capacity_version: int | None = None, *, include_stale: bool = False) -> list[dict[str, Any]]:
        sql = "SELECT * FROM shard_outcomes WHERE 1=1"
        params: list[Any] = []
        if capacity_version is not None:
            sql += " AND capacity_version=?"
            params.append(capacity_version)
        if not include_stale:
            sql += " AND stale=0"
        sql += " ORDER BY shard_id"
        rows = self.conn.execute(sql, params).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            for key in ("water_json", "error_json", "semantic_json"):
                if item.get(key):
                    item[key.removesuffix("_json")] = json.loads(item[key])
            result.append(item)
        return result

    def semantic_view(self, capacity_version: int) -> dict[str, dict[str, Any]]:
        view: dict[str, dict[str, Any]] = {}
        for row in self.list_outcomes(capacity_version):
            semantic = json.loads(row["semantic_json"]) if row.get("semantic_json") else {}
            view[row["shard_id"]] = {
                "shard_id": row["shard_id"],
                "status": row["status"],
                "capacity_version": row["capacity_version"],
                "node_id": row["node_id"],
                "lease_id": row["lease_id"],
                "lease_version": row["lease_version"],
                "resource_units": row["resource_units"],
                "summary": row["summary"],
                "summary_prefix": row["summary_prefix"],
                "sample_snapshot": row["sample_snapshot"],
                "water": json.loads(row["water_json"]) if row.get("water_json") else {},
                "error": json.loads(row["error_json"]) if row.get("error_json") else {},
                "semantic": semantic,
            }
        return view

    def save_receipts(self, receipts: list[dict[str, Any]], capacity_version: int | None = None) -> None:
        now = monotonic_ms()
        with self.lock:
            self.conn.execute("BEGIN IMMEDIATE")
            for receipt in receipts:
                cap = int(capacity_version if capacity_version is not None else receipt.get("capacity_version", 0))
                self.conn.execute(
                    """INSERT INTO receipt_conclusions(receipt_id,shard_id,capacity_version,format_version,adopted,stale,payload_json,updated_ms)
                       VALUES(?,?,?,?,?,0,?,?)
                       ON CONFLICT(receipt_id) DO UPDATE SET shard_id=excluded.shard_id,capacity_version=excluded.capacity_version,
                         format_version=excluded.format_version,adopted=excluded.adopted,payload_json=excluded.payload_json,
                         updated_ms=excluded.updated_ms""",
                    (
                        receipt["receipt_id"],
                        receipt.get("shard_id"),
                        cap,
                        receipt["format_version"],
                        1 if receipt.get("adopted") else 0,
                        json.dumps(receipt, ensure_ascii=False, sort_keys=True),
                        now,
                    ),
                )
            self.conn.commit()

    def list_events(self, limit: int = 200) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM events ORDER BY event_seq DESC LIMIT ?", (limit,)).fetchall()
        result = []
        for row in reversed(rows):
            item = dict(row)
            item["detail"] = json.loads(item.pop("detail_json"))
            result.append(item)
        return result
