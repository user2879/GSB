"""Versioned SQLite ledger with deterministic business-content digests."""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, Mapping

from .errors import BusinessError
from .util import canonical_json, digest_obj, is_int, monotonic_ms

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS results (
    shard_id TEXT PRIMARY KEY,
    outcome TEXT NOT NULL,
    summary_json TEXT NOT NULL,
    node_id TEXT NOT NULL,
    lease_id TEXT NOT NULL,
    lease_version INTEGER NOT NULL,
    config_version INTEGER NOT NULL,
    resource_units INTEGER NOT NULL,
    generation INTEGER NOT NULL,
    completion_event_seq INTEGER NOT NULL,
    elapsed_monotonic_ms INTEGER NOT NULL,
    wait_start_ms INTEGER,
    wait_end_ms INTEGER,
    completion_monotonic_ms INTEGER NOT NULL,
    result_digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS result_history (
    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
    shard_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    outcome TEXT NOT NULL,
    summary_json TEXT NOT NULL,
    node_id TEXT NOT NULL,
    lease_id TEXT NOT NULL,
    lease_version INTEGER NOT NULL,
    config_version INTEGER NOT NULL,
    resource_units INTEGER NOT NULL,
    completion_event_seq INTEGER NOT NULL,
    completion_monotonic_ms INTEGER NOT NULL,
    UNIQUE(shard_id, generation)
);
CREATE TABLE IF NOT EXISTS leases (
    lease_id TEXT PRIMARY KEY,
    shard_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    granted_ms INTEGER NOT NULL,
    expires_ms INTEGER NOT NULL,
    version INTEGER NOT NULL,
    state TEXT NOT NULL,
    config_version INTEGER NOT NULL,
    generation INTEGER NOT NULL,
    resource_units INTEGER NOT NULL,
    event_seq INTEGER NOT NULL,
    updated_monotonic_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS attempts (
    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
    shard_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    attempt_no INTEGER NOT NULL,
    status TEXT NOT NULL,
    lease_id TEXT,
    detail TEXT,
    started_monotonic_ms INTEGER NOT NULL,
    updated_monotonic_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS node_quota (
    node_id TEXT PRIMARY KEY,
    concurrency INTEGER NOT NULL,
    lease_limit INTEGER NOT NULL,
    capacity_version INTEGER NOT NULL,
    active_shards INTEGER NOT NULL DEFAULT 0,
    active_leases INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS budget (
    budget_id INTEGER PRIMARY KEY CHECK (budget_id = 1),
    global_budget INTEGER NOT NULL,
    used_units INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS invalidations (
    event_seq INTEGER NOT NULL,
    shard_id TEXT NOT NULL,
    direct INTEGER NOT NULL,
    reason TEXT NOT NULL,
    topo_rank INTEGER NOT NULL,
    PRIMARY KEY(event_seq, shard_id)
);
CREATE TABLE IF NOT EXISTS incremental_checks (
    event_seq INTEGER NOT NULL,
    shard_id TEXT NOT NULL,
    digest_before TEXT NOT NULL,
    digest_after TEXT NOT NULL,
    unchanged INTEGER NOT NULL,
    PRIMARY KEY(event_seq, shard_id)
);
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    after_event_seq INTEGER NOT NULL,
    created_monotonic_ms INTEGER NOT NULL,
    business_digest TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS exit_reports (
    report_id INTEGER PRIMARY KEY AUTOINCREMENT,
    reason TEXT NOT NULL,
    created_monotonic_ms INTEGER NOT NULL,
    clean_exit INTEGER NOT NULL,
    ledger_digest TEXT NOT NULL,
    report_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS recovery_events (
    recovery_id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_monotonic_ms INTEGER NOT NULL,
    unknown_count INTEGER NOT NULL,
    detail_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cursor (
    cursor_id INTEGER PRIMARY KEY CHECK (cursor_id = 1),
    last_event_seq INTEGER NOT NULL,
    completed_count INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS applied_events (
    event_seq INTEGER PRIMARY KEY,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    applied_monotonic_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS dispatch_requests (
    shard_id TEXT PRIMARY KEY,
    event_seq INTEGER NOT NULL,
    revision_version INTEGER NOT NULL,
    generation INTEGER NOT NULL,
    requested INTEGER NOT NULL,
    ttl_ms INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS cancelled_shards (
    shard_id TEXT PRIMARY KEY,
    event_seq INTEGER NOT NULL,
    reason TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_attempts_shard ON attempts(shard_id, attempt_id);
CREATE INDEX IF NOT EXISTS idx_history_shard ON result_history(shard_id, history_id);
"""

BUSINESS_RESULT_FIELDS = (
    "outcome",
    "summary_json",
    "node_id",
    "lease_id",
    "lease_version",
    "config_version",
    "resource_units",
    "generation",
    "completion_event_seq",
    "result_digest",
)


def _business_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row[field] for field in BUSINESS_RESULT_FIELDS}


class Ledger:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.db_path), timeout=30.0, isolation_level=None, check_same_thread=False)
        try:
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=FULL")
            self.conn.execute("PRAGMA foreign_keys=ON")
            self.conn.executescript(SCHEMA)
            self._initialize_meta()
            self.verify_content_digest()
        except Exception:
            try:
                self.conn.close()
            finally:
                raise

    # ------------------------------------------------------------------ core
    def _transaction(self):
        return _Transaction(self)

    def _initialize_meta(self) -> None:
        with self._lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                row = self.conn.execute("SELECT value FROM meta WHERE key='ledger_format_version'").fetchone()
                if row is None:
                    defaults = {
                        "ledger_format_version": str(SCHEMA_VERSION),
                        "input_digest": "",
                        "clean_shutdown": "1",
                        "active_revision": "1",
                    }
                    self.conn.executemany("INSERT INTO meta(key,value) VALUES(?,?)", defaults.items())
                    self.conn.execute("INSERT OR IGNORE INTO cursor(cursor_id,last_event_seq,completed_count) VALUES(1,0,0)")
                    self.conn.execute("INSERT OR IGNORE INTO budget(budget_id,global_budget,used_units) VALUES(1,1,0)")
                    digest = digest_obj(self._digest_payload(self.conn))
                    self.conn.execute(
                        "INSERT INTO meta(key,value) VALUES('content_digest',?) "
                        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        (digest,),
                    )
                else:
                    raw_version = row["value"]
                    try:
                        version = int(raw_version)
                    except (TypeError, ValueError) as exc:
                        raise BusinessError(
                            "E_SNAPSHOT",
                            "ledger format version is not an integer",
                            position=str(self.db_path),
                            field="ledger_format_version",
                        ) from exc
                    if version != SCHEMA_VERSION:
                        raise BusinessError(
                            "E_LEDGER",
                            f"unsupported ledger format version: {version}",
                            position=str(self.db_path),
                            field="ledger_format_version",
                            detail={"supported": SCHEMA_VERSION, "actual": version},
                        )
                    self.conn.execute("INSERT OR IGNORE INTO cursor(cursor_id,last_event_seq,completed_count) VALUES(1,0,0)")
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise

    def _meta(self, conn: sqlite3.Connection | None = None, key: str | None = None) -> Any:
        target = conn or self.conn
        if key is not None:
            row = target.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
            return None if row is None else row["value"]
        return {row["key"]: row["value"] for row in target.execute("SELECT key,value FROM meta ORDER BY key")}

    def _set_meta(self, conn: sqlite3.Connection, key: str, value: str) -> None:
        conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def _digest_payload(self, conn: sqlite3.Connection) -> dict[str, Any]:
        results = [
            {
                "shard_id": row["shard_id"],
                **_business_row(row),
            }
            for row in conn.execute("SELECT * FROM results ORDER BY shard_id")
        ]
        leases = [
            {
                "lease_id": row["lease_id"],
                "shard_id": row["shard_id"],
                "node_id": row["node_id"],
                "version": row["version"],
                "state": row["state"],
                "config_version": row["config_version"],
                "generation": row["generation"],
                "resource_units": row["resource_units"],
                "event_seq": row["event_seq"],
            }
            for row in conn.execute("SELECT * FROM leases ORDER BY lease_id")
        ]
        quotas = [
            {
                "node_id": row["node_id"],
                "concurrency": row["concurrency"],
                "lease_limit": row["lease_limit"],
                "capacity_version": row["capacity_version"],
            }
            for row in conn.execute("SELECT * FROM node_quota ORDER BY node_id")
        ]
        budget_row = conn.execute("SELECT global_budget FROM budget WHERE budget_id=1").fetchone()
        cursor_row = conn.execute("SELECT last_event_seq,completed_count FROM cursor WHERE cursor_id=1").fetchone()
        invalidations = [
            {
                "event_seq": row["event_seq"],
                "shard_id": row["shard_id"],
                "direct": row["direct"],
                "reason": row["reason"],
                "topo_rank": row["topo_rank"],
            }
            for row in conn.execute("SELECT * FROM invalidations ORDER BY event_seq,shard_id")
        ]
        checks = [
            {
                "event_seq": row["event_seq"],
                "shard_id": row["shard_id"],
                "digest_before": row["digest_before"],
                "digest_after": row["digest_after"],
                "unchanged": row["unchanged"],
            }
            for row in conn.execute("SELECT * FROM incremental_checks ORDER BY event_seq,shard_id")
        ]
        applied_events = [
            {
                "event_seq": row["event_seq"],
                "event_type": row["event_type"],
                "payload_json": row["payload_json"],
            }
            for row in conn.execute("SELECT * FROM applied_events ORDER BY event_seq")
        ]
        dispatch_requests = [
            {
                "shard_id": row["shard_id"],
                "event_seq": row["event_seq"],
                "revision_version": row["revision_version"],
                "generation": row["generation"],
                "requested": row["requested"],
                "ttl_ms": row["ttl_ms"],
            }
            for row in conn.execute("SELECT * FROM dispatch_requests ORDER BY shard_id")
        ]
        cancellations = [
            {
                "shard_id": row["shard_id"],
                "event_seq": row["event_seq"],
                "reason": row["reason"],
            }
            for row in conn.execute("SELECT * FROM cancelled_shards ORDER BY shard_id")
        ]
        meta = self._meta(conn)
        return {
            "ledger_format_version": int(meta.get("ledger_format_version", SCHEMA_VERSION)),
            "input_digest": meta.get("input_digest", ""),
            "active_revision": int(meta.get("active_revision", "1")),
            "results": results,
            "leases": leases,
            "node_quota": quotas,
            "global_budget": int(budget_row["global_budget"]) if budget_row else 1,
            "cursor": {
                "last_event_seq": int(cursor_row["last_event_seq"]) if cursor_row else 0,
                "completed_count": int(cursor_row["completed_count"]) if cursor_row else 0,
            },
            "invalidations": invalidations,
            "incremental_checks": checks,
            "applied_events": applied_events,
            "dispatch_requests": dispatch_requests,
            "cancellations": cancellations,
        }

    def _update_digest(self, conn: sqlite3.Connection) -> str:
        digest = digest_obj(self._digest_payload(conn))
        self._set_meta(conn, "content_digest", digest)
        return digest

    def _write_snapshot(self, conn: sqlite3.Connection, after_event_seq: int) -> str:
        payload = self._digest_payload(conn)
        digest = digest_obj(payload)
        conn.execute(
            "INSERT INTO snapshots(after_event_seq,created_monotonic_ms,business_digest,payload_json) VALUES(?,?,?,?)",
            (after_event_seq, monotonic_ms(), digest, canonical_json(payload)),
        )
        return digest

    # -------------------------------------------------------------- lifecycle
    def content_digest(self) -> str:
        with self._lock:
            value = self._meta(key="content_digest")
            if value is None:
                raise BusinessError("E_SNAPSHOT", "ledger has no content digest", position=str(self.db_path), field="content_digest")
            return str(value)

    def verify_content_digest(self) -> str:
        with self._lock:
            expected = self._meta(key="content_digest")
            actual = digest_obj(self._digest_payload(self.conn))
            if expected is None or expected != actual:
                raise BusinessError(
                    "E_SNAPSHOT",
                    "ledger content digest mismatch",
                    position=str(self.db_path),
                    field="content_digest",
                    detail={"expected": expected, "actual": actual},
                )
            return actual

    def begin_run(self, input_digest: str, nodes: Mapping[str, Mapping[str, int]], global_budget: int) -> dict[str, Any]:
        with self._lock, self._transaction() as conn:
            existing = self._meta(conn, "input_digest")
            if existing and existing != input_digest:
                raise BusinessError(
                    "E_LEDGER",
                    "work directory belongs to a different input sample",
                    position=str(self.db_path),
                    field="input_digest",
                    detail={"existing": existing, "incoming": input_digest},
                )
            fresh = not bool(existing)
            was_clean = self._meta(conn, "clean_shutdown") == "1"
            unknown_attempts = [
                dict(row)
                for row in conn.execute(
                    "SELECT attempt_id,shard_id,generation,attempt_no,lease_id,status FROM attempts "
                    "WHERE status='in_flight' ORDER BY shard_id,attempt_id"
                )
            ]
            now = monotonic_ms()
            conn.execute(
                "UPDATE attempts SET status='unknown', updated_monotonic_ms=?, detail=? WHERE status='in_flight'",
                (now, "process terminated before transaction commit"),
            )
            conn.execute(
                "UPDATE leases SET state='unknown', updated_monotonic_ms=? WHERE state='active'",
                (now,),
            )
            conn.execute("UPDATE node_quota SET active_shards=0, active_leases=0")
            conn.execute("UPDATE budget SET used_units=0 WHERE budget_id=1")
            for node_id in sorted(nodes):
                values = nodes[node_id]
                if fresh:
                    conn.execute(
                        "INSERT INTO node_quota(node_id,concurrency,lease_limit,capacity_version,active_shards,active_leases) "
                        "VALUES(?,?,?,?,0,0) ON CONFLICT(node_id) DO UPDATE SET "
                        "concurrency=excluded.concurrency, lease_limit=excluded.lease_limit, "
                        "capacity_version=excluded.capacity_version, active_shards=0, active_leases=0",
                        (node_id, int(values["concurrency"]), int(values["lease_limit"]), int(values["capacity_version"])),
                    )
                else:
                    conn.execute(
                        "INSERT INTO node_quota(node_id,concurrency,lease_limit,capacity_version,active_shards,active_leases) "
                        "VALUES(?,?,?,?,0,0) ON CONFLICT(node_id) DO UPDATE SET active_shards=0, active_leases=0",
                        (node_id, int(values["concurrency"]), int(values["lease_limit"]), int(values["capacity_version"])),
                    )
            if fresh:
                conn.execute(
                    "INSERT INTO budget(budget_id,global_budget,used_units) VALUES(1,?,0) "
                    "ON CONFLICT(budget_id) DO UPDATE SET global_budget=excluded.global_budget, used_units=0",
                    (int(global_budget),),
                )
            else:
                conn.execute("UPDATE budget SET used_units=0 WHERE budget_id=1")
            self._set_meta(conn, "input_digest", input_digest)
            self._set_meta(conn, "clean_shutdown", "0")
            if unknown_attempts:
                conn.execute(
                    "INSERT INTO recovery_events(started_monotonic_ms,unknown_count,detail_json) VALUES(?,?,?)",
                    (now, len(unknown_attempts), canonical_json(unknown_attempts)),
                )
            self._update_digest(conn)
            return {
                "fresh": fresh,
                "was_clean": was_clean,
                "recovery": (not fresh) and (not was_clean),
                "unknown_attempts": unknown_attempts,
                "unknown_shards": sorted({row["shard_id"] for row in unknown_attempts}),
                "recovered_monotonic_ms": now,
            }

    def finish_run(self, report: Mapping[str, Any] | None = None, *, clean: bool = True) -> None:
        with self._lock, self._transaction() as conn:
            self._set_meta(conn, "clean_shutdown", "1" if clean else "0")
            if report is not None:
                conn.execute(
                    "INSERT INTO exit_reports(reason,created_monotonic_ms,clean_exit,ledger_digest,report_json) "
                    "VALUES(?,?,?,?,?)",
                    (
                        str(report.get("reason", "normal")),
                        monotonic_ms(),
                        1 if report.get("clean_exit") else 0,
                        self.content_digest(),
                        canonical_json(report),
                    ),
                )

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    # ------------------------------------------------------------- scheduling
    def update_quota(
        self,
        *,
        event_seq: int,
        global_budget: int,
        nodes: Mapping[str, Mapping[str, int]],
    ) -> None:
        with self._lock, self._transaction() as conn:
            budget = conn.execute("SELECT global_budget,used_units FROM budget WHERE budget_id=1").fetchone()
            if budget is None:
                raise BusinessError("E_LEDGER", "missing global budget row", field="budget")
            if global_budget < budget["used_units"]:
                raise BusinessError(
                    "E_LEASE",
                    "global budget is below currently reserved units",
                    field="global_budget",
                    detail={"incoming": global_budget, "reserved": budget["used_units"]},
                )
            for node_id in sorted(nodes):
                values = nodes[node_id]
                row = conn.execute("SELECT * FROM node_quota WHERE node_id=?", (node_id,)).fetchone()
                if row is None:
                    raise BusinessError("E_LEDGER", f"missing node quota row: {node_id}", field=node_id)
                if int(values["concurrency"]) < row["active_shards"]:
                    raise BusinessError(
                        "E_LEASE",
                        "concurrency below active shard count",
                        field="concurrency",
                        detail={"node_id": node_id, "active": row["active_shards"]},
                    )
                if int(values["lease_limit"]) < row["active_leases"]:
                    raise BusinessError(
                        "E_LEASE",
                        "lease_limit below active lease count",
                        field="lease_limit",
                        detail={"node_id": node_id, "active": row["active_leases"]},
                    )
                conn.execute(
                    "UPDATE node_quota SET concurrency=?,lease_limit=?,capacity_version=? WHERE node_id=?",
                    (
                        int(values["concurrency"]),
                        int(values["lease_limit"]),
                        int(values["capacity_version"]),
                        node_id,
                    ),
                )
            conn.execute(
                "UPDATE budget SET global_budget=? WHERE budget_id=1",
                (int(global_budget),),
            )
            conn.execute(
                "UPDATE cursor SET last_event_seq=MAX(last_event_seq,?) WHERE cursor_id=1",
                (int(event_seq),),
            )
            self._update_digest(conn)

    def reserve_dispatch(
        self,
        *,
        event_seq: int,
        shard_id: str,
        generation: int,
        node_id: str,
        lease_id: str,
        granted_ms: int,
        expires_ms: int,
        config_version: int,
        units: int,
    ) -> None:
        with self._lock, self._transaction() as conn:
            node = conn.execute("SELECT * FROM node_quota WHERE node_id=?", (node_id,)).fetchone()
            if node is None:
                raise BusinessError("E_LEDGER", f"unknown node quota row: {node_id}", field=node_id, shard_id=shard_id)
            budget = conn.execute("SELECT * FROM budget WHERE budget_id=1").fetchone()
            if node["active_shards"] + 1 > node["concurrency"]:
                raise BusinessError(
                    "E_LEASE",
                    "node concurrency quota exhausted",
                    field="concurrency",
                    shard_id=shard_id,
                    detail={"node_id": node_id},
                )
            if node["active_leases"] + 1 > node["lease_limit"]:
                raise BusinessError(
                    "E_LEASE",
                    "node lease quota exhausted",
                    field="lease_limit",
                    shard_id=shard_id,
                    detail={"node_id": node_id},
                )
            if budget["used_units"] + units > budget["global_budget"]:
                raise BusinessError(
                    "E_LEASE",
                    "global resource budget exhausted",
                    field="global_budget",
                    shard_id=shard_id,
                    detail={"used": budget["used_units"], "requested": units},
                )
            row = conn.execute("SELECT * FROM leases WHERE lease_id=?", (lease_id,)).fetchone()
            if row is not None and row["state"] == "active":
                raise BusinessError("E_LEASE", "lease id already active", field="lease_id", shard_id=shard_id)
            active_same = conn.execute(
                "SELECT lease_id FROM leases WHERE shard_id=? AND state='active'",
                (shard_id,),
            ).fetchone()
            if active_same is not None:
                raise BusinessError("E_LEASE", "shard already has an active lease", field="lease_id", shard_id=shard_id)
            conn.execute(
                "INSERT INTO leases(lease_id,shard_id,node_id,granted_ms,expires_ms,version,state,config_version,generation,resource_units,event_seq,updated_monotonic_ms) "
                "VALUES(?,?,?,?,?,1,'active',?,?,?,?,?) ON CONFLICT(lease_id) DO UPDATE SET "
                "shard_id=excluded.shard_id,node_id=excluded.node_id,granted_ms=excluded.granted_ms,"
                "expires_ms=excluded.expires_ms,version=1,state='active',config_version=excluded.config_version,"
                "generation=excluded.generation,resource_units=excluded.resource_units,event_seq=excluded.event_seq,"
                "updated_monotonic_ms=excluded.updated_monotonic_ms",
                (lease_id, shard_id, node_id, granted_ms, expires_ms, config_version, generation, int(units), event_seq, monotonic_ms()),
            )
            conn.execute(
                "UPDATE node_quota SET active_shards=active_shards+1, active_leases=active_leases+1 WHERE node_id=?",
                (node_id,),
            )
            conn.execute("UPDATE budget SET used_units=used_units+? WHERE budget_id=1", (int(units),))
            attempt_no = conn.execute("SELECT COUNT(*)+1 FROM attempts WHERE shard_id=?", (shard_id,)).fetchone()[0]
            conn.execute(
                "INSERT INTO attempts(shard_id,generation,attempt_no,status,lease_id,detail,started_monotonic_ms,updated_monotonic_ms) "
                "VALUES(?,?,?,'in_flight',?,NULL,?,?)",
                (shard_id, generation, attempt_no, lease_id, monotonic_ms(), monotonic_ms()),
            )
            conn.execute("UPDATE cursor SET last_event_seq=MAX(last_event_seq,?) WHERE cursor_id=1", (int(event_seq),))
            self._update_digest(conn)

    def commit_result(
        self,
        *,
        shard_id: str,
        outcome: str,
        summary: Mapping[str, Any],
        node_id: str,
        lease_id: str,
        lease_version: int,
        config_version: int,
        resource_units: int,
        generation: int,
        completion_event_seq: int,
        elapsed_monotonic_ms: int,
        wait_start_ms: int | None,
        wait_end_ms: int | None,
    ) -> dict[str, Any]:
        with self._lock, self._transaction() as conn:
            lease = conn.execute("SELECT * FROM leases WHERE lease_id=?", (lease_id,)).fetchone()
            if lease is None or lease["state"] != "active":
                raise BusinessError(
                    "E_LEASE",
                    "cannot commit a result without an active lease",
                    field="state",
                    shard_id=shard_id,
                )
            if lease["shard_id"] != shard_id:
                raise BusinessError("E_LEASE", "lease shard mismatch", field="shard_id", shard_id=shard_id)
            if int(lease["version"]) != int(lease_version):
                raise BusinessError(
                    "E_LEASE",
                    "lease version mismatch at commit",
                    field="lease_version",
                    shard_id=shard_id,
                )
            summary_json = canonical_json(summary)
            business = {
                "outcome": outcome,
                "summary_json": summary_json,
                "node_id": node_id,
                "lease_id": lease_id,
                "lease_version": int(lease_version),
                "config_version": int(config_version),
                "resource_units": int(resource_units),
                "generation": int(generation),
                "completion_event_seq": int(completion_event_seq),
            }
            result_digest = digest_obj(business)
            completion_ms = monotonic_ms()
            conn.execute(
                "INSERT INTO results(shard_id,outcome,summary_json,node_id,lease_id,lease_version,config_version,resource_units,generation,completion_event_seq,elapsed_monotonic_ms,wait_start_ms,wait_end_ms,completion_monotonic_ms,result_digest) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(shard_id) DO UPDATE SET "
                "outcome=excluded.outcome,summary_json=excluded.summary_json,node_id=excluded.node_id,"
                "lease_id=excluded.lease_id,lease_version=excluded.lease_version,config_version=excluded.config_version,"
                "resource_units=excluded.resource_units,generation=excluded.generation,"
                "completion_event_seq=excluded.completion_event_seq,elapsed_monotonic_ms=excluded.elapsed_monotonic_ms,"
                "wait_start_ms=excluded.wait_start_ms,wait_end_ms=excluded.wait_end_ms,"
                "completion_monotonic_ms=excluded.completion_monotonic_ms,result_digest=excluded.result_digest",
                (
                    shard_id,
                    outcome,
                    summary_json,
                    node_id,
                    lease_id,
                    int(lease_version),
                    int(config_version),
                    int(resource_units),
                    int(generation),
                    int(completion_event_seq),
                    int(elapsed_monotonic_ms),
                    wait_start_ms,
                    wait_end_ms,
                    completion_ms,
                    result_digest,
                ),
            )
            conn.execute(
                "INSERT INTO result_history(shard_id,generation,outcome,summary_json,node_id,lease_id,lease_version,config_version,resource_units,completion_event_seq,completion_monotonic_ms) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(shard_id,generation) DO UPDATE SET "
                "outcome=excluded.outcome,summary_json=excluded.summary_json,node_id=excluded.node_id,"
                "lease_id=excluded.lease_id,lease_version=excluded.lease_version,config_version=excluded.config_version,"
                "resource_units=excluded.resource_units,completion_event_seq=excluded.completion_event_seq,"
                "completion_monotonic_ms=excluded.completion_monotonic_ms",
                (
                    shard_id,
                    int(generation),
                    outcome,
                    summary_json,
                    node_id,
                    lease_id,
                    int(lease_version),
                    int(config_version),
                    int(resource_units),
                    int(completion_event_seq),
                    completion_ms,
                ),
            )
            conn.execute(
                "UPDATE leases SET state='completed',updated_monotonic_ms=? WHERE lease_id=? AND state='active'",
                (completion_ms, lease_id),
            )
            conn.execute(
                "UPDATE node_quota SET active_shards=active_shards-1,active_leases=active_leases-1 WHERE node_id=?",
                (node_id,),
            )
            conn.execute("UPDATE budget SET used_units=used_units-? WHERE budget_id=1", (int(resource_units),))
            conn.execute(
                "UPDATE attempts SET status='committed',updated_monotonic_ms=? WHERE attempt_id=("
                "SELECT attempt_id FROM attempts WHERE shard_id=? ORDER BY attempt_id DESC LIMIT 1)",
                (completion_ms, shard_id),
            )
            conn.execute(
                "UPDATE cursor SET completed_count=(SELECT COUNT(*) FROM results),last_event_seq=MAX(last_event_seq,?) WHERE cursor_id=1",
                (int(completion_event_seq),),
            )
            digest = self._write_snapshot(conn, int(completion_event_seq))
            self._set_meta(conn, "content_digest", digest)
            return {
                "result_digest": result_digest,
                "ledger_digest": digest,
                "completion_monotonic_ms": completion_ms,
                "generation": int(generation),
            }

    def release_lease(self, *, lease_id: str, state: str, detail: str | None = None, event_seq: int | None = None) -> bool:
        if state not in {"cancelled", "expired", "failed"}:
            raise ValueError(f"invalid release state: {state}")
        with self._lock, self._transaction() as conn:
            row = conn.execute("SELECT * FROM leases WHERE lease_id=?", (lease_id,)).fetchone()
            if row is None or row["state"] != "active":
                if event_seq is not None:
                    conn.execute("UPDATE cursor SET last_event_seq=MAX(last_event_seq,?) WHERE cursor_id=1", (int(event_seq),))
                    self._update_digest(conn)
                return False
            now = monotonic_ms()
            conn.execute("UPDATE leases SET state=?,updated_monotonic_ms=? WHERE lease_id=?", (state, now, lease_id))
            conn.execute(
                "UPDATE node_quota SET active_shards=active_shards-1,active_leases=active_leases-1 WHERE node_id=?",
                (row["node_id"],),
            )
            conn.execute(
                "UPDATE node_quota SET active_shards=MAX(active_shards,0),active_leases=MAX(active_leases,0)"
            )
            units = int(row["resource_units"])
            conn.execute("UPDATE budget SET used_units=MAX(used_units-?,0) WHERE budget_id=1", (units,))
            conn.execute(
                "UPDATE attempts SET status=?,updated_monotonic_ms=?,detail=? WHERE attempt_id=("
                "SELECT attempt_id FROM attempts WHERE shard_id=? ORDER BY attempt_id DESC LIMIT 1)",
                (state, now, detail, row["shard_id"]),
            )
            if event_seq is not None:
                conn.execute("UPDATE cursor SET last_event_seq=MAX(last_event_seq,?) WHERE cursor_id=1", (int(event_seq),))
            self._update_digest(conn)
            return True

    def release_reservation(self, *, shard_id: str, lease_id: str, units: int, state: str, detail: str | None = None, event_seq: int | None = None) -> bool:
        """Release a reserved lease and its known units exactly once."""
        if state not in {"cancelled", "expired", "failed"}:
            raise ValueError(f"invalid release state: {state}")
        with self._lock, self._transaction() as conn:
            row = conn.execute("SELECT * FROM leases WHERE lease_id=?", (lease_id,)).fetchone()
            if row is None or row["state"] != "active":
                if event_seq is not None:
                    conn.execute("UPDATE cursor SET last_event_seq=MAX(last_event_seq,?) WHERE cursor_id=1", (int(event_seq),))
                    self._update_digest(conn)
                return False
            now = monotonic_ms()
            conn.execute("UPDATE leases SET state=?,updated_monotonic_ms=? WHERE lease_id=?", (state, now, lease_id))
            conn.execute(
                "UPDATE node_quota SET active_shards=MAX(active_shards-1,0),active_leases=MAX(active_leases-1,0) WHERE node_id=?",
                (row["node_id"],),
            )
            actual_units = int(row["resource_units"])
            conn.execute("UPDATE budget SET used_units=MAX(used_units-?,0) WHERE budget_id=1", (actual_units,))
            conn.execute(
                "UPDATE attempts SET status=?,updated_monotonic_ms=?,detail=? WHERE attempt_id=("
                "SELECT attempt_id FROM attempts WHERE shard_id=? ORDER BY attempt_id DESC LIMIT 1)",
                (state, now, detail, shard_id),
            )
            if event_seq is not None:
                conn.execute("UPDATE cursor SET last_event_seq=MAX(last_event_seq,?) WHERE cursor_id=1", (int(event_seq),))
            self._update_digest(conn)
            return True

    def mark_attempt_failed(self, *, shard_id: str, detail: str) -> None:
        with self._lock, self._transaction() as conn:
            conn.execute(
                "UPDATE attempts SET status='failed',updated_monotonic_ms=?,detail=? WHERE attempt_id=("
                "SELECT attempt_id FROM attempts WHERE shard_id=? ORDER BY attempt_id DESC LIMIT 1)",
                (monotonic_ms(), detail, shard_id),
            )

    # ------------------------------------------------------------ event journal
    def applied_event_seqs(self) -> list[int]:
        with self._lock:
            return [int(row["event_seq"]) for row in self.conn.execute("SELECT event_seq FROM applied_events ORDER BY event_seq")]

    def active_revision(self) -> int:
        with self._lock:
            return int(self._meta(key="active_revision") or "1")

    def dispatch_requests(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {row["shard_id"]: dict(row) for row in self.conn.execute("SELECT * FROM dispatch_requests ORDER BY shard_id")}

    def cancelled_shards(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {row["shard_id"]: dict(row) for row in self.conn.execute("SELECT * FROM cancelled_shards ORDER BY shard_id")}

    def _record_applied_event(self, conn: sqlite3.Connection, event: Mapping[str, Any]) -> None:
        conn.execute(
            "INSERT INTO applied_events(event_seq,event_type,payload_json,applied_monotonic_ms) VALUES(?,?,?,?) "
            "ON CONFLICT(event_seq) DO UPDATE SET event_type=excluded.event_type,payload_json=excluded.payload_json",
            (
                int(event["seq"]),
                str(event["type"]),
                canonical_json(event),
                monotonic_ms(),
            ),
        )
        conn.execute("UPDATE cursor SET last_event_seq=MAX(last_event_seq,?) WHERE cursor_id=1", (int(event["seq"]),))

    def apply_dispatch_event(
        self,
        event: Mapping[str, Any],
        *,
        revision_version: int,
        default_ttl_ms: int,
    ) -> dict[str, Any]:
        with self._lock, self._transaction() as conn:
            shard_id = str(event["shard_id"])
            cancelled = conn.execute("SELECT 1 FROM cancelled_shards WHERE shard_id=?", (shard_id,)).fetchone()
            result = conn.execute("SELECT generation FROM results WHERE shard_id=?", (shard_id,)).fetchone()
            prior = conn.execute("SELECT generation FROM dispatch_requests WHERE shard_id=?", (shard_id,)).fetchone()
            base_generation = max(
                int(result["generation"]) if result is not None else 0,
                int(prior["generation"]) if prior is not None else 0,
                1,
            )
            requested = 0 if cancelled is not None else 1
            ttl_ms = int(event.get("ttl_ms", default_ttl_ms))
            conn.execute(
                "INSERT INTO dispatch_requests(shard_id,event_seq,revision_version,generation,requested,ttl_ms) VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(shard_id) DO UPDATE SET event_seq=excluded.event_seq,revision_version=excluded.revision_version,"
                "generation=MAX(dispatch_requests.generation,excluded.generation),requested=excluded.requested,"
                "ttl_ms=excluded.ttl_ms",
                (shard_id, int(event["seq"]), int(revision_version), base_generation, requested, ttl_ms),
            )
            self._record_applied_event(conn, event)
            self._update_digest(conn)
            return {"requested": bool(requested), "generation": base_generation}

    def apply_cancel_event(self, event: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock, self._transaction() as conn:
            shard_id = str(event["shard_id"])
            reason = str(event.get("reason", "cancelled"))
            conn.execute(
                "INSERT INTO cancelled_shards(shard_id,event_seq,reason) VALUES(?,?,?) "
                "ON CONFLICT(shard_id) DO UPDATE SET event_seq=excluded.event_seq,reason=excluded.reason",
                (shard_id, int(event["seq"]), reason),
            )
            conn.execute("UPDATE dispatch_requests SET requested=0 WHERE shard_id=?", (shard_id,))
            row = conn.execute(
                "SELECT * FROM leases WHERE shard_id=? AND state='active' ORDER BY lease_id DESC LIMIT 1",
                (shard_id,),
            ).fetchone()
            released: dict[str, Any] | None = None
            if row is not None:
                now = monotonic_ms()
                conn.execute("UPDATE leases SET state='cancelled',updated_monotonic_ms=? WHERE lease_id=?", (now, row["lease_id"]))
                conn.execute(
                    "UPDATE node_quota SET active_shards=MAX(active_shards-1,0),active_leases=MAX(active_leases-1,0) WHERE node_id=?",
                    (row["node_id"],),
                )
                conn.execute("UPDATE budget SET used_units=MAX(used_units-?,0) WHERE budget_id=1", (int(row["resource_units"]),))
                conn.execute(
                    "UPDATE attempts SET status='cancelled',updated_monotonic_ms=?,detail=? WHERE attempt_id=("
                    "SELECT attempt_id FROM attempts WHERE shard_id=? ORDER BY attempt_id DESC LIMIT 1)",
                    (now, reason, shard_id),
                )
                released = {
                    "lease_id": row["lease_id"],
                    "node_id": row["node_id"],
                    "resource_units": int(row["resource_units"]),
                    "generation": int(row["generation"]),
                }
            self._record_applied_event(conn, event)
            self._update_digest(conn)
            return {"cancelled": True, "released": released}

    def apply_revision_event(
        self,
        event: Mapping[str, Any],
        *,
        revision_version: int,
        global_budget: int,
        nodes: Mapping[str, Mapping[str, int]],
        invalidation_entries: Iterable[Mapping[str, Any]],
        requested_generations: Mapping[str, int],
        checks: Iterable[Mapping[str, Any]] = (),
        event_type: str = "revision",
    ) -> None:
        with self._lock, self._transaction() as conn:
            budget = conn.execute("SELECT used_units FROM budget WHERE budget_id=1").fetchone()
            if budget is not None and int(global_budget) < budget["used_units"]:
                raise BusinessError("E_LEASE", "global budget below reserved units", field="global_budget")
            for node_id in sorted(nodes):
                values = nodes[node_id]
                row = conn.execute("SELECT * FROM node_quota WHERE node_id=?", (node_id,)).fetchone()
                if row is None:
                    raise BusinessError("E_LEDGER", f"missing node quota row: {node_id}", field=node_id)
                if int(values["concurrency"]) < row["active_shards"] or int(values["lease_limit"]) < row["active_leases"]:
                    raise BusinessError("E_LEASE", "quota transition would be below in-flight usage", field="quota", detail={"node_id": node_id})
                conn.execute(
                    "UPDATE node_quota SET concurrency=?,lease_limit=?,capacity_version=? WHERE node_id=?",
                    (int(values["concurrency"]), int(values["lease_limit"]), int(values["capacity_version"]), node_id),
                )
            conn.execute("UPDATE budget SET global_budget=? WHERE budget_id=1", (int(global_budget),))
            for entry in invalidation_entries:
                conn.execute(
                    "INSERT INTO invalidations(event_seq,shard_id,direct,reason,topo_rank) VALUES(?,?,?,?,?) "
                    "ON CONFLICT(event_seq,shard_id) DO UPDATE SET direct=excluded.direct,reason=excluded.reason,topo_rank=excluded.topo_rank",
                    (
                        int(event["seq"]),
                        str(entry["shard_id"]),
                        1 if entry["direct"] else 0,
                        str(entry["reason"]),
                        int(entry["topo_rank"]),
                    ),
                )
            for check in checks:
                conn.execute(
                    "INSERT INTO incremental_checks(event_seq,shard_id,digest_before,digest_after,unchanged) VALUES(?,?,?,?,?) "
                    "ON CONFLICT(event_seq,shard_id) DO UPDATE SET digest_before=excluded.digest_before,"
                    "digest_after=excluded.digest_after,unchanged=excluded.unchanged",
                    (
                        int(event["seq"]),
                        str(check["shard_id"]),
                        str(check["digest_before"]),
                        str(check["digest_after"]),
                        1 if check["unchanged"] else 0,
                    ),
                )
            for shard_id in sorted(requested_generations):
                generation = int(requested_generations[shard_id])
                existing = conn.execute("SELECT * FROM dispatch_requests WHERE shard_id=?", (shard_id,)).fetchone()
                cancelled = conn.execute("SELECT 1 FROM cancelled_shards WHERE shard_id=?", (shard_id,)).fetchone()
                should_request = cancelled is None and existing is not None and (
                    bool(existing["requested"])
                    or conn.execute("SELECT 1 FROM results WHERE shard_id=?", (shard_id,)).fetchone() is not None
                )
                conn.execute(
                    "INSERT INTO dispatch_requests(shard_id,event_seq,revision_version,generation,requested,ttl_ms) VALUES(?,?,?,?,?,0) "
                    "ON CONFLICT(shard_id) DO UPDATE SET event_seq=excluded.event_seq,revision_version=excluded.revision_version,"
                    "generation=MAX(dispatch_requests.generation,excluded.generation),requested=excluded.requested",
                    (shard_id, int(event["seq"]), int(revision_version), generation, 1 if should_request else 0),
                )
            self._set_meta(conn, "active_revision", str(int(revision_version)))
            self._record_applied_event(conn, event)
            # event_type is retained in the applied event payload/type; this argument
            # documents whether a revision or quota parameter change drove invalidation.
            _ = event_type
            self._update_digest(conn)

    # ---------------------------------------------------------- invalidation
    def record_invalidation(
        self,
        *,
        event_seq: int,
        entries: Iterable[Mapping[str, Any]],
        checks: Iterable[Mapping[str, Any]] = (),
    ) -> None:
        with self._lock, self._transaction() as conn:
            for entry in entries:
                conn.execute(
                    "INSERT INTO invalidations(event_seq,shard_id,direct,reason,topo_rank) VALUES(?,?,?,?,?) "
                    "ON CONFLICT(event_seq,shard_id) DO UPDATE SET direct=excluded.direct,reason=excluded.reason,topo_rank=excluded.topo_rank",
                    (
                        int(event_seq),
                        str(entry["shard_id"]),
                        1 if entry["direct"] else 0,
                        str(entry["reason"]),
                        int(entry["topo_rank"]),
                    ),
                )
            for check in checks:
                conn.execute(
                    "INSERT INTO incremental_checks(event_seq,shard_id,digest_before,digest_after,unchanged) VALUES(?,?,?,?,?) "
                    "ON CONFLICT(event_seq,shard_id) DO UPDATE SET digest_before=excluded.digest_before,"
                    "digest_after=excluded.digest_after,unchanged=excluded.unchanged",
                    (
                        int(event_seq),
                        str(check["shard_id"]),
                        str(check["digest_before"]),
                        str(check["digest_after"]),
                        1 if check["unchanged"] else 0,
                    ),
                )
            conn.execute("UPDATE cursor SET last_event_seq=MAX(last_event_seq,?) WHERE cursor_id=1", (int(event_seq),))
            self._update_digest(conn)

    # --------------------------------------------------------------- queries
    def results(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {row["shard_id"]: dict(row) for row in self.conn.execute("SELECT * FROM results ORDER BY shard_id")}

    def result_digests(self) -> dict[str, str]:
        with self._lock:
            return {row["shard_id"]: row["result_digest"] for row in self.conn.execute("SELECT shard_id,result_digest FROM results ORDER BY shard_id")}

    def leases(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self.conn.execute("SELECT * FROM leases ORDER BY lease_id")]

    def node_quotas(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self.conn.execute("SELECT * FROM node_quota ORDER BY node_id")]

    def budget_state(self) -> dict[str, Any]:
        with self._lock:
            row = self.conn.execute("SELECT * FROM budget WHERE budget_id=1").fetchone()
            return dict(row) if row else {}

    def snapshots(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT snapshot_id,after_event_seq,created_monotonic_ms,business_digest FROM snapshots "
                "ORDER BY snapshot_id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
            return [dict(row) for row in reversed(rows)]

    def invalidations(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self.conn.execute("SELECT * FROM invalidations ORDER BY event_seq,topo_rank,shard_id")]

    def incremental_checks(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self.conn.execute("SELECT * FROM incremental_checks ORDER BY event_seq,shard_id")]

    def result_history(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                dict(row)
                for row in self.conn.execute("SELECT * FROM result_history ORDER BY shard_id,generation")
            ]

    def unknown_attempts(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self.conn.execute("SELECT * FROM attempts WHERE status IN ('unknown','in_flight') ORDER BY shard_id,attempt_id")]

    def attempts(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self.conn.execute("SELECT * FROM attempts ORDER BY attempt_id")]

    def exit_reports(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self.conn.execute("SELECT * FROM exit_reports ORDER BY report_id")]

    def recovery_events(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self.conn.execute("SELECT * FROM recovery_events ORDER BY recovery_id")]

    def next_generation(self, shard_id: str) -> int:
        with self._lock:
            row = self.conn.execute(
                "SELECT generation FROM results WHERE shard_id=? UNION ALL "
                "SELECT generation FROM result_history WHERE shard_id=? ORDER BY generation DESC LIMIT 1",
                (shard_id, shard_id),
            ).fetchone()
            return int(row["generation"]) + 1 if row else 1

    def state_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "ledger_format_version": int(self._meta(key="ledger_format_version")),
                "content_digest": self.content_digest(),
                "clean_shutdown": self._meta(key="clean_shutdown"),
                "results": self.results(),
                "leases": self.leases(),
                "node_quota": self.node_quotas(),
                "budget": self.budget_state(),
                "invalidations": self.invalidations(),
                "incremental_checks": self.incremental_checks(),
                "unknown_attempts": self.unknown_attempts(),
                "snapshots": self.snapshots(),
                "active_revision": self.active_revision(),
                "applied_events": self.applied_event_seqs(),
                "dispatch_requests": self.dispatch_requests(),
                "cancelled_shards": self.cancelled_shards(),
            }

    def verify_snapshots(self) -> dict[str, Any]:
        with self._lock:
            digest = self.verify_content_digest()
            rows = [dict(row) for row in self.conn.execute("SELECT * FROM snapshots ORDER BY snapshot_id")]
            for row in rows:
                payload_digest = digest_obj(__import__("json").loads(row["payload_json"]))
                if payload_digest != row["business_digest"]:
                    raise BusinessError(
                        "E_SNAPSHOT",
                        "snapshot payload digest mismatch",
                        position=str(self.db_path),
                        field="business_digest",
                        detail={"snapshot_id": row["snapshot_id"], "expected": row["business_digest"], "actual": payload_digest},
                    )
            return {
                "content_digest": digest,
                "snapshot_count": len(rows),
                "latest_snapshot_digest": rows[-1]["business_digest"] if rows else None,
            }

    def inject_digest_corruption(self) -> None:
        """Test/demo hook: deliberately make the persisted content digest inconsistent."""
        with self._lock, self._transaction() as conn:
            self._set_meta(conn, "content_digest", "0" * 64)


class _Transaction:
    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    def __enter__(self) -> sqlite3.Connection:
        self.ledger.conn.execute("BEGIN IMMEDIATE")
        return self.ledger.conn

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.ledger.conn.execute("COMMIT")
        else:
            self.ledger.conn.execute("ROLLBACK")
        return False


def compare_replay(baseline_path: str | Path, candidate_path: str | Path) -> dict[str, Any]:
    """Compare uninterrupted and recovered ledgers field by field."""
    baseline = Ledger(baseline_path)
    candidate = Ledger(candidate_path)
    try:
        base_version = baseline._meta(key="ledger_format_version")
        cand_version = candidate._meta(key="ledger_format_version")
        if base_version != cand_version:
            raise BusinessError(
                "E_REPLAY",
                "ledger format version mismatch",
                shard_id="<ledger>",
                field="ledger_format_version",
                detail={"baseline": base_version, "candidate": cand_version},
            )
        base_results = baseline.results()
        cand_results = candidate.results()
        compared: list[str] = []
        for shard_id in sorted(set(base_results) | set(cand_results)):
            if shard_id not in base_results:
                raise BusinessError(
                    "E_REPLAY",
                    "candidate has a shard absent from baseline",
                    shard_id=shard_id,
                    field="presence",
                    detail={"baseline": None, "candidate": "present"},
                )
            if shard_id not in cand_results:
                raise BusinessError(
                    "E_REPLAY",
                    "baseline shard missing from candidate",
                    shard_id=shard_id,
                    field="presence",
                    detail={"baseline": "present", "candidate": None},
                )
            before = base_results[shard_id]
            after = cand_results[shard_id]
            for field_name in (
                "outcome",
                "summary_json",
                "node_id",
                "lease_id",
                "lease_version",
                "config_version",
                "resource_units",
                "generation",
                "completion_event_seq",
            ):
                if before[field_name] != after[field_name]:
                    raise BusinessError(
                        "E_REPLAY",
                        f"replay mismatch in {field_name}",
                        shard_id=shard_id,
                        field=field_name,
                        detail={"baseline": before[field_name], "candidate": after[field_name]},
                    )
            compared.append(shard_id)
        base_quota = {row["node_id"]: row for row in baseline.node_quotas()}
        cand_quota = {row["node_id"]: row for row in candidate.node_quotas()}
        first_shard_by_node: dict[str, str] = {}
        for shard in sorted(base_results):
            first_shard_by_node.setdefault(base_results[shard]["node_id"], shard)
        for node_id in sorted(set(base_quota) | set(cand_quota)):
            if base_quota.get(node_id) != cand_quota.get(node_id):
                b = base_quota.get(node_id, {})
                c = cand_quota.get(node_id, {})
                for field_name in ("concurrency", "lease_limit", "capacity_version", "active_shards", "active_leases"):
                    if b.get(field_name) != c.get(field_name):
                        raise BusinessError(
                            "E_REPLAY",
                            f"quota occupancy mismatch in {field_name}",
                            shard_id=first_shard_by_node.get(node_id, f"<node:{node_id}>"),
                            field=field_name,
                            detail={"node_id": node_id, "baseline": b.get(field_name), "candidate": c.get(field_name)},
                        )
        base_budget = baseline.budget_state()
        cand_budget = candidate.budget_state()
        for field_name in ("global_budget", "used_units"):
            if base_budget.get(field_name) != cand_budget.get(field_name):
                raise BusinessError(
                    "E_REPLAY",
                    f"quota occupancy mismatch in budget.{field_name}",
                    shard_id="<ledger>",
                    field=field_name,
                    detail={"baseline": base_budget.get(field_name), "candidate": cand_budget.get(field_name)},
                )
        base_digest = baseline.content_digest()
        cand_digest = candidate.content_digest()
        if base_digest != cand_digest:
            raise BusinessError(
                "E_REPLAY",
                "ledger business digest mismatch",
                shard_id="<ledger>",
                field="content_digest",
                detail={"baseline": base_digest, "candidate": cand_digest},
            )
        return {
            "ok": True,
            "compared_shards": compared,
            "shard_count": len(compared),
            "ledger_digest": base_digest,
            "quota": {
                "global_budget": base_budget.get("global_budget"),
                "used_units": base_budget.get("used_units"),
                "nodes": [
                    {
                        key: row.get(key)
                        for key in ("node_id", "concurrency", "lease_limit", "capacity_version", "active_shards", "active_leases")
                    }
                    for row in sorted(base_quota.values(), key=lambda item: item["node_id"])
                ],
            },
        }
    finally:
        baseline.close()
        candidate.close()
