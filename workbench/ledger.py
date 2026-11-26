from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from . import LEDGER_FORMAT_VERSION
from .canonical import canonical_json, content_digest

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS shards (
  shard_id TEXT PRIMARY KEY,
  topo_index INTEGER NOT NULL,
  status TEXT NOT NULL,
  generation INTEGER NOT NULL DEFAULT 1,
  node_id TEXT,
  lease_id TEXT,
  sample_rows INTEGER,
  participating_rows INTEGER,
  capacity_version INTEGER,
  summary_digest TEXT,
  summary_json TEXT,
  wait_started_ms INTEGER,
  wait_ended_ms INTEGER,
  wait_ms INTEGER,
  error_code TEXT,
  updated_at_ms INTEGER NOT NULL,
  format_version INTEGER NOT NULL,
  content_digest TEXT
);
CREATE TABLE IF NOT EXISTS leases (
  lease_id TEXT PRIMARY KEY,
  shard_id TEXT NOT NULL,
  node_id TEXT NOT NULL,
  generation INTEGER NOT NULL DEFAULT 1,
  granted_at_ms INTEGER NOT NULL,
  expires_at_ms INTEGER NOT NULL,
  renew_count INTEGER NOT NULL,
  lease_version INTEGER NOT NULL,
  capacity_version INTEGER,
  state TEXT NOT NULL,
  reap_reason TEXT,
  format_version INTEGER NOT NULL,
  content_digest TEXT NOT NULL,
  UNIQUE(shard_id, generation)
);
CREATE TABLE IF NOT EXISTS resource_occupancy (
  shard_id TEXT NOT NULL,
  generation INTEGER NOT NULL DEFAULT 1,
  lease_id TEXT,
  node_id TEXT,
  resource_units INTEGER NOT NULL,
  global_budget INTEGER NOT NULL,
  quota_slots INTEGER NOT NULL DEFAULT 1,
  lease_slots INTEGER NOT NULL DEFAULT 1,
  state TEXT NOT NULL,
  started_at_ms INTEGER NOT NULL,
  ended_at_ms INTEGER,
  scale_exponent INTEGER NOT NULL,
  scaled_resource_units INTEGER NOT NULL,
  saturated INTEGER NOT NULL DEFAULT 0,
  format_version INTEGER NOT NULL,
  content_digest TEXT NOT NULL,
  PRIMARY KEY(shard_id, generation)
);
CREATE TABLE IF NOT EXISTS water_records (
  record_key TEXT PRIMARY KEY,
  scope_type TEXT NOT NULL,
  scope_id TEXT NOT NULL,
  shard_id TEXT,
  record_json TEXT NOT NULL,
  format_version INTEGER NOT NULL,
  content_digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS error_attribution (
  shard_id TEXT PRIMARY KEY,
  record_json TEXT NOT NULL,
  format_version INTEGER NOT NULL,
  content_digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS outcome_commits (
  shard_id TEXT PRIMARY KEY,
  format_version INTEGER NOT NULL,
  content_digest TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  committed_at_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS progress_cursor (
  id INTEGER PRIMARY KEY CHECK(id = 1),
  last_event_seq INTEGER NOT NULL,
  last_topo_index INTEGER,
  last_shard_id TEXT,
  completed_count INTEGER NOT NULL,
  active_lease_count INTEGER NOT NULL,
  active_resource_units INTEGER NOT NULL,
  format_version INTEGER NOT NULL,
  content_digest TEXT NOT NULL,
  updated_at_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS invalidations (
  ordinal INTEGER PRIMARY KEY AUTOINCREMENT,
  spec_digest TEXT NOT NULL,
  shard_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  basis_json TEXT NOT NULL,
  created_at_ms INTEGER NOT NULL
);
"""

EXPORT_TABLES = [
    "meta",
    "shards",
    "leases",
    "resource_occupancy",
    "water_records",
    "error_attribution",
    "outcome_commits",
    "progress_cursor",
    "invalidations",
]


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(
            str(self.path), timeout=30.0, isolation_level=None, check_same_thread=False
        )
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA user_version=1")
        self.conn.executescript(SCHEMA_SQL)

    def close(self) -> None:
        with self.lock:
            self.conn.close()

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        with self.lock:
            row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
            return str(row["value"]) if row else default

    def set_meta(self, key: str, value: str) -> None:
        with self.lock, self.conn:
            self.conn.execute(
                "INSERT INTO meta(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, str(value)),
            )

    def initialize_spec(self, spec_digest: str, spec_json: str, now_ms: int) -> bool:
        with self.lock, self.conn:
            existing = self.get_meta("spec_digest")
            if existing is None:
                self.conn.execute(
                    "INSERT INTO meta(key,value) VALUES('ledger_format_version',?)",
                    (str(LEDGER_FORMAT_VERSION),),
                )
                self.conn.execute("INSERT INTO meta(key,value) VALUES('spec_digest',?)", (spec_digest,))
                self.conn.execute("INSERT INTO meta(key,value) VALUES('spec_json',?)", (spec_json,))
                self.conn.execute(
                    "INSERT INTO meta(key,value) VALUES('initialized_at_ms',?)", (str(now_ms),)
                )
                return True
            return existing == spec_digest

    def upsert_dispatch(self, dispatch: dict[str, Any]) -> None:
        now_ms = int(dispatch["now_ms"])
        generation = int(dispatch.get("generation", 1))
        with self.lock, self.conn:
            self.conn.execute(
                """
                INSERT INTO shards(shard_id, topo_index, status, generation, node_id, lease_id,
                  sample_rows, capacity_version, wait_started_ms, wait_ended_ms, wait_ms,
                  error_code, updated_at_ms, format_version, content_digest)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(shard_id) DO UPDATE SET
                  topo_index=excluded.topo_index, status=excluded.status,
                  generation=excluded.generation, node_id=excluded.node_id,
                  lease_id=excluded.lease_id, sample_rows=excluded.sample_rows,
                  capacity_version=excluded.capacity_version,
                  wait_started_ms=COALESCE(shards.wait_started_ms, excluded.wait_started_ms),
                  wait_ended_ms=excluded.wait_ended_ms, wait_ms=excluded.wait_ms,
                  error_code=NULL, updated_at_ms=excluded.updated_at_ms,
                  format_version=excluded.format_version, content_digest=excluded.content_digest
                """,
                (
                    dispatch["shard_id"],
                    int(dispatch["topo_index"]),
                    "in_flight",
                    generation,
                    dispatch["node_id"],
                    dispatch["lease_id"],
                    int(dispatch["sample_rows"]),
                    int(dispatch["capacity_version"]),
                    dispatch.get("wait_started_ms"),
                    dispatch.get("wait_ended_ms"),
                    int(dispatch.get("wait_ms", 0)),
                    None,
                    now_ms,
                    LEDGER_FORMAT_VERSION,
                    content_digest(dispatch),
                ),
            )
            lease = dispatch["lease"]
            lease_digest = content_digest(lease)
            self.conn.execute(
                """
                INSERT INTO leases(lease_id, shard_id, node_id, generation, granted_at_ms,
                  expires_at_ms, renew_count, lease_version, capacity_version, state,
                  reap_reason, format_version, content_digest)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(lease_id) DO UPDATE SET
                  shard_id=excluded.shard_id, node_id=excluded.node_id,
                  generation=excluded.generation,
                  granted_at_ms=excluded.granted_at_ms,
                  expires_at_ms=excluded.expires_at_ms, renew_count=excluded.renew_count,
                  lease_version=excluded.lease_version, state='active', reap_reason=NULL,
                  capacity_version=excluded.capacity_version,
                  format_version=excluded.format_version, content_digest=excluded.content_digest
                """,
                (
                    lease["lease_id"],
                    lease["shard_id"],
                    lease["node_id"],
                    generation,
                    int(lease["granted_at_ms"]),
                    int(lease["expires_at_ms"]),
                    int(lease["renew_count"]),
                    int(lease["lease_version"]),
                    int(lease.get("capacity_version", dispatch["capacity_version"])),
                    "active",
                    None,
                    LEDGER_FORMAT_VERSION,
                    lease_digest,
                ),
            )
            occupancy = dispatch["occupancy"]
            occ_digest = content_digest(occupancy)
            self.conn.execute(
                """
                INSERT INTO resource_occupancy(shard_id, generation, lease_id, node_id,
                  resource_units, global_budget, quota_slots, lease_slots, state,
                  started_at_ms, ended_at_ms, scale_exponent, scaled_resource_units,
                  saturated, format_version, content_digest)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(shard_id, generation) DO UPDATE SET
                  lease_id=excluded.lease_id, state='active', ended_at_ms=NULL,
                  node_id=excluded.node_id, resource_units=excluded.resource_units,
                  global_budget=excluded.global_budget,
                  scale_exponent=excluded.scale_exponent,
                  scaled_resource_units=excluded.scaled_resource_units,
                  saturated=excluded.saturated, started_at_ms=excluded.started_at_ms,
                  content_digest=excluded.content_digest
                """,
                (
                    dispatch["shard_id"],
                    generation,
                    dispatch["lease_id"],
                    dispatch["node_id"],
                    int(occupancy["resource_units"]),
                    int(occupancy["global_budget"]),
                    1,
                    1,
                    "active",
                    now_ms,
                    None,
                    int(occupancy["scale_exponent"]),
                    int(occupancy["scaled_resource_units"]),
                    1 if occupancy.get("saturated") else 0,
                    LEDGER_FORMAT_VERSION,
                    occ_digest,
                ),
            )

    def get_shard(self, shard_id: str) -> dict[str, Any] | None:
        with self.lock:
            row = self.conn.execute(
                "SELECT * FROM shards WHERE shard_id=?", (shard_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_shards(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute("SELECT * FROM shards ORDER BY topo_index, shard_id").fetchall()
            return [dict(row) for row in rows]

    def get_lease(self, lease_id: str) -> dict[str, Any] | None:
        with self.lock:
            row = self.conn.execute("SELECT * FROM leases WHERE lease_id=?", (lease_id,)).fetchone()
            return dict(row) if row else None

    def list_leases(self, active_only: bool = False) -> list[dict[str, Any]]:
        sql = "SELECT * FROM leases"
        if active_only:
            sql += " WHERE state='active'"
        sql += " ORDER BY granted_at_ms, lease_id"
        with self.lock:
            return [dict(row) for row in self.conn.execute(sql).fetchall()]

    def active_resource_sum(self) -> int:
        with self.lock:
            row = self.conn.execute(
                "SELECT COALESCE(SUM(resource_units),0) AS total FROM resource_occupancy WHERE state='active'"
            ).fetchone()
            return int(row["total"])

    def active_node_counts(self) -> dict[str, dict[str, int]]:
        with self.lock:
            rows = self.conn.execute(
                """
                SELECT node_id,
                       SUM(CASE WHEN state='active' THEN 1 ELSE 0 END) AS count
                FROM resource_occupancy
                WHERE state='active'
                GROUP BY node_id
                """
            ).fetchall()
            return {str(row["node_id"]): {"shards": int(row["count"]), "leases": int(row["count"])} for row in rows}

    def active_node_resources(self) -> dict[str, int]:
        with self.lock:
            rows = self.conn.execute(
                """
                SELECT node_id, COALESCE(SUM(resource_units), 0) AS total
                FROM resource_occupancy
                WHERE state='active'
                GROUP BY node_id
                """
            ).fetchall()
            return {str(row["node_id"]): int(row["total"]) for row in rows}

    def transition_lease(
        self,
        lease_id: str,
        *,
        new_state: str,
        reap_reason: str,
        now_ms: int,
        expected_version: int | None = None,
    ) -> bool:
        with self.lock, self.conn:
            sql = "UPDATE leases SET state=?, reap_reason=? WHERE lease_id=? AND state='active'"
            params: list[Any] = [new_state, reap_reason, lease_id]
            if expected_version is not None:
                sql += " AND lease_version=?"
                params.append(expected_version)
            changed = self.conn.execute(sql, params).rowcount
            if changed:
                self.conn.execute(
                    """
                    UPDATE resource_occupancy
                    SET state=?, ended_at_ms=?
                    WHERE lease_id=? AND state='active'
                    """,
                    ("released" if new_state == "released" else "reclaimed", now_ms, lease_id),
                )
            return bool(changed)

    def renew_lease_row(
        self,
        lease_id: str,
        *,
        expected_version: int,
        new_version: int,
        renew_count: int,
        expires_at_ms: int,
        now_ms: int,
    ) -> bool:
        with self.lock, self.conn:
            changed = self.conn.execute(
                """
                UPDATE leases
                SET lease_version=?, renew_count=?, expires_at_ms=?, content_digest=?
                WHERE lease_id=? AND state='active' AND lease_version=?
                """,
                (
                    new_version,
                    renew_count,
                    expires_at_ms,
                    content_digest(
                        {
                            "lease_id": lease_id,
                            "lease_version": new_version,
                            "renew_count": renew_count,
                            "expires_at_ms": expires_at_ms,
                        }
                    ),
                    lease_id,
                    expected_version,
                ),
            ).rowcount
            if changed:
                self.conn.execute(
                    "UPDATE shards SET updated_at_ms=? WHERE lease_id=?", (now_ms, lease_id)
                )
            return bool(changed)

    def reap_expired(self, now_ms: int) -> list[str]:
        reaped: list[str] = []
        with self.lock:
            rows = self.conn.execute(
                "SELECT lease_id FROM leases WHERE state='active' AND expires_at_ms <= ?",
                (now_ms,),
            ).fetchall()
            for row in rows:
                if self.transition_lease(
                    row["lease_id"], new_state="expired", reap_reason="monotonic_ttl", now_ms=now_ms
                ):
                    reaped.append(row["lease_id"])
        return reaped

    def reap_version_generation(self, node_id: str, capacity_version: int, now_ms: int) -> list[str]:
        reaped: list[str] = []
        with self.lock:
            rows = self.conn.execute(
                """
                SELECT lease_id FROM leases
                WHERE state='active' AND node_id=? AND capacity_version=?
                """,
                (node_id, int(capacity_version)),
            ).fetchall()
            for row in rows:
                if self.transition_lease(
                    row["lease_id"],
                    new_state="superseded",
                    reap_reason="capacity_version_generation",
                    now_ms=now_ms,
                ):
                    reaped.append(row["lease_id"])
        return reaped

    def recover_uncommitted(self, now_ms: int) -> list[dict[str, Any]]:
        recovered: list[dict[str, Any]] = []
        with self.lock, self.conn:
            rows = self.conn.execute(
                "SELECT * FROM shards WHERE status IN ('in_flight','dispatched','unknown')"
            ).fetchall()
            for row in rows:
                shard = dict(row)
                lease_id = shard.get("lease_id")
                if lease_id:
                    self.conn.execute(
                        """
                        UPDATE leases SET state='reclaimed', reap_reason='forced_termination_recovery'
                        WHERE lease_id=? AND state='active'
                        """,
                        (lease_id,),
                    )
                    self.conn.execute(
                        """
                        UPDATE resource_occupancy SET state='reclaimed', ended_at_ms=?
                        WHERE lease_id=? AND state='active'
                        """,
                        (now_ms, lease_id),
                    )
                self.conn.execute(
                    """
                    UPDATE shards SET status='unknown', error_code='E_REPLAY_RERUN_REQUIRED',
                      updated_at_ms=?, content_digest=?
                    WHERE shard_id=?
                    """,
                    (now_ms, content_digest({"shard_id": shard["shard_id"], "status": "unknown"}), shard["shard_id"]),
                )
                recovered.append(shard)
        return recovered

    def commit_shard_outcome(self, outcome: dict[str, Any]) -> str:
        now_ms = int(outcome["committed_at_ms"])
        shard_id = str(outcome["shard_id"])
        payload = dict(outcome)
        payload_digest = content_digest(payload)
        conclusion_digest = str(outcome.get("conclusion_digest") or payload_digest)
        summary_json = json.dumps(outcome["summary"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self.lock, self.conn:
            self.conn.execute(
                """
                UPDATE shards SET status='completed', node_id=?, lease_id=?, sample_rows=?,
                  participating_rows=?, capacity_version=?, summary_digest=?, summary_json=?,
                  wait_started_ms=?, wait_ended_ms=?, wait_ms=?, error_code=NULL,
                  updated_at_ms=?, format_version=?, content_digest=?
                WHERE shard_id=?
                """,
                (
                    outcome["node_id"],
                    outcome["lease_id"],
                    int(outcome["sample_rows"]),
                    int(outcome["participating_rows"]),
                    int(outcome["capacity_version"]),
                    outcome["summary_digest"],
                    summary_json,
                    outcome.get("wait_started_ms"),
                    outcome.get("wait_ended_ms"),
                    int(outcome.get("wait_ms", 0)),
                    now_ms,
                    LEDGER_FORMAT_VERSION,
                    conclusion_digest,
                    shard_id,
                ),
            )
            self.conn.execute(
                """
                UPDATE leases SET state='released', reap_reason='completed',
                  expires_at_ms=?, renew_count=?, lease_version=?, content_digest=?
                WHERE lease_id=?
                """,
                (
                    int(outcome["lease"]["expires_at_ms"]),
                    int(outcome["lease"]["renew_count"]),
                    int(outcome["lease"]["lease_version"]),
                    content_digest(outcome["lease"]),
                    outcome["lease_id"],
                ),
            )
            self.conn.execute(
                """
                UPDATE resource_occupancy SET state='released', ended_at_ms=?,
                  scale_exponent=?, scaled_resource_units=?, saturated=?
                WHERE shard_id=? AND generation=?
                """,
                (
                    now_ms,
                    int(outcome["water"]["scale_exponent"]),
                    int(outcome["scaled_resource_units"]),
                    1 if outcome.get("saturated") else 0,
                    shard_id,
                    int(outcome.get("generation", 1)),
                ),
            )
            error_json = json.dumps(outcome["error_attribution"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            self.conn.execute(
                """
                INSERT INTO error_attribution(shard_id, record_json, format_version, content_digest)
                VALUES(?,?,?,?) ON CONFLICT(shard_id) DO UPDATE SET
                  record_json=excluded.record_json, format_version=excluded.format_version,
                  content_digest=excluded.content_digest
                """,
                (shard_id, error_json, LEDGER_FORMAT_VERSION, content_digest(outcome["error_attribution"])),
            )
            for record in outcome["water_records"]:
                record_json = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                key = f"{record['scope_type']}:{record['scope_id']}"
                self.conn.execute(
                    """
                    INSERT INTO water_records(record_key, scope_type, scope_id, shard_id,
                      record_json, format_version, content_digest)
                    VALUES(?,?,?,?,?,?,?)
                    ON CONFLICT(record_key) DO UPDATE SET record_json=excluded.record_json,
                      content_digest=excluded.content_digest
                    """,
                    (
                        key,
                        record["scope_type"],
                        str(record["scope_id"]),
                        record.get("shard_id"),
                        record_json,
                        LEDGER_FORMAT_VERSION,
                        content_digest(record),
                    ),
                )
            self.conn.execute(
                """
                INSERT INTO outcome_commits(shard_id, format_version, content_digest,
                  payload_json, committed_at_ms)
                VALUES(?,?,?,?,?) ON CONFLICT(shard_id) DO UPDATE SET
                  format_version=excluded.format_version, content_digest=excluded.content_digest,
                  payload_json=excluded.payload_json, committed_at_ms=excluded.committed_at_ms
                """,
                (shard_id, LEDGER_FORMAT_VERSION, payload_digest, canonical_json(payload), now_ms),
            )
            completed = int(outcome["cursor"]["completed_count"])
            active_leases = len([l for l in self.list_leases(active_only=True)])
            active_resources = self.active_resource_sum()
            cursor_digest = content_digest(
                {
                    "last_event_seq": outcome["cursor"].get("last_event_seq", 0),
                    "last_topo_index": outcome["cursor"].get("last_topo_index"),
                    "last_shard_id": shard_id,
                    "completed_count": completed,
                    "active_lease_count": active_leases,
                    "active_resource_units": active_resources,
                }
            )
            self.conn.execute(
                """
                INSERT INTO progress_cursor(id, last_event_seq, last_topo_index, last_shard_id,
                  completed_count, active_lease_count, active_resource_units, format_version,
                  content_digest, updated_at_ms)
                VALUES(1,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET last_event_seq=excluded.last_event_seq,
                  last_topo_index=excluded.last_topo_index, last_shard_id=excluded.last_shard_id,
                  completed_count=excluded.completed_count,
                  active_lease_count=excluded.active_lease_count,
                  active_resource_units=excluded.active_resource_units,
                  format_version=excluded.format_version, content_digest=excluded.content_digest,
                  updated_at_ms=excluded.updated_at_ms
                """,
                (
                    int(outcome["cursor"].get("last_event_seq", 0)),
                    outcome["cursor"].get("last_topo_index"),
                    shard_id,
                    completed,
                    active_leases,
                    active_resources,
                    LEDGER_FORMAT_VERSION,
                    cursor_digest,
                    now_ms,
                ),
            )
        return conclusion_digest

    def record_invalidation(self, spec_digest: str, shard_id: str, kind: str, basis: dict[str, Any], now_ms: int) -> None:
        with self.lock, self.conn:
            self.conn.execute(
                "INSERT INTO invalidations(spec_digest, shard_id, kind, basis_json, created_at_ms) VALUES(?,?,?,?,?)",
                (spec_digest, shard_id, kind, canonical_json(basis), now_ms),
            )

    def mark_invalid_for_recompute(self, shard_id: str, reason: str, now_ms: int) -> None:
        with self.lock, self.conn:
            self.conn.execute(
                """
                UPDATE shards
                SET status='stale', error_code=?, updated_at_ms=?, content_digest=?
                WHERE shard_id=?
                """,
                (
                    reason,
                    now_ms,
                    content_digest({"shard_id": shard_id, "status": "stale", "reason": reason}),
                    shard_id,
                ),
            )

    def export_view(self) -> dict[str, list[dict[str, Any]]]:
        view: dict[str, list[dict[str, Any]]] = {}
        with self.lock:
            for table in EXPORT_TABLES:
                rows = self.conn.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
                view[table] = [dict(row) for row in rows]
        return view

    def canonical_view(self) -> str:
        return canonical_json(self.export_view())

    def close(self) -> None:  # noqa: F811
        with self.lock:
            self.conn.close()
