from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from . import LEDGER_FORMAT_VERSION
from .canonical import canonical_dumps, sha256_hex
from .errors import LedgerError


def monotonic_ms() -> int:
    return time.monotonic_ns() // 1_000_000


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ledger_header (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    format_version INTEGER NOT NULL,
    spec_digest TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    updated_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS shard_records (
    shard_id TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    node_id TEXT,
    topo_rank INTEGER,
    attempt INTEGER NOT NULL DEFAULT 1,
    wait_started_ms INTEGER,
    wait_ended_ms INTEGER,
    started_ms INTEGER,
    finished_ms INTEGER,
    demand_units INTEGER,
    demand_q INTEGER,
    active_q INTEGER NOT NULL DEFAULT 0,
    lease_json TEXT,
    lease_ownership_json TEXT,
    occupancy_json TEXT,
    watermark_json TEXT,
    error_json TEXT,
    summary_json TEXT,
    receipt_json TEXT,
    recovery_json TEXT,
    outcome_digest TEXT,
    updated_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS leases (
    lease_id TEXT PRIMARY KEY,
    shard_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    lease_version INTEGER NOT NULL,
    renew_count INTEGER NOT NULL,
    state TEXT NOT NULL,
    lease_json TEXT NOT NULL,
    updated_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS ledger_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    at_ms INTEGER NOT NULL,
    kind TEXT NOT NULL,
    shard_id TEXT,
    node_id TEXT,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS progress_cursor (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    last_completed_shard TEXT,
    completed_count INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    updated_ms INTEGER NOT NULL
);
"""


class Ledger:
    def __init__(self, path: str | Path, conn: sqlite3.Connection, replaced: bool = False) -> None:
        self.path = Path(path)
        self.conn = conn
        self.replaced = replaced

    @classmethod
    def open_for_session(
        cls,
        path: str | Path,
        *,
        spec_digest: str,
        replace_on_mismatch: bool = True,
    ) -> "Ledger":
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        existed = path.exists()
        replaced = False
        if existed:
            header = cls.read_header(path)
            mismatch = (
                header.get("ledger_format_version") != str(LEDGER_FORMAT_VERSION)
                or header.get("spec_digest") != spec_digest
            )
            if mismatch:
                if not replace_on_mismatch:
                    raise LedgerError(
                        "ledger version or spec digest mismatch",
                        field="ledger",
                        details={"header": header, "expected_spec_digest": spec_digest},
                    )
                for suffix in ("", "-wal", "-shm"):
                    stale = Path(str(path) + suffix)
                    if stale.exists() or stale.is_symlink():
                        stale.unlink()
                replaced = True

        conn = sqlite3.connect(str(path), timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.executescript(SCHEMA_SQL)
        ledger = cls(path, conn, replaced=replaced)
        ledger._initialize_meta(spec_digest)
        return ledger

    @staticmethod
    def read_header(path: str | Path) -> dict[str, str]:
        path = Path(path)
        if not path.exists():
            return {}
        try:
            conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
            try:
                rows = conn.execute("SELECT key, value FROM meta").fetchall()
                return {str(row[0]): str(row[1]) for row in rows}
            finally:
                conn.close()
        except sqlite3.DatabaseError as exc:
            raise LedgerError("cannot read SQLite ledger header", field="ledger") from exc

    def _initialize_meta(self, spec_digest: str) -> None:
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO ledger_header(id, format_version, spec_digest, content_digest, updated_ms)
                VALUES(1, ?, ?, ?, ?)
                """,
                (LEDGER_FORMAT_VERSION, spec_digest, sha256_hex(""), monotonic_ms()),
            )
            self.conn.execute(
                "INSERT OR IGNORE INTO meta(key,value) VALUES('ledger_format_version', ?)",
                (str(LEDGER_FORMAT_VERSION),),
            )
            self.conn.execute(
                "INSERT OR IGNORE INTO meta(key,value) VALUES('spec_digest', ?)",
                (spec_digest,),
            )
            self.conn.execute(
                "INSERT OR IGNORE INTO meta(key,value) VALUES('content_digest', ?)",
                (sha256_hex(""),),
            )
            self.conn.execute(
                "INSERT OR IGNORE INTO progress_cursor(id,last_completed_shard,completed_count,payload_json,updated_ms)"
                " VALUES(1,NULL,0,?,?)",
                (canonical_dumps({"phase": "initialized"}), monotonic_ms()),
            )
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def add_event(self, kind: str, payload: dict[str, Any], *, shard_id: str | None = None, node_id: str | None = None) -> None:
        self.conn.execute(
            "INSERT INTO ledger_events(at_ms,kind,shard_id,node_id,payload_json) VALUES(?,?,?,?,?)",
            (monotonic_ms(), kind, shard_id, node_id, canonical_dumps(payload)),
        )

    def begin_attempt(
        self,
        *,
        shard_id: str,
        node_id: str,
        topo_rank: int,
        attempt: int,
        demand_units: int,
        demand_q: int,
        wait_started_ms: int | None,
        wait_ended_ms: int | None,
        started_ms: int,
        lease: dict[str, Any],
        lease_ownership: dict[str, Any],
    ) -> None:
        lease_json = canonical_dumps(lease)
        ownership_json = canonical_dumps(lease_ownership)
        occupancy = {
            "state": "active",
            "demand_units": demand_units,
            "demand_q": demand_q,
            "active_q": demand_q,
        }
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.conn.execute(
                """
                INSERT INTO shard_records(
                    shard_id,state,node_id,topo_rank,attempt,wait_started_ms,wait_ended_ms,
                    started_ms,demand_units,demand_q,active_q,lease_json,lease_ownership_json,
                    occupancy_json,updated_ms
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(shard_id) DO UPDATE SET
                    state=excluded.state,node_id=excluded.node_id,topo_rank=excluded.topo_rank,
                    attempt=excluded.attempt,wait_started_ms=excluded.wait_started_ms,
                    wait_ended_ms=excluded.wait_ended_ms,started_ms=excluded.started_ms,
                    demand_units=excluded.demand_units,demand_q=excluded.demand_q,
                    active_q=excluded.active_q,lease_json=excluded.lease_json,
                    lease_ownership_json=excluded.lease_ownership_json,
                    occupancy_json=excluded.occupancy_json,updated_ms=excluded.updated_ms
                """,
                (
                    shard_id,
                    "running",
                    node_id,
                    topo_rank,
                    attempt,
                    wait_started_ms,
                    wait_ended_ms,
                    started_ms,
                    demand_units,
                    demand_q,
                    demand_q,
                    lease_json,
                    ownership_json,
                    canonical_dumps(occupancy),
                    monotonic_ms(),
                ),
            )
            self.conn.execute(
                """
                INSERT INTO leases(lease_id,shard_id,node_id,lease_version,renew_count,state,lease_json,updated_ms)
                VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(lease_id) DO UPDATE SET
                    shard_id=excluded.shard_id,node_id=excluded.node_id,
                    lease_version=excluded.lease_version,renew_count=excluded.renew_count,
                    state=excluded.state,lease_json=excluded.lease_json,updated_ms=excluded.updated_ms
                """,
                (
                    lease["lease_id"],
                    shard_id,
                    node_id,
                    lease["lease_version"],
                    lease["renew_count"],
                    "active",
                    lease_json,
                    monotonic_ms(),
                ),
            )
            self.add_event("attempt_begin", {"attempt": attempt}, shard_id=shard_id, node_id=node_id)
            self._update_content_digest_locked()
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise

    def commit_outcome(
        self,
        *,
        shard_id: str,
        lease: dict[str, Any],
        lease_ownership: dict[str, Any],
        occupancy: dict[str, Any],
        watermark: dict[str, Any],
        error_attribution: dict[str, Any],
        summary: dict[str, Any],
        receipt: dict[str, Any],
        wait_started_ms: int | None,
        wait_ended_ms: int | None,
        finished_ms: int,
        last_completed_shard: str,
        completed_count: int,
        cursor_payload: dict[str, Any],
    ) -> str:
        lease_json = canonical_dumps(lease)
        ownership_json = canonical_dumps(lease_ownership)
        occupancy_json = canonical_dumps(occupancy)
        watermark_json = canonical_dumps(watermark)
        error_json = canonical_dumps(error_attribution)
        summary_json = canonical_dumps(summary)
        receipt_json = canonical_dumps(receipt)
        outcome_digest = sha256_hex(
            {
                "shard_id": shard_id,
                "lease_ownership": lease_ownership,
                "occupancy": occupancy,
                "watermark": watermark,
                "error": error_attribution,
                "summary": summary,
            }
        )
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            cursor = self.conn.execute(
                """
                UPDATE shard_records SET
                    state='completed', wait_started_ms=?, wait_ended_ms=?, finished_ms=?,
                    active_q=0, lease_json=?, lease_ownership_json=?, occupancy_json=?,
                    watermark_json=?, error_json=?, summary_json=?, receipt_json=?,
                    outcome_digest=?, recovery_json=NULL, updated_ms=?
                WHERE shard_id=?
                """,
                (
                    wait_started_ms,
                    wait_ended_ms,
                    finished_ms,
                    lease_json,
                    ownership_json,
                    occupancy_json,
                    watermark_json,
                    error_json,
                    summary_json,
                    receipt_json,
                    outcome_digest,
                    monotonic_ms(),
                    shard_id,
                ),
            )
            if cursor.rowcount != 1:
                raise LedgerError("begin_attempt record is missing", field=shard_id)
            self.conn.execute(
                """
                INSERT INTO leases(lease_id,shard_id,node_id,lease_version,renew_count,state,lease_json,updated_ms)
                VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(lease_id) DO UPDATE SET
                    lease_version=excluded.lease_version,renew_count=excluded.renew_count,
                    state='completed',lease_json=excluded.lease_json,updated_ms=excluded.updated_ms
                """,
                (
                    lease["lease_id"],
                    shard_id,
                    lease["node_id"],
                    lease["lease_version"],
                    lease["renew_count"],
                    "completed",
                    lease_json,
                    monotonic_ms(),
                ),
            )
            self.add_event("outcome_committed", {"outcome_digest": outcome_digest}, shard_id=shard_id, node_id=lease["node_id"])
            self.conn.execute(
                """
                UPDATE progress_cursor SET last_completed_shard=?, completed_count=?, payload_json=?, updated_ms=?
                WHERE id=1
                """,
                (
                    last_completed_shard,
                    completed_count,
                    canonical_dumps(cursor_payload),
                    monotonic_ms(),
                ),
            )
            self._update_content_digest_locked()
            self.conn.commit()
            return outcome_digest
        except Exception:
            self.conn.rollback()
            raise

    def recover_incomplete(self) -> list[dict[str, Any]]:
        now = monotonic_ms()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            rows = self.conn.execute(
                "SELECT shard_id,node_id,attempt,lease_json FROM shard_records WHERE state IN ('running','unknown')"
            ).fetchall()
            reclaimed: list[dict[str, Any]] = []
            for row in rows:
                lease = json.loads(row["lease_json"]) if row["lease_json"] else None
                recovery = {
                    "classification": "unknown_uncommitted_outcome",
                    "action": "rerun_required",
                    "quota_reclaimed_before_scheduling": True,
                    "reclaimed_at_ms": now,
                    "previous_attempt": row["attempt"],
                    "old_lease_id": lease.get("lease_id") if lease else None,
                    "warning": "no force-termination exit report can be reconstructed",
                }
                self.conn.execute(
                    """
                    UPDATE shard_records
                    SET state='unknown', active_q=0, recovery_json=?, updated_ms=?
                    WHERE shard_id=?
                    """,
                    (canonical_dumps(recovery), now, row["shard_id"]),
                )
                if lease:
                    lease["recovery_state"] = "quota_reclaimed"
                    self.conn.execute(
                        "UPDATE leases SET state='reclaimed', lease_json=?, updated_ms=? WHERE lease_id=?",
                        (canonical_dumps(lease), now, lease["lease_id"]),
                    )
                self.add_event(
                    "quota_reclaimed",
                    recovery,
                    shard_id=row["shard_id"],
                    node_id=row["node_id"],
                )
                reclaimed.append({"shard_id": row["shard_id"], "node_id": row["node_id"], **recovery})
            cursor = {
                "phase": "recovery",
                "reclaimed_shards": [item["shard_id"] for item in reclaimed],
            }
            self.conn.execute(
                "UPDATE progress_cursor SET payload_json=?, updated_ms=? WHERE id=1",
                (canonical_dumps(cursor), now),
            )
            self._update_content_digest_locked()
            self.conn.commit()
            return reclaimed
        except Exception:
            self.conn.rollback()
            raise

    def _update_content_digest_locked(self) -> str:
        rows = self.conn.execute(
            "SELECT shard_id,state,outcome_digest,recovery_json,updated_ms FROM shard_records ORDER BY topo_rank,shard_id"
        ).fetchall()
        material = []
        for row in rows:
            material.append(
                {
                    "shard_id": row["shard_id"],
                    "state": row["state"],
                    "outcome_digest": row["outcome_digest"],
                    "recovery_json": row["recovery_json"],
                }
            )
        digest = sha256_hex(material)
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES('content_digest',?)",
            (digest,),
        )
        self.conn.execute(
            "UPDATE ledger_header SET content_digest=?, updated_ms=? WHERE id=1",
            (digest, monotonic_ms()),
        )
        return digest

    def verify_digest(self) -> dict[str, Any]:
        header = self.read_header(self.path)
        stored = header.get("content_digest")
        recomputed = self._update_content_digest_locked()
        self.conn.commit()
        return {"stored_content_digest": stored, "recomputed_content_digest": recomputed, "ok": stored == recomputed}

    def completed_shard_ids(self) -> set[str]:
        rows = self.conn.execute("SELECT shard_id FROM shard_records WHERE state='completed'").fetchall()
        return {str(row["shard_id"]) for row in rows}

    def copy_completed_records_from(self, source: "Ledger", shard_ids: list[str] | tuple[str, ...] | set[str]) -> dict[str, Any]:
        copied: list[str] = []
        lease_copied = 0
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            for shard_id in sorted(shard_ids):
                row = source.conn.execute(
                    "SELECT * FROM shard_records WHERE shard_id=? AND state='completed'",
                    (shard_id,),
                ).fetchone()
                if row is None:
                    raise LedgerError(
                        "cannot preserve missing completed record",
                        field=shard_id,
                    )
                columns = row.keys()
                column_sql = ",".join(columns)
                placeholder_sql = ",".join("?" for _ in columns)
                self.conn.execute(
                    f"INSERT OR REPLACE INTO shard_records({column_sql}) VALUES({placeholder_sql})",
                    tuple(row[column] for column in columns),
                )
                lease_rows = source.conn.execute(
                    "SELECT * FROM leases WHERE shard_id=?",
                    (shard_id,),
                ).fetchall()
                for lease_row in lease_rows:
                    lease_columns = lease_row.keys()
                    lease_sql = ",".join(lease_columns)
                    lease_placeholders = ",".join("?" for _ in lease_columns)
                    self.conn.execute(
                        f"INSERT OR REPLACE INTO leases({lease_sql}) VALUES({lease_placeholders})",
                        tuple(lease_row[column] for column in lease_columns),
                    )
                    lease_copied += 1
                copied.append(shard_id)
            self._update_content_digest_locked()
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        return {"copied_shard_ids": copied, "copied_lease_rows": lease_copied}

    def records(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM shard_records ORDER BY topo_rank,shard_id").fetchall()
        return [dict(row) for row in rows]

    def snapshot(self) -> dict[str, Any]:
        return {
            "ledger_format_version": LEDGER_FORMAT_VERSION,
            "header": self.read_header(self.path),
            "shards": self.records(),
            "events": [dict(row) for row in self.conn.execute("SELECT * FROM ledger_events ORDER BY event_id")],
            "cursor": dict(self.conn.execute("SELECT * FROM progress_cursor WHERE id=1").fetchone()),
        }


def replace_json_snapshot_if_stale(
    path: str | Path,
    *,
    expected_version: int,
    spec_digest: str,
    fresh_payload: dict[str, Any],
) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            current = {}
        if current.get("snapshot_format_version") != expected_version or current.get("spec_digest") != spec_digest:
            replacement = {
                "snapshot_format_version": expected_version,
                "spec_digest": spec_digest,
                "replaced_due_to": "version_or_digest_mismatch",
                "payload": fresh_payload,
            }
            path.write_text(canonical_dumps(replacement) + "\n", encoding="utf-8")
            return {"ok": True, "replaced": True, "path": str(path)}
        return {"ok": True, "replaced": False, "path": str(path)}
    payload = {
        "snapshot_format_version": expected_version,
        "spec_digest": spec_digest,
        "replaced_due_to": "new_snapshot",
        "payload": fresh_payload,
    }
    path.write_text(canonical_dumps(payload) + "\n", encoding="utf-8")
    return {"ok": True, "replaced": True, "path": str(path)}
