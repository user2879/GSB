from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from . import LEDGER_FORMAT_VERSION
from .canonical import canonical_dumps
from .errors import LeaseError
from .timeutil import monotonic_ms


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
    occurred_ms INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    quota_version INTEGER NOT NULL,
    node_id TEXT,
    shard_id TEXT,
    lease_id TEXT,
    decision_basis TEXT,
    payload_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS leases (
    lease_id TEXT PRIMARY KEY,
    quota_version INTEGER NOT NULL,
    lease_version INTEGER NOT NULL,
    node_id TEXT NOT NULL,
    shard_id TEXT NOT NULL,
    slot_index INTEGER NOT NULL,
    byte_count INTEGER NOT NULL,
    granted_ms INTEGER NOT NULL,
    expires_ms INTEGER NOT NULL,
    renewal_count INTEGER NOT NULL,
    state TEXT NOT NULL,
    decision_basis TEXT,
    terminal_event_seq INTEGER,
    updated_event_seq INTEGER NOT NULL,
    stale INTEGER NOT NULL DEFAULT 0
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_active_lease_shard
    ON leases(quota_version, node_id, shard_id) WHERE state='active';
CREATE UNIQUE INDEX IF NOT EXISTS idx_active_lease_slot
    ON leases(quota_version, node_id, slot_index) WHERE state='active';
CREATE UNIQUE INDEX IF NOT EXISTS idx_active_lease_shard_global
    ON leases(shard_id) WHERE state='active';
CREATE TABLE IF NOT EXISTS slots (
    quota_version INTEGER NOT NULL,
    node_id TEXT NOT NULL,
    slot_index INTEGER NOT NULL,
    lease_id TEXT,
    shard_id TEXT,
    state TEXT NOT NULL,
    updated_event_seq INTEGER NOT NULL,
    PRIMARY KEY(quota_version, node_id, slot_index)
);
CREATE TABLE IF NOT EXISTS budget_state (
    budget_id INTEGER PRIMARY KEY CHECK (budget_id = 1),
    quota_version INTEGER NOT NULL,
    bytes_reserved INTEGER NOT NULL,
    bytes_verified INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS capacity_margins (
    margin_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_seq INTEGER NOT NULL,
    quota_version INTEGER NOT NULL,
    scope_node TEXT NOT NULL,
    capacity_slots INTEGER NOT NULL,
    active_leases INTEGER NOT NULL,
    available_slots INTEGER NOT NULL,
    global_active INTEGER NOT NULL,
    global_max_active INTEGER NOT NULL,
    bytes_reserved INTEGER NOT NULL,
    bytes_verified INTEGER NOT NULL,
    byte_budget INTEGER NOT NULL,
    stale INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS progress_cursor (
    cursor_id INTEGER PRIMARY KEY CHECK (cursor_id = 1),
    last_event_seq INTEGER NOT NULL,
    updated_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS shard_results (
    quota_version INTEGER NOT NULL,
    node_id TEXT NOT NULL,
    shard_id TEXT NOT NULL,
    shard_seq INTEGER NOT NULL,
    digest TEXT,
    content_bytes_digest TEXT,
    proof_json TEXT,
    source_format TEXT NOT NULL,
    status TEXT NOT NULL,
    unknown INTEGER NOT NULL DEFAULT 0,
    event_seq INTEGER NOT NULL,
    stale INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(quota_version, node_id, shard_id)
);
CREATE TABLE IF NOT EXISTS adjudications (
    event_seq INTEGER PRIMARY KEY,
    lease_id TEXT NOT NULL,
    winner TEXT NOT NULL,
    basis TEXT NOT NULL,
    quota_version INTEGER NOT NULL,
    old_quota_version INTEGER,
    competing_action TEXT
);
CREATE TABLE IF NOT EXISTS receipts (
    receipt_id INTEGER PRIMARY KEY AUTOINCREMENT,
    receipt_key TEXT NOT NULL,
    quota_version INTEGER NOT NULL,
    node_id TEXT NOT NULL,
    shard_id TEXT NOT NULL,
    source_format TEXT NOT NULL,
    source_line INTEGER NOT NULL,
    normalized_hash TEXT NOT NULL,
    canonical_json TEXT NOT NULL,
    adopted INTEGER NOT NULL,
    current INTEGER NOT NULL DEFAULT 0,
    reason TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    stale INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_receipt_key ON receipts(receipt_key, receipt_id);
"""


class Ledger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA busy_timeout=30000")
        self._init_schema()

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        content_digest: str,
        format_version: int = LEDGER_FORMAT_VERSION,
        backup_dir: str | Path | None = None,
    ) -> "Ledger":
        p = Path(path)
        replaced = False
        backup_path: Path | None = None
        if p.exists():
            mismatch = True
            try:
                check = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
                rows = dict(check.execute("SELECT key,value FROM meta").fetchall())
                check.close()
                mismatch = (
                    rows.get("ledger_format_version") != str(format_version)
                    or rows.get("content_digest") != content_digest
                )
            except (sqlite3.DatabaseError, OSError):
                mismatch = True
            if mismatch:
                destination_dir = Path(backup_dir) if backup_dir is not None else p.parent / "backups"
                destination_dir.mkdir(parents=True, exist_ok=True)
                stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 1_000_000:06d}"
                backup_path = destination_dir / f"{p.stem}-{stamp}.db"
                os.replace(p, backup_path)
                for suffix in ("-wal", "-shm"):
                    side = p.with_name(p.name + suffix)
                    if side.exists():
                        try:
                            os.replace(side, backup_path.with_name(backup_path.name + suffix))
                        except OSError:
                            pass
                backup_path.chmod(0o444)
                replaced = True
        ledger = cls(p)
        with ledger.lock, ledger.conn:
            ledger.conn.execute(
                "INSERT OR IGNORE INTO meta(key,value) VALUES('ledger_format_version',?)",
                (str(format_version),),
            )
            ledger.conn.execute(
                "INSERT OR IGNORE INTO meta(key,value) VALUES('content_digest',?)",
                (content_digest,),
            )
            ledger.conn.execute(
                "INSERT OR IGNORE INTO meta(key,value) VALUES('opened_ms',?)",
                (str(monotonic_ms()),),
            )
            ledger.conn.execute(
                "INSERT OR IGNORE INTO budget_state(budget_id,quota_version,bytes_reserved,bytes_verified) VALUES(1,0,0,0)",
            )
        if replaced and backup_path is not None:
            ledger.append_event(
                "snapshot_replaced",
                quota_version=0,
                payload={
                    "reason": "format_version_or_content_digest_mismatch",
                    "backup_path": str(backup_path),
                    "expected_format_version": format_version,
                    "expected_content_digest": content_digest,
                },
                basis="meta_version_or_digest_mismatch:whole_snapshot_replaced",
            )
        return ledger

    def _init_schema(self) -> None:
        with self.lock, self.conn:
            self.conn.executescript(SCHEMA_SQL)
            self._migrate_columns()

    def _migrate_columns(self) -> None:
        additions = {
            "leases": [("stale", "INTEGER NOT NULL DEFAULT 0")],
            "capacity_margins": [("stale", "INTEGER NOT NULL DEFAULT 0")],
            "shard_results": [("stale", "INTEGER NOT NULL DEFAULT 0")],
            "receipts": [("stale", "INTEGER NOT NULL DEFAULT 0")],
        }
        for table, columns in additions.items():
            existing = {row["name"] for row in self.conn.execute(f"PRAGMA table_info({table})")}
            for name, declaration in columns:
                if name not in existing:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")

    def mark_quota_stale(self, quota_version: int) -> dict[str, int]:
        counts: dict[str, int] = {}
        with self.lock, self.conn:
            for table in ("leases", "capacity_margins", "shard_results", "receipts"):
                cur = self.conn.execute(
                    f"UPDATE {table} SET stale=1 WHERE quota_version<=? AND stale=0",
                    (quota_version,),
                )
                counts[table] = cur.rowcount
            return counts

    def close(self) -> None:
        with self.lock:
            self.conn.close()

    def meta(self) -> dict[str, str]:
        with self.lock:
            return {row["key"]: row["value"] for row in self.conn.execute("SELECT key,value FROM meta")}

    def append_event(
        self,
        event_type: str,
        *,
        quota_version: int,
        occurred_ms: int | None = None,
        node_id: str | None = None,
        shard_id: str | None = None,
        lease_id: str | None = None,
        basis: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> int:
        occurred = monotonic_ms() if occurred_ms is None else occurred_ms
        with self.lock, self.conn:
            cur = self.conn.execute(
                """INSERT INTO events(occurred_ms,event_type,quota_version,node_id,shard_id,lease_id,
                   decision_basis,payload_json) VALUES(?,?,?,?,?,?,?,?)""",
                (
                    occurred,
                    event_type,
                    quota_version,
                    node_id,
                    shard_id,
                    lease_id,
                    basis,
                    canonical_dumps(payload or {}),
                ),
            )
            seq = int(cur.lastrowid)
            self.conn.execute(
                """INSERT INTO progress_cursor(cursor_id,last_event_seq,updated_ms)
                   VALUES(1,?,?) ON CONFLICT(cursor_id) DO UPDATE SET
                   last_event_seq=excluded.last_event_seq,updated_ms=excluded.updated_ms""",
                (seq, occurred),
            )
            return seq

    def _write_margins(self, seq: int, quota_version: int, policy: Any, nodes: Iterable[Any]) -> None:
        row = self.conn.execute("SELECT bytes_reserved,bytes_verified FROM budget_state WHERE budget_id=1").fetchone()
        reserved = int(row["bytes_reserved"])
        verified = int(row["bytes_verified"])
        global_active = int(
            self.conn.execute("SELECT COUNT(*) AS c FROM leases WHERE state='active'").fetchone()["c"]
        )
        for node in nodes:
            active = int(
                self.conn.execute(
                    "SELECT COUNT(*) AS c FROM leases WHERE state='active' AND node_id=?",
                    (node.node_id,),
                ).fetchone()["c"]
            )
            self.conn.execute(
                """INSERT INTO capacity_margins(event_seq,quota_version,scope_node,capacity_slots,
                   active_leases,available_slots,global_active,global_max_active,bytes_reserved,
                   bytes_verified,byte_budget) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    seq,
                    quota_version,
                    node.node_id,
                    node.capacity_slots,
                    active,
                    max(0, node.capacity_slots - active),
                    global_active,
                    policy.global_max_active_leases,
                    reserved,
                    verified,
                    policy.global_byte_budget,
                ),
            )

    def record_grant(self, lease: Any, node: Any, policy: Any) -> int:
        with self.lock, self.conn:
            event_seq = self._insert_event_locked(
                event_type="lease_granted",
                quota_version=lease.quota_version,
                occurred_ms=lease.granted_ms,
                node_id=lease.node_id,
                shard_id=lease.shard_id,
                lease_id=lease.lease_id,
                basis="capacity_and_budget_reserved:monotonic_grant",
                payload=lease.to_dict(),
            )
            self.conn.execute(
                """INSERT INTO leases(lease_id,quota_version,lease_version,node_id,shard_id,slot_index,
                   byte_count,granted_ms,expires_ms,renewal_count,state,decision_basis,
                   terminal_event_seq,updated_event_seq)
                   VALUES(?,?,?,?,?,?,?,?,?,?, 'active', NULL, NULL, ?)""",
                (
                    lease.lease_id,
                    lease.quota_version,
                    lease.lease_version,
                    lease.node_id,
                    lease.shard_id,
                    lease.slot_index,
                    lease.byte_count,
                    lease.granted_ms,
                    lease.expires_ms,
                    lease.renewal_count,
                    event_seq,
                ),
            )
            self.conn.execute(
                """INSERT INTO slots(quota_version,node_id,slot_index,lease_id,shard_id,state,updated_event_seq)
                   VALUES(?,?,?,?,?, 'occupied', ?)
                   ON CONFLICT(quota_version,node_id,slot_index) DO UPDATE SET
                   lease_id=excluded.lease_id,shard_id=excluded.shard_id,
                   state='occupied',updated_event_seq=excluded.updated_event_seq""",
                (lease.quota_version, lease.node_id, lease.slot_index, lease.lease_id, lease.shard_id, event_seq),
            )
            self.conn.execute(
                "UPDATE budget_state SET quota_version=?, bytes_reserved=bytes_reserved+? WHERE budget_id=1",
                (lease.quota_version, lease.byte_count),
            )
            self._write_margins(event_seq, lease.quota_version, policy, [node])
            self._update_cursor(event_seq, lease.granted_ms)
            return event_seq

    def _insert_event_locked(self, *args: Any, **kwargs: Any) -> int:
        cur = self.conn.execute(
            """INSERT INTO events(occurred_ms,event_type,quota_version,node_id,shard_id,lease_id,
               decision_basis,payload_json) VALUES(?,?,?,?,?,?,?,?)""",
            (
                kwargs.get("occurred_ms", monotonic_ms()),
                kwargs["event_type"] if False else kwargs.get("event_type", "event"),
                kwargs["quota_version"],
                kwargs.get("node_id"),
                kwargs.get("shard_id"),
                kwargs.get("lease_id"),
                kwargs.get("basis"),
                canonical_dumps(kwargs.get("payload") or {}),
            ),
        )
        return int(cur.lastrowid)

    def _update_cursor(self, seq: int, occurred_ms: int) -> None:
        self.conn.execute(
            """INSERT INTO progress_cursor(cursor_id,last_event_seq,updated_ms) VALUES(1,?,?)
               ON CONFLICT(cursor_id) DO UPDATE SET last_event_seq=excluded.last_event_seq,
               updated_ms=excluded.updated_ms""",
            (seq, occurred_ms),
        )

    def record_renewal(
        self,
        *,
        lease_id: str,
        expected_version: int,
        new_version: int,
        renewal_count: int,
        new_expires_ms: int,
        occurred_ms: int,
        policy: Any,
        node: Any,
    ) -> int:
        with self.lock, self.conn:
            cur = self.conn.execute(
                """UPDATE leases SET lease_version=?, renewal_count=?, expires_ms=?, updated_event_seq=?
                   WHERE lease_id=? AND lease_version=? AND state='active'""",
                (new_version, renewal_count, new_expires_ms, -1, lease_id, expected_version),
            )
            if cur.rowcount != 1:
                row = self.conn.execute("SELECT lease_version,state FROM leases WHERE lease_id=?", (lease_id,)).fetchone()
                actual = dict(row) if row else None
                raise LeaseError(
                    f"optimistic renewal rejected for {lease_id}: expected v{expected_version}, actual={actual}",
                    field="lease_version",
                )
            seq = self._insert_event_locked(
                event_type="lease_renewed",
                quota_version=node and self._lease_quota(lease_id),
                occurred_ms=occurred_ms,
                node_id=node.node_id,
                lease_id=lease_id,
                basis="node_optimistic_version_match+monotonic_renew",
                payload={
                    "expected_version": expected_version,
                    "new_version": new_version,
                    "renewal_count": renewal_count,
                    "expires_ms": new_expires_ms,
                },
            )
            self.conn.execute("UPDATE leases SET updated_event_seq=? WHERE lease_id=?", (seq, lease_id))
            self._write_margins(seq, self._lease_quota(lease_id), policy, [node])
            self._update_cursor(seq, occurred_ms)
            return seq

    def _lease_quota(self, lease_id: str) -> int:
        row = self.conn.execute("SELECT quota_version FROM leases WHERE lease_id=?", (lease_id,)).fetchone()
        return int(row["quota_version"]) if row else 0

    def terminalize_lease(
        self,
        lease_id: str,
        *,
        new_state: str,
        event_type: str,
        winner: str,
        basis: str,
        occurred_ms: int,
        policy: Any,
        node: Any,
        event_quota_version: int | None = None,
        old_quota_version: int | None = None,
        competing_action: str | None = None,
        extra_payload: dict[str, Any] | None = None,
        mark_unknown: bool = False,
    ) -> int | None:
        with self.lock, self.conn:
            lease_row = self.conn.execute("SELECT * FROM leases WHERE lease_id=?", (lease_id,)).fetchone()
            if lease_row is None:
                return None
            quota_version = int(lease_row["quota_version"])
            cur = self.conn.execute(
                """UPDATE leases SET state=?, decision_basis=?, terminal_event_seq=?
                   WHERE lease_id=? AND state='active'""",
                (new_state, basis, -1, lease_id),
            )
            if cur.rowcount != 1:
                return None
            event_quota = quota_version if event_quota_version is None else event_quota_version
            payload = dict(extra_payload or {})
            payload.update(
                {
                    "winner": winner,
                    "basis": basis,
                    "lease_quota_version": quota_version,
                    "slot_index": lease_row["slot_index"],
                }
            )
            seq = self._insert_event_locked(
                event_type=event_type,
                quota_version=event_quota,
                occurred_ms=occurred_ms,
                node_id=lease_row["node_id"],
                shard_id=lease_row["shard_id"],
                lease_id=lease_id,
                basis=basis,
                payload=payload,
            )
            self.conn.execute(
                "UPDATE leases SET terminal_event_seq=?, updated_event_seq=? WHERE lease_id=?",
                (seq, seq, lease_id),
            )
            self.conn.execute(
                """UPDATE slots SET state='free', lease_id=NULL, shard_id=NULL, updated_event_seq=?
                   WHERE quota_version=? AND node_id=? AND slot_index=?""",
                (seq, quota_version, lease_row["node_id"], lease_row["slot_index"]),
            )
            self.conn.execute(
                "UPDATE budget_state SET bytes_reserved=MAX(0, bytes_reserved-?) WHERE budget_id=1",
                (int(lease_row["byte_count"]),),
            )
            self.conn.execute(
                """INSERT INTO adjudications(event_seq,lease_id,winner,basis,quota_version,
                   old_quota_version,competing_action) VALUES(?,?,?,?,?,?,?)""",
                (seq, lease_id, winner, basis, event_quota, old_quota_version, competing_action),
            )
            if mark_unknown:
                self.conn.execute(
                    """INSERT INTO shard_results(quota_version,node_id,shard_id,shard_seq,digest,
                       content_bytes_digest,proof_json,source_format,status,unknown,event_seq)
                       VALUES(?,?,?,?,NULL,NULL,NULL,'recovery','unknown',1,?)
                       ON CONFLICT(quota_version,node_id,shard_id) DO UPDATE SET
                       status='unknown',unknown=1,digest=NULL,content_bytes_digest=NULL,
                       proof_json=NULL,source_format='recovery',event_seq=excluded.event_seq""",
                    (
                        quota_version,
                        lease_row["node_id"],
                        lease_row["shard_id"],
                        _seq_from_shard_id(lease_row["shard_id"]),
                        seq,
                    ),
                )
            self._write_margins(seq, event_quota, policy, [node])
            self._update_cursor(seq, occurred_ms)
            return seq

    def commit_result(
        self,
        lease: Any,
        *,
        proof_json: str,
        digest: str,
        content_digest: str,
        source_format: str,
        occurred_ms: int,
        policy: Any,
        node: Any,
    ) -> int | None:
        with self.lock, self.conn:
            row = self.conn.execute(
                "SELECT * FROM leases WHERE lease_id=? AND state='active' AND lease_version=?",
                (lease.lease_id, lease.lease_version),
            ).fetchone()
            if row is None:
                return None
            seq = self._insert_event_locked(
                event_type="shard_verified",
                quota_version=lease.quota_version,
                occurred_ms=occurred_ms,
                node_id=lease.node_id,
                shard_id=lease.shard_id,
                lease_id=lease.lease_id,
                basis="verification_committed",
                payload={"digest": digest, "content_bytes_digest": content_digest, "format": source_format},
            )
            self.conn.execute(
                """INSERT INTO shard_results(quota_version,node_id,shard_id,shard_seq,digest,
                   content_bytes_digest,proof_json,source_format,status,unknown,event_seq)
                   VALUES(?,?,?,?,?,?,?,?, 'verified',0,?)
                   ON CONFLICT(quota_version,node_id,shard_id) DO UPDATE SET
                   digest=excluded.digest,content_bytes_digest=excluded.content_bytes_digest,
                   proof_json=excluded.proof_json,source_format=excluded.source_format,
                   status='verified',unknown=0,event_seq=excluded.event_seq""",
                (
                    lease.quota_version,
                    lease.node_id,
                    lease.shard_id,
                    lease.shard_seq,
                    digest,
                    content_digest,
                    proof_json,
                    source_format,
                    seq,
                ),
            )
            self.conn.execute(
                "UPDATE budget_state SET bytes_reserved=MAX(0, bytes_reserved-?), bytes_verified=bytes_verified+? WHERE budget_id=1",
                (lease.byte_count, lease.byte_count),
            )
            self.conn.execute(
                """UPDATE leases SET state='completed', decision_basis='verification_committed',
                   terminal_event_seq=?, updated_event_seq=? WHERE lease_id=?""",
                (seq, seq, lease.lease_id),
            )
            self.conn.execute(
                """UPDATE slots SET state='free', lease_id=NULL, shard_id=NULL, updated_event_seq=?
                   WHERE quota_version=? AND node_id=? AND slot_index=?""",
                (seq, lease.quota_version, lease.node_id, lease.slot_index),
            )
            self.conn.execute(
                """INSERT INTO adjudications(event_seq,lease_id,winner,basis,quota_version)
                   VALUES(?,?, 'completed', 'verification_committed', ?)""",
                (seq, lease.lease_id, lease.quota_version),
            )
            self._write_margins(seq, lease.quota_version, policy, [node])
            self._update_cursor(seq, occurred_ms)
            return seq

    def record_quota_revision(
        self, old_version: int, new_version: int, occurred_ms: int, reason: str, node_ids: list[str]
    ) -> int:
        return self.append_event(
            "quota_revision",
            quota_version=new_version,
            occurred_ms=occurred_ms,
            basis=reason,
            payload={"old_quota_version": old_version, "new_quota_version": new_version, "node_ids": node_ids},
        )

    def insert_receipt(self, receipt: Any, adopted: bool, reason: str) -> int:
        canonical = canonical_dumps(receipt.to_canonical())
        key = f"{receipt.quota_version}|{receipt.node_id}|{receipt.shard_id}"
        with self.lock, self.conn:
            if adopted:
                self.conn.execute("UPDATE receipts SET current=0 WHERE receipt_key=? AND adopted=1", (key,))
            cur = self.conn.execute(
                """INSERT INTO receipts(receipt_key,quota_version,node_id,shard_id,source_format,
                   source_line,normalized_hash,canonical_json,adopted,current,reason,created_ms)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    key,
                    receipt.quota_version,
                    receipt.node_id or "",
                    receipt.shard_id,
                    receipt.source_format,
                    receipt.source_line,
                    receipt.normalized_hash,
                    canonical,
                    1 if adopted else 0,
                    1 if adopted else 0,
                    reason,
                    monotonic_ms(),
                ),
            )
            return int(cur.lastrowid)

    def replace_receipt(self, receipt_id: int, replacement: Any, adopted: bool, reason: str) -> int:
        with self.lock, self.conn:
            row = self.conn.execute("SELECT receipt_key FROM receipts WHERE receipt_id=?", (receipt_id,)).fetchone()
            if row is None:
                raise KeyError(receipt_id)
            self.conn.execute("UPDATE receipts SET current=0 WHERE receipt_id=?", (receipt_id,))
            return self.insert_receipt(replacement, adopted, reason + ":replaced_receipt_id=" + str(receipt_id))

    def active_lease_rows(self) -> list[sqlite3.Row]:
        with self.lock:
            return list(self.conn.execute("SELECT * FROM leases WHERE state='active' ORDER BY lease_id"))

    def incomplete_lease_rows(self) -> list[sqlite3.Row]:
        return self.active_lease_rows()

    def fetch_all(self, table: str) -> list[sqlite3.Row]:
        allowed = {
            "events",
            "leases",
            "slots",
            "capacity_margins",
            "progress_cursor",
            "shard_results",
            "adjudications",
            "receipts",
            "budget_state",
        }
        if table not in allowed:
            raise ValueError(f"table {table} is not replay-readable")
        with self.lock:
            return list(self.conn.execute(f"SELECT * FROM {table} ORDER BY 1"))

    def latest_cursor(self) -> sqlite3.Row | None:
        with self.lock:
            return self.conn.execute("SELECT * FROM progress_cursor WHERE cursor_id=1").fetchone()


def _seq_from_shard_id(shard_id: str) -> int:
    try:
        return int(shard_id.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return -1
