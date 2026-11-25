"""SQLite ledger with atomic journals, checkpoints and crash recovery."""

from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .content import canonical_json, manifest_digest
from .model import Manifest
from .protocol import monotonic_ms

SCHEMA_VERSION = 1


class Ledger:
    def __init__(self, db_path: str | Path, manifest: Manifest) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest = manifest
        self.digest = manifest_digest(manifest)
        self.lock = threading.RLock()
        self.reinitialized = False
        self.reinitialize_reason: str | None = None
        self.recovered_unknown: list[dict[str, Any]] = []
        self._prepare_database()
        self.conn = sqlite3.connect(
            str(self.path),
            timeout=15.0,
            check_same_thread=False,
            isolation_level=None,
        )
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA busy_timeout=15000")
        self._create_schema()
        self._bind_or_reinitialize()
        self.recovered_unknown = self.recover_open_batches()

    def _prepare_database(self) -> None:
        if not self.path.exists():
            return
        probe: sqlite3.Connection | None = None
        mismatch_reason: str | None = None
        try:
            probe = sqlite3.connect(str(self.path), timeout=2.0)
            row = probe.execute("SELECT value FROM meta WHERE key='manifest_digest'").fetchone()
            version = probe.execute("SELECT value FROM meta WHERE key='format_version'").fetchone()
            if row is not None and version is not None and (row[0] != self.digest or version[0] != str(SCHEMA_VERSION)):
                mismatch_reason = "format_version or manifest_digest changed"
        except sqlite3.DatabaseError:
            mismatch_reason = "existing ledger is not a compatible SQLite ledger"
        finally:
            if probe is not None:
                probe.close()
        if mismatch_reason is not None:
            self.reinitialize_reason = mismatch_reason
            self._archive_and_remove()

    def _archive_and_remove(self) -> None:
        suffix = time.strftime("%Y%m%dT%H%M%S", time.localtime()) + f"-{time.time_ns() % 1_000_000_000:09d}"
        archive = self.path.with_name(self.path.name + f".stale-{suffix}")
        try:
            shutil.copy2(self.path, archive)
        except OSError:
            pass
        for candidate in (self.path, Path(str(self.path) + "-wal"), Path(str(self.path) + "-shm")):
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass
        self.reinitialized = True

    def _create_schema(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            format_version INTEGER NOT NULL,
            current_quota_version INTEGER NOT NULL,
            active_nodes_json TEXT NOT NULL,
            progress_cursor INTEGER NOT NULL,
            requires_rerun INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS journal (
            ledger_seq INTEGER PRIMARY KEY AUTOINCREMENT,
            event_seq INTEGER NOT NULL,
            kind TEXT NOT NULL,
            lease_id TEXT,
            node_id TEXT,
            payload_json TEXT NOT NULL,
            format_version INTEGER NOT NULL,
            content_digest TEXT NOT NULL,
            mono_ms INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS checkpoints (
            ledger_seq INTEGER PRIMARY KEY REFERENCES journal(ledger_seq) ON DELETE CASCADE,
            event_seq INTEGER NOT NULL,
            progress_cursor INTEGER NOT NULL,
            capacity_json TEXT NOT NULL,
            node_leases_json TEXT NOT NULL,
            format_version INTEGER NOT NULL,
            content_digest TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS batches (
            batch_id TEXT PRIMARY KEY,
            event_seq INTEGER NOT NULL,
            node_id TEXT NOT NULL,
            status TEXT NOT NULL,
            created_ms INTEGER NOT NULL,
            closed_ms INTEGER,
            request_json TEXT NOT NULL,
            error_code TEXT,
            attempt INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS leases (
            lease_id TEXT PRIMARY KEY,
            queue_id TEXT,
            batch_id TEXT,
            node_id TEXT NOT NULL,
            content_key TEXT NOT NULL,
            shard_index INTEGER NOT NULL,
            slot_index INTEGER NOT NULL,
            bytes INTEGER NOT NULL,
            granted_ms INTEGER NOT NULL,
            expires_ms INTEGER NOT NULL,
            lease_version INTEGER NOT NULL,
            quota_version INTEGER NOT NULL,
            digest TEXT NOT NULL,
            checksum TEXT NOT NULL,
            status TEXT NOT NULL,
            claim_kind TEXT,
            claim_basis TEXT,
            claim_event_seq INTEGER,
            claim_quota_version INTEGER,
            attempt INTEGER NOT NULL DEFAULT 0,
            last_ledger_seq INTEGER
        );
        CREATE TABLE IF NOT EXISTS current_view (
            lease_id TEXT PRIMARY KEY,
            node_id TEXT NOT NULL,
            content_key TEXT NOT NULL,
            shard_index INTEGER NOT NULL,
            slot_index INTEGER NOT NULL,
            bytes INTEGER NOT NULL,
            expires_ms INTEGER NOT NULL,
            lease_version INTEGER NOT NULL,
            quota_version INTEGER NOT NULL,
            digest TEXT NOT NULL,
            status TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS slot_ownership (
            lease_id TEXT PRIMARY KEY,
            node_id TEXT NOT NULL,
            slot_index INTEGER NOT NULL,
            active INTEGER NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS slot_ownership_active
            ON slot_ownership(node_id, slot_index) WHERE active = 1;
        CREATE TABLE IF NOT EXISTS queue_waits (
            queue_id TEXT PRIMARY KEY,
            event_seq INTEGER NOT NULL,
            node_id TEXT NOT NULL,
            content_key TEXT NOT NULL,
            shard_index INTEGER NOT NULL,
            wait_start_ms INTEGER NOT NULL,
            wait_end_ms INTEGER,
            status TEXT NOT NULL,
            quota_version INTEGER NOT NULL DEFAULT 1,
            attempt INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS slot_claims (
            lease_id TEXT PRIMARY KEY,
            node_id TEXT NOT NULL,
            slot_index INTEGER NOT NULL,
            claim_kind TEXT NOT NULL,
            basis TEXT NOT NULL,
            decision_event_seq INTEGER NOT NULL,
            quota_version INTEGER NOT NULL,
            mono_ms INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS quota_history (
            quota_version INTEGER PRIMARY KEY,
            event_seq INTEGER NOT NULL,
            node_ids_json TEXT NOT NULL,
            added_nodes_json TEXT NOT NULL,
            removed_nodes_json TEXT NOT NULL,
            new_leases_json TEXT NOT NULL,
            revoked_leases_json TEXT NOT NULL,
            retained_leases_json TEXT NOT NULL,
            created_ms INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS stale_views (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            quota_version INTEGER NOT NULL,
            reason TEXT NOT NULL,
            event_seq INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            created_ms INTEGER NOT NULL
        );
        """
        self.conn.executescript(schema)
        columns = {row["name"] for row in self.conn.execute("PRAGMA table_info(queue_waits)").fetchall()}
        if "quota_version" not in columns:
            self.conn.execute(
                "ALTER TABLE queue_waits ADD COLUMN quota_version INTEGER NOT NULL DEFAULT 1"
            )

    def _bind_or_reinitialize(self) -> None:
        row = self.conn.execute("SELECT value FROM meta WHERE key='manifest_digest'").fetchone()
        version = self.conn.execute("SELECT value FROM meta WHERE key='format_version'").fetchone()
        if row is None or version is None:
            with self._write_tx():
                self.conn.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('manifest_digest',?)", (self.digest,))
                self.conn.execute(
                    "INSERT OR REPLACE INTO meta(key,value) VALUES('format_version',?)", (str(SCHEMA_VERSION),)
                )
                self.conn.execute(
                    "INSERT OR REPLACE INTO meta(key,value) VALUES('created_mono_ms',?)", (str(monotonic_ms()),)
                )
                self.conn.execute(
                    "INSERT OR IGNORE INTO state(id,format_version,current_quota_version,active_nodes_json,progress_cursor,requires_rerun) VALUES(1,?,?,?,0,0)",
                    (
                        SCHEMA_VERSION,
                        self.manifest.initial_quota_version,
                        canonical_json([node.node_id for node in self.manifest.nodes]),
                    ),
                )
            return
        if row[0] != self.digest or version[0] != str(SCHEMA_VERSION):
            # _prepare_database handles the normal path; this protects callers that bypass it.
            self.conn.close()
            self.reinitialize_reason = "format_version or manifest_digest changed"
            self._archive_and_remove()
            self.conn = sqlite3.connect(str(self.path), timeout=15.0, check_same_thread=False, isolation_level=None)
            self.conn.row_factory = sqlite3.Row
            self._create_schema()
            self._bind_or_reinitialize()
            return
        state = self.conn.execute("SELECT * FROM state WHERE id=1").fetchone()
        if state is None:
            with self._write_tx():
                self.conn.execute(
                    "INSERT INTO state(id,format_version,current_quota_version,active_nodes_json,progress_cursor,requires_rerun) VALUES(1,?,?,?,0,0)",
                    (
                        SCHEMA_VERSION,
                        self.manifest.initial_quota_version,
                        canonical_json([node.node_id for node in self.manifest.nodes]),
                    ),
                )

    @contextmanager
    def _write_tx(self) -> Iterator[None]:
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield
        except Exception:
            self.conn.rollback()
            raise
        else:
            self.conn.commit()

    def close(self) -> None:
        with self.lock:
            try:
                self.conn.close()
            except sqlite3.Error:
                pass

    def state(self) -> dict[str, Any]:
        with self.lock:
            row = self.conn.execute("SELECT * FROM state WHERE id=1").fetchone()
            if row is None:
                raise RuntimeError("ledger state row is missing")
            return {
                "format_version": int(row["format_version"]),
                "current_quota_version": int(row["current_quota_version"]),
                "active_nodes": json.loads(row["active_nodes_json"]),
                "progress_cursor": int(row["progress_cursor"]),
                "requires_rerun": bool(row["requires_rerun"]),
            }

    def _update_cursor(self, event_seq: int) -> None:
        self.conn.execute(
            "UPDATE state SET progress_cursor=MAX(progress_cursor,?) WHERE id=1", (int(event_seq),)
        )

    def _capacity_conn(self) -> dict[str, Any]:
        rows = self.conn.execute("SELECT * FROM current_view").fetchall()
        cfg = self.manifest.config
        by_node: dict[str, dict[str, Any]] = {}
        for node in self.manifest.nodes:
            by_node[node.node_id] = {
                "slots_declared": node.slots,
                "lease_limit": node.lease_limit,
                "active_leases": 0,
                "active_bytes": 0,
                "used_slots": [],
                "remaining_slots": node.slots,
                "remaining_leases": node.lease_limit,
            }
        global_active = 0
        global_bytes = 0
        for row in rows:
            node_capacity = by_node.get(row["node_id"])
            if node_capacity is None:
                continue
            global_active += 1
            global_bytes += int(row["bytes"])
            node_capacity["active_leases"] += 1
            node_capacity["active_bytes"] += int(row["bytes"])
            node_capacity["used_slots"].append(int(row["slot_index"]))
        for data in by_node.values():
            data["used_slots"] = sorted(set(data["used_slots"]))
            data["remaining_slots"] = max(0, data["slots_declared"] - len(data["used_slots"]))
            data["remaining_leases"] = max(0, data["lease_limit"] - data["active_leases"])
        return {
            "global_active_leases": global_active,
            "global_active_bytes": global_bytes,
            "global_lease_limit": cfg.global_lease_limit,
            "global_remaining_leases": max(0, cfg.global_lease_limit - global_active),
            "global_byte_budget": cfg.global_byte_budget,
            "global_remaining_bytes": max(0, cfg.global_byte_budget - global_bytes),
            "nodes": {key: by_node[key] for key in sorted(by_node)},
        }

    def capacity(self) -> dict[str, Any]:
        with self.lock:
            return self._capacity_conn()

    def _node_leases_conn(self) -> dict[str, list[dict[str, Any]]]:
        result: dict[str, list[dict[str, Any]]] = {node.node_id: [] for node in self.manifest.nodes}
        rows = self.conn.execute("SELECT * FROM current_view ORDER BY node_id,slot_index,lease_id").fetchall()
        for row in rows:
            result.setdefault(row["node_id"], []).append(
                {
                    "lease_id": row["lease_id"],
                    "content_key": row["content_key"],
                    "shard_index": int(row["shard_index"]),
                    "slot_index": int(row["slot_index"]),
                    "bytes": int(row["bytes"]),
                    "expires_ms": int(row["expires_ms"]),
                    "lease_version": int(row["lease_version"]),
                    "quota_version": int(row["quota_version"]),
                    "digest": row["digest"],
                    "status": row["status"],
                }
            )
        return result

    def node_leases(self) -> dict[str, list[dict[str, Any]]]:
        with self.lock:
            return self._node_leases_conn()

    def _append(
        self,
        kind: str,
        event_seq: int,
        payload: dict[str, Any],
        *,
        lease_id: str | None = None,
        node_id: str | None = None,
        content_digest: str | None = None,
    ) -> int:
        self._update_cursor(event_seq)
        cursor = self.conn.execute(
            "INSERT INTO journal(event_seq,kind,lease_id,node_id,payload_json,format_version,content_digest,mono_ms) VALUES(?,?,?,?,?,?,?,?)",
            (
                int(event_seq),
                kind,
                lease_id,
                node_id,
                canonical_json(payload),
                SCHEMA_VERSION,
                content_digest or self.digest,
                monotonic_ms(),
            ),
        )
        ledger_seq = int(cursor.lastrowid)
        self.conn.execute(
            "INSERT INTO checkpoints(ledger_seq,event_seq,progress_cursor,capacity_json,node_leases_json,format_version,content_digest) VALUES(?,?,?,?,?,?,?)",
            (
                ledger_seq,
                int(event_seq),
                int(event_seq),
                canonical_json(self._capacity_conn()),
                canonical_json(self._node_leases_conn()),
                SCHEMA_VERSION,
                content_digest or self.digest,
            ),
        )
        return ledger_seq

    def record_batch_open(
        self,
        batch_id: str,
        event_seq: int,
        node_id: str,
        request: dict[str, Any],
        attempt: int = 0,
    ) -> None:
        with self.lock, self._write_tx():
            now = monotonic_ms()
            self.conn.execute(
                "INSERT INTO batches(batch_id,event_seq,node_id,status,created_ms,request_json,attempt) VALUES(?,?,?,'OPEN',?,?,?)",
                (batch_id, event_seq, node_id, now, canonical_json(request), attempt),
            )
            self._append(
                "BATCH_OPEN",
                event_seq,
                {"batch_id": batch_id, "node_id": node_id, "attempt": attempt, "request": request},
                node_id=node_id,
            )

    def record_queue_start(
        self,
        queue_id: str,
        event_seq: int,
        node_id: str,
        content_key: str,
        shard_index: int,
        wait_start_ms: int,
        attempt: int = 0,
        quota_version: int | None = None,
    ) -> None:
        with self.lock, self._write_tx():
            effective_quota = self.state()["current_quota_version"] if quota_version is None else int(quota_version)
            self.conn.execute(
                "INSERT INTO queue_waits(queue_id,event_seq,node_id,content_key,shard_index,wait_start_ms,status,attempt,quota_version) VALUES(?,?,?,?,?,?,'WAITING',?,?)",
                (queue_id, event_seq, node_id, content_key, shard_index, wait_start_ms, attempt, effective_quota),
            )
            self._append(
                "QUEUE_START",
                event_seq,
                {
                    "queue_id": queue_id,
                    "node_id": node_id,
                    "content_key": content_key,
                    "shard_index": shard_index,
                    "wait_start_ms": wait_start_ms,
                    "attempt": attempt,
                    "quota_version": effective_quota,
                },
                node_id=node_id,
            )

    def record_grant(self, lease: dict[str, Any]) -> None:
        with self.lock, self._write_tx():
            self.conn.execute(
                """
                INSERT INTO leases(lease_id,queue_id,batch_id,node_id,content_key,shard_index,slot_index,bytes,
                    granted_ms,expires_ms,lease_version,quota_version,digest,checksum,status,attempt)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?, 'RESERVED',?)
                """,
                (
                    lease["lease_id"],
                    lease["queue_id"],
                    lease["batch_id"],
                    lease["node_id"],
                    lease["content_key"],
                    lease["shard_index"],
                    lease["slot_index"],
                    lease["bytes"],
                    lease["granted_ms"],
                    lease["expires_ms"],
                    lease["lease_version"],
                    lease["quota_version"],
                    lease["digest"],
                    str(lease["checksum"]),
                    lease.get("attempt", 0),
                ),
            )
            self.conn.execute(
                "INSERT INTO current_view(lease_id,node_id,content_key,shard_index,slot_index,bytes,expires_ms,lease_version,quota_version,digest,status) VALUES(?,?,?,?,?,?,?,?,?,?,'RESERVED')",
                (
                    lease["lease_id"],
                    lease["node_id"],
                    lease["content_key"],
                    lease["shard_index"],
                    lease["slot_index"],
                    lease["bytes"],
                    lease["expires_ms"],
                    lease["lease_version"],
                    lease["quota_version"],
                    lease["digest"],
                ),
            )
            self.conn.execute(
                "INSERT INTO slot_ownership(lease_id,node_id,slot_index,active) VALUES(?,?,?,1)",
                (lease["lease_id"], lease["node_id"], lease["slot_index"]),
            )
            self.conn.execute(
                "UPDATE queue_waits SET wait_end_ms=?,status='GRANTED' WHERE queue_id=?",
                (lease["granted_ms"], lease["queue_id"]),
            )
            self._append(
                "LEASE_GRANT",
                int(lease["event_seq"]),
                dict(lease),
                lease_id=lease["lease_id"],
                node_id=lease["node_id"],
                content_digest=lease["digest"],
            )
            self.conn.execute(
                "UPDATE leases SET last_ledger_seq=(SELECT MAX(ledger_seq) FROM journal) WHERE lease_id=?",
                (lease["lease_id"],),
            )

    def record_batch_commit(
        self,
        batch_id: str,
        event_seq: int,
        node_results: list[dict[str, Any]] | None = None,
    ) -> None:
        with self.lock, self._write_tx():
            rows = self.conn.execute(
                "SELECT lease_id FROM leases WHERE batch_id=? AND status='RESERVED' ORDER BY lease_id", (batch_id,)
            ).fetchall()
            lease_ids = [row["lease_id"] for row in rows]
            self.conn.execute("UPDATE leases SET status='ACTIVE' WHERE batch_id=? AND status='RESERVED'", (batch_id,))
            self.conn.execute("UPDATE current_view SET status='ACTIVE' WHERE lease_id IN (SELECT lease_id FROM leases WHERE batch_id=?)", (batch_id,))
            self.conn.execute(
                "UPDATE batches SET status='COMMITTED',closed_ms=? WHERE batch_id=?", (monotonic_ms(), batch_id)
            )
            node_row = self.conn.execute("SELECT node_id FROM batches WHERE batch_id=?", (batch_id,)).fetchone()
            self._append(
                "BATCH_COMMIT",
                event_seq,
                {"batch_id": batch_id, "lease_ids": lease_ids, "node_results": node_results or []},
                node_id=node_row["node_id"] if node_row else None,
            )

    def record_renew(self, lease_id: str, event_seq: int, old_version: int, new_version: int, old_expiry: int, new_expiry: int) -> None:
        with self.lock, self._write_tx():
            row = self.conn.execute("SELECT node_id,digest FROM leases WHERE lease_id=?", (lease_id,)).fetchone()
            if row is None:
                raise KeyError(lease_id)
            self.conn.execute(
                "UPDATE leases SET lease_version=?,expires_ms=? WHERE lease_id=?",
                (new_version, new_expiry, lease_id),
            )
            self.conn.execute(
                "UPDATE current_view SET lease_version=?,expires_ms=? WHERE lease_id=?",
                (new_version, new_expiry, lease_id),
            )
            self._append(
                "LEASE_RENEW",
                event_seq,
                {
                    "lease_id": lease_id,
                    "old_version": old_version,
                    "new_version": new_version,
                    "old_expires_ms": old_expiry,
                    "new_expires_ms": new_expiry,
                },
                lease_id=lease_id,
                node_id=row["node_id"],
                content_digest=row["digest"],
            )
            self.conn.execute(
                "UPDATE leases SET last_ledger_seq=(SELECT MAX(ledger_seq) FROM journal) WHERE lease_id=?",
                (lease_id,),
            )

    def record_claim(
        self,
        lease_id: str,
        claim_kind: str,
        basis: str,
        event_seq: int,
        *,
        mono_ms: int | None = None,
    ) -> bool:
        with self.lock, self._write_tx():
            row = self.conn.execute("SELECT * FROM leases WHERE lease_id=?", (lease_id,)).fetchone()
            if row is None or row["status"] not in {"RESERVED", "ACTIVE"}:
                return False
            existing = self.conn.execute("SELECT 1 FROM slot_claims WHERE lease_id=?", (lease_id,)).fetchone()
            if existing is not None:
                return False
            status = {
                "EXPIRED": "EXPIRED",
                "QUOTA_REVOKED": "QUOTA_REVOKED",
                "RELEASED": "RELEASED",
                "UNKNOWN": "UNKNOWN",
            }.get(claim_kind, claim_kind)
            at = monotonic_ms() if mono_ms is None else int(mono_ms)
            self.conn.execute(
                "UPDATE leases SET status=?,claim_kind=?,claim_basis=?,claim_event_seq=?,claim_quota_version=? WHERE lease_id=?",
                (status, claim_kind, basis, event_seq, row["quota_version"], lease_id),
            )
            self.conn.execute("DELETE FROM current_view WHERE lease_id=?", (lease_id,))
            self.conn.execute("UPDATE slot_ownership SET active=0 WHERE lease_id=?", (lease_id,))
            self.conn.execute(
                "INSERT INTO slot_claims(lease_id,node_id,slot_index,claim_kind,basis,decision_event_seq,quota_version,mono_ms) VALUES(?,?,?,?,?,?,?,?)",
                (
                    lease_id,
                    row["node_id"],
                    row["slot_index"],
                    claim_kind,
                    basis,
                    event_seq,
                    row["quota_version"],
                    at,
                ),
            )
            self._append(
                "LEASE_CLAIM",
                event_seq,
                {
                    "lease_id": lease_id,
                    "claim_kind": claim_kind,
                    "basis": basis,
                    "slot_index": row["slot_index"],
                    "bytes": row["bytes"],
                    "claim_quota_version": row["quota_version"],
                },
                lease_id=lease_id,
                node_id=row["node_id"],
                content_digest=row["digest"],
            )
            self.conn.execute(
                "UPDATE leases SET last_ledger_seq=(SELECT MAX(ledger_seq) FROM journal) WHERE lease_id=?",
                (lease_id,),
            )
            return True

    def record_batch_unknown(
        self,
        batch_id: str,
        basis: str,
        event_seq: int,
        *,
        error_code: str = "E_NODE",
    ) -> list[dict[str, Any]]:
        with self.lock, self._write_tx():
            rows = self.conn.execute(
                "SELECT * FROM leases WHERE batch_id=? AND status='RESERVED' ORDER BY lease_id", (batch_id,)
            ).fetchall()
            unknown: list[dict[str, Any]] = []
            now = monotonic_ms()
            for row in rows:
                lease_id = row["lease_id"]
                self.conn.execute(
                    "UPDATE leases SET status='UNKNOWN',claim_kind='UNKNOWN',claim_basis=?,claim_event_seq=?,claim_quota_version=? WHERE lease_id=?",
                    (basis, event_seq, row["quota_version"], lease_id),
                )
                self.conn.execute("DELETE FROM current_view WHERE lease_id=?", (lease_id,))
                self.conn.execute("UPDATE slot_ownership SET active=0 WHERE lease_id=?", (lease_id,))
                self.conn.execute(
                    "INSERT OR IGNORE INTO slot_claims(lease_id,node_id,slot_index,claim_kind,basis,decision_event_seq,quota_version,mono_ms) VALUES(?,?,?,'UNKNOWN',?,?,?,?)",
                    (
                        lease_id,
                        row["node_id"],
                        row["slot_index"],
                        basis,
                        event_seq,
                        row["quota_version"],
                        now,
                    ),
                )
                unknown.append(
                    {
                        "lease_id": lease_id,
                        "node_id": row["node_id"],
                        "content_key": row["content_key"],
                        "shard_index": int(row["shard_index"]),
                        "slot_index": int(row["slot_index"]),
                    }
                )
            self.conn.execute(
                "UPDATE batches SET status='UNKNOWN',closed_ms=?,error_code=? WHERE batch_id=?",
                (now, error_code, batch_id),
            )
            self.conn.execute("UPDATE state SET requires_rerun=1 WHERE id=1")
            self._append(
                "LEASE_UNKNOWN",
                event_seq,
                {"batch_id": batch_id, "basis": basis, "error_code": error_code, "leases": unknown},
            )
            return unknown

    def cancel_queues(self, queue_ids: list[str], event_seq: int, basis: str) -> list[str]:
        if not queue_ids:
            return []
        with self.lock, self._write_tx():
            changed: list[str] = []
            now = monotonic_ms()
            for queue_id in sorted(set(queue_ids)):
                cursor = self.conn.execute(
                    "UPDATE queue_waits SET wait_end_ms=?,status='CANCELLED_STALE' WHERE queue_id=? AND status='WAITING'",
                    (now, queue_id),
                )
                if cursor.rowcount:
                    changed.append(queue_id)
            if changed:
                self._append(
                    "QUEUE_CANCEL",
                    event_seq,
                    {"queue_ids": changed, "basis": basis},
                )
            return changed

    def record_quota_transition(
        self,
        event_seq: int,
        quota_version: int,
        node_ids: list[str],
        added_nodes: list[str],
        removed_nodes: list[str],
        new_leases: list[str],
        revoked_leases: list[str],
        retained_leases: list[str],
        *,
        old_view: dict[str, Any] | None = None,
        old_quota_version: int | None = None,
    ) -> int:
        with self.lock, self._write_tx():
            stale_id: int | None = None
            if old_view is not None:
                cursor = self.conn.execute(
                    "INSERT INTO stale_views(quota_version,reason,event_seq,payload_json,created_ms) VALUES(?,?,?,?,?)",
                    (
                        int(old_quota_version if old_quota_version is not None else quota_version - 1),
                        "quota_version_or_node_set_replaced",
                        event_seq,
                        canonical_json(old_view),
                        monotonic_ms(),
                    ),
                )
                stale_id = int(cursor.lastrowid)
            before_digest = canonical_json(self._node_leases_conn())
            self.conn.execute("DELETE FROM current_view")
            self.conn.execute(
                "UPDATE state SET current_quota_version=?,active_nodes_json=? WHERE id=1",
                (quota_version, canonical_json(sorted(node_ids))),
            )
            after_digest = canonical_json(self._node_leases_conn())
            self.conn.execute(
                "INSERT INTO quota_history(quota_version,event_seq,node_ids_json,added_nodes_json,removed_nodes_json,new_leases_json,revoked_leases_json,retained_leases_json,created_ms) VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    quota_version,
                    event_seq,
                    canonical_json(sorted(node_ids)),
                    canonical_json(sorted(added_nodes)),
                    canonical_json(sorted(removed_nodes)),
                    canonical_json(sorted(new_leases)),
                    canonical_json(sorted(revoked_leases)),
                    canonical_json(sorted(retained_leases)),
                    monotonic_ms(),
                ),
            )
            return self._append(
                "QUOTA_CHANGE",
                event_seq,
                {
                    "from_quota_version": old_quota_version,
                    "to_quota_version": quota_version,
                    "node_ids": sorted(node_ids),
                    "added_nodes": sorted(added_nodes),
                    "removed_nodes": sorted(removed_nodes),
                    "new_leases": sorted(new_leases),
                    "revoked_leases": sorted(revoked_leases),
                    "retained_leases": sorted(retained_leases),
                    "stale_view_id": stale_id,
                    "view_before_digest": __import__("hashlib").sha256(before_digest.encode()).hexdigest(),
                    "view_after_digest": __import__("hashlib").sha256(after_digest.encode()).hexdigest(),
                    "replacement": "whole_view",
                },
            )

    def record_reject(self, lease_id: str | None, event_seq: int, node_id: str | None, reason: str) -> None:
        with self.lock, self._write_tx():
            self._append(
                "LEASE_REJECT",
                event_seq,
                {"lease_id": lease_id, "reason": reason},
                lease_id=lease_id,
                node_id=node_id,
            )

    def mark_rerun_resolved(self, event_seq: int) -> None:
        with self.lock, self._write_tx():
            self.conn.execute("UPDATE state SET requires_rerun=0 WHERE id=1")
            self._append("RERUN_RESOLVED", event_seq, {"requires_rerun": False})

    def recover_open_batches(self) -> list[dict[str, Any]]:
        recovered: list[dict[str, Any]] = []
        with self.lock:
            rows = self.conn.execute("SELECT * FROM batches WHERE status='OPEN' ORDER BY created_ms,batch_id").fetchall()
            for row in rows:
                recovered.extend(
                    self.record_batch_unknown(
                        row["batch_id"],
                        "open_batch_found_on_snapshot_restore; occupancy reclaimed before rerun",
                        int(row["event_seq"]),
                        error_code="E_UNKNOWN",
                    )
                )
        return recovered

    def current_leases(self, *, node_id: str | None = None) -> list[dict[str, Any]]:
        with self.lock:
            if node_id is None:
                rows = self.conn.execute("SELECT * FROM current_view ORDER BY node_id,slot_index,lease_id").fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT * FROM current_view WHERE node_id=? ORDER BY node_id,slot_index,lease_id", (node_id,)
                ).fetchall()
            return [dict(row) for row in rows]

    def lease(self, lease_id: str) -> dict[str, Any] | None:
        with self.lock:
            row = self.conn.execute("SELECT * FROM leases WHERE lease_id=?", (lease_id,)).fetchone()
            return dict(row) if row is not None else None

    def leases_by_status(self, status: str) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM leases WHERE status=? ORDER BY node_id,slot_index,lease_id", (status,)
            ).fetchall()
            return [dict(row) for row in rows]

    def unknown_batches(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM batches WHERE status='UNKNOWN' ORDER BY created_ms,batch_id"
            ).fetchall()
            return [dict(row) for row in rows]

    def claims(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM slot_claims ORDER BY decision_event_seq,lease_id"
            ).fetchall()
            return [dict(row) for row in rows]

    def queued(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM queue_waits WHERE status='WAITING' ORDER BY event_seq,node_id,shard_index,queue_id"
            ).fetchall()
            return [dict(row) for row in rows]

    def active_lease_rows(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM leases WHERE status IN ('ACTIVE','RESERVED') ORDER BY node_id,slot_index,lease_id"
            ).fetchall()
            return [dict(row) for row in rows]

    def waiting_queues(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM queue_waits WHERE status='WAITING' ORDER BY event_seq,node_id,shard_index,queue_id"
            ).fetchall()
            return [dict(row) for row in rows]

    def unresolved_unknown_requests(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                """
                SELECT b.* FROM batches b
                WHERE b.status='UNKNOWN'
                  AND NOT EXISTS (
                    SELECT 1 FROM batches r
                    WHERE r.event_seq=b.event_seq AND r.node_id=b.node_id
                      AND r.attempt>b.attempt AND r.status IN ('OPEN','COMMITTED')
                  )
                ORDER BY b.event_seq,b.created_ms,b.batch_id
                """
            ).fetchall()
            requests: dict[tuple[int, str, str, int], dict[str, Any]] = {}
            for row in rows:
                request = json.loads(row["request_json"])
                for item in request.get("items", []):
                    candidate = {
                        "event_seq": int(row["event_seq"]),
                        "node_id": row["node_id"],
                        "content_key": item["content_key"],
                        "shard_index": int(item["shard_index"]),
                        "quota_version": int(request.get("quota_version", 1)),
                        "attempt": int(row["attempt"]) + 1,
                    }
                    key = (
                        candidate["event_seq"],
                        candidate["node_id"],
                        candidate["content_key"],
                        candidate["shard_index"],
                    )
                    previous = requests.get(key)
                    if previous is None or candidate["attempt"] > previous["attempt"]:
                        requests[key] = candidate
            return [requests[key] for key in sorted(requests)]

    def batches(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM batches ORDER BY created_ms,batch_id"
            ).fetchall()
            return [dict(row) for row in rows]

    def stale_view_count(self) -> int:
        with self.lock:
            row = self.conn.execute("SELECT COUNT(*) AS count FROM stale_views").fetchone()
            return int(row["count"]) if row is not None else 0

    def quota_history(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM quota_history ORDER BY event_seq,quota_version"
            ).fetchall()
            result: list[dict[str, Any]] = []
            for row in rows:
                item = dict(row)
                for key in (
                    "node_ids_json",
                    "added_nodes_json",
                    "removed_nodes_json",
                    "new_leases_json",
                    "revoked_leases_json",
                    "retained_leases_json",
                ):
                    item[key.removesuffix("_json")] = json.loads(item.pop(key))
                result.append(item)
            return result

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            state = self.state()
            return {
                "quota_version": state["current_quota_version"],
                "active_nodes": state["active_nodes"],
                "leases": self.current_leases(),
                "queued": self.queued(),
                "claims": self.claims(),
                "capacity": self._capacity_conn(),
                "node_leases": self._node_leases_conn(),
                "stale": True,
            }

    def report(self) -> dict[str, Any]:
        with self.lock:
            state = self.state()
            return {
                "ok": True,
                "format_version": SCHEMA_VERSION,
                "manifest_digest": self.digest,
                "reinitialized": self.reinitialized,
                "reinitialize_reason": self.reinitialize_reason,
                "current_quota_version": state["current_quota_version"],
                "active_nodes": state["active_nodes"],
                "progress_cursor": state["progress_cursor"],
                "requires_rerun": state["requires_rerun"],
                "capacity": self._capacity_conn(),
                "current_leases": self.current_leases(),
                "node_leases": self._node_leases_conn(),
                "queued": self.queued(),
                "unknown_leases": self.leases_by_status("UNKNOWN"),
                "unknown_batches": self.unknown_batches(),
                "batches": self.batches(),
                "claims": self.claims(),
                "quota_transitions": self.quota_history(),
                "stale_view_count": self.stale_view_count(),
                "recovered_unknown": self.recovered_unknown,
            }
