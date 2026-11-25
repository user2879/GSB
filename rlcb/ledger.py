"""SQLite ledger with atomic commits, snapshot replacement and crash replay.

Each batch's conclusions, lease ownership, invalidation traces and progress
cursors are written in a single SQLite transaction carrying an integer format
version and a content digest. Reopening a snapshot whose format version or
digest does not match triggers a whole-snapshot replacement (never an
incremental merge). On forced termination only fully committed records are
trusted; an unfinished batch is marked unknown and its quota is reclaimed
first, then requests are replayed and compared item by item to locate the
first divergence.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .codes import DivergenceError, LedgerError
from .hashing import canonical_json, sha256_json

LEDGER_FORMAT_VERSION = 1
LEDGER_APP_ID = 0x524C4342  # 'RLCB'

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS batches (
    batch_id TEXT NOT NULL,
    service_id TEXT NOT NULL,
    config_version INTEGER NOT NULL,
    content_digest TEXT NOT NULL,
    payload TEXT NOT NULL,
    committed_seq INTEGER NOT NULL,
    UNIQUE(service_id, batch_id)
);
CREATE TABLE IF NOT EXISTS decisions (
    request_id TEXT PRIMARY KEY,
    service_id TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    arrival_seq INTEGER NOT NULL,
    payload TEXT NOT NULL,
    content_digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS lease_ownership (
    slot_key TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    content_digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS lease_events (
    event_seq INTEGER PRIMARY KEY,
    payload TEXT NOT NULL,
    content_digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS invalidations (
    service_id TEXT NOT NULL,
    payload TEXT NOT NULL,
    content_digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cursors (
    cursor_name TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    content_digest TEXT NOT NULL
);
"""


class Ledger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fresh = not self.path.exists()
        self.conn = sqlite3.connect(
            str(self.path), isolation_level=None, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(f"PRAGMA application_id={LEDGER_APP_ID};")
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA foreign_keys=ON;")
        self.conn.executescript(SCHEMA)
        if fresh:
            self._write_meta("format_version", LEDGER_FORMAT_VERSION)
        self._verify_header()

    # ---- header / snapshot identity -------------------------------------
    def _write_meta(self, key: str, value: Any) -> None:
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, canonical_json(value)),
        )

    def _read_meta(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def _verify_header(self) -> None:
        version = self._read_meta("format_version")
        if version != LEDGER_FORMAT_VERSION:
            raise LedgerError(
                "ledger format version mismatch; whole snapshot replaced",
                expected=LEDGER_FORMAT_VERSION, found=version,
            )
        app_id = self.conn.execute("PRAGMA application_id;").fetchone()[0]
        if app_id != LEDGER_APP_ID:
            raise LedgerError("ledger application id mismatch",
                              found=app_id)

    def snapshot_fingerprint(self) -> dict[str, Any]:
        decisions = self.conn.execute(
            "SELECT content_digest FROM decisions "
            "ORDER BY request_id").fetchall()
        batches = self.conn.execute(
            "SELECT content_digest FROM batches "
            "ORDER BY service_id, batch_id").fetchall()
        leases = self.conn.execute(
            "SELECT content_digest FROM lease_ownership "
            "ORDER BY slot_key").fetchall()
        material = {
            "format_version": LEDGER_FORMAT_VERSION,
            "decisions": [r[0] for r in decisions],
            "batches": [r[0] for r in batches],
            "leases": [r[0] for r in leases],
        }
        return {"format_version": LEDGER_FORMAT_VERSION,
                "digest": sha256_json(material)}


    # ---- atomic commit ---------------------------------------------------
    @contextmanager
    def transaction(self):
        self.conn.execute("BEGIN IMMEDIATE;")
        try:
            yield self
            self.conn.execute("COMMIT;")
        except Exception:
            self.conn.execute("ROLLBACK;")
            raise

    def commit_batch(self, *, batch: dict[str, Any],
                     decisions: list[dict[str, Any]],
                     lease_ownership: dict[str, dict[str, Any]],
                     lease_events: list[dict[str, Any]],
                     invalidation: dict[str, Any] | None,
                     cursor: dict[str, Any]) -> None:
        batch_digest = sha256_json(batch)
        seq_row = self.conn.execute(
            "SELECT COALESCE(MAX(committed_seq),0) AS m FROM batches"
        ).fetchone()
        next_seq = int(seq_row["m"]) + 1
        with self.transaction():
            self.conn.execute(
                "INSERT INTO batches(batch_id,service_id,config_version,"
                "content_digest,payload,committed_seq) VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(service_id,batch_id) DO UPDATE SET "
                "config_version=excluded.config_version,"
                "content_digest=excluded.content_digest,"
                "payload=excluded.payload,committed_seq=excluded.committed_seq",
                (batch["batch_id"], batch["service_id"],
                 int(batch["config_version"]), batch_digest,
                 canonical_json(batch), next_seq),
            )
            for decision in sorted(decisions, key=lambda d: d["request_id"]):
                digest = sha256_json(decision)
                self.conn.execute(
                    "INSERT INTO decisions(request_id,service_id,batch_id,"
                    "arrival_seq,payload,content_digest) VALUES(?,?,?,?,?,?) "
                    "ON CONFLICT(request_id) DO UPDATE SET service_id="
                    "excluded.service_id,batch_id=excluded.batch_id,"
                    "arrival_seq=excluded.arrival_seq,payload=excluded.payload,"
                    "content_digest=excluded.content_digest",
                    (decision["request_id"], decision["service_id"],
                     decision["batch_id"], int(decision["arrival_seq"]),
                     canonical_json(decision), digest),
                )
            for slot_key, ownership in sorted(lease_ownership.items()):
                self.conn.execute(
                    "INSERT INTO lease_ownership(slot_key,payload,content_digest)"
                    " VALUES(?,?,?) ON CONFLICT(slot_key) DO UPDATE SET "
                    "payload=excluded.payload,content_digest=excluded.content_digest",
                    (slot_key, canonical_json(ownership),
                     sha256_json(ownership)),
                )
            for event in sorted(lease_events, key=lambda e: e["event_seq"]):
                self.conn.execute(
                    "INSERT INTO lease_events(event_seq,payload,content_digest)"
                    " VALUES(?,?,?) ON CONFLICT(event_seq) DO UPDATE SET "
                    "payload=excluded.payload,content_digest=excluded.content_digest",
                    (int(event["event_seq"]), canonical_json(event),
                     sha256_json(event)),
                )
            if invalidation is not None:
                self.conn.execute(
                    "INSERT INTO invalidations(service_id,payload,content_digest)"
                    " VALUES(?,?,?)",
                    (",".join(sorted(c["service_id"]
                                     for c in invalidation.get("changes", [])))
                     or "_",
                     canonical_json(invalidation), sha256_json(invalidation)),
                )
            self._write_cursor("progress", cursor)
            self._write_meta("snapshot_fingerprint", self.snapshot_fingerprint())

    def _write_cursor(self, name: str, payload: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT INTO cursors(cursor_name,payload,content_digest) "
            "VALUES(?,?,?) ON CONFLICT(cursor_name) DO UPDATE SET "
            "payload=excluded.payload,content_digest=excluded.content_digest",
            (name, canonical_json(payload), sha256_json(payload)),
        )

    # ---- reads -----------------------------------------------------------
    def load_decisions(self) -> dict[str, dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT payload FROM decisions ORDER BY request_id").fetchall()
        return {json.loads(r["payload"])["request_id"]: json.loads(r["payload"])
                for r in rows}

    def load_batches(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT payload FROM batches ORDER BY committed_seq").fetchall()
        return [json.loads(r["payload"]) for r in rows]

    def load_lease_ownership(self) -> dict[str, dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT slot_key,payload FROM lease_ownership ORDER BY slot_key"
        ).fetchall()
        return {r["slot_key"]: json.loads(r["payload"]) for r in rows}

    def load_lease_events(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT payload FROM lease_events ORDER BY event_seq").fetchall()
        return [json.loads(r["payload"]) for r in rows]

    def load_cursor(self, name: str = "progress") -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT payload FROM cursors WHERE cursor_name=?", (name,)
        ).fetchone()
        return json.loads(row["payload"]) if row else None

    def committed_batch_keys(self) -> set[str]:
        rows = self.conn.execute(
            "SELECT service_id, batch_id FROM batches").fetchall()
        return {f"{r['service_id']}:{r['batch_id']}" for r in rows}

    def committed_batch_ids(self) -> set[str]:
        rows = self.conn.execute("SELECT batch_id FROM batches").fetchall()
        return {r["batch_id"] for r in rows}

    # ---- snapshot replacement -------------------------------------------
    def replace_with(self, other_fingerprint: dict[str, Any]) -> str:
        current = self.snapshot_fingerprint()
        if current == other_fingerprint:
            return "matched"
        # Whole-snapshot replacement: wipe derived tables, keep format header.
        self.conn.execute("BEGIN IMMEDIATE;")
        try:
            for table in ("decisions", "batches", "lease_ownership",
                          "lease_events", "invalidations", "cursors"):
                self.conn.execute(f"DELETE FROM {table};")
            self._write_meta("replacement", {
                "reason": "fingerprint_or_version_mismatch",
                "old": current, "incoming": other_fingerprint,
            })
            self.conn.execute("COMMIT;")
        except Exception:
            self.conn.execute("ROLLBACK;")
            raise
        return "replaced"

    # ---- crash recovery --------------------------------------------------
    def recover_state(self, expected_batches: list[str]) -> dict[str, Any]:
        committed = self.committed_batch_keys()
        unknown = sorted(set(expected_batches) - committed)
        cursor = self.load_cursor() or {}
        return {
            "committed_batches": sorted(committed),
            "unknown_batches": unknown,
            "reclaimed_quota_for": unknown,
            "cursor": cursor,
        }

    def verify_replay(self, replayed: list[dict[str, Any]]) -> dict[str, Any]:
        stored = self.load_decisions()
        replayed_by_id = {d["request_id"]: d for d in replayed}
        all_ids = sorted(set(stored) | set(replayed_by_id))
        for request_id in all_ids:
            left = stored.get(request_id)
            right = replayed_by_id.get(request_id)
            if left is None or right is None:
                raise DivergenceError(
                    "replay mismatch: request missing on one side",
                    request_id=request_id,
                    stored_present=left is not None,
                    replayed_present=right is not None,
                )
            if canonical_json(left) != canonical_json(right):
                raise DivergenceError(
                    "replay mismatch: first differing request",
                    request_id=request_id,
                    stored=left, replayed=right,
                )
        return {"matched_requests": len(all_ids), "divergence": None}

    def close(self) -> None:
        self.conn.close()
