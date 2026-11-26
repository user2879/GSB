"""Requirement 8: durable SQLite ledger with atomic batch transactions.

Every completed batch is committed in a single transaction that writes batch
ownership, parameter summary, loss summary, checkpoint state, lease ownership
and the progress cursor together with an integer format version and a content
digest. Reopening a snapshot whose version/digest does not match triggers a
whole-snapshot replacement (never incremental merge).
"""
from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .digest import canonical, digest, digest_text
from .errors import ReplayError, SpecError

LEDGER_FORMAT_VERSION = 1
UNKNOWN = "unknown"
COMMITTED = "committed"


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS batches (
    batch_id TEXT PRIMARY KEY,
    shard_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    cost_units INTEGER NOT NULL,
    param_version INTEGER NOT NULL,
    param_summary TEXT NOT NULL,
    loss_summary TEXT NOT NULL,
    total_loss INTEGER NOT NULL,
    sample_ids TEXT NOT NULL,
    scope_layers TEXT NOT NULL,
    elapsed_ms INTEGER NOT NULL,
    lease_id TEXT,
    lease_version INTEGER,
    content_digest TEXT NOT NULL,
    status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    shard_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    param_version INTEGER NOT NULL,
    loss_baseline TEXT NOT NULL,
    state TEXT NOT NULL,
    content_digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS leases (
    lease_id TEXT PRIMARY KEY,
    shard_id TEXT NOT NULL,
    lease_version INTEGER NOT NULL,
    param_version INTEGER,
    granted_ms INTEGER,
    expires_ms INTEGER,
    state TEXT NOT NULL,
    content_digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ownership (
    unit_index INTEGER NOT NULL,
    owner_kind TEXT NOT NULL,
    shard_id TEXT,
    lease_id TEXT,
    lease_version INTEGER,
    param_version INTEGER,
    event_seq INTEGER NOT NULL,
    basis TEXT NOT NULL,
    PRIMARY KEY (unit_index, event_seq)
);
CREATE TABLE IF NOT EXISTS decisions (
    event_seq INTEGER NOT NULL,
    kind TEXT NOT NULL,
    basis TEXT NOT NULL,
    param_version INTEGER,
    lease_version INTEGER,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS progress (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    last_seq INTEGER NOT NULL,
    cursor_digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pending (
    batch_id TEXT PRIMARY KEY,
    seq INTEGER NOT NULL,
    note TEXT NOT NULL
);
"""


@dataclass
class ReplayOutcome:
    verified_batches: int
    first_diff: str | None
    rerun_required: list[str]


class Ledger:
    def __init__(self, path: str | Path, *, replace: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if replace and self.path.exists():
            # Whole-snapshot replacement: delete the file rather than merge.
            self.path.unlink()
        self._write_lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self._init_meta()

    def _init_meta(self) -> None:
        existing = self.conn.execute(
            "SELECT value FROM meta WHERE key='format_version'").fetchone()
        if existing is None:
            self.conn.execute(
                "INSERT INTO meta(key,value) VALUES('format_version',?)",
                (str(LEDGER_FORMAT_VERSION),))
            self.conn.commit()

    def expected_version(self) -> int:
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key='format_version'").fetchone()
        return int(row["value"]) if row else LEDGER_FORMAT_VERSION

    def snapshot_digest(self) -> str:
        rows = self.conn.execute(
            "SELECT * FROM batches ORDER BY batch_id").fetchall()
        parts = [canonical(dict(row)) for row in rows]
        cp_rows = self.conn.execute(
            "SELECT * FROM checkpoints ORDER BY checkpoint_id").fetchall()
        parts.extend(canonical(dict(row)) for row in cp_rows)
        own_rows = self.conn.execute(
            "SELECT * FROM ownership ORDER BY unit_index,event_seq").fetchall()
        parts.extend(canonical(dict(row)) for row in own_rows)
        return digest_text("|".join(parts))

    def open_snapshot(self, expected_version: int,
                      expected_digest: str | None) -> "Ledger":
        """Reopen with version recognition; replace whole snapshot on mismatch."""
        version = self.expected_version()
        if version != expected_version:
            return self._replace(expected_version, reason="format_version_mismatch")
        if expected_digest is not None and self.snapshot_digest() != expected_digest:
            return self._replace(expected_version, reason="content_digest_mismatch")
        return self

    def _replace(self, expected_version: int, *, reason: str) -> "Ledger":
        path = self.path
        self.conn.close()
        replacement = Ledger(path, replace=True)
        replacement.conn.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)",
            ("last_replace_reason", reason))
        replacement.conn.commit()
        return replacement

    def mark_pending(self, batch_id: str, seq: int, note: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO pending(batch_id,seq,note) VALUES(?,?,?)",
            (batch_id, seq, note))
        self.conn.commit()

    def commit_batch(self, *, seq: int, batch: dict, lease: dict | None,
                     ownership_records: list[dict], decisions: list[dict],
                     checkpoint_states: list[dict]) -> None:
        batch_digest = digest(batch)
        with self._write_lock:
            cur = self.conn.cursor()
            cur.execute("BEGIN IMMEDIATE")
            try:
                cur.execute(
                    """INSERT OR REPLACE INTO batches(batch_id,shard_id,seq,cost_units,
                       param_version,param_summary,loss_summary,total_loss,sample_ids,
                       scope_layers,elapsed_ms,lease_id,lease_version,content_digest,status)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (batch["batch_id"], batch["shard_id"], seq, batch["cost_units"],
                     batch["param_version"], batch["param_summary"], batch["loss_summary"],
                     batch["total_loss"], json.dumps(batch["sample_ids"]),
                     json.dumps(batch["scope_layers"]), batch["elapsed_ms"],
                     (lease or {}).get("lease_id"), (lease or {}).get("lease_version"),
                     batch_digest, COMMITTED))
                if lease is not None:
                    lease_digest = digest(lease)
                    cur.execute(
                        """INSERT OR REPLACE INTO leases(lease_id,shard_id,lease_version,
                           param_version,granted_ms,expires_ms,state,content_digest)
                           VALUES(?,?,?,?,?,?,?,?)""",
                        (lease["lease_id"], lease["shard_id"], lease["lease_version"],
                         lease.get("param_version"), lease.get("granted_ms"),
                         lease.get("expires_ms"), lease.get("state", "active"),
                         lease_digest))
                for record in ownership_records:
                    cur.execute(
                        """INSERT OR REPLACE INTO ownership(unit_index,owner_kind,shard_id,
                           lease_id,lease_version,param_version,event_seq,basis)
                           VALUES(?,?,?,?,?,?,?,?)""",
                        (record["unit_index"], record["owner_kind"], record.get("shard_id"),
                         record.get("lease_id"), record.get("lease_version"),
                         record.get("param_version"), record["event_seq"], record["basis"]))
                for decision in decisions:
                    cur.execute(
                        """INSERT INTO decisions(event_seq,kind,basis,param_version,
                           lease_version,payload) VALUES(?,?,?,?,?,?)""",
                        (decision["event_seq"], decision["kind"], decision["basis"],
                         decision.get("param_version"), decision.get("lease_version"),
                         json.dumps(decision, sort_keys=True)))
                for cp in checkpoint_states:
                    cur.execute(
                        """INSERT OR REPLACE INTO checkpoints(checkpoint_id,shard_id,
                           generation,param_version,loss_baseline,state,content_digest)
                           VALUES(?,?,?,?,?,?,?)""",
                        (cp["checkpoint_id"], cp["shard_id"], cp["generation"],
                         cp["param_version"], cp["loss_baseline"], cp["state"],
                         digest(cp)))
                cursor_digest = digest({"last_seq": seq, "batch_id": batch["batch_id"]})
                cur.execute(
                    "INSERT OR REPLACE INTO progress(id,last_seq,cursor_digest) "
                    "VALUES(1,?,?)", (seq, cursor_digest))
                cur.execute("DELETE FROM pending WHERE batch_id=?", (batch["batch_id"],))
                cur.execute("COMMIT")
            except Exception:
                cur.execute("ROLLBACK")
                raise

    def committed_batches(self) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT * FROM batches WHERE status=? ORDER BY seq", (COMMITTED,)))

    def progress(self) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM progress WHERE id=1").fetchone()

    def pending_batches(self) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM pending ORDER BY seq"))

    def recover_after_crash(self) -> ReplayOutcome:
        """Only fully committed rows are trusted; pending batches need re-run."""
        rerun = [row["batch_id"] for row in self.pending_batches()]
        progress = self.progress()
        last_seq = progress["last_seq"] if progress else -1
        return ReplayOutcome(
            verified_batches=len(self.committed_batches()),
            first_diff=None,
            rerun_required=rerun,
        )

    def verify_replay(self, recompute: Callable[[sqlite3.Row], dict]) -> ReplayOutcome:
        """Recompute every committed batch; E_REPLAY at the first divergence."""
        outcome = self.recover_after_crash()
        for row in self.committed_batches():
            recomputed = recompute(row)
            for field_name in ("param_summary", "loss_summary", "total_loss",
                               "param_version"):
                if recomputed[field_name] != row[field_name]:
                    raise ReplayError(
                        f"replay mismatch on field {field_name}",
                        f"batch:{row['batch_id']}", field_name,
                        detail={"ledger": row[field_name],
                                "recomputed": recomputed[field_name]})
            stored_sample_ids = json.loads(row["sample_ids"])
            if canonical(sorted(recomputed["sample_ids"])) != \
                    canonical(sorted(stored_sample_ids)):
                raise ReplayError("replay mismatch on sample_ids",
                                  f"batch:{row['batch_id']}", "sample_ids")
        return outcome

    def close(self) -> None:
        self.conn.close()
