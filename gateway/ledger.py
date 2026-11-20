"""SQLite hit-ledger persistence, config versioning and recovery scan.

Every finished batch is committed in ONE SQLite transaction: batch row,
per-request ledger rows and the progress cursor. The ledger carries an
integer format version and a content digest. After a forced termination only
fully committed records are visible; batches absent from the ledger are
UNKNOWN and must be re-run.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from typing import Dict, List, Optional

from .models import LEDGER_FORMAT_VERSION, BatchResult, RoutedRequest

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
    config_version INTEGER PRIMARY KEY,
    config_hash TEXT NOT NULL,
    created_at_commit INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS batches (
    config_version INTEGER NOT NULL,
    batch_id TEXT NOT NULL,
    backend_id TEXT NOT NULL,
    rule_id TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    wait_start_ms INTEGER NOT NULL,
    wait_end_ms INTEGER NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    duration_ms INTEGER NOT NULL,
    output_digest TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    PRIMARY KEY (config_version, batch_id)
);
CREATE TABLE IF NOT EXISTS ledger (
    config_version INTEGER NOT NULL,
    request_seq INTEGER NOT NULL,
    rule_id TEXT NOT NULL,
    backend_id TEXT NOT NULL,
    variant TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    output_summary INTEGER NOT NULL,
    duration_ms INTEGER NOT NULL,
    PRIMARY KEY (config_version, request_seq)
);
CREATE TABLE IF NOT EXISTS cursor (
    config_version INTEGER PRIMARY KEY,
    committed_batches INTEGER NOT NULL,
    content_digest TEXT NOT NULL
);
"""


def config_hash(decl, strategy: str) -> str:
    """Hash of the knobs whose change must bump the config version:
    canary weights, batch sizes and concurrency quotas."""
    parts = [strategy]
    for b in decl.backends:
        parts.append(f"b:{b.model_id}:{b.max_concurrency}:{b.batch_size}")
    for r in decl.rules:
        parts.append(f"r:{r.rule_id}:{r.canary_weight_bp}:{r.batch_size}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


class Ledger:
    def __init__(self, path: str):
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('format_version', ?)",
            (str(LEDGER_FORMAT_VERSION),))
        self.conn.commit()

    # -- config version / staleness -------------------------------------
    def current_version(self) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(config_version), 0) FROM snapshots").fetchone()
        return int(row[0])

    def ensure_version(self, new_hash: str) -> int:
        """Return the active config version; bump monotonically on change."""
        version = self.current_version()
        if version == 0:
            self.conn.execute(
                "INSERT INTO snapshots(config_version, config_hash, "
                "created_at_commit) VALUES(1, ?, 0)", (new_hash,))
            self.conn.commit()
            return 1
        row = self.conn.execute(
            "SELECT config_hash FROM snapshots WHERE config_version=?",
            (version,)).fetchone()
        if row[0] != new_hash:
            version += 1
            self.conn.execute(
                "INSERT INTO snapshots(config_version, config_hash, "
                "created_at_commit) VALUES(?, ?, 0)", (version, new_hash))
            self.conn.commit()
        return version

    def snapshots(self) -> List[dict]:
        rows = self.conn.execute(
            "SELECT config_version, config_hash, created_at_commit "
            "FROM snapshots ORDER BY config_version").fetchall()
        return [{"config_version": r[0], "config_hash": r[1],
                 "created_at_commit": r[2]} for r in rows]

    # -- commit ----------------------------------------------------------
    def commit_batch(self, version: int, result: BatchResult,
                     routed: Dict[int, RoutedRequest],
                     summaries: Dict[int, int]) -> None:
        """Atomically commit one batch: batch row + ledger rows + cursor."""
        b = result.batch
        with self._lock, self.conn:  # serialized single transaction
            self.conn.execute(
                "INSERT INTO batches(config_version, batch_id, backend_id, "
                "rule_id, fingerprint, wait_start_ms, wait_end_ms, start_ms, "
                "end_ms, duration_ms, output_digest, attempt) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (version, b.batch_id, b.backend_id, b.rule_id,
                 result.fingerprint, result.wait_start_ms, result.wait_end_ms,
                 result.start_ms, result.end_ms, result.duration_ms,
                 result.output_digest, result.attempt))
            for offset, seq in enumerate(b.request_seqs):
                rr = routed[seq]
                self.conn.execute(
                    "INSERT INTO ledger(config_version, request_seq, rule_id, "
                    "backend_id, variant, batch_id, fingerprint, "
                    "output_summary, duration_ms) VALUES(?,?,?,?,?,?,?,?,?)",
                    (version, seq, rr.rule_id, rr.backend_id, rr.variant,
                     b.batch_id, result.fingerprint, summaries[offset],
                     result.duration_ms))
            row = self.conn.execute(
                "SELECT committed_batches FROM cursor WHERE config_version=?",
                (version,)).fetchone()
            committed = (row[0] if row else 0) + 1
            digest = self._content_digest(version)
            self.conn.execute(
                "INSERT OR REPLACE INTO cursor(config_version, "
                "committed_batches, content_digest) VALUES(?,?,?)",
                (version, committed, digest))

    def _content_digest(self, version: int) -> str:
        rows = self.conn.execute(
            "SELECT request_seq, rule_id, backend_id, variant, batch_id, "
            "fingerprint, output_summary FROM ledger WHERE config_version=? "
            "ORDER BY request_seq", (version,)).fetchall()
        payload = "\n".join("|".join(str(c) for c in row) for row in rows)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    # -- recovery scan ----------------------------------------------------
    def committed_batch_ids(self, version: int) -> List[str]:
        rows = self.conn.execute(
            "SELECT batch_id FROM batches WHERE config_version=? "
            "ORDER BY batch_id", (version,)).fetchall()
        return [r[0] for r in rows]

    def ledger_rows(self, version: Optional[int] = None) -> List[dict]:
        sql = ("SELECT config_version, request_seq, rule_id, backend_id, "
               "variant, batch_id, fingerprint, output_summary, duration_ms "
               "FROM ledger")
        args: tuple = ()
        if version is not None:
            sql += " WHERE config_version=?"
            args = (version,)
        sql += " ORDER BY request_seq"
        cols = ["config_version", "request_seq", "rule_id", "backend_id",
                "variant", "batch_id", "fingerprint", "output_summary",
                "duration_ms"]
        return [dict(zip(cols, row))
                for row in self.conn.execute(sql, args).fetchall()]

    def batch_rows(self, version: int) -> List[dict]:
        cols = ["batch_id", "backend_id", "rule_id", "fingerprint",
                "wait_start_ms", "wait_end_ms", "start_ms", "end_ms",
                "duration_ms", "output_digest", "attempt"]
        rows = self.conn.execute(
            "SELECT " + ", ".join(cols) +
            " FROM batches WHERE config_version=? ORDER BY batch_id",
            (version,)).fetchall()
        return [dict(zip(cols, row)) for row in rows]

    def cursor_info(self, version: int) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT committed_batches, content_digest FROM cursor "
            "WHERE config_version=?", (version,)).fetchone()
        if row is None:
            return None
        return {"committed_batches": row[0], "content_digest": row[1]}

    def close(self) -> None:
        self.conn.close()
