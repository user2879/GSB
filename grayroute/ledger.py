"""SQLite audit ledger with atomic, content-addressed commits.

All rows produced when one batch finishes (attribution, adopted receipt,
invalidated propagation, progress cursor) are written in one deferred
transaction and only become visible to other readers at COMMIT.  Each committed
ledger carries an integer format version and a content summary.
"""
from __future__ import annotations

import sqlite3
import threading
from .canonical import content_hash
from .constants import LEDGER_FORMAT_VERSION, UNKNOWN_BATCH

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS batch_attribution (
    batch_id TEXT PRIMARY KEY,
    worker_id TEXT NOT NULL,
    backend_id TEXT NOT NULL,
    rule_id TEXT NOT NULL,
    rule_revision INTEGER NOT NULL,
    param_version_seq INTEGER NOT NULL,
    param_summary TEXT NOT NULL,
    batch_index INTEGER NOT NULL,
    request_ids TEXT NOT NULL,
    arrival_seqs TEXT NOT NULL,
    status TEXT NOT NULL,
    wait_start_ms INTEGER,
    wait_end_ms INTEGER,
    content_summary TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS request_attribution (
    request_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    worker_id TEXT NOT NULL,
    param_version_seq INTEGER NOT NULL,
    pool TEXT NOT NULL,
    bucket INTEGER NOT NULL,
    FOREIGN KEY(batch_id) REFERENCES batch_attribution(batch_id)
);
CREATE TABLE IF NOT EXISTS adopted_receipt (
    batch_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    source_format INTEGER NOT NULL,
    content_summary TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY(batch_id, request_id)
);
CREATE TABLE IF NOT EXISTS rejected_receipt (
    batch_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    source_format INTEGER NOT NULL,
    param_version_seq INTEGER NOT NULL,
    reason TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY(batch_id, request_id, source_format, param_version_seq)
);
CREATE TABLE IF NOT EXISTS invalidation (
    worker_id TEXT NOT NULL,
    param_version_seq INTEGER NOT NULL,
    batch_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    PRIMARY KEY(worker_id, param_version_seq, batch_id, request_id)
);
CREATE TABLE IF NOT EXISTS progress_cursor (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    last_batch_id TEXT NOT NULL,
    last_arrival_seq INTEGER NOT NULL,
    committed_count INTEGER NOT NULL,
    content_summary TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS unknown_batch (
    batch_id TEXT PRIMARY KEY,
    note TEXT NOT NULL
);
"""


class Ledger:
    def __init__(self, path: str):
        self.path = path
        self._local = threading.local()
        self._commit_lock = threading.Lock()
        self._init_conn()
        self._init_meta()

    def _new_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def _init_conn(self) -> sqlite3.Connection:
        conn = self._new_connection()
        conn.executescript(SCHEMA)
        return conn

    @property
    def conn(self) -> sqlite3.Connection:
        """Thread-local connection; safe under the parallel scheduler."""
        current = getattr(self._local, "conn", None)
        if current is None:
            current = self._init_conn()
            self._local.conn = current
        return current

    def _init_meta(self) -> None:
        existing = self.conn.execute(
            "SELECT value FROM meta WHERE key='format_version'").fetchone()
        if existing is None:
            self.conn.execute(
                "INSERT INTO meta(key,value) VALUES('format_version',?)",
                (str(LEDGER_FORMAT_VERSION),))
            self.conn.commit()

    def close(self) -> None:
        current = getattr(self._local, "conn", None)
        if current is not None:
            current.close()
            self._local.conn = None

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def format_version(self) -> int:
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key='format_version'").fetchone()
        return int(row["value"])

    # -- recovery queries -------------------------------------------------- #
    def committed_batch_ids(self) -> set[str]:
        return {row[0] for row in self.conn.execute(
            "SELECT batch_id FROM batch_attribution")}

    def is_committed(self, batch_id: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM batch_attribution WHERE batch_id=?",
            (batch_id,)).fetchone() is not None

    def mark_unknown(self, batch_id: str, note: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO unknown_batch(batch_id,note) VALUES(?,?)",
            (batch_id, note))
        self.conn.commit()

    def unknown_batches(self) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT batch_id,note FROM unknown_batch ORDER BY batch_id"))

    def ledger_content_summary(self) -> str:
        tables = ("batch_attribution", "request_attribution",
                  "adopted_receipt", "rejected_receipt", "invalidation",
                  "progress_cursor")
        material = {}
        for table in tables:
            rows = self.conn.execute(
                f"SELECT * FROM {table} ORDER BY 1,2,3,4").fetchall()
            material[table] = [dict(row) for row in rows]
        return content_hash({"format_version": LEDGER_FORMAT_VERSION,
                             "tables": material})

    # -- atomic batch commit ---------------------------------------------- #
    def commit_batch(self, *, batch, hits, adopted: list[dict],
                     rejected: list[dict], invalidations: list[dict],
                     wait_start_ms: int, wait_end_ms: int) -> str:
        """Write every artifact of one finished batch in one transaction."""
        import json
        from .canonical import canonical_bytes

        batch_summary = content_hash(batch.to_dict())
        with self._commit_lock:
            conn = self.conn
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute(
                    """INSERT INTO batch_attribution(
                        batch_id, worker_id, backend_id, rule_id, rule_revision,
                        param_version_seq, param_summary, batch_index,
                        request_ids, arrival_seqs, status, wait_start_ms,
                        wait_end_ms, content_summary)
                       VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (batch.batch_id, batch.worker_id, batch.backend_id,
                     batch.rule_id, batch.rule_revision, batch.param_version_seq,
                     batch.param_summary, batch.batch_index,
                     json.dumps(list(batch.request_ids)),
                     json.dumps(list(batch.arrival_seqs)),
                     "committed", wait_start_ms, wait_end_ms, batch_summary))

                for request_id in batch.request_ids:
                    hit = hits[request_id]
                    conn.execute(
                        """INSERT INTO request_attribution(
                            request_id, batch_id, worker_id, param_version_seq,
                            pool, bucket) VALUES(?,?,?,?,?,?)""",
                        (request_id, batch.batch_id, hit.worker_id,
                         hit.param_version_seq, hit.pool, hit.bucket))

                for record in adopted:
                    conn.execute(
                        """INSERT INTO adopted_receipt(
                            batch_id, request_id, source_format,
                            content_summary, payload)
                           VALUES(?,?,?,?,?)""",
                        (record["batch_id"], record["request_id"],
                         record["source_format"], record["content_summary"],
                         canonical_bytes(record["normalized"]).decode("utf-8")))

                for record in rejected:
                    conn.execute(
                        """INSERT INTO rejected_receipt(
                            batch_id, request_id, source_format,
                            param_version_seq, reason, payload)
                           VALUES(?,?,?,?,?,?)""",
                        (record["batch_id"], record["request_id"],
                         record["source_format"], record["param_version_seq"],
                         record["reason"],
                         canonical_bytes(record["normalized"]).decode("utf-8")))

                for inv in invalidations:
                    conn.execute(
                        """INSERT OR REPLACE INTO invalidation(
                            worker_id, param_version_seq, batch_id, request_id,
                            reason) VALUES(?,?,?,?,?)""",
                        (inv["worker_id"], inv["param_version_seq"],
                         inv["batch_id"], inv["request_id"], inv["reason"]))

                prior = conn.execute(
                    "SELECT committed_count FROM progress_cursor WHERE id=1"
                ).fetchone()
                count = (prior["committed_count"] if prior else 0) + 1
                max_seq = max(batch.arrival_seqs)
                cursor_summary = content_hash({
                    "last_batch_id": batch.batch_id,
                    "last_arrival_seq": max_seq,
                    "committed_count": count,
                })
                conn.execute(
                    """INSERT INTO progress_cursor(
                        id, last_batch_id, last_arrival_seq, committed_count,
                        content_summary)
                       VALUES(1,?,?,?,?)
                       ON CONFLICT(id) DO UPDATE SET
                         last_batch_id=excluded.last_batch_id,
                         last_arrival_seq=excluded.last_arrival_seq,
                         committed_count=excluded.committed_count,
                         content_summary=excluded.content_summary""",
                    (batch.batch_id, max_seq, count, cursor_summary))
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        return batch_summary

    def adopted_payloads(self, batch_id: str) -> list[dict]:
        import json
        rows = self.conn.execute(
            "SELECT payload, content_summary FROM adopted_receipt "
            "WHERE batch_id=? ORDER BY request_id",
            (batch_id,)).fetchall()
        payloads = []
        for row in rows:
            payload = json.loads(row["payload"])
            payload["content_summary"] = row["content_summary"]
            payloads.append(payload)
        return payloads

    def cursor(self) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM progress_cursor WHERE id=1").fetchone()
        return dict(row) if row else None
