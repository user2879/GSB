"""SQLite ledger with batch-atomic commits and crash recovery.

Batch protocol:
1. begin_batch writes a marker row (committed=0, seq range) in its own
   transaction, so a kill mid-batch leaves visible evidence.
2. commit_batch writes every per-request row of the batch, flips
   committed=1, advances the progress cursor and updates the running
   content digest, all inside ONE SQLite transaction.

Recovery after a forced kill: only fully committed batches are trusted.
Requests inside an uncommitted batch are marked UNKNOWN and must be
re-run. No exit report is claimed for the killed run.

The ledger carries an integer format version and a content digest chained
over committed request rows. The declared clock base is stored in meta;
the source of truth for latencies is the per-observation rows here.
"""

import hashlib
import sqlite3

from . import LEDGER_FORMAT_VERSION, UNKNOWN
from .clock import CLOCK_BASE, monotonic_ms

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS batches (
    batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
    seq_from INTEGER NOT NULL,
    seq_to INTEGER NOT NULL,
    committed INTEGER NOT NULL DEFAULT 0,
    started_ms INTEGER NOT NULL,
    finished_ms INTEGER
);
CREATE TABLE IF NOT EXISTS requests (
    seq INTEGER PRIMARY KEY,
    batch_id INTEGER,
    client_key TEXT,
    service_id TEXT,
    bucket_key TEXT,
    arrival_ms INTEGER,
    token_before INTEGER,
    token_refilled INTEGER,
    allowed INTEGER,
    token_after INTEGER,
    auth_ok INTEGER,
    auth_reason TEXT,
    decision_code TEXT NOT NULL,
    receipt_summary TEXT,
    wait_start_ms INTEGER,
    wait_end_ms INTEGER,
    decide_start_ms INTEGER,
    decide_end_ms INTEGER,
    fwd_start_ms INTEGER,
    fwd_end_ms INTEGER,
    status TEXT NOT NULL DEFAULT 'done'
);
CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    scope TEXT NOT NULL,
    elapsed_ms INTEGER NOT NULL,
    at_ms INTEGER NOT NULL
);
"""

REQUEST_COLUMNS = (
    "seq", "batch_id", "client_key", "service_id", "bucket_key",
    "arrival_ms", "token_before", "token_refilled", "allowed",
    "token_after", "auth_ok", "auth_reason", "decision_code",
    "receipt_summary", "wait_start_ms", "wait_end_ms",
    "decide_start_ms", "decide_end_ms", "fwd_start_ms", "fwd_end_ms",
    "status")


def _row_digest_line(row):
    return "|".join(str(row[c]) for c in REQUEST_COLUMNS)


class Ledger:
    def __init__(self, path, scale=None, services_digest=None,
                 keys_digest=None):
        self.path = path
        self.conn = sqlite3.connect(path, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        existing = self._get_meta("format_version")
        if existing is None:
            if scale is None:
                raise ValueError("new ledger requires scale")
            self._set_meta("format_version", str(LEDGER_FORMAT_VERSION))
            self._set_meta("scale", str(scale))
            self._set_meta("clock_base", CLOCK_BASE)
            self._set_meta("content_digest", "0" * 64)
            self._set_meta("cursor", "0")
            self._set_meta("services_digest", services_digest or "")
            self._set_meta("keys_digest", keys_digest or "")
            self.conn.commit()
        else:
            if int(existing) != LEDGER_FORMAT_VERSION:
                raise ValueError(
                    f"ledger format version {existing} != "
                    f"{LEDGER_FORMAT_VERSION}")
            if scale is not None and int(self._get_meta("scale")) != scale:
                raise ValueError("scale factor mismatch with ledger")

    def _get_meta(self, key):
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def _set_meta(self, key, value):
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value))

    def meta(self):
        return {r["key"]: r["value"] for r in
                self.conn.execute("SELECT key,value FROM meta")}

    @property
    def scale(self):
        return int(self._get_meta("scale"))

    @property
    def cursor(self):
        return int(self._get_meta("cursor"))

    def begin_batch(self, seq_from, seq_to):
        cur = self.conn.execute(
            "INSERT INTO batches(seq_from,seq_to,committed,started_ms) "
            "VALUES(?,?,0,?)", (seq_from, seq_to, monotonic_ms()))
        self.conn.commit()
        return cur.lastrowid

    def commit_batch(self, batch_id, rows):
        """Write all rows of the batch in ONE SQLite transaction."""
        digest = self._get_meta("content_digest")
        cursor_after = 0
        with self.conn:
            for row in rows:
                row = dict(row)
                row["batch_id"] = batch_id
                row["status"] = "done"
                cursor_after = max(cursor_after, row["seq"])
                digest = hashlib.sha256(
                    (digest + "|" + _row_digest_line(row)).encode(
                        "utf-8")).hexdigest()
                self.conn.execute(
                    "INSERT OR REPLACE INTO requests("
                    + ",".join(REQUEST_COLUMNS) + ") VALUES("
                    + ",".join("?" for _ in REQUEST_COLUMNS) + ")",
                    tuple(row[c] for c in REQUEST_COLUMNS))
            self.conn.execute(
                "UPDATE batches SET committed=1, finished_ms=? "
                "WHERE batch_id=?", (monotonic_ms(), batch_id))
            self._set_meta("cursor", str(cursor_after))
            self._set_meta("content_digest", digest)

    def recover_uncommitted(self):
        """Mark requests of uncommitted batches UNKNOWN; return seqs."""
        unknown = []
        rows = self.conn.execute(
            "SELECT batch_id, seq_from, seq_to FROM batches "
            "WHERE committed=0 ORDER BY batch_id").fetchall()
        with self.conn:
            for row in rows:
                for seq in range(row["seq_from"], row["seq_to"] + 1):
                    unknown.append(seq)
                    self.conn.execute(
                        "INSERT OR REPLACE INTO requests"
                        "(seq,batch_id,decision_code,status) "
                        "VALUES(?,?,?,?)",
                        (seq, row["batch_id"], UNKNOWN, "unknown"))
                self.conn.execute(
                    "DELETE FROM batches WHERE batch_id=?",
                    (row["batch_id"],))
        return unknown

    def iter_requests(self):
        return self.conn.execute(
            "SELECT * FROM requests ORDER BY seq").fetchall()

    def done_requests(self):
        return self.conn.execute(
            "SELECT * FROM requests WHERE status='done' "
            "ORDER BY seq").fetchall()

    def record_observation(self, kind, scope, elapsed_ms):
        with self.conn:
            self.conn.execute(
                "INSERT INTO observations(kind,scope,elapsed_ms,at_ms) "
                "VALUES(?,?,?,?)",
                (kind, scope, int(elapsed_ms), monotonic_ms()))

    def observations(self, kind=None):
        if kind is None:
            return self.conn.execute(
                "SELECT * FROM observations ORDER BY id").fetchall()
        return self.conn.execute(
            "SELECT * FROM observations WHERE kind=? ORDER BY id",
            (kind,)).fetchall()

    def close(self):
        self.conn.close()
