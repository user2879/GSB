"""SQLite ledger: index entries, leases, snapshots, conclusions, progress.

Every shard batch is committed in a single SQLite transaction covering the
batch's index positions, lease states, convergence conclusions and the
progress cursor. The ledger carries an integer format version and a content
digest over its business tables.
"""

import sqlite3
import threading

from .canon import content_digest

FORMAT_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    k TEXT PRIMARY KEY,
    v TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS index_entries (
    shard_id TEXT NOT NULL,
    rule TEXT NOT NULL,
    key_digest TEXT NOT NULL,
    position INTEGER NOT NULL,
    index_version INTEGER NOT NULL,
    batch_id INTEGER NOT NULL,
    PRIMARY KEY (shard_id, rule, key_digest)
);
CREATE TABLE IF NOT EXISTS leases (
    lease_id TEXT PRIMARY KEY,
    shard_id TEXT NOT NULL,
    key_digest TEXT NOT NULL,
    version INTEGER NOT NULL,
    issued_mono_ms INTEGER NOT NULL,
    expiry_mono_ms INTEGER NOT NULL,
    state TEXT NOT NULL,
    batch_id INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id TEXT PRIMARY KEY,
    parent_id TEXT,
    shard_id TEXT NOT NULL,
    index_pos_lo INTEGER NOT NULL,
    index_pos_hi INTEGER NOT NULL,
    lease_expiry_pos INTEGER NOT NULL,
    content_digest TEXT NOT NULL,
    batch_id INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS conclusions (
    shard_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    index_pos INTEGER NOT NULL,
    lease_expiry_pos INTEGER NOT NULL,
    diff INTEGER NOT NULL,
    basis TEXT NOT NULL,
    batch_id INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS progress (
    shard_id TEXT PRIMARY KEY,
    batch_id INTEGER NOT NULL,
    state TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS wait_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    shard_id TEXT NOT NULL,
    task_kind TEXT NOT NULL,
    wait_start_mono_ms INTEGER NOT NULL,
    wait_end_mono_ms INTEGER NOT NULL
);
"""

BUSINESS_TABLES = ("index_entries", "leases", "snapshots", "conclusions", "progress")


class Ledger:
    def __init__(self, path):
        self.path = str(path)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        with self._lock:
            self.conn.executescript(SCHEMA)
            self.conn.commit()
        if self.get_meta("format_version") is None:
            with self._lock:
                self.conn.execute(
                    "INSERT INTO meta(k, v) VALUES ('format_version', ?)",
                    (str(FORMAT_VERSION),))
                self.conn.execute(
                    "INSERT INTO meta(k, v) VALUES ('content_digest', ?)", ("",))
                self.conn.commit()

    def close(self):
        with self._lock:
            self.conn.close()

    def get_meta(self, key):
        with self._lock:
            row = self.conn.execute("SELECT v FROM meta WHERE k=?", (key,)).fetchone()
        return row["v"] if row else None

    def format_version(self):
        v = self.get_meta("format_version")
        return int(v) if v is not None else None

    def _table_dump(self, table):
        cur = self.conn.execute("SELECT * FROM %s" % table)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def dump_state(self):
        """Deterministic dump of all business tables (sorted rows)."""
        with self._lock:
            state = {}
            for table in BUSINESS_TABLES:
                rows = self._table_dump(table)
                rows.sort(key=lambda r: tuple(sorted((k, str(v)) for k, v in r.items())))
                state[table] = rows
            return state

    def compute_digest(self):
        return content_digest(self.dump_state())

    def commit_batch(self, batch_id, *, index_entries=(), leases=(), snapshots=(),
                     conclusions=(), progress=(), wait_events=()):
        """Atomically persist one shard batch and refresh the content digest."""
        with self._lock:
            try:
                self.conn.execute("BEGIN")
                for e in index_entries:
                    self.conn.execute(
                        "INSERT OR REPLACE INTO index_entries"
                        " (shard_id, rule, key_digest, position, index_version, batch_id)"
                        " VALUES (?,?,?,?,?,?)",
                        (e["shard_id"], e["rule"], e["key_digest"], e["position"],
                         e["index_version"], batch_id))
                for l in leases:
                    self.conn.execute(
                        "INSERT OR REPLACE INTO leases"
                        " (lease_id, shard_id, key_digest, version, issued_mono_ms,"
                        "  expiry_mono_ms, state, batch_id)"
                        " VALUES (?,?,?,?,?,?,?,?)",
                        (l["lease_id"], l["shard_id"], l["key_digest"], l["version"],
                         l["issued_mono_ms"], l["expiry_mono_ms"], l["state"], batch_id))
                for s in snapshots:
                    self.conn.execute(
                        "INSERT OR REPLACE INTO snapshots"
                        " (snapshot_id, parent_id, shard_id, index_pos_lo, index_pos_hi,"
                        "  lease_expiry_pos, content_digest, batch_id)"
                        " VALUES (?,?,?,?,?,?,?,?)",
                        (s["snapshot_id"], s["parent_id"], s["shard_id"],
                         s["index_pos_lo"], s["index_pos_hi"], s["lease_expiry_pos"],
                         s["content_digest"], batch_id))
                for c in conclusions:
                    self.conn.execute(
                        "INSERT OR REPLACE INTO conclusions"
                        " (shard_id, status, index_pos, lease_expiry_pos, diff, basis,"
                        "  batch_id) VALUES (?,?,?,?,?,?,?)",
                        (c["shard_id"], c["status"], c["index_pos"],
                         c["lease_expiry_pos"], c["diff"], c["basis"], batch_id))
                for p in progress:
                    self.conn.execute(
                        "INSERT OR REPLACE INTO progress (shard_id, batch_id, state)"
                        " VALUES (?,?,?)",
                        (p["shard_id"], batch_id, p["state"]))
                for w in wait_events:
                    self.conn.execute(
                        "INSERT INTO wait_events"
                        " (shard_id, task_kind, wait_start_mono_ms, wait_end_mono_ms)"
                        " VALUES (?,?,?,?)",
                        (w["shard_id"], w["task_kind"], w["wait_start_mono_ms"],
                         w["wait_end_mono_ms"]))
                digest = self.compute_digest()
                self.conn.execute(
                    "INSERT OR REPLACE INTO meta(k, v) VALUES ('content_digest', ?)",
                    (digest,))
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise
        return digest

    def replace_all(self, state):
        """Wholesale replacement of business tables (recovery, never merge)."""
        with self._lock:
            try:
                self.conn.execute("BEGIN")
                for table in BUSINESS_TABLES:
                    self.conn.execute("DELETE FROM %s" % table)
                for e in state.get("index_entries", []):
                    self.conn.execute(
                        "INSERT INTO index_entries VALUES (?,?,?,?,?,?)",
                        (e["shard_id"], e["rule"], e["key_digest"], e["position"],
                         e["index_version"], e["batch_id"]))
                for l in state.get("leases", []):
                    self.conn.execute(
                        "INSERT INTO leases VALUES (?,?,?,?,?,?,?,?)",
                        (l["lease_id"], l["shard_id"], l["key_digest"], l["version"],
                         l["issued_mono_ms"], l["expiry_mono_ms"], l["state"],
                         l["batch_id"]))
                for s in state.get("snapshots", []):
                    self.conn.execute(
                        "INSERT INTO snapshots VALUES (?,?,?,?,?,?,?,?)",
                        (s["snapshot_id"], s["parent_id"], s["shard_id"],
                         s["index_pos_lo"], s["index_pos_hi"], s["lease_expiry_pos"],
                         s["content_digest"], s["batch_id"]))
                for c in state.get("conclusions", []):
                    self.conn.execute(
                        "INSERT INTO conclusions VALUES (?,?,?,?,?,?,?)",
                        (c["shard_id"], c["status"], c["index_pos"],
                         c["lease_expiry_pos"], c["diff"], c["basis"], c["batch_id"]))
                for p in state.get("progress", []):
                    self.conn.execute(
                        "INSERT INTO progress VALUES (?,?,?)",
                        (p["shard_id"], p["batch_id"], p["state"]))
                digest = self.compute_digest()
                self.conn.execute(
                    "INSERT OR REPLACE INTO meta(k, v) VALUES ('content_digest', ?)",
                    (digest,))
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise
        return digest

    def committed_shards(self):
        with self._lock:
            rows = self.conn.execute(
                "SELECT shard_id FROM progress WHERE state='committed'").fetchall()
        return sorted(r["shard_id"] for r in rows)

    def query(self, sql, params=()):
        with self._lock:
            cur = self.conn.execute(sql, params)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
