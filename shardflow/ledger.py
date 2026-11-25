"""SQLite durable ledger with format identity, digest and crash recovery.

Every shard terminal outcome is inserted in one SQLite transaction together
with its lease, resource footprint, capacity version and progress cursor.
The ledger stores an integer format version and a content digest; opening a
database whose format or digest does not match causes a whole-file replacement
rather than incremental merging.
"""

import os
import sqlite3

from . import LEDGER_FORMAT_VERSION
from .errors import LedgerError
from .util import canonical_json, sha256_hex, wall_ms

DIGEST_TABLES = ("lease_events", "outcomes", "invalidations", "snapshots",
                 "progress")


class Ledger:
    def __init__(self, path, spec_digest=None, replace_on_mismatch=True):
        self.path = os.path.abspath(path)
        self.spec_digest = spec_digest
        self.replace_on_mismatch = replace_on_mismatch
        self.replaced = False
        self.conn = None
        self._open()

    def _connect(self):
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        return conn

    def _create_schema(self, conn):
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS lease_events (
                row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS outcomes (
                row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                shard_id TEXT NOT NULL,
                attempt INTEGER NOT NULL,
                status TEXT NOT NULL,
                node_id TEXT NOT NULL,
                config_version INTEGER NOT NULL,
                capacity_version INTEGER NOT NULL,
                resource_units INTEGER NOT NULL,
                summary_json TEXT,
                outcome_digest TEXT,
                lease_json TEXT,
                wait_start_ms INTEGER,
                wait_end_ms INTEGER,
                dispatch_ms INTEGER,
                completed_ms INTEGER,
                node_elapsed_ms INTEGER,
                event_seq INTEGER
            );
            CREATE TABLE IF NOT EXISTS invalidations (
                row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                revision INTEGER,
                shard_id TEXT NOT NULL,
                reason TEXT NOT NULL,
                basis_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS snapshots (
                row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                note TEXT NOT NULL,
                wall_ms INTEGER NOT NULL,
                monotonic_ms INTEGER NOT NULL,
                content_digest TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS progress (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                event_seq INTEGER NOT NULL,
                wall_ms INTEGER NOT NULL
            );
            """
        )

    def _open(self):
        existed = os.path.exists(self.path)
        if existed:
            conn = self._connect()
            try:
                format_row = conn.execute(
                    "SELECT value FROM meta WHERE key='format_version'"
                ).fetchone()
                if format_row is None:
                    raise LedgerError("ledger is missing format_version metadata",
                                      field="format_version")
                if int(format_row[0]) != LEDGER_FORMAT_VERSION:
                    raise LedgerError(
                        "ledger format version %s is not supported" % format_row[0],
                        field="format_version",
                    )
                stored = conn.execute(
                    "SELECT value FROM meta WHERE key='content_digest'"
                ).fetchone()
                actual = self._compute_digest(conn)
                if stored is None or stored[0] != actual:
                    raise LedgerError(
                        "ledger content digest does not match disk state",
                        field="content_digest",
                    )
                self.conn = conn
                return
            except LedgerError:
                conn.close()
                if not self.replace_on_mismatch:
                    raise
                self._replace_file()
        self.conn = self._connect()
        self._create_schema(self.conn)
        with self.conn:
            self.conn.executemany(
                "INSERT INTO meta(key, value) VALUES(?, ?)",
                [
                    ("format_version", str(LEDGER_FORMAT_VERSION)),
                    ("spec_digest", self.spec_digest or ""),
                    ("created_wall_ms", str(wall_ms())),
                    ("content_digest", ""),
                ],
            )
            self.conn.execute(
                "INSERT OR REPLACE INTO progress(id, event_seq, wall_ms) "
                "VALUES(1, 0, ?)",
                (wall_ms(),),
            )
            self._refresh_digest_locked()

    def _replace_file(self):
        for suffix in ("", "-wal", "-shm"):
            target = self.path + suffix
            if os.path.exists(target):
                os.remove(target)
        self.replaced = True

    def _compute_digest(self, conn):
        sections = {}
        for table in DIGEST_TABLES:
            if table == "progress":
                rows = conn.execute(
                    "SELECT id, event_seq FROM progress ORDER BY id"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM %s ORDER BY row_id" % table
                ).fetchall()
            sections[table] = [dict(row) for row in rows]
        return sha256_hex(
            {"format_version": LEDGER_FORMAT_VERSION, "tables": sections}
        )

    def _refresh_digest_locked(self):
        digest = self._compute_digest(self.conn)
        self.conn.execute(
            "UPDATE meta SET value=? WHERE key='content_digest'", (digest,)
        )
        return digest

    def record_lease_event(self, kind, payload):
        with self.conn:
            self.conn.execute(
                "INSERT INTO lease_events(kind, payload_json) VALUES(?, ?)",
                (kind, canonical_json(payload)),
            )
            return self._refresh_digest_locked()

    def record_outcome(self, outcome, event_seq):
        lease = outcome.get("lease")
        summary = outcome.get("summary")
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO outcomes(
                    shard_id, attempt, status, node_id, config_version,
                    capacity_version, resource_units, summary_json,
                    outcome_digest, lease_json, wait_start_ms, wait_end_ms,
                    dispatch_ms, completed_ms, node_elapsed_ms, event_seq
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    outcome["shard_id"],
                    outcome["attempt"],
                    outcome["status"],
                    outcome["node_id"],
                    outcome["config_version"],
                    outcome["capacity_version"],
                    outcome["resource_units"],
                    canonical_json(summary) if summary is not None else None,
                    outcome.get("outcome_digest"),
                    canonical_json(lease) if lease is not None else None,
                    outcome.get("wait_start_ms"),
                    outcome.get("wait_end_ms"),
                    outcome.get("dispatch_ms"),
                    outcome.get("completed_ms"),
                    outcome.get("node_elapsed_ms"),
                    event_seq,
                ),
            )
            self.conn.execute(
                "UPDATE progress SET event_seq=?, wall_ms=? WHERE id=1",
                (event_seq, wall_ms()),
            )
            return self._refresh_digest_locked()

    def mark_unknown(self, shard_id, attempt, node_id, lease, event_seq):
        with self.conn:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO outcomes(
                    shard_id, attempt, status, node_id, config_version,
                    capacity_version, resource_units, summary_json,
                    outcome_digest, lease_json, wait_start_ms, wait_end_ms,
                    dispatch_ms, completed_ms, node_elapsed_ms, event_seq
                ) VALUES(?, ?, 'unknown', ?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?,
                        ?, NULL, ?)
                """,
                (
                    shard_id,
                    attempt,
                    node_id,
                    lease.get("config_version", 0) if lease else 0,
                    lease.get("capacity_version", 0) if lease else 0,
                    lease.get("resource_units", 0) if lease else 0,
                    canonical_json(lease) if lease else None,
                    lease.get("wait_start_ms") if lease else None,
                    lease.get("wait_end_ms") if lease else None,
                    lease.get("granted_ms") if lease else None,
                    lease.get("granted_ms") if lease else None,
                    event_seq,
                ),
            )
            return self._refresh_digest_locked()

    def record_invalidation(self, revision, shard_id, reason, basis):
        with self.conn:
            self.conn.execute(
                "INSERT INTO invalidations(revision, shard_id, reason, basis_json)"
                " VALUES(?, ?, ?, ?)",
                (revision, shard_id, reason, canonical_json(basis)),
            )
            return self._refresh_digest_locked()

    def add_snapshot(self, note, monotonic_ms):
        with self.conn:
            digest = self._refresh_digest_locked()
            self.conn.execute(
                "INSERT INTO snapshots(note, wall_ms, monotonic_ms, content_digest)"
                " VALUES(?, ?, ?, ?)",
                (note, wall_ms(), monotonic_ms, digest),
            )
            return self._refresh_digest_locked()

    def fetch_lease_events(self):
        rows = self.conn.execute(
            "SELECT row_id, kind, payload_json FROM lease_events ORDER BY row_id"
        ).fetchall()
        return [
            {"row_id": row["row_id"], "kind": row["kind"],
             "payload": __import__("json").loads(row["payload_json"])}
            for row in rows
        ]

    def fetch_outcomes(self):
        rows = self.conn.execute("SELECT * FROM outcomes ORDER BY row_id").fetchall()
        import json
        result = []
        for row in rows:
            item = dict(row)
            if item["summary_json"] is not None:
                item["summary"] = json.loads(item.pop("summary_json"))
            if item["lease_json"] is not None:
                item["lease"] = json.loads(item.pop("lease_json"))
            result.append(item)
        return result

    def fetch_invalidations(self):
        import json
        rows = self.conn.execute(
            "SELECT * FROM invalidations ORDER BY row_id"
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["basis"] = json.loads(item.pop("basis_json"))
            result.append(item)
        return result

    def fetch_snapshots(self):
        return [
            dict(row)
            for row in self.conn.execute(
                "SELECT * FROM snapshots ORDER BY row_id"
            ).fetchall()
        ]

    def progress_cursor(self):
        row = self.conn.execute("SELECT event_seq FROM progress WHERE id=1").fetchone()
        return int(row["event_seq"]) if row else 0

    def format_version(self):
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key='format_version'"
        ).fetchone()
        return int(row["value"])

    def content_digest(self):
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key='content_digest'"
        ).fetchone()
        return row["value"] if row else None

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None
