"""SQLite ledger: the single source of truth.

Each request batch and every breaker migration is written inside one SQLite
transaction together with its receipts, budget consumption and progress
cursor.  The ledger carries an integer format version and a BLAKE2b content
digest.  Recovery can therefore rely solely on fully committed rows; requests
left in the pending table without a committed batch are reported as *unknown*
and must be rerun rather than guessed.
"""

import sqlite3
import threading

from .canonical import content_digest, digest_bytes

LEDGER_FORMAT_VERSION = 1


class Ledger:
    def __init__(self, path):
        self.path = path
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_schema()

    # ------------------------------------------------------------------ schema
    def _create_schema(self):
        with self._lock, self.conn:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS rule_versions (
                    version INTEGER PRIMARY KEY,
                    config_digest TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS pending_requests (
                    request_id TEXT PRIMARY KEY,
                    route_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    batch_id TEXT,
                    state TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS batches (
                    batch_id TEXT PRIMARY KEY,
                    rule_version INTEGER NOT NULL,
                    created_at_ms INTEGER NOT NULL,
                    cursor_sequence INTEGER NOT NULL,
                    batch_digest TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS requests (
                    request_id TEXT PRIMARY KEY,
                    batch_id TEXT NOT NULL,
                    route_id TEXT NOT NULL,
                    client_key TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    service_id TEXT NOT NULL,
                    instance_id TEXT NOT NULL,
                    final_outcome TEXT NOT NULL,
                    total_elapsed_ms INTEGER NOT NULL,
                    started_clock_ms INTEGER NOT NULL,
                    finished_clock_ms INTEGER NOT NULL,
                    detail TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    instance_id TEXT NOT NULL,
                    elapsed_ms INTEGER NOT NULL,
                    outcome TEXT NOT NULL,
                    error_code TEXT NOT NULL,
                    detail TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS breaker_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id TEXT,
                    service_id TEXT NOT NULL,
                    instance_id TEXT NOT NULL,
                    from_state TEXT NOT NULL,
                    to_state TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    mono_ms INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS breaker_state (
                    service_id TEXT NOT NULL,
                    instance_id TEXT NOT NULL,
                    rule_version INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    consecutive_failures INTEGER NOT NULL,
                    opened_at_ms INTEGER,
                    stale INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (service_id, instance_id, rule_version)
                );
                CREATE TABLE IF NOT EXISTS receipts (
                    window_id TEXT NOT NULL,
                    rule_version INTEGER NOT NULL,
                    route_id TEXT NOT NULL,
                    window_index INTEGER NOT NULL,
                    canonical_bytes BLOB NOT NULL,
                    digest TEXT NOT NULL,
                    stale INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (window_id, rule_version)
                );
                CREATE TABLE IF NOT EXISTS view_timings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    started_mono_ms INTEGER NOT NULL,
                    elapsed_ms INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS hash_chain (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id TEXT,
                    entry_digest TEXT NOT NULL,
                    chain_digest TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_requests_route
                    ON requests(route_id, started_clock_ms);
                """
            )
            row = self.conn.execute(
                "SELECT value FROM meta WHERE key='format_version'"
            ).fetchone()
            if row is None:
                self.conn.execute(
                    "INSERT INTO meta(key, value) VALUES('format_version', ?)",
                    (str(LEDGER_FORMAT_VERSION),),
                )
                self.conn.execute(
                    "INSERT INTO meta(key, value) VALUES('head_digest', ?)",
                    ("GENESIS",),
                )

    # ------------------------------------------------------------- housekeeping
    def format_version(self):
        with self._lock:
            row = self.conn.execute(
                "SELECT value FROM meta WHERE key='format_version'"
            ).fetchone()
            return int(row["value"])

    def head_digest(self):
        with self._lock:
            row = self.conn.execute(
                "SELECT value FROM meta WHERE key='head_digest'"
            ).fetchone()
            return row["value"] if row else "GENESIS"

    def close(self):
        with self._lock:
            self.conn.close()

    def record_rule_version(self, version, config_digest, created_at_ms):
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO rule_versions(version, config_digest, created_at_ms)"
                " VALUES(?, ?, ?)",
                (version, config_digest, created_at_ms),
            )

    def rule_versions(self):
        with self._lock:
            return [dict(row) for row in self.conn.execute(
                "SELECT version, config_digest, created_at_ms FROM rule_versions"
                " ORDER BY version"
            )]

    # ------------------------------------------------------------- batch writes
    def mark_pending(self, requests, rule_version):
        with self._lock, self.conn:
            for request in requests:
                self.conn.execute(
                    "INSERT OR IGNORE INTO pending_requests"
                    "(request_id, route_id, sequence, batch_id, state)"
                    " VALUES(?, ?, ?, NULL, 'in_flight')",
                    (request.request_id, request.route_id, request.sequence),
                )

    def append_batch(self, batch_id, rule_version, created_at_ms, cursor_sequence,
                     results, migrations, receipts, breaker_snapshots):
        """Atomically persist a full batch: requests, attempts, breakers, receipts."""
        digest_payload = {
            "batch_id": batch_id,
            "rule_version": rule_version,
            "created_at_ms": created_at_ms,
            "cursor_sequence": cursor_sequence,
            "results": [item.as_ledger_dict() for item in results],
            "migrations": migrations,
            "receipts": sorted(receipt.digest for receipt in receipts),
        }
        batch_digest = content_digest(digest_payload)
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO batches(batch_id, rule_version, created_at_ms,"
                " cursor_sequence, batch_digest) VALUES(?, ?, ?, ?, ?)",
                (batch_id, rule_version, created_at_ms, cursor_sequence, batch_digest),
            )
            for result in results:
                self.conn.execute(
                    "INSERT INTO requests(request_id, batch_id, route_id, client_key,"
                    " sequence, service_id, instance_id, final_outcome, total_elapsed_ms,"
                    " started_clock_ms, finished_clock_ms, detail) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        result.request_id, batch_id, result.route_id, result.client_key,
                        result.sequence, result.service_id, result.instance_id,
                        result.final_outcome, result.total_elapsed_ms,
                        result.started_clock_ms, result.finished_clock_ms,
                        result.final_outcome,
                    ),
                )
                for attempt in result.attempts:
                    self.conn.execute(
                        "INSERT INTO attempts(request_id, attempt, instance_id, elapsed_ms,"
                        " outcome, error_code, detail) VALUES(?,?,?,?,?,?,?)",
                        (
                            result.request_id, attempt.attempt, attempt.instance_id,
                            attempt.elapsed_ms, attempt.outcome, attempt.error_code,
                            attempt.detail,
                        ),
                    )
                self.conn.execute(
                    "UPDATE pending_requests SET batch_id=?, state='committed'"
                    " WHERE request_id=?",
                    (batch_id, result.request_id),
                )
            for migration in migrations:
                self.conn.execute(
                    "INSERT INTO breaker_events(batch_id, service_id, instance_id,"
                    " from_state, to_state, reason, mono_ms) VALUES(?,?,?,?,?,?,?)",
                    (
                        batch_id, migration.get("service_id"), migration["instance_id"],
                        migration["from_state"], migration["to_state"],
                        migration["reason"], migration["mono_ms"],
                    ),
                )
            for snap in breaker_snapshots:
                self.conn.execute(
                    "INSERT INTO breaker_state(service_id, instance_id, rule_version, state,"
                    " consecutive_failures, opened_at_ms, stale) VALUES(?,?,?,?,?,?,0)"
                    " ON CONFLICT(service_id, instance_id, rule_version) DO UPDATE SET"
                    " state=excluded.state, consecutive_failures=excluded.consecutive_failures,"
                    " opened_at_ms=excluded.opened_at_ms, stale=0",
                    (
                        snap["service_id"], snap["instance_id"], rule_version, snap["state"],
                        snap["consecutive_failures"], snap.get("opened_at_ms"),
                    ),
                )
            for receipt in receipts:
                self.conn.execute(
                    "INSERT INTO receipts(window_id, rule_version, route_id, window_index,"
                    " canonical_bytes, digest, stale) VALUES(?,?,?,?,?,?,0)"
                    " ON CONFLICT(window_id, rule_version) DO UPDATE SET"
                    " canonical_bytes=excluded.canonical_bytes, digest=excluded.digest, stale=0",
                    (
                        receipt.window_id, receipt.rule_version, receipt.route_id,
                        receipt.window_index, receipt.canonical_bytes, receipt.digest,
                    ),
                )
            previous_head = self.head_digest()
            entry_digest = digest_bytes(previous_head.encode("utf-8") + batch_digest.encode("utf-8"))
            self.conn.execute(
                "INSERT INTO hash_chain(batch_id, entry_digest, chain_digest)"
                " VALUES(?, ?, ?)",
                (batch_id, entry_digest, entry_digest),
            )
            self.conn.execute(
                "UPDATE meta SET value=? WHERE key='head_digest'", (entry_digest,)
            )
        return batch_digest

    # ------------------------------------------------------------------ staleness
    def mark_stale(self, route_ids, old_version, new_version):
        """Wholesale replacement marker for windows and breaker state."""
        affected_windows = []
        affected_instances = []
        with self._lock, self.conn:
            for route_id in sorted(route_ids):
                rows = self.conn.execute(
                    "SELECT DISTINCT window_id FROM receipts WHERE route_id=?"
                    " AND rule_version=? AND stale=0",
                    (route_id, old_version),
                ).fetchall()
                for row in rows:
                    affected_windows.append(row["window_id"])
                    self.conn.execute(
                        "UPDATE receipts SET stale=1 WHERE window_id=? AND rule_version=?",
                        (row["window_id"], old_version),
                    )
                inst_rows = self.conn.execute(
                    "SELECT DISTINCT service_id, instance_id FROM breaker_state"
                    " WHERE rule_version=? AND stale=0",
                    (old_version,),
                ).fetchall()
                for row in inst_rows:
                    affected_instances.append((row["service_id"], row["instance_id"]))
            self.conn.execute(
                "UPDATE breaker_state SET stale=1 WHERE rule_version=?", (old_version,)
            )
        return sorted(set(affected_windows)), sorted(set(affected_instances))

    # ------------------------------------------------------------------- readers
    def current_receipts(self):
        with self._lock:
            rows = self.conn.execute(
                "SELECT window_id, rule_version, route_id, window_index,"
                " canonical_bytes, digest FROM receipts WHERE stale=0"
                " ORDER BY route_id, window_index"
            ).fetchall()
            return [dict(row) for row in rows]

    def stored_receipts_all(self):
        with self._lock:
            rows = self.conn.execute(
                "SELECT window_id, rule_version, route_id, window_index,"
                " canonical_bytes, digest, stale FROM receipts"
                " ORDER BY route_id, rule_version, window_index"
            ).fetchall()
            return [dict(row) for row in rows]

    def requests_for_route(self, route_id, rule_version):
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM requests WHERE route_id=? AND batch_id IN"
                " (SELECT batch_id FROM batches WHERE rule_version=?)"
                " ORDER BY sequence",
                (route_id, rule_version),
            ).fetchall()
            return [dict(row) for row in rows]

    def all_committed_requests(self):
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM requests ORDER BY route_id, sequence").fetchall()
            return [dict(row) for row in rows]

    def breaker_state_rows(self, include_stale=False):
        sql = "SELECT * FROM breaker_state"
        if not include_stale:
            sql += " WHERE stale=0"
        sql += " ORDER BY service_id, instance_id, rule_version"
        with self._lock:
            return [dict(row) for row in self.conn.execute(sql)]

    def cursor_sequence(self):
        with self._lock:
            row = self.conn.execute(
                "SELECT MAX(cursor_sequence) AS c FROM batches"
            ).fetchone()
            return row["c"] or 0

    def unknown_requests(self):
        """Rows that were marked in-flight but never reached a committed batch."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT request_id, route_id, sequence FROM pending_requests"
                " WHERE state='in_flight' ORDER BY sequence"
            ).fetchall()
            return [dict(row) for row in rows]

    def record_view_timing(self, kind, started_mono_ms, elapsed_ms):
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO view_timings(kind, started_mono_ms, elapsed_ms) VALUES(?,?,?)",
                (kind, started_mono_ms, elapsed_ms),
            )

    def view_timings(self):
        with self._lock:
            return [dict(row) for row in self.conn.execute(
                "SELECT * FROM view_timings ORDER BY id")]

    # ---------------------------------------------------------------- integrity
    def verify_integrity(self):
        """Recompute the hash chain and format version.

        Raises :class:`SnapshotError` (E_SNAPSHOT) on the first mismatch and
        returns integrity metadata when the ledger is intact.
        """
        from .errors import SnapshotError
        with self._lock:
            version = self.format_version()
            chain_rows = self.conn.execute(
                "SELECT batch_id, entry_digest FROM hash_chain ORDER BY seq"
            ).fetchall()
            head = "GENESIS"
            for row in chain_rows:
                batch_row = self.conn.execute(
                    "SELECT batch_digest FROM batches WHERE batch_id=?",
                    (row["batch_id"],),
                ).fetchone()
                expected = digest_bytes(
                    head.encode("utf-8") + batch_row["batch_digest"].encode("utf-8")
                )
                if expected != row["entry_digest"]:
                    raise SnapshotError(
                        "ledger hash-chain content digest mismatch",
                        batch_id=row["batch_id"], stored=row["entry_digest"],
                        expected=expected,
                    )
                head = expected
            if head != self.head_digest():
                raise SnapshotError(
                    "ledger head content digest mismatch",
                    stored=self.head_digest(), expected=head,
                )
            return {"format_version": version, "head_digest": head,
                    "entries": len(chain_rows)}

    def tamper_head_digest(self):
        """Fault-injection helper: corrupt the stored content summary."""
        with self._lock, self.conn:
            self.conn.execute(
                "UPDATE meta SET value='TAMPERED' WHERE key='head_digest'"
            )
