"""Execution ledger and session snapshot recovery (SQLite).
Ledger tables (single SQLite database):
  meta(key PK, value)                 ledger format version + content digest
  batches(task_id, batch_index, row_start, row_end, outcome, mem_units,
          result_digest, result_payload, cursor, queued_ms, start_ms, end_ms,
          PRIMARY KEY(task_id, batch_index))
  results(task_id PK, final_digest, agg, agg_value, normalized, scale_exp)
  snapshot(id PK, snapshot_version, state_digest, taken_at)
  recovery_log(id PK, event, detail)  stable, sorted diff listings

Rules:
  * Every finished batch is written together with its progress cursor in ONE
    SQLite transaction.
  * The ledger carries an integer format version and a content digest.
  * On reopen, the stored snapshot version is compared with the current one;
    a mismatch triggers WHOLESALE replacement of batch traces and results
    (never incremental merge) and the replaced keys are listed in a stable
    sorted order in recovery_log.
  * Replay: remaining batches (those without a 'done' record) are re-executed
    from the ledger cursor; the final result must be byte-identical to an
    uninterrupted run, otherwise E_REPLAY locates the divergence.
"""

import hashlib
import json
import sqlite3
import threading

from .errors import ReplayError

LEDGER_FORMAT_VERSION = 1
SNAPSHOT_VERSION = 3  # current session/snapshot schema version

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS batches(
  task_id       TEXT NOT NULL,
  batch_index   INTEGER NOT NULL,
  row_start     INTEGER NOT NULL,
  row_end       INTEGER NOT NULL,
  outcome       TEXT NOT NULL,
  mem_units     INTEGER NOT NULL,
  result_digest TEXT NOT NULL,
  result_payload TEXT NOT NULL DEFAULT '',
  cursor        INTEGER NOT NULL,
  queued_ms     INTEGER,
  start_ms      INTEGER,
  end_ms        INTEGER,
  PRIMARY KEY(task_id, batch_index)
);
CREATE TABLE IF NOT EXISTS results(
  task_id      TEXT PRIMARY KEY,
  final_digest TEXT NOT NULL,
  agg          TEXT NOT NULL,
  agg_value    INTEGER,
  normalized   TEXT NOT NULL,
  scale_exp    INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshot(
  id INTEGER PRIMARY KEY CHECK(id=1),
  snapshot_version INTEGER NOT NULL,
  state_digest TEXT NOT NULL,
  taken_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE TABLE IF NOT EXISTS recovery_log(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event TEXT NOT NULL,
  detail TEXT NOT NULL
);
"""


class Ledger:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.RLock()
        self.conn.executescript(_SCHEMA)
        self.conn.commit()
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES('format_version', ?)",
                (str(LEDGER_FORMAT_VERSION),))

    def record_batch(self, task_id, batch, outcome, mem_units, digest, cursor,
                     payload="", timing=None):
        """One transaction: batch record + progress cursor."""
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO batches VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (task_id, batch.index, batch.row_start, batch.row_end, outcome,
                 mem_units, digest, payload, cursor,
                 None if timing is None else timing.queued_ms,
                 None if timing is None else timing.start_ms,
                 None if timing is None else timing.end_ms))

    def record_result(self, task_id, final_digest, agg, agg_value, normalized,
                      scale_exp):
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO results VALUES(?,?,?,?,?,?)",
                (task_id, final_digest, agg,
                 None if agg_value is None else int(agg_value),
                 normalized, scale_exp))

    def done_batches(self, task_id):
        return {r[0] for r in self.conn.execute(
            "SELECT batch_index FROM batches WHERE task_id=? AND outcome='done'",
            (task_id,))}

    def done_batch_payloads(self, task_id):
        """batch_index -> (row_start, payload_json) for finished batches."""
        return {r[0]: (r[1], r[2]) for r in self.conn.execute(
            "SELECT batch_index, row_start, result_payload FROM batches"
            " WHERE task_id=? AND outcome='done'", (task_id,))}

    def get_result(self, task_id):
        return self.conn.execute(
            "SELECT final_digest, agg, agg_value, normalized, scale_exp"
            " FROM results WHERE task_id=?", (task_id,)).fetchone()

    def batch_records(self):
        return self.conn.execute(
            "SELECT task_id, batch_index, row_start, row_end, outcome, mem_units,"
            " result_digest, cursor, queued_ms, start_ms, end_ms"
            " FROM batches ORDER BY task_id, batch_index").fetchall()

    def content_digest(self):
        """Digest over the whole ledger content (batches + results)."""
        h = hashlib.sha256()
        for row in self.batch_records():
            h.update(json.dumps(row, sort_keys=True).encode() + b"\n")
        for row in self.conn.execute(
                "SELECT * FROM results ORDER BY task_id"):
            h.update(json.dumps(row, sort_keys=True).encode() + b"\n")
        return h.hexdigest()

    def save_snapshot(self, snapshot_version=SNAPSHOT_VERSION):
        digest = self.content_digest()
        with self.conn:
            self.conn.execute("DELETE FROM snapshot")
            self.conn.execute(
                "INSERT INTO snapshot(id, snapshot_version, state_digest)"
                " VALUES(1, ?, ?)", (snapshot_version, digest))
            self.conn.execute(
                "UPDATE meta SET value=? WHERE key='format_version'",
                (str(LEDGER_FORMAT_VERSION),))
        return digest

    def load_snapshot(self):
        return self.conn.execute(
            "SELECT snapshot_version, state_digest FROM snapshot WHERE id=1"
        ).fetchone()

    def check_snapshot(self, current_version=SNAPSHOT_VERSION):
        """Compare stored snapshot version with the current one.

        On mismatch: wholesale replacement of batch traces and results (no
        incremental merge). Returns a report dict with a stable sorted diff.
        """
        snap = self.load_snapshot()
        if snap is None:
            return {"replaced": False, "reason": "no-snapshot", "diff": []}
        stored_version, stored_digest = snap
        if stored_version == current_version:
            return {"replaced": False, "reason": "version-match",
                    "version": stored_version, "diff": []}
        replaced_batches = sorted(
            f"{t}#{i}" for t, i in self.conn.execute(
                "SELECT task_id, batch_index FROM batches"))
        replaced_results = sorted(
            r[0] for r in self.conn.execute("SELECT task_id FROM results"))
        diff = {"batches": replaced_batches, "results": replaced_results}
        with self.conn:
            self.conn.execute("DELETE FROM batches")
            self.conn.execute("DELETE FROM results")
            self.conn.execute("DELETE FROM snapshot")
            self.conn.execute(
                "INSERT INTO recovery_log(event, detail) VALUES(?, ?)",
                ("wholesale-replace", json.dumps({
                    "stored_version": stored_version,
                    "current_version": current_version,
                    "stored_digest": stored_digest,
                    "diff": diff,
                }, sort_keys=True)))
        return {"replaced": True, "stored_version": stored_version,
                "current_version": current_version, "diff": diff}

    def verify_replay(self, task_id, expected_final_digest):
        row = self.get_result(task_id)
        if row is None:
            raise ReplayError("no recorded result to replay against",
                              task_id=task_id)
        recorded = row[0]
        if recorded != expected_final_digest:
            last = self.conn.execute(
                "SELECT batch_index, result_digest FROM batches"
                " WHERE task_id=? ORDER BY batch_index DESC LIMIT 1",
                (task_id,)).fetchone()
            raise ReplayError("replayed result diverges from ledger",
                              task_id=task_id,
                              last_batch=None if last is None else last[0],
                              recorded=recorded, replayed=expected_final_digest)

    def close(self):
        self.conn.close()
