"""SQLite ledger: transactional batch commits and snapshot recovery
(requirement 8).

Layout (one SQLite file):
  meta(key, value)                 - session snapshot identity: spec_version,
                                     content_digest, format_version
  batches(batch_id, spec_version, format_version, content_digest, committed,
          started_ms, committed_ms)
  stage_summaries(stage_id, window_id, batch_id, summary, null_count,
                  null_rule, PRIMARY KEY(stage_id, window_id))
  metric_stats(metric_key, window_id, batch_id, canonical,
               PRIMARY KEY(metric_key, window_id))
  error_attributions(metric_key, spec_version, batch_id, report_json,
                     PRIMARY KEY(metric_key, spec_version))  -- old versions kept
  progress_cursor(id, batch_id, stage_order, window_order)

Rules:
  * Every batch commits stage summaries, metric stats, error attributions and
    the progress cursor in ONE SQLite transaction, stamped with the integer
    FORMAT_VERSION and the content digest.
  * Reopening a snapshot whose stored spec_version or content_digest does not
    match the expected identity replaces the snapshot wholesale (all current
    state rows deleted), never incremental merges.
  * A batch row is first written committed=0 (own transaction) and flipped to
    committed=1 only inside the batch commit transaction. After a forced kill,
    committed=0 batches are reported as "unknown" and must be re-run; only
    fully committed batches are used for recovery. No exit report is claimed
    for the killed process.
  * Recovery verification recomputes from the ledger inputs and reports the
    first mismatching (stage_id, window_id, field).
"""

from __future__ import annotations

import json
import sqlite3
import time

FORMAT_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS batches (
    batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
    spec_version INTEGER NOT NULL,
    format_version INTEGER NOT NULL,
    content_digest TEXT NOT NULL,
    committed INTEGER NOT NULL DEFAULT 0,
    started_ms INTEGER NOT NULL,
    committed_ms INTEGER
);
CREATE TABLE IF NOT EXISTS stage_summaries (
    stage_id TEXT NOT NULL,
    window_id TEXT NOT NULL,
    batch_id INTEGER NOT NULL,
    summary TEXT NOT NULL,
    null_count INTEGER NOT NULL,
    null_rule TEXT NOT NULL,
    PRIMARY KEY (stage_id, window_id)
);
CREATE TABLE IF NOT EXISTS metric_stats (
    metric_key TEXT NOT NULL,
    window_id TEXT NOT NULL,
    batch_id INTEGER NOT NULL,
    canonical TEXT NOT NULL,
    PRIMARY KEY (metric_key, window_id)
);
CREATE TABLE IF NOT EXISTS error_attributions (
    metric_key TEXT NOT NULL,
    spec_version INTEGER NOT NULL,
    batch_id INTEGER NOT NULL,
    report_json TEXT NOT NULL,
    PRIMARY KEY (metric_key, spec_version)
);
CREATE TABLE IF NOT EXISTS progress_cursor (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    batch_id INTEGER NOT NULL,
    stage_order INTEGER NOT NULL,
    window_order INTEGER NOT NULL
);
"""

# Snapshot state replaced wholesale on identity mismatch. error_attributions
# is deliberately excluded: it is the versioned history store (keyed by
# spec_version) that keeps old error bounds for comparison (requirement 5).
_STATE_TABLES = ("stage_summaries", "metric_stats", "progress_cursor")


def _now_ms():
    return time.monotonic_ns() // 1_000_000


class Ledger:
    def __init__(self, path):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.executescript(_SCHEMA)
        self.conn.commit()

    def close(self):
        self.conn.close()

    # -- snapshot identity -------------------------------------------------

    def snapshot_identity(self):
        rows = dict(self.conn.execute("SELECT key, value FROM meta"))
        if not rows:
            return None
        return {
            "spec_version": int(rows["spec_version"]),
            "content_digest": rows["content_digest"],
            "format_version": int(rows["format_version"]),
        }

    def open_snapshot(self, spec_version, content_digest):
        """Returns True if the stored snapshot matches and is kept.

        On any mismatch (or first open) the snapshot state is replaced
        wholesale: all current-state rows are deleted and the new identity is
        written. Nothing is merged incrementally from a mismatched snapshot.
        """
        identity = self.snapshot_identity()
        expected = {
            "spec_version": spec_version,
            "content_digest": content_digest,
            "format_version": FORMAT_VERSION,
        }
        if identity == expected:
            return True
        with self.conn:
            for table in _STATE_TABLES:
                self.conn.execute("DELETE FROM %s" % table)
            self.conn.execute("DELETE FROM meta")
            self.conn.execute("DELETE FROM batches")
            self.conn.executemany(
                "INSERT INTO meta(key, value) VALUES (?, ?)",
                [(k, str(v)) for k, v in expected.items()],
            )
        return False

    # -- batches -----------------------------------------------------------

    def begin_batch(self, spec_version, content_digest):
        """Register a new uncommitted batch (its own transaction)."""
        with self.conn:
            cursor = self.conn.execute(
                "INSERT INTO batches(spec_version, format_version, content_digest,"
                " committed, started_ms) VALUES (?, ?, ?, 0, ?)",
                (spec_version, FORMAT_VERSION, content_digest, _now_ms()),
            )
        return cursor.lastrowid

    def commit_batch(self, batch_id, spec_version, content_digest, stage_summaries,
                     metric_stats, error_attributions, cursor_position):
        """Commit one batch atomically: summaries, stats, attributions, cursor."""
        with self.conn:
            for row in stage_summaries:
                self.conn.execute(
                    "INSERT OR REPLACE INTO stage_summaries"
                    "(stage_id, window_id, batch_id, summary, null_count, null_rule)"
                    " VALUES (?, ?, ?, ?, ?, ?)",
                    (row.stage_id, row.window_id, batch_id, row.summary,
                     row.null_count, row.null_rule),
                )
            for stats in metric_stats:
                self.conn.execute(
                    "INSERT OR REPLACE INTO metric_stats"
                    "(metric_key, window_id, batch_id, canonical) VALUES (?, ?, ?, ?)",
                    (stats.metric_key, stats.window_id, batch_id, stats.canonical()),
                )
            for metric_key, report in error_attributions:
                self.conn.execute(
                    "INSERT OR REPLACE INTO error_attributions"
                    "(metric_key, spec_version, batch_id, report_json) VALUES (?, ?, ?, ?)",
                    (metric_key, spec_version, batch_id, json.dumps(report, sort_keys=True)),
                )
            stage_order, window_order = cursor_position
            self.conn.execute(
                "INSERT OR REPLACE INTO progress_cursor(id, batch_id, stage_order,"
                " window_order) VALUES (1, ?, ?, ?)",
                (batch_id, stage_order, window_order),
            )
            self.conn.execute(
                "UPDATE batches SET committed = 1, committed_ms = ? WHERE batch_id = ?",
                (_now_ms(), batch_id),
            )

    # -- recovery ----------------------------------------------------------

    def recovery_status(self):
        """Classify batches after (re)opening: committed vs unknown."""
        rows = self.conn.execute(
            "SELECT batch_id, spec_version, committed FROM batches ORDER BY batch_id"
        ).fetchall()
        committed = [r[0] for r in rows if r[2] == 1]
        unknown = [r[0] for r in rows if r[2] == 0]
        return {
            "committed_batches": committed,
            "unknown_batches": unknown,
            "requires_rerun": bool(unknown),
            "note": "only fully committed batches are trustworthy; unknown batches"
            " must be re-run; no exit report exists for a killed process",
        }

    def load_stage_summaries(self):
        return {
            (r[0], r[1]): {"summary": r[2], "null_count": r[3], "null_rule": r[4]}
            for r in self.conn.execute(
                "SELECT stage_id, window_id, summary, null_count, null_rule"
                " FROM stage_summaries"
            )
        }

    def load_metric_stats(self):
        return {
            (r[0], r[1]): r[2]
            for r in self.conn.execute("SELECT metric_key, window_id, canonical FROM metric_stats")
        }

    def load_error_attributions(self, spec_version=None):
        if spec_version is None:
            rows = self.conn.execute(
                "SELECT metric_key, spec_version, report_json FROM error_attributions"
            )
        else:
            rows = self.conn.execute(
                "SELECT metric_key, spec_version, report_json FROM error_attributions"
                " WHERE spec_version = ?",
                (spec_version,),
            )
        return {(r[0], r[1]): json.loads(r[2]) for r in rows}

    def load_cursor(self):
        row = self.conn.execute(
            "SELECT batch_id, stage_order, window_order FROM progress_cursor WHERE id = 1"
        ).fetchone()
        return tuple(row) if row else None

    def verify_stage_summaries(self, recomputed):
        """Diff recomputed {key: StageResult-like} against the ledger.

        Returns None when consistent, else the first mismatch as
        (stage_id, window_id, field, ledger_value, recomputed_value) ordered by
        (stage_id, window_id).
        """
        stored = self.load_stage_summaries()
        for key in sorted(set(stored) | set(recomputed)):
            stage_id, window_id = key
            if key not in stored:
                return (stage_id, window_id, "summary", None, "present")
            if key not in recomputed:
                return (stage_id, window_id, "summary", stored[key]["summary"], None)
            result = recomputed[key]
            if stored[key]["summary"] != result.summary:
                return (stage_id, window_id, "summary", stored[key]["summary"], result.summary)
            if stored[key]["null_count"] != result.null_count:
                return (stage_id, window_id, "null_count", stored[key]["null_count"], result.null_count)
        return None
