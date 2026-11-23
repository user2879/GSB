"""SQLite ledger: atomic batch commits, recovery, snapshot verification.

One batch == one SQLite transaction writing batch summary, aggregate
intermediate state, correlation conclusions and the progress cursor together.
A forced termination can only leave either a fully committed batch or an
``UNKNOWN`` marker written ahead of work; UNKNOWN batches are never trusted and
must be re-run. No exit report is fabricated for an unknown batch.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from .errors import SnapshotError
from .util import canonical_digest

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS batches (
    batch_id INTEGER PRIMARY KEY,
    declared_version INTEGER NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('COMMITTED', 'UNKNOWN')),
    cursor_line INTEGER NOT NULL,
    summary_json TEXT NOT NULL,
    digest TEXT NOT NULL,
    stale INTEGER NOT NULL DEFAULT 0,
    committed_clock_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS aggregates (
    batch_id INTEGER NOT NULL,
    aggregate_id TEXT NOT NULL,
    declared_version INTEGER NOT NULL,
    digest TEXT NOT NULL,
    record_json TEXT NOT NULL,
    stale INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (batch_id, declared_version, aggregate_id)
);
CREATE TABLE IF NOT EXISTS correlations (
    batch_id INTEGER NOT NULL,
    aggregate_id TEXT NOT NULL,
    declared_version INTEGER NOT NULL,
    digest TEXT NOT NULL,
    record_json TEXT NOT NULL,
    stale INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (batch_id, declared_version, aggregate_id)
);
CREATE TABLE IF NOT EXISTS dedup_state (
    batch_id INTEGER PRIMARY KEY,
    declared_version INTEGER NOT NULL,
    state_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS progress (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    declared_version INTEGER NOT NULL,
    last_committed_batch INTEGER NOT NULL,
    cursor_line INTEGER NOT NULL,
    stream_digest TEXT NOT NULL,
    graph_digest TEXT NOT NULL,
    rules_digest TEXT NOT NULL
);
"""


@dataclass
class RecoveryState:
    declared_version: int
    last_committed_batch: int
    cursor_line: int
    unknown_batches: list[int]
    stream_digest: str
    graph_digest: str
    rules_digest: str


def record_digest(record: dict) -> str:
    return canonical_digest(record)


class Ledger:
    def __init__(self, path):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------ writes
    def mark_unknown(self, batch_id: int, declared_version: int,
                     cursor_line: int) -> None:
        """Write the pre-work UNKNOWN marker (used for crash injection)."""
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO batches (batch_id, declared_version, "
                "state, cursor_line, summary_json, digest, stale, "
                "committed_clock_ms) VALUES (?, ?, 'UNKNOWN', ?, '{}', '', 0, 0)",
                (batch_id, declared_version, cursor_line))

    def commit_batch(self, batch_id: int, declared_version: int,
                     cursor_line: int, summary: dict,
                     aggregate_records: list[dict],
                     correlation_records: list[dict],
                     digests: dict[str, str], clock_ms: int,
                     dedup_state: list[dict] | None = None) -> None:
        batch_digest = canonical_digest({
            "summary": summary,
            "aggregates": sorted(record_digest(r) for r in aggregate_records),
            "correlations": sorted(
                record_digest(r) for r in correlation_records),
        })
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO batches (batch_id, declared_version, "
                "state, cursor_line, summary_json, digest, stale, "
                "committed_clock_ms) VALUES (?, ?, 'COMMITTED', ?, ?, ?, 0, ?)",
                (batch_id, declared_version, cursor_line,
                 json.dumps(summary, sort_keys=True), batch_digest, clock_ms))
            # Only same-batch rows are replaced; rows from older declared
            # versions keep their stale=1 marker (whole-view replacement).
            self.conn.execute(
                "DELETE FROM aggregates WHERE batch_id = ? AND "
                "declared_version = ?", (batch_id, declared_version))
            self.conn.execute(
                "DELETE FROM correlations WHERE batch_id = ? AND "
                "declared_version = ?", (batch_id, declared_version))
            self.conn.execute(
                "INSERT OR REPLACE INTO dedup_state (batch_id, "
                "declared_version, state_json) VALUES (?, ?, ?)",
                (batch_id, declared_version,
                 json.dumps(dedup_state or [], sort_keys=True,
                            ensure_ascii=False)))
            for rec in aggregate_records:
                self.conn.execute(
                    "INSERT INTO aggregates (batch_id, aggregate_id, "
                    "declared_version, digest, record_json, stale) "
                    "VALUES (?, ?, ?, ?, ?, 0)",
                    (batch_id, rec["aggregate_id"], declared_version,
                     record_digest(rec),
                     json.dumps(rec, sort_keys=True, ensure_ascii=False)))
            for rec in correlation_records:
                self.conn.execute(
                    "INSERT INTO correlations (batch_id, aggregate_id, "
                    "declared_version, digest, record_json, stale) "
                    "VALUES (?, ?, ?, ?, ?, 0)",
                    (batch_id, rec["aggregate_id"], declared_version,
                     record_digest(rec),
                     json.dumps(rec, sort_keys=True, ensure_ascii=False)))
            self.conn.execute(
                "INSERT OR REPLACE INTO progress (id, declared_version, "
                "last_committed_batch, cursor_line, stream_digest, "
                "graph_digest, rules_digest) VALUES (1, ?, ?, ?, ?, ?, ?)",
                (declared_version, batch_id, cursor_line,
                 digests["stream"], digests["graph"], digests["rules"]))

    def mark_all_stale(self) -> int:
        """Evolution guard: stamp every prior conclusion stale in one tx."""
        with self.conn:
            n = 0
            for table in ("batches", "aggregates", "correlations"):
                cur = self.conn.execute(
                    f"UPDATE {table} SET stale = 1 WHERE stale = 0")
                n += cur.rowcount
        return n

    # ------------------------------------------------------------------ reads
    def recovery_state(self) -> RecoveryState | None:
        row = self.conn.execute(
            "SELECT declared_version, last_committed_batch, cursor_line, "
            "stream_digest, graph_digest, rules_digest FROM progress "
            "WHERE id = 1").fetchone()
        unknown = [r[0] for r in self.conn.execute(
            "SELECT batch_id FROM batches WHERE state = 'UNKNOWN' ORDER BY 1")]
        if row is None:
            return None
        return RecoveryState(row[0], row[1], row[2], unknown, row[3], row[4],
                             row[5])

    def verify_snapshots(self) -> None:
        """Recompute every stored digest; first mismatch -> E_SNAPSHOT."""
        for table, id_col in (("aggregates", "aggregate_id"),
                              ("correlations", "aggregate_id")):
            for batch_id, key, stored, payload in self.conn.execute(
                    f"SELECT batch_id, {id_col}, digest, record_json "
                    f"FROM {table} ORDER BY batch_id, {id_col}"):
                actual = canonical_digest(json.loads(payload))
                if actual != stored:
                    raise SnapshotError(
                        "ledger content digest mismatch", batch_id,
                        f"{table}.{id_col}", key)
        for batch_id, stored, payload in self.conn.execute(
                "SELECT batch_id, digest, summary_json FROM batches "
                "WHERE state = 'COMMITTED' ORDER BY batch_id"):
            # Batch digest covers child rows; recompute from child tables.
            agg = [r[0] for r in self.conn.execute(
                "SELECT digest FROM aggregates WHERE batch_id = ? "
                "ORDER BY aggregate_id", (batch_id,))]
            corr = [r[0] for r in self.conn.execute(
                "SELECT digest FROM correlations WHERE batch_id = ? "
                "ORDER BY aggregate_id", (batch_id,))]
            agg.sort()
            corr.sort()
            actual = canonical_digest({
                "summary": json.loads(payload),
                "aggregates": agg, "correlations": corr})
            if actual != stored:
                raise SnapshotError("batch summary digest mismatch", batch_id,
                                    "batches.digest")

    def last_committed_cursor(self, version: int) -> int:
        row = self.conn.execute(
            "SELECT MAX(cursor_line) FROM batches WHERE state='COMMITTED' "
            "AND declared_version = ?", (version,)).fetchone()
        return row[0] if row and row[0] is not None else 0

    def latest_dedup_state(self, version: int) -> list[dict]:
        row = self.conn.execute(
            "SELECT state_json FROM dedup_state WHERE declared_version = ? "
            "ORDER BY batch_id DESC LIMIT 1", (version,)).fetchone()
        return json.loads(row[0]) if row else []

    def load_committed(self, version: int):
        batches = [dict(batch_id=r[0], cursor_line=r[1],
                        summary=json.loads(r[2]))
                   for r in self.conn.execute(
                       "SELECT batch_id, cursor_line, summary_json FROM batches "
                       "WHERE state = 'COMMITTED' AND declared_version = ? "
                       "ORDER BY batch_id", (version,))]
        agg_rows = self.conn.execute(
            "SELECT record_json FROM aggregates WHERE declared_version = ? "
            "ORDER BY batch_id, aggregate_id", (version,)).fetchall()
        corr_rows = self.conn.execute(
            "SELECT record_json FROM correlations WHERE declared_version = ? "
            "ORDER BY batch_id, aggregate_id", (version,)).fetchall()
        aggregates = [json.loads(r[0]) for r in agg_rows]
        correlations = [json.loads(r[0]) for r in corr_rows]
        return batches, aggregates, correlations

    def current_view(self, version: int):
        _, aggregates, correlations = self.load_committed(version)
        # Stale rows are excluded from the current presentation/summary.
        return aggregates, correlations

    def corrupt_for_injection(self, table: str, batch_id: int) -> None:
        """Test helper: tamper with stored payload without updating digest."""
        with self.conn:
            self.conn.execute(
                f"UPDATE {table} SET record_json = ? WHERE batch_id = ?",
                ("{\"tampered\": true}", batch_id))
