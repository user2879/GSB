"""SQLite ledger with atomic commit, snapshot replace and replay recovery.

All facts for one batch (receipt, adoption conclusion, lease state, progress
cursor) are written inside a single SQLite transaction carrying an integer
format version and a content digest.  Recovery trusts only committed rows;
uncommitted in-flight batches are marked ``unknown`` and their quota is
reclaimed first.  Replaying recomputed facts and comparing field by field
raises :class:`ReplayError` (``E_REPLAY``) at the first mismatch.
"""

import json
import sqlite3

from .digest import canonical_json, digest
from .errors import ReplayError

FORMAT_VERSION = 1


def _content_digest(payload):
    return digest({"format_version": FORMAT_VERSION, "payload": payload})


class Ledger:
    def __init__(self, path):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._create_schema()

    def _create_schema(self):
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS snapshots (
                snapshot_id INTEGER PRIMARY KEY,
                plan_signature TEXT NOT NULL,
                plan_canonical TEXT NOT NULL,
                created_at_ms INTEGER NOT NULL,
                format_version INTEGER NOT NULL,
                content_digest TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS batches (
                batch_id TEXT NOT NULL,
                snapshot_id INTEGER NOT NULL,
                batch_index INTEGER NOT NULL,
                worker_id TEXT NOT NULL,
                receipt_json TEXT NOT NULL,
                adoption_json TEXT NOT NULL,
                lease_json TEXT NOT NULL,
                progress_cursor INTEGER NOT NULL,
                format_version INTEGER NOT NULL,
                content_digest TEXT NOT NULL,
                committed INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (snapshot_id, batch_id)
            );
            CREATE TABLE IF NOT EXISTS progress (
                snapshot_id INTEGER PRIMARY KEY,
                cursor INTEGER NOT NULL,
                last_batch_id TEXT
            );
            """
        )
        self.conn.commit()

    def replace_snapshot(self, plan, created_at_ms):
        """Atomically replace the active snapshot, dropping all prior facts."""
        snapshot_id = self.current_snapshot_id()
        next_id = 1 if snapshot_id is None else snapshot_id + 1
        canonical = plan.canonical_bytes().decode("utf-8")
        payload = {
            "plan_signature": plan.signature,
            "plan_canonical": canonical,
            "created_at_ms": int(created_at_ms),
        }
        with self.conn:
            self.conn.execute("DELETE FROM batches")
            self.conn.execute("DELETE FROM progress")
            self.conn.execute("DELETE FROM snapshots")
            self.conn.execute(
                """INSERT INTO snapshots
                   (snapshot_id, plan_signature, plan_canonical,
                    created_at_ms, format_version, content_digest)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    next_id,
                    plan.signature,
                    canonical,
                    int(created_at_ms),
                    FORMAT_VERSION,
                    _content_digest(payload),
                ),
            )
            self.conn.execute(
                "INSERT INTO progress (snapshot_id, cursor, last_batch_id) "
                "VALUES (?, 0, NULL)",
                (next_id,),
            )
        return next_id

    def current_snapshot_id(self):
        row = self.conn.execute(
            "SELECT MAX(snapshot_id) FROM snapshots"
        ).fetchone()
        return row[0]

    def mark_in_flight(self, snapshot_id, batch, cursor):
        """Reserve a row as uncommitted (simulates a crash mid-batch)."""
        placeholder = json.dumps({"status": "in_flight"}, sort_keys=True)
        payload = {
            "batch_id": batch.batch_id,
            "batch_index": batch.batch_index,
            "worker_id": batch.worker_id,
            "receipt": placeholder,
            "adoption": placeholder,
            "lease": placeholder,
            "progress_cursor": cursor,
        }
        with self.conn:
            self.conn.execute(
                """INSERT INTO batches
                   (batch_id, snapshot_id, batch_index, worker_id,
                    receipt_json, adoption_json, lease_json,
                    progress_cursor, format_version, content_digest,
                    committed)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)""",
                (
                    batch.batch_id,
                    snapshot_id,
                    batch.batch_index,
                    batch.worker_id,
                    placeholder,
                    placeholder,
                    placeholder,
                    cursor,
                    FORMAT_VERSION,
                    _content_digest(payload),
                ),
            )

    def commit_batch(self, snapshot_id, batch, cursor, receipt, adoption,
                     lease):
        """Atomically persist all four fact groups for one batch."""
        receipt_json = json.dumps(receipt, sort_keys=True)
        adoption_json = json.dumps(adoption, sort_keys=True)
        lease_json = json.dumps(lease, sort_keys=True)
        payload = {
            "batch_id": batch.batch_id,
            "batch_index": batch.batch_index,
            "worker_id": batch.worker_id,
            "receipt": receipt,
            "adoption": adoption,
            "lease": lease,
            "progress_cursor": cursor,
        }
        with self.conn:
            self.conn.execute(
                """INSERT OR REPLACE INTO batches
                   (batch_id, snapshot_id, batch_index, worker_id,
                    receipt_json, adoption_json, lease_json,
                    progress_cursor, format_version, content_digest,
                    committed)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)""",
                (
                    batch.batch_id,
                    snapshot_id,
                    batch.batch_index,
                    batch.worker_id,
                    receipt_json,
                    adoption_json,
                    lease_json,
                    cursor,
                    FORMAT_VERSION,
                    _content_digest(payload),
                ),
            )
            self.conn.execute(
                "UPDATE progress SET cursor=?, last_batch_id=? "
                "WHERE snapshot_id=?",
                (cursor, batch.batch_id, snapshot_id),
            )

    def recover(self):
        """Rebuild state from committed rows only.

        Uncommitted rows are reported as ``unknown`` batches whose quota must
        be reclaimed before resuming.
        """
        snapshot_id = self.current_snapshot_id()
        if snapshot_id is None:
            return {"snapshot_id": None, "committed": [], "unknown": [],
                    "cursor": 0}
        rows = self.conn.execute(
            """SELECT batch_id, worker_id, receipt_json, adoption_json,
                      lease_json, progress_cursor, committed, content_digest
               FROM batches WHERE snapshot_id=?
               ORDER BY progress_cursor, batch_id""",
            (snapshot_id,),
        ).fetchall()
        committed, unknown = [], []
        for (batch_id, worker_id, receipt_json, adoption_json, lease_json,
             cursor, is_committed, stored_digest) in rows:
            record = {
                "batch_id": batch_id,
                "worker_id": worker_id,
                "receipt": json.loads(receipt_json),
                "adoption": json.loads(adoption_json),
                "lease": json.loads(lease_json),
                "progress_cursor": cursor,
                "content_digest": stored_digest,
            }
            if is_committed:
                committed.append(record)
            else:
                unknown.append(record)
        cursor = self.conn.execute(
            "SELECT cursor FROM progress WHERE snapshot_id=?", (snapshot_id,)
        ).fetchone()[0]
        return {
            "snapshot_id": snapshot_id,
            "committed": committed,
            "unknown": unknown,
            "cursor": cursor,
        }

    def replay_verify(self, expected):
        """Compare recomputed facts against committed recovery state.

        Raises ``E_REPLAY`` at the first inconsistent batch and field.
        """
        state = self.recover()
        actual = {row["batch_id"]: row for row in state["committed"]}
        for batch_record in expected:
            batch_id = batch_record["batch_id"]
            if batch_id not in actual:
                raise ReplayError(
                    "replay missing committed batch",
                    field="batch_id",
                )
            stored = actual[batch_id]
            for field in ("receipt", "adoption", "lease",
                          "progress_cursor"):
                if canonical_json(stored[field]) != canonical_json(
                    batch_record[field]
                ):
                    raise ReplayError(
                        f"replay mismatch at batch {batch_id!r}",
                        field=field,
                    )
        if len(actual) != len(expected):
            extra = sorted(set(actual) - {r["batch_id"] for r in expected})
            raise ReplayError(
                f"replay found unexpected committed batches {extra}",
                field="batch_id",
            )
        return True

    def close(self):
        self.conn.close()
