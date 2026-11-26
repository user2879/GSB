"""SQLite ledger: atomic batch commits, snapshot replacement and replay."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from .digest import canonical_bytes, sha256_hex

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    integer_value INTEGER,
    text_value TEXT
);
CREATE TABLE IF NOT EXISTS dispatches (
    dispatch_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    subscriber_id TEXT NOT NULL,
    channel_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    batch_index INTEGER NOT NULL,
    attempt INTEGER NOT NULL,
    event_seqs TEXT NOT NULL,
    payload_summary TEXT NOT NULL,
    merge_seq INTEGER NOT NULL,
    enqueued_at_ms INTEGER,
    dequeued_at_ms INTEGER,
    granted_at_ms INTEGER,
    completion_ms INTEGER,
    format_version INTEGER NOT NULL,
    digest TEXT NOT NULL,
    status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS receipts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT NOT NULL,
    format_version TEXT NOT NULL,
    source_format TEXT NOT NULL,
    normalized TEXT NOT NULL,
    backfilled_fields TEXT NOT NULL,
    backfill_basis TEXT,
    content_digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS adoptions (
    batch_id TEXT PRIMARY KEY,
    winning_format TEXT NOT NULL,
    decision TEXT NOT NULL,
    decision_bytes TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    error_margin_applied INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS lease_states (
    batch_id TEXT PRIMARY KEY,
    lease_json TEXT NOT NULL,
    version INTEGER NOT NULL,
    expires_at_ms INTEGER NOT NULL,
    state TEXT NOT NULL,
    content_digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS trajectory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    round_index INTEGER NOT NULL,
    payload TEXT NOT NULL,
    content_digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id TEXT PRIMARY KEY,
    schema_version INTEGER NOT NULL,
    session_digest TEXT NOT NULL,
    payload TEXT NOT NULL,
    replaced INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS replay_findings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT,
    field_path TEXT,
    expected TEXT,
    actual TEXT,
    kind TEXT
);
"""


class Ledger:
    def __init__(self, path, session_digest):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session_digest = session_digest
        # The bus writes trajectory rows and batch transactions from worker
        # threads; a lock serializes access while keeping one connection.
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.replaced = False
        self._init_schema()

    def _init_schema(self):
        has_meta = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'"
        ).fetchone()
        if has_meta is None:
            self.conn.executescript(_SCHEMA)
            self.conn.execute(
                "INSERT INTO meta(key, integer_value, text_value) VALUES(?,?,?)",
                ("schema_version", SCHEMA_VERSION, self.session_digest),
            )
            self.conn.commit()
            return
        existing = self.conn.execute(
            "SELECT text_value FROM meta WHERE key='schema_version'"
        ).fetchone()
        if existing is None or existing[0] != self.session_digest:
            # Whole-snapshot replacement, never incremental merging.
            self._wipe_and_replace()

    def _wipe_and_replace(self):
        self.conn.executescript(
            "DROP TABLE IF EXISTS meta; DROP TABLE IF EXISTS dispatches; "
            "DROP TABLE IF EXISTS receipts; DROP TABLE IF EXISTS adoptions; "
            "DROP TABLE IF EXISTS lease_states; DROP TABLE IF EXISTS trajectory; "
            "DROP TABLE IF EXISTS snapshots; DROP TABLE IF EXISTS replay_findings;"
        )
        self.conn.executescript(_SCHEMA)
        self.conn.execute(
            "INSERT INTO meta(key, integer_value, text_value) VALUES(?,?,?)",
            ("schema_version", SCHEMA_VERSION, self.session_digest),
        )
        self.replaced = True
        self.conn.commit()

    def record_snapshot(self, snapshot_id, payload):
        with self._lock:
            body = canonical_bytes(payload)
        self.conn.execute(
            "INSERT OR REPLACE INTO snapshots VALUES(?,?,?,?,?)",
            (
                snapshot_id,
                SCHEMA_VERSION,
                self.session_digest,
                body.decode("utf-8"),
                1 if self.replaced else 0,
            ),
        )
        self.conn.commit()

    def register_dispatch(self, dispatch):
        """Durably register a dispatch before receipts so x1 can be backfilled."""
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO dispatches VALUES("
                "?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    dispatch["dispatch_id"],
                    dispatch["batch_id"],
                    dispatch["subscriber_id"],
                    dispatch["channel_id"],
                    dispatch["topic"],
                    dispatch["batch_index"],
                    dispatch["attempt"],
                    json.dumps(dispatch["event_seqs"]),
                    json.dumps(dispatch["payload_summary"], sort_keys=True),
                    dispatch["merge_seq"],
                    dispatch.get("enqueued_at_ms"),
                    dispatch.get("dequeued_at_ms"),
                    dispatch.get("granted_at_ms"),
                    dispatch.get("completion_ms"),
                    dispatch["format_version"],
                    dispatch["digest"],
                    dispatch["status"],
                ),
            )

    def mark_completion(self, dispatch_id, completion_ms):
        with self._lock, self.conn:
            self.conn.execute(
                "UPDATE dispatches SET completion_ms=? WHERE dispatch_id=?",
                (completion_ms, dispatch_id),
            )

    def lookup_dispatch(self, batch_id):
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM dispatches WHERE batch_id=? ORDER BY attempt DESC LIMIT 1",
                (batch_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "dispatch_id": row["dispatch_id"],
            "batch_id": row["batch_id"],
            "subscriber_id": row["subscriber_id"],
            "channel_id": row["channel_id"],
            "topic": row["topic"],
            "batch_index": row["batch_index"],
            "attempt": row["attempt"],
            "event_seqs": json.loads(row["event_seqs"]),
            "payload_summary": json.loads(row["payload_summary"]),
            "merge_seq": row["merge_seq"],
            "enqueued_at_ms": row["enqueued_at_ms"],
            "dequeued_at_ms": row["dequeued_at_ms"],
            "granted_at_ms": row["granted_at_ms"],
            "completion_ms": (
                row["completion_ms"]
                if "completion_ms" in row.keys()
                else row["granted_at_ms"]
            ),
            "format_version": row["format_version"],
            "digest": row["digest"],
            "status": row["status"],
            "caliber_version": "v1",
            "covered_refs": [],
        }

    def commit_batch(self, batch_id, receipts, adoption, lease, dispatch_update):
        """Dispatch, three receipts, adoption and lease in one transaction."""
        with self._lock, self.conn:
            self.conn.execute(
                "UPDATE dispatches SET status=?, dequeued_at_ms=?, granted_at_ms=? "
                "WHERE dispatch_id=?",
                (
                    dispatch_update["status"],
                    dispatch_update.get("dequeued_at_ms"),
                    dispatch_update.get("granted_at_ms"),
                    dispatch_update["dispatch_id"],
                ),
            )
            for normalized in receipts:
                content = canonical_bytes(normalized)
                self.conn.execute(
                    "INSERT INTO receipts(batch_id, format_version, source_format,"
                    " normalized, backfilled_fields, backfill_basis, content_digest)"
                    " VALUES(?,?,?,?,?,?,?)",
                    (
                        batch_id,
                        normalized["format_version"],
                        normalized["source_format"],
                        content.decode("utf-8"),
                        json.dumps(normalized["backfilled_fields"]),
                        normalized["backfill_basis"],
                        sha256_hex(content),
                    ),
                )
            decision_bytes = canonical_bytes(adoption["decision"])
            self.conn.execute(
                "INSERT OR REPLACE INTO adoptions VALUES(?,?,?,?,?,?)",
                (
                    batch_id,
                    adoption["winning_format"],
                    json.dumps(adoption["decision"], sort_keys=True),
                    decision_bytes.decode("utf-8"),
                    sha256_hex(decision_bytes),
                    1 if adoption.get("error_margin_applied") else 0,
                ),
            )
            lease_doc = lease.to_dict()
            self.conn.execute(
                "INSERT OR REPLACE INTO lease_states VALUES(?,?,?,?,?,?)",
                (
                    batch_id,
                    json.dumps(lease_doc, sort_keys=True),
                    lease.version,
                    lease.expires_at_ms,
                    "committed",
                    sha256_hex(lease_doc),
                ),
            )

    def append_trajectory(self, round_index, payload):
        body = canonical_bytes(payload)
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO trajectory(round_index, payload, content_digest)"
                " VALUES(?,?,?)",
                (round_index, body.decode("utf-8"), sha256_hex(body)),
            )

    def committed_batch_ids(self):
        with self._lock:
            rows = self.conn.execute(
                "SELECT DISTINCT batch_id FROM adoptions"
            ).fetchall()
        return {row[0] for row in rows}

    def dispatch_status(self):
        with self._lock:
            return {
                row[0]: row[1]
                for row in self.conn.execute("SELECT batch_id, status FROM dispatches")
            }

    def close(self):
        self.conn.close()

    def record_replay_finding(self, batch_id, field_path, expected, actual, kind):
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO replay_findings(batch_id, field_path, expected,"
                " actual, kind) VALUES(?,?,?,?,?)",
                (
                    batch_id,
                    field_path,
                    json.dumps(expected, sort_keys=True),
                    json.dumps(actual, sort_keys=True),
                    kind,
                ),
            )


def diff_replay(expected_batches, recovered_records, ledger=None):
    """Replay planned batches against recovered committed records.

    Returns the first mismatch as a dict, or None when everything matches.
    Unknown (never committed) batches are reported as kind='unknown'.
    """
    recovered = {record["batch_id"]: record for record in recovered_records}
    for batch in expected_batches:
        record = recovered.get(batch.batch_id)
        if record is None:
            finding = {
                "batch_id": batch.batch_id,
                "field_path": "<batch>",
                "expected": "committed",
                "actual": "unknown",
                "kind": "unknown",
            }
            if ledger:
                ledger.record_replay_finding(
                    batch.batch_id, "<batch>", "committed", "unknown", "unknown"
                )
            return finding
        if list(record["event_seqs"]) != list(batch.event_seqs):
            finding = {
                "batch_id": batch.batch_id,
                "field_path": "event_seqs",
                "expected": list(batch.event_seqs),
                "actual": list(record["event_seqs"]),
                "kind": "mismatch",
            }
            if ledger:
                ledger.record_replay_finding(**finding)
            return finding
        if record["digest"] != batch.digest:
            finding = {
                "batch_id": batch.batch_id,
                "field_path": "digest",
                "expected": batch.digest,
                "actual": record["digest"],
                "kind": "mismatch",
            }
            if ledger:
                ledger.record_replay_finding(**finding)
            return finding
    return None
