"""SQLite attribution ledger with versioned, whole-snapshot replacement.

Each completed batch is committed in one transaction containing the batch
attribution, its receipt digest, the invalidation trail and the progress
cursor. The ledger stores an integer format version and a content digest;
reopening a snapshot whose version or digest disagrees triggers a *whole*
replacement (never an incremental merge) and lists the divergence sources.
"""

import sqlite3
from dataclasses import dataclass

from .canonical import dump_bytes, dumps, short_digest
from .errors import ReplayError

LEDGER_FORMAT_VERSION = 1


@dataclass
class SnapshotStatus:
    path: str
    action: str
    format_version: int
    content_digest: str
    stored_digest: str
    differences: list


_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS batches (
    batch_id TEXT PRIMARY KEY,
    worker_id TEXT NOT NULL,
    batch_seq INTEGER NOT NULL,
    rank_order TEXT NOT NULL,
    sorted_requests TEXT NOT NULL,
    param_version INTEGER NOT NULL,
    param_digest TEXT NOT NULL,
    output_digest TEXT NOT NULL,
    basis TEXT NOT NULL,
    result_json TEXT NOT NULL,
    committed_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS receipts (
    batch_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    source_format TEXT NOT NULL,
    normalized_json TEXT NOT NULL,
    PRIMARY KEY (batch_id, request_id)
);
CREATE TABLE IF NOT EXISTS invalidation_traces (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS progress (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    committed_batch_ids TEXT NOT NULL,
    updated_ms INTEGER NOT NULL
);
"""


class Ledger:
    def __init__(self, path, replace=False):
        self.path = str(path)
        if replace:
            self._destroy()
        self.conn = sqlite3.connect(self.path, timeout=10.0)
        self.conn.execute("PRAGMA busy_timeout=10000")
        self.conn.execute("PRAGMA journal_mode=DELETE")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(_SCHEMA)
        self._ensure_meta()

    def _destroy(self):
        # Whole-snapshot replacement: the previous probe connection is
        # already closed, so remove the file (and any journal sidecars) and
        # let __init__ recreate a blank, current-version database. In-place
        # DDL after a same-process reader hits SQLite table locks on some
        # builds, so file replacement is the deterministic path.
        from pathlib import Path
        for suffix in ("", "-journal", "-wal", "-shm"):
            sidecar = Path(self.path + suffix)
            if sidecar.exists():
                sidecar.unlink()

    def _ensure_meta(self):
        row = self.conn.execute(
            "SELECT value FROM schema_meta WHERE key='format_version'"
        ).fetchone()
        if row is None:
            with self.conn:
                self.conn.execute(
                    "INSERT INTO schema_meta(key,value) VALUES(?,?)",
                    ("format_version", str(LEDGER_FORMAT_VERSION)))
                self.conn.execute(
                    "INSERT INTO schema_meta(key,value) VALUES(?,?)",
                    ("content_digest", short_digest([], 32)))
                self.conn.execute(
                    "INSERT INTO progress(id,committed_batch_ids,updated_ms)"
                    " VALUES(1,'[]',0)")

    @property
    def format_version(self):
        return int(self.conn.execute(
            "SELECT value FROM schema_meta WHERE key='format_version'"
        ).fetchone()[0])

    def committed_batches(self):
        rows = self.conn.execute(
            "SELECT batch_id, result_json, committed_ms FROM batches "
            "ORDER BY worker_id, batch_seq"
        ).fetchall()
        return rows

    def _batch_rows_for_digest(self):
        return self.conn.execute(
            "SELECT batch_id, worker_id, batch_seq, rank_order, "
            "sorted_requests, param_version, param_digest, output_digest, "
            "basis, result_json FROM batches "
            "ORDER BY worker_id, batch_seq"
        ).fetchall()

    def batch_ids(self):
        return [row[0] for row in self.conn.execute(
            "SELECT batch_id FROM batches ORDER BY worker_id, batch_seq")]

    def get_batch(self, batch_id):
        row = self.conn.execute(
            "SELECT result_json FROM batches WHERE batch_id=?",
            (batch_id,)).fetchone()
        return row[0] if row else None

    def get_receipts(self, batch_id):
        return [row[0] for row in self.conn.execute(
            "SELECT normalized_json FROM receipts WHERE batch_id=? "
            "ORDER BY request_id", (batch_id,))]

    def traces(self):
        return [row[0] for row in self.conn.execute(
            "SELECT trace_json FROM invalidation_traces ORDER BY seq")]

    def content_digest(self):
        payload = {
            "batches": [list(row) for row in self._batch_rows_for_digest()],
            "receipts": [row[0] for row in self.conn.execute(
                "SELECT normalized_json FROM receipts "
                "ORDER BY batch_id, request_id")],
            "traces": self.traces(),
            "progress": self.conn.execute(
                "SELECT committed_batch_ids FROM progress WHERE id=1"
            ).fetchone()[0],
        }
        return short_digest(payload, 32)

    def stored_digest(self):
        return self.conn.execute(
            "SELECT value FROM schema_meta WHERE key='content_digest'"
        ).fetchone()[0]

    def commit_batch(self, result, receipts=None, traces=None):
        """Atomically write batch + receipts + traces + progress cursor."""
        receipts = receipts or []
        traces = traces or []
        batch = {
            "batch_id": result.batch_id,
            "worker_id": result.worker_id,
            "param_version": result.param_version,
            "param_digest": result.param_digest,
            "rank_order": list(result.rank_order),
            "sorted_requests": list(result.sorted_requests),
            "per_request_digest": result.per_request_digest,
            "output_digest": result.output_digest,
            "basis": result.basis,
        }
        import time
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO batches VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (result.batch_id, result.worker_id,
                 int(result.batch_id.split(":")[2]),
                 dumps(list(result.rank_order)),
                 dumps(list(result.sorted_requests)),
                 result.param_version, result.param_digest,
                 result.output_digest, result.basis, dumps(batch),
                 time.monotonic_ns() // 1_000_000))
            for receipt in receipts:
                normalized = receipt.to_dict() \
                    if hasattr(receipt, "to_dict") else receipt
                self.conn.execute(
                    "INSERT OR REPLACE INTO receipts VALUES(?,?,?,?)",
                    (result.batch_id, normalized["request_id"],
                     normalized["source_format"], dumps(normalized)))
            for trace in traces:
                self.conn.execute(
                    "INSERT INTO invalidation_traces(trace_json) VALUES(?)",
                    (dumps(trace),))
            ids = self.batch_ids()
            if result.batch_id not in ids:
                ids = sorted(ids + [result.batch_id])
            else:
                ids = sorted(ids)
            self.conn.execute(
                "UPDATE progress SET committed_batch_ids=?, updated_ms=? "
                "WHERE id=1",
                (dumps(ids), time.monotonic_ns() // 1_000_000))
            self.conn.execute(
                "UPDATE schema_meta SET value=? WHERE key='content_digest'",
                (self.content_digest(),))

    def append_traces(self, traces):
        with self.conn:
            for trace in traces:
                self.conn.execute(
                    "INSERT INTO invalidation_traces(trace_json) VALUES(?)",
                    (dumps(trace),))
            self.conn.execute(
                "UPDATE schema_meta SET value=? WHERE key='content_digest'",
                (self.content_digest(),))

    def close(self):
        self.conn.close()


def open_ledger(path, expected_version=LEDGER_FORMAT_VERSION):
    """Open a snapshot, replacing it wholesale when version/digest diverge."""
    differences = []
    action = "opened"
    stored_digest = ""
    probe = Ledger(path)
    try:
        tables = {row[0] for row in probe.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if "schema_meta" not in tables:
            differences.append({"source": "schema_meta",
                                "reason": "missing schema_meta table"})
        else:
            version_row = probe.conn.execute(
                "SELECT value FROM schema_meta WHERE key='format_version'"
            ).fetchone()
            if version_row is None or int(version_row[0]) != expected_version:
                differences.append({
                    "source": "schema_meta.format_version",
                    "reason": f"stored={version_row and version_row[0]} "
                              f"expected={expected_version}",
                })
            digest_row = probe.conn.execute(
                "SELECT value FROM schema_meta WHERE key='content_digest'"
            ).fetchone()
            stored_digest = digest_row[0] if digest_row else ""
        if not differences:
            actual = probe.content_digest()
            if actual != stored_digest:
                differences.append({
                    "source": "schema_meta.content_digest",
                    "reason": "stored digest does not match recomputed "
                              "content",
                    "stored": stored_digest,
                    "recomputed": actual,
                })
    finally:
        probe.close()

    if differences:
        action = "replaced"
        ledger = Ledger(path, replace=True)
    else:
        ledger = Ledger(path)
    return ledger, SnapshotStatus(
        path=str(path), action=action,
        format_version=LEDGER_FORMAT_VERSION,
        content_digest=ledger.content_digest(),
        stored_digest=stored_digest,
        differences=differences)


def replay_check(ledger, plan, results):
    """Recompute committed batches and locate the first divergence."""
    for batch in sorted(plan.batches, key=lambda b: (b.worker_id, b.batch_seq)):
        stored_raw = ledger.get_batch(batch.batch_id)
        if stored_raw is None:
            continue
        import json
        stored = json.loads(stored_raw)
        recomputed = results[batch.batch_id]
        checks = (
            ("worker_id", stored["worker_id"], recomputed.worker_id),
            ("param_version", stored["param_version"],
             recomputed.param_version),
            ("param_digest", stored["param_digest"],
             recomputed.param_digest),
            ("rank_order", stored["rank_order"],
             list(recomputed.rank_order)),
            ("sorted_requests", stored["sorted_requests"],
             list(recomputed.sorted_requests)),
            ("per_request_digest", stored["per_request_digest"],
             recomputed.per_request_digest),
            ("output_digest", stored["output_digest"],
             recomputed.output_digest),
        )
        for field_name, stored_value, actual_value in checks:
            if dump_bytes(stored_value) != dump_bytes(actual_value):
                if field_name == "per_request_digest":
                    request_id = next(
                        key for key in stored_value
                        if dump_bytes(stored_value[key])
                        != dump_bytes(actual_value[key]))
                elif field_name in ("rank_order", "sorted_requests"):
                    request_id = next(
                        item for pair in zip(stored_value, actual_value)
                        for item in [pair[0]]
                        if dump_bytes(pair[0]) != dump_bytes(pair[1]))
                else:
                    request_id = batch.requests[0]
                raise ReplayError(
                    "replay divergence: stored ledger does not match "
                    "recomputed batch",
                    field=field_name,
                    extra={
                        "batch_id": batch.batch_id,
                        "request_id": request_id,
                        "field": field_name,
                        "stored": stored_value,
                        "recomputed": actual_value,
                        "basis": "recompute from worker declaration + "
                                 "deterministic forward pass",
                    })
    return True
