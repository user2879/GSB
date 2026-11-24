"""SQLite persistence ledger for obsflow.

All per-window conclusions, lease ownership, propagation trails and progress
cursors are written inside a single SQLite transaction per batch and carry
an integer format version plus a content digest. Reopening a snapshot
compares source-set / format-set / scale-convention versions and the content
digest; any mismatch triggers a whole-state replacement with a read-only
backup - incremental merging is explicitly forbidden.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict
import sqlite3
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from .canonical import canonical_bytes, content_digest
from .errors import SnapshotError
from .models import Lease, PropagationEvent, WindowResult

LEDGER_FORMAT_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS batches (
    batch_id INTEGER PRIMARY KEY,
    status TEXT NOT NULL,
    committed_ms INTEGER NOT NULL,
    source_ids TEXT NOT NULL,
    record_count INTEGER NOT NULL,
    seq_cursor INTEGER NOT NULL,
    content_digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS windows (
    metric_key TEXT NOT NULL,
    window_index INTEGER NOT NULL,
    start_source_id TEXT NOT NULL,
    end_source_id TEXT NOT NULL,
    start_seq INTEGER NOT NULL,
    end_seq INTEGER NOT NULL,
    count INTEGER NOT NULL,
    sum_q INTEGER NOT NULL,
    mean_q INTEGER NOT NULL,
    value_error_bound_q INTEGER NOT NULL,
    sum_error_bound_q INTEGER NOT NULL,
    scale INTEGER NOT NULL,
    truncation TEXT NOT NULL,
    saturation TEXT NOT NULL,
    window_version INTEGER NOT NULL,
    status TEXT NOT NULL,
    input_digest TEXT NOT NULL,
    result_digest TEXT NOT NULL,
    lease_id TEXT,
    batch_id INTEGER,
    PRIMARY KEY (metric_key, window_index)
);
CREATE TABLE IF NOT EXISTS leases (
    lease_id TEXT PRIMARY KEY,
    metric_key TEXT NOT NULL,
    window_index INTEGER NOT NULL,
    start_seq INTEGER NOT NULL,
    end_seq INTEGER NOT NULL,
    granted_ms INTEGER NOT NULL,
    expiry_ms INTEGER NOT NULL,
    version INTEGER NOT NULL,
    state TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS propagation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id INTEGER NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cursors (
    scope TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    last_seq INTEGER NOT NULL,
    last_line_no INTEGER NOT NULL,
    batch_id INTEGER NOT NULL
);
"""


@dataclass(frozen=True, slots=True)
class SnapshotIdentity:
    source_set_version: int
    format_versions: tuple[tuple[str, int], ...]
    scale_convention_version: int
    session_scale: int
    content_digest: str


def _result_to_row(result: WindowResult,
                   batch_id: Optional[int]) -> tuple[Any, ...]:
    return (
        result.metric_key, result.window_index, result.start_source_id,
        result.end_source_id, result.start_seq, result.end_seq, result.count,
        result.sum_q, result.mean_q, result.value_error_bound_q,
        result.sum_error_bound_q, result.scale, result.truncation,
        result.saturation, result.window_version, result.status,
        result.input_digest, result.result_digest, result.lease_id, batch_id,
    )


def _row_to_result(row: sqlite3.Row) -> WindowResult:
    return WindowResult(
        metric_key=row["metric_key"],
        window_index=row["window_index"],
        start_source_id=row["start_source_id"],
        end_source_id=row["end_source_id"],
        start_seq=row["start_seq"],
        end_seq=row["end_seq"],
        count=row["count"],
        sum_q=row["sum_q"],
        mean_q=row["mean_q"],
        value_error_bound_q=row["value_error_bound_q"],
        sum_error_bound_q=row["sum_error_bound_q"],
        scale=row["scale"],
        truncation=row["truncation"],
        saturation=row["saturation"],
        window_version=row["window_version"],
        status=row["status"],
        input_digest=row["input_digest"],
        result_digest=row["result_digest"],
        lease_id=row["lease_id"],
    )


def _lease_to_row(lease: Lease) -> tuple[Any, ...]:
    return (
        lease.lease_id, lease.metric_key, lease.window_index, lease.start_seq,
        lease.end_seq, lease.granted_ms, lease.expiry_ms, lease.version,
        lease.state,
    )


def _row_to_lease(row: sqlite3.Row) -> Lease:
    return Lease(
        lease_id=row["lease_id"],
        metric_key=row["metric_key"],
        window_index=row["window_index"],
        start_seq=row["start_seq"],
        end_seq=row["end_seq"],
        granted_ms=row["granted_ms"],
        expiry_ms=row["expiry_ms"],
        version=row["version"],
        state=row["state"],
    )


class Ledger:
    def __init__(self, path: str):
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------ meta
    def set_meta(self, key: str, value: Any) -> None:
        self._conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value, sort_keys=True)))

    def get_meta(self, key: str, default: Any = None) -> Any:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        return json.loads(row["value"])

    def snapshot_identity(self) -> Optional[SnapshotIdentity]:
        digest = self.get_meta("content_digest")
        if digest is None:
            return None
        return SnapshotIdentity(
            source_set_version=int(self.get_meta("source_set_version")),
            format_versions=tuple(
                sorted((item["source_id"], item["format_version"])
                       for item in self.get_meta("format_versions"))),
            scale_convention_version=int(
                self.get_meta("scale_convention_version")),
            session_scale=int(self.get_meta("session_scale")),
            content_digest=digest,
        )

    def write_identity(self, identity: SnapshotIdentity) -> None:
        self.set_meta("source_set_version", identity.source_set_version)
        self.set_meta("format_versions",
                      [{"source_id": s, "format_version": v}
                       for s, v in identity.format_versions])
        self.set_meta("scale_convention_version",
                      identity.scale_convention_version)
        self.set_meta("session_scale", identity.session_scale)
        self.set_meta("content_digest", identity.content_digest)
        self.set_meta("ledger_format_version", LEDGER_FORMAT_VERSION)

    @staticmethod
    def build_identity(source_set_version: int,
                       format_versions: dict[str, int],
                       scale_convention_version: int,
                       session_scale: int,
                       window_payload: list[dict[str, Any]]) -> SnapshotIdentity:
        digest = content_digest({
            "source_set_version": source_set_version,
            "format_versions": sorted(format_versions.items()),
            "scale_convention_version": scale_convention_version,
            "session_scale": session_scale,
            "windows": window_payload,
        })
        return SnapshotIdentity(
            source_set_version=source_set_version,
            format_versions=tuple(sorted(format_versions.items())),
            scale_convention_version=scale_convention_version,
            session_scale=session_scale,
            content_digest=digest,
        )

    # -------------------------------------------------------------- batches
    def begin_batch(self, batch_id: int, source_ids: list[str],
                    record_count: int, seq_cursor: int,
                    digest: str, committed_ms: int) -> None:
        self._conn.execute(
            "INSERT INTO batches(batch_id, status, committed_ms, source_ids, "
            "record_count, seq_cursor, content_digest) "
            "VALUES(?, 'in_progress', ?, ?, ?, ?, ?)",
            (batch_id, committed_ms, json.dumps(sorted(source_ids)),
             record_count, seq_cursor, digest))

    def commit_batch(self, batch_id: int,
                     windows: list[WindowResult],
                     leases: list[Lease],
                     propagation: list[PropagationEvent],
                     cursors: dict[str, tuple[int, int]],
                     committed_ms: int) -> None:
        """Atomically commit every artefact of one batch."""
        try:
            self._conn.execute(
                "UPDATE batches SET status='committed', committed_ms=? "
                "WHERE batch_id=?", (committed_ms, batch_id))
            for result in windows:
                self._conn.execute(
                    "INSERT INTO windows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, "
                    "?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(metric_key, window_index) DO UPDATE SET "
                    "start_source_id=excluded.start_source_id, "
                    "end_source_id=excluded.end_source_id, "
                    "start_seq=excluded.start_seq, end_seq=excluded.end_seq, "
                    "count=excluded.count, sum_q=excluded.sum_q, "
                    "mean_q=excluded.mean_q, "
                    "value_error_bound_q=excluded.value_error_bound_q, "
                    "sum_error_bound_q=excluded.sum_error_bound_q, "
                    "scale=excluded.scale, truncation=excluded.truncation, "
                    "saturation=excluded.saturation, "
                    "window_version=excluded.window_version, "
                    "status=excluded.status, "
                    "input_digest=excluded.input_digest, "
                    "result_digest=excluded.result_digest, "
                    "lease_id=excluded.lease_id, batch_id=excluded.batch_id",
                    _result_to_row(result, batch_id))
            for lease in leases:
                self._conn.execute(
                    "INSERT INTO leases VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(lease_id) DO UPDATE SET "
                    "metric_key=excluded.metric_key, "
                    "window_index=excluded.window_index, "
                    "start_seq=excluded.start_seq, end_seq=excluded.end_seq, "
                    "granted_ms=excluded.granted_ms, "
                    "expiry_ms=excluded.expiry_ms, version=excluded.version, "
                    "state=excluded.state",
                    _lease_to_row(lease))
            for event in propagation:
                self._conn.execute(
                    "INSERT INTO propagation(batch_id, payload) VALUES(?, ?)",
                    (batch_id, json.dumps(asdict(event), sort_keys=True)))
            for source_id, (last_seq, last_line) in cursors.items():
                self._conn.execute(
                    "INSERT INTO cursors(scope, source_id, last_seq, "
                    "last_line_no, batch_id) VALUES('source', ?, ?, ?, ?) "
                    "ON CONFLICT(scope) DO UPDATE SET source_id=excluded.source_id, "
                    "last_seq=excluded.last_seq, last_line_no=excluded.last_line_no, "
                    "batch_id=excluded.batch_id",
                    (source_id, last_seq, last_line, batch_id))
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def latest_batch_id(self) -> int:
        row = self._conn.execute(
            "SELECT COALESCE(MAX(batch_id), 0) AS max_id FROM batches"
        ).fetchone()
        return int(row["max_id"])

    def in_progress_batches(self) -> list[sqlite3.Row]:
        return list(self._conn.execute(
            "SELECT * FROM batches WHERE status='in_progress' "
            "ORDER BY batch_id"))

    # -------------------------------------------------------------- reads
    def all_windows(self) -> list[WindowResult]:
        rows = self._conn.execute(
            "SELECT * FROM windows ORDER BY metric_key, window_index")
        return [_row_to_result(row) for row in rows]

    def all_leases(self) -> list[Lease]:
        rows = self._conn.execute(
            "SELECT * FROM leases ORDER BY metric_key, window_index")
        return [_row_to_lease(row) for row in rows]

    def propagation_trail(self) -> list[PropagationEvent]:
        rows = self._conn.execute(
            "SELECT payload FROM propagation ORDER BY id")
        events = []
        for row in rows:
            data = json.loads(row["payload"])
            data["affected_windows"] = [
                tuple(item) for item in data["affected_windows"]]
            data["affected_windows"] = tuple(data["affected_windows"])
            events.append(PropagationEvent(**data))
        return events

    def cursors(self) -> dict[str, tuple[int, int]]:
        rows = self._conn.execute("SELECT * FROM cursors ORDER BY source_id")
        return {row["source_id"]: (row["last_seq"], row["last_line_no"])
                for row in rows}

    def mark_windows_unknown(self,
                             refs: Iterable[tuple[str, int]]) -> None:
        refs = list(refs)
        if not refs:
            return
        try:
            for metric_key, window_index in refs:
                self._conn.execute(
                    "UPDATE windows SET status='unknown', lease_id=NULL "
                    "WHERE metric_key=? AND window_index=?",
                    (metric_key, window_index))
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def replace_state(self, backup_dir: str) -> str:
        """Whole-state replacement with a read-only backup (no merging)."""
        self._conn.close()
        os.makedirs(backup_dir, exist_ok=True)
        backup_path = os.path.join(
            backup_dir,
            f"ledger-backup-{content_digest(os.path.basename(self.path))[:12]}-"
            f"{os.getpid()}.db")
        shutil.copy2(self.path, backup_path)
        os.chmod(backup_path, 0o444)
        os.remove(self.path)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        return backup_path

    def verify_content_digest(self, expected_identity: SnapshotIdentity
                              ) -> None:
        """Raise E_SNAPSHOT when the stored digest fails verification."""
        stored = self.snapshot_identity()
        if stored is None:
            raise SnapshotError("snapshot has no content digest",
                                field="content_digest")
        if stored.content_digest != expected_identity.content_digest:
            raise SnapshotError(
                "snapshot content digest does not match current state",
                field="content_digest",
                detail={"stored": stored.content_digest,
                        "expected": expected_identity.content_digest})
        if stored.source_set_version != expected_identity.source_set_version:
            raise SnapshotError("source set version mismatch",
                                field="source_set_version")
        if stored.format_versions != expected_identity.format_versions:
            raise SnapshotError("format version set mismatch",
                                field="format_versions")
        if stored.scale_convention_version != \
                expected_identity.scale_convention_version:
            raise SnapshotError("scale convention version mismatch",
                                field="scale_convention_version")

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def canonical_window_payload(
            windows: list[WindowResult]) -> list[dict[str, Any]]:
        return [
            {
                "metric_key": w.metric_key,
                "window_index": w.window_index,
                "start_seq": w.start_seq,
                "end_seq": w.end_seq,
                "count": w.count,
                "sum_q": w.sum_q,
                "mean_q": w.mean_q,
                "value_error_bound_q": w.value_error_bound_q,
                "sum_error_bound_q": w.sum_error_bound_q,
                "window_version": w.window_version,
                "status": w.status,
                "result_digest": w.result_digest,
            }
            for w in sorted(windows, key=lambda w: (w.metric_key,
                                                    w.window_index))
        ]
