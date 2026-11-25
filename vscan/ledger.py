from __future__ import annotations

import os
import shutil
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .canonical import canonical_json, digest_value
from .errors import DivergenceError, LedgerError, RecoveryBatch, RecoveryReport
from .executor import BatchResult
from .planner import BatchPlan


LEDGER_FORMAT_VERSION = 1
SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS batches (
  query_id TEXT NOT NULL,
  segment_id TEXT NOT NULL,
  batch_no INTEGER NOT NULL,
  start_row INTEGER NOT NULL,
  end_row INTEGER NOT NULL,
  row_count INTEGER NOT NULL,
  state TEXT NOT NULL,
  plan_digest TEXT NOT NULL,
  fingerprint TEXT NOT NULL DEFAULT '',
  passed_rows INTEGER NOT NULL DEFAULT 0,
  decoded_bytes INTEGER NOT NULL DEFAULT 0,
  wait_start_monotonic_ms INTEGER NOT NULL DEFAULT 0,
  wait_end_monotonic_ms INTEGER NOT NULL DEFAULT 0,
  wait_ms INTEGER NOT NULL DEFAULT 0,
  aggregate_digest TEXT NOT NULL DEFAULT '',
  aggregate_json TEXT NOT NULL DEFAULT '{}',
  progress_cursor TEXT NOT NULL DEFAULT '{}',
  committed_at_ms INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (query_id, segment_id, batch_no)
);
"""


@dataclass(frozen=True)
class LedgerOpenResult:
    replaced: bool
    backup_path: str | None
    snapshot_digest: str
    content_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "replaced": self.replaced,
            "backup_path": self.backup_path,
            "snapshot_digest": self.snapshot_digest,
            "content_digest": self.content_digest,
            "format_version": LEDGER_FORMAT_VERSION,
        }


class Ledger:
    def __init__(
        self,
        path: str | Path,
        *,
        snapshot_digest: str,
        spec_digest: str,
    ) -> None:
        self.path = Path(path)
        self.snapshot_digest = snapshot_digest
        self.spec_digest = spec_digest
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self.open_result = self._open()

    def _open(self) -> LedgerOpenResult:
        needs_replace = False
        old_content = ""
        reason = ""
        backup_path: str | None = None
        if self.path.exists():
            try:
                conn = self._connect(self.path)
                meta = self._read_meta(conn)
                if int(meta.get("format_version", -1)) != LEDGER_FORMAT_VERSION:
                    needs_replace = True
                    reason = "format version mismatch"
                elif meta.get("snapshot_digest") != self.snapshot_digest:
                    needs_replace = True
                    reason = "snapshot digest mismatch"
                else:
                    actual = self._compute_content_digest(conn)
                    old_content = meta.get("content_digest", "")
                    if actual != old_content:
                        needs_replace = True
                        reason = "content digest mismatch"
                if needs_replace:
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                conn.close()
            except (sqlite3.DatabaseError, ValueError, OSError):
                needs_replace = True
                reason = "unreadable snapshot"
        else:
            needs_replace = True
            reason = "new ledger"

        if needs_replace:
            if self.path.exists() and reason != "new ledger":
                backup_path = str(self._make_backup(old_content or reason))
            self._replace_database()
            self._conn = self._connect(self.path)
            content = self._read_meta(self._conn).get("content_digest", "")
            return LedgerOpenResult(True, backup_path, self.snapshot_digest, content)

        self._conn = self._connect(self.path)
        content = self._read_meta(self._conn).get("content_digest", "")
        return LedgerOpenResult(False, None, self.snapshot_digest, content)

    @staticmethod
    def _connect(path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(path, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @staticmethod
    def _read_meta(conn: sqlite3.Connection) -> dict[str, str]:
        try:
            rows = conn.execute("SELECT key, value FROM meta").fetchall()
        except sqlite3.DatabaseError:
            return {}
        return {str(row["key"]): str(row["value"]) for row in rows}

    def _replace_database(self) -> None:
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(self.path) + suffix)
            if sidecar.exists():
                sidecar.unlink()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(f".{self.path.name}.init-{os.getpid()}-{time.monotonic_ns()}")
        if temp.exists():
            temp.unlink()
        conn = sqlite3.connect(temp)
        try:
            conn.executescript(SCHEMA)
            conn.executemany(
                "INSERT INTO meta(key, value) VALUES(?, ?)",
                [
                    ("format_version", str(LEDGER_FORMAT_VERSION)),
                    ("snapshot_digest", self.snapshot_digest),
                    ("spec_digest", self.spec_digest),
                    ("content_digest", digest_value([])),
                    ("progress_seq", "0"),
                    ("progress_cursor", "{}"),
                ],
            )
            conn.execute(f"PRAGMA user_version={LEDGER_FORMAT_VERSION}")
            conn.commit()
        finally:
            conn.close()
        os.replace(temp, self.path)

    def _make_backup(self, material: str) -> Path:
        suffix = digest_value({"material": material, "path_time": time.time_ns()})[:12]
        backup = self.path.with_name(f"{self.path.name}.backup-{suffix}")
        counter = 0
        while backup.exists():
            counter += 1
            backup = self.path.with_name(f"{self.path.name}.backup-{suffix}-{counter}")
        shutil.copy2(self.path, backup)
        os.chmod(backup, 0o444)
        return backup

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise LedgerError("ledger is closed")
        return self._conn

    @staticmethod
    def _committed_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
        rows = conn.execute(
            """
            SELECT query_id, segment_id, batch_no, start_row, end_row, row_count,
                   state, plan_digest, fingerprint, passed_rows, decoded_bytes,
                   aggregate_digest, aggregate_json, progress_cursor
            FROM batches WHERE state = 'committed'
            ORDER BY query_id, segment_id, batch_no
            """
        ).fetchall()
        return [
            {
                "query_id": row["query_id"],
                "segment_id": row["segment_id"],
                "batch_no": row["batch_no"],
                "start_row": row["start_row"],
                "end_row": row["end_row"],
                "row_count": row["row_count"],
                "state": row["state"],
                "plan_digest": row["plan_digest"],
                "fingerprint": row["fingerprint"],
                "passed_rows": row["passed_rows"],
                "decoded_bytes": row["decoded_bytes"],
                "aggregate_digest": row["aggregate_digest"],
                "aggregate_json": row["aggregate_json"],
                "progress_cursor": row["progress_cursor"],
            }
            for row in rows
        ]

    @classmethod
    def _compute_content_digest(cls, conn: sqlite3.Connection) -> str:
        return digest_value(cls._committed_rows(conn))

    def begin_batch(self, batch: BatchPlan, fingerprint: str) -> None:
        with self._lock:
            with self.conn:
                self.conn.execute(
                    """
                    INSERT INTO batches(query_id, segment_id, batch_no, start_row, end_row,
                                        row_count, state, plan_digest, fingerprint)
                    VALUES(?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                    ON CONFLICT(query_id, segment_id, batch_no) DO UPDATE SET
                      start_row=excluded.start_row,
                      end_row=excluded.end_row,
                      row_count=excluded.row_count,
                      state=CASE WHEN batches.state='committed' THEN batches.state ELSE 'pending' END,
                      plan_digest=CASE WHEN batches.state='committed' THEN batches.plan_digest ELSE excluded.plan_digest END,
                      fingerprint=CASE WHEN batches.state='committed' THEN batches.fingerprint ELSE excluded.fingerprint END,
                      passed_rows=CASE WHEN batches.state='committed' THEN batches.passed_rows ELSE 0 END,
                      decoded_bytes=CASE WHEN batches.state='committed' THEN batches.decoded_bytes ELSE 0 END,
                      aggregate_digest=CASE WHEN batches.state='committed' THEN batches.aggregate_digest ELSE '' END,
                      aggregate_json=CASE WHEN batches.state='committed' THEN batches.aggregate_json ELSE '{}' END,
                      progress_cursor=CASE WHEN batches.state='committed' THEN batches.progress_cursor ELSE '{}' END,
                      committed_at_ms=CASE WHEN batches.state='committed' THEN batches.committed_at_ms ELSE 0 END
                    """,
                    (
                        batch.query_id,
                        batch.segment_id,
                        batch.batch_no,
                        batch.start_row,
                        batch.end_row,
                        batch.rows,
                        batch.plan_digest,
                        fingerprint,
                    ),
                )

    def commit_batch(self, result: BatchResult) -> None:
        batch = result.plan
        aggregate_payload = result.computation.aggregate_payload()
        aggregate_json = canonical_json(aggregate_payload)
        committed_at_ms = time.time_ns() // 1_000_000
        with self._lock:
            conn = self.conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                sequence = int(self._read_meta(conn).get("progress_seq", "0")) + 1
                cursor = canonical_json(
                    {
                        "seq": sequence,
                        "query_id": batch.query_id,
                        "segment_id": batch.segment_id,
                        "batch_no": batch.batch_no,
                    }
                )
                conn.execute(
                    """
                    INSERT INTO batches(
                      query_id, segment_id, batch_no, start_row, end_row, row_count,
                      state, plan_digest, fingerprint, passed_rows, decoded_bytes,
                      wait_start_monotonic_ms, wait_end_monotonic_ms, wait_ms,
                      aggregate_digest, aggregate_json, progress_cursor, committed_at_ms
                    ) VALUES(?, ?, ?, ?, ?, ?, 'committed', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(query_id, segment_id, batch_no) DO UPDATE SET
                      start_row=excluded.start_row,
                      end_row=excluded.end_row,
                      row_count=excluded.row_count,
                      state='committed',
                      plan_digest=excluded.plan_digest,
                      fingerprint=excluded.fingerprint,
                      passed_rows=excluded.passed_rows,
                      decoded_bytes=excluded.decoded_bytes,
                      wait_start_monotonic_ms=excluded.wait_start_monotonic_ms,
                      wait_end_monotonic_ms=excluded.wait_end_monotonic_ms,
                      wait_ms=excluded.wait_ms,
                      aggregate_digest=excluded.aggregate_digest,
                      aggregate_json=excluded.aggregate_json,
                      progress_cursor=excluded.progress_cursor,
                      committed_at_ms=excluded.committed_at_ms
                    """,
                    (
                        batch.query_id,
                        batch.segment_id,
                        batch.batch_no,
                        batch.start_row,
                        batch.end_row,
                        batch.rows,
                        batch.plan_digest,
                        result.fingerprint,
                        result.computation.passed_rows,
                        result.computation.actual_decoded_bytes,
                        result.wait_start_ms,
                        result.wait_end_ms,
                        result.wait_ms,
                        result.computation.aggregate_digest,
                        aggregate_json,
                        cursor,
                        committed_at_ms,
                    ),
                )
                content_digest = self._compute_content_digest(conn)
                conn.executemany(
                    "UPDATE meta SET value=? WHERE key=?",
                    [
                        (str(sequence), "progress_seq"),
                        (cursor, "progress_cursor"),
                        (content_digest, "content_digest"),
                    ],
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def reconcile_expected(
        self,
        expected: Iterable[tuple[BatchPlan, str]],
    ) -> RecoveryReport:
        expected_by_key = {batch.key: (batch, fingerprint) for batch, fingerprint in expected}
        report = RecoveryReport()
        with self._lock:
            conn = self.conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                all_rows = {
                    (row["query_id"], row["segment_id"], row["batch_no"]): row
                    for row in conn.execute("SELECT * FROM batches").fetchall()
                }
                for key, (batch, fingerprint) in sorted(expected_by_key.items()):
                    row = all_rows.get(key)
                    if row is None or row["state"] != "committed":
                        state = "unknown"
                        if row is not None and row["state"] == "pending":
                            detail = "pending transaction was not committed; rerun required"
                        elif row is None:
                            detail = "no committed ledger record; rerun required"
                        else:
                            detail = f"ledger state {row['state']!r} is not committed; rerun required"
                        conn.execute(
                            """
                            INSERT INTO batches(query_id, segment_id, batch_no, start_row,
                                                end_row, row_count, state, plan_digest, fingerprint)
                            VALUES(?, ?, ?, ?, ?, ?, 'unknown', ?, ?)
                            ON CONFLICT(query_id, segment_id, batch_no) DO UPDATE SET
                              state='unknown', passed_rows=0, decoded_bytes=0,
                              aggregate_digest='', aggregate_json='{}',
                              progress_cursor='{}', committed_at_ms=0
                            """,
                            (
                                batch.query_id,
                                batch.segment_id,
                                batch.batch_no,
                                batch.start_row,
                                batch.end_row,
                                batch.rows,
                                batch.plan_digest,
                                fingerprint,
                            ),
                        )
                        report.unknown.append(
                            RecoveryBatch(
                                batch.query_id,
                                batch.segment_id,
                                batch.batch_no,
                                state,
                                detail,
                            )
                        )
                    else:
                        if row["plan_digest"] != batch.plan_digest or row["fingerprint"] != fingerprint:
                            conn.execute(
                                """
                                UPDATE batches SET state='unknown', plan_digest=?, fingerprint=?,
                                  passed_rows=0, decoded_bytes=0, aggregate_digest='',
                                  aggregate_json='{}', progress_cursor='{}', committed_at_ms=0
                                WHERE query_id=? AND segment_id=? AND batch_no=?
                                """,
                                (
                                    batch.plan_digest,
                                    fingerprint,
                                    batch.query_id,
                                    batch.segment_id,
                                    batch.batch_no,
                                ),
                            )
                            report.unknown.append(
                                RecoveryBatch(
                                    batch.query_id,
                                    batch.segment_id,
                                    batch.batch_no,
                                    "unknown",
                                    "committed plan/fingerprint differs from current state; rerun required",
                                )
                            )
                        else:
                            report.committed.append(
                                RecoveryBatch(
                                    batch.query_id,
                                    batch.segment_id,
                                    batch.batch_no,
                                    "committed",
                                    f"fingerprint={fingerprint}",
                                )
                            )
                content_digest = self._compute_content_digest(conn)
                conn.execute(
                    "UPDATE meta SET value=? WHERE key='content_digest'",
                    (content_digest,),
                )
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        return report

    def committed_records(self) -> dict[tuple[str, str, int], dict[str, Any]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM batches WHERE state='committed' ORDER BY query_id, segment_id, batch_no"
            ).fetchall()
        return {
            (row["query_id"], row["segment_id"], row["batch_no"]): dict(row)
            for row in rows
        }

    def current_content_digest(self) -> str:
        with self._lock:
            return self._compute_content_digest(self.conn)

    def verify_results(self, results: Iterable[BatchResult]) -> None:
        records = self.committed_records()
        for result in sorted(results, key=lambda item: item.key):
            record = records.get(result.key)
            if record is None:
                raise DivergenceError(
                    "state",
                    segment_id=result.plan.segment_id,
                    batch_no=result.plan.batch_no,
                    expected="committed",
                    actual="missing",
                    message="replay comparison found a missing committed batch",
                )
            self.verify_result_against_record(result, record)

    def verify_result_against_record(self, result: BatchResult, record: Mapping[str, Any]) -> None:
        comparisons = {
                "start_row": (result.plan.start_row, record["start_row"]),
                "end_row": (result.plan.end_row, record["end_row"]),
                "plan_digest": (result.plan.plan_digest, record["plan_digest"]),
                "fingerprint": (result.fingerprint, record["fingerprint"]),
                "passed_rows": (result.computation.passed_rows, record["passed_rows"]),
                "decoded_bytes": (
                    result.computation.actual_decoded_bytes,
                    record["decoded_bytes"],
                ),
                "aggregate_digest": (
                    result.computation.aggregate_digest,
                    record["aggregate_digest"],
                ),
        }
        for field, (expected, actual) in comparisons.items():
            if expected != actual:
                raise DivergenceError(
                    field,
                    segment_id=result.plan.segment_id,
                    batch_no=result.plan.batch_no,
                    expected=expected,
                    actual=actual,
                    message="recomputed/replayed batch differs from the committed ledger record",
                )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            meta = self._read_meta(self.conn)
            rows = self.committed_records()
        return {
            "format_version": LEDGER_FORMAT_VERSION,
            "meta": meta,
            "batches": [rows[key] for key in sorted(rows)],
        }

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
