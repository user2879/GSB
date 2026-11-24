"""SQLite receipt ledger with versioned, whole-snapshot replacement recovery.

Every completed batch is written in one transaction covering attribution,
receipt summary, invalidation trace, budget consumption and the progress
cursor. The ledger header carries an integer format version and a content
digest. On reopen, worker-set version, batch-size version and content digest are
compared against the live specification: a mismatch triggers a read-only backup
plus a full replacement (never an incremental merge). Batches present in the
plan but never fully committed are reported as ``unknown`` and must be rerun.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from typing import Any

from .canonical import sha256_hex
from .errors import SnapshotError

LEDGER_FORMAT_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS header (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    format_version INTEGER NOT NULL,
    worker_set_version TEXT NOT NULL,
    batch_size_version TEXT NOT NULL,
    content_digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS batches (
    batch_id TEXT PRIMARY KEY,
    worker_id TEXT NOT NULL,
    batch_index INTEGER NOT NULL,
    request_ids TEXT NOT NULL,
    arrival_seqs TEXT NOT NULL,
    remainder INTEGER NOT NULL,
    rationale TEXT NOT NULL,
    batch_version INTEGER NOT NULL,
    status TEXT NOT NULL,
    output_digest TEXT NOT NULL,
    param_version INTEGER NOT NULL,
    param_digest TEXT NOT NULL,
    intermediate_dim INTEGER NOT NULL,
    elapsed_ms INTEGER NOT NULL,
    wait_started_ms INTEGER NOT NULL,
    wait_ended_ms INTEGER NOT NULL,
    attempts INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS receipts (
    batch_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    source_version TEXT NOT NULL,
    normalized_json TEXT NOT NULL,
    PRIMARY KEY (batch_id, request_id)
);
CREATE TABLE IF NOT EXISTS invalidation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT NOT NULL,
    step_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS budget_usage (
    batch_id TEXT PRIMARY KEY,
    elapsed_ms INTEGER NOT NULL,
    attempts INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS progress (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    last_committed_batch TEXT,
    committed_count INTEGER NOT NULL
);
"""


@dataclass(frozen=True)
class SnapshotHeader:
    format_version: int
    worker_set_version: str
    batch_size_version: str
    content_digest: str


class SnapshotLedger:
    def __init__(self, path: str, connection: sqlite3.Connection, header: SnapshotHeader) -> None:
        self.path = os.path.abspath(path)
        self.conn = connection
        self.header = header

    # ------------------------------------------------------------------ open
    @classmethod
    def create(cls, path: str, spec) -> "SnapshotLedger":
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        if os.path.exists(path):
            os.remove(path)
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA)
        header = SnapshotHeader(
            format_version=LEDGER_FORMAT_VERSION,
            worker_set_version=spec.worker_set_fingerprint(),
            batch_size_version=spec.batch_size_fingerprint(),
            content_digest=spec.content_fingerprint(),
        )
        conn.execute(
            "INSERT INTO header VALUES (1, ?, ?, ?, ?)",
            (
                header.format_version,
                header.worker_set_version,
                header.batch_size_version,
                header.content_digest,
            ),
        )
        conn.execute("INSERT INTO progress VALUES (1, NULL, 0)")
        conn.commit()
        return cls(path, conn, header)

    @classmethod
    def open(cls, path: str, spec, *, on_mismatch: str = "replace") -> "SnapshotLedger":
        """Open an existing snapshot.

        ``on_mismatch`` is ``"replace"`` (backup + full replacement) or
        ``"raise"`` (surface ``E_SNAPSHOT``, used by snapshot-digest injection).
        """

        if not os.path.exists(path):
            return cls.create(path, spec)
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA journal_mode=WAL")
        row = conn.execute(
            "SELECT format_version, worker_set_version, batch_size_version, content_digest "
            "FROM header WHERE id = 1"
        ).fetchone()
        if row is None:
            raise SnapshotError("ledger header missing", entry=path, field="header")
        stored = SnapshotHeader(*row)
        expected = SnapshotHeader(
            format_version=LEDGER_FORMAT_VERSION,
            worker_set_version=spec.worker_set_fingerprint(),
            batch_size_version=spec.batch_size_fingerprint(),
            content_digest=spec.content_fingerprint(),
        )
        mismatches = [
            name
            for name, left, right in (
                ("format_version", stored.format_version, expected.format_version),
                ("worker_set_version", stored.worker_set_version, expected.worker_set_version),
                ("batch_size_version", stored.batch_size_version, expected.batch_size_version),
                ("content_digest", stored.content_digest, expected.content_digest),
            )
            if left != right
        ]
        if mismatches:
            if on_mismatch == "raise":
                conn.close()
                raise SnapshotError(
                    "snapshot version/content digest mismatch",
                    entry=path,
                    field=",".join(mismatches),
                    stored={
                        "format_version": stored.format_version,
                        "worker_set_version": stored.worker_set_version,
                        "batch_size_version": stored.batch_size_version,
                        "content_digest": stored.content_digest,
                    },
                )
            conn.close()
            backup_path = cls._backup(path)
            fresh = cls.create(path, spec)
            fresh.backup_path = backup_path
            fresh.replaced_fields = tuple(mismatches)
            return fresh
        return cls(path, conn, stored)

    @staticmethod
    def _backup(path: str) -> str:
        backup_path = f"{path}.readonly.bak"
        counter = 1
        while os.path.exists(backup_path):
            backup_path = f"{path}.readonly.{counter}.bak"
            counter += 1
        shutil.copy2(path, backup_path)
        os.chmod(backup_path, 0o444)
        return os.path.abspath(backup_path)

    def close(self) -> None:
        self.conn.close()

    def update_header(self, spec) -> SnapshotHeader:
        """Move the snapshot header to the current specification version.

        Used after a validated invalidating rerun so the ledger stays
        self-consistent; whole-snapshot replacement remains reserved for
        foreign snapshots that cannot be incrementally reconciled.
        """

        self.header = SnapshotHeader(
            format_version=LEDGER_FORMAT_VERSION,
            worker_set_version=spec.worker_set_fingerprint(),
            batch_size_version=spec.batch_size_fingerprint(),
            content_digest=spec.content_fingerprint(),
        )
        self.conn.execute(
            "UPDATE header SET format_version = ?, worker_set_version = ?, "
            "batch_size_version = ?, content_digest = ? WHERE id = 1",
            (
                self.header.format_version,
                self.header.worker_set_version,
                self.header.batch_size_version,
                self.header.content_digest,
            ),
        )
        self.conn.commit()
        return self.header

    # -------------------------------------------------------------- commits
    def commit_batch(
        self,
        batch,
        forward_result,
        normalized_receipts,
        invalidation_steps,
        timing,
    ) -> None:
        """Atomically persist one fully completed batch."""

        from .canonical import canonical_json

        with closing(self.conn.cursor()) as cursor:
            cursor.execute("BEGIN")
            try:
                cursor.execute(
                    "INSERT OR REPLACE INTO batches VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        batch.batch_id,
                        batch.worker_id,
                        batch.batch_index,
                        canonical_json(list(batch.request_ids)).decode("utf-8"),
                        canonical_json(list(batch.arrival_seqs)).decode("utf-8"),
                        1 if batch.remainder else 0,
                        batch.rationale,
                        forward_result.param_version,
                        "committed",
                        forward_result.output_digest,
                        forward_result.param_version,
                        forward_result.param_digest,
                        forward_result.intermediate_dim,
                        timing["elapsed_ms"],
                        timing["wait_started_ms"],
                        timing["wait_ended_ms"],
                        timing["attempts"],
                    ),
                )
                cursor.execute("DELETE FROM receipts WHERE batch_id = ?", (batch.batch_id,))
                for receipt in normalized_receipts:
                    cursor.execute(
                        "INSERT INTO receipts VALUES (?, ?, ?, ?)",
                        (
                            batch.batch_id,
                            receipt.request_id,
                            receipt.source_version,
                            canonical_json(receipt.to_dict()).decode("utf-8"),
                        ),
                    )
                cursor.execute("DELETE FROM invalidation WHERE batch_id = ?", (batch.batch_id,))
                for step in invalidation_steps:
                    cursor.execute(
                        "INSERT INTO invalidation (batch_id, step_json) VALUES (?, ?)",
                        (batch.batch_id, canonical_json(step.to_dict()).decode("utf-8")),
                    )
                cursor.execute(
                    "INSERT OR REPLACE INTO budget_usage VALUES (?, ?, ?)",
                    (batch.batch_id, timing["elapsed_ms"], timing["attempts"]),
                )
                committed_count = cursor.execute(
                    "SELECT COUNT(*) FROM batches WHERE status = 'committed'"
                ).fetchone()[0]
                cursor.execute(
                    "UPDATE progress SET last_committed_batch = ?, committed_count = ? WHERE id = 1",
                    (batch.batch_id, committed_count),
                )
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    # ------------------------------------------------------------- queries
    def committed_batch_ids(self) -> tuple[str, ...]:
        rows = self.conn.execute(
            "SELECT batch_id FROM batches WHERE status = 'committed' ORDER BY worker_id, batch_index"
        ).fetchall()
        return tuple(row[0] for row in rows)

    def unknown_batches(self, plan) -> tuple[str, ...]:
        """Planned batches lacking a full commit; these must be rerun."""

        committed = set(self.committed_batch_ids())
        return tuple(
            batch.batch_id
            for batch in sorted(plan.batches, key=lambda item: (item.worker_id, item.batch_index))
            if batch.batch_id not in committed
        )

    def load_batch(self, batch_id: str) -> dict[str, Any] | None:
        cursor = self.conn.execute("SELECT * FROM batches WHERE batch_id = ?", (batch_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        columns = [info[0] for info in cursor.description]
        return dict(zip(columns, row))

    def load_receipts(self, batch_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT normalized_json FROM receipts WHERE batch_id = ? ORDER BY request_id",
            (batch_id,),
        ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def load_invalidation(self, batch_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT step_json FROM invalidation WHERE batch_id = ? ORDER BY id",
            (batch_id,),
        ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def total_budget_ms(self) -> int:
        row = self.conn.execute("SELECT COALESCE(SUM(elapsed_ms), 0) FROM budget_usage").fetchone()
        return int(row[0])

    def recompute_content_digest(self) -> str:
        """Digest over committed content, independent of the stored header."""

        payload = {
            "batches": [
                list(row)
                for row in self.conn.execute(
                    "SELECT batch_id, worker_id, batch_index, output_digest, param_version, "
                    "param_digest, intermediate_dim, elapsed_ms, status FROM batches "
                    "ORDER BY batch_id"
                ).fetchall()
            ],
            "receipts": [
                list(row)
                for row in self.conn.execute(
                    "SELECT batch_id, request_id, source_version, normalized_json "
                    "FROM receipts ORDER BY batch_id, request_id"
                ).fetchall()
            ],
        }
        return sha256_hex(payload)

    def tamper_header_digest(self) -> None:
        """Injection helper: corrupt the stored content digest."""

        self.conn.execute(
            "UPDATE header SET content_digest = ? WHERE id = 1",
            ("tampered-" + self.header.content_digest[:24],),
        )
        self.conn.commit()
