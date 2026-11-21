"""Local SQLite ledger with batch-atomic commits and crash recovery.

Durability contract
-------------------
Every processed sample batch is persisted in a *single* transaction that
inserts both the batch progress cursor and the folded weights accumulated so
far. A forced kill can therefore leave only two observable states:

* the batch's transaction fully committed -> the batch is recoverable;
* nothing from the batch landed -> samples past the last committed cursor are
  marked ``unknown`` and must be re-run.

The ledger stores integer format versions and content digests so that stale or
foreign state can never be silently merged into a new session.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

from .errors import LedgerError
from .folding import STACK_SEP, FoldResult, FrameMetric
from .fixedpoint import ShareReport
from .ranking import FrameContext, FrameRank

LEDGER_FORMAT_VERSION = 1
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BatchRecord:
    batch_seq: int
    start_line: int
    end_line: int
    sample_count: int
    weight_added: int
    committed_total_weight: int


@dataclass(frozen=True)
class RecoveryInfo:
    recoverable_batches: int
    last_committed_line: int
    committed_weight: int
    unknown_after_line: int
    needs_rerun_from_line: int


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(
            str(self.path), isolation_level=None)  # autocommit, explicit txn
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    # ------------------------------------------------------------------ schema
    def _create_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS batches (
                batch_seq INTEGER PRIMARY KEY,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                sample_count INTEGER NOT NULL,
                weight_added INTEGER NOT NULL,
                committed_total_weight INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS stack_weights (
                stack_key TEXT PRIMARY KEY,
                leaf TEXT NOT NULL,
                weight INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS frame_metrics (
                frame TEXT PRIMARY KEY,
                self_weight INTEGER NOT NULL,
                cumulative_weight INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS edges (
                frame TEXT NOT NULL,
                direction TEXT NOT NULL CHECK (direction IN ('caller','callee')),
                counterpart TEXT NOT NULL,
                edge_weight INTEGER NOT NULL,
                call_count INTEGER NOT NULL,
                PRIMARY KEY (frame, direction, counterpart)
            );
            CREATE TABLE IF NOT EXISTS ranks (
                rank INTEGER PRIMARY KEY,
                frame TEXT NOT NULL,
                self_weight INTEGER NOT NULL,
                cumulative_weight INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS annotations (
                frame TEXT NOT NULL,
                kind TEXT NOT NULL,
                payload TEXT NOT NULL,
                PRIMARY KEY (frame, kind)
            );
            """
        )

    # -------------------------------------------------------------------- meta
    def init_session(self, session_id: str, interval_canonical: str,
                     interval_version: int, sample_digest: str,
                     extra: Optional[Mapping[str, object]] = None) -> None:
        existing = self.get_meta("session_id")
        if existing is not None and existing != session_id:
            raise LedgerError(
                f"ledger already bound to session {existing!r}; use a fresh "
                "path or replace the snapshot",
                field="session_id",
            )
        payload = {
            "session_id": session_id,
            "ledger_format_version": LEDGER_FORMAT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "interval_canonical": interval_canonical,
            "interval_version": interval_version,
            "sample_digest": sample_digest,
        }
        if extra:
            payload.update(extra)
        self.conn.execute("BEGIN")
        try:
            for key, value in payload.items():
                self.conn.execute(
                    "INSERT INTO meta(key, value) VALUES(?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (key, json.dumps(value, sort_keys=True)),
                )
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    def get_meta(self, key: str) -> Optional[object]:
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def all_meta(self) -> dict:
        return {
            key: json.loads(value)
            for key, value in self.conn.execute(
                "SELECT key, value FROM meta ORDER BY key")
        }

    # -------------------------------------------------------------- batch write
    def commit_batch(self, batch_seq: int, start_line: int, end_line: int,
                     sample_count: int, weight_added: int,
                     stack_deltas: Mapping[Tuple[str, ...], int],
                     frame_self_deltas: Mapping[str, int],
                     frame_cumulative_deltas: Mapping[str, int],
                     committed_total_weight: int) -> None:
        """Atomically persist one batch: cursor + all folded deltas."""

        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.conn.execute(
                "INSERT INTO batches(batch_seq, start_line, end_line, "
                "sample_count, weight_added, committed_total_weight) "
                "VALUES(?,?,?,?,?,?)",
                (batch_seq, start_line, end_line, sample_count,
                 weight_added, committed_total_weight),
            )
            for frames, weight in stack_deltas.items():
                key = STACK_SEP.join(frames)
                self.conn.execute(
                    "INSERT INTO stack_weights(stack_key, leaf, weight) "
                    "VALUES(?,?,?) ON CONFLICT(stack_key) DO UPDATE SET "
                    "weight=weight+excluded.weight",
                    (key, frames[-1], weight),
                )
            for frame, weight in frame_self_deltas.items():
                self.conn.execute(
                    "INSERT INTO frame_metrics(frame, self_weight, "
                    "cumulative_weight) VALUES(?, ?, 0) ON CONFLICT(frame) DO "
                    "UPDATE SET self_weight=self_weight+excluded.self_weight",
                    (frame, weight),
                )
            for frame, weight in frame_cumulative_deltas.items():
                self.conn.execute(
                    "INSERT INTO frame_metrics(frame, self_weight, "
                    "cumulative_weight) VALUES(?, 0, ?) ON CONFLICT(frame) DO "
                    "UPDATE SET cumulative_weight="
                    "cumulative_weight+excluded.cumulative_weight",
                    (frame, weight),
                )
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    # ------------------------------------------------------------- result write
    def replace_results(self, result: FoldResult,
                        ranks: Sequence[FrameRank],
                        contexts: Mapping[str, FrameContext],
                        report: ShareReport,
                        path_choice: Mapping[str, object]) -> None:
        """Persist final conclusions in one transaction (idempotent replace)."""

        self.conn.execute("BEGIN")
        try:
            self.conn.execute("DELETE FROM edges")
            self.conn.execute("DELETE FROM ranks")
            self.conn.execute("DELETE FROM annotations")
            for rank in ranks:
                self.conn.execute(
                    "INSERT INTO ranks(rank, frame, self_weight, "
                    "cumulative_weight) VALUES(?,?,?,?)",
                    (rank.rank, rank.frame, rank.self_weight,
                     rank.cumulative_weight),
                )
            for frame, context in contexts.items():
                for edge in context.callers:
                    self.conn.execute(
                        "INSERT INTO edges(frame, direction, counterpart, "
                        "edge_weight, call_count) VALUES(?,?,?,?,?)",
                        (frame, "caller", edge.counterpart, edge.edge_weight,
                         edge.call_count),
                    )
                for edge in context.callees:
                    self.conn.execute(
                        "INSERT INTO edges(frame, direction, counterpart, "
                        "edge_weight, call_count) VALUES(?,?,?,?,?)",
                        (frame, "callee", edge.counterpart, edge.edge_weight,
                         edge.call_count),
                    )
            for share in report.self_shares:
                self.conn.execute(
                    "INSERT INTO annotations(frame, kind, payload) "
                    "VALUES(?,?,?) ON CONFLICT(frame, kind) DO UPDATE SET "
                    "payload=excluded.payload",
                    (share.frame, "self_share",
                     json.dumps(share.__dict__, sort_keys=True)),
                )
            for share in report.cumulative_shares:
                self.conn.execute(
                    "INSERT INTO annotations(frame, kind, payload) "
                    "VALUES(?,?,?) ON CONFLICT(frame, kind) DO UPDATE SET "
                    "payload=excluded.payload",
                    (share.frame, "cumulative_share",
                     json.dumps(share.__dict__, sort_keys=True)),
                )
            for key, value in path_choice.items():
                self.conn.execute(
                    "INSERT INTO meta(key, value) VALUES(?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (key, json.dumps(value, sort_keys=True)),
                )
            self.conn.execute(
                "INSERT INTO meta(key, value) VALUES('error_affected_pairs',"
                " ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (json.dumps(report.error_affected_pairs),),
            )
            self.conn.execute("COMMIT")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

    # ------------------------------------------------------------------ reads
    def list_batches(self) -> List[BatchRecord]:
        rows = self.conn.execute(
            "SELECT batch_seq, start_line, end_line, sample_count, "
            "weight_added, committed_total_weight FROM batches "
            "ORDER BY batch_seq").fetchall()
        return [BatchRecord(*row) for row in rows]

    def load_stack_weights(self) -> dict:
        out = {}
        for key, leaf, weight in self.conn.execute(
                "SELECT stack_key, leaf, weight FROM stack_weights "
                "ORDER BY stack_key"):
            out[tuple(key.split(STACK_SEP))] = weight
        return out

    def load_frame_metrics(self) -> Tuple[FrameMetric, ...]:
        rows = self.conn.execute(
            "SELECT frame, self_weight, cumulative_weight FROM frame_metrics "
            "ORDER BY frame").fetchall()
        return tuple(FrameMetric(*row) for row in rows)

    def recovery_info(self) -> RecoveryInfo:
        batches = self.list_batches()
        if not batches:
            return RecoveryInfo(0, 0, 0, 0, 1)
        last = batches[-1]
        return RecoveryInfo(
            recoverable_batches=len(batches),
            last_committed_line=last.end_line,
            committed_weight=last.committed_total_weight,
            unknown_after_line=last.end_line,
            needs_rerun_from_line=last.end_line + 1,
        )

    def close(self) -> None:
        self.conn.close()
