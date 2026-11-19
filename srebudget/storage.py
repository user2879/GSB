"""SQLite persistence for crash-safe batches and whole-state snapshots."""

from __future__ import annotations

import copy
import json
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .engine import DiagnosticEngine
from .errors import SreError, snapshot_error
from .models import ChainSpec, TraceStream, canonical_json, sha256_json

SNAPSHOT_SCHEMA = "srebudget.snapshot.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class RecoveryReport:
    session_id: str
    committed_cursor: int
    recovered_batches: tuple[dict[str, Any], ...]
    unknown_batches: tuple[dict[str, Any], ...]

    @property
    def rerun_required(self) -> bool:
        return bool(self.unknown_batches)

    def as_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "committed_cursor": self.committed_cursor,
            "recovered_batches": list(self.recovered_batches),
            "unknown_batches": list(self.unknown_batches),
            "rerun_required": self.rerun_required,
        }


@dataclass(frozen=True)
class SnapshotRecord:
    snapshot_id: str
    session_id: str
    chain_version: str
    chain_digest: str
    trace_digest: str
    cursor: int
    clock_ms: int
    state: dict[str, Any]
    chain: dict[str, Any]
    conclusion_summary: dict[str, Any]
    content_digest: str
    created_at: str

    def as_dict(self, *, include_state: bool = False) -> dict[str, Any]:
        value = {
            "snapshot_id": self.snapshot_id,
            "session_id": self.session_id,
            "chain_version": self.chain_version,
            "chain_digest": self.chain_digest,
            "trace_digest": self.trace_digest,
            "cursor": self.cursor,
            "clock_ms": self.clock_ms,
            "conclusion_summary": self.conclusion_summary,
            "content_digest": self.content_digest,
            "created_at": self.created_at,
        }
        if include_state:
            value["state"] = self.state
            value["chain"] = self.chain
        return value


@dataclass(frozen=True)
class RestoreResult:
    snapshot_id: str
    mode: str
    reason: str
    affected_services: tuple[str, ...]
    invalidated_conclusions: int
    cursor: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "mode": self.mode,
            "reason": self.reason,
            "affected_services": list(self.affected_services),
            "invalidated_conclusions": self.invalidated_conclusions,
            "cursor": self.cursor,
        }


class SessionStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path), timeout=30)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = FULL")
        self._create_schema()

    def __enter__(self) -> "SessionStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                chain_version TEXT NOT NULL,
                chain_digest TEXT NOT NULL,
                trace_digest TEXT NOT NULL,
                total_calls INTEGER NOT NULL,
                committed_cursor INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS batches (
                session_id TEXT NOT NULL,
                batch_no INTEGER NOT NULL,
                start_call INTEGER NOT NULL,
                end_call_exclusive INTEGER NOT NULL,
                status TEXT NOT NULL,
                conclusions_json TEXT,
                state_json TEXT,
                state_digest TEXT,
                started_at TEXT NOT NULL,
                committed_at TEXT,
                PRIMARY KEY (session_id, batch_no),
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE TABLE IF NOT EXISTS conclusions (
                session_id TEXT NOT NULL,
                call_index INTEGER NOT NULL,
                batch_no INTEGER NOT NULL,
                status TEXT NOT NULL,
                decision_json TEXT NOT NULL,
                valid INTEGER NOT NULL DEFAULT 1,
                invalidation_reason TEXT,
                PRIMARY KEY (session_id, call_index),
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE TABLE IF NOT EXISTS unknown_batches (
                session_id TEXT NOT NULL,
                batch_no INTEGER NOT NULL,
                start_call INTEGER NOT NULL,
                end_call_exclusive INTEGER NOT NULL,
                reason TEXT NOT NULL,
                detected_at TEXT NOT NULL,
                PRIMARY KEY (session_id, batch_no),
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE TABLE IF NOT EXISTS unknown_calls (
                session_id TEXT NOT NULL,
                call_index INTEGER NOT NULL,
                batch_no INTEGER NOT NULL,
                reason TEXT NOT NULL,
                detected_at TEXT NOT NULL,
                PRIMARY KEY (session_id, call_index),
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE TABLE IF NOT EXISTS session_checkpoints (
                session_id TEXT PRIMARY KEY,
                cursor INTEGER NOT NULL,
                state_json TEXT NOT NULL,
                state_digest TEXT NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE TABLE IF NOT EXISTS snapshots (
                snapshot_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                chain_version TEXT NOT NULL,
                chain_digest TEXT NOT NULL,
                trace_digest TEXT NOT NULL,
                cursor INTEGER NOT NULL,
                clock_ms INTEGER NOT NULL,
                state_json TEXT NOT NULL,
                chain_json TEXT NOT NULL,
                conclusion_summary_json TEXT NOT NULL,
                content_digest TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE TABLE IF NOT EXISTS snapshot_restorations (
                restoration_id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                mode TEXT NOT NULL,
                reason TEXT NOT NULL,
                affected_services_json TEXT NOT NULL,
                invalidated_conclusions INTEGER NOT NULL,
                cursor INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (snapshot_id) REFERENCES snapshots(snapshot_id)
            );
            """
        )
        self.connection.commit()

    # ------------------------------------------------------------------
    # Sessions and batches
    # ------------------------------------------------------------------
    def create_session(
        self,
        chain: ChainSpec,
        trace: TraceStream,
        *,
        session_id: str | None = None,
    ) -> str:
        sid = session_id or f"session-{secrets.token_hex(6)}"
        now = _utc_now()
        try:
            self.connection.execute(
                """
                INSERT INTO sessions(
                    session_id, chain_version, chain_digest, trace_digest,
                    total_calls, committed_cursor, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (sid, chain.version, chain.digest, trace.content_digest, len(trace), now, now),
            )
            self.connection.commit()
        except sqlite3.IntegrityError as exc:
            raise SreError(
                "E_SESSION",
                "会话标识已存在",
                {"session_id": sid},
            ) from exc
        return sid

    def get_session(self, session_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            raise SreError("E_SESSION", "会话不存在", {"session_id": session_id})
        return dict(row)

    def update_session_identity(
        self, session_id: str, chain: ChainSpec, trace: TraceStream
    ) -> None:
        self.connection.execute(
            """
            UPDATE sessions
            SET chain_version = ?, chain_digest = ?, trace_digest = ?,
                total_calls = ?, updated_at = ?
            WHERE session_id = ?
            """,
            (
                chain.version,
                chain.digest,
                trace.content_digest,
                len(trace),
                _utc_now(),
                session_id,
            ),
        )
        self.connection.commit()

    def next_batch_no(self, session_id: str) -> int:
        row = self.connection.execute(
            "SELECT COALESCE(MAX(batch_no), 0) + 1 AS value FROM batches WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row["value"])

    def start_batch_attempt(
        self, session_id: str, *, start_call: int, end_call_exclusive: int
    ) -> int:
        session = self.get_session(session_id)
        if start_call != int(session["committed_cursor"]) + 1:
            raise SreError(
                "E_SESSION",
                "批次起点与已提交游标不连续",
                {
                    "session_id": session_id,
                    "expected_start": int(session["committed_cursor"]) + 1,
                    "actual_start": start_call,
                },
            )
        if end_call_exclusive <= start_call:
            raise SreError("E_SESSION", "批次范围为空", {"session_id": session_id})
        batch_no = self.next_batch_no(session_id)
        try:
            self.connection.execute(
                """
                INSERT INTO batches(
                    session_id, batch_no, start_call, end_call_exclusive,
                    status, started_at
                ) VALUES (?, ?, ?, ?, 'started', ?)
                """,
                (session_id, batch_no, start_call, end_call_exclusive, _utc_now()),
            )
            self.connection.commit()
        except sqlite3.Error:
            self.connection.rollback()
            raise
        return batch_no

    def commit_batch(
        self,
        session_id: str,
        batch_no: int,
        decisions: Iterable[dict[str, Any]],
        engine: DiagnosticEngine,
    ) -> None:
        decision_rows = list(decisions)
        if not decision_rows:
            raise SreError("E_SESSION", "批次没有调用结论", {"session_id": session_id})
        state = engine.export_state()
        state_json = canonical_json(state)
        state_digest = sha256_json(state)
        conclusions_json = canonical_json(decision_rows)
        end_cursor = max(int(item["call_index"]) for item in decision_rows)

        try:
            self.connection.execute("BEGIN IMMEDIATE")
            batch = self.connection.execute(
                "SELECT * FROM batches WHERE session_id = ? AND batch_no = ?",
                (session_id, batch_no),
            ).fetchone()
            if batch is None or batch["status"] != "started":
                raise SreError(
                    "E_SESSION",
                    "批次不处于 started 状态",
                    {"session_id": session_id, "batch_no": batch_no},
                )
            expected_start = int(batch["start_call"])
            if int(decision_rows[0]["call_index"]) != expected_start:
                raise SreError(
                    "E_SESSION",
                    "批次调用序号与预留范围不一致",
                    {"expected": expected_start, "actual": decision_rows[0]["call_index"]},
                )
            if end_cursor + 1 != int(batch["end_call_exclusive"]):
                raise SreError(
                    "E_SESSION",
                    "批次结束调用序号与预留范围不一致",
                    {
                        "expected_exclusive": int(batch["end_call_exclusive"]),
                        "actual_exclusive": end_cursor + 1,
                    },
                )
            for item in decision_rows:
                call_index = int(item["call_index"])
                status = (
                    "fast_failed"
                    if bool(item.get("fast_failed"))
                    else str(item.get("outcome", "ok"))
                )
                self.connection.execute(
                    """
                    INSERT INTO conclusions(
                        session_id, call_index, batch_no, status, decision_json,
                        valid, invalidation_reason
                    ) VALUES (?, ?, ?, ?, ?, 1, NULL)
                    ON CONFLICT(session_id, call_index) DO UPDATE SET
                        batch_no = excluded.batch_no,
                        status = excluded.status,
                        decision_json = excluded.decision_json,
                        valid = 1,
                        invalidation_reason = NULL
                    """,
                    (
                        session_id,
                        call_index,
                        batch_no,
                        status,
                        canonical_json(item),
                    ),
                )
            self.connection.execute(
                """
                UPDATE batches
                SET status = 'committed', conclusions_json = ?, state_json = ?,
                    state_digest = ?, committed_at = ?
                WHERE session_id = ? AND batch_no = ?
                """,
                (
                    conclusions_json,
                    state_json,
                    state_digest,
                    _utc_now(),
                    session_id,
                    batch_no,
                ),
            )
            self.connection.execute(
                """
                UPDATE sessions SET committed_cursor = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (end_cursor, _utc_now(), session_id),
            )
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.rollback()
            raise

    def load_latest_state(self, session_id: str) -> dict[str, Any] | None:
        batch_row = self.connection.execute(
            """
            SELECT state_json, end_call_exclusive, batch_no FROM batches
            WHERE session_id = ? AND status = 'committed'
            ORDER BY end_call_exclusive DESC, batch_no DESC
            LIMIT 1
            """,
            (session_id,),
        ).fetchone()
        checkpoint_row = self.connection.execute(
            "SELECT state_json, cursor FROM session_checkpoints WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if batch_row is None and checkpoint_row is None:
            return None
        if checkpoint_row is not None:
            if batch_row is None or int(checkpoint_row["cursor"]) >= int(
                batch_row["end_call_exclusive"]
            ):
                return json.loads(checkpoint_row["state_json"])
        return json.loads(batch_row["state_json"])

    def save_checkpoint(
        self,
        session_id: str,
        cursor: int,
        state_json: str,
        state_digest: str,
        reason: str,
        *,
        commit: bool = True,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO session_checkpoints(
                session_id, cursor, state_json, state_digest, reason, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                cursor = excluded.cursor,
                state_json = excluded.state_json,
                state_digest = excluded.state_digest,
                reason = excluded.reason,
                created_at = excluded.created_at
            """,
            (session_id, cursor, state_json, state_digest, reason, _utc_now()),
        )
        if commit:
            self.connection.commit()

    def clear_checkpoint(self, session_id: str, *, commit: bool = True) -> None:
        self.connection.execute(
            "DELETE FROM session_checkpoints WHERE session_id = ?", (session_id,)
        )
        if commit:
            self.connection.commit()

    def recover(self, session_id: str) -> RecoveryReport:
        now = _utc_now()
        unknown_batches: list[dict[str, Any]] = []
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            started = self.connection.execute(
                """
                SELECT * FROM batches
                WHERE session_id = ? AND status = 'started'
                ORDER BY batch_no ASC
                """,
                (session_id,),
            ).fetchall()
            for batch in started:
                self.connection.execute(
                    """
                    UPDATE batches SET status = 'unknown'
                    WHERE session_id = ? AND batch_no = ?
                    """,
                    (session_id, batch["batch_no"]),
                )
                self.connection.execute(
                    """
                    INSERT OR IGNORE INTO unknown_batches(
                        session_id, batch_no, start_call, end_call_exclusive,
                        reason, detected_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        batch["batch_no"],
                        batch["start_call"],
                        batch["end_call_exclusive"],
                        "uncommitted_batch_after_process_termination",
                        now,
                    ),
                )
                call_indexes: list[int] = []
                for call_index in range(
                    int(batch["start_call"]), int(batch["end_call_exclusive"])
                ):
                    existing = self.connection.execute(
                        "SELECT 1 FROM conclusions WHERE session_id = ? AND call_index = ?",
                        (session_id, call_index),
                    ).fetchone()
                    if existing is None:
                        call_indexes.append(call_index)
                        self.connection.execute(
                            """
                            INSERT OR IGNORE INTO unknown_calls(
                                session_id, call_index, batch_no, reason, detected_at
                            ) VALUES (?, ?, ?, ?, ?)
                            """,
                            (
                                session_id,
                                call_index,
                                batch["batch_no"],
                                "call_not_in_committed_transaction",
                                now,
                            ),
                        )
                unknown_batches.append(
                    {
                        "batch_no": int(batch["batch_no"]),
                        "start_call": int(batch["start_call"]),
                        "end_call_exclusive": int(batch["end_call_exclusive"]),
                        "unknown_call_indexes": call_indexes,
                        "requires_rerun": bool(call_indexes),
                    }
                )
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.rollback()
            raise

        session = self.get_session(session_id)
        recovered_rows = self.connection.execute(
            """
            SELECT batch_no, start_call, end_call_exclusive, committed_at
            FROM batches
            WHERE session_id = ? AND status = 'committed'
            ORDER BY batch_no ASC
            """,
            (session_id,),
        ).fetchall()
        recovered_batches = tuple(
            {
                "batch_no": int(row["batch_no"]),
                "start_call": int(row["start_call"]),
                "end_call_exclusive": int(row["end_call_exclusive"]),
                "committed_at": row["committed_at"],
            }
            for row in recovered_rows
        )
        return RecoveryReport(
            session_id=session_id,
            committed_cursor=int(session["committed_cursor"]),
            recovered_batches=recovered_batches,
            unknown_batches=tuple(unknown_batches),
        )

    def list_unknown(self, session_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT batch_no, start_call, end_call_exclusive, reason, detected_at
            FROM unknown_batches WHERE session_id = ? ORDER BY batch_no ASC
            """,
            (session_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def invalidate_conclusions_after(
        self, session_id: str, cursor: int, reason: str, *, commit: bool = True
    ) -> int:
        result = self.connection.execute(
            """
            UPDATE conclusions
            SET valid = 0, invalidation_reason = ?
            WHERE session_id = ? AND call_index > ? AND valid = 1
            """,
            (reason, session_id, cursor),
        )
        if commit:
            self.connection.commit()
        return int(result.rowcount)

    def invalidate_all_conclusions(
        self, session_id: str, reason: str, *, commit: bool = True
    ) -> int:
        result = self.connection.execute(
            """
            UPDATE conclusions
            SET valid = 0, invalidation_reason = ?
            WHERE session_id = ? AND valid = 1
            """,
            (reason, session_id),
        )
        if commit:
            self.connection.commit()
        return int(result.rowcount)

    def set_cursor(self, session_id: str, cursor: int, *, commit: bool = True) -> None:
        self.connection.execute(
            "UPDATE sessions SET committed_cursor = ?, updated_at = ? WHERE session_id = ?",
            (cursor, _utc_now(), session_id),
        )
        if commit:
            self.connection.commit()

    def list_conclusions(
        self,
        session_id: str,
        *,
        valid_only: bool = True,
        limit: int | None = None,
        descending: bool = False,
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM conclusions WHERE session_id = ?"
        params: list[Any] = [session_id]
        if valid_only:
            sql += " AND valid = 1"
        sql += " ORDER BY call_index " + ("DESC" if descending else "ASC")
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        rows = self.connection.execute(sql, params).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            result.append(
                {
                    "call_index": int(row["call_index"]),
                    "batch_no": int(row["batch_no"]),
                    "status": row["status"],
                    "valid": bool(row["valid"]),
                    "invalidation_reason": row["invalidation_reason"],
                    "decision": json.loads(row["decision_json"]),
                }
            )
        return result

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------
    def _snapshot_content(self, record: dict[str, Any]) -> dict[str, Any]:
        return {
            "snapshot_schema": SNAPSHOT_SCHEMA,
            "chain_version": record["chain_version"],
            "chain_digest": record["chain_digest"],
            "trace_digest": record["trace_digest"],
            "cursor": int(record["cursor"]),
            "clock_ms": int(record["clock_ms"]),
            "state": record["state"],
            "chain": record["chain"],
            "conclusion_summary": record["conclusion_summary"],
        }

    def _conclusion_summary(self, session_id: str, cursor: int) -> dict[str, Any]:
        rows = self.connection.execute(
            """
            SELECT status, decision_json FROM conclusions
            WHERE session_id = ? AND call_index <= ? AND valid = 1
            ORDER BY call_index ASC
            """,
            (session_id, cursor),
        ).fetchall()
        by_outcome: dict[str, int] = {}
        fast_failed_calls = 0
        fast_failed_services = 0
        for row in rows:
            decision = json.loads(row["decision_json"])
            outcome = str(decision.get("outcome", "unknown"))
            by_outcome[outcome] = by_outcome.get(outcome, 0) + 1
            if decision.get("fast_failed"):
                fast_failed_calls += 1
                fast_failed_services += len(decision.get("fast_fail_chain", []))
        return {
            "cursor": cursor,
            "conclusion_count": len(rows),
            "by_outcome": dict(sorted(by_outcome.items())),
            "fast_failed_calls": fast_failed_calls,
            "fast_failed_services": fast_failed_services,
            "first_call_index": (
                int(json.loads(rows[0]["decision_json"])["call_index"])
                if rows
                else None
            ),
            "last_call_index": (
                int(json.loads(rows[-1]["decision_json"])["call_index"]) if rows else None
            ),
        }

    def create_snapshot(
        self,
        session_id: str,
        engine: DiagnosticEngine,
        chain: ChainSpec,
        trace: TraceStream,
    ) -> SnapshotRecord:
        self.get_session(session_id)
        row = self.connection.execute(
            "SELECT COALESCE(MAX(CAST(SUBSTR(snapshot_id, 6) AS INTEGER)), 0) AS value FROM snapshots"
        ).fetchone()
        snapshot_id = f"snap-{int(row['value']) + 1:06d}"
        cursor = engine.cursor
        summary = self._conclusion_summary(session_id, cursor)
        state = engine.export_state()
        record = {
            "snapshot_id": snapshot_id,
            "session_id": session_id,
            "chain_version": chain.version,
            "chain_digest": chain.digest,
            "trace_digest": trace.content_digest,
            "cursor": cursor,
            "clock_ms": int(state.get("clock_ms", 0)),
            "state": state,
            "chain": chain.canonical_dict(),
            "conclusion_summary": summary,
        }
        content_digest = sha256_json(self._snapshot_content(record))
        created_at = _utc_now()
        try:
            self.connection.execute(
                """
                INSERT INTO snapshots(
                    snapshot_id, session_id, chain_version, chain_digest,
                    trace_digest, cursor, clock_ms, state_json, chain_json,
                    conclusion_summary_json, content_digest, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    session_id,
                    chain.version,
                    chain.digest,
                    trace.content_digest,
                    cursor,
                    int(state.get("clock_ms", 0)),
                    canonical_json(state),
                    canonical_json(chain.canonical_dict()),
                    canonical_json(summary),
                    content_digest,
                    created_at,
                ),
            )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        return SnapshotRecord(
            snapshot_id=snapshot_id,
            session_id=session_id,
            chain_version=chain.version,
            chain_digest=chain.digest,
            trace_digest=trace.content_digest,
            cursor=cursor,
            clock_ms=int(state.get("clock_ms", 0)),
            state=state,
            chain=chain.canonical_dict(),
            conclusion_summary=summary,
            content_digest=content_digest,
            created_at=created_at,
        )

    def list_snapshots(self, session_id: str | None = None) -> list[dict[str, Any]]:
        if session_id is None:
            rows = self.connection.execute(
                "SELECT * FROM snapshots ORDER BY created_at ASC, snapshot_id ASC"
            ).fetchall()
        else:
            rows = self.connection.execute(
                """
                SELECT * FROM snapshots WHERE session_id = ?
                ORDER BY created_at ASC, snapshot_id ASC
                """,
                (session_id,),
            ).fetchall()
        return [
            {
                "snapshot_id": row["snapshot_id"],
                "session_id": row["session_id"],
                "chain_version": row["chain_version"],
                "chain_digest": row["chain_digest"],
                "trace_digest": row["trace_digest"],
                "cursor": int(row["cursor"]),
                "clock_ms": int(row["clock_ms"]),
                "conclusion_summary": json.loads(row["conclusion_summary_json"]),
                "content_digest": row["content_digest"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def load_snapshot(self, snapshot_id: str) -> SnapshotRecord:
        row = self.connection.execute(
            "SELECT * FROM snapshots WHERE snapshot_id = ?", (snapshot_id,)
        ).fetchone()
        if row is None:
            raise snapshot_error(
                "快照不存在", snapshot_id=snapshot_id, reason="snapshot_not_found"
            )
        state = json.loads(row["state_json"])
        chain = json.loads(row["chain_json"])
        summary = json.loads(row["conclusion_summary_json"])
        record = {
            "snapshot_id": row["snapshot_id"],
            "session_id": row["session_id"],
            "chain_version": row["chain_version"],
            "chain_digest": row["chain_digest"],
            "trace_digest": row["trace_digest"],
            "cursor": int(row["cursor"]),
            "clock_ms": int(row["clock_ms"]),
            "state": state,
            "chain": chain,
            "conclusion_summary": summary,
        }
        actual_digest = sha256_json(self._snapshot_content(record))
        expected_digest = row["content_digest"]
        if actual_digest != expected_digest:
            raise snapshot_error(
                "快照内容摘要不符",
                snapshot_id=snapshot_id,
                field_path="content_digest",
                expected=expected_digest,
                actual=actual_digest,
            )
        return SnapshotRecord(
            snapshot_id=snapshot_id,
            session_id=row["session_id"],
            chain_version=row["chain_version"],
            chain_digest=row["chain_digest"],
            trace_digest=row["trace_digest"],
            cursor=int(row["cursor"]),
            clock_ms=int(row["clock_ms"]),
            state=state,
            chain=chain,
            conclusion_summary=summary,
            content_digest=expected_digest,
            created_at=row["created_at"],
        )

    def corrupt_snapshot_digest(
        self, snapshot_id: str, replacement: str = "0" * 64
    ) -> None:
        result = self.connection.execute(
            "UPDATE snapshots SET content_digest = ? WHERE snapshot_id = ?",
            (replacement, snapshot_id),
        )
        self.connection.commit()
        if result.rowcount != 1:
            raise snapshot_error("快照不存在", snapshot_id=snapshot_id)

    def _state_service_map(self, state: dict[str, Any]) -> dict[str, Any]:
        return {
            item["id"]: item
            for item in state.get("services", [])
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }

    def _restore_reason(
        self,
        snapshot: SnapshotRecord,
        chain: ChainSpec,
        trace: TraceStream,
    ) -> str | None:
        if snapshot.chain_version != chain.version:
            return "chain_version_changed"
        if snapshot.chain_digest != chain.digest:
            old_budget = {
                service.get("id"): service.get("budget", {})
                for service in snapshot.chain.get("services", [])
            }
            new_budget = chain.budget_signature()
            if old_budget != new_budget:
                return "window_parameters_changed"
            return "chain_content_changed"
        if snapshot.trace_digest != trace.content_digest:
            return "trace_content_changed"
        return None

    def restore_snapshot(
        self,
        snapshot_id: str,
        engine: DiagnosticEngine,
        chain: ChainSpec,
        trace: TraceStream,
    ) -> RestoreResult:
        snapshot = self.load_snapshot(snapshot_id)
        session_id = snapshot.session_id
        reason = self._restore_reason(snapshot, chain, trace)
        current_state = engine.export_state()
        current_services = self._state_service_map(current_state)
        snapshot_services = self._state_service_map(snapshot.state)

        if reason is not None:
            affected = sorted(
                set(current_services)
                | set(snapshot_services)
                | set(chain.service_ids)
            )
            try:
                self.connection.execute("BEGIN IMMEDIATE")
                invalidated = self.invalidate_all_conclusions(
                    session_id, reason, commit=False
                )
                self.connection.execute(
                    "UPDATE batches SET status = 'superseded' WHERE session_id = ?",
                    (session_id,),
                )
                self.connection.execute(
                    """
                    UPDATE sessions SET chain_version = ?, chain_digest = ?, trace_digest = ?,
                        total_calls = ?, committed_cursor = 0, updated_at = ?
                    WHERE session_id = ?
                    """,
                    (
                        chain.version,
                        chain.digest,
                        trace.content_digest,
                        len(trace),
                        _utc_now(),
                        session_id,
                    ),
                )
                self.connection.execute(
                    """
                    INSERT INTO snapshot_restorations(
                        snapshot_id, session_id, mode, reason,
                        affected_services_json, invalidated_conclusions,
                        cursor, created_at
                    ) VALUES (?, ?, 'whole_state_reset', ?, ?, ?, 0, ?)
                    """,
                    (
                        snapshot_id,
                        session_id,
                        reason,
                        canonical_json(affected),
                        invalidated,
                        _utc_now(),
                    ),
                )
                fresh_state = DiagnosticEngine(chain, engine.clock).export_state()
                self.save_checkpoint(
                    session_id,
                    0,
                    canonical_json(fresh_state),
                    sha256_json(fresh_state),
                    reason,
                    commit=False,
                )
                self.connection.execute("COMMIT")
            except Exception:
                self.connection.rollback()
                raise
            engine.reset()
            return RestoreResult(
                snapshot_id=snapshot_id,
                mode="whole_state_reset",
                reason=reason,
                affected_services=tuple(affected),
                invalidated_conclusions=invalidated,
                cursor=0,
            )

        affected = sorted(
            service_id
            for service_id in set(current_services) | set(snapshot_services)
            if current_services.get(service_id) != snapshot_services.get(service_id)
        )
        same_state = sha256_json(current_state) == sha256_json(snapshot.state)
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            invalidated = self.invalidate_conclusions_after(
                session_id,
                snapshot.cursor,
                "snapshot_restore_rewinds_conclusions",
                commit=False,
            )
            self.connection.execute(
                """
                UPDATE batches SET status = 'superseded'
                WHERE session_id = ? AND end_call_exclusive > ?
                """,
                (session_id, snapshot.cursor),
            )
            self.connection.execute(
                """
                UPDATE sessions SET committed_cursor = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (snapshot.cursor, _utc_now(), session_id),
            )
            mode = "unchanged" if same_state else "state_replaced"
            self.connection.execute(
                """
                INSERT INTO snapshot_restorations(
                    snapshot_id, session_id, mode, reason,
                    affected_services_json, invalidated_conclusions,
                    cursor, created_at
                ) VALUES (?, ?, ?, 'snapshot_state_whole_replacement', ?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    session_id,
                    mode,
                    canonical_json(affected),
                    invalidated,
                    snapshot.cursor,
                    _utc_now(),
                ),
            )
            self.save_checkpoint(
                session_id,
                snapshot.cursor,
                canonical_json(snapshot.state),
                sha256_json(snapshot.state),
                "snapshot_restore",
                commit=False,
            )
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.rollback()
            raise

        engine.replace_state(copy.deepcopy(snapshot.state))
        return RestoreResult(
            snapshot_id=snapshot_id,
            mode=mode,
            reason="snapshot_state_whole_replacement",
            affected_services=tuple(affected),
            invalidated_conclusions=invalidated,
            cursor=snapshot.cursor,
        )
