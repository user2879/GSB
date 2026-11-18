"""SQLite persistence for sessions, current-generation views, and the ledger."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .errors import E_LEDGER, E_PLAN, E_STAGE, BusinessError
from .models import (
    EventDefinition,
    EventRuntime,
    LedgerEntry,
    Plan,
    SessionRecord,
    SessionView,
    StageRuntime,
    canonical_json,
)
from .plan import validate_plan_object


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    plan_id TEXT NOT NULL,
    plan_version INTEGER NOT NULL,
    plan_digest TEXT NOT NULL,
    clock_ms INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    generation INTEGER NOT NULL DEFAULT 1,
    last_clock_source TEXT NOT NULL DEFAULT 'logical'
);

CREATE TABLE IF NOT EXISTS plan_versions (
    session_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    plan_json TEXT NOT NULL,
    plan_digest TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 0,
    stale INTEGER NOT NULL DEFAULT 0,
    created_ms INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (session_id, version),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS phase_states (
    session_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    stage_seq INTEGER NOT NULL,
    status TEXT NOT NULL,
    skip_reason TEXT,
    business_code TEXT,
    skip_from_stage_seq INTEGER,
    unknown_reason TEXT,
    updated_ms INTEGER NOT NULL DEFAULT 0,
    stale INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (session_id, generation, stage_seq),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS events (
    session_id TEXT NOT NULL,
    generation INTEGER NOT NULL,
    injection_seq INTEGER NOT NULL,
    event_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    offset_ms INTEGER NOT NULL,
    stage_seq INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    source TEXT NOT NULL DEFAULT 'plan',
    stale INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (session_id, generation, injection_seq),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS ledger (
    ledger_pk INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    generation INTEGER NOT NULL,
    kind TEXT NOT NULL,
    stage_seq INTEGER,
    from_status TEXT,
    to_status TEXT,
    reason TEXT,
    code TEXT,
    logical_ms INTEGER NOT NULL,
    plan_version INTEGER NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}',
    is_complete INTEGER NOT NULL DEFAULT 1,
    UNIQUE (session_id, seq),
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);

CREATE INDEX IF NOT EXISTS idx_ledger_session_generation
    ON ledger(session_id, generation, seq);

CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_pk INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id TEXT NOT NULL UNIQUE,
    session_id TEXT NOT NULL,
    file_path TEXT NOT NULL UNIQUE,
    plan_version INTEGER NOT NULL,
    content_digest TEXT NOT NULL,
    created_ms INTEGER NOT NULL,
    generation INTEGER NOT NULL,
    read_only INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
"""


class Repository:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect()
        try:
            connection.executescript(SCHEMA)
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.db_path,
            timeout=30.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        return connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    @contextmanager
    def read_connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _next_seq(connection: sqlite3.Connection, session_id: str) -> int:
        row = connection.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 AS next_seq FROM ledger WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row["next_seq"])

    def session_exists(self, session_id: str) -> bool:
        with self.read_connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            return row is not None

    def create_session(
        self,
        session_id: str,
        plan: Plan,
        events: list[tuple[EventDefinition, str]],
        *,
        generation: int = 1,
    ) -> None:
        plan_json = canonical_json(plan.to_dict())
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if existing:
                raise BusinessError(
                    E_PLAN,
                    f"会话 {session_id} 已存在，拒绝重复初始化",
                    field_path="session_id",
                )
            connection.execute(
                """
                INSERT INTO sessions
                    (session_id, plan_id, plan_version, plan_digest, clock_ms, status, generation)
                VALUES (?, ?, ?, ?, 0, 'created', ?)
                """,
                (session_id, plan.plan_id, plan.version, plan.digest, generation),
            )
            connection.execute(
                """
                INSERT INTO plan_versions
                    (session_id, version, plan_json, plan_digest, active, stale, created_ms)
                VALUES (?, ?, ?, ?, 1, 0, 0)
                """,
                (session_id, plan.version, plan_json, plan.digest),
            )
            for stage in plan.stages:
                connection.execute(
                    """
                    INSERT INTO phase_states
                        (session_id, generation, stage_seq, status, updated_ms, stale)
                    VALUES (?, ?, ?, 'pending', 0, 0)
                    """,
                    (session_id, generation, stage.seq),
                )
            for injection_seq, (event, source) in enumerate(events, start=1):
                connection.execute(
                    """
                    INSERT INTO events
                        (session_id, generation, injection_seq, event_id, event_type,
                         offset_ms, stage_seq, status, source, stale)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, 0)
                    """,
                    (
                        session_id,
                        generation,
                        injection_seq,
                        event.event_id,
                        event.type,
                        event.offset_ms,
                        event.stage_seq,
                        source,
                    ),
                )
            seq = self._next_seq(connection, session_id)
            connection.execute(
                """
                INSERT INTO ledger
                    (session_id, seq, generation, kind, stage_seq, from_status, to_status,
                     reason, code, logical_ms, plan_version, details_json, is_complete)
                VALUES (?, ?, ?, 'session_created', NULL, NULL, NULL, ?, NULL,
                        0, ?, ?, 1)
                """,
                (
                    session_id,
                    seq,
                    generation,
                    "创建演练会话并整体装载当前预案",
                    plan.version,
                    canonical_json({"plan_id": plan.plan_id, "plan_digest": plan.digest}),
                ),
            )

    def get_session(self, session_id: str) -> SessionRecord | None:
        with self.read_connection() as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        if row is None:
            return None
        return SessionRecord(
            session_id=row["session_id"],
            plan_id=row["plan_id"],
            plan_version=int(row["plan_version"]),
            plan_digest=row["plan_digest"],
            clock_ms=int(row["clock_ms"]),
            status=row["status"],
            generation=int(row["generation"]),
        )

    def require_session(self, session_id: str) -> SessionRecord:
        session = self.get_session(session_id)
        if session is None:
            raise BusinessError(
                E_PLAN,
                f"会话 {session_id} 不存在",
                field_path="session_id",
            )
        return session

    def get_plan(self, session_id: str, version: int | None = None) -> Plan:
        session = self.require_session(session_id)
        target_version = version if version is not None else session.plan_version
        with self.read_connection() as connection:
            row = connection.execute(
                "SELECT plan_json FROM plan_versions WHERE session_id = ? AND version = ?",
                (session_id, target_version),
            ).fetchone()
        if row is None:
            raise BusinessError(
                E_PLAN,
                f"预案版本 {target_version} 不在会话 {session_id} 中",
                field_path="plan_version",
            )
        try:
            raw = json.loads(row["plan_json"])
        except json.JSONDecodeError as exc:
            raise BusinessError(
                E_PLAN,
                "会话内预案 JSON 已损坏",
                field_path="plan_json",
            ) from exc
        return validate_plan_object(raw)

    def max_plan_version(self, session_id: str) -> int:
        with self.read_connection() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(version), 0) AS max_version FROM plan_versions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return int(row["max_version"])

    def get_view(self, session_id: str) -> SessionView:
        session = self.require_session(session_id)
        plan = self.get_plan(session_id, session.plan_version)
        with self.read_connection() as connection:
            stage_rows = connection.execute(
                """
                SELECT * FROM phase_states
                WHERE session_id = ? AND generation = ? AND stale = 0
                ORDER BY stage_seq
                """,
                (session_id, session.generation),
            ).fetchall()
            event_rows = connection.execute(
                """
                SELECT * FROM events
                WHERE session_id = ? AND generation = ? AND stale = 0
                ORDER BY injection_seq
                """,
                (session_id, session.generation),
            ).fetchall()
        stages = [
            StageRuntime(
                seq=int(row["stage_seq"]),
                status=row["status"],
                skip_reason=row["skip_reason"],
                business_code=row["business_code"],
                skip_from_stage_seq=row["skip_from_stage_seq"],
                unknown_reason=row["unknown_reason"],
                updated_ms=int(row["updated_ms"]),
                stale=bool(row["stale"]),
            )
            for row in stage_rows
        ]
        events = [
            EventRuntime(
                injection_seq=int(row["injection_seq"]),
                event_id=row["event_id"],
                type=row["event_type"],
                offset_ms=int(row["offset_ms"]),
                stage_seq=int(row["stage_seq"]),
                status=row["status"],
                source=row["source"],
                stale=bool(row["stale"]),
            )
            for row in event_rows
        ]
        return SessionView(session=session, plan=plan, stages=stages, events=events)

    def get_stage_state(
        self, session_id: str, generation: int, stage_seq: int
    ) -> StageRuntime:
        with self.read_connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM phase_states
                WHERE session_id = ? AND generation = ? AND stage_seq = ? AND stale = 0
                """,
                (session_id, generation, stage_seq),
            ).fetchone()
        if row is None:
            raise BusinessError(
                E_STAGE,
                f"阶段 {stage_seq} 不存在",
                stage_seq=stage_seq,
                field_path="phase_states.stage_seq",
            )
        return StageRuntime(
            seq=int(row["stage_seq"]),
            status=row["status"],
            skip_reason=row["skip_reason"],
            business_code=row["business_code"],
            skip_from_stage_seq=row["skip_from_stage_seq"],
            unknown_reason=row["unknown_reason"],
            updated_ms=int(row["updated_ms"]),
            stale=bool(row["stale"]),
        )

    @staticmethod
    def _allowed_transition(from_status: str, to_status: str) -> bool:
        return (from_status, to_status) in {
            ("pending", "running"),
            ("pending", "skipped"),
            ("running", "succeeded"),
            ("running", "failed"),
            ("unknown", "running"),
        }

    def transition_stage(
        self,
        session_id: str,
        generation: int,
        stage_seq: int,
        to_status: str,
        *,
        reason: str,
        logical_ms: int,
        code: str | None = None,
        skip_from_stage_seq: int | None = None,
        details: dict[str, Any] | None = None,
        kind: str = "stage_transition",
    ) -> LedgerEntry:
        with self.transaction() as connection:
            session = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if session is None:
                raise BusinessError(E_STAGE, "会话不存在", field_path="session_id")
            row = connection.execute(
                """
                SELECT * FROM phase_states
                WHERE session_id = ? AND generation = ? AND stage_seq = ? AND stale = 0
                """,
                (session_id, generation, stage_seq),
            ).fetchone()
            if row is None:
                raise BusinessError(
                    E_STAGE,
                    f"阶段 {stage_seq} 在当前视图中不存在",
                    stage_seq=stage_seq,
                    field_path="phase_states.stage_seq",
                )
            from_status = row["status"]
            if not self._allowed_transition(from_status, to_status):
                raise BusinessError(
                    E_STAGE,
                    f"非法状态迁移 {from_status} -> {to_status}",
                    stage_seq=stage_seq,
                    field_path="phase_states.status",
                )
            seq = self._next_seq(connection, session_id)
            transition_details = dict(details or {})
            if skip_from_stage_seq is not None:
                transition_details["skip_from_stage_seq"] = skip_from_stage_seq
            connection.execute(
                """
                INSERT INTO ledger
                    (session_id, seq, generation, kind, stage_seq, from_status, to_status,
                     reason, code, logical_ms, plan_version, details_json, is_complete)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    session_id,
                    seq,
                    generation,
                    kind,
                    stage_seq,
                    from_status,
                    to_status,
                    reason,
                    code,
                    logical_ms,
                    int(session["plan_version"]),
                    canonical_json(transition_details),
                ),
            )
            skip_reason = row["skip_reason"]
            business_code = row["business_code"]
            skip_from = row["skip_from_stage_seq"]
            unknown_reason = row["unknown_reason"]
            if to_status == "skipped":
                skip_reason = reason
                business_code = code
                skip_from = skip_from_stage_seq
            elif to_status in {"running", "succeeded", "failed"}:
                skip_reason = None
                business_code = None
                skip_from = None
            if to_status == "running":
                unknown_reason = None
            connection.execute(
                """
                UPDATE phase_states
                SET status = ?, skip_reason = ?, business_code = ?,
                    skip_from_stage_seq = ?, unknown_reason = ?, updated_ms = ?, stale = 0
                WHERE session_id = ? AND generation = ? AND stage_seq = ?
                """,
                (
                    to_status,
                    skip_reason,
                    business_code,
                    skip_from,
                    unknown_reason,
                    logical_ms,
                    session_id,
                    generation,
                    stage_seq,
                ),
            )
            connection.execute(
                """
                UPDATE sessions
                SET clock_ms = ?, last_clock_source = 'logical'
                WHERE session_id = ?
                """,
                (logical_ms, session_id),
            )
        return LedgerEntry(
            seq=seq,
            kind=kind,
            stage_seq=stage_seq,
            from_status=from_status,
            to_status=to_status,
            reason=reason,
            code=code,
            logical_ms=logical_ms,
            plan_version=int(session["plan_version"]),
            details=transition_details,
        )

    def mark_unknown(
        self,
        session_id: str,
        generation: int,
        stage_seq: int,
        *,
        reason: str,
        logical_ms: int,
    ) -> None:
        with self.transaction() as connection:
            session = connection.execute(
                "SELECT plan_version FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if session is None:
                raise BusinessError(E_STAGE, "会话不存在", field_path="session_id")
            row = connection.execute(
                """
                SELECT status FROM phase_states
                WHERE session_id = ? AND generation = ? AND stage_seq = ? AND stale = 0
                """,
                (session_id, generation, stage_seq),
            ).fetchone()
            if row is None:
                raise BusinessError(E_STAGE, "阶段不存在", stage_seq=stage_seq)
            if row["status"] == "unknown":
                return
            if row["status"] != "running":
                raise BusinessError(
                    E_LEDGER,
                    "只有 running 阶段可标注为 unknown",
                    stage_seq=stage_seq,
                    field_path="phase_states.status",
                )
            seq = self._next_seq(connection, session_id)
            connection.execute(
                """
                INSERT INTO ledger
                    (session_id, seq, generation, kind, stage_seq, from_status, to_status,
                     reason, code, logical_ms, plan_version, details_json, is_complete)
                VALUES (?, ?, ?, 'recovery_unknown', ?, 'running', 'unknown',
                        ?, ?, ?, ?, '{}', 1)
                """,
                (
                    session_id,
                    seq,
                    generation,
                    stage_seq,
                    reason,
                    E_LEDGER,
                    logical_ms,
                    int(session["plan_version"]),
                ),
            )
            connection.execute(
                """
                UPDATE phase_states
                SET status = 'unknown', unknown_reason = ?, updated_ms = ?
                WHERE session_id = ? AND generation = ? AND stage_seq = ?
                """,
                (reason, logical_ms, session_id, generation, stage_seq),
            )

    def set_session_status(
        self,
        session_id: str,
        status: str,
        *,
        kind: str,
        reason: str,
        logical_ms: int | None = None,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        with self.transaction() as connection:
            session = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if session is None:
                raise BusinessError(E_STAGE, "会话不存在", field_path="session_id")
            clock = int(session["clock_ms"]) if logical_ms is None else logical_ms
            seq = self._next_seq(connection, session_id)
            connection.execute(
                """
                INSERT INTO ledger
                    (session_id, seq, generation, kind, stage_seq, from_status, to_status,
                     reason, code, logical_ms, plan_version, details_json, is_complete)
                VALUES (?, ?, ?, ?, NULL, NULL, NULL, ?, ?, ?, ?, ?, 1)
                """,
                (
                    session_id,
                    seq,
                    int(session["generation"]),
                    kind,
                    reason,
                    code,
                    clock,
                    int(session["plan_version"]),
                    canonical_json(details or {}),
                ),
            )
            connection.execute(
                """
                UPDATE sessions
                SET status = ?, clock_ms = ?, last_clock_source = 'logical'
                WHERE session_id = ?
                """,
                (status, clock, session_id),
            )

    def trigger_event(
        self,
        session_id: str,
        generation: int,
        injection_seq: int,
        *,
        status: str,
        reason: str,
        logical_ms: int,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> LedgerEntry:
        with self.transaction() as connection:
            session = connection.execute(
                "SELECT plan_version FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            row = connection.execute(
                """
                SELECT * FROM events
                WHERE session_id = ? AND generation = ? AND injection_seq = ? AND stale = 0
                """,
                (session_id, generation, injection_seq),
            ).fetchone()
            if row is None:
                raise BusinessError(E_LEDGER, "注入事件不存在", field_path="events.injection_seq")
            if row["status"] != "pending":
                raise BusinessError(
                    E_LEDGER,
                    f"注入事件状态不是 pending: {row['status']}",
                    stage_seq=int(row["stage_seq"]),
                    field_path="events.status",
                )
            event_details = dict(details or {})
            event_details.update(
                {
                    "event_id": row["event_id"],
                    "event_type": row["event_type"],
                    "scheduled_offset_ms": int(row["offset_ms"]),
                    "target_stage_seq": int(row["stage_seq"]),
                }
            )
            seq = self._next_seq(connection, session_id)
            connection.execute(
                """
                INSERT INTO ledger
                    (session_id, seq, generation, kind, stage_seq, from_status, to_status,
                     reason, code, logical_ms, plan_version, details_json, is_complete)
                VALUES (?, ?, ?, 'event_triggered', ?, NULL, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    session_id,
                    seq,
                    generation,
                    int(row["stage_seq"]),
                    status,
                    reason,
                    code,
                    logical_ms,
                    int(session["plan_version"]),
                    canonical_json(event_details),
                ),
            )
            connection.execute(
                """
                UPDATE events SET status = ?
                WHERE session_id = ? AND generation = ? AND injection_seq = ?
                """,
                (status, session_id, generation, injection_seq),
            )
            connection.execute(
                "UPDATE sessions SET clock_ms = ? WHERE session_id = ?",
                (logical_ms, session_id),
            )
        return LedgerEntry(
            seq=seq,
            kind="event_triggered",
            stage_seq=int(row["stage_seq"]),
            from_status=None,
            to_status=status,
            reason=reason,
            code=code,
            logical_ms=logical_ms,
            plan_version=int(session["plan_version"]),
            details=event_details,
        )

    def cancel_events_for_stage(
        self,
        session_id: str,
        generation: int,
        stage_seq: int,
        *,
        reason: str,
        logical_ms: int,
    ) -> list[LedgerEntry]:
        cancelled: list[LedgerEntry] = []
        with self.transaction() as connection:
            session = connection.execute(
                "SELECT plan_version FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            rows = connection.execute(
                """
                SELECT * FROM events
                WHERE session_id = ? AND generation = ? AND stage_seq = ?
                  AND stale = 0 AND status = 'pending'
                ORDER BY injection_seq
                """,
                (session_id, generation, stage_seq),
            ).fetchall()
            for row in rows:
                seq = self._next_seq(connection, session_id)
                details = {
                    "event_id": row["event_id"],
                    "event_type": row["event_type"],
                    "target_stage_seq": stage_seq,
                }
                connection.execute(
                    """
                    INSERT INTO ledger
                        (session_id, seq, generation, kind, stage_seq, from_status, to_status,
                         reason, code, logical_ms, plan_version, details_json, is_complete)
                    VALUES (?, ?, ?, 'event_cancelled', ?, 'pending', 'cancelled',
                            ?, ?, ?, ?, ?, 1)
                    """,
                    (
                        session_id,
                        seq,
                        generation,
                        stage_seq,
                        reason,
                        E_STAGE,
                        logical_ms,
                        int(session["plan_version"]),
                        canonical_json(details),
                    ),
                )
                connection.execute(
                    """
                    UPDATE events SET status = 'cancelled'
                    WHERE session_id = ? AND generation = ? AND injection_seq = ?
                    """,
                    (session_id, generation, int(row["injection_seq"])),
                )
                cancelled.append(
                    LedgerEntry(
                        seq=seq,
                        kind="event_cancelled",
                        stage_seq=stage_seq,
                        from_status="pending",
                        to_status="cancelled",
                        reason=reason,
                        code=E_STAGE,
                        logical_ms=logical_ms,
                        plan_version=int(session["plan_version"]),
                        details=details,
                    )
                )
        return cancelled

    def inject_event(
        self,
        session_id: str,
        event: EventDefinition,
        *,
        logical_ms: int,
        source: str = "manual",
    ) -> EventRuntime:
        with self.transaction() as connection:
            session = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if session is None:
                raise BusinessError(E_PLAN, "会话不存在", field_path="session_id")
            generation = int(session["generation"])
            duplicate = connection.execute(
                """
                SELECT injection_seq FROM events
                WHERE session_id = ? AND generation = ? AND stale = 0 AND event_id = ?
                """,
                (session_id, generation, event.event_id),
            ).fetchone()
            if duplicate:
                raise BusinessError(
                    E_PLAN,
                    f"event_id 已在当前注入序列中: {event.event_id}",
                    stage_seq=event.stage_seq,
                    field_path="events.event_id",
                )
            row = connection.execute(
                """
                SELECT COALESCE(MAX(injection_seq), 0) + 1 AS next_seq
                FROM events WHERE session_id = ? AND generation = ?
                """,
                (session_id, generation),
            ).fetchone()
            injection_seq = int(row["next_seq"])
            connection.execute(
                """
                INSERT INTO events
                    (session_id, generation, injection_seq, event_id, event_type,
                     offset_ms, stage_seq, status, source, stale)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, 0)
                """,
                (
                    session_id,
                    generation,
                    injection_seq,
                    event.event_id,
                    event.type,
                    event.offset_ms,
                    event.stage_seq,
                    source,
                ),
            )
            seq = self._next_seq(connection, session_id)
            connection.execute(
                """
                INSERT INTO ledger
                    (session_id, seq, generation, kind, stage_seq, from_status, to_status,
                     reason, code, logical_ms, plan_version, details_json, is_complete)
                VALUES (?, ?, ?, 'event_injected', ?, 'absent', 'pending', ?, NULL, ?, ?, ?, 1)
                """,
                (
                    session_id,
                    seq,
                    generation,
                    event.stage_seq,
                    "注入演练事件",
                    logical_ms,
                    int(session["plan_version"]),
                    canonical_json(event.to_dict() | {"injection_seq": injection_seq, "source": source}),
                ),
            )
            connection.execute(
                "UPDATE sessions SET clock_ms = ? WHERE session_id = ?",
                (logical_ms, session_id),
            )
        return EventRuntime(
            injection_seq=injection_seq,
            event_id=event.event_id,
            type=event.type,
            offset_ms=event.offset_ms,
            stage_seq=event.stage_seq,
            status="pending",
            source=source,
        )

    def replace_plan_version(
        self,
        session_id: str,
        plan: Plan,
        *,
        diff: dict[str, Any],
    ) -> int:
        with self.transaction() as connection:
            session = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if session is None:
                raise BusinessError(E_PLAN, "会话不存在", field_path="session_id")
            old_generation = int(session["generation"])
            new_generation = old_generation + 1
            if plan.version <= int(
                connection.execute(
                    "SELECT COALESCE(MAX(version), 0) AS max_version FROM plan_versions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()["max_version"]
            ):
                raise BusinessError(
                    E_PLAN,
                    "预案版本必须单调递增",
                    field_path="version",
                )
            connection.execute(
                "UPDATE phase_states SET stale = 1 WHERE session_id = ? AND generation = ?",
                (session_id, old_generation),
            )
            connection.execute(
                "UPDATE events SET stale = 1 WHERE session_id = ? AND generation = ?",
                (session_id, old_generation),
            )
            connection.execute(
                "UPDATE plan_versions SET active = 0, stale = 1 WHERE session_id = ?",
                (session_id,),
            )
            connection.execute(
                """
                INSERT INTO plan_versions
                    (session_id, version, plan_json, plan_digest, active, stale, created_ms)
                VALUES (?, ?, ?, ?, 1, 0, 0)
                """,
                (session_id, plan.version, canonical_json(plan.to_dict()), plan.digest),
            )
            for stage in plan.stages:
                connection.execute(
                    """
                    INSERT INTO phase_states
                        (session_id, generation, stage_seq, status, updated_ms, stale)
                    VALUES (?, ?, ?, 'pending', 0, 0)
                    """,
                    (session_id, new_generation, stage.seq),
                )
            for injection_seq, event in enumerate(plan.events, start=1):
                connection.execute(
                    """
                    INSERT INTO events
                        (session_id, generation, injection_seq, event_id, event_type,
                         offset_ms, stage_seq, status, source, stale)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', 'plan', 0)
                    """,
                    (
                        session_id,
                        new_generation,
                        injection_seq,
                        event.event_id,
                        event.type,
                        event.offset_ms,
                        event.stage_seq,
                    ),
                )
            connection.execute(
                """
                UPDATE sessions
                SET plan_id = ?, plan_version = ?, plan_digest = ?, clock_ms = 0,
                    status = 'created', generation = ?, last_clock_source = 'logical'
                WHERE session_id = ?
                """,
                (
                    plan.plan_id,
                    plan.version,
                    plan.digest,
                    new_generation,
                    session_id,
                ),
            )
            seq = self._next_seq(connection, session_id)
            connection.execute(
                """
                INSERT INTO ledger
                    (session_id, seq, generation, kind, stage_seq, from_status, to_status,
                     reason, code, logical_ms, plan_version, details_json, is_complete)
                VALUES (?, ?, ?, 'plan_replaced', NULL, NULL, NULL, ?, NULL, 0, ?, ?, 1)
                """,
                (
                    session_id,
                    seq,
                    new_generation,
                    "整体替换预案、事件与当前账本视图",
                    plan.version,
                    canonical_json(diff),
                ),
            )
        return new_generation

    def ledger_rows(
        self,
        session_id: str,
        generation: int,
    ) -> list[LedgerEntry]:
        with self.read_connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM ledger
                WHERE session_id = ? AND generation = ?
                ORDER BY seq
                """,
                (session_id, generation),
            ).fetchall()
        entries: list[LedgerEntry] = []
        for row in rows:
            try:
                details = json.loads(row["details_json"])
                parse_ok = isinstance(details, dict)
            except (json.JSONDecodeError, TypeError):
                details = {"_parse_error": True}
                parse_ok = False
            entry = LedgerEntry(
                seq=int(row["seq"]),
                kind=row["kind"],
                stage_seq=int(row["stage_seq"]) if row["stage_seq"] is not None else None,
                from_status=row["from_status"],
                to_status=row["to_status"],
                reason=row["reason"],
                code=row["code"],
                logical_ms=int(row["logical_ms"]),
                plan_version=int(row["plan_version"]),
                details=details,
                is_complete=bool(row["is_complete"]) and parse_ok,
            )
            entries.append(entry)
        return entries

    def append_incomplete_tail(
        self,
        session_id: str,
        generation: int,
        *,
        stage_seq: int,
        logical_ms: int,
    ) -> int:
        with self.transaction() as connection:
            session = connection.execute(
                "SELECT plan_version FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            seq = self._next_seq(connection, session_id)
            connection.execute(
                """
                INSERT INTO ledger
                    (session_id, seq, generation, kind, stage_seq, from_status, to_status,
                     reason, code, logical_ms, plan_version, details_json, is_complete)
                VALUES (?, ?, ?, 'crash_tail', ?, NULL, NULL, ?, ?, ?, ?, '{}', 0)
                """,
                (
                    session_id,
                    seq,
                    generation,
                    stage_seq,
                    "进程强制终止时未完整提交的账本尾记录",
                    E_LEDGER,
                    logical_ms,
                    int(session["plan_version"]),
                ),
            )
        return seq

    def discard_incomplete_tail(
        self,
        session_id: str,
        generation: int,
        *,
        tail_seqs: list[int],
        logical_ms: int,
        reason: str,
    ) -> None:
        """Quarantine uncommitted tail rows so later recovery rows stay contiguous."""

        if not tail_seqs:
            return
        with self.transaction() as connection:
            session = connection.execute(
                "SELECT plan_version FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            placeholders = ",".join("?" for _ in tail_seqs)
            connection.execute(
                f"""
                DELETE FROM ledger
                WHERE session_id = ? AND generation = ? AND seq IN ({placeholders})
                """,
                (session_id, generation, *tail_seqs),
            )
            seq = self._next_seq(connection, session_id)
            connection.execute(
                """
                INSERT INTO ledger
                    (session_id, seq, generation, kind, stage_seq, from_status, to_status,
                     reason, code, logical_ms, plan_version, details_json, is_complete)
                VALUES (?, ?, ?, 'incomplete_tail_discarded', NULL, NULL, NULL,
                        ?, ?, ?, ?, ?, 1)
                """,
                (
                    session_id,
                    seq,
                    generation,
                    reason,
                    E_LEDGER,
                    logical_ms,
                    int(session["plan_version"]),
                    canonical_json({"discarded_seq": tail_seqs}),
                ),
            )

    def next_snapshot_number(self) -> int:
        with self.read_connection() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(snapshot_pk), 0) + 1 AS next_number FROM snapshots"
            ).fetchone()
            return int(row["next_number"])

    def register_snapshot(
        self,
        *,
        snapshot_id: str,
        session_id: str,
        file_path: str,
        plan_version: int,
        content_digest: str,
        created_ms: int,
        generation: int,
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                "UPDATE snapshots SET read_only = 1 WHERE session_id = ?",
                (session_id,),
            )
            connection.execute(
                """
                INSERT INTO snapshots
                    (snapshot_id, session_id, file_path, plan_version, content_digest,
                     created_ms, generation, read_only)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    snapshot_id,
                    session_id,
                    file_path,
                    plan_version,
                    content_digest,
                    created_ms,
                    generation,
                ),
            )

    def list_snapshots(self, session_id: str | None = None) -> list[dict[str, Any]]:
        with self.read_connection() as connection:
            if session_id is None:
                rows = connection.execute(
                    "SELECT * FROM snapshots ORDER BY snapshot_pk"
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM snapshots WHERE session_id = ? ORDER BY snapshot_pk",
                    (session_id,),
                ).fetchall()
        return [
            {
                "snapshot_id": row["snapshot_id"],
                "session_id": row["session_id"],
                "file_path": row["file_path"],
                "plan_version": int(row["plan_version"]),
                "content_digest": row["content_digest"],
                "created_ms": int(row["created_ms"]),
                "generation": int(row["generation"]),
                "read_only": bool(row["read_only"]),
            }
            for row in rows
        ]

    def mark_snapshots_read_only(self, session_id: str) -> None:
        with self.transaction() as connection:
            connection.execute(
                "UPDATE snapshots SET read_only = 1 WHERE session_id = ?",
                (session_id,),
            )

    def replace_from_snapshot(
        self,
        session_id: str,
        snapshot: dict[str, Any],
        *,
        previous_plan_version: int,
        previous_plan_digest: str,
    ) -> int:
        plan = validate_plan_object(snapshot["plan"])
        with self.transaction() as connection:
            session = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if session is None:
                raise BusinessError(E_PLAN, "会话不存在", field_path="session_id")
            generation = int(session["generation"]) + 1
            existing_plan = connection.execute(
                "SELECT plan_digest FROM plan_versions WHERE session_id = ? AND version = ?",
                (session_id, plan.version),
            ).fetchone()
            if existing_plan and existing_plan["plan_digest"] != plan.digest:
                raise BusinessError(
                    "E_SNAPSHOT",
                    "快照预案版本与已有版本内容冲突",
                    field_path="plan_digest",
                )
            connection.execute(
                "UPDATE phase_states SET stale = 1 WHERE session_id = ? AND generation = ?",
                (session_id, int(session["generation"])),
            )
            connection.execute(
                "UPDATE events SET stale = 1 WHERE session_id = ? AND generation = ?",
                (session_id, int(session["generation"])),
            )
            connection.execute(
                "UPDATE plan_versions SET active = 0, stale = 1 WHERE session_id = ?",
                (session_id,),
            )
            if existing_plan:
                connection.execute(
                    """
                    UPDATE plan_versions SET active = 1, stale = 0
                    WHERE session_id = ? AND version = ?
                    """,
                    (session_id, plan.version),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO plan_versions
                        (session_id, version, plan_json, plan_digest, active, stale, created_ms)
                    VALUES (?, ?, ?, ?, 1, 0, ?)
                    """,
                    (
                        session_id,
                        plan.version,
                        canonical_json(plan.to_dict()),
                        plan.digest,
                        int(snapshot["logical_clock_ms"]),
                    ),
                )
            for stage in snapshot["phase_states"]:
                connection.execute(
                    """
                    INSERT INTO phase_states
                        (session_id, generation, stage_seq, status, skip_reason,
                         business_code, skip_from_stage_seq, unknown_reason, updated_ms, stale)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                    """,
                    (
                        session_id,
                        generation,
                        int(stage["seq"]),
                        stage["status"],
                        stage.get("skip_reason"),
                        stage.get("business_code"),
                        stage.get("skip_from_stage_seq"),
                        stage.get("unknown_reason"),
                        int(stage.get("updated_ms", 0)),
                    ),
                )
            for event in snapshot["events"]:
                connection.execute(
                    """
                    INSERT INTO events
                        (session_id, generation, injection_seq, event_id, event_type,
                         offset_ms, stage_seq, status, source, stale)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                    """,
                    (
                        session_id,
                        generation,
                        int(event["injection_seq"]),
                        event["event_id"],
                        event["type"],
                        int(event["offset_ms"]),
                        int(event["stage_seq"]),
                        event.get("status", "pending"),
                        event.get("source", "snapshot"),
                    ),
                )
            connection.execute(
                """
                UPDATE sessions
                SET plan_id = ?, plan_version = ?, plan_digest = ?, clock_ms = ?,
                    status = 'restored', generation = ?, last_clock_source = 'logical'
                WHERE session_id = ?
                """,
                (
                    plan.plan_id,
                    plan.version,
                    plan.digest,
                    int(snapshot["logical_clock_ms"]),
                    generation,
                    session_id,
                ),
            )
            seq = self._next_seq(connection, session_id)
            connection.execute(
                """
                INSERT INTO ledger
                    (session_id, seq, generation, kind, stage_seq, from_status, to_status,
                     reason, code, logical_ms, plan_version, details_json, is_complete)
                VALUES (?, ?, ?, 'snapshot_restored', NULL, NULL, NULL, ?, NULL, ?, ?, ?, 1)
                """,
                (
                    session_id,
                    seq,
                    generation,
                    "从快照整体替换预案、事件与阶段状态",
                    int(snapshot["logical_clock_ms"]),
                    plan.version,
                    canonical_json(
                        {
                            "snapshot_id": snapshot["snapshot_id"],
                            "snapshot_plan_version": plan.version,
                            "previous_plan_version": previous_plan_version,
                            "previous_plan_digest": previous_plan_digest,
                            "content_digest": snapshot["content_digest"],
                        }
                    ),
                ),
            )
            for entry in snapshot.get("ledger_trace", []):
                if not entry.get("is_complete", True):
                    continue
                seq = self._next_seq(connection, session_id)
                details = dict(entry.get("details") or {})
                details["imported_from_snapshot"] = snapshot["snapshot_id"]
                details["original_seq"] = int(entry["seq"])
                connection.execute(
                    """
                    INSERT INTO ledger
                        (session_id, seq, generation, kind, stage_seq, from_status, to_status,
                         reason, code, logical_ms, plan_version, details_json, is_complete)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                    """,
                    (
                        session_id,
                        seq,
                        generation,
                        entry["kind"],
                        entry.get("stage_seq"),
                        entry.get("from_status"),
                        entry.get("to_status"),
                        entry.get("reason"),
                        entry.get("code"),
                        int(entry["logical_ms"]),
                        int(entry.get("plan_version", plan.version)),
                        canonical_json(details),
                    ),
                )
        return generation
