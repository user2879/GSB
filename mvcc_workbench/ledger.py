"""SQLite round ledger with atomic commit and crash recovery semantics."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .canonical import canonical_json, sha256_digest
from .errors import LedgerError

LEDGER_FORMAT_VERSION = 1
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LedgerRound:
    round_id: str
    spec_digest: str
    selected_path: str
    conclusion_digest: str
    results: tuple[dict[str, Any], ...]
    plan: dict[str, Any]
    schedule: dict[str, Any]
    budget: dict[str, Any]
    progress: dict[str, Any]
    committed_wall_ms: int
    format_version: int = LEDGER_FORMAT_VERSION

    def payload_without_digest(self) -> dict[str, Any]:
        return {
            "round_id": self.round_id,
            "format_version": self.format_version,
            "spec_digest": self.spec_digest,
            "selected_path": self.selected_path,
            "conclusion_digest": self.conclusion_digest,
            "results": [dict(item) for item in sorted(self.results, key=lambda x: x["query_id"])],
            "plan": self.plan,
            "schedule": self.schedule,
            "budget": self.budget,
            "progress": self.progress,
            "committed_wall_ms": self.committed_wall_ms,
        }

    @property
    def content_digest(self) -> str:
        return sha256_digest(self.payload_without_digest())

    def summary(self) -> dict[str, Any]:
        return {
            "round_id": self.round_id,
            "format_version": self.format_version,
            "spec_digest": self.spec_digest,
            "selected_path": self.selected_path,
            "conclusion_digest": self.conclusion_digest,
            "content_digest": self.content_digest,
            "conclusion_count": len(self.results),
            "committed_wall_ms": self.committed_wall_ms,
        }


@dataclass(frozen=True)
class RecoveryReport:
    latest_complete: LedgerRound | None
    unknown_rounds: tuple[dict[str, Any], ...]
    ledger_cursor: int
    requires_rerun: bool
    code: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "ledger_cursor": self.ledger_cursor,
            "requires_rerun": self.requires_rerun,
            "latest_complete": (
                self.latest_complete.summary() if self.latest_complete is not None else None
            ),
            "latest_complete_round": (
                self.latest_complete.payload_without_digest()
                if self.latest_complete is not None
                else None
            ),
            "unknown_rounds": list(self.unknown_rounds),
        }


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.path,
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._create_schema()

    def _create_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS ledger_info (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS pending_rounds (
                    round_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    started_wall_ms INTEGER NOT NULL,
                    spec_digest TEXT NOT NULL,
                    query_ids_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS rounds (
                    round_seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    round_id TEXT NOT NULL UNIQUE,
                    format_version INTEGER NOT NULL,
                    spec_digest TEXT NOT NULL,
                    selected_path TEXT NOT NULL,
                    conclusion_digest TEXT NOT NULL,
                    content_digest TEXT NOT NULL,
                    committed_wall_ms INTEGER NOT NULL,
                    plan_json TEXT NOT NULL,
                    schedule_json TEXT NOT NULL,
                    budget_json TEXT NOT NULL,
                    progress_json TEXT NOT NULL,
                    progress_cursor INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS conclusions (
                    round_id TEXT NOT NULL,
                    query_id TEXT NOT NULL,
                    row_key TEXT NOT NULL,
                    read_point_id TEXT NOT NULL,
                    visible_version INTEGER,
                    visible_tx_id TEXT,
                    visible_value_digest TEXT,
                    backtrack_steps INTEGER NOT NULL,
                    inspected_versions INTEGER NOT NULL,
                    predicate_result INTEGER NOT NULL,
                    terminal_code TEXT NOT NULL,
                    exclusion_reasons_json TEXT NOT NULL,
                    reason_chain_digest TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    PRIMARY KEY (round_id, query_id),
                    FOREIGN KEY (round_id) REFERENCES rounds(round_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_rounds_round_seq ON rounds(round_seq);
                CREATE INDEX IF NOT EXISTS idx_conclusions_round ON conclusions(round_id);
                """
            )
            round_columns = {
                row["name"] for row in self._conn.execute("PRAGMA table_info(rounds)")
            }
            if "progress_cursor" not in round_columns:
                self._conn.execute(
                    "ALTER TABLE rounds ADD COLUMN progress_cursor INTEGER NOT NULL DEFAULT 0"
                )
            existing = self._conn.execute(
                "SELECT value FROM ledger_info WHERE key='format_version'"
            ).fetchone()
            if existing is None:
                self._conn.execute(
                    "INSERT INTO ledger_info(key, value) VALUES('format_version', ?)",
                    (str(LEDGER_FORMAT_VERSION),),
                )
                self._conn.execute(
                    "INSERT INTO ledger_info(key, value) VALUES('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
            else:
                try:
                    stored_format = int(existing["value"])
                except (TypeError, ValueError) as exc:
                    raise LedgerError("账本格式版本不是整数") from exc
                if stored_format != LEDGER_FORMAT_VERSION:
                    raise LedgerError(
                        "账本格式版本不兼容",
                        details={
                            "expected": LEDGER_FORMAT_VERSION,
                            "actual": stored_format,
                        },
                    )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def begin_pending(
        self,
        *,
        round_id: str,
        spec_digest: str,
        query_ids: Iterable[str],
        started_wall_ms: int | None = None,
    ) -> None:
        timestamp = started_wall_ms if started_wall_ms is not None else int(time.time() * 1000)
        ordered_ids = sorted(query_ids)
        with self._lock:
            try:
                self._conn.execute(
                    """
                    INSERT INTO pending_rounds(
                        round_id, status, started_wall_ms, spec_digest, query_ids_json
                    ) VALUES(?, 'building', ?, ?, ?)
                    """,
                    (round_id, timestamp, spec_digest, canonical_json(ordered_ids)),
                )
            except sqlite3.IntegrityError as exc:
                raise LedgerError(
                    "轮次标识已存在，不能覆盖",
                    details={"round_id": round_id},
                ) from exc

    def commit_round(self, round_record: LedgerRound) -> str:
        if round_record.format_version != LEDGER_FORMAT_VERSION:
            raise LedgerError(
                "非法的轮次格式版本",
                details={"format_version": round_record.format_version},
            )
        ordered_results = sorted(round_record.results, key=lambda x: x["query_id"])
        computed_digest = round_record.content_digest
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                pending = self._conn.execute(
                    "SELECT status FROM pending_rounds WHERE round_id=?",
                    (round_record.round_id,),
                ).fetchone()
                if pending is None:
                    raise LedgerError(
                        "轮次未登记 pending 记录",
                        details={"round_id": round_record.round_id},
                    )
                if pending["status"] != "building":
                    raise LedgerError(
                        "轮次 pending 状态不允许提交",
                        details={
                            "round_id": round_record.round_id,
                            "status": pending["status"],
                        },
                    )
                self._conn.execute(
                    """
                    INSERT INTO rounds(
                        round_id, format_version, spec_digest, selected_path,
                        conclusion_digest, content_digest, committed_wall_ms,
                        plan_json, schedule_json, budget_json, progress_json,
                        progress_cursor
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        round_record.round_id,
                        round_record.format_version,
                        round_record.spec_digest,
                        round_record.selected_path,
                        round_record.conclusion_digest,
                        computed_digest,
                        round_record.committed_wall_ms,
                        canonical_json(round_record.plan),
                        canonical_json(round_record.schedule),
                        canonical_json(round_record.budget),
                        canonical_json(round_record.progress),
                        int(round_record.progress["cursor"]),
                    ),
                )
                for result in ordered_results:
                    self._conn.execute(
                        """
                        INSERT INTO conclusions(
                            round_id, query_id, row_key, read_point_id,
                            visible_version, visible_tx_id, visible_value_digest,
                            backtrack_steps, inspected_versions, predicate_result,
                            terminal_code, exclusion_reasons_json,
                            reason_chain_digest, result_json
                        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            round_record.round_id,
                            result["query_id"],
                            result["row_key"],
                            result["read_point_id"],
                            result["visible_version"],
                            result["visible_tx_id"],
                            result["visible_value_digest"],
                            result["backtrack_steps"],
                            result["inspected_versions"],
                            1 if result["predicate_result"] else 0,
                            result["terminal_code"],
                            canonical_json(result["exclusion_reasons"]),
                            result["reason_chain_digest"],
                            canonical_json(result),
                        ),
                    )
                self._conn.execute(
                    "UPDATE pending_rounds SET status='committed' WHERE round_id=?",
                    (round_record.round_id,),
                )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
        return computed_digest

    def ledger_cursor(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COALESCE(MAX(round_seq), 0) AS seq FROM rounds").fetchone()
            return int(row["seq"])

    def _round_from_row(
        self,
        round_row: sqlite3.Row,
        conclusion_rows: list[sqlite3.Row],
    ) -> LedgerRound:
        try:
            stored_format = int(round_row["format_version"])
            plan = json.loads(round_row["plan_json"])
            schedule = json.loads(round_row["schedule_json"])
            budget = json.loads(round_row["budget_json"])
            progress = json.loads(round_row["progress_json"])
            results = tuple(json.loads(row["result_json"]) for row in conclusion_rows)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise LedgerError(
                "账本轮次 JSON 无法解析",
                details={"round_id": round_row["round_id"]},
            ) from exc
        record = LedgerRound(
            round_id=round_row["round_id"],
            format_version=stored_format,
            spec_digest=round_row["spec_digest"],
            selected_path=round_row["selected_path"],
            conclusion_digest=round_row["conclusion_digest"],
            results=tuple(sorted(results, key=lambda x: x["query_id"])),
            plan=plan,
            schedule=schedule,
            budget=budget,
            progress=progress,
            committed_wall_ms=int(round_row["committed_wall_ms"]),
        )
        if stored_format != LEDGER_FORMAT_VERSION:
            raise LedgerError(
                "账本轮次格式版本不兼容",
                details={
                    "round_id": record.round_id,
                    "expected": LEDGER_FORMAT_VERSION,
                    "actual": stored_format,
                },
            )
        if record.content_digest != round_row["content_digest"]:
            raise LedgerError(
                "账本内容摘要不一致",
                details={
                    "round_id": record.round_id,
                    "expected": round_row["content_digest"],
                    "actual": record.content_digest,
                },
            )
        if len(record.results) != len(conclusion_rows):
            raise LedgerError(
                "账本结论数量与轮次记录不一致",
                details={"round_id": record.round_id},
            )
        if int(round_row["progress_cursor"]) != int(record.progress.get("cursor", -1)):
            raise LedgerError(
                "账本进度游标与进度 JSON 不一致",
                details={
                    "round_id": record.round_id,
                    "progress_cursor": int(round_row["progress_cursor"]),
                    "progress_json_cursor": record.progress.get("cursor"),
                },
            )
        return record

    def load_latest_committed(self) -> LedgerRound | None:
        with self._lock:
            round_row = self._conn.execute(
                "SELECT * FROM rounds ORDER BY round_seq DESC LIMIT 1"
            ).fetchone()
            if round_row is None:
                return None
            conclusion_rows = self._conn.execute(
                "SELECT * FROM conclusions WHERE round_id=? ORDER BY query_id",
                (round_row["round_id"],),
            ).fetchall()
            return self._round_from_row(round_row, list(conclusion_rows))

    def load_round(self, round_id: str) -> LedgerRound | None:
        with self._lock:
            round_row = self._conn.execute(
                "SELECT * FROM rounds WHERE round_id=?", (round_id,)
            ).fetchone()
            if round_row is None:
                return None
            conclusion_rows = self._conn.execute(
                "SELECT * FROM conclusions WHERE round_id=? ORDER BY query_id",
                (round_id,),
            ).fetchall()
            return self._round_from_row(round_row, list(conclusion_rows))

    def pending_rounds(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT round_id, status, started_wall_ms, spec_digest, query_ids_json
                FROM pending_rounds WHERE status='building' ORDER BY round_id
                """
            ).fetchall()
        payload: list[dict[str, Any]] = []
        for row in rows:
            query_ids = json.loads(row["query_ids_json"])
            payload.append(
                {
                    "round_id": row["round_id"],
                    "status": "UNKNOWN_RERUN_REQUIRED",
                    "started_wall_ms": int(row["started_wall_ms"]),
                    "spec_digest": row["spec_digest"],
                    "query_ids": sorted(query_ids),
                    "business_code": "UNKNOWN_RERUN_REQUIRED",
                }
            )
        return tuple(payload)

    def recover(self) -> RecoveryReport:
        latest = self.load_latest_committed()
        unknown = self.pending_rounds()
        cursor = self.ledger_cursor()
        return RecoveryReport(
            latest_complete=latest,
            unknown_rounds=unknown,
            ledger_cursor=cursor,
            requires_rerun=bool(unknown),
            code="OK_RERUN_REQUIRED" if unknown else "OK_RECOVERED",
        )

    def round_count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS count FROM rounds").fetchone()
            return int(row["count"])
