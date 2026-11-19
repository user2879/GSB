"""SQLite task ledger with atomic commit points and crash recovery."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .audit import utc_now


class Ledger:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS event_log (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    code TEXT NOT NULL,
                    shard_id TEXT,
                    version INTEGER,
                    timestamp TEXT NOT NULL,
                    event_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS task_claims (
                    ledger_task_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    base_task_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    logical_id TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    claimed_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_claims_run_status
                    ON task_claims(run_id, status);
                CREATE TABLE IF NOT EXISTS task_commits (
                    ledger_task_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    base_task_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    logical_id TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    outcome TEXT NOT NULL,
                    bytes_completed INTEGER NOT NULL,
                    budget_consumed INTEGER NOT NULL,
                    progress_cursor INTEGER NOT NULL,
                    committed_at TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    FOREIGN KEY(ledger_task_id) REFERENCES task_claims(ledger_task_id)
                );
                CREATE INDEX IF NOT EXISTS idx_commits_run
                    ON task_commits(run_id, base_task_id);
                CREATE TABLE IF NOT EXISTS run_progress (
                    run_id TEXT PRIMARY KEY,
                    completed_bytes INTEGER NOT NULL,
                    cursor INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def record_event(self, event: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO event_log(code, shard_id, version, timestamp, event_json) VALUES(?,?,?,?,?)",
                (
                    event["code"],
                    event.get("shard_id"),
                    event.get("version"),
                    event["timestamp"],
                    json.dumps(event, ensure_ascii=False, sort_keys=True),
                ),
            )

    def claim(
        self,
        *,
        ledger_task_id: str,
        run_id: str,
        base_task_id: str,
        node_id: str,
        logical_id: str,
        attempt: int,
    ) -> None:
        now = utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO task_claims(
                    ledger_task_id, run_id, base_task_id, node_id, logical_id,
                    attempt, status, claimed_at, updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(ledger_task_id) DO UPDATE SET
                    status='running', updated_at=excluded.updated_at
                """,
                (
                    ledger_task_id,
                    run_id,
                    base_task_id,
                    node_id,
                    logical_id,
                    attempt,
                    "running",
                    now,
                    now,
                ),
            )

    def commit(
        self,
        *,
        ledger_task_id: str,
        run_id: str,
        base_task_id: str,
        node_id: str,
        logical_id: str,
        attempt: int,
        outcome: str,
        bytes_completed: int,
        budget_consumed: int,
        result: dict[str, Any],
    ) -> int:
        """Commit the task result, claim state and progress in one transaction."""

        now = utc_now()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT completed_bytes, cursor FROM run_progress WHERE run_id=?",
                (run_id,),
            ).fetchone()
            completed_bytes = int(row["completed_bytes"]) if row else 0
            cursor = int(row["cursor"]) if row else 0
            cursor += 1
            completed_bytes += max(0, int(bytes_completed))
            connection.execute(
                """
                INSERT INTO task_commits(
                    ledger_task_id, run_id, base_task_id, node_id, logical_id,
                    attempt, outcome, bytes_completed, budget_consumed,
                    progress_cursor, committed_at, result_json
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(ledger_task_id) DO UPDATE SET
                    outcome=excluded.outcome,
                    bytes_completed=excluded.bytes_completed,
                    budget_consumed=excluded.budget_consumed,
                    progress_cursor=excluded.progress_cursor,
                    committed_at=excluded.committed_at,
                    result_json=excluded.result_json
                """,
                (
                    ledger_task_id,
                    run_id,
                    base_task_id,
                    node_id,
                    logical_id,
                    attempt,
                    outcome,
                    int(bytes_completed),
                    int(budget_consumed),
                    cursor,
                    now,
                    json.dumps(result, ensure_ascii=False, sort_keys=True),
                ),
            )
            connection.execute(
                "UPDATE task_claims SET status=?, updated_at=? WHERE ledger_task_id=?",
                ("committed", now, ledger_task_id),
            )
            connection.execute(
                """
                INSERT INTO run_progress(run_id, completed_bytes, cursor, updated_at)
                VALUES(?,?,?,?)
                ON CONFLICT(run_id) DO UPDATE SET
                    completed_bytes=excluded.completed_bytes,
                    cursor=excluded.cursor,
                    updated_at=excluded.updated_at
                """,
                (run_id, completed_bytes, cursor, now),
            )
            connection.commit()
            return cursor

    def recover(self, run_id: str, logger: Any | None = None) -> dict[str, Any]:
        """Mark uncommitted running claims unknown; never invent an exit report."""

        now = utc_now()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT c.* FROM task_claims c
                LEFT JOIN task_commits k ON k.ledger_task_id=c.ledger_task_id
                WHERE c.run_id=? AND c.status='running' AND k.ledger_task_id IS NULL
                ORDER BY c.claimed_at, c.ledger_task_id
                """,
                (run_id,),
            ).fetchall()
            unknown = [dict(row) for row in rows]
            connection.executemany(
                "UPDATE task_claims SET status='unknown_requires_rerun', updated_at=? WHERE ledger_task_id=?",
                [(now, row["ledger_task_id"]) for row in rows],
            )
            commits = [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM task_commits WHERE run_id=? ORDER BY progress_cursor",
                    (run_id,),
                ).fetchall()
            ]
        report = {
            "run_id": run_id,
            "committed": commits,
            "unknown_requires_rerun": unknown,
            "exit_report_available": False,
            "note": "A forced termination produces no trustworthy exit report; only committed task rows are resumed.",
        }
        if logger is not None:
            for task in unknown:
                logger.emit(
                    "E_UNKNOWN_REQUIRES_RERUN",
                    shard_id=task.get("logical_id"),
                    version=1,
                    task_id=task["ledger_task_id"],
                    node_id=task["node_id"],
                )
        return report

    def completed_task_ids(self, run_id: str) -> set[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT base_task_id FROM task_commits WHERE run_id=? AND outcome='success'",
                (run_id,),
            ).fetchall()
        return {str(row["base_task_id"]) for row in rows}

    def completed_success_rows(self, run_id: str) -> dict[str, dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM task_commits WHERE run_id=? AND outcome='success' ORDER BY progress_cursor",
                (run_id,),
            ).fetchall()
        return {str(row["base_task_id"]): dict(row) for row in rows}

    def committed_results(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM task_commits WHERE run_id=? ORDER BY progress_cursor",
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]
