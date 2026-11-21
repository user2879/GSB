from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from .budget import BudgetPlan
from .canonical import canonical_json, digest_json, utc_now_iso
from .errors import DiagnosticError
from .model import Dataset
from .scheduler import TaskTrace
from .visibility import VisibilityDecision, decisions_digest


SCHEMA_VERSION = 1


DDL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS rounds (
    ledger_cursor INTEGER PRIMARY KEY,
    generation INTEGER NOT NULL,
    round_no INTEGER NOT NULL,
    schema_version INTEGER NOT NULL,
    allocation_json TEXT NOT NULL,
    schedule_json TEXT NOT NULL,
    decisions_json TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    committed_at TEXT NOT NULL,
    UNIQUE(generation, round_no)
);

CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_no INTEGER PRIMARY KEY,
    ledger_cursor INTEGER NOT NULL,
    generation INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS inflight (
    inflight_id INTEGER PRIMARY KEY AUTOINCREMENT,
    generation INTEGER NOT NULL,
    round_no INTEGER NOT NULL,
    query_id TEXT NOT NULL,
    read_point_id TEXT NOT NULL,
    row_key TEXT NOT NULL,
    owner_pid INTEGER NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS unknown_tasks (
    unknown_id INTEGER PRIMARY KEY AUTOINCREMENT,
    generation INTEGER NOT NULL,
    round_no INTEGER NOT NULL,
    query_id TEXT NOT NULL,
    read_point_id TEXT NOT NULL,
    row_key TEXT NOT NULL,
    status TEXT NOT NULL,
    reason TEXT NOT NULL,
    marked_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reconciliations (
    reconciliation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    generation INTEGER NOT NULL,
    old_snapshot_digest TEXT,
    new_dataset_digest TEXT NOT NULL,
    location_json TEXT NOT NULL,
    affected_json TEXT NOT NULL,
    changes_json TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_rounds_generation ON rounds(generation, round_no);
CREATE INDEX IF NOT EXISTS idx_snapshots_generation ON snapshots(generation, ledger_cursor);
CREATE INDEX IF NOT EXISTS idx_inflight_status ON inflight(status);
"""


class Ledger:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = FULL")
        self._initialize_or_verify()
        self.verify_integrity()
        self.recovered_inflight = self.mark_running_unknown()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _initialize_or_verify(self) -> None:
        version = int(self.connection.execute("PRAGMA user_version").fetchone()[0])
        if version == 0:
            self.connection.executescript(DDL)
            self.connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self._set_meta("format_version", str(SCHEMA_VERSION))
            self._set_meta("created_at", utc_now_iso())
            self._set_meta("ledger_seq", "0")
            return
        if version != SCHEMA_VERSION:
            raise DiagnosticError(
                "E_SNAPSHOT",
                {"read_point_id": "<ledger>", "row_key": "<ledger>", "field": "schema_version"},
                "unsupported ledger schema version",
                expected=SCHEMA_VERSION,
                actual=version,
            )
        stored = self._get_meta("format_version")
        if stored != str(SCHEMA_VERSION):
            raise DiagnosticError(
                "E_SNAPSHOT",
                {"read_point_id": "<ledger>", "row_key": "<ledger>", "field": "format_version"},
                "ledger integer format version is missing or unsupported",
                expected=str(SCHEMA_VERSION),
                actual=stored,
            )

    def _set_meta(self, key: str, value: str) -> None:
        self.connection.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def _get_meta(self, key: str) -> str | None:
        row = self.connection.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def _next_ledger_cursor(self) -> int:
        return int(self._get_meta("ledger_seq") or "0") + 1

    def verify_integrity(self) -> None:
        try:
            check = self.connection.execute("PRAGMA integrity_check").fetchone()[0]
        except sqlite3.DatabaseError as exc:
            raise DiagnosticError(
                "E_SNAPSHOT",
                {"read_point_id": "<ledger>", "row_key": "<ledger>", "field": "sqlite_integrity"},
                "SQLite integrity check failed",
                error=str(exc),
            ) from exc
        if check != "ok":
            raise DiagnosticError(
                "E_SNAPSHOT",
                {"read_point_id": "<ledger>", "row_key": "<ledger>", "field": "sqlite_integrity"},
                "SQLite integrity check failed",
                result=check,
            )

        for row in self.connection.execute("SELECT * FROM rounds ORDER BY ledger_cursor"):
            payload = {
                "schema_version": int(row["schema_version"]),
                "ledger_cursor": int(row["ledger_cursor"]),
                "generation": int(row["generation"]),
                "round_no": int(row["round_no"]),
                "allocation": json.loads(row["allocation_json"]),
                "schedule": json.loads(row["schedule_json"]),
                "decisions": json.loads(row["decisions_json"]),
            }
            actual = digest_json(payload)
            if actual != row["content_digest"]:
                raise DiagnosticError(
                    "E_SNAPSHOT",
                    {
                        "read_point_id": "<ledger>",
                        "row_key": "<ledger>",
                        "field": "round.content_digest",
                        "ledger_cursor": int(row["ledger_cursor"]),
                    },
                    "ledger round content digest does not match stored content",
                    expected=row["content_digest"],
                    actual=actual,
                )

        for row in self.connection.execute("SELECT * FROM snapshots ORDER BY snapshot_no"):
            try:
                payload = json.loads(row["payload_json"])
            except json.JSONDecodeError as exc:
                raise DiagnosticError(
                    "E_SNAPSHOT",
                    {
                        "read_point_id": "<ledger>",
                        "row_key": "<ledger>",
                        "field": "snapshot.payload_json",
                    },
                    "snapshot payload is not valid JSON",
                ) from exc
            embedded = payload.get("content_digest")
            base = dict(payload)
            base.pop("content_digest", None)
            actual = digest_json(base)
            if actual != row["content_digest"] or embedded != actual:
                raise DiagnosticError(
                    "E_SNAPSHOT",
                    {
                        "read_point_id": "<ledger>",
                        "row_key": "<ledger>",
                        "field": "snapshot.content_digest",
                        "snapshot_no": int(row["snapshot_no"]),
                    },
                    "snapshot content digest does not match stored content",
                    expected=row["content_digest"],
                    actual=actual,
                )

    def mark_running_unknown(self) -> list[dict[str, Any]]:
        rows = self.connection.execute("SELECT * FROM inflight WHERE status = 'RUNNING'").fetchall()
        if not rows:
            return []
        marked: list[dict[str, Any]] = []
        now = utc_now_iso()
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            for row in rows:
                reason = "process terminated before round commit"
                self.connection.execute(
                    "UPDATE inflight SET status = ?, updated_at = ? WHERE inflight_id = ?",
                    ("UNKNOWN_REQUIRES_RERUN", now, row["inflight_id"]),
                )
                self.connection.execute(
                    """
                    INSERT INTO unknown_tasks(
                        generation, round_no, query_id, read_point_id, row_key,
                        status, reason, marked_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["generation"],
                        row["round_no"],
                        row["query_id"],
                        row["read_point_id"],
                        row["row_key"],
                        "UNKNOWN_REQUIRES_RERUN",
                        reason,
                        now,
                    ),
                )
                marked.append(
                    {
                        "generation": int(row["generation"]),
                        "round_no": int(row["round_no"]),
                        "query_id": row["query_id"],
                        "read_point_id": row["read_point_id"],
                        "row_key": row["row_key"],
                        "status": "UNKNOWN_REQUIRES_RERUN",
                        "reason": reason,
                    }
                )
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise
        return marked

    def current_generation(self) -> int | None:
        value = self._get_meta("current_generation")
        return None if value is None else int(value)

    def current_round_no(self) -> int:
        return int(self._get_meta("current_round_no") or "0")

    def ledger_seq(self) -> int:
        return int(self._get_meta("ledger_seq") or "0")

    def latest_snapshot(self, generation: int | None = None) -> dict[str, Any] | None:
        if generation is None:
            generation = self.current_generation()
        if generation is None:
            return None
        row = self.connection.execute(
            "SELECT * FROM snapshots WHERE generation = ? ORDER BY ledger_cursor DESC LIMIT 1",
            (generation,),
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(row["payload_json"])
        payload["_stored_content_digest"] = row["content_digest"]
        payload["_snapshot_no"] = int(row["snapshot_no"])
        return payload

    def completed_rounds(self, generation: int) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for row in self.connection.execute(
            "SELECT * FROM rounds WHERE generation = ? ORDER BY round_no",
            (generation,),
        ):
            result.append(
                {
                    "ledger_cursor": int(row["ledger_cursor"]),
                    "generation": int(row["generation"]),
                    "round_no": int(row["round_no"]),
                    "schema_version": int(row["schema_version"]),
                    "allocation": json.loads(row["allocation_json"]),
                    "schedule": json.loads(row["schedule_json"]),
                    "decisions": json.loads(row["decisions_json"]),
                    "content_digest": row["content_digest"],
                    "committed_at": row["committed_at"],
                }
            )
        return result

    def all_committed_decisions(self, generation: int) -> list[dict[str, Any]]:
        decisions: list[dict[str, Any]] = []
        for round_row in self.completed_rounds(generation):
            decisions.extend(round_row["decisions"])
        return sorted(
            decisions,
            key=lambda item: (
                item.get("read_point_id", ""),
                item.get("row_key", ""),
                item.get("query_id", ""),
            ),
        )

    def all_schedule_traces(self, generation: int) -> list[dict[str, Any]]:
        traces: list[dict[str, Any]] = []
        for round_row in self.completed_rounds(generation):
            traces.extend(round_row["schedule"])
        return traces

    def unknown_tasks(self) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM unknown_tasks WHERE status = 'UNKNOWN_REQUIRES_RERUN' ORDER BY unknown_id"
            )
        ]

    def all_unknown_task_history(self) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM unknown_tasks ORDER BY unknown_id"
            )
        ]

    def insert_inflight(
        self,
        generation: int,
        round_no: int,
        allocations: Iterable[Any],
    ) -> None:
        now = utc_now_iso()
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            for allocation in allocations:
                self.connection.execute(
                    """
                    INSERT INTO inflight(
                        generation, round_no, query_id, read_point_id, row_key,
                        owner_pid, status, created_at, updated_at
                    ) VALUES(?, ?, ?, ?, ?, ?, 'RUNNING', ?, ?)
                    """,
                    (
                        generation,
                        round_no,
                        allocation.query_id,
                        allocation.read_point_id,
                        allocation.row_key,
                        os.getpid(),
                        now,
                        now,
                    ),
                )
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

    def commit_round(
        self,
        *,
        dataset: Dataset,
        generation: int,
        round_no: int,
        budget_round: Any,
        budget_plan: BudgetPlan,
        traces: tuple[TaskTrace, ...],
        decisions: list[VisibilityDecision],
        all_decisions: list[VisibilityDecision],
    ) -> dict[str, Any]:
        ledger_cursor = self._next_ledger_cursor()
        allocation_payload = budget_round.as_dict()
        schedule_payload = [trace.as_dict() for trace in traces]
        decision_payload = [decision.as_dict() for decision in decisions]
        round_digest = digest_json(
            {
                "schema_version": SCHEMA_VERSION,
                "ledger_cursor": ledger_cursor,
                "generation": generation,
                "round_no": round_no,
                "allocation": allocation_payload,
                "schedule": schedule_payload,
                "decisions": decision_payload,
            }
        )
        snapshot_base = {
            "schema_version": SCHEMA_VERSION,
            "generation": generation,
            "ledger_cursor": ledger_cursor,
            "cursor": {"generation": generation, "round_no": round_no, "ledger_cursor": ledger_cursor},
            "manifest": dataset.manifest.as_dict(),
            "metadata": dataset.metadata_digest(),
            "declarations": dataset.as_declaration_dict(),
            "budget_plan": budget_plan.as_dict(),
            "conclusions": [item.as_dict() for item in _sorted_decisions(all_decisions)],
            "conclusion_digest": decisions_digest(all_decisions),
            "round_digests": [
                {
                    "round_no": int(row["round_no"]),
                    "ledger_cursor": int(row["ledger_cursor"]),
                    "content_digest": row["content_digest"],
                }
                for row in self.connection.execute(
                    "SELECT round_no, ledger_cursor, content_digest FROM rounds WHERE generation = ? ORDER BY round_no",
                    (generation,),
                )
            ]
            + [
                {
                    "round_no": round_no,
                    "ledger_cursor": ledger_cursor,
                    "content_digest": round_digest,
                }
            ],
        }
        snapshot_digest = digest_json(snapshot_base)
        snapshot_payload = dict(snapshot_base)
        snapshot_payload["content_digest"] = snapshot_digest
        now = utc_now_iso()

        try:
            self.connection.execute("BEGIN IMMEDIATE")
            self.connection.execute(
                """
                INSERT INTO rounds(
                    ledger_cursor, generation, round_no, schema_version,
                    allocation_json, schedule_json, decisions_json,
                    content_digest, committed_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ledger_cursor,
                    generation,
                    round_no,
                    SCHEMA_VERSION,
                    canonical_json(allocation_payload),
                    canonical_json(schedule_payload),
                    canonical_json(decision_payload),
                    round_digest,
                    now,
                ),
            )
            self.connection.execute(
                """
                INSERT INTO snapshots(
                    snapshot_no, ledger_cursor, generation, payload_json, content_digest, created_at
                ) VALUES(?, ?, ?, ?, ?, ?)
                """,
                (
                    ledger_cursor,
                    ledger_cursor,
                    generation,
                    canonical_json(snapshot_payload),
                    snapshot_digest,
                    now,
                ),
            )
            self.connection.execute(
                "DELETE FROM inflight WHERE generation = ? AND round_no = ?",
                (generation, round_no),
            )
            self.connection.execute(
                """
                UPDATE unknown_tasks
                   SET status = 'RERUN_COMPLETED',
                       reason = 'the unfinished round was rerun and committed'
                 WHERE generation = ? AND round_no = ?
                   AND status = 'UNKNOWN_REQUIRES_RERUN'
                """,
                (generation, round_no),
            )
            self.connection.execute(
                "UPDATE inflight SET status = 'COMMITTED', updated_at = ? "
                "WHERE generation = ? AND round_no = ? AND status = 'RUNNING'",
                (now, generation, round_no),
            )
            self._set_meta("current_generation", str(generation))
            self._set_meta("current_round_no", str(round_no))
            self._set_meta("ledger_seq", str(ledger_cursor))
            self._set_meta("last_commit_at", now)
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise
        return {
            "ledger_cursor": ledger_cursor,
            "generation": generation,
            "round_no": round_no,
            "round_content_digest": round_digest,
            "snapshot_content_digest": snapshot_digest,
            "conclusion_digest": snapshot_base["conclusion_digest"],
            "committed_at": now,
        }

    def record_reconciliation(
        self,
        *,
        generation: int,
        old_snapshot_digest: str | None,
        new_dataset_digest: str,
        location: dict[str, Any],
        affected: dict[str, Any],
        changes: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "generation": generation,
            "old_snapshot_digest": old_snapshot_digest,
            "new_dataset_digest": new_dataset_digest,
            "location": location,
            "affected": affected,
            "changes": changes,
        }
        content_digest = digest_json(payload)
        now = utc_now_iso()
        cursor = self.connection.execute(
            """
            INSERT INTO reconciliations(
                generation, old_snapshot_digest, new_dataset_digest,
                location_json, affected_json, changes_json, content_digest, created_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                generation,
                old_snapshot_digest,
                new_dataset_digest,
                canonical_json(location),
                canonical_json(affected),
                canonical_json(changes),
                content_digest,
                now,
            ),
        ).lastrowid
        return {
            "reconciliation_id": int(cursor),
            "content_digest": content_digest,
            "created_at": now,
            **payload,
        }

    def latest_reconciliation(self) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM reconciliations ORDER BY reconciliation_id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return {
            "reconciliation_id": int(row["reconciliation_id"]),
            "generation": int(row["generation"]),
            "old_snapshot_digest": row["old_snapshot_digest"],
            "new_dataset_digest": row["new_dataset_digest"],
            "location": json.loads(row["location_json"]),
            "affected": json.loads(row["affected_json"]),
            "changes": json.loads(row["changes_json"]),
            "content_digest": row["content_digest"],
            "created_at": row["created_at"],
        }


def _sorted_decisions(decisions: Iterable[VisibilityDecision]) -> list[VisibilityDecision]:
    return sorted(decisions, key=lambda item: (item.read_point_id, item.row_key, item.query_id))
