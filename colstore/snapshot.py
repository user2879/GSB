"""SQLite-persisted scan session snapshots with whole-state replacement.

A snapshot stores: segment file-set summaries and format versions, encoding
selection (+ its version), budget parameters (+ version), task traces, result
summary and statistics.  On reopen, snapshot versions are compared against the
current disk state; any mismatch replaces the whole in-memory view instead of
incremental merging, and the previous snapshot stays readable with a diff list.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional

from .errors import SnapshotError
from .model import BudgetParams, ScanResult


class SnapshotStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)) or ".", exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._create_schema()

    def close(self) -> None:
        self._conn.close()

    def _create_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS snapshots (
                snapshot_id TEXT PRIMARY KEY,
                created_at_ms REAL NOT NULL,
                read_only INTEGER NOT NULL DEFAULT 0,
                superseded_by TEXT,
                budget_version INTEGER NOT NULL,
                encoding_selection_version INTEGER NOT NULL,
                budget_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                current_state_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS segment_summaries (
                snapshot_id TEXT NOT NULL,
                segment TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                encoding_version INTEGER NOT NULL,
                PRIMARY KEY (snapshot_id, segment),
                FOREIGN KEY (snapshot_id) REFERENCES snapshots(snapshot_id)
            );
            CREATE TABLE IF NOT EXISTS task_traces (
                snapshot_id TEXT NOT NULL,
                task_id INTEGER NOT NULL,
                task_json TEXT NOT NULL,
                PRIMARY KEY (snapshot_id, task_id),
                FOREIGN KEY (snapshot_id) REFERENCES snapshots(snapshot_id)
            );
            CREATE TABLE IF NOT EXISTS replacement_diffs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                detail_json TEXT NOT NULL,
                FOREIGN KEY (snapshot_id) REFERENCES snapshots(snapshot_id)
            );
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_session(
        self,
        snapshot_id: str,
        result: ScanResult,
        budget: BudgetParams,
        segment_summaries: List[Dict[str, Any]],
        encoding_version: int,
        current_state: Dict[str, Any],
    ) -> None:
        existing = self._conn.execute(
            "SELECT snapshot_id FROM snapshots WHERE snapshot_id = ?", (snapshot_id,)
        ).fetchone()
        if existing is not None:
            raise SnapshotError(
                "snapshot id already exists; sessions are write-once",
                {"snapshot_id": snapshot_id},
            )
        self._conn.execute(
            """
            INSERT INTO snapshots (snapshot_id, created_at_ms, read_only,
                superseded_by, budget_version, encoding_selection_version,
                budget_json, result_json, current_state_json)
            VALUES (?, ?, 0, NULL, ?, ?, ?, ?, ?)
            """,
            (
                snapshot_id,
                time.monotonic() * 1000.0,
                result.budget_version,
                encoding_version,
                json.dumps(budget.to_dict(), sort_keys=True),
                json.dumps(result.to_dict(), sort_keys=True),
                json.dumps(current_state, sort_keys=True),
            ),
        )
        for summary in segment_summaries:
            self._conn.execute(
                """
                INSERT INTO segment_summaries
                    (snapshot_id, segment, summary_json, encoding_version)
                VALUES (?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    summary["segment"],
                    json.dumps(summary, sort_keys=True),
                    summary.get("encoding_selection_version", encoding_version),
                ),
            )
        for task in result.tasks:
            self._conn.execute(
                "INSERT INTO task_traces (snapshot_id, task_id, task_json) VALUES (?, ?, ?)",
                (snapshot_id, task.task_id, json.dumps(task.to_dict(), sort_keys=True)),
            )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Reopen / compare / whole replacement
    # ------------------------------------------------------------------

    def list_snapshots(self) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT snapshot_id, created_at_ms, read_only, superseded_by, "
            "budget_version, encoding_selection_version FROM snapshots ORDER BY created_at_ms"
        ).fetchall()
        return [dict(row) for row in rows]

    def load_session(self, snapshot_id: str) -> Dict[str, Any]:
        row = self._conn.execute(
            "SELECT * FROM snapshots WHERE snapshot_id = ?", (snapshot_id,)
        ).fetchone()
        if row is None:
            raise SnapshotError("snapshot not found", {"snapshot_id": snapshot_id})
        segments = [
            json.loads(item["summary_json"])
            for item in self._conn.execute(
                "SELECT summary_json FROM segment_summaries WHERE snapshot_id = ?",
                (snapshot_id,),
            ).fetchall()
        ]
        tasks = [
            json.loads(item["task_json"])
            for item in self._conn.execute(
                "SELECT task_json FROM task_traces WHERE snapshot_id = ? ORDER BY task_id",
                (snapshot_id,),
            ).fetchall()
        ]
        diffs = [
            {"kind": item["kind"], "detail": json.loads(item["detail_json"])}
            for item in self._conn.execute(
                "SELECT kind, detail_json FROM replacement_diffs WHERE snapshot_id = ?",
                (snapshot_id,),
            ).fetchall()
        ]
        return {
            "snapshot_id": snapshot_id,
            "read_only": bool(row["read_only"]),
            "superseded_by": row["superseded_by"],
            "budget_version": row["budget_version"],
            "encoding_selection_version": row["encoding_selection_version"],
            "budget": json.loads(row["budget_json"]),
            "result": json.loads(row["result_json"]),
            "current_state": json.loads(row["current_state_json"]),
            "segment_summaries": segments,
            "task_traces": tasks,
            "replacement_diffs": diffs,
        }

    def compare_with_current(
        self, snapshot_id: str, current_segments: List[Dict[str, Any]], current_encoding_version: int
    ) -> List[Dict[str, Any]]:
        """Return structured diffs; empty list means snapshot matches disk."""
        session = self.load_session(snapshot_id)
        diffs: List[Dict[str, Any]] = []
        snap_index = {item["segment"]: item for item in session["segment_summaries"]}
        current_index = {item["segment"]: item for item in current_segments}
        all_names = sorted(set(snap_index) | set(current_index))
        for name in all_names:
            snap = snap_index.get(name)
            current = current_index.get(name)
            if snap is None:
                diffs.append({"kind": "segment_added", "segment": name, "snapshot": None,
                              "current": current})
                continue
            if current is None:
                diffs.append({"kind": "segment_removed", "segment": name,
                              "snapshot": snap, "current": None})
                continue
            for field_name in ("file_hash", "format_version", "total_rows", "compression"):
                if snap.get(field_name) != current.get(field_name):
                    diffs.append({
                        "kind": "segment_field_changed",
                        "segment": name,
                        "field": field_name,
                        "snapshot": snap.get(field_name),
                        "current": current.get(field_name),
                    })
        if session["encoding_selection_version"] != current_encoding_version:
            diffs.append({
                "kind": "encoding_version_changed",
                "segment": "*",
                "snapshot": session["encoding_selection_version"],
                "current": current_encoding_version,
            })
        return diffs

    def mark_superseded(
        self, old_snapshot_id: str, new_snapshot_id: str, diffs: List[Dict[str, Any]]
    ) -> None:
        """Freeze the old snapshot read-only and record replacement diffs."""
        self._conn.execute(
            "UPDATE snapshots SET read_only = 1, superseded_by = ? WHERE snapshot_id = ?",
            (new_snapshot_id, old_snapshot_id),
        )
        for diff in diffs:
            self._conn.execute(
                "INSERT INTO replacement_diffs (snapshot_id, kind, detail_json) VALUES (?, ?, ?)",
                (old_snapshot_id, diff["kind"], json.dumps(diff, sort_keys=True)),
            )
        self._conn.commit()

    def assert_not_read_only(self, snapshot_id: str) -> None:
        row = self._conn.execute(
            "SELECT read_only FROM snapshots WHERE snapshot_id = ?", (snapshot_id,)
        ).fetchone()
        if row is not None and row["read_only"]:
            raise SnapshotError(
                "snapshot is read-only (superseded); create a new session instead",
                {"snapshot_id": snapshot_id},
            )
