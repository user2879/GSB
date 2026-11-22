"""SQLite ledger with atomic batch commits and whole-snapshot replacement."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from . import LEDGER_FORMAT_VERSION
from .canonical import canonical_json, digest_value, sha256_bytes
from .engine import Execution
from .errors import EReplay
from .fixedpoint import FIXED_POINT_VERSION


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


SCHEMA = """
CREATE TABLE IF NOT EXISTS ledger_meta (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    format_version INTEGER NOT NULL,
    plan_version_counter INTEGER NOT NULL DEFAULT 0,
    content_digest TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_version INTEGER NOT NULL UNIQUE,
    format_version INTEGER NOT NULL,
    catalog_digest TEXT NOT NULL,
    catalog_path TEXT NOT NULL,
    catalog_json TEXT NOT NULL,
    dataset_id TEXT NOT NULL,
    rows_per_page INTEGER NOT NULL,
    group_key TEXT NOT NULL,
    budget INTEGER NOT NULL,
    route TEXT NOT NULL,
    route_decision_json TEXT NOT NULL,
    batch_size INTEGER NOT NULL,
    fixed_point_version TEXT NOT NULL,
    scan_digest TEXT NOT NULL,
    status TEXT NOT NULL,
    cursor INTEGER NOT NULL DEFAULT 0,
    final_result_json TEXT,
    result_digest TEXT,
    result_bytes_sha256 TEXT,
    replacement_reason TEXT,
    previous_plan_version INTEGER,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS batches (
    snapshot_id INTEGER NOT NULL,
    batch_index INTEGER NOT NULL,
    cursor_start INTEGER NOT NULL,
    cursor_end INTEGER NOT NULL,
    fixed_point_version TEXT NOT NULL,
    group_state_digest TEXT NOT NULL,
    batch_digest TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (snapshot_id, batch_index),
    FOREIGN KEY (snapshot_id) REFERENCES snapshots(snapshot_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS page_outcomes (
    snapshot_id INTEGER NOT NULL,
    batch_index INTEGER NOT NULL,
    access_seq INTEGER NOT NULL,
    page_id INTEGER NOT NULL,
    outcome TEXT NOT NULL,
    cache_hit INTEGER NOT NULL,
    physical_scan INTEGER NOT NULL,
    slot INTEGER,
    payload_json TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    PRIMARY KEY (snapshot_id, batch_index, access_seq),
    FOREIGN KEY (snapshot_id, batch_index) REFERENCES batches(snapshot_id, batch_index) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS eviction_events (
    snapshot_id INTEGER NOT NULL,
    batch_index INTEGER NOT NULL,
    event_seq INTEGER NOT NULL,
    access_seq INTEGER NOT NULL,
    slot INTEGER NOT NULL,
    victim_page_id INTEGER NOT NULL,
    incoming_page_id INTEGER NOT NULL,
    reason TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    PRIMARY KEY (snapshot_id, batch_index, event_seq),
    FOREIGN KEY (snapshot_id, batch_index) REFERENCES batches(snapshot_id, batch_index) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS rescan_events (
    snapshot_id INTEGER NOT NULL,
    batch_index INTEGER NOT NULL,
    event_seq INTEGER NOT NULL,
    access_seq INTEGER NOT NULL,
    page_id INTEGER NOT NULL,
    reason TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    PRIMARY KEY (snapshot_id, batch_index, event_seq),
    FOREIGN KEY (snapshot_id, batch_index) REFERENCES batches(snapshot_id, batch_index) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS group_state (
    snapshot_id INTEGER NOT NULL,
    batch_index INTEGER NOT NULL,
    group_key INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    PRIMARY KEY (snapshot_id, batch_index, group_key),
    FOREIGN KEY (snapshot_id, batch_index) REFERENCES batches(snapshot_id, batch_index) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS page_group_refs (
    snapshot_id INTEGER NOT NULL,
    page_id INTEGER NOT NULL,
    group_key INTEGER NOT NULL,
    PRIMARY KEY (snapshot_id, page_id, group_key),
    FOREIGN KEY (snapshot_id) REFERENCES snapshots(snapshot_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS batch_attempts (
    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER NOT NULL,
    batch_index INTEGER NOT NULL,
    cursor_start INTEGER NOT NULL,
    cursor_end INTEGER NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    error_code TEXT,
    FOREIGN KEY (snapshot_id) REFERENCES snapshots(snapshot_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS replacements (
    replacement_id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_plan_version INTEGER,
    to_plan_version INTEGER NOT NULL,
    reason TEXT NOT NULL,
    old_catalog_digest TEXT,
    new_catalog_digest TEXT NOT NULL,
    old_result_digest TEXT,
    new_result_digest TEXT,
    diff_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_page_outcomes_snapshot ON page_outcomes(snapshot_id, access_seq);
CREATE INDEX IF NOT EXISTS idx_evictions_snapshot ON eviction_events(snapshot_id, event_seq);
CREATE INDEX IF NOT EXISTS idx_rescans_snapshot ON rescan_events(snapshot_id, event_seq);
CREATE INDEX IF NOT EXISTS idx_group_state_snapshot ON group_state(snapshot_id, batch_index, group_key);
CREATE INDEX IF NOT EXISTS idx_attempts_snapshot ON batch_attempts(snapshot_id, batch_index, status);
"""


class Ledger:
    def __init__(self, path: str | Path, *, verify: bool = True) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(
            str(self.path),
            timeout=30,
            isolation_level=None,
        )
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = FULL")
        self.connection.execute("PRAGMA busy_timeout = 30000")
        self._initialize()
        if verify:
            self.verify_content_digest()
        self.mark_started_unknown()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _initialize(self) -> None:
        existing = self.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ledger_meta'"
        ).fetchone()
        self.connection.executescript(SCHEMA)
        row = self.connection.execute("SELECT * FROM ledger_meta WHERE id=1").fetchone()
        if row is None:
            self._begin()
            try:
                self.connection.execute(
                    "INSERT INTO ledger_meta(id, format_version, plan_version_counter, content_digest, updated_at) "
                    "VALUES(1, ?, 0, NULL, ?)",
                    (LEDGER_FORMAT_VERSION, utc_now()),
                )
                self.refresh_content_digest()
                self.connection.execute("COMMIT")
            except Exception:
                self.connection.execute("ROLLBACK")
                raise
        else:
            if int(row["format_version"]) != LEDGER_FORMAT_VERSION:
                raise EReplay(
                    "ledger format version is unsupported",
                    field="ledger.format_version",
                    details={
                        "actual": int(row["format_version"]),
                        "expected": LEDGER_FORMAT_VERSION,
                    },
                )
            if existing is None:
                raise EReplay("ledger metadata exists without a supported schema", field="ledger.schema")

    def _begin(self) -> None:
        self.connection.execute("BEGIN IMMEDIATE")

    def _rows(self, query: str, parameters: Iterable[Any] = ()) -> list[dict[str, Any]]:
        return [dict(row) for row in self.connection.execute(query, tuple(parameters)).fetchall()]

    def _state_payload(self) -> dict[str, Any]:
        meta = self.connection.execute(
            "SELECT format_version, plan_version_counter FROM ledger_meta WHERE id=1"
        ).fetchone()
        return {
            "format_version": int(meta["format_version"]),
            "plan_version_counter": int(meta["plan_version_counter"]),
            "snapshots": self._rows("SELECT * FROM snapshots ORDER BY snapshot_id"),
            "batches": self._rows("SELECT * FROM batches ORDER BY snapshot_id, batch_index"),
            "page_outcomes": self._rows(
                "SELECT * FROM page_outcomes ORDER BY snapshot_id, batch_index, access_seq"
            ),
            "eviction_events": self._rows(
                "SELECT * FROM eviction_events ORDER BY snapshot_id, batch_index, event_seq"
            ),
            "rescan_events": self._rows(
                "SELECT * FROM rescan_events ORDER BY snapshot_id, batch_index, event_seq"
            ),
            "group_state": self._rows(
                "SELECT * FROM group_state ORDER BY snapshot_id, batch_index, group_key"
            ),
            "page_group_refs": self._rows(
                "SELECT * FROM page_group_refs ORDER BY snapshot_id, page_id, group_key"
            ),
            "batch_attempts": self._rows("SELECT * FROM batch_attempts ORDER BY attempt_id"),
            "replacements": self._rows("SELECT * FROM replacements ORDER BY replacement_id"),
        }

    def compute_content_digest(self) -> str:
        return digest_value(self._state_payload())

    def refresh_content_digest(self) -> str:
        value = self.compute_content_digest()
        self.connection.execute(
            "UPDATE ledger_meta SET content_digest=?, updated_at=? WHERE id=1",
            (value, utc_now()),
        )
        return value

    def verify_content_digest(self) -> str:
        row = self.connection.execute(
            "SELECT content_digest FROM ledger_meta WHERE id=1"
        ).fetchone()
        expected = row["content_digest"] if row else None
        actual = self.compute_content_digest()
        if expected is None:
            self._begin()
            try:
                self.refresh_content_digest()
                self.connection.execute("COMMIT")
            except Exception:
                self.connection.execute("ROLLBACK")
                raise
            return actual
        if expected != actual:
            raise EReplay(
                "ledger content digest mismatch",
                field="ledger.content_digest",
                details={"stored": expected, "computed": actual},
            )
        return expected

    def mark_started_unknown(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM batch_attempts WHERE status='STARTED' ORDER BY attempt_id"
        ).fetchall()
        if not rows:
            return []
        self._begin()
        try:
            self.connection.execute(
                "UPDATE batch_attempts SET status='UNKNOWN', finished_at=?, "
                "error_code='FORCED_TERMINATION_OR_UNCOMMITTED' WHERE status='STARTED'",
                (utc_now(),),
            )
            self.refresh_content_digest()
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise
        return [dict(row) for row in rows]

    def unknown_attempts(self) -> list[dict[str, Any]]:
        return self._rows(
            "SELECT * FROM batch_attempts WHERE status='UNKNOWN' ORDER BY attempt_id"
        )

    def get_active_snapshot(self) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM snapshots ORDER BY snapshot_id DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def snapshot_config(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        return {
            "catalog_digest": snapshot["catalog_digest"],
            "rows_per_page": int(snapshot["rows_per_page"]),
            "group_key": snapshot["group_key"],
            "budget": int(snapshot["budget"]),
            "route": snapshot["route"],
            "batch_size": int(snapshot["batch_size"]),
        }

    def create_snapshot(
        self,
        execution: Execution,
        *,
        catalog_path: str,
        replacement_reason: str | None,
    ) -> tuple[int, int, dict[str, Any] | None]:
        old = self.get_active_snapshot()
        old_version = int(old["plan_version"]) if old else None
        meta = self.connection.execute(
            "SELECT plan_version_counter FROM ledger_meta WHERE id=1"
        ).fetchone()
        plan_version = int(meta["plan_version_counter"]) + 1
        self._begin()
        try:
            if old is not None:
                self.connection.execute(
                    "DELETE FROM snapshots WHERE snapshot_id=?", (old["snapshot_id"],)
                )
            self.connection.execute(
                "UPDATE ledger_meta SET plan_version_counter=? WHERE id=1", (plan_version,)
            )
            scan_digest = digest_value(
                {
                    "scan_sequence": list(execution.catalog.scan_sequence),
                    "catalog_digest": execution.catalog.digest,
                }
            )
            cursor = self.connection.execute(
                """
                INSERT INTO snapshots(
                    plan_version, format_version, catalog_digest, catalog_path,
                    catalog_json, dataset_id, rows_per_page, group_key, budget,
                    route, route_decision_json, batch_size, fixed_point_version,
                    scan_digest, status, cursor, replacement_reason,
                    previous_plan_version, created_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    plan_version,
                    LEDGER_FORMAT_VERSION,
                    execution.catalog.digest,
                    catalog_path,
                    canonical_json(execution.catalog.raw),
                    execution.catalog.dataset_id,
                    execution.catalog.rows_per_page,
                    execution.catalog.group_key,
                    execution.budget,
                    execution.route.selected_route,
                    canonical_json(execution.route.payload()),
                    _effective_batch_size(execution),
                    FIXED_POINT_VERSION,
                    scan_digest,
                    "BUILDING",
                    0,
                    replacement_reason,
                    old_version,
                    utc_now(),
                ),
            )
            snapshot_id = int(cursor.lastrowid)
            self.refresh_content_digest()
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise
        return snapshot_id, plan_version, old

    def _next_attempt(
        self,
        snapshot_id: int,
        batch_index: int,
        cursor_start: int,
        cursor_end: int,
    ) -> int:
        self._begin()
        try:
            cursor = self.connection.execute(
                "INSERT INTO batch_attempts(snapshot_id, batch_index, cursor_start, cursor_end, status, started_at) "
                "VALUES(?,?,?,?,?,?)",
                (snapshot_id, batch_index, cursor_start, cursor_end, "STARTED", utc_now()),
            )
            attempt_id = int(cursor.lastrowid)
            self.refresh_content_digest()
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise
        return attempt_id

    def persist_batches(
        self,
        execution: Execution,
        snapshot_id: int,
        *,
        start_batch_index: int = 1,
        crash_during_batch: int | None = None,
    ) -> list[int]:
        committed: list[int] = []
        for batch in execution.batches:
            if batch.batch_index < start_batch_index:
                continue
            attempt_id = self._next_attempt(
                snapshot_id,
                batch.batch_index,
                batch.cursor_start,
                batch.cursor_end,
            )
            self._begin()
            try:
                self.connection.execute(
                    "INSERT INTO batches(snapshot_id, batch_index, cursor_start, cursor_end, "
                    "fixed_point_version, group_state_digest, batch_digest, created_at) "
                    "VALUES(?,?,?,?,?,?,?,?)",
                    (
                        snapshot_id,
                        batch.batch_index,
                        batch.cursor_start,
                        batch.cursor_end,
                        FIXED_POINT_VERSION,
                        batch.group_state_digest,
                        batch.batch_digest,
                        utc_now(),
                    ),
                )
                for outcome in batch.page_outcomes:
                    payload_json = canonical_json(outcome)
                    self.connection.execute(
                        "INSERT INTO page_outcomes(snapshot_id, batch_index, access_seq, page_id, "
                        "outcome, cache_hit, physical_scan, slot, payload_json, payload_sha256) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (
                            snapshot_id,
                            batch.batch_index,
                            outcome["access_seq"],
                            outcome["page_id"],
                            outcome["outcome"],
                            1 if outcome["cache_hit"] else 0,
                            1 if outcome["physical_scan"] else 0,
                            outcome["slot"],
                            payload_json,
                            sha256_bytes(payload_json.encode("utf-8")),
                        ),
                    )
                for event in batch.eviction_events:
                    payload_json = canonical_json(event)
                    self.connection.execute(
                        "INSERT INTO eviction_events(snapshot_id, batch_index, event_seq, access_seq, "
                        "slot, victim_page_id, incoming_page_id, reason, payload_json, payload_sha256) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (
                            snapshot_id,
                            batch.batch_index,
                            event["event_seq"],
                            event["access_seq"],
                            event["slot"],
                            event["victim_page_id"],
                            event["incoming_page_id"],
                            event["reason"],
                            payload_json,
                            sha256_bytes(payload_json.encode("utf-8")),
                        ),
                    )
                for event in batch.rescan_events:
                    payload_json = canonical_json(event)
                    self.connection.execute(
                        "INSERT INTO rescan_events(snapshot_id, batch_index, event_seq, access_seq, "
                        "page_id, reason, payload_json, payload_sha256) "
                        "VALUES(?,?,?,?,?,?,?,?)",
                        (
                            snapshot_id,
                            batch.batch_index,
                            event["event_seq"],
                            event["access_seq"],
                            event["page_id"],
                            event["reason"],
                            payload_json,
                            sha256_bytes(payload_json.encode("utf-8")),
                        ),
                    )
                for group in batch.group_state:
                    payload_json = canonical_json(group)
                    self.connection.execute(
                        "INSERT INTO group_state(snapshot_id, batch_index, group_key, payload_json, payload_sha256) "
                        "VALUES(?,?,?,?,?)",
                        (
                            snapshot_id,
                            batch.batch_index,
                            int(group["group_key"]),
                            payload_json,
                            sha256_bytes(payload_json.encode("utf-8")),
                        ),
                    )
                final_batch = batch.batch_index == execution.batches[-1].batch_index
                if final_batch:
                    for group in execution.final_group_state:
                        group_key = int(group["group_key"])
                        for page_id in execution.catalog.pages:
                            page = execution.catalog.pages[page_id]
                            if bool((page.columns[execution.catalog.group_key] == group_key).any()):
                                self.connection.execute(
                                    "INSERT OR IGNORE INTO page_group_refs(snapshot_id, page_id, group_key) "
                                    "VALUES(?,?,?)",
                                    (snapshot_id, page_id, group_key),
                                )
                    result_json = canonical_json(execution.result)
                    self.connection.execute(
                        "UPDATE snapshots SET status='COMPLETE', cursor=?, final_result_json=?, "
                        "result_digest=?, result_bytes_sha256=? WHERE snapshot_id=?",
                        (
                            batch.cursor_end,
                            result_json,
                            execution.result_digest,
                            sha256_bytes(execution.result_bytes()),
                            snapshot_id,
                        ),
                    )
                else:
                    self.connection.execute(
                        "UPDATE snapshots SET status='IN_PROGRESS', cursor=? WHERE snapshot_id=?",
                        (batch.cursor_end, snapshot_id),
                    )
                self.connection.execute(
                    "UPDATE batch_attempts SET status='COMMITTED', finished_at=? WHERE attempt_id=?",
                    (utc_now(), attempt_id),
                )
                if crash_during_batch == batch.batch_index:
                    # Simulate an abrupt process kill while the transaction is open.
                    os._exit(137)
                self.refresh_content_digest()
                self.connection.execute("COMMIT")
            except BaseException:
                try:
                    self.connection.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise
            committed.append(batch.batch_index)
        return committed

    def remaining_batches(self, execution: Execution) -> list[int]:
        snapshot = self.get_active_snapshot()
        if snapshot is None:
            return [batch.batch_index for batch in execution.batches]
        cursor = int(snapshot["cursor"])
        return [batch.batch_index for batch in execution.batches if batch.cursor_start >= cursor]

    def committed_batch_count(self, snapshot_id: int) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) AS count FROM batches WHERE snapshot_id=?", (snapshot_id,)
        ).fetchone()
        return int(row["count"])

    def read_batches(self, snapshot_id: int) -> list[dict[str, Any]]:
        return self._rows(
            "SELECT * FROM batches WHERE snapshot_id=? ORDER BY batch_index", (snapshot_id,)
        )

    def read_page_outcomes(self, snapshot_id: int) -> list[dict[str, Any]]:
        return self._rows(
            "SELECT * FROM page_outcomes WHERE snapshot_id=? ORDER BY batch_index, access_seq",
            (snapshot_id,),
        )

    def read_evictions(self, snapshot_id: int) -> list[dict[str, Any]]:
        return self._rows(
            "SELECT * FROM eviction_events WHERE snapshot_id=? ORDER BY event_seq",
            (snapshot_id,),
        )

    def read_rescans(self, snapshot_id: int) -> list[dict[str, Any]]:
        return self._rows(
            "SELECT * FROM rescan_events WHERE snapshot_id=? ORDER BY event_seq",
            (snapshot_id,),
        )

    def read_group_states(self, snapshot_id: int, batch_index: int | None = None) -> list[dict[str, Any]]:
        if batch_index is None:
            return self._rows(
                "SELECT * FROM group_state WHERE snapshot_id=? ORDER BY batch_index, group_key",
                (snapshot_id,),
            )
        return self._rows(
            "SELECT * FROM group_state WHERE snapshot_id=? AND batch_index=? ORDER BY group_key",
            (snapshot_id, batch_index),
        )

    def page_group_refs(self, snapshot_id: int) -> list[dict[str, Any]]:
        return self._rows(
            "SELECT * FROM page_group_refs WHERE snapshot_id=? ORDER BY page_id, group_key",
            (snapshot_id,),
        )

    def record_replacement(
        self,
        *,
        from_plan_version: int | None,
        to_plan_version: int,
        reason: str,
        old_catalog_digest: str | None,
        new_catalog_digest: str,
        old_result_digest: str | None,
        new_result_digest: str | None,
        diff: dict[str, Any],
    ) -> None:
        self._begin()
        try:
            self.connection.execute(
                "INSERT INTO replacements(from_plan_version, to_plan_version, reason, old_catalog_digest, "
                "new_catalog_digest, old_result_digest, new_result_digest, diff_json, created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    from_plan_version,
                    to_plan_version,
                    reason,
                    old_catalog_digest,
                    new_catalog_digest,
                    old_result_digest,
                    new_result_digest,
                    canonical_json(diff),
                    utc_now(),
                ),
            )
            self.refresh_content_digest()
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

    def ledger_summary(self) -> dict[str, Any]:
        meta = self.connection.execute("SELECT * FROM ledger_meta WHERE id=1").fetchone()
        snapshot = self.get_active_snapshot()
        return {
            "path": str(self.path),
            "format_version": int(meta["format_version"]),
            "content_digest": meta["content_digest"],
            "plan_version_counter": int(meta["plan_version_counter"]),
            "active_snapshot_id": snapshot["snapshot_id"] if snapshot else None,
            "active_plan_version": snapshot["plan_version"] if snapshot else None,
            "active_status": snapshot["status"] if snapshot else None,
            "progress_cursor": snapshot["cursor"] if snapshot else 0,
            "unknown_attempts": self.unknown_attempts(),
        }

    def sqlite_integrity_check(self) -> str:
        row = self.connection.execute("PRAGMA integrity_check").fetchone()
        return str(row[0]) if row else "unknown"


def _effective_batch_size(execution: Execution) -> int:
    if execution.batches:
        return execution.batches[0].cursor_end - execution.batches[0].cursor_start
    return execution.catalog.batch_size
