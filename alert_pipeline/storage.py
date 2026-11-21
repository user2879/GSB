from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .dedup import DedupState
from .engine import BatchExecution, PipelineEngine
from .errors import PipelineError
from .models import (
    AggregateAlert,
    aggregate_from_public_dict,
)
from .spec import LoadedDataset
from .util import canonical_json, digest_json, epoch_ms


SCHEMA_VERSION = 2


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS generations (
    generation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    declaration_version INTEGER NOT NULL UNIQUE,
    semantic_digest TEXT NOT NULL,
    semantic_payload_json TEXT NOT NULL,
    config_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('building', 'complete')),
    is_current INTEGER NOT NULL DEFAULT 0,
    is_stale INTEGER NOT NULL DEFAULT 0,
    seed INTEGER NOT NULL,
    format_version INTEGER NOT NULL,
    default_silence_window_ms INTEGER NOT NULL,
    batch_size INTEGER NOT NULL,
    alert_count INTEGER NOT NULL DEFAULT 0,
    created_at_ms INTEGER NOT NULL,
    completed_at_ms INTEGER
);

CREATE TABLE IF NOT EXISTS batches (
    generation_id INTEGER NOT NULL,
    batch_index INTEGER NOT NULL,
    start_cursor INTEGER NOT NULL,
    end_cursor INTEGER NOT NULL,
    status TEXT NOT NULL,
    business_code TEXT NOT NULL,
    summary_digest TEXT NOT NULL,
    conclusion_digest TEXT NOT NULL,
    aggregate_count INTEGER NOT NULL,
    ledger_count INTEGER NOT NULL,
    normalized_count INTEGER NOT NULL,
    timeline_json TEXT NOT NULL,
    peak_global INTEGER NOT NULL,
    peak_by_service_json TEXT NOT NULL,
    summary_json TEXT NOT NULL,
    committed_at_ms INTEGER NOT NULL,
    is_stale INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (generation_id, batch_index),
    FOREIGN KEY (generation_id) REFERENCES generations(generation_id)
);

CREATE TABLE IF NOT EXISTS batch_attempts (
    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
    generation_id INTEGER NOT NULL,
    batch_index INTEGER NOT NULL,
    start_cursor INTEGER NOT NULL,
    end_cursor INTEGER NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('unknown', 'committed', 'superseded')),
    started_at_ms INTEGER NOT NULL,
    finished_at_ms INTEGER,
    details_json TEXT,
    FOREIGN KEY (generation_id) REFERENCES generations(generation_id)
);

CREATE TABLE IF NOT EXISTS progress (
    generation_id INTEGER PRIMARY KEY,
    next_cursor INTEGER NOT NULL,
    last_batch_index INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    FOREIGN KEY (generation_id) REFERENCES generations(generation_id)
);

CREATE TABLE IF NOT EXISTS aggregates (
    generation_id INTEGER NOT NULL,
    alert_id TEXT NOT NULL,
    service_id TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    first_timestamp_ms INTEGER NOT NULL,
    suppressed_count INTEGER NOT NULL,
    content_json TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    is_stale INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (generation_id, alert_id),
    FOREIGN KEY (generation_id) REFERENCES generations(generation_id)
);

CREATE TABLE IF NOT EXISTS ledger (
    generation_id INTEGER NOT NULL,
    alert_id TEXT NOT NULL,
    service_id TEXT NOT NULL,
    candidate_services_json TEXT NOT NULL,
    paths_json TEXT NOT NULL,
    rule_id TEXT NOT NULL,
    rule_version INTEGER NOT NULL,
    graph_version INTEGER NOT NULL,
    declared_version INTEGER NOT NULL,
    format_version INTEGER NOT NULL,
    content_json TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    is_stale INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (generation_id, alert_id),
    FOREIGN KEY (generation_id) REFERENCES generations(generation_id)
);

CREATE TABLE IF NOT EXISTS batch_aggregates (
    generation_id INTEGER NOT NULL,
    batch_index INTEGER NOT NULL,
    alert_id TEXT NOT NULL,
    service_id TEXT NOT NULL,
    content_json TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    PRIMARY KEY (generation_id, batch_index, alert_id),
    FOREIGN KEY (generation_id) REFERENCES generations(generation_id)
);

CREATE TABLE IF NOT EXISTS batch_ledger (
    generation_id INTEGER NOT NULL,
    batch_index INTEGER NOT NULL,
    alert_id TEXT NOT NULL,
    service_id TEXT NOT NULL,
    content_json TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    PRIMARY KEY (generation_id, batch_index, alert_id),
    FOREIGN KEY (generation_id) REFERENCES generations(generation_id)
);

CREATE TABLE IF NOT EXISTS replacement_runs (
    generation_id INTEGER PRIMARY KEY,
    previous_generation_id INTEGER,
    changed_config_json TEXT NOT NULL,
    affected_count INTEGER NOT NULL,
    report_json TEXT NOT NULL,
    created_at_ms INTEGER NOT NULL,
    FOREIGN KEY (generation_id) REFERENCES generations(generation_id)
);

CREATE TABLE IF NOT EXISTS replacements (
    replacement_id INTEGER PRIMARY KEY AUTOINCREMENT,
    generation_id INTEGER NOT NULL,
    previous_generation_id INTEGER,
    alert_id TEXT NOT NULL,
    changed_fields_json TEXT NOT NULL,
    before_json TEXT,
    after_json TEXT,
    affected INTEGER NOT NULL,
    FOREIGN KEY (generation_id) REFERENCES generations(generation_id)
);

CREATE INDEX IF NOT EXISTS idx_batches_generation
ON batches(generation_id, batch_index);

CREATE INDEX IF NOT EXISTS idx_attempts_generation
ON batch_attempts(generation_id, batch_index, status);

CREATE INDEX IF NOT EXISTS idx_aggregates_current
ON aggregates(generation_id, is_stale, alert_id);

CREATE INDEX IF NOT EXISTS idx_ledger_current
ON ledger(generation_id, is_stale, alert_id);

CREATE VIEW IF NOT EXISTS current_aggregate_view AS
SELECT a.*
FROM aggregates a
JOIN generations g ON g.generation_id = a.generation_id
WHERE g.is_current = 1 AND g.status = 'complete' AND a.is_stale = 0;

CREATE VIEW IF NOT EXISTS current_ledger_view AS
SELECT l.*
FROM ledger l
JOIN generations g ON g.generation_id = l.generation_id
WHERE g.is_current = 1 AND g.status = 'complete' AND l.is_stale = 0;

"""


@dataclass(frozen=True)
class Generation:
    generation_id: int
    declaration_version: int
    semantic_digest: str
    semantic_payload: dict[str, Any]
    config: dict[str, Any]
    status: str
    is_current: bool
    is_stale: bool
    seed: int
    format_version: int
    default_silence_window_ms: int
    batch_size: int
    alert_count: int


@dataclass(frozen=True)
class ReplayCheck:
    generation_id: int
    batches_checked: int
    alerts_checked: int
    conclusion_digest: str
    unknown_batches: tuple[int, ...]
    exit_report_available: bool

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "business_code": "OK",
            "generation_id": self.generation_id,
            "batches_checked": self.batches_checked,
            "alerts_checked": self.alerts_checked,
            "conclusion_digest": self.conclusion_digest,
            "unknown_batches": list(self.unknown_batches),
            "exit_report_available": self.exit_report_available,
        }


def _as_generation(row: sqlite3.Row) -> Generation:
    return Generation(
        generation_id=row["generation_id"],
        declaration_version=row["declaration_version"],
        semantic_digest=row["semantic_digest"],
        semantic_payload=json.loads(row["semantic_payload_json"]),
        config=json.loads(row["config_json"]),
        status=row["status"],
        is_current=bool(row["is_current"]),
        is_stale=bool(row["is_stale"]),
        seed=row["seed"],
        format_version=row["format_version"],
        default_silence_window_ms=row["default_silence_window_ms"],
        batch_size=row["batch_size"],
        alert_count=row["alert_count"],
    )


def _first_value_diff(
    before: Any, after: Any, prefix: str = ""
) -> tuple[str, Any, Any] | None:
    if type(before) is not type(after):
        return prefix or "$", before, after
    if isinstance(before, dict):
        for key in sorted(set(before) | set(after)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in before:
                return path, None, after[key]
            if key not in after:
                return path, before[key], None
            found = _first_value_diff(before[key], after[key], path)
            if found is not None:
                return found
        return None
    if isinstance(before, list):
        if len(before) != len(after):
            return prefix or "$", before, after
        for index, (left, right) in enumerate(zip(before, after)):
            found = _first_value_diff(left, right, f"{prefix}[{index}]")
            if found is not None:
                return found
        return None
    if before != after:
        return prefix or "$", before, after
    return None


def _changed_paths(before: Any, after: Any, prefix: str = "") -> list[str]:
    if type(before) is not type(after):
        return [prefix or "$"]
    if isinstance(before, dict):
        paths: list[str] = []
        for key in sorted(set(before) | set(after)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in before or key not in after:
                paths.append(path)
            else:
                paths.extend(_changed_paths(before[key], after[key], path))
        return paths or ([prefix or "$"] if before != after else [])
    if isinstance(before, list):
        if len(before) != len(after):
            return [prefix or "$"]
        paths = []
        for index, (left, right) in enumerate(zip(before, after)):
            paths.extend(_changed_paths(left, right, f"{prefix}[{index}]"))
        return paths
    return [prefix or "$"] if before != after else []


class LedgerStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path), timeout=30)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript(SCHEMA_SQL)
        batch_columns = {
            row["name"]
            for row in self.connection.execute("PRAGMA table_info(batches)").fetchall()
        }
        if "is_stale" not in batch_columns:
            self.connection.execute(
                "ALTER TABLE batches ADD COLUMN is_stale INTEGER NOT NULL DEFAULT 0"
            )
        self.connection.execute(
            """
            CREATE VIEW IF NOT EXISTS current_batch_view AS
            SELECT b.*
            FROM batches b
            JOIN generations g ON g.generation_id = b.generation_id
            WHERE g.is_current = 1 AND g.status = 'complete' AND b.is_stale = 0
            """
        )
        self.connection.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self.connection.commit()
        self._attempts: dict[tuple[int, int], int] = {}
        self.crash_before_commit_on_batch: int | None = None

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "LedgerStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _transaction(self):
        return self.connection

    def open_generation(self, config: PipelineConfig, dataset: LoadedDataset) -> Generation:
        semantic_payload = config.semantic_payload(dataset)
        semantic_digest = digest_json(semantic_payload)
        config_json = config.public_dict(dataset)
        row = self.connection.execute(
            "SELECT * FROM generations ORDER BY declaration_version DESC LIMIT 1"
        ).fetchone()
        if row is not None:
            latest = _as_generation(row)
            if config.declaration_version < latest.declaration_version:
                raise PipelineError(
                    "E_VERSION",
                    "declaration version must increase monotonically",
                    field="declaration_version",
                    details={
                        "requested": config.declaration_version,
                        "latest": latest.declaration_version,
                    },
                )
            if config.declaration_version == latest.declaration_version:
                if semantic_digest != latest.semantic_digest:
                    raise PipelineError(
                        "E_VERSION",
                        "semantic configuration changed without incrementing declaration_version",
                        field="declaration_version",
                        details={
                            "declaration_version": config.declaration_version,
                            "stored_digest": latest.semantic_digest,
                            "requested_digest": semantic_digest,
                        },
                    )
                return latest
            return self._create_generation(
                config,
                semantic_payload,
                semantic_digest,
                config_json,
            )
        return self._create_generation(
            config,
            semantic_payload,
            semantic_digest,
            config_json,
        )

    def _create_generation(
        self,
        config: PipelineConfig,
        semantic_payload: dict[str, Any],
        semantic_digest: str,
        config_json: dict[str, Any],
    ) -> Generation:
        now = epoch_ms()
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            self.connection.execute(
                "UPDATE aggregates SET is_stale = 1 WHERE is_stale = 0"
            )
            self.connection.execute(
                "UPDATE ledger SET is_stale = 1 WHERE is_stale = 0"
            )
            self.connection.execute("UPDATE batches SET is_stale = 1 WHERE is_stale = 0")
            self.connection.execute(
                "UPDATE generations SET is_current = 0, is_stale = 1"
            )
            cursor = self.connection.execute(
                """
                INSERT INTO generations(
                    declaration_version, semantic_digest, semantic_payload_json,
                    config_json, status, is_current, is_stale, seed, format_version,
                    default_silence_window_ms, batch_size, alert_count,
                    created_at_ms, completed_at_ms
                ) VALUES(?,?,?,?, 'building', 0, 0, ?, ?, ?, ?, 0, ?, NULL)
                """,
                (
                    config.declaration_version,
                    semantic_digest,
                    canonical_json(semantic_payload),
                    canonical_json(config_json),
                    config.seed,
                    config.format_version,
                    config.default_silence_window_ms,
                    config.batch_size,
                    now,
                ),
            )
            generation_id = int(cursor.lastrowid)
            self.connection.execute(
                """
                INSERT INTO progress(generation_id, next_cursor, last_batch_index, updated_at_ms)
                VALUES(?, 0, -1, ?)
                """,
                (generation_id, now),
            )
        row = self.connection.execute(
            "SELECT * FROM generations WHERE generation_id = ?", (generation_id,)
        ).fetchone()
        assert row is not None
        return _as_generation(row)

    def begin_attempt(
        self, generation_id: int, batch_index: int, start_cursor: int, end_cursor: int
    ) -> int:
        now = epoch_ms()
        with self.connection:
            cursor = self.connection.execute(
                """
                INSERT INTO batch_attempts(
                    generation_id, batch_index, start_cursor, end_cursor,
                    status, started_at_ms, finished_at_ms, details_json
                ) VALUES(?,?,?,?, 'unknown', ?, NULL, ?)
                """,
                (
                    generation_id,
                    batch_index,
                    start_cursor,
                    end_cursor,
                    now,
                    canonical_json({"reason": "batch intent committed before execution"}),
                ),
            )
        attempt_id = int(cursor.lastrowid)
        self._attempts[(generation_id, batch_index)] = attempt_id
        return attempt_id

    def commit_batch(self, generation_id: int, execution: BatchExecution) -> None:
        outcome = execution.outcome
        if (
            self.crash_before_commit_on_batch is not None
            and outcome.batch_index == self.crash_before_commit_on_batch
        ):
            os._exit(91)
        now = epoch_ms()
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            self.connection.execute(
                "DELETE FROM batch_aggregates WHERE generation_id=? AND batch_index=?",
                (generation_id, outcome.batch_index),
            )
            self.connection.execute(
                "DELETE FROM batch_ledger WHERE generation_id=? AND batch_index=?",
                (generation_id, outcome.batch_index),
            )
            for aggregate in execution.aggregates:
                content = canonical_json(aggregate.to_public_dict())
                self.connection.execute(
                    """
                    INSERT INTO batch_aggregates(
                        generation_id, batch_index, alert_id, service_id,
                        content_json, content_digest
                    ) VALUES(?,?,?,?,?,?)
                    """,
                    (
                        generation_id,
                        outcome.batch_index,
                        aggregate.alert_id,
                        aggregate.service_id,
                        content,
                        aggregate.content_digest(),
                    ),
                )
                self.connection.execute(
                    """
                    INSERT INTO aggregates(
                        generation_id, alert_id, service_id, fingerprint,
                        first_timestamp_ms, suppressed_count, content_json,
                        content_digest, is_stale
                    ) VALUES(?,?,?,?,?,?,?,?,0)
                    ON CONFLICT(generation_id, alert_id) DO UPDATE SET
                        service_id=excluded.service_id,
                        fingerprint=excluded.fingerprint,
                        first_timestamp_ms=excluded.first_timestamp_ms,
                        suppressed_count=excluded.suppressed_count,
                        content_json=excluded.content_json,
                        content_digest=excluded.content_digest,
                        is_stale=0
                    """,
                    (
                        generation_id,
                        aggregate.alert_id,
                        aggregate.service_id,
                        aggregate.fingerprint,
                        aggregate.first_timestamp_ms,
                        aggregate.suppressed_count,
                        content,
                        aggregate.content_digest(),
                    ),
                )
            for entry in execution.ledger:
                content_dict = entry.to_public_dict()
                content = canonical_json(content_dict)
                self.connection.execute(
                    """
                    INSERT INTO batch_ledger(
                        generation_id, batch_index, alert_id, service_id,
                        content_json, content_digest
                    ) VALUES(?,?,?,?,?,?)
                    """,
                    (
                        generation_id,
                        outcome.batch_index,
                        entry.alert_id,
                        entry.aggregate.service_id,
                        content,
                        entry.content_digest,
                    ),
                )
                self.connection.execute(
                    """
                    INSERT INTO ledger(
                        generation_id, alert_id, service_id, candidate_services_json,
                        paths_json, rule_id, rule_version, graph_version,
                        declared_version, format_version, content_json,
                        content_digest, is_stale
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,0)
                    ON CONFLICT(generation_id, alert_id) DO UPDATE SET
                        service_id=excluded.service_id,
                        candidate_services_json=excluded.candidate_services_json,
                        paths_json=excluded.paths_json,
                        rule_id=excluded.rule_id,
                        rule_version=excluded.rule_version,
                        graph_version=excluded.graph_version,
                        declared_version=excluded.declared_version,
                        format_version=excluded.format_version,
                        content_json=excluded.content_json,
                        content_digest=excluded.content_digest,
                        is_stale=0
                    """,
                    (
                        generation_id,
                        entry.alert_id,
                        entry.aggregate.service_id,
                        canonical_json([item.service_id for item in entry.candidates]),
                        canonical_json([item.to_public_dict() for item in entry.candidates]),
                        entry.rule_id,
                        entry.rule_version,
                        entry.graph_version,
                        entry.declared_version,
                        entry.format_version,
                        content,
                        entry.content_digest,
                    ),
                )
            summary = outcome.summary_dict()
            self.connection.execute(
                """
                INSERT INTO batches(
                    generation_id, batch_index, start_cursor, end_cursor, status,
                    business_code, summary_digest, conclusion_digest,
                    aggregate_count, ledger_count, normalized_count,
                    timeline_json, peak_global, peak_by_service_json,
                    summary_json, committed_at_ms, is_stale
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)
                ON CONFLICT(generation_id, batch_index) DO UPDATE SET
                    start_cursor=excluded.start_cursor,
                    end_cursor=excluded.end_cursor,
                    status=excluded.status,
                    business_code=excluded.business_code,
                    summary_digest=excluded.summary_digest,
                    conclusion_digest=excluded.conclusion_digest,
                    aggregate_count=excluded.aggregate_count,
                    ledger_count=excluded.ledger_count,
                    normalized_count=excluded.normalized_count,
                    timeline_json=excluded.timeline_json,
                    peak_global=excluded.peak_global,
                    peak_by_service_json=excluded.peak_by_service_json,
                    summary_json=excluded.summary_json,
                    committed_at_ms=excluded.committed_at_ms,
                    is_stale=0
                """,
                (
                    generation_id,
                    outcome.batch_index,
                    outcome.start_cursor,
                    outcome.end_cursor,
                    outcome.status,
                    outcome.business_code,
                    outcome.summary_digest,
                    outcome.conclusion_digest,
                    outcome.aggregate_count,
                    outcome.ledger_count,
                    outcome.normalized_count,
                    canonical_json(summary["timeline"]),
                    outcome.schedule.peak_global,
                    canonical_json(outcome.schedule.peak_by_service),
                    canonical_json(summary),
                    now,
                ),
            )
            attempt_id = self._attempts.get((generation_id, outcome.batch_index))
            self.connection.execute(
                """
                UPDATE batch_attempts
                SET status='superseded'
                WHERE generation_id=? AND batch_index=? AND status='unknown'
                  AND attempt_id<>?
                """,
                (generation_id, outcome.batch_index, attempt_id or -1),
            )
            if attempt_id is not None:
                self.connection.execute(
                    """
                    UPDATE batch_attempts
                    SET status='committed', finished_at_ms=?
                    WHERE attempt_id=?
                    """,
                    (now, attempt_id),
                )
            self.connection.execute(
                """
                INSERT INTO progress(generation_id, next_cursor, last_batch_index, updated_at_ms)
                VALUES(?,?,?,?)
                ON CONFLICT(generation_id) DO UPDATE SET
                    next_cursor=excluded.next_cursor,
                    last_batch_index=excluded.last_batch_index,
                    updated_at_ms=excluded.updated_at_ms
                """,
                (
                    generation_id,
                    outcome.end_cursor,
                    outcome.batch_index,
                    now,
                ),
            )

    def unknown_attempts(self, generation_id: int) -> tuple[dict[str, Any], ...]:
        rows = self.connection.execute(
            """
            SELECT * FROM batch_attempts
            WHERE generation_id=? AND status='unknown'
            ORDER BY batch_index, attempt_id
            """,
            (generation_id,),
        ).fetchall()
        return tuple(
            {
                "attempt_id": row["attempt_id"],
                "batch_index": row["batch_index"],
                "start_cursor": row["start_cursor"],
                "end_cursor": row["end_cursor"],
                "started_at_ms": row["started_at_ms"],
            }
            for row in rows
        )

    def progress(self, generation_id: int) -> tuple[int, int]:
        row = self.connection.execute(
            "SELECT * FROM progress WHERE generation_id=?", (generation_id,)
        ).fetchone()
        if row is None:
            return 0, -1
        return int(row["next_cursor"]), int(row["last_batch_index"])

    def committed_batches(self, generation_id: int) -> tuple[dict[str, Any], ...]:
        rows = self.connection.execute(
            """
            SELECT * FROM batches WHERE generation_id=? AND status='committed'
            ORDER BY batch_index
            """,
            (generation_id,),
        ).fetchall()
        return tuple(
            {
                "batch_index": row["batch_index"],
                "start_cursor": row["start_cursor"],
                "end_cursor": row["end_cursor"],
                "business_code": row["business_code"],
                "is_stale": bool(row["is_stale"]),
                "summary_digest": row["summary_digest"],
                "conclusion_digest": row["conclusion_digest"],
                "aggregate_count": row["aggregate_count"],
                "ledger_count": row["ledger_count"],
                "normalized_count": row["normalized_count"],
                "timeline": json.loads(row["timeline_json"]),
                "peak_global": row["peak_global"],
                "peak_by_service": json.loads(row["peak_by_service_json"]),
                "summary": json.loads(row["summary_json"]),
            }
            for row in rows
        )

    def load_batch_aggregates(
        self, generation_id: int, batch_index: int
    ) -> tuple[AggregateAlert, ...]:
        rows = self.connection.execute(
            """
            SELECT content_json FROM batch_aggregates
            WHERE generation_id=? AND batch_index=?
            ORDER BY alert_id
            """,
            (generation_id, batch_index),
        ).fetchall()
        return tuple(aggregate_from_public_dict(json.loads(row["content_json"])) for row in rows)

    def load_batch_ledger(
        self, generation_id: int, batch_index: int
    ) -> dict[str, dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT alert_id, content_json FROM batch_ledger
            WHERE generation_id=? AND batch_index=?
            ORDER BY alert_id
            """,
            (generation_id, batch_index),
        ).fetchall()
        return {row["alert_id"]: json.loads(row["content_json"]) for row in rows}

    def load_state(self, generation_id: int, batch_index: int) -> DedupState:
        aggregates = self.load_batch_aggregates(generation_id, batch_index)
        state: dict[tuple[str, str], list[AggregateAlert]] = {}
        for aggregate in aggregates:
            state.setdefault((aggregate.service_id, aggregate.fingerprint), []).append(
                aggregate
            )
        return {
            key: tuple(
                sorted(value, key=lambda item: (item.first_timestamp_ms, item.alert_id))
            )
            for key, value in sorted(state.items())
        }

    def replay_generation(
        self,
        generation: Generation,
        config: PipelineConfig,
        dataset: LoadedDataset,
    ) -> ReplayCheck:
        requested_digest = config.semantic_digest(dataset)
        if (
            config.declaration_version != generation.declaration_version
            or config.seed != generation.seed
            or requested_digest != generation.semantic_digest
        ):
            raise PipelineError(
                "E_REPLAY",
                "replay configuration does not match the declared ledger generation",
                field="declaration_version/seed/semantic_digest",
                details={
                    "generation_id": generation.generation_id,
                    "declaration_version": generation.declaration_version,
                    "seed": generation.seed,
                    "stored_semantic_digest": generation.semantic_digest,
                    "requested_semantic_digest": requested_digest,
                },
            )
        batches = self.committed_batches(generation.generation_id)
        next_cursor, last_batch_index = self.progress(generation.generation_id)
        expected_cursor = 0
        expected_index = 0
        for batch in batches:
            if (
                batch["batch_index"] != expected_index
                or batch["start_cursor"] != expected_cursor
                or batch["end_cursor"] < batch["start_cursor"]
            ):
                raise PipelineError(
                    "E_REPLAY",
                    "committed batch chain is not contiguous",
                    field="cursor/batch_index",
                    details={
                        "expected_batch_index": expected_index,
                        "actual_batch_index": batch["batch_index"],
                        "expected_start_cursor": expected_cursor,
                        "actual_start_cursor": batch["start_cursor"],
                    },
                )
            expected_cursor = batch["end_cursor"]
            expected_index += 1
        if batches and (
            next_cursor != expected_cursor or last_batch_index != expected_index - 1
        ):
            raise PipelineError(
                "E_REPLAY",
                "progress cursor does not match the last committed batch",
                field="cursor",
                details={
                    "progress_cursor": next_cursor,
                    "batch_cursor": expected_cursor,
                    "progress_batch_index": last_batch_index,
                    "committed_batch_count": expected_index,
                },
            )

        engine = PipelineEngine(
            config,
            dataset,
            mode="serial",
            global_limit=1,
            per_service_limit=1,
        )
        state: DedupState = {}
        alerts_checked = 0
        aggregate_checks = 0
        for batch in batches:
            execution = engine.process_batch(
                batch_index=batch["batch_index"],
                start_cursor=batch["start_cursor"],
                end_cursor=batch["end_cursor"],
                state=state,
            )
            stored_aggregates = {
                item.alert_id: item.to_public_dict()
                for item in self.load_batch_aggregates(
                    generation.generation_id, batch["batch_index"]
                )
            }
            computed_aggregates = {
                item.alert_id: item.to_public_dict() for item in execution.aggregates
            }
            compared_ids = sorted(set(stored_aggregates) | set(computed_aggregates))
            for alert_id in compared_ids:
                before = stored_aggregates.get(alert_id)
                after = computed_aggregates.get(alert_id)
                diff = _first_value_diff(before, after, "aggregate") if before is not None and after is not None else None
                if before is None or after is None or diff is not None:
                    field = (
                        diff[0]
                        if diff is not None
                        else ("aggregate.alert_id" if before is None else "aggregate.presence")
                    )
                    raise PipelineError(
                        "E_REPLAY",
                        "replayed aggregate differs from the committed ledger",
                        field=field,
                        details={
                            "alert_id": alert_id,
                            "batch_index": batch["batch_index"],
                            "stored": before,
                            "recomputed": after,
                            "basis": "declared version, seed, source streams and committed batch state",
                        },
                    )
                aggregate_checks += 1

            stored_ledger = self.load_batch_ledger(
                generation.generation_id, batch["batch_index"]
            )
            computed_ledger = {
                entry.alert_id: entry.to_public_dict() for entry in execution.ledger
            }
            for alert_id in sorted(set(stored_ledger) | set(computed_ledger)):
                before = stored_ledger.get(alert_id)
                after = computed_ledger.get(alert_id)
                diff = _first_value_diff(before, after, "ledger") if before is not None and after is not None else None
                if before is None or after is None or diff is not None:
                    field = (
                        diff[0]
                        if diff is not None
                        else ("ledger.alert_id" if before is None else "ledger.presence")
                    )
                    raise PipelineError(
                        "E_REPLAY",
                        "replayed correlation ledger differs from committed conclusions",
                        field=field,
                        details={
                            "alert_id": alert_id,
                            "batch_index": batch["batch_index"],
                            "stored": before,
                            "recomputed": after,
                            "basis": "declared graph/rule versions and deterministic replay",
                        },
                    )
            if execution.outcome.conclusion_digest != batch["conclusion_digest"]:
                raise PipelineError(
                    "E_REPLAY",
                    "batch conclusion digest differs during replay",
                    field="conclusion_digest",
                    details={
                        "batch_index": batch["batch_index"],
                        "stored": batch["conclusion_digest"],
                        "recomputed": execution.outcome.conclusion_digest,
                        "checked_alerts": aggregate_checks,
                    },
                )
            if execution.outcome.summary_digest != batch["summary_digest"]:
                raise PipelineError(
                    "E_REPLAY",
                    "batch summary digest differs during replay",
                    field="summary_digest",
                    details={
                        "batch_index": batch["batch_index"],
                        "stored": batch["summary_digest"],
                        "recomputed": execution.outcome.summary_digest,
                    },
                )
            state = execution.state
            alerts_checked += batch["end_cursor"] - batch["start_cursor"]

        unknown = self.unknown_attempts(generation.generation_id)
        aggregates = self.load_current_aggregates(generation.generation_id)
        ledger_rows = self.load_current_ledger(generation.generation_id)
        digest = digest_json(
            {
                "aggregates": aggregates,
                "ledger": ledger_rows,
                "batches": [batch["conclusion_digest"] for batch in batches],
            }
        )
        return ReplayCheck(
            generation_id=generation.generation_id,
            batches_checked=len(batches),
            alerts_checked=alerts_checked,
            conclusion_digest=digest,
            unknown_batches=tuple(item["batch_index"] for item in unknown),
            exit_report_available=len(unknown) == 0 and generation.status == "complete",
        )

    def _previous_complete_generation(self, generation: Generation) -> Generation | None:
        row = self.connection.execute(
            """
            SELECT * FROM generations
            WHERE declaration_version < ? AND status='complete'
            ORDER BY declaration_version DESC LIMIT 1
            """,
            (generation.declaration_version,),
        ).fetchone()
        return _as_generation(row) if row is not None else None

    def _contents(self, generation_id: int) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
        aggregate_rows = self.connection.execute(
            "SELECT alert_id, content_json FROM aggregates WHERE generation_id=? ORDER BY alert_id",
            (generation_id,),
        ).fetchall()
        ledger_rows = self.connection.execute(
            "SELECT alert_id, content_json FROM ledger WHERE generation_id=? ORDER BY alert_id",
            (generation_id,),
        ).fetchall()
        return (
            {row["alert_id"]: json.loads(row["content_json"]) for row in aggregate_rows},
            {row["alert_id"]: json.loads(row["content_json"]) for row in ledger_rows},
        )

    def finalize_generation(
        self,
        generation: Generation,
        config: PipelineConfig,
        dataset: LoadedDataset,
    ) -> dict[str, Any]:
        next_cursor, last_batch_index = self.progress(generation.generation_id)
        if next_cursor != dataset.alert_count:
            raise PipelineError(
                "E_STATE",
                "cannot finalize an incomplete generation",
                field="next_cursor",
                details={"next_cursor": next_cursor, "alert_count": dataset.alert_count},
            )
        expected_last = (dataset.alert_count - 1) // config.batch_size if dataset.alert_count else -1
        if last_batch_index != expected_last:
            raise PipelineError(
                "E_STATE",
                "cannot finalize: committed batch index does not cover all alerts",
                field="last_batch_index",
                details={"last_batch_index": last_batch_index, "expected": expected_last},
            )

        previous = self._previous_complete_generation(generation)
        old_config: dict[str, Any] = previous.semantic_payload if previous else {}
        new_config_payload = config.semantic_payload(dataset)
        changed_config = {
            key: {"before": old_config.get(key), "after": new_config_payload.get(key)}
            for key in sorted(set(old_config) | set(new_config_payload))
            if old_config.get(key) != new_config_payload.get(key)
        }
        replacements: list[dict[str, Any]] = []
        if previous is not None:
            old_aggregates, old_ledger = self._contents(previous.generation_id)
            new_aggregates, new_ledger = self._contents(generation.generation_id)
            for alert_id in sorted(set(old_aggregates) | set(new_aggregates) | set(old_ledger) | set(new_ledger)):
                before = {
                    "aggregate": old_aggregates.get(alert_id),
                    "ledger": old_ledger.get(alert_id),
                }
                after = {
                    "aggregate": new_aggregates.get(alert_id),
                    "ledger": new_ledger.get(alert_id),
                }
                if canonical_json(before) == canonical_json(after):
                    continue
                changed_fields = sorted(_changed_paths(before, after))
                replacements.append(
                    {
                        "alert_id": alert_id,
                        "changed_fields": changed_fields,
                        "before": before,
                        "after": after,
                    }
                )
        report = {
            "generation_id": generation.generation_id,
            "declaration_version": generation.declaration_version,
            "previous_generation_id": previous.generation_id if previous else None,
            "previous_declaration_version": previous.declaration_version if previous else None,
            "changed_config_fields": changed_config,
            "affected_count": len(replacements),
            "execution_report_available": True,
            "affected_alerts": replacements,
        }
        now = epoch_ms()
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            self.connection.execute(
                "UPDATE generations SET status='complete', is_current=1, is_stale=0, alert_count=?, completed_at_ms=? WHERE generation_id=?",
                (dataset.alert_count, now, generation.generation_id),
            )
            self.connection.execute(
                "UPDATE generations SET is_current=0, is_stale=1 WHERE generation_id<>?",
                (generation.generation_id,),
            )
            self.connection.execute(
                "UPDATE aggregates SET is_stale=0 WHERE generation_id=?",
                (generation.generation_id,),
            )
            self.connection.execute(
                "UPDATE ledger SET is_stale=0 WHERE generation_id=?",
                (generation.generation_id,),
            )
            self.connection.execute(
                "UPDATE aggregates SET is_stale=1 WHERE generation_id<>?",
                (generation.generation_id,),
            )
            self.connection.execute(
                "UPDATE ledger SET is_stale=1 WHERE generation_id<>?",
                (generation.generation_id,),
            )
            self.connection.execute(
                "UPDATE batches SET is_stale=1 WHERE generation_id<>?",
                (generation.generation_id,),
            )
            self.connection.execute(
                "UPDATE batches SET is_stale=0 WHERE generation_id=?",
                (generation.generation_id,),
            )
            self.connection.execute(
                "DELETE FROM replacements WHERE generation_id=?",
                (generation.generation_id,),
            )
            for replacement in replacements:
                self.connection.execute(
                    """
                    INSERT INTO replacements(
                        generation_id, previous_generation_id, alert_id,
                        changed_fields_json, before_json, after_json, affected
                    ) VALUES(?,?,?,?,?,?,1)
                    """,
                    (
                        generation.generation_id,
                        previous.generation_id if previous else None,
                        replacement["alert_id"],
                        canonical_json(replacement["changed_fields"]),
                        canonical_json(replacement["before"]),
                        canonical_json(replacement["after"]),
                    ),
                )
            self.connection.execute(
                """
                INSERT OR REPLACE INTO replacement_runs(
                    generation_id, previous_generation_id, changed_config_json,
                    affected_count, report_json, created_at_ms
                ) VALUES(?,?,?,?,?,?)
                """,
                (
                    generation.generation_id,
                    previous.generation_id if previous else None,
                    canonical_json(changed_config),
                    len(replacements),
                    canonical_json(report),
                    now,
                ),
            )
        return report

    def load_current_aggregates(
        self, generation_id: int | None = None
    ) -> tuple[dict[str, Any], ...]:
        if generation_id is None:
            rows = self.connection.execute(
                "SELECT content_json FROM current_aggregate_view ORDER BY first_timestamp_ms, service_id, alert_id"
            ).fetchall()
        else:
            rows = self.connection.execute(
                """
                SELECT content_json FROM aggregates
                WHERE generation_id=? AND is_stale=0
                ORDER BY first_timestamp_ms, service_id, alert_id
                """,
                (generation_id,),
            ).fetchall()
        return tuple(json.loads(row["content_json"]) for row in rows)

    def load_current_ledger(
        self, generation_id: int | None = None
    ) -> tuple[dict[str, Any], ...]:
        if generation_id is None:
            rows = self.connection.execute(
                "SELECT content_json FROM current_ledger_view ORDER BY alert_id"
            ).fetchall()
        else:
            rows = self.connection.execute(
                "SELECT content_json FROM ledger WHERE generation_id=? AND is_stale=0 ORDER BY alert_id",
                (generation_id,),
            ).fetchall()
        return tuple(json.loads(row["content_json"]) for row in rows)

    def load_current_aggregate_records(self) -> tuple[dict[str, Any], ...]:
        rows = self.connection.execute(
            """
            SELECT * FROM current_aggregate_view
            ORDER BY first_timestamp_ms, service_id, alert_id
            """
        ).fetchall()
        return tuple(
            {
                "alert_id": row["alert_id"],
                "service_id": row["service_id"],
                "first_timestamp_ms": row["first_timestamp_ms"],
                "suppressed_count": row["suppressed_count"],
                "content_digest": row["content_digest"],
                "content": json.loads(row["content_json"]),
            }
            for row in rows
        )

    def load_current_ledger_records(self) -> tuple[dict[str, Any], ...]:
        rows = self.connection.execute(
            "SELECT * FROM current_ledger_view ORDER BY alert_id"
        ).fetchall()
        return tuple(
            {
                "alert_id": row["alert_id"],
                "service_id": row["service_id"],
                "format_version": row["format_version"],
                "rule_id": row["rule_id"],
                "rule_version": row["rule_version"],
                "graph_version": row["graph_version"],
                "content_digest": row["content_digest"],
                "content": json.loads(row["content_json"]),
            }
            for row in rows
        )

    def stale_counts(self) -> dict[str, int]:
        return {
            "aggregates": int(
                self.connection.execute(
                    "SELECT COUNT(*) FROM aggregates WHERE is_stale=1"
                ).fetchone()[0]
            ),
            "ledger": int(
                self.connection.execute(
                    "SELECT COUNT(*) FROM ledger WHERE is_stale=1"
                ).fetchone()[0]
            ),
            "generations": int(
                self.connection.execute(
                    "SELECT COUNT(*) FROM generations WHERE is_stale=1"
                ).fetchone()[0]
            ),
        }

    def current_generation(self) -> Generation | None:
        row = self.connection.execute(
            "SELECT * FROM generations WHERE is_current=1 AND status='complete' LIMIT 1"
        ).fetchone()
        return _as_generation(row) if row is not None else None

    def latest_generation(self) -> Generation | None:
        row = self.connection.execute(
            "SELECT * FROM generations ORDER BY declaration_version DESC LIMIT 1"
        ).fetchone()
        return _as_generation(row) if row is not None else None

    def generation_for_version(self, declaration_version: int) -> Generation:
        row = self.connection.execute(
            "SELECT * FROM generations WHERE declaration_version=?",
            (declaration_version,),
        ).fetchone()
        if row is None:
            raise PipelineError(
                "E_STATE",
                "no generation exists for the requested declaration_version",
                field="declaration_version",
                details={"declaration_version": declaration_version},
            )
        return _as_generation(row)

    def replacement_report(self, generation_id: int) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT report_json FROM replacement_runs WHERE generation_id=?",
            (generation_id,),
        ).fetchone()
        return json.loads(row["report_json"]) if row is not None else None

    def generation_report(self, generation_id: int) -> dict[str, Any]:
        generation_row = self.connection.execute(
            "SELECT * FROM generations WHERE generation_id=?", (generation_id,)
        ).fetchone()
        if generation_row is None:
            raise PipelineError("E_STATE", "generation not found", details={"generation_id": generation_id})
        generation = _as_generation(generation_row)
        batches = self.committed_batches(generation_id)
        unknown = self.unknown_attempts(generation_id)
        aggregates = self.load_current_aggregates(generation_id)
        ledger = self.load_current_ledger(generation_id)
        return {
            "generation_id": generation.generation_id,
            "declaration_version": generation.declaration_version,
            "status": generation.status,
            "is_current": generation.is_current,
            "is_stale": generation.is_stale,
            "seed": generation.seed,
            "format_version": generation.format_version,
            "semantic_digest": generation.semantic_digest,
            "alert_count": generation.alert_count,
            "aggregate_count": len(aggregates),
            "ledger_count": len(ledger),
            "batches": list(batches),
            "unknown_attempts": list(unknown),
            "exit_report_available": generation.status == "complete" and not unknown,
            "replacement": self.replacement_report(generation_id),
        }
