from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .codec import parse_document
from .derive import changed_fields, resolve_effective
from .engine import JudgmentEngine
from .errors import AppError, ConfigFormatError, ReplayError
from .models import (
    AccessEvent,
    ConfigDocument,
    DECISION_ALLOW,
    DECISION_OVER_QUOTA,
    DECISION_RATE_LIMITED,
    Judgment,
    VolumeUsage,
    judgment_map,
)
from .utils import canonical_json, sha256_text, utc_now


SCHEMA_VERSION = 1


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS config_versions (
    revision INTEGER PRIMARY KEY,
    schema_version INTEGER NOT NULL,
    original_version INTEGER NOT NULL,
    name TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    config_json TEXT NOT NULL,
    source_path TEXT,
    created_at TEXT NOT NULL,
    stale INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS upgrade_checkpoints (
    checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS active_state (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    revision INTEGER NOT NULL,
    generation INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS config_replacements (
    replacement_id INTEGER PRIMARY KEY AUTOINCREMENT,
    from_revision INTEGER,
    to_revision INTEGER NOT NULL,
    affected_volumes_json TEXT NOT NULL,
    changed_fields_json TEXT NOT NULL,
    basis TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS event_batches (
    batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_revision INTEGER NOT NULL,
    start_index INTEGER NOT NULL,
    first_seq INTEGER NOT NULL,
    last_seq INTEGER NOT NULL,
    event_count INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'committed', 'unknown', 'superseded')),
    digest TEXT,
    created_at TEXT NOT NULL,
    committed_at TEXT,
    note TEXT
);

CREATE INDEX IF NOT EXISTS idx_event_batches_revision_status
ON event_batches(config_revision, status, batch_id);

CREATE TABLE IF NOT EXISTS event_inputs (
    config_revision INTEGER NOT NULL,
    seq INTEGER NOT NULL,
    batch_id INTEGER NOT NULL,
    event_json TEXT NOT NULL,
    PRIMARY KEY (config_revision, seq)
);

CREATE TABLE IF NOT EXISTS judgments (
    judgment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_revision INTEGER NOT NULL,
    batch_id INTEGER NOT NULL,
    seq INTEGER NOT NULL,
    timestamp_ms INTEGER NOT NULL,
    volume_id TEXT NOT NULL,
    decision TEXT NOT NULL,
    reason TEXT NOT NULL,
    operation_count INTEGER NOT NULL,
    write_bytes INTEGER NOT NULL,
    cumulative_write_bytes INTEGER NOT NULL,
    tokens_scaled INTEGER NOT NULL,
    tokens_capacity_scaled INTEGER NOT NULL,
    last_refill_ms INTEGER,
    single_error_upper INTEGER NOT NULL,
    aggregate_error_upper INTEGER NOT NULL,
    stale INTEGER NOT NULL DEFAULT 0,
    UNIQUE (config_revision, seq)
);

CREATE INDEX IF NOT EXISTS idx_judgments_revision_stale_seq
ON judgments(config_revision, stale, seq);

CREATE TABLE IF NOT EXISTS current_judgments (
    config_revision INTEGER NOT NULL,
    batch_id INTEGER NOT NULL,
    seq INTEGER NOT NULL,
    timestamp_ms INTEGER NOT NULL,
    volume_id TEXT NOT NULL,
    decision TEXT NOT NULL,
    reason TEXT NOT NULL,
    operation_count INTEGER NOT NULL,
    write_bytes INTEGER NOT NULL,
    cumulative_write_bytes INTEGER NOT NULL,
    tokens_scaled INTEGER NOT NULL,
    tokens_capacity_scaled INTEGER NOT NULL,
    last_refill_ms INTEGER,
    single_error_upper INTEGER NOT NULL,
    aggregate_error_upper INTEGER NOT NULL,
    PRIMARY KEY (config_revision, seq)
);

CREATE TABLE IF NOT EXISTS current_usage (
    config_revision INTEGER NOT NULL,
    volume_id TEXT NOT NULL,
    cumulative_write_bytes INTEGER NOT NULL,
    tokens_scaled INTEGER NOT NULL,
    last_refill_ms INTEGER,
    allowed_events INTEGER NOT NULL,
    rate_limited_events INTEGER NOT NULL,
    over_quota_events INTEGER NOT NULL,
    allowed_write_bytes INTEGER NOT NULL,
    allowed_operations INTEGER NOT NULL,
    PRIMARY KEY (config_revision, volume_id)
);

CREATE TABLE IF NOT EXISTS progress_cursor (
    config_revision INTEGER PRIMARY KEY,
    next_index INTEGER NOT NULL,
    next_seq INTEGER,
    last_batch_id INTEGER,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS structured_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL,
    message TEXT NOT NULL,
    operation TEXT,
    volume_id TEXT,
    field_path TEXT,
    details_json TEXT,
    created_at TEXT NOT NULL
);
"""


@dataclass
class IngestReport:
    config_revision: int
    batches_committed: int
    judgments_written: int
    next_index: int
    next_seq: int | None
    decisions: dict[str, int]
    error_bound_single: int = 0
    error_bound_aggregate: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "config_revision": self.config_revision,
            "batches_committed": self.batches_committed,
            "judgments_written": self.judgments_written,
            "next_index": self.next_index,
            "next_seq": self.next_seq,
            "decisions": self.decisions,
            "error_bound_single": self.error_bound_single,
            "error_bound_aggregate": self.error_bound_aggregate,
        }


@dataclass
class RecoveryReport:
    config_revision: int | None
    unknown_batches: list[dict[str, Any]] = field(default_factory=list)
    committed_batches: int = 0
    next_index: int = 0
    exit_report_available: bool = True
    rerun_required: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "config_revision": self.config_revision,
            "unknown_batches": self.unknown_batches,
            "committed_batches": self.committed_batches,
            "next_index": self.next_index,
            "exit_report_available": self.exit_report_available,
            "rerun_required": self.rerun_required,
        }


@dataclass
class ReplayReport:
    config_revision: int
    events_replayed: int
    judgments_compared: int
    mismatches: list[dict[str, Any]] = field(default_factory=list)
    errors: list[AppError] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, Any]:
        return {
            "config_revision": self.config_revision,
            "events_replayed": self.events_replayed,
            "judgments_compared": self.judgments_compared,
            "mismatches": self.mismatches,
            "errors": [error.as_dict() for error in self.errors],
            "ok": self.ok,
        }


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=10.0)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA synchronous = FULL")
        self.connection.execute("PRAGMA busy_timeout = 5000")
        if self.path != ":memory:":
            try:
                self.connection.execute("PRAGMA journal_mode = WAL")
            except sqlite3.DatabaseError:
                pass
        self._initialize()

    def _initialize(self) -> None:
        self.connection.executescript(SCHEMA_SQL)
        row = self.connection.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        if row is None:
            self.connection.execute(
                "INSERT INTO meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
        else:
            existing = int(row["value"])
            if existing != SCHEMA_VERSION:
                raise ConfigFormatError(
                    f"unsupported ledger schema version {existing}",
                    field_path=str(self.path),
                )
        policy = {
            "decision_policy": "capacity_before_rate_limit",
            "capacity_boundary": "projected_write_bytes <= capacity_bytes",
            "token_boundary": "cost_scaled <= token_balance_scaled",
            "token_scale": "1000",
            "timestamp_unit": "integer_milliseconds",
            "truncation_direction": "floor_to_1/1000_unit",
            "single_error_upper": "0",
            "aggregate_error_upper": "0",
        }
        for key, value in policy.items():
            self.connection.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES(?, ?)", (key, value)
            )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _begin(self) -> None:
        self.connection.execute("BEGIN IMMEDIATE")

    def _rollback(self) -> None:
        try:
            self.connection.rollback()
        except sqlite3.Error:
            pass

    def record_structured_event(
        self,
        code: str,
        message: str,
        *,
        operation: str | None = None,
        volume_id: str | None = None,
        field_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO structured_events(
                code, message, operation, volume_id, field_path, details_json, created_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (
                code,
                message,
                operation,
                volume_id,
                field_path,
                canonical_json(details) if details else None,
                utc_now(),
            ),
        )
        self.connection.commit()

    def record_error(self, error: AppError, *, operation: str) -> None:
        details = error.as_dict().get("details")
        self.record_structured_event(
            error.code,
            error.message,
            operation=operation,
            volume_id=error.volume_id,
            field_path=error.field_path,
            details=details,
        )

    def record_upgrade_checkpoint(self, payload: dict[str, Any]) -> None:
        text = canonical_json(payload)
        fingerprint = sha256_text(text)
        latest = self.connection.execute(
            "SELECT fingerprint FROM upgrade_checkpoints ORDER BY checkpoint_id DESC LIMIT 1"
        ).fetchone()
        if latest is not None and latest["fingerprint"] == fingerprint:
            return
        self.connection.execute(
            """
            INSERT INTO upgrade_checkpoints(fingerprint, payload_json, recorded_at)
            VALUES(?, ?, ?)
            """,
            (fingerprint, text, utc_now()),
        )
        self.connection.commit()

    def register_config(self, document: ConfigDocument, *, source: str = "workspace") -> bool:
        """Persist an immutable snapshot and activate it if newer.

        Returns True when the active view was wholly replaced.
        """

        # Validate derivation before persisting: a document that cannot produce
        # effective quotas must not become active.
        resolve_effective(document)
        config_text = canonical_json(document.to_json())
        fingerprint = document.fingerprint
        existing = self.connection.execute(
            "SELECT fingerprint FROM config_versions WHERE revision = ?",
            (document.revision,),
        ).fetchone()
        if existing is not None and existing["fingerprint"] != fingerprint:
            raise ConfigFormatError(
                "configuration revision was reused with different content",
                field_path="$.revision",
                details={"revision": document.revision},
            )

        if existing is None:
            self.connection.execute(
                """
                INSERT INTO config_versions(
                    revision, schema_version, original_version, name, fingerprint,
                    config_json, source_path, created_at, stale
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    document.revision,
                    document.schema_version,
                    document.original_version,
                    document.name,
                    fingerprint,
                    config_text,
                    document.source_path or source,
                    utc_now(),
                ),
            )
            self.connection.commit()

        active_row = self.connection.execute(
            "SELECT revision FROM active_state WHERE singleton = 1"
        ).fetchone()
        if active_row is None:
            self._activate_config(None, document)
            return True
        active_revision = int(active_row["revision"])
        if document.revision < active_revision:
            raise ConfigFormatError(
                "configuration revision must increase monotonically",
                field_path="$.revision",
                details={"revision": document.revision, "active_revision": active_revision},
            )
        if document.revision == active_revision:
            return False
        self._activate_config(self.load_config(active_revision), document)
        return True

    def _activate_config(self, old: ConfigDocument | None, new: ConfigDocument) -> None:
        changes = changed_fields(old, new)
        affected_volumes = sorted(change["volume_id"] for change in changes)
        old_revision = old.revision if old is not None else None
        self._begin()
        try:
            self.connection.execute("UPDATE judgments SET stale = 1")
            self.connection.execute("UPDATE config_versions SET stale = 1")
            self.connection.execute(
                "UPDATE config_versions SET stale = 0 WHERE revision = ?", (new.revision,)
            )
            self.connection.execute("DELETE FROM current_judgments")
            self.connection.execute("DELETE FROM current_usage")
            self.connection.execute("DELETE FROM progress_cursor")
            active = self.connection.execute(
                "SELECT generation FROM active_state WHERE singleton = 1"
            ).fetchone()
            generation = (int(active["generation"]) + 1) if active is not None else 1
            self.connection.execute(
                """
                INSERT INTO config_replacements(
                    from_revision, to_revision, affected_volumes_json,
                    changed_fields_json, basis, created_at
                ) VALUES(?, ?, ?, ?, ?, ?)
                """,
                (
                    old_revision,
                    new.revision,
                    canonical_json(affected_volumes),
                    canonical_json(changes),
                    "whole current view replacement; previous revisions are stale",
                    utc_now(),
                ),
            )
            self.connection.execute(
                """
                INSERT INTO active_state(singleton, revision, generation, updated_at)
                VALUES(1, ?, ?, ?)
                ON CONFLICT(singleton) DO UPDATE SET
                    revision = excluded.revision,
                    generation = excluded.generation,
                    updated_at = excluded.updated_at
                """,
                (new.revision, generation, utc_now()),
            )
            self.connection.execute(
                """
                INSERT INTO progress_cursor(config_revision, next_index, next_seq, last_batch_id, updated_at)
                VALUES(?, 0, NULL, NULL, ?)
                """,
                (new.revision, utc_now()),
            )
            self.connection.commit()
        except BaseException:
            self._rollback()
            raise

    def active_revision(self) -> int | None:
        row = self.connection.execute(
            "SELECT revision FROM active_state WHERE singleton = 1"
        ).fetchone()
        return int(row["revision"]) if row is not None else None

    def load_config(self, revision: int) -> ConfigDocument:
        row = self.connection.execute(
            "SELECT config_json, source_path FROM config_versions WHERE revision = ?",
            (revision,),
        ).fetchone()
        if row is None:
            raise ConfigFormatError(
                f"configuration revision {revision} is not in the ledger",
                field_path="$.revision",
            )
        raw = json.loads(row["config_json"])
        return parse_document(raw, source_path=row["source_path"])

    def list_config_versions(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT revision, schema_version, original_version, name, fingerprint,
                   source_path, created_at, stale
            FROM config_versions ORDER BY revision
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def list_replacements(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT replacement_id, from_revision, to_revision,
                   affected_volumes_json, changed_fields_json, basis, created_at
            FROM config_replacements ORDER BY replacement_id
            """
        ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["affected_volumes"] = json.loads(item.pop("affected_volumes_json"))
            item["changed_fields"] = json.loads(item.pop("changed_fields_json"))
            result.append(item)
        return result

    def _recover_pending_locked(self, revision: int) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT batch_id, start_index, first_seq, last_seq, event_count, created_at
            FROM event_batches
            WHERE config_revision = ? AND status = 'pending'
            ORDER BY batch_id
            """,
            (revision,),
        ).fetchall()
        unknown: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["status"] = "unknown"
            unknown.append(item)
            self.connection.execute(
                "UPDATE event_batches SET status = 'unknown', note = ? WHERE batch_id = ?",
                (
                    "process stopped before commit; rerun is required and no exit-time report is available",
                    row["batch_id"],
                ),
            )
        return unknown

    def recover_incomplete(self) -> RecoveryReport:
        revision = self.active_revision()
        if revision is None:
            return RecoveryReport(config_revision=None)
        self._begin()
        try:
            unknown = self._recover_pending_locked(revision)
            committed = self.connection.execute(
                "SELECT COUNT(*) AS count FROM event_batches WHERE config_revision = ? AND status = 'committed'",
                (revision,),
            ).fetchone()["count"]
            progress = self.connection.execute(
                "SELECT next_index FROM progress_cursor WHERE config_revision = ?",
                (revision,),
            ).fetchone()
            self.connection.commit()
        except BaseException:
            self._rollback()
            raise
        return RecoveryReport(
            config_revision=revision,
            unknown_batches=unknown,
            committed_batches=int(committed),
            next_index=int(progress["next_index"]) if progress is not None else 0,
            exit_report_available=not unknown,
            rerun_required=bool(unknown),
        )

    def ingest_events(
        self,
        document: ConfigDocument,
        events: Sequence[AccessEvent],
        *,
        batch_size: int = 3,
        crash_hook: Callable[[str, int], None] | None = None,
    ) -> IngestReport:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.active_revision() != document.revision:
            raise ConfigFormatError(
                "document is not the active ledger configuration",
                field_path="$.revision",
                details={
                    "document_revision": document.revision,
                    "active_revision": self.active_revision(),
                },
            )
        effective = resolve_effective(document)
        self._validate_event_sequence(events)
        for event in events:
            if event.volume_id not in effective:
                raise ConfigFormatError(
                    f"event references unknown volume {event.volume_id!r}",
                    volume_id=event.volume_id,
                    field_path=f"event:{event.seq}.volume_id",
                    details={"seq": event.seq},
                )
        self.recover_incomplete()

        progress = self.connection.execute(
            "SELECT next_index, next_seq, last_batch_id FROM progress_cursor WHERE config_revision = ?",
            (document.revision,),
        ).fetchone()
        start_index = int(progress["next_index"]) if progress is not None else 0
        if start_index > len(events):
            raise ReplayError(
                "progress cursor is beyond the supplied event stream",
                details={"next_index": start_index, "event_count": len(events)},
            )
        if start_index > 0 and progress is None:
            raise ReplayError(
                "progress exists without a committed ledger cursor",
                details={"start_index": start_index},
            )

        usage = self.load_current_usage(document.revision)
        if start_index > 0 and set(usage) != set(effective):
            raise ReplayError(
                "committed progress has no complete persisted usage snapshot",
                details={
                    "next_index": start_index,
                    "effective_volumes": sorted(effective),
                    "persisted_usage_volumes": sorted(usage),
                },
            )
        engine = JudgmentEngine(effective, usage)
        committed_now = 0
        written = 0
        decision_counts = {
            DECISION_ALLOW: 0,
            DECISION_RATE_LIMITED: 0,
            DECISION_OVER_QUOTA: 0,
        }
        end_index = start_index
        last_seq: int | None = int(progress["next_seq"]) if progress and progress["next_seq"] is not None else None

        while end_index < len(events):
            batch = list(events[end_index : end_index + batch_size])
            if not batch:
                break
            first_seq = batch[0].seq
            last_batch_seq = batch[-1].seq
            digest = sha256_text(canonical_json([event.to_json() for event in batch]))
            cursor = self.connection.execute(
                """
                INSERT INTO event_batches(
                    config_revision, start_index, first_seq, last_seq, event_count,
                    status, digest, created_at
                ) VALUES(?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    document.revision,
                    end_index,
                    first_seq,
                    last_batch_seq,
                    len(batch),
                    digest,
                    utc_now(),
                ),
            )
            self.connection.commit()
            batch_id = int(cursor.lastrowid)
            if crash_hook is not None:
                crash_hook("after_pending", batch_id)

            try:
                judgments = [engine.evaluate(event) for event in batch]
            except BaseException as exc:
                self.connection.execute(
                    "UPDATE event_batches SET status = 'unknown', note = ? WHERE batch_id = ?",
                    (f"evaluation interrupted: {exc}", batch_id),
                )
                self.connection.commit()
                raise
            if crash_hook is not None:
                crash_hook("after_evaluate", batch_id)

            new_end_index = end_index + len(batch)
            snapshot = engine.snapshot_usage()
            try:
                self._begin()
                for event, judgment in zip(batch, judgments):
                    self.connection.execute(
                        """
                        INSERT INTO event_inputs(config_revision, seq, batch_id, event_json)
                        VALUES(?, ?, ?, ?)
                        """,
                        (
                            document.revision,
                            event.seq,
                            batch_id,
                            canonical_json(event.to_json()),
                        ),
                    )
                    self._insert_judgment(document.revision, batch_id, judgment, table="judgments")
                    self._insert_judgment(
                        document.revision, batch_id, judgment, table="current_judgments"
                    )

                # The usage current view is replaced wholesale on every commit;
                # it is never patched by adding deltas to old rows.
                self.connection.execute("DELETE FROM current_usage")
                for volume_id in sorted(snapshot):
                    using = snapshot[volume_id]
                    self.connection.execute(
                        """
                        INSERT INTO current_usage(
                            config_revision, volume_id, cumulative_write_bytes,
                            tokens_scaled, last_refill_ms, allowed_events,
                            rate_limited_events, over_quota_events,
                            allowed_write_bytes, allowed_operations
                        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            document.revision,
                            using.volume_id,
                            using.cumulative_write_bytes,
                            using.tokens_scaled,
                            using.last_refill_ms,
                            using.allowed_events,
                            using.rate_limited_events,
                            using.over_quota_events,
                            using.allowed_write_bytes,
                            using.allowed_operations,
                        ),
                    )
                self.connection.execute(
                    """
                    INSERT INTO progress_cursor(
                        config_revision, next_index, next_seq, last_batch_id, updated_at
                    ) VALUES(?, ?, ?, ?, ?)
                    ON CONFLICT(config_revision) DO UPDATE SET
                        next_index = excluded.next_index,
                        next_seq = excluded.next_seq,
                        last_batch_id = excluded.last_batch_id,
                        updated_at = excluded.updated_at
                    """,
                    (
                        document.revision,
                        new_end_index,
                        last_batch_seq,
                        batch_id,
                        utc_now(),
                    ),
                )
                self.connection.execute(
                    """
                    UPDATE event_batches
                    SET status = 'committed', committed_at = ?, note = NULL
                    WHERE batch_id = ?
                    """,
                    (utc_now(), batch_id),
                )
                if crash_hook is not None:
                    crash_hook("before_commit", batch_id)
                self.connection.commit()
            except BaseException:
                self._rollback()
                raise

            committed_now += 1
            written += len(judgments)
            for judgment in judgments:
                decision_counts[judgment.decision] += 1
            end_index = new_end_index
            last_seq = last_batch_seq

        return IngestReport(
            config_revision=document.revision,
            batches_committed=committed_now,
            judgments_written=written,
            next_index=end_index,
            next_seq=last_seq,
            decisions=decision_counts,
        )

    def _validate_event_sequence(self, events: Sequence[AccessEvent]) -> None:
        previous_seq: int | None = None
        previous_ts: int | None = None
        for event in events:
            if previous_seq is not None and event.seq <= previous_seq:
                from .errors import OrderError

                raise OrderError(
                    "event sequence must be strictly increasing",
                    volume_id=event.volume_id,
                    field_path=f"event:{event.seq}.seq",
                    details={"previous_seq": previous_seq, "seq": event.seq},
                )
            if previous_ts is not None and event.timestamp_ms < previous_ts:
                from .errors import OrderError

                raise OrderError(
                    "event timestamp moved backwards",
                    volume_id=event.volume_id,
                    field_path=f"event:{event.seq}.timestamp_ms",
                    details={"previous_timestamp_ms": previous_ts, "timestamp_ms": event.timestamp_ms},
                )
            previous_seq = event.seq
            previous_ts = event.timestamp_ms

    def _insert_judgment(
        self,
        revision: int,
        batch_id: int,
        judgment: Judgment,
        *,
        table: str,
    ) -> None:
        if table not in {"judgments", "current_judgments"}:
            raise ValueError(table)
        stale_column = ", stale" if table == "judgments" else ""
        stale_value = ", 0" if table == "judgments" else ""
        self.connection.execute(
            f"""
            INSERT INTO {table}(
                config_revision, batch_id, seq, timestamp_ms, volume_id, decision,
                reason, operation_count, write_bytes, cumulative_write_bytes,
                tokens_scaled, tokens_capacity_scaled, last_refill_ms,
                single_error_upper, aggregate_error_upper{stale_column}
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?{stale_value})
            """,
            (
                revision,
                batch_id,
                judgment.seq,
                judgment.timestamp_ms,
                judgment.volume_id,
                judgment.decision,
                judgment.reason,
                judgment.operation_count,
                judgment.write_bytes,
                judgment.cumulative_write_bytes,
                judgment.tokens_scaled,
                judgment.tokens_capacity_scaled,
                judgment.last_refill_ms,
                judgment.single_error_upper,
                judgment.aggregate_error_upper,
            ),
        )

    def load_current_judgments(self, revision: int | None = None) -> list[Judgment]:
        revision = self.active_revision() if revision is None else revision
        if revision is None:
            return []
        rows = self.connection.execute(
            """
            SELECT seq, timestamp_ms, volume_id, decision, reason, operation_count,
                   write_bytes, cumulative_write_bytes, tokens_scaled,
                   tokens_capacity_scaled, last_refill_ms, single_error_upper,
                   aggregate_error_upper
            FROM current_judgments WHERE config_revision = ? ORDER BY seq
            """,
            (revision,),
        ).fetchall()
        return [self._row_to_judgment(row) for row in rows]

    def load_all_judgments(self, revision: int | None = None) -> list[dict[str, Any]]:
        if revision is None:
            revision = self.active_revision()
        if revision is None:
            return []
        rows = self.connection.execute(
            "SELECT * FROM judgments WHERE config_revision = ? ORDER BY seq",
            (revision,),
        ).fetchall()
        return [dict(row) for row in rows]

    def _row_to_judgment(self, row: sqlite3.Row) -> Judgment:
        return Judgment(
            seq=int(row["seq"]),
            timestamp_ms=int(row["timestamp_ms"]),
            volume_id=row["volume_id"],
            decision=row["decision"],
            reason=row["reason"],
            operation_count=int(row["operation_count"]),
            write_bytes=int(row["write_bytes"]),
            cumulative_write_bytes=int(row["cumulative_write_bytes"]),
            tokens_scaled=int(row["tokens_scaled"]),
            tokens_capacity_scaled=int(row["tokens_capacity_scaled"]),
            last_refill_ms=(
                None if row["last_refill_ms"] is None else int(row["last_refill_ms"])
            ),
            single_error_upper=int(row["single_error_upper"]),
            aggregate_error_upper=int(row["aggregate_error_upper"]),
        )

    def load_current_usage(self, revision: int | None = None) -> dict[str, VolumeUsage]:
        revision = self.active_revision() if revision is None else revision
        if revision is None:
            return {}
        rows = self.connection.execute(
            "SELECT * FROM current_usage WHERE config_revision = ? ORDER BY volume_id",
            (revision,),
        ).fetchall()
        return {
            row["volume_id"]: VolumeUsage(
                volume_id=row["volume_id"],
                cumulative_write_bytes=int(row["cumulative_write_bytes"]),
                tokens_scaled=int(row["tokens_scaled"]),
                last_refill_ms=(
                    None if row["last_refill_ms"] is None else int(row["last_refill_ms"])
                ),
                allowed_events=int(row["allowed_events"]),
                rate_limited_events=int(row["rate_limited_events"]),
                over_quota_events=int(row["over_quota_events"]),
                allowed_write_bytes=int(row["allowed_write_bytes"]),
                allowed_operations=int(row["allowed_operations"]),
            )
            for row in rows
        }

    def load_event_inputs(self, revision: int | None = None) -> list[AccessEvent]:
        revision = self.active_revision() if revision is None else revision
        if revision is None:
            return []
        rows = self.connection.execute(
            "SELECT event_json FROM event_inputs WHERE config_revision = ? ORDER BY seq",
            (revision,),
        ).fetchall()
        events: list[AccessEvent] = []
        for row in rows:
            raw = json.loads(row["event_json"])
            from .events import parse_event

            events.append(parse_event(raw, line_number=len(events) + 1, source="ledger"))
        return events

    def current_progress(self) -> dict[str, Any] | None:
        revision = self.active_revision()
        if revision is None:
            return None
        row = self.connection.execute(
            "SELECT * FROM progress_cursor WHERE config_revision = ?", (revision,)
        ).fetchone()
        return dict(row) if row is not None else None

    def verify_replay(self) -> ReplayReport:
        revision = self.active_revision()
        if revision is None:
            return ReplayReport(
                config_revision=0,
                events_replayed=0,
                judgments_compared=0,
                errors=[ReplayError("ledger has no active configuration", details={})],
            )
        document = self.load_config(revision)
        effective = resolve_effective(document)
        events = self.load_event_inputs(revision)
        saved = judgment_map(self.load_current_judgments(revision))
        engine = JudgmentEngine(effective)
        mismatches: list[dict[str, Any]] = []
        errors: list[AppError] = []
        compared = 0
        for event in events:
            replayed = engine.evaluate(event)
            current = saved.get(event.seq)
            if current is None:
                mismatch = {
                    "volume_id": event.volume_id,
                    "seq": event.seq,
                    "difference_fields": ["missing_current_judgment"],
                }
                mismatches.append(mismatch)
                if not errors:
                    errors.append(
                        ReplayError(
                            "saved judgment cannot be reproduced",
                            volume_id=event.volume_id,
                            field_path=f"event:{event.seq}",
                            details=mismatch,
                        )
                    )
                continue
            diff_fields = replayed.difference_fields(current)
            compared += 1
            if diff_fields:
                mismatch = {
                    "volume_id": event.volume_id,
                    "seq": event.seq,
                    "difference_fields": list(diff_fields),
                    "replayed": replayed.comparable(),
                    "saved": current.comparable(),
                }
                mismatches.append(mismatch)
                if not errors:
                    errors.append(
                        ReplayError(
                            "saved judgment cannot be reproduced",
                            volume_id=event.volume_id,
                            field_path=f"event:{event.seq}",
                            details=mismatch,
                        )
                    )
        saved_seqs = set(saved)
        replayed_seqs = {event.seq for event in events}
        for seq in sorted(saved_seqs - replayed_seqs):
            current = saved[seq]
            mismatch = {
                "volume_id": current.volume_id,
                "seq": seq,
                "difference_fields": ["missing_event_input"],
            }
            mismatches.append(mismatch)
            if not errors:
                errors.append(
                    ReplayError(
                        "saved judgment has no reproducible event input",
                        volume_id=current.volume_id,
                        field_path=f"event:{seq}",
                        details=mismatch,
                    )
                )
        progress = self.current_progress()
        expected_next_index = len(events)
        if progress is not None and int(progress["next_index"]) != expected_next_index:
            mismatch = {
                "volume_id": None,
                "seq": progress.get("next_seq"),
                "difference_fields": ["progress.next_index"],
                "saved_next_index": int(progress["next_index"]),
                "replayed_next_index": expected_next_index,
            }
            mismatches.append(mismatch)
            if not errors:
                errors.append(
                    ReplayError(
                        "progress cursor does not match persisted event inputs",
                        field_path="progress_cursor.next_index",
                        details=mismatch,
                    )
                )

        saved_usage = self.load_current_usage(revision)
        if events or saved_usage:
            replayed_usage = engine.snapshot_usage()
            for volume_id in sorted(set(saved_usage) | set(effective)):
                saved = saved_usage.get(volume_id)
                replayed = replayed_usage.get(volume_id)
                if saved is None or replayed is None:
                    difference_fields = [
                        "missing_saved_usage" if saved is None else "missing_replayed_usage"
                    ]
                    saved_payload = saved.to_json() if saved is not None else None
                    replayed_payload = replayed.to_json() if replayed is not None else None
                else:
                    saved_payload = saved.to_json()
                    replayed_payload = replayed.to_json()
                    difference_fields = sorted(
                        key
                        for key in saved_payload
                        if key != "volume_id" and saved_payload[key] != replayed_payload[key]
                    )
                if not difference_fields:
                    continue
                mismatch = {
                    "volume_id": volume_id,
                    "seq": progress.get("next_seq") if progress is not None else None,
                    "difference_fields": difference_fields,
                    "saved_usage": saved_payload,
                    "replayed_usage": replayed_payload,
                }
                mismatches.append(mismatch)
                if not errors:
                    errors.append(
                        ReplayError(
                            "persisted usage cannot be reproduced",
                            volume_id=volume_id,
                            field_path=f"volume:{volume_id}.usage",
                            details=mismatch,
                        )
                    )
        return ReplayReport(
            config_revision=revision,
            events_replayed=len(events),
            judgments_compared=compared,
            mismatches=mismatches,
            errors=errors,
        )

    def current_view(self) -> dict[str, Any]:
        revision = self.active_revision()
        if revision is None:
            return {
                "active_revision": None,
                "config_versions": self.list_config_versions(),
                "replacements": self.list_replacements(),
                "judgments": [],
                "usage": {},
                "progress": None,
                "structured_events": [],
            }
        document = self.load_config(revision)
        effective = resolve_effective(document)
        judgments = self.load_current_judgments(revision)
        usage = self.load_current_usage(revision)
        event_rows = self.connection.execute(
            "SELECT event_id, code, message, operation, volume_id, field_path, details_json, created_at "
            "FROM structured_events ORDER BY event_id"
        ).fetchall()
        events = []
        for row in event_rows:
            item = dict(row)
            item["details"] = json.loads(item.pop("details_json")) if item["details_json"] else None
            events.append(item)
        policy_rows = self.connection.execute(
            "SELECT key, value FROM meta ORDER BY key"
        ).fetchall()
        return {
            "active_revision": revision,
            "config": document.to_json(),
            "effective_volumes": {
                volume_id: effective[volume_id].to_json()
                for volume_id in sorted(effective)
            },
            "config_versions": self.list_config_versions(),
            "replacements": self.list_replacements(),
            "judgments": [judgment.to_json() for judgment in judgments],
            "usage": {volume_id: usage[volume_id].to_json() for volume_id in sorted(usage)},
            "progress": self.current_progress(),
            "structured_events": events,
            "error_bounds": {"single": 0, "aggregate": 0},
            "calculation_policy": {row["key"]: row["value"] for row in policy_rows},
        }
