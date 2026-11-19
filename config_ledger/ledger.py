"""SQLite ledger with transactional rounds, stale isolation and replay state."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .errors import RecoveryInfo, SnapshotError
from .latency import LatencySample, aggregate_stats, build_fetch_aggregates
from .merge import MergeResult, canonical_json
from .observer import FetchBatch
from .spec import Spec
from .structured_log import StructuredLogger, utc_now_text


FORMAT_VERSION = 1
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ActivationResult:
    previous_version: int
    current_version: int
    previous_round_id: str | None
    changed: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "previous_version": self.previous_version,
            "current_version": self.current_version,
            "previous_round_id": self.previous_round_id,
            "changed": self.changed,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class CommitResult:
    round_id: str
    content_digest: str
    parameter_version: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "round_id": self.round_id,
            "content_digest": self.content_digest,
            "parameter_version": self.parameter_version,
        }


class Ledger:
    """One local SQLite connection guarded by a re-entrant lock."""

    def __init__(self, path: str | os.PathLike[str], *, logger: StructuredLogger | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logger or StructuredLogger()
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(
            str(self.path), timeout=15.0, check_same_thread=False, isolation_level=None
        )
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self._create_schema()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield self.conn
        except Exception:
            self.conn.execute("ROLLBACK")
            raise
        else:
            self.conn.execute("COMMIT")

    def _create_schema(self) -> None:
        with self._lock:
            version = int(self.conn.execute("PRAGMA user_version").fetchone()[0])
            if version not in (0, SCHEMA_VERSION):
                raise SnapshotError(
                    f"unsupported SQLite user_version {version}; expected {SCHEMA_VERSION}"
                )
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ledger_info (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    format_version INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS parameter_versions (
                    version INTEGER PRIMARY KEY,
                    fingerprint TEXT NOT NULL,
                    spec_json TEXT NOT NULL,
                    activated_at TEXT NOT NULL,
                    is_current INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS attempts (
                    attempt_id TEXT PRIMARY KEY,
                    parameter_version INTEGER NOT NULL,
                    window_id TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    finished_at TEXT,
                    note TEXT
                );
                CREATE TABLE IF NOT EXISTS rounds (
                    round_id TEXT PRIMARY KEY,
                    attempt_id TEXT NOT NULL UNIQUE,
                    parameter_version INTEGER NOT NULL,
                    parameter_fingerprint TEXT NOT NULL,
                    window_id TEXT NOT NULL,
                    is_stale INTEGER NOT NULL DEFAULT 0,
                    stale_reason TEXT,
                    committed_at TEXT NOT NULL,
                    content_digest TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS observation_requests (
                    round_id TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    status TEXT NOT NULL,
                    observations_json TEXT NOT NULL,
                    error_code TEXT,
                    error_reason TEXT,
                    diagnostic_code TEXT,
                    http_status INTEGER,
                    request_start_ms INTEGER NOT NULL,
                    request_end_ms INTEGER NOT NULL,
                    request_latency_us INTEGER NOT NULL,
                    queue_wait_start_ms INTEGER NOT NULL,
                    queue_wait_end_ms INTEGER NOT NULL,
                    queued INTEGER NOT NULL,
                    PRIMARY KEY (round_id, node_id, key),
                    FOREIGN KEY (round_id) REFERENCES rounds(round_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS conclusions (
                    round_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    status TEXT NOT NULL,
                    divergent INTEGER NOT NULL,
                    quorum_met INTEGER NOT NULL,
                    effective_value_json TEXT,
                    decision_json TEXT NOT NULL,
                    PRIMARY KEY (round_id, key),
                    FOREIGN KEY (round_id) REFERENCES rounds(round_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS node_comparisons (
                    round_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    request_status TEXT NOT NULL,
                    http_status INTEGER,
                    error_reason TEXT,
                    value_json TEXT,
                    observation_seq INTEGER,
                    timestamp_ms INTEGER,
                    duplicate_count INTEGER NOT NULL,
                    adopted INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    latest_candidate INTEGER NOT NULL,
                    PRIMARY KEY (round_id, key, node_id),
                    FOREIGN KEY (round_id) REFERENCES rounds(round_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS latency_samples (
                    round_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    label TEXT NOT NULL,
                    duration_us INTEGER NOT NULL,
                    monotonic_start_ms INTEGER NOT NULL,
                    monotonic_end_ms INTEGER NOT NULL,
                    over_budget INTEGER NOT NULL DEFAULT 0,
                    overrun_us INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (round_id, kind, node_id, label),
                    FOREIGN KEY (round_id) REFERENCES rounds(round_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS latency_aggregates (
                    round_id TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    node_id TEXT,
                    sample_count INTEGER NOT NULL,
                    mean_us INTEGER NOT NULL,
                    p95_us INTEGER NOT NULL,
                    max_us INTEGER NOT NULL,
                    quantization_error_bound_fixed6_ms INTEGER NOT NULL,
                    stats_json TEXT NOT NULL,
                    FOREIGN KEY (round_id) REFERENCES rounds(round_id) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS ui_latency_samples (
                    sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    parameter_version INTEGER NOT NULL,
                    round_id TEXT,
                    duration_us INTEGER NOT NULL,
                    monotonic_start_ms INTEGER NOT NULL,
                    monotonic_end_ms INTEGER NOT NULL,
                    over_budget INTEGER NOT NULL,
                    overrun_us INTEGER NOT NULL,
                    is_stale INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS replacement_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    from_version INTEGER NOT NULL,
                    to_version INTEGER NOT NULL,
                    round_id TEXT NOT NULL,
                    previous_round_id TEXT,
                    affected_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS cursors (
                    parameter_version INTEGER PRIMARY KEY,
                    round_id TEXT,
                    phase TEXT NOT NULL,
                    last_key TEXT,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS replay_runs (
                    replay_id TEXT PRIMARY KEY,
                    round_id TEXT NOT NULL,
                    parameter_version INTEGER NOT NULL,
                    window_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    first_diff_json TEXT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_rounds_version ON rounds(parameter_version, is_stale);
                CREATE INDEX IF NOT EXISTS idx_ui_latency_version ON ui_latency_samples(parameter_version, is_stale);
                """
            )
            if version == 0:
                self.conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            defaults = {
                "format_version": str(FORMAT_VERSION),
                "current_parameter_version": "0",
                "current_parameter_fingerprint": "",
                "current_round_id": "",
                "current_content_digest": "NONE",
                "last_committed_round_id": "",
                "last_committed_parameter_version": "0",
            }
            with self._transaction():
                for key, value in defaults.items():
                    self.conn.execute(
                        "INSERT OR IGNORE INTO meta(key, value) VALUES(?, ?)", (key, value)
                    )
                self.conn.execute(
                    "INSERT OR IGNORE INTO ledger_info(singleton, format_version, created_at) "
                    "VALUES(1, ?, ?)",
                    (FORMAT_VERSION, utc_now_text()),
                )

    def _meta(self, conn: sqlite3.Connection | None = None) -> dict[str, str]:
        connection = conn or self.conn
        rows = connection.execute("SELECT key, value FROM meta ORDER BY key").fetchall()
        return {row["key"]: row["value"] for row in rows}

    def current_meta(self) -> dict[str, Any]:
        with self._lock:
            raw = self._meta()
            version = self._parse_int_meta(raw, "current_parameter_version")
            format_version = self._parse_int_meta(raw, "format_version")
            try:
                last_committed_version = int(
                    raw.get("last_committed_parameter_version", "0") or 0
                )
            except ValueError:
                last_committed_version = 0
            return {
                "format_version": format_version,
                "schema_version": SCHEMA_VERSION,
                "current_parameter_version": version,
                "current_parameter_fingerprint": raw.get("current_parameter_fingerprint", ""),
                "current_round_id": raw.get("current_round_id") or None,
                "current_content_digest": raw.get("current_content_digest", "NONE"),
                "last_committed_round_id": raw.get("last_committed_round_id") or None,
                "last_committed_parameter_version": last_committed_version,
            }

    def last_committed_context(self) -> tuple[int, str | None]:
        with self._lock:
            raw = self._meta()
            try:
                version = int(raw.get("last_committed_parameter_version", "0") or 0)
            except ValueError:
                version = 0
            return version, raw.get("last_committed_round_id") or None

    @staticmethod
    def _parse_int_meta(raw: dict[str, str], key: str) -> int:
        try:
            return int(raw[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise SnapshotError(f"ledger meta {key!r} is not an integer") from exc

    def verify_snapshot(self) -> dict[str, Any]:
        """Validate ledger format and recompute the current round digest."""

        with self._lock:
            raw = self._meta()
            info_row = self.conn.execute(
                "SELECT format_version FROM ledger_info WHERE singleton=1"
            ).fetchone()
            if info_row is None:
                self._invalidate_snapshot("ledger_info integer format version is missing")
                raise SnapshotError("ledger_info integer format version is missing")
            try:
                integer_format_version = int(info_row["format_version"])
            except (TypeError, ValueError) as exc:
                self._invalidate_snapshot("ledger_info format_version is not an integer")
                raise SnapshotError(
                    "ledger_info format_version is not an integer"
                ) from exc
            try:
                format_version = int(raw["format_version"])
            except (KeyError, ValueError) as exc:
                self._invalidate_snapshot("format_version is missing or not an integer")
                raise SnapshotError("ledger format_version is missing or not an integer") from exc
            if format_version != FORMAT_VERSION:
                self._invalidate_snapshot(f"unsupported format_version {format_version}")
                raise SnapshotError(
                    f"unsupported ledger format_version {format_version}; expected {FORMAT_VERSION}"
                )
            if integer_format_version != FORMAT_VERSION:
                self._invalidate_snapshot(
                    f"unsupported integer format version {integer_format_version}"
                )
                raise SnapshotError(
                    "unsupported integer ledger format version; expected "
                    f"{FORMAT_VERSION}"
                )
            current_version = self._parse_int_meta(raw, "current_parameter_version")
            current_round_id = raw.get("current_round_id", "")
            meta_digest = raw.get("current_content_digest", "NONE")
            if not current_round_id:
                return {
                    "format_version": format_version,
                    "parameter_version": current_version,
                    "round_id": None,
                    "content_digest": meta_digest,
                    "valid": meta_digest in ("NONE", "INVALID"),
                    "invalidated": meta_digest == "INVALID",
                }

            round_row = self.conn.execute(
                "SELECT * FROM rounds WHERE round_id=?", (current_round_id,)
            ).fetchone()
            if round_row is None:
                self._invalidate_snapshot("current round row is missing")
                raise SnapshotError("current round row is missing from the ledger")
            if int(round_row["is_stale"]) != 0:
                self._invalidate_snapshot("current round is stale")
                raise SnapshotError("current round is marked stale")
            if int(round_row["parameter_version"]) != current_version:
                self._invalidate_snapshot("current round parameter version differs from meta")
                raise SnapshotError("current round parameter version differs from ledger meta")
            if round_row["parameter_fingerprint"] != raw.get("current_parameter_fingerprint", ""):
                self._invalidate_snapshot("current round fingerprint differs from meta")
                raise SnapshotError("current round fingerprint differs from ledger meta")
            try:
                content = self._build_round_content(self.conn, current_round_id)
                computed = hashlib.sha256(
                    canonical_json(content).encode("utf-8")
                ).hexdigest()
            except Exception as exc:
                self._invalidate_snapshot(
                    "current round content cannot be canonicalised",
                    details={"error": f"{type(exc).__name__}: {exc}"},
                )
                raise SnapshotError(
                    f"current round content cannot be canonicalised: {exc}"
                ) from exc
            if computed != round_row["content_digest"] or meta_digest != computed:
                self._invalidate_snapshot(
                    "current content digest mismatch",
                    details={
                        "stored_round_digest": round_row["content_digest"],
                        "stored_meta_digest": meta_digest,
                        "recomputed_digest": computed,
                    },
                )
                raise SnapshotError(
                    "current content digest mismatch; snapshot is not trusted"
                )
            return {
                "format_version": format_version,
                "parameter_version": current_version,
                "round_id": current_round_id,
                "content_digest": computed,
                "valid": True,
                "invalidated": False,
            }

    def _invalidate_snapshot(self, reason: str, *, details: dict[str, Any] | None = None) -> None:
        with self._lock:
            try:
                with self._transaction():
                    current_round = self.conn.execute(
                        "SELECT value FROM meta WHERE key='current_round_id'"
                    ).fetchone()
                    version_row = self.conn.execute(
                        "SELECT value FROM meta WHERE key='current_parameter_version'"
                    ).fetchone()
                    if current_round and current_round["value"]:
                        self.conn.execute(
                            "UPDATE rounds SET is_stale=1, stale_reason=? WHERE round_id=?",
                            ("SNAPSHOT_INVALIDATED:" + reason, current_round["value"]),
                        )
                        self.conn.execute(
                            "UPDATE ui_latency_samples SET is_stale=1 WHERE round_id=?",
                            (current_round["value"],),
                        )
                    if version_row:
                        try:
                            version = int(version_row["value"])
                        except ValueError:
                            version = 0
                        self.conn.execute(
                            """
                            INSERT INTO cursors(parameter_version, round_id, phase, last_key, updated_at)
                            VALUES(?, NULL, 'RERUN_REQUIRED', NULL, ?)
                            ON CONFLICT(parameter_version) DO UPDATE SET
                                round_id=NULL, phase='RERUN_REQUIRED', last_key=NULL,
                                updated_at=excluded.updated_at
                            """,
                            (version, utc_now_text()),
                        )
                    self.conn.execute(
                        "UPDATE meta SET value='' WHERE key='current_round_id'"
                    )
                    self.conn.execute(
                        "UPDATE meta SET value='INVALID' WHERE key='current_content_digest'"
                    )
                    self.conn.execute(
                        "UPDATE meta SET value='' WHERE key='last_committed_round_id'"
                    )
                    self.conn.execute(
                        "UPDATE meta SET value='0' "
                        "WHERE key='last_committed_parameter_version'"
                    )
            except sqlite3.Error:
                pass
            self.logger.error(
                "E_SNAPSHOT",
                "ledger snapshot rejected and current view invalidated",
                reason=reason,
                **(details or {}),
            )

    def recover_incomplete_attempts(self) -> RecoveryInfo:
        with self._lock:
            rows = self.conn.execute(
                "SELECT attempt_id, parameter_version, window_id, started_at "
                "FROM attempts WHERE status='IN_PROGRESS' ORDER BY attempt_id"
            ).fetchall()
            unknown: list[dict[str, Any]] = []
            if rows:
                now = utc_now_text()
                with self._transaction():
                    for row in rows:
                        self.conn.execute(
                            "UPDATE attempts SET status='UNKNOWN', finished_at=?, note=? "
                            "WHERE attempt_id=? AND status='IN_PROGRESS'",
                            (
                                now,
                                "process terminated before transaction commit; rerun required; no exit report exists",
                                row["attempt_id"],
                            ),
                        )
                        unknown.append(
                            {
                                "attempt_id": row["attempt_id"],
                                "parameter_version": int(row["parameter_version"]),
                                "window_id": row["window_id"],
                                "started_at": row["started_at"],
                                "status": "UNKNOWN",
                                "requires_rerun": True,
                            }
                        )
            for item in unknown:
                self.logger.warning(
                    "E_RECOVERY",
                    "uncommitted attempt marked UNKNOWN; only committed rounds are restorable",
                    **item,
                )
            return RecoveryInfo(unknown_attempts=unknown)

    def activate_parameter(self, spec: Spec) -> ActivationResult:
        """Activate a validated parameter set and stale all prior views if changed."""

        with self._lock:
            raw = self._meta()
            current_version = self._parse_int_meta(raw, "current_parameter_version")
            current_fp = raw.get("current_parameter_fingerprint", "")
            previous_round_id = raw.get("current_round_id") or None
            if previous_round_id is None:
                previous_round_id = raw.get("last_committed_round_id") or None
            declared = spec.parameter_version
            fingerprint = spec.fingerprint

            if current_version == 0:
                target_version = declared
                changed = True
                reason = "INITIAL"
            elif fingerprint != current_fp:
                target_version = max(current_version + 1, declared)
                changed = True
                reason = "PARAMETER_CHANGED"
            elif declared > current_version:
                target_version = declared
                changed = True
                reason = "PARAMETER_VERSION_BUMP"
            else:
                target_version = current_version
                changed = False
                reason = "UNCHANGED"

            if not changed:
                row = self.conn.execute(
                    "SELECT * FROM parameter_versions WHERE version=?", (current_version,)
                ).fetchone()
                if row is None:
                    self._invalidate_snapshot("current parameter version row is missing")
                    raise SnapshotError("current parameter version row is missing")
                return ActivationResult(
                    previous_version=current_version,
                    current_version=current_version,
                    previous_round_id=previous_round_id,
                    changed=False,
                    reason=reason,
                )

            effective_spec = spec.with_parameter_version(target_version)
            now = utc_now_text()
            with self._transaction():
                if current_version:
                    old_rounds = self.conn.execute(
                        "SELECT round_id FROM rounds WHERE is_stale=0"
                    ).fetchall()
                    for row in old_rounds:
                        self.conn.execute(
                            "UPDATE rounds SET is_stale=1, stale_reason=? WHERE round_id=?",
                            (reason, row["round_id"]),
                        )
                    self.conn.execute(
                        "UPDATE ui_latency_samples SET is_stale=1 WHERE parameter_version=?",
                        (current_version,),
                    )
                self.conn.execute(
                    "UPDATE parameter_versions SET is_current=0 WHERE is_current=1"
                )
                self.conn.execute(
                    """
                    INSERT INTO parameter_versions(version, fingerprint, spec_json, activated_at, is_current)
                    VALUES(?, ?, ?, ?, 1)
                    """,
                    (target_version, fingerprint, effective_spec.canonical_json(), now),
                )
                self.conn.execute(
                    "UPDATE meta SET value=? WHERE key='current_parameter_version'",
                    (str(target_version),),
                )
                self.conn.execute(
                    "UPDATE meta SET value=? WHERE key='current_parameter_fingerprint'",
                    (fingerprint,),
                )
                self.conn.execute(
                    "UPDATE meta SET value='' WHERE key='current_round_id'"
                )
                self.conn.execute(
                    "UPDATE meta SET value='NONE' WHERE key='current_content_digest'"
                )
                self.conn.execute(
                    """
                    INSERT INTO cursors(parameter_version, round_id, phase, last_key, updated_at)
                    VALUES(?, NULL, 'WAITING_FOR_ROUND', NULL, ?)
                    ON CONFLICT(parameter_version) DO UPDATE SET
                        round_id=NULL, phase='WAITING_FOR_ROUND', last_key=NULL,
                        updated_at=excluded.updated_at
                    """,
                    (target_version, now),
                )
            self.logger.info(
                "PARAMETER_ACTIVATED",
                "parameter set activated; previous conclusions are stale",
                previous_version=current_version,
                current_version=target_version,
                changed=True,
                reason=reason,
            )
            return ActivationResult(
                previous_version=current_version,
                current_version=target_version,
                previous_round_id=previous_round_id,
                changed=True,
                reason=reason,
            )

    def begin_attempt(self, *, parameter_version: int, window_id: str) -> str:
        attempt_id = uuid.uuid4().hex
        with self._lock:
            with self._transaction():
                self.conn.execute(
                    """
                    INSERT INTO attempts(attempt_id, parameter_version, window_id, started_at, status)
                    VALUES(?, ?, ?, ?, 'IN_PROGRESS')
                    """,
                    (attempt_id, parameter_version, window_id, utc_now_text()),
                )
        return attempt_id

    def mark_attempt_failed(self, attempt_id: str, note: str) -> None:
        with self._lock:
            with self._transaction():
                self.conn.execute(
                    "UPDATE attempts SET status='FAILED', finished_at=?, note=? "
                    "WHERE attempt_id=? AND status='IN_PROGRESS'",
                    (utc_now_text(), note[:2000], attempt_id),
                )

    def commit_round(
        self,
        *,
        spec: Spec,
        attempt_id: str,
        window_id: str,
        batch: FetchBatch,
        merge_result: MergeResult,
        previous_version: int | None = None,
        previous_round_id: str | None = None,
    ) -> CommitResult:
        """Atomically write observations, decisions, latency stats and cursor."""

        with self._lock:
            raw = self._meta()
            current_version = self._parse_int_meta(raw, "current_parameter_version")
            if current_version != spec.parameter_version:
                raise SnapshotError(
                    "cannot commit a round for a parameter version other than the active one"
                )
            if raw.get("current_parameter_fingerprint") != spec.fingerprint:
                raise SnapshotError("active parameter fingerprint differs from round spec")
            attempt = self.conn.execute(
                "SELECT * FROM attempts WHERE attempt_id=?", (attempt_id,)
            ).fetchone()
            if attempt is None or attempt["status"] != "IN_PROGRESS":
                raise RuntimeError("round attempt is not IN_PROGRESS")
            if int(attempt["parameter_version"]) != spec.parameter_version:
                raise RuntimeError("attempt parameter version differs from round spec")

            round_id = uuid.uuid4().hex
            now = utc_now_text()
            old_current_round = raw.get("current_round_id") or None
            if previous_round_id is None:
                previous_round_id = old_current_round
            if previous_version is None:
                previous_version = current_version

            fetch_samples = [
                LatencySample(
                    kind="fetch",
                    node_id=outcome.node_id,
                    label=outcome.key,
                    duration_us=outcome.request_latency_us,
                    monotonic_start_ms=outcome.request_start_ms,
                    monotonic_end_ms=outcome.request_end_ms,
                )
                for outcome in batch.ordered()
            ]
            aggregates = build_fetch_aggregates(fetch_samples, spec.sorted_node_ids)

            with self._transaction():
                self.conn.execute(
                    """
                    INSERT INTO rounds(
                        round_id, attempt_id, parameter_version, parameter_fingerprint,
                        window_id, is_stale, stale_reason, committed_at, content_digest
                    ) VALUES(?, ?, ?, ?, ?, 0, NULL, ?, 'PENDING')
                    """,
                    (
                        round_id,
                        attempt_id,
                        spec.parameter_version,
                        spec.fingerprint,
                        window_id,
                        now,
                    ),
                )
                for outcome in batch.ordered():
                    self.conn.execute(
                        """
                        INSERT INTO observation_requests(
                            round_id, node_id, key, status, observations_json,
                            error_code, error_reason, diagnostic_code, http_status,
                            request_start_ms, request_end_ms, request_latency_us,
                            queue_wait_start_ms, queue_wait_end_ms, queued
                        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            round_id,
                            outcome.node_id,
                            outcome.key,
                            outcome.status,
                            canonical_json(outcome.observations_as_list()),
                            None if outcome.ok else "E_OBSERVE",
                            outcome.error_reason,
                            outcome.diagnostic_code,
                            outcome.http_status,
                            outcome.request_start_ms,
                            outcome.request_end_ms,
                            outcome.request_latency_us,
                            outcome.queue_wait_start_ms,
                            outcome.queue_wait_end_ms,
                            1 if outcome.queued else 0,
                        ),
                    )
                for decision in merge_result.decisions:
                    effective = decision.get("effective_value")
                    self.conn.execute(
                        """
                        INSERT INTO conclusions(
                            round_id, key, status, divergent, quorum_met,
                            effective_value_json, decision_json
                        ) VALUES(?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            round_id,
                            decision["key"],
                            decision["status"],
                            1 if decision["divergent"] else 0,
                            1 if decision["quorum_met"] else 0,
                            None if effective is None else canonical_json(effective),
                            canonical_json(decision),
                        ),
                    )
                for comparison in merge_result.comparisons:
                    self.conn.execute(
                        """
                        INSERT INTO node_comparisons(
                            round_id, key, node_id, request_status, http_status,
                            error_reason, value_json, observation_seq, timestamp_ms,
                            duplicate_count, adopted, reason, latest_candidate
                        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            round_id,
                            comparison["key"],
                            comparison["node_id"],
                            comparison["request_status"],
                            comparison["http_status"],
                            comparison["error_reason"],
                            None
                            if comparison["value"] is None
                            else canonical_json(comparison["value"]),
                            comparison["observation_seq"],
                            comparison["timestamp_ms"],
                            comparison["duplicate_count"],
                            1 if comparison["adopted"] else 0,
                            comparison["reason"],
                            1 if comparison["latest_candidate"] else 0,
                        ),
                    )
                for sample in fetch_samples:
                    self.conn.execute(
                        """
                        INSERT INTO latency_samples(
                            round_id, kind, node_id, label, duration_us,
                            monotonic_start_ms, monotonic_end_ms, over_budget, overrun_us
                        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            round_id,
                            sample.kind,
                            sample.node_id or "",
                            sample.label,
                            sample.duration_us,
                            sample.monotonic_start_ms,
                            sample.monotonic_end_ms,
                            1 if sample.over_budget else 0,
                            sample.overrun_us,
                        ),
                    )
                for aggregate in aggregates:
                    self.conn.execute(
                        """
                        INSERT INTO latency_aggregates(
                            round_id, scope, node_id, sample_count, mean_us, p95_us,
                            max_us, quantization_error_bound_fixed6_ms, stats_json
                        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            round_id,
                            aggregate["scope"],
                            aggregate.get("node_id"),
                            int(aggregate["sample_count"]),
                            int(aggregate["mean_us"]),
                            int(aggregate["p95_us"]),
                            int(aggregate["max_us"]),
                            int(aggregate["quantization_error_bound_fixed6_ms"]),
                            canonical_json(aggregate),
                        ),
                    )

                self.conn.execute(
                    "UPDATE rounds SET is_stale=1, stale_reason='ROUND_SUPERSEDED' "
                    "WHERE round_id<>? AND is_stale=0",
                    (round_id,),
                )
                self.conn.execute(
                    "UPDATE ui_latency_samples SET is_stale=1 "
                    "WHERE is_stale=0 AND (round_id IS NULL OR round_id<>?)",
                    (round_id,),
                )
                self.conn.execute(
                    """
                    INSERT INTO cursors(parameter_version, round_id, phase, last_key, updated_at)
                    VALUES(?, ?, 'COMMITTED', ?, ?)
                    ON CONFLICT(parameter_version) DO UPDATE SET
                        round_id=excluded.round_id, phase='COMMITTED',
                        last_key=excluded.last_key, updated_at=excluded.updated_at
                    """,
                    (
                        spec.parameter_version,
                        round_id,
                        spec.sorted_keys[-1] if spec.sorted_keys else None,
                        now,
                    ),
                )
                content = self._build_round_content(self.conn, round_id)
                digest = hashlib.sha256(canonical_json(content).encode("utf-8")).hexdigest()
                self.conn.execute(
                    "UPDATE rounds SET content_digest=? WHERE round_id=?", (digest, round_id)
                )
                self.conn.execute(
                    "UPDATE attempts SET status='COMMITTED', finished_at=?, note=NULL "
                    "WHERE attempt_id=?",
                    (now, attempt_id),
                )
                self.conn.execute(
                    "UPDATE meta SET value=? WHERE key='current_round_id'", (round_id,)
                )
                self.conn.execute(
                    "UPDATE meta SET value=? WHERE key='current_content_digest'", (digest,)
                )
                self.conn.execute(
                    "UPDATE meta SET value=? WHERE key='last_committed_round_id'",
                    (round_id,),
                )
                self.conn.execute(
                    "UPDATE meta SET value=? "
                    "WHERE key='last_committed_parameter_version'",
                    (str(spec.parameter_version),),
                )
                self._write_replacement_event(
                    from_version=int(previous_version),
                    to_version=spec.parameter_version,
                    round_id=round_id,
                    previous_round_id=previous_round_id,
                    new_spec=spec,
                    decision_rows=merge_result.decisions,
                )

            self.logger.info(
                "ROUND_COMMITTED",
                "round, conclusions, latency and cursor committed atomically",
                round_id=round_id,
                parameter_version=spec.parameter_version,
                content_digest=digest,
                attempt_id=attempt_id,
            )
            return CommitResult(
                round_id=round_id,
                content_digest=digest,
                parameter_version=spec.parameter_version,
            )

    def _write_replacement_event(
        self,
        *,
        from_version: int,
        to_version: int,
        round_id: str,
        previous_round_id: str | None,
        new_spec: Spec,
        decision_rows: tuple[dict[str, Any], ...],
    ) -> None:
        old_decisions: dict[str, dict[str, Any]] = {}
        old_spec_data: dict[str, Any] | None = None
        if previous_round_id:
            rows = self.conn.execute(
                "SELECT key, decision_json FROM conclusions WHERE round_id=? ORDER BY key",
                (previous_round_id,),
            ).fetchall()
            old_decisions = {
                row["key"]: json.loads(row["decision_json"]) for row in rows
            }
        if from_version > 0:
            old_spec_row = self.conn.execute(
                "SELECT spec_json FROM parameter_versions WHERE version=?", (from_version,)
            ).fetchone()
            if old_spec_row:
                old_spec_data = json.loads(old_spec_row["spec_json"])

        parameter_differences: list[dict[str, Any]] = []
        parameter_keys: set[str] = set()
        all_keys_changed = False
        if old_spec_data is not None:
            old_keys = old_spec_data.get("keys", {})
            new_keys = new_spec.to_dict()["keys"]
            for key in sorted(set(old_keys) | set(new_keys)):
                old_value = old_keys.get(key)
                new_value = new_keys.get(key)
                if old_value != new_value:
                    parameter_keys.add(key)
                    parameter_differences.append(
                        {"key": key, "old": old_value, "new": new_value}
                    )
            old_nodes = old_spec_data.get("nodes", [])
            new_nodes = new_spec.to_dict()["nodes"]
            globals_to_compare = (
                "observation_window",
                "global_concurrency",
                "observation_timeout_ms",
                "ui_latency_budget_ms",
                "arbitration_rule",
            )
            for field in globals_to_compare:
                if old_spec_data.get(field) != new_spec.to_dict().get(field):
                    all_keys_changed = True
                    parameter_differences.append(
                        {
                            "key": "<global>",
                            "field": field,
                            "old": old_spec_data.get(field),
                            "new": new_spec.to_dict().get(field),
                        }
                    )
            if old_nodes != new_nodes:
                all_keys_changed = True
                parameter_differences.append(
                    {"key": "<global>", "field": "nodes", "old": old_nodes, "new": new_nodes}
                )

        new_decisions = {decision["key"]: decision for decision in decision_rows}
        decision_changed_keys = {
            key
            for key in sorted(set(old_decisions) | set(new_decisions))
            if old_decisions.get(key) != new_decisions.get(key)
        }
        if all_keys_changed:
            affected = sorted(set(old_decisions) | set(new_decisions))
        else:
            affected = sorted(parameter_keys | decision_changed_keys)
        if not affected and not parameter_differences:
            return

        affected_rows: list[dict[str, Any]] = []
        for key in affected:
            old_decision = old_decisions.get(key)
            new_decision = new_decisions.get(key)
            old_summary = self._decision_summary(old_decision)
            new_summary = self._decision_summary(new_decision)
            affected_rows.append(
                {
                    "key": key,
                    "old": old_summary,
                    "new": new_summary,
                    "decision_flip": old_summary != new_summary,
                    "parameter_caused": key in parameter_keys or all_keys_changed,
                }
            )
        payload = {
            "parameter_differences": parameter_differences,
            "affected_keys": affected_rows,
            "replacement_semantics": "whole-current-view",
        }
        self.conn.execute(
            """
            INSERT INTO replacement_events(
                from_version, to_version, round_id, previous_round_id, affected_json, created_at
            ) VALUES(?, ?, ?, ?, ?, ?)
            """,
            (
                from_version,
                to_version,
                round_id,
                previous_round_id,
                canonical_json(payload),
                utc_now_text(),
            ),
        )

    @staticmethod
    def _decision_summary(decision: dict[str, Any] | None) -> dict[str, Any] | None:
        if decision is None:
            return None
        return {
            "status": decision.get("status"),
            "effective_value": decision.get("effective_value"),
            "divergent": decision.get("divergent"),
            "quorum_met": decision.get("quorum_met"),
            "expected_matches": decision.get("expected_matches"),
            "attributions": decision.get("attributions", []),
        }

    def _build_round_content(
        self, conn: sqlite3.Connection, round_id: str
    ) -> dict[str, Any]:
        round_row = conn.execute(
            "SELECT * FROM rounds WHERE round_id=?", (round_id,)
        ).fetchone()
        if round_row is None:
            raise SnapshotError(f"round {round_id} is missing")
        spec_row = conn.execute(
            "SELECT spec_json FROM parameter_versions WHERE version=?",
            (int(round_row["parameter_version"]),),
        ).fetchone()
        if spec_row is None:
            raise SnapshotError(
                f"parameter version {round_row['parameter_version']} is missing"
            )
        observation_rows = conn.execute(
            "SELECT * FROM observation_requests WHERE round_id=? ORDER BY node_id, key",
            (round_id,),
        ).fetchall()
        conclusion_rows = conn.execute(
            "SELECT * FROM conclusions WHERE round_id=? ORDER BY key", (round_id,)
        ).fetchall()
        comparison_rows = conn.execute(
            "SELECT * FROM node_comparisons WHERE round_id=? ORDER BY key, node_id",
            (round_id,),
        ).fetchall()
        latency_rows = conn.execute(
            "SELECT * FROM latency_samples WHERE round_id=? "
            "ORDER BY node_id, label, kind, monotonic_start_ms",
            (round_id,),
        ).fetchall()
        aggregate_rows = conn.execute(
            "SELECT * FROM latency_aggregates WHERE round_id=? "
            "ORDER BY scope, COALESCE(node_id, '')",
            (round_id,),
        ).fetchall()
        cursor_row = conn.execute(
            "SELECT * FROM cursors WHERE parameter_version=?",
            (int(round_row["parameter_version"]),),
        ).fetchone()
        return {
            "content_version": 1,
            "round": {
                "round_id": round_row["round_id"],
                "parameter_version": int(round_row["parameter_version"]),
                "parameter_fingerprint": round_row["parameter_fingerprint"],
                "window_id": round_row["window_id"],
            },
            "parameter_spec": json.loads(spec_row["spec_json"]),
            "cursor": (
                None
                if cursor_row is None
                else {
                    "parameter_version": int(cursor_row["parameter_version"]),
                    "round_id": cursor_row["round_id"],
                    "phase": cursor_row["phase"],
                    "last_key": cursor_row["last_key"],
                    "updated_at": cursor_row["updated_at"],
                }
            ),
            "observation_requests": [
                {
                    "node_id": row["node_id"],
                    "key": row["key"],
                    "status": row["status"],
                    "observations": json.loads(row["observations_json"]),
                    "error_code": row["error_code"],
                    "error_reason": row["error_reason"],
                    "diagnostic_code": row["diagnostic_code"],
                    "http_status": row["http_status"],
                    "request_start_ms": int(row["request_start_ms"]),
                    "request_end_ms": int(row["request_end_ms"]),
                    "request_latency_us": int(row["request_latency_us"]),
                    "queue_wait_start_ms": int(row["queue_wait_start_ms"]),
                    "queue_wait_end_ms": int(row["queue_wait_end_ms"]),
                    "queued": bool(row["queued"]),
                }
                for row in observation_rows
            ],
            "conclusions": [
                json.loads(row["decision_json"]) for row in conclusion_rows
            ],
            "node_comparisons": [
                {
                    "key": row["key"],
                    "node_id": row["node_id"],
                    "request_status": row["request_status"],
                    "http_status": row["http_status"],
                    "error_reason": row["error_reason"],
                    "value": None if row["value_json"] is None else json.loads(row["value_json"]),
                    "observation_seq": row["observation_seq"],
                    "timestamp_ms": row["timestamp_ms"],
                    "duplicate_count": int(row["duplicate_count"]),
                    "adopted": bool(row["adopted"]),
                    "reason": row["reason"],
                    "latest_candidate": bool(row["latest_candidate"]),
                }
                for row in comparison_rows
            ],
            "latency_samples": [
                {
                    "kind": row["kind"],
                    "node_id": row["node_id"],
                    "label": row["label"],
                    "duration_us": int(row["duration_us"]),
                    "monotonic_start_ms": int(row["monotonic_start_ms"]),
                    "monotonic_end_ms": int(row["monotonic_end_ms"]),
                    "over_budget": bool(row["over_budget"]),
                    "overrun_us": int(row["overrun_us"]),
                }
                for row in latency_rows
            ],
            "latency_aggregates": [
                json.loads(row["stats_json"]) for row in aggregate_rows
            ],
        }

    def record_ui_latency(
        self,
        *,
        parameter_version: int,
        round_id: str | None,
        duration_us: int,
        monotonic_start_ms: int,
        monotonic_end_ms: int,
        over_budget: bool,
        overrun_us: int,
    ) -> int:
        with self._lock:
            with self._transaction():
                cursor = self.conn.execute(
                    """
                    INSERT INTO ui_latency_samples(
                        parameter_version, round_id, duration_us,
                        monotonic_start_ms, monotonic_end_ms, over_budget, overrun_us,
                        is_stale, created_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, 0, ?)
                    """,
                    (
                        parameter_version,
                        round_id,
                        max(0, duration_us),
                        monotonic_start_ms,
                        monotonic_end_ms,
                        1 if over_budget else 0,
                        max(0, overrun_us),
                        utc_now_text(),
                    ),
                )
                return int(cursor.lastrowid)

    def ui_latency_summary(self, *, parameter_version: int, round_id: str | None) -> dict[str, Any]:
        with self._lock:
            rows = self.conn.execute(
                """
                SELECT duration_us FROM ui_latency_samples
                WHERE parameter_version=? AND round_id IS ? AND is_stale=0
                ORDER BY sample_id
                """,
                (parameter_version, round_id),
            ).fetchall()
            stats = aggregate_stats(row["duration_us"] for row in rows)
            stats["kind"] = "ui_query"
            stats["true_source"] = "ui_latency_samples rows in the SQLite ledger"
            return stats

    def load_dashboard(self) -> dict[str, Any]:
        with self._lock:
            raw = self._meta()
            parameter_version = self._parse_int_meta(raw, "current_parameter_version")
            round_id = raw.get("current_round_id") or None
            current_round = None
            conclusions: list[dict[str, Any]] = []
            comparisons: list[dict[str, Any]] = []
            fetch_latency: list[dict[str, Any]] = []
            if round_id:
                round_row = self.conn.execute(
                    "SELECT * FROM rounds WHERE round_id=? AND is_stale=0", (round_id,)
                ).fetchone()
                if round_row is not None:
                    current_round = {
                        "round_id": round_row["round_id"],
                        "parameter_version": int(round_row["parameter_version"]),
                        "parameter_fingerprint": round_row["parameter_fingerprint"],
                        "window_id": round_row["window_id"],
                        "committed_at": round_row["committed_at"],
                        "content_digest": round_row["content_digest"],
                    }
                    conclusions = [
                        json.loads(row["decision_json"])
                        for row in self.conn.execute(
                            "SELECT decision_json FROM conclusions WHERE round_id=? ORDER BY key",
                            (round_id,),
                        ).fetchall()
                    ]
                    comparisons = [
                        self._comparison_from_row(row)
                        for row in self.conn.execute(
                            "SELECT * FROM node_comparisons WHERE round_id=? "
                            "ORDER BY key, node_id",
                            (round_id,),
                        ).fetchall()
                    ]
                    fetch_latency = [
                        json.loads(row["stats_json"])
                        for row in self.conn.execute(
                            "SELECT stats_json FROM latency_aggregates WHERE round_id=? "
                            "ORDER BY scope, COALESCE(node_id, '')",
                            (round_id,),
                        ).fetchall()
                    ]
            versions = [
                {
                    "version": int(row["version"]),
                    "fingerprint": row["fingerprint"],
                    "is_current": bool(row["is_current"]),
                    "activated_at": row["activated_at"],
                }
                for row in self.conn.execute(
                    "SELECT version, fingerprint, activated_at, is_current "
                    "FROM parameter_versions ORDER BY version DESC"
                ).fetchall()
            ]
            replacements = [
                {
                    "event_id": int(row["event_id"]),
                    "from_version": int(row["from_version"]),
                    "to_version": int(row["to_version"]),
                    "round_id": row["round_id"],
                    "previous_round_id": row["previous_round_id"],
                    "detail": json.loads(row["affected_json"]),
                    "created_at": row["created_at"],
                }
                for row in self.conn.execute(
                    "SELECT * FROM replacement_events ORDER BY event_id DESC"
                ).fetchall()
            ]
            unknown_attempts = [
                dict(row)
                for row in self.conn.execute(
                    "SELECT attempt_id, parameter_version, window_id, started_at, status, note "
                    "FROM attempts WHERE status='UNKNOWN' ORDER BY attempt_id"
                ).fetchall()
            ]
            cursor_rows = self.conn.execute(
                "SELECT * FROM cursors ORDER BY parameter_version DESC"
            ).fetchall()
            cursor = None
            for row in cursor_rows:
                if int(row["parameter_version"]) == parameter_version:
                    cursor = dict(row)
                    break
            return {
                "meta": {
                    "format_version": self._parse_int_meta(raw, "format_version"),
                    "parameter_version": parameter_version,
                    "parameter_fingerprint": raw.get("current_parameter_fingerprint", ""),
                    "current_round_id": round_id,
                    "current_content_digest": raw.get("current_content_digest", "NONE"),
                },
                "round": current_round,
                "conclusions": conclusions,
                "comparisons": comparisons,
                "fetch_latency": fetch_latency,
                "ui_latency": self.ui_latency_summary(
                    parameter_version=parameter_version, round_id=round_id
                ),
                "parameter_versions": versions,
                "replacement_events": replacements,
                "unknown_attempts": unknown_attempts,
                "cursor": cursor,
            }

    @staticmethod
    def _comparison_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "key": row["key"],
            "node_id": row["node_id"],
            "request_status": row["request_status"],
            "http_status": row["http_status"],
            "error_reason": row["error_reason"],
            "value": None if row["value_json"] is None else json.loads(row["value_json"]),
            "observation_seq": row["observation_seq"],
            "timestamp_ms": row["timestamp_ms"],
            "duplicate_count": int(row["duplicate_count"]),
            "adopted": bool(row["adopted"]),
            "reason": row["reason"],
            "latest_candidate": bool(row["latest_candidate"]),
        }

    def load_current_bundle(self) -> dict[str, Any] | None:
        with self._lock:
            raw = self._meta()
            round_id = raw.get("current_round_id") or None
            if not round_id:
                return None
            round_row = self.conn.execute(
                "SELECT * FROM rounds WHERE round_id=? AND is_stale=0", (round_id,)
            ).fetchone()
            if round_row is None:
                return None
            spec_row = self.conn.execute(
                "SELECT spec_json FROM parameter_versions WHERE version=?",
                (int(round_row["parameter_version"]),),
            ).fetchone()
            observations = [
                {
                    "node_id": row["node_id"],
                    "key": row["key"],
                    "status": row["status"],
                    "observations": json.loads(row["observations_json"]),
                    "error_reason": row["error_reason"],
                    "diagnostic_code": row["diagnostic_code"],
                }
                for row in self.conn.execute(
                    "SELECT * FROM observation_requests WHERE round_id=? ORDER BY node_id, key",
                    (round_id,),
                ).fetchall()
            ]
            decisions = [
                json.loads(row["decision_json"])
                for row in self.conn.execute(
                    "SELECT decision_json FROM conclusions WHERE round_id=? ORDER BY key",
                    (round_id,),
                ).fetchall()
            ]
            comparisons = [
                self._comparison_from_row(row)
                for row in self.conn.execute(
                    "SELECT * FROM node_comparisons WHERE round_id=? ORDER BY key, node_id",
                    (round_id,),
                ).fetchall()
            ]
            latency_samples = [
                {
                    "kind": row["kind"],
                    "node_id": row["node_id"],
                    "label": row["label"],
                    "duration_us": int(row["duration_us"]),
                    "monotonic_start_ms": int(row["monotonic_start_ms"]),
                    "monotonic_end_ms": int(row["monotonic_end_ms"]),
                    "over_budget": bool(row["over_budget"]),
                    "overrun_us": int(row["overrun_us"]),
                }
                for row in self.conn.execute(
                    "SELECT * FROM latency_samples WHERE round_id=? "
                    "ORDER BY node_id, label, kind, monotonic_start_ms",
                    (round_id,),
                ).fetchall()
            ]
            latency_aggregates = [
                json.loads(row["stats_json"])
                for row in self.conn.execute(
                    "SELECT stats_json FROM latency_aggregates WHERE round_id=? "
                    "ORDER BY scope, COALESCE(node_id, '')",
                    (round_id,),
                ).fetchall()
            ]
            return {
                "round_id": round_id,
                "parameter_version": int(round_row["parameter_version"]),
                "parameter_fingerprint": round_row["parameter_fingerprint"],
                "window_id": round_row["window_id"],
                "content_digest": round_row["content_digest"],
                "spec": None if spec_row is None else json.loads(spec_row["spec_json"]),
                "observations": observations,
                "decisions": decisions,
                "comparisons": comparisons,
                "latency_samples": latency_samples,
                "latency_aggregates": latency_aggregates,
            }

    def record_replay_run(
        self,
        *,
        round_id: str,
        parameter_version: int,
        window_id: str,
        status: str,
        first_diff: dict[str, Any] | None,
        started_at: str,
        finished_at: str,
    ) -> str:
        replay_id = uuid.uuid4().hex
        with self._lock:
            with self._transaction():
                self.conn.execute(
                    """
                    INSERT INTO replay_runs(
                        replay_id, round_id, parameter_version, window_id,
                        status, first_diff_json, started_at, finished_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        replay_id,
                        round_id,
                        parameter_version,
                        window_id,
                        status,
                        None if first_diff is None else canonical_json(first_diff),
                        started_at,
                        finished_at,
                    ),
                )
        return replay_id
