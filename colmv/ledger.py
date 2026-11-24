"""SQLite ledger, session snapshots, crash recovery, and scheme invalidation."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import canonical_json, object_digest, sha256_hex
from .errors import ColmvError
from .model import RuntimeConfig


LEDGER_FORMAT_VERSION = 1
SNAPSHOT_FORMAT_VERSION = 1
GENESIS_DIGEST = "0" * 64


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS schemes (
    scheme_version INTEGER PRIMARY KEY,
    fingerprint TEXT NOT NULL,
    views_json TEXT NOT NULL,
    catalog_digest TEXT NOT NULL,
    views_digest TEXT NOT NULL,
    scale_power INTEGER NOT NULL,
    budget INTEGER NOT NULL,
    fixed_contract TEXT NOT NULL,
    is_current INTEGER NOT NULL,
    supersedes INTEGER,
    created_batch INTEGER NOT NULL DEFAULT 0,
    invalidation_json TEXT
);

CREATE TABLE IF NOT EXISTS batch_attempts (
    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_no INTEGER NOT NULL,
    scheme_version INTEGER NOT NULL,
    start_seq INTEGER,
    end_seq INTEGER,
    path TEXT,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS attempt_groups (
    attempt_id INTEGER NOT NULL,
    scheme_version INTEGER NOT NULL,
    batch_no INTEGER NOT NULL,
    view_id TEXT NOT NULL,
    group_sort TEXT NOT NULL,
    group_key_json TEXT NOT NULL,
    status TEXT NOT NULL,
    PRIMARY KEY (attempt_id, view_id, group_sort)
);

CREATE TABLE IF NOT EXISTS batches (
    scheme_version INTEGER NOT NULL,
    batch_no INTEGER NOT NULL,
    start_seq INTEGER NOT NULL,
    end_seq INTEGER NOT NULL,
    change_count INTEGER NOT NULL,
    cursor INTEGER NOT NULL,
    path TEXT NOT NULL,
    decision_json TEXT NOT NULL,
    events_json TEXT NOT NULL,
    groups_json TEXT NOT NULL,
    result_digest TEXT NOT NULL,
    materialized_digest TEXT NOT NULL,
    materialized_json TEXT NOT NULL,
    fixed_version TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    is_stale INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (scheme_version, batch_no)
);

CREATE TABLE IF NOT EXISTS group_states (
    scheme_version INTEGER NOT NULL,
    batch_no INTEGER NOT NULL,
    view_id TEXT NOT NULL,
    group_sort TEXT NOT NULL,
    group_key_json TEXT NOT NULL,
    state_json TEXT NOT NULL,
    status TEXT NOT NULL,
    is_stale INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (scheme_version, batch_no, view_id, group_sort)
);

CREATE TABLE IF NOT EXISTS events (
    scheme_version INTEGER NOT NULL,
    batch_no INTEGER NOT NULL,
    event_seq INTEGER NOT NULL,
    kind TEXT NOT NULL,
    view_id TEXT,
    group_sort TEXT,
    event_json TEXT NOT NULL,
    is_stale INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (scheme_version, batch_no, event_seq)
);

CREATE TABLE IF NOT EXISTS snapshot (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    format_version INTEGER NOT NULL,
    scheme_version INTEGER NOT NULL,
    snapshot_digest TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS recovery_log (
    recovery_id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    details_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS invalidation_runs (
    invalidation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    old_scheme_version INTEGER,
    new_scheme_version INTEGER NOT NULL,
    plan_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class LedgerStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Default deferred transactions are important: every ``with
        # self.connection`` block below is one atomic SQLite transaction.
        self.connection = sqlite3.connect(str(self.path), timeout=30)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript(SCHEMA_SQL)
        self._initialize_meta()
        self.snapshot_replaced = False
        self.snapshot_replacement_details: dict[str, Any] | None = None

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "LedgerStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _initialize_meta(self) -> None:
        with self.connection:
            existing = self.connection.execute(
                "SELECT value FROM meta WHERE key='ledger_format_version'"
            ).fetchone()
            if existing is None:
                self.connection.executemany(
                    "INSERT INTO meta(key, value) VALUES (?, ?)",
                    [
                        ("ledger_format_version", str(LEDGER_FORMAT_VERSION)),
                        ("snapshot_format_version", str(SNAPSHOT_FORMAT_VERSION)),
                        ("content_digest", GENESIS_DIGEST),
                        ("last_batch", "0"),
                        ("cursor", "0"),
                    ],
                )
            elif int(existing["value"]) != LEDGER_FORMAT_VERSION:
                raise ColmvError(
                    "E_LEDGER",
                    "ledger integer format version is unsupported",
                    field="ledger_format_version",
                    details={"expected": LEDGER_FORMAT_VERSION, "actual": existing["value"]},
                )
            meta = {
                row["key"]: row["value"]
                for row in self.connection.execute("SELECT key, value FROM meta")
            }
            for key, value in (
                ("content_digest", GENESIS_DIGEST),
                ("last_batch", "0"),
                ("cursor", "0"),
            ):
                if key not in meta:
                    self.connection.execute("INSERT INTO meta(key, value) VALUES (?, ?)", (key, value))

    # ------------------------------------------------------------------
    # Scheme registration and invalidation
    # ------------------------------------------------------------------
    def register_scheme(
        self,
        config: RuntimeConfig,
        *,
        budget: int,
        invalidation: Mapping[str, Any] | None = None,
    ) -> int:
        fingerprint = config.scheme_fingerprint(budget)
        views_object = {
            "format_version": 1,
            "scale_power": config.scale_power,
            "views": [view.to_dict() for view in config.views],
        }
        with self.connection:
            current = self.connection.execute(
                "SELECT * FROM schemes WHERE is_current=1 ORDER BY scheme_version DESC LIMIT 1"
            ).fetchone()
            if current is not None and current["fingerprint"] == fingerprint:
                return int(current["scheme_version"])

            max_row = self.connection.execute(
                "SELECT COALESCE(MAX(scheme_version), 0) AS version FROM schemes"
            ).fetchone()
            next_version = int(max_row["version"]) + 1
            old_version = int(current["scheme_version"]) if current is not None else None
            invalidation_payload = canonical_json(invalidation) if invalidation is not None else None

            self.connection.execute("UPDATE schemes SET is_current=0")
            self.connection.execute(
                "UPDATE batches SET is_stale=1 WHERE is_stale=0"
            )
            self.connection.execute(
                "UPDATE group_states SET is_stale=1 WHERE is_stale=0"
            )
            self.connection.execute(
                "UPDATE events SET is_stale=1 WHERE is_stale=0"
            )
            self.connection.execute("DELETE FROM snapshot")
            self.connection.execute("UPDATE meta SET value='0' WHERE key IN ('last_batch', 'cursor')")
            self.connection.execute(
                """
                INSERT INTO schemes(
                    scheme_version, fingerprint, views_json, catalog_digest,
                    views_digest, scale_power, budget, fixed_contract,
                    is_current, supersedes, invalidation_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    next_version,
                    fingerprint,
                    canonical_json(views_object),
                    config.catalog_digest,
                    config.views_digest,
                    config.scale_power,
                    budget,
                    f"fixed-v1;scale=10^{config.scale_power};trunc=toward-zero;saturate=int64",
                    old_version,
                    invalidation_payload,
                ),
            )
            if invalidation is not None:
                self.connection.execute(
                    """
                    INSERT INTO invalidation_runs(old_scheme_version, new_scheme_version, plan_json, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (old_version, next_version, canonical_json(invalidation), _utc_now()),
                )
            return next_version

    def current_scheme(self) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM schemes WHERE is_current=1 ORDER BY scheme_version DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row is not None else None

    def scheme(self, scheme_version: int) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT * FROM schemes WHERE scheme_version=?", (scheme_version,)
        ).fetchone()
        return dict(row) if row is not None else None

    def schemes(self) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM schemes ORDER BY scheme_version"
            )
        ]

    # ------------------------------------------------------------------
    # In-flight intent and recovery
    # ------------------------------------------------------------------
    def begin_batch_attempt(
        self,
        *,
        scheme_version: int,
        batch_no: int,
        start_seq: int,
        end_seq: int,
        path: str,
        planned_groups: Sequence[Mapping[str, Any]],
    ) -> int:
        with self.connection:
            cursor = self.connection.execute(
                """
                INSERT INTO batch_attempts(
                    batch_no, scheme_version, start_seq, end_seq, path, status, created_at
                ) VALUES (?, ?, ?, ?, ?, 'inflight', ?)
                """,
                (batch_no, scheme_version, start_seq, end_seq, path, _utc_now()),
            )
            attempt_id = int(cursor.lastrowid)
            for group in planned_groups:
                self.connection.execute(
                    """
                    INSERT OR REPLACE INTO attempt_groups(
                        attempt_id, scheme_version, batch_no, view_id,
                        group_sort, group_key_json, status
                    ) VALUES (?, ?, ?, ?, ?, ?, 'planned')
                    """,
                    (
                        attempt_id,
                        scheme_version,
                        batch_no,
                        group["view_id"],
                        object_digest(group),
                        canonical_json(group),
                    ),
                )
            return attempt_id

    def recover_uncommitted(self) -> dict[str, Any]:
        attempts = [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM batch_attempts WHERE status='inflight' ORDER BY attempt_id"
            )
        ]
        unknown_attempts: list[dict[str, Any]] = []
        with self.connection:
            for attempt in attempts:
                groups = [
                    dict(row)
                    for row in self.connection.execute(
                        "SELECT * FROM attempt_groups WHERE attempt_id=? ORDER BY view_id, group_sort",
                        (attempt["attempt_id"],),
                    )
                ]
                self.connection.execute(
                    "UPDATE batch_attempts SET status='unknown' WHERE attempt_id=?",
                    (attempt["attempt_id"],),
                )
                self.connection.execute(
                    "UPDATE attempt_groups SET status='unknown' WHERE attempt_id=?",
                    (attempt["attempt_id"],),
                )
                payload = {
                    "attempt_id": attempt["attempt_id"],
                    "batch_no": attempt["batch_no"],
                    "scheme_version": attempt["scheme_version"],
                    "unknown_groups": [
                        {"view_id": group["view_id"], "group": json.loads(group["group_key_json"])}
                        for group in groups
                    ],
                }
                self.connection.execute(
                    "INSERT INTO recovery_log(kind, details_json, created_at) VALUES ('uncommitted_batch', ?, ?)",
                    (canonical_json(payload), _utc_now()),
                )
                unknown_attempts.append(payload)
        last = self.last_committed_batch()
        return {
            "recovered": bool(unknown_attempts),
            "unknown_attempts": unknown_attempts,
            "resume_from_cursor": int(last["cursor"]) if last is not None else 0,
            "resume_from_batch": int(last["batch_no"]) if last is not None else 0,
            "exit_report_available": False,
            "message": (
                "uncommitted batch groups are marked unknown and must be rerun"
                if unknown_attempts
                else "no uncommitted batch was found"
            ),
        }

    def unknown_groups(self) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM attempt_groups WHERE status='unknown' ORDER BY batch_no, view_id, group_sort"
            )
        ]

    # ------------------------------------------------------------------
    # Batch commit
    # ------------------------------------------------------------------
    def commit_batch(
        self,
        *,
        attempt_id: int,
        scheme_version: int,
        batch_result: Mapping[str, Any],
        materialized_result: Mapping[str, Any],
    ) -> dict[str, Any]:
        materialized_json = canonical_json(materialized_result)
        materialized_digest = sha256_hex(materialized_json.encode("utf-8"))
        events_json = canonical_json(batch_result["events"])
        groups_json = canonical_json(batch_result["groups"])
        result_payload = {
            "batch_no": batch_result["batch_no"],
            "cursor": batch_result["cursor"],
            "events_digest": sha256_hex(events_json.encode("utf-8")),
            "groups_digest": sha256_hex(groups_json.encode("utf-8")),
            "materialized_digest": materialized_digest,
            "path": batch_result["path"],
            "scheme_fingerprint": batch_result["scheme_fingerprint"],
        }
        batch_digest = object_digest(result_payload)
        snapshot_payload = {
            "format_version": SNAPSHOT_FORMAT_VERSION,
            "scheme_version": scheme_version,
            "cursor": int(batch_result["cursor"]),
            "batch_no": int(batch_result["batch_no"]),
            "materialized": materialized_result,
        }
        snapshot_json = canonical_json(snapshot_payload)
        snapshot_digest = sha256_hex(snapshot_json.encode("utf-8"))

        with self.connection:
            attempt = self.connection.execute(
                "SELECT * FROM batch_attempts WHERE attempt_id=?", (attempt_id,)
            ).fetchone()
            if attempt is None:
                raise ColmvError(
                    "E_LEDGER",
                    "batch attempt does not exist",
                    details={"attempt_id": attempt_id},
                )
            old_digest_row = self.connection.execute(
                "SELECT value FROM meta WHERE key='content_digest'"
            ).fetchone()
            old_digest = old_digest_row["value"] if old_digest_row is not None else GENESIS_DIGEST
            content_digest = sha256_hex((old_digest + batch_digest).encode("utf-8"))

            self.connection.execute(
                """
                INSERT INTO batches(
                    scheme_version, batch_no, start_seq, end_seq, change_count,
                    cursor, path, decision_json, events_json, groups_json,
                    result_digest, materialized_digest, materialized_json,
                    fixed_version, content_digest, is_stale
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    scheme_version,
                    batch_result["batch_no"],
                    batch_result["start_seq"],
                    batch_result["end_seq"],
                    batch_result["change_count"],
                    batch_result["cursor"],
                    batch_result["path"],
                    canonical_json(batch_result["decision"]),
                    events_json,
                    groups_json,
                    batch_digest,
                    materialized_digest,
                    materialized_json,
                    batch_result["fixed_point"]["version"],
                    content_digest,
                ),
            )

            self.connection.execute(
                "DELETE FROM group_states WHERE scheme_version=? AND batch_no=?",
                (scheme_version, batch_result["batch_no"]),
            )
            self.connection.execute(
                "DELETE FROM events WHERE scheme_version=? AND batch_no=?",
                (scheme_version, batch_result["batch_no"]),
            )
            for group in batch_result["groups"]:
                view_id = group["view_id"]
                group_sort = object_digest(group["group_key_typed"])
                self.connection.execute(
                    """
                    INSERT INTO group_states(
                        scheme_version, batch_no, view_id, group_sort,
                        group_key_json, state_json, status, is_stale
                    ) VALUES (?, ?, ?, ?, ?, ?, 'committed', 0)
                    """,
                    (
                        scheme_version,
                        batch_result["batch_no"],
                        view_id,
                        group_sort,
                        canonical_json(group["group_key_typed"]),
                        canonical_json(group),
                    ),
                )
            for event in batch_result["events"]:
                view_id = event.get("view_id")
                typed_key = event.get("group_key", event.get("evicted_group_key"))
                group_sort = object_digest(typed_key) if typed_key is not None else None
                self.connection.execute(
                    """
                    INSERT INTO events(
                        scheme_version, batch_no, event_seq, kind,
                        view_id, group_sort, event_json, is_stale
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                    """,
                    (
                        scheme_version,
                        batch_result["batch_no"],
                        event["event_seq"],
                        event["kind"],
                        view_id,
                        group_sort,
                        canonical_json(event),
                    ),
                )
            self.connection.execute(
                "UPDATE batch_attempts SET status='committed' WHERE attempt_id=?",
                (attempt_id,),
            )
            self.connection.execute(
                "UPDATE attempt_groups SET status='committed' WHERE attempt_id=?",
                (attempt_id,),
            )
            self.connection.executemany(
                "UPDATE meta SET value=? WHERE key=?",
                [
                    (str(batch_result["batch_no"]), "last_batch"),
                    (str(batch_result["cursor"]), "cursor"),
                    (content_digest, "content_digest"),
                ],
            )
            self.connection.execute(
                "UPDATE schemes SET created_batch=? WHERE scheme_version=?",
                (batch_result["batch_no"], scheme_version),
            )
            self.connection.execute(
                """
                INSERT INTO snapshot(
                    singleton, format_version, scheme_version,
                    snapshot_digest, payload_json, updated_at
                ) VALUES (1, ?, ?, ?, ?, ?)
                ON CONFLICT(singleton) DO UPDATE SET
                    format_version=excluded.format_version,
                    scheme_version=excluded.scheme_version,
                    snapshot_digest=excluded.snapshot_digest,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (
                    SNAPSHOT_FORMAT_VERSION,
                    scheme_version,
                    snapshot_digest,
                    snapshot_json,
                    _utc_now(),
                ),
            )
        return {
            "batch_digest": batch_digest,
            "content_digest": content_digest,
            "materialized_digest": materialized_digest,
            "snapshot_digest": snapshot_digest,
        }

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def last_committed_batch(self, scheme_version: int | None = None) -> dict[str, Any] | None:
        if scheme_version is None:
            current = self.current_scheme()
            if current is None:
                return None
            scheme_version = int(current["scheme_version"])
        row = self.connection.execute(
            """
            SELECT * FROM batches
            WHERE scheme_version=? AND is_stale=0
            ORDER BY batch_no DESC LIMIT 1
            """,
            (scheme_version,),
        ).fetchone()
        return dict(row) if row is not None else None

    def committed_batches(self, scheme_version: int) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.connection.execute(
                """
                SELECT * FROM batches
                WHERE scheme_version=? AND is_stale=0
                ORDER BY batch_no
                """,
                (scheme_version,),
            )
        ]

    def group_states_for_batch(self, scheme_version: int, batch_no: int) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.connection.execute(
                """
                SELECT * FROM group_states
                WHERE scheme_version=? AND batch_no=? AND is_stale=0
                ORDER BY view_id, group_sort
                """,
                (scheme_version, batch_no),
            )
        ]

    def events_for_batch(self, scheme_version: int, batch_no: int) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.connection.execute(
                """
                SELECT * FROM events
                WHERE scheme_version=? AND batch_no=? AND is_stale=0
                ORDER BY event_seq
                """,
                (scheme_version, batch_no),
            )
        ]

    def ledger_meta(self) -> dict[str, str]:
        return {
            row["key"]: row["value"]
            for row in self.connection.execute("SELECT key, value FROM meta ORDER BY key")
        }

    # ------------------------------------------------------------------
    # Session snapshot verification / whole replacement
    # ------------------------------------------------------------------
    def verify_snapshot(self) -> dict[str, Any]:
        row = self.connection.execute("SELECT * FROM snapshot WHERE singleton=1").fetchone()
        current = self.current_scheme()
        meta_snapshot_version = self.ledger_meta().get("snapshot_format_version")
        parsed_payload: dict[str, Any] | None = None
        reason: str | None = None
        if row is None:
            reason = "missing"
        elif meta_snapshot_version != str(SNAPSHOT_FORMAT_VERSION):
            reason = "meta_snapshot_format_version_mismatch"
        elif int(row["format_version"]) != SNAPSHOT_FORMAT_VERSION:
            reason = "format_version_mismatch"
        elif sha256_hex(row["payload_json"].encode("utf-8")) != row["snapshot_digest"]:
            reason = "digest_mismatch"
        elif current is None or int(row["scheme_version"]) != int(current["scheme_version"]):
            reason = "scheme_version_mismatch"
        else:
            try:
                parsed_payload = json.loads(row["payload_json"])
            except json.JSONDecodeError:
                reason = "payload_json_invalid"
            if reason is None and (
                not isinstance(parsed_payload, dict)
                or parsed_payload.get("format_version") != SNAPSHOT_FORMAT_VERSION
                or parsed_payload.get("scheme_version") != int(row["scheme_version"])
            ):
                reason = "payload_version_mismatch"

        if reason is None:
            return {
                "valid": True,
                "replaced": False,
                "scheme_version": int(row["scheme_version"]),
                "snapshot_digest": row["snapshot_digest"],
            }

        replacement = self._replace_snapshot_from_latest_batch()
        details = {"reason": reason, "replacement": replacement}
        with self.connection:
            self.connection.execute(
                "INSERT INTO recovery_log(kind, details_json, created_at) VALUES ('snapshot_replaced', ?, ?)",
                (canonical_json(details), _utc_now()),
            )
        self.snapshot_replaced = True
        self.snapshot_replacement_details = details
        return {"valid": False, "replaced": True, **details}

    def _replace_snapshot_from_latest_batch(self) -> dict[str, Any]:
        current = self.current_scheme()
        if current is None:
            with self.connection:
                self.connection.execute("DELETE FROM snapshot")
            return {"available": False, "reason": "no_current_scheme"}
        latest = self.last_committed_batch(int(current["scheme_version"]))
        with self.connection:
            self.connection.execute("DELETE FROM snapshot")
            if latest is None:
                return {"available": False, "reason": "no_committed_batch"}
            materialized = json.loads(latest["materialized_json"])
            payload = {
                "format_version": SNAPSHOT_FORMAT_VERSION,
                "scheme_version": int(current["scheme_version"]),
                "cursor": int(latest["cursor"]),
                "batch_no": int(latest["batch_no"]),
                "materialized": materialized,
            }
            payload_json = canonical_json(payload)
            digest = sha256_hex(payload_json.encode("utf-8"))
            self.connection.execute(
                """
                INSERT INTO snapshot(
                    singleton, format_version, scheme_version,
                    snapshot_digest, payload_json, updated_at
                ) VALUES (1, ?, ?, ?, ?, ?)
                """,
                (
                    SNAPSHOT_FORMAT_VERSION,
                    int(current["scheme_version"]),
                    digest,
                    payload_json,
                    _utc_now(),
                ),
            )
        return {
            "available": True,
            "scheme_version": int(current["scheme_version"]),
            "batch_no": int(latest["batch_no"]),
            "cursor": int(latest["cursor"]),
            "snapshot_digest": digest,
        }

    def snapshot(self) -> dict[str, Any] | None:
        row = self.connection.execute("SELECT * FROM snapshot WHERE singleton=1").fetchone()
        if row is None:
            return None
        return {
            "format_version": int(row["format_version"]),
            "scheme_version": int(row["scheme_version"]),
            "snapshot_digest": row["snapshot_digest"],
            "payload": json.loads(row["payload_json"]),
        }

    def record_invalidation(self, plan: Mapping[str, Any], *, new_scheme_version: int, old_scheme_version: int | None) -> None:
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO invalidation_runs(
                    old_scheme_version, new_scheme_version, plan_json, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    old_scheme_version,
                    new_scheme_version,
                    canonical_json(plan),
                    _utc_now(),
                ),
            )

    def verify_content_chain(self) -> dict[str, Any]:
        rows = [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM batches ORDER BY scheme_version, batch_no"
            )
        ]
        scheme_fingerprints = {
            int(row["scheme_version"]): row["fingerprint"]
            for row in self.connection.execute(
                "SELECT scheme_version, fingerprint FROM schemes"
            )
        }
        digest = GENESIS_DIGEST
        for row in rows:
            events_digest = sha256_hex(row["events_json"].encode("utf-8"))
            groups_digest = sha256_hex(row["groups_json"].encode("utf-8"))
            materialized_digest = sha256_hex(row["materialized_json"].encode("utf-8"))
            if materialized_digest != row["materialized_digest"]:
                return {
                    "valid": False,
                    "first_scheme_version": int(row["scheme_version"]),
                    "first_batch": int(row["batch_no"]),
                    "field": "materialized_digest",
                    "expected": row["materialized_digest"],
                    "actual": materialized_digest,
                }
            result_payload = {
                "batch_no": row["batch_no"],
                "cursor": row["cursor"],
                "events_digest": events_digest,
                "groups_digest": groups_digest,
                "materialized_digest": row["materialized_digest"],
                "path": row["path"],
                "scheme_fingerprint": scheme_fingerprints.get(int(row["scheme_version"]), ""),
            }
            batch_digest = object_digest(result_payload)
            if batch_digest != row["result_digest"]:
                return {
                    "valid": False,
                    "first_scheme_version": int(row["scheme_version"]),
                    "first_batch": int(row["batch_no"]),
                    "field": "result_digest",
                    "expected": row["result_digest"],
                    "actual": batch_digest,
                }
            digest = sha256_hex((digest + batch_digest).encode("utf-8"))
            if digest != row["content_digest"]:
                return {
                    "valid": False,
                    "first_scheme_version": int(row["scheme_version"]),
                    "first_batch": int(row["batch_no"]),
                    "field": "content_digest",
                    "expected": row["content_digest"],
                    "actual": digest,
                }
        stored = self.ledger_meta().get("content_digest", GENESIS_DIGEST)
        return {"valid": stored == digest, "content_digest": digest, "stored": stored}
