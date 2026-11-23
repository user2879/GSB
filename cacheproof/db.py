"""SQLite schema and ledger-state digest helpers."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator

from .canonical import digest_value, sha256_hex, utf8_key
from .errors import DiagnosticError


LEDGER_FORMAT_VERSION = 1
SCHEMA_VERSION = 1


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ledger_state (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    format_version INTEGER NOT NULL,
    batch_id INTEGER NOT NULL,
    segment_set_version INTEGER NOT NULL,
    completed_cursor INTEGER NOT NULL,
    spec_digest TEXT NOT NULL,
    content_digest TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS segments (
    segment_id TEXT PRIMARY KEY,
    capacity_bytes INTEGER NOT NULL CHECK (capacity_bytes > 0),
    used_bytes INTEGER NOT NULL CHECK (used_bytes >= 0),
    entry_count INTEGER NOT NULL CHECK (entry_count >= 0),
    segment_version INTEGER NOT NULL,
    proof_json TEXT NOT NULL,
    proof_digest TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS entries (
    segment_id TEXT NOT NULL,
    entry_key TEXT NOT NULL,
    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
    content_seed TEXT NOT NULL,
    digest TEXT NOT NULL,
    file_relpath TEXT NOT NULL UNIQUE,
    committed_cursor INTEGER NOT NULL,
    PRIMARY KEY (segment_id, entry_key),
    FOREIGN KEY (segment_id) REFERENCES segments(segment_id) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS operations (
    operation_seq INTEGER PRIMARY KEY AUTOINCREMENT,
    operation_id TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    result_code TEXT NOT NULL,
    segment_id TEXT,
    entry_key TEXT,
    size_bytes INTEGER,
    content_seed TEXT,
    expected_digest TEXT,
    file_relpath TEXT,
    old_file_relpath TEXT,
    required_bytes INTEGER,
    available_bytes INTEGER,
    committed_cursor INTEGER,
    started_monotonic_ns INTEGER NOT NULL,
    finished_monotonic_ns INTEGER,
    elapsed_ns INTEGER,
    details_json TEXT NOT NULL DEFAULT '{}'
);

CREATE UNIQUE INDEX IF NOT EXISTS operations_committed_cursor_unique
ON operations(committed_cursor) WHERE committed_cursor IS NOT NULL;

CREATE INDEX IF NOT EXISTS operations_status_seq
ON operations(status, operation_seq);

CREATE TABLE IF NOT EXISTS unknown_entries (
    segment_id TEXT NOT NULL,
    entry_key TEXT NOT NULL,
    operation_id TEXT NOT NULL DEFAULT '',
    expected_digest TEXT,
    observed_digest TEXT,
    observed_size_bytes INTEGER,
    observed_file_relpath TEXT NOT NULL DEFAULT '',
    reason TEXT NOT NULL,
    first_seen_monotonic_ns INTEGER NOT NULL,
    PRIMARY KEY (segment_id, entry_key, operation_id, observed_file_relpath)
);

CREATE TABLE IF NOT EXISTS read_views (
    view_id TEXT PRIMARY KEY,
    view_version INTEGER NOT NULL,
    segment_set_version INTEGER NOT NULL,
    snapshot_json TEXT NOT NULL,
    snapshot_digest TEXT NOT NULL,
    last_differences_json TEXT NOT NULL DEFAULT '{}',
    updated_monotonic_ns INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS spec_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def connect(database: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        str(database), timeout=30.0, isolation_level=None
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 30000")
    connection.execute("PRAGMA synchronous = FULL")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def ensure_schema(connection: sqlite3.Connection) -> None:
    current = connection.execute("PRAGMA user_version").fetchone()[0]
    if current not in (0, SCHEMA_VERSION):
        raise DiagnosticError(
            "E_SNAPSHOT",
            "unsupported SQLite ledger schema version",
            field="ledger.user_version",
            details={"found": current, "supported": SCHEMA_VERSION},
        )
    connection.executescript(SCHEMA_SQL)
    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


@contextmanager
def transaction(connection: sqlite3.Connection, *, immediate: bool = True) -> Iterator[None]:
    connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
    try:
        yield
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    else:
        connection.execute("COMMIT")


def get_ledger_state(connection: sqlite3.Connection) -> sqlite3.Row | None:
    return connection.execute(
        "SELECT * FROM ledger_state WHERE singleton = 1"
    ).fetchone()


def _views_payload(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        "SELECT view_id, view_version, segment_set_version, snapshot_digest, last_differences_json FROM read_views"
    ).fetchall()
    payload = []
    for row in rows:
        payload.append(
            {
                "view_id": row["view_id"],
                "view_version": row["view_version"],
                "segment_set_version": row["segment_set_version"],
                "snapshot_digest": row["snapshot_digest"],
                "last_differences_digest": "sha256:"
                + sha256_hex(row["last_differences_json"].encode("utf-8")),
            }
        )
    payload.sort(key=lambda item: utf8_key(item["view_id"]))
    return payload


def ledger_content_digest(connection: sqlite3.Connection) -> tuple[str, dict[str, Any]]:
    state = get_ledger_state(connection)
    if state is None:
        raise DiagnosticError(
            "E_NOT_FOUND", "SQLite ledger has not been initialized", field="ledger_state"
        )

    segments = []
    for row in connection.execute("SELECT * FROM segments").fetchall():
        segments.append(
            {
                "segment_id": row["segment_id"],
                "capacity_bytes": row["capacity_bytes"],
                "used_bytes": row["used_bytes"],
                "entry_count": row["entry_count"],
                "segment_version": row["segment_version"],
                "proof_digest": row["proof_digest"],
            }
        )
    segments.sort(key=lambda item: utf8_key(item["segment_id"]))

    entries = []
    for row in connection.execute("SELECT * FROM entries").fetchall():
        entries.append(
            {
                "segment_id": row["segment_id"],
                "entry_key": row["entry_key"],
                "size_bytes": row["size_bytes"],
                "content_seed": row["content_seed"],
                "digest": row["digest"],
                "file_relpath": row["file_relpath"],
                "committed_cursor": row["committed_cursor"],
            }
        )
    entries.sort(key=lambda item: (utf8_key(item["segment_id"]), utf8_key(item["entry_key"])))

    unknown_entries = []
    for row in connection.execute("SELECT * FROM unknown_entries").fetchall():
        unknown_entries.append(
            {
                "segment_id": row["segment_id"],
                "entry_key": row["entry_key"],
                "operation_id": row["operation_id"],
                "expected_digest": row["expected_digest"],
                "observed_digest": row["observed_digest"],
                "observed_size_bytes": row["observed_size_bytes"],
                "observed_file_relpath": row["observed_file_relpath"],
                "reason": row["reason"],
            }
        )
    unknown_entries.sort(
        key=lambda item: (
            utf8_key(item["segment_id"]),
            utf8_key(item["entry_key"]),
            item["operation_id"],
        )
    )

    spec_digest_row = connection.execute(
        "SELECT value FROM spec_meta WHERE key = 'spec_digest'"
    ).fetchone()
    payload = {
        "format_version": state["format_version"],
        "batch_id": state["batch_id"],
        "segment_set_version": state["segment_set_version"],
        "completed_cursor": state["completed_cursor"],
        "spec_digest": state["spec_digest"],
        "segments": segments,
        "entries": entries,
        "views": _views_payload(connection),
        "unknown_entries": unknown_entries,
        "stored_spec_digest": None if spec_digest_row is None else spec_digest_row["value"],
    }
    return digest_value(payload), payload


def refresh_ledger_digest(
    connection: sqlite3.Connection,
    *,
    increment_batch: bool = True,
) -> str:
    state = get_ledger_state(connection)
    if state is None:
        raise DiagnosticError("E_NOT_FOUND", "ledger state is missing", field="ledger_state")
    if increment_batch:
        connection.execute(
            "UPDATE ledger_state SET batch_id = batch_id + 1 WHERE singleton = 1"
        )
    digest, _ = ledger_content_digest(connection)
    connection.execute(
        "UPDATE ledger_state SET content_digest = ? WHERE singleton = 1", (digest,)
    )
    return digest


def verify_ledger_digest(connection: sqlite3.Connection) -> dict[str, Any]:
    state = get_ledger_state(connection)
    if state is None:
        raise DiagnosticError(
            "E_NOT_FOUND", "SQLite ledger has not been initialized", field="ledger_state"
        )
    if state["format_version"] != LEDGER_FORMAT_VERSION:
        raise DiagnosticError(
            "E_SNAPSHOT",
            "unsupported ledger format version",
            field="ledger.format_version",
            details={
                "stored": state["format_version"],
                "supported": LEDGER_FORMAT_VERSION,
            },
        )
    computed, _ = ledger_content_digest(connection)
    if state["content_digest"] != computed:
        raise DiagnosticError(
            "E_SNAPSHOT",
            "SQLite ledger content digest mismatch",
            field="ledger.content_digest",
            details={
                "stored_digest": state["content_digest"],
                "computed_digest": computed,
                "batch_id": state["batch_id"],
            },
        )
    return {
        "format_version": state["format_version"],
        "batch_id": state["batch_id"],
        "segment_set_version": state["segment_set_version"],
        "completed_cursor": state["completed_cursor"],
        "content_digest": computed,
    }


def loads_json(value: str, *, field: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise DiagnosticError(
            "E_SNAPSHOT",
            "stored JSON is not valid",
            field=field,
            details={"error": str(exc)},
        ) from exc
