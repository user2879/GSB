"""SQLite-backed diagnostic session and observation ledger."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .errors import WorkbenchError, fail
from .logging_utils import JsonlLogger, utc_now
from .query import Query, is_slow, parse_select


INTERNAL_PREFIX = "__sw_"
FORMAT_VERSION = 1
CLOCK_BASIS = "local monotonic counter, perf_counter_ns()/1_000_000 milliseconds"
DEFAULT_THRESHOLD_MS = 2
DEFAULT_THRESHOLD_ROWS = 1000
DEFAULT_REPLAY_TIMEOUT_MS = 250
DEFAULT_REPLAY_BUDGET_MS = 3000

VERSION_FIELDS = ("data_version", "index_version", "stats_version")

REQUIRED_META = (
    "format_version",
    "session_id",
    "session_name",
    "data_version",
    "index_version",
    "stats_version",
    "schema_json",
    "schema_hash",
    "index_set_json",
    "index_set_hash",
    "clock_basis",
    "slow_ms_threshold",
    "slow_scan_threshold",
    "replay_timeout_ms",
    "replay_budget_ms",
)


INTERNAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS __sw_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS __sw_version_guard (
    field TEXT PRIMARY KEY,
    high_water INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS __sw_change_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    source_version_field TEXT NOT NULL,
    old_value INTEGER NOT NULL,
    new_value INTEGER NOT NULL,
    affected_query_ids_json TEXT NOT NULL,
    affected_tables_json TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS __sw_queries (
    query_id TEXT PRIMARY KEY,
    canonical_sql TEXT NOT NULL,
    tables_json TEXT NOT NULL,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    execution_count INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS __sw_executions (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id TEXT NOT NULL UNIQUE,
    query_id TEXT NOT NULL,
    canonical_sql TEXT NOT NULL,
    executed_sql TEXT NOT NULL,
    kind TEXT NOT NULL,
    candidate_id TEXT,
    started_mono_ms REAL NOT NULL,
    ended_mono_ms REAL NOT NULL,
    elapsed_ms REAL NOT NULL,
    scanned_rows INTEGER NOT NULL,
    result_rows INTEGER NOT NULL,
    outcome TEXT NOT NULL,
    error_code TEXT,
    error_message TEXT,
    timeout_ms REAL,
    data_version INTEGER NOT NULL,
    index_version INTEGER NOT NULL,
    stats_version INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS __sw_exec_query_idx ON __sw_executions(query_id, seq);
CREATE TABLE IF NOT EXISTS __sw_candidates (
    candidate_id TEXT PRIMARY KEY,
    query_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    rewritten_sql TEXT NOT NULL,
    rationale TEXT NOT NULL,
    referenced_indexes_json TEXT NOT NULL,
    index_version INTEGER NOT NULL,
    stats_version INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'generated',
    elapsed_ms REAL,
    result_rows INTEGER,
    execution_id TEXT,
    error_code TEXT,
    reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS __sw_conclusions (
    query_id TEXT PRIMARY KEY,
    conclusion_id TEXT NOT NULL UNIQUE,
    original_execution_id TEXT NOT NULL,
    original_elapsed_ms REAL NOT NULL,
    original_scanned_rows INTEGER NOT NULL,
    original_result_rows INTEGER NOT NULL,
    candidate_ids_json TEXT NOT NULL,
    optimal_candidate_id TEXT,
    optimal_elapsed_ms REAL,
    speedup REAL,
    threshold_ms INTEGER NOT NULL,
    threshold_rows INTEGER NOT NULL,
    data_version INTEGER NOT NULL,
    index_version INTEGER NOT NULL,
    stats_version INTEGER NOT NULL,
    is_stale INTEGER NOT NULL DEFAULT 0,
    stale_event_id INTEGER,
    stale_reason TEXT,
    stale_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS __sw_stats_snapshots (
    version INTEGER PRIMARY KEY,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS __sw_replay_validations (
    validation_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    match INTEGER NOT NULL,
    report_json TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class Execution:
    execution_id: str
    query_id: str
    canonical_sql: str
    executed_sql: str
    kind: str
    candidate_id: str | None
    started_mono_ms: float
    ended_mono_ms: float
    elapsed_ms: float
    scanned_rows: int
    result_rows: int
    outcome: str
    error_code: str | None
    error_message: str | None
    timeout_ms: float | None
    data_version: int
    index_version: int
    stats_version: int
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "query_id": self.query_id,
            "canonical_sql": self.canonical_sql,
            "executed_sql": self.executed_sql,
            "kind": self.kind,
            "candidate_id": self.candidate_id,
            "started_mono_ms": self.started_mono_ms,
            "ended_mono_ms": self.ended_mono_ms,
            "elapsed_ms": self.elapsed_ms,
            "scanned_rows": self.scanned_rows,
            "result_rows": self.result_rows,
            "outcome": self.outcome,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "timeout_ms": self.timeout_ms,
            "data_version": self.data_version,
            "index_version": self.index_version,
            "stats_version": self.stats_version,
            "created_at": self.created_at,
        }


def _quote_identifier(identifier: str) -> str:
    if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", identifier):
        raise fail("E_DATA", f"invalid SQLite identifier: {identifier!r}")
    if identifier.startswith(INTERNAL_PREFIX):
        raise fail("E_DATA", "internal workbench objects cannot be modified")
    return '"' + identifier.replace('"', '""') + '"'


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def _split_sql_statements(script: str) -> list[str]:
    statements: list[str] = []
    current: list[str] = []
    for char in script:
        current.append(char)
        text = "".join(current)
        if sqlite3.complete_statement(text):
            statement = text.strip()
            if statement and statement != ";":
                statements.append(statement)
            current = []
    tail = "".join(current).strip()
    if tail:
        if not sqlite3.complete_statement(tail + ";"):
            raise fail("E_DATA", "incomplete SQL statement in schema/data file")
        if tail != ";":
            statements.append(tail + ("" if tail.endswith(";") else ";"))
    return statements


def _result_dict(cursor: sqlite3.Cursor, row: Sequence[Any]) -> dict[str, Any]:
    return {description[0]: row[index] for index, description in enumerate(cursor.description)}


class Session:
    """An open diagnostic session stored in one local SQLite file."""

    def __init__(self, path: str | Path, connection: sqlite3.Connection, logger: JsonlLogger | None = None) -> None:
        self.path = Path(path).resolve()
        self.conn = connection
        self.conn.row_factory = sqlite3.Row
        self.logger = logger or JsonlLogger(self.path.with_name(self.path.name + ".log.jsonl"))
        self._closed = False

    @classmethod
    def create(
        cls,
        path: str | Path,
        name: str | None = None,
        *,
        overwrite: bool = False,
        logger: JsonlLogger | None = None,
    ) -> "Session":
        target = Path(path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size > 0 and not overwrite:
            raise fail("E_SESSION", f"session file already exists: {target}")
        if overwrite and target.exists():
            target.unlink()
        connection = sqlite3.connect(str(target))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        session_logger = logger or JsonlLogger(target.with_name(target.name + ".log.jsonl"))
        try:
            connection.executescript(INTERNAL_SCHEMA)
            now = utc_now()
            session_name = name or target.stem
            initial = {
                "format_version": str(FORMAT_VERSION),
                "session_id": "s_" + uuid.uuid4().hex,
                "session_name": session_name,
                "data_version": "0",
                "index_version": "0",
                "stats_version": "0",
                "schema_json": "[]",
                "schema_hash": _digest([]),
                "index_set_json": "[]",
                "index_set_hash": _digest([]),
                "clock_basis": CLOCK_BASIS,
                "slow_ms_threshold": str(DEFAULT_THRESHOLD_MS),
                "slow_scan_threshold": str(DEFAULT_THRESHOLD_ROWS),
                "replay_timeout_ms": str(DEFAULT_REPLAY_TIMEOUT_MS),
                "replay_budget_ms": str(DEFAULT_REPLAY_BUDGET_MS),
                "created_at": now,
                "updated_at": now,
            }
            with connection:
                connection.executemany(
                    "INSERT INTO __sw_meta(key, value) VALUES (?, ?)",
                    [(key, value) for key, value in initial.items()],
                )
                connection.executemany(
                    "INSERT INTO __sw_version_guard(field, high_water, updated_at) VALUES (?, 0, ?)",
                    [(field, now) for field in VERSION_FIELDS],
                )
        except Exception:
            connection.close()
            raise
        session_logger.info("session_created", "diagnostic session created", session_path=str(target), session_name=session_name)
        return cls(target, connection, logger=session_logger)

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        expected_versions: Mapping[str, int] | None = None,
        logger: JsonlLogger | None = None,
    ) -> "Session":
        target = Path(path).resolve()
        if not target.exists():
            raise fail("E_SESSION", f"session file does not exist: {target}")
        session_logger = logger or JsonlLogger(target.with_name(target.name + ".log.jsonl"))
        connection = sqlite3.connect(str(target))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        session = cls(target, connection, logger=session_logger)
        try:
            session._validate_open(expected_versions=expected_versions)
        except Exception as exc:
            connection.close()
            session._closed = True
            if isinstance(exc, WorkbenchError):
                session_logger.error(
                    "session_open_failed",
                    exc.message,
                    code=exc.code,
                    details=exc.details,
                    session_path=str(target),
                )
            raise
        session_logger.info(
            "session_opened",
            "diagnostic session opened",
            session_path=str(target),
            session_name=session.meta_value("session_name"),
            versions=session.version_info(),
        )
        return session

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.conn.commit()
        finally:
            self.conn.close()
            self._closed = True
            self.logger.info("session_closed", "diagnostic session closed", session_path=str(self.path))

    @property
    def closed(self) -> bool:
        return self._closed

    def _ensure_open(self) -> None:
        if self._closed:
            raise fail("E_SESSION", "session is closed")

    # ------------------------------------------------------------------ metadata
    def meta_value(self, key: str) -> str:
        row = self.conn.execute("SELECT value FROM __sw_meta WHERE key = ?", (key,)).fetchone()
        if row is None:
            raise fail("E_SESSION", f"required session metadata is missing: {key}", {"field": key})
        return str(row["value"])

    def _meta_int(self, key: str) -> int:
        raw = self.meta_value(key)
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise fail("E_SESSION", f"session metadata is not an integer: {key}", {"field": key, "value": raw}) from exc
        return value

    def set_meta(self, key: str, value: str) -> None:
        self._ensure_open()
        with self.conn:
            self.conn.execute(
                "INSERT INTO __sw_meta(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, str(value)),
            )
            self.conn.execute("UPDATE __sw_meta SET value = ? WHERE key = 'updated_at'", (utc_now(),))

    def _set_meta_no_tx(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO __sw_meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, str(value)),
        )

    def version_info(self) -> dict[str, Any]:
        return {
            "data_version": self._meta_int("data_version"),
            "index_version": self._meta_int("index_version"),
            "stats_version": self._meta_int("stats_version"),
            "schema_hash": self.meta_value("schema_hash"),
            "index_set_hash": self.meta_value("index_set_hash"),
        }

    def session_info(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "session_id": self.meta_value("session_id"),
            "session_name": self.meta_value("session_name"),
            "format_version": self._meta_int("format_version"),
            "clock_basis": self.meta_value("clock_basis"),
            "thresholds": self.thresholds(),
            "replay_defaults": {
                "timeout_ms": self._meta_int("replay_timeout_ms"),
                "budget_ms": self._meta_int("replay_budget_ms"),
            },
            "versions": self.version_info(),
            "tables": [item["name"] for item in self._schema_snapshot()],
            "indexes": self.index_snapshot(),
        }

    def thresholds(self) -> dict[str, int]:
        return {
            "threshold_ms": self._meta_int("slow_ms_threshold"),
            "threshold_rows": self._meta_int("slow_scan_threshold"),
        }

    def set_thresholds(self, threshold_ms: int, threshold_rows: int) -> dict[str, int]:
        if not isinstance(threshold_ms, int) or isinstance(threshold_ms, bool) or threshold_ms < 0:
            raise fail("E_SESSION", "threshold_ms must be a non-negative integer")
        if not isinstance(threshold_rows, int) or isinstance(threshold_rows, bool) or threshold_rows < 0:
            raise fail("E_SESSION", "threshold_rows must be a non-negative integer")
        self._ensure_open()
        with self.conn:
            self._set_meta_no_tx("slow_ms_threshold", str(threshold_ms))
            self._set_meta_no_tx("slow_scan_threshold", str(threshold_rows))
            self._set_meta_no_tx("updated_at", utc_now())
        self.logger.info(
            "thresholds_changed",
            "slow-query thresholds changed; ledger observations remain raw",
            threshold_ms=threshold_ms,
            threshold_rows=threshold_rows,
        )
        return self.thresholds()

    def _validate_open(self, expected_versions: Mapping[str, int] | None = None) -> None:
        try:
            rows = {row["key"]: row["value"] for row in self.conn.execute("SELECT key, value FROM __sw_meta")}
        except sqlite3.Error as exc:
            raise fail("E_SESSION", "file is not a valid diagnostic session", {"error": str(exc)}) from exc
        missing = [key for key in REQUIRED_META if key not in rows]
        if missing:
            raise fail("E_SESSION", "required session metadata is missing", {"missing_fields": missing})
        try:
            format_version = int(rows["format_version"])
        except (TypeError, ValueError) as exc:
            raise fail("E_SESSION", "session format version is not an integer") from exc
        if format_version != FORMAT_VERSION:
            raise fail(
                "E_SESSION",
                "unsupported session format version",
                {"expected": FORMAT_VERSION, "actual": rows["format_version"]},
            )
        try:
            guards = {
                row["field"]: int(row["high_water"])
                for row in self.conn.execute("SELECT field, high_water FROM __sw_version_guard")
            }
        except sqlite3.Error as exc:
            raise fail("E_SESSION", "session version guard is unreadable", {"error": str(exc)}) from exc
        rollback: list[dict[str, Any]] = []
        for field in VERSION_FIELDS:
            if field not in guards:
                raise fail("E_SESSION", f"version high-water mark is missing: {field}", {"field": field})
            try:
                current = int(rows[field])
            except (TypeError, ValueError) as exc:
                raise fail("E_SESSION", f"version is not an integer: {field}", {"field": field, "value": rows[field]}) from exc
            if current < 0:
                raise fail("E_SESSION", f"version must be non-negative: {field}", {"field": field, "value": current})
            if current < guards[field]:
                rollback.append({"field": field, "current": current, "high_water": guards[field]})
        if rollback:
            raise fail("E_SESSION", "session version rollback detected", {"rollbacks": rollback})
        for field in VERSION_FIELDS:
            current = int(rows[field])
            if current > guards[field]:
                with self.conn:
                    self.conn.execute(
                        "UPDATE __sw_version_guard SET high_water = ?, updated_at = ? WHERE field = ?",
                        (current, utc_now(), field),
                    )
        if expected_versions:
            differences = []
            for field, expected in expected_versions.items():
                if field not in VERSION_FIELDS:
                    continue
                actual = int(rows[field])
                if actual != int(expected):
                    differences.append({"field": field, "expected": int(expected), "actual": actual})
            if differences:
                raise fail("E_VERSION", "session version mismatch", {"differences": differences})
        actual_schema = self._schema_snapshot()
        actual_schema_hash = _digest(actual_schema)
        if actual_schema_hash != rows["schema_hash"]:
            raise fail(
                "E_VERSION",
                "table schema does not match persisted schema snapshot",
                {"field": "schema_hash", "stored": rows["schema_hash"], "actual": actual_schema_hash},
            )
        actual_indexes = self.index_snapshot()
        actual_index_hash = _digest(actual_indexes)
        if actual_index_hash != rows["index_set_hash"]:
            raise fail(
                "E_VERSION",
                "index set does not match persisted index snapshot",
                {
                    "field": "index_set_hash",
                    "stored": rows["index_set_hash"],
                    "actual": actual_index_hash,
                },
            )
        stats_version = int(rows["stats_version"])
        if stats_version > 0:
            snapshot_row = self.conn.execute(
                "SELECT version FROM __sw_stats_snapshots WHERE version = ?", (stats_version,)
            ).fetchone()
            if snapshot_row is None:
                raise fail(
                    "E_SESSION",
                    "statistics snapshot is missing for the recorded stats_version",
                    {"field": "stats_version", "version": stats_version},
                )
        try:
            json.loads(rows["schema_json"])
            json.loads(rows["index_set_json"])
        except json.JSONDecodeError as exc:
            raise fail("E_SESSION", "session metadata JSON is damaged", {"error": str(exc)}) from exc

    # ----------------------------------------------------------------- versions
    def _bump_version_no_tx(self, field: str) -> tuple[int, int]:
        if field not in VERSION_FIELDS:
            raise ValueError(field)
        current = self._meta_int(field)
        guard_row = self.conn.execute(
            "SELECT high_water FROM __sw_version_guard WHERE field = ?", (field,)
        ).fetchone()
        if guard_row is None:
            raise fail("E_SESSION", f"version high-water mark is missing: {field}", {"field": field})
        high_water = int(guard_row["high_water"])
        new_value = max(current, high_water) + 1
        self._set_meta_no_tx(field, str(new_value))
        self.conn.execute(
            "UPDATE __sw_version_guard SET high_water = ?, updated_at = ? WHERE field = ?",
            (new_value, utc_now(), field),
        )
        self._set_meta_no_tx("updated_at", utc_now())
        return current, new_value

    def _set_initial_version_no_tx(self, field: str, value: int) -> None:
        self._set_meta_no_tx(field, str(value))
        self.conn.execute(
            "UPDATE __sw_version_guard SET high_water = ?, updated_at = ? WHERE field = ?",
            (value, utc_now(), field),
        )

    # --------------------------------------------------------------- schema/index
    def _user_tables(self) -> list[str]:
        return [
            str(row["name"])
            for row in self.conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type = 'table' AND substr(name, 1, 5) != '__sw_' "
                "AND substr(name, 1, 7) != 'sqlite_' ORDER BY name"
            )
        ]

    def _schema_snapshot(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for table in self._user_tables():
            quoted = _quote_identifier(table)
            sql_row = self.conn.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
            ).fetchone()
            columns = [
                {
                    "cid": int(row["cid"]),
                    "name": str(row["name"]),
                    "type": str(row["type"]),
                    "notnull": int(row["notnull"]),
                    "default": row["dflt_value"],
                    "pk": int(row["pk"]),
                }
                for row in self.conn.execute(f"PRAGMA table_info({quoted})")
            ]
            result.append({"name": table, "sql": sql_row["sql"] if sql_row else None, "columns": columns})
        return result

    def schema_snapshot(self) -> list[dict[str, Any]]:
        return self._schema_snapshot()

    def index_snapshot(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for table in self._user_tables():
            quoted = _quote_identifier(table)
            for row in self.conn.execute(f"PRAGMA index_list({quoted})"):
                name = str(row["name"])
                if name.startswith("sqlite_autoindex_"):
                    continue
                columns = [
                    str(info["name"])
                    for info in self.conn.execute(f"PRAGMA index_info({_quote_identifier(name)})")
                    if info["name"] is not None
                ]
                sql_row = self.conn.execute(
                    "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?", (name,)
                ).fetchone()
                result.append(
                    {
                        "name": name,
                        "table": table,
                        "columns": columns,
                        "unique": bool(row["unique"]),
                        "origin": str(row["origin"]),
                        "sql": sql_row["sql"] if sql_row else None,
                    }
                )
        return sorted(result, key=lambda item: (item["table"], item["name"]))

    def list_indexes(self) -> list[dict[str, Any]]:
        return self.index_snapshot()

    def _refresh_schema_meta_no_tx(self) -> str:
        snapshot = self._schema_snapshot()
        digest = _digest(snapshot)
        self._set_meta_no_tx("schema_json", _stable_json(snapshot))
        self._set_meta_no_tx("schema_hash", digest)
        return digest

    def _refresh_index_meta_no_tx(self) -> str:
        snapshot = self.index_snapshot()
        digest = _digest(snapshot)
        self._set_meta_no_tx("index_set_json", _stable_json(snapshot))
        self._set_meta_no_tx("index_set_hash", digest)
        return digest

    def _table_columns(self, table: str) -> list[str]:
        if table not in self._user_tables():
            raise fail("E_DATA", f"unknown table: {table}")
        quoted = _quote_identifier(table)
        return [str(row["name"]) for row in self.conn.execute(f"PRAGMA table_info({quoted})")]

    def create_domain_table(self, sql: str) -> dict[str, Any]:
        self._ensure_open()
        statement = sql.strip()
        if not re.match(r"(?is)^create\s+table\s+", statement):
            raise fail("E_PARSE", "only CREATE TABLE is allowed in this editor")
        if statement.count(";") > 1 or (statement.count(";") == 1 and not statement.endswith(";")):
            raise fail("E_PARSE", "exactly one CREATE TABLE statement is allowed")
        match = re.match(r"(?is)^create\s+table\s+(?:if\s+not\s+exists\s+)?([A-Za-z_][A-Za-z0-9_$]*)", statement)
        if not match:
            raise fail("E_PARSE", "cannot determine table name")
        table = match.group(1)
        if table.startswith(INTERNAL_PREFIX):
            raise fail("E_DATA", "internal workbench tables are reserved")
        previous_hash = self.meta_value("schema_hash")
        old_version = self._meta_int("data_version")
        try:
            with self.conn:
                self.conn.execute(statement)
                self._refresh_schema_meta_no_tx()
                new_hash = self.meta_value("schema_hash")
                if new_hash != previous_hash:
                    _, new_version = self._bump_version_no_tx("data_version")
                    self._record_invalidation_no_tx(
                        "schema_change",
                        "data_version",
                        old_version,
                        new_version,
                        [table.lower()],
                        f"table schema changed: {table}",
                    )
        except sqlite3.Error as exc:
            raise fail("E_DATA", f"CREATE TABLE failed: {exc}") from exc
        self.logger.info("table_created", "domain table created", table=table, versions=self.version_info())
        return {"table": table, "versions": self.version_info()}

    def add_index(
        self,
        table: str,
        columns: Sequence[str],
        *,
        name: str | None = None,
        unique: bool = False,
    ) -> dict[str, Any]:
        self._ensure_open()
        table = table.lower()
        valid_columns = set(self._table_columns(table))
        normalized_columns = [str(column).lower() for column in columns]
        if not normalized_columns or any(column not in valid_columns for column in normalized_columns):
            raise fail("E_DATA", "index columns must exist in the target table", {"columns": list(columns)})
        index_name = name or "idx_" + table + "_" + "_".join(normalized_columns)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", index_name) or index_name.startswith(INTERNAL_PREFIX):
            raise fail("E_DATA", f"invalid index name: {index_name!r}")
        if any(item["name"] == index_name for item in self.index_snapshot()):
            raise fail("E_DATA", f"index already exists: {index_name}")
        old_version = self._meta_int("index_version")
        column_sql = ", ".join(_quote_identifier(column) for column in normalized_columns)
        unique_sql = "UNIQUE " if unique else ""
        try:
            with self.conn:
                self.conn.execute(
                    f"CREATE {unique_sql}INDEX {_quote_identifier(index_name)} "
                    f"ON {_quote_identifier(table)} ({column_sql})"
                )
                self._refresh_index_meta_no_tx()
                _, new_version = self._bump_version_no_tx("index_version")
                self._record_invalidation_no_tx(
                    "index_change",
                    "index_version",
                    old_version,
                    new_version,
                    [table],
                    f"index added: {index_name}",
                )
        except sqlite3.Error as exc:
            raise fail("E_DATA", f"CREATE INDEX failed: {exc}") from exc
        self.logger.info("index_added", "index added", index=index_name, table=table, versions=self.version_info())
        return {
            "name": index_name,
            "table": table,
            "columns": normalized_columns,
            "versions": self.version_info(),
        }

    def drop_index(self, name: str) -> dict[str, Any]:
        self._ensure_open()
        indexes = {item["name"]: item for item in self.index_snapshot()}
        if name not in indexes:
            raise fail("E_DATA", f"unknown index: {name}")
        table = indexes[name]["table"]
        old_version = self._meta_int("index_version")
        try:
            with self.conn:
                self.conn.execute(f"DROP INDEX {_quote_identifier(name)}")
                self._refresh_index_meta_no_tx()
                _, new_version = self._bump_version_no_tx("index_version")
                self._record_invalidation_no_tx(
                    "index_change",
                    "index_version",
                    old_version,
                    new_version,
                    [table],
                    f"index dropped: {name}",
                )
        except sqlite3.Error as exc:
            raise fail("E_DATA", f"DROP INDEX failed: {exc}") from exc
        self.logger.info("index_dropped", "index dropped", index=name, table=table, versions=self.version_info())
        return {"name": name, "table": table, "versions": self.version_info()}

    def append_rows(
        self,
        table: str,
        columns: Sequence[str],
        rows: Iterable[Sequence[Any]],
    ) -> dict[str, Any]:
        self._ensure_open()
        table = table.lower()
        valid_columns = set(self._table_columns(table))
        normalized_columns = [str(column).lower() for column in columns]
        if not normalized_columns or any(column not in valid_columns for column in normalized_columns):
            raise fail("E_DATA", "insert columns must exist in the target table")
        materialized = [tuple(row) for row in rows]
        for row in materialized:
            if len(row) != len(normalized_columns):
                raise fail("E_DATA", "row width does not match column list")
        old_version = self._meta_int("data_version")
        column_sql = ", ".join(_quote_identifier(column) for column in normalized_columns)
        placeholders = ", ".join("?" for _ in normalized_columns)
        try:
            with self.conn:
                if materialized:
                    self.conn.executemany(
                        f"INSERT INTO {_quote_identifier(table)} ({column_sql}) VALUES ({placeholders})",
                        materialized,
                    )
                _, new_version = self._bump_version_no_tx("data_version")
                self._record_invalidation_no_tx(
                    "data_change",
                    "data_version",
                    old_version,
                    new_version,
                    [table],
                    f"rows appended: {len(materialized)}",
                )
        except sqlite3.Error as exc:
            raise fail("E_DATA", f"INSERT failed: {exc}") from exc
        self.logger.info(
            "rows_appended",
            "rows appended to domain table",
            table=table,
            count=len(materialized),
            versions=self.version_info(),
        )
        return {"table": table, "count": len(materialized), "versions": self.version_info()}

    def refresh_statistics(self, table: str | None = None) -> dict[str, Any]:
        self._ensure_open()
        targets = self._user_tables() if table is None else [table.lower()]
        for target in targets:
            if target not in self._user_tables():
                raise fail("E_DATA", f"unknown table: {target}")
        old_version = self._meta_int("stats_version")
        try:
            with self.conn:
                if table is None:
                    self.conn.execute("ANALYZE")
                else:
                    self.conn.execute(f"ANALYZE {_quote_identifier(table.lower())}")
                _, new_version = self._bump_version_no_tx("stats_version")
                payload = self._build_stats_payload_no_tx(new_version)
                self.conn.execute(
                    "INSERT OR REPLACE INTO __sw_stats_snapshots(version, payload_json, created_at) VALUES (?, ?, ?)",
                    (new_version, _stable_json(payload), payload["created_at"]),
                )
                self._record_invalidation_no_tx(
                    "statistics_refresh",
                    "stats_version",
                    old_version,
                    new_version,
                    targets,
                    "statistics refreshed: " + (table or "all tables"),
                )
        except sqlite3.Error as exc:
            raise fail("E_DATA", f"ANALYZE failed: {exc}") from exc
        self.logger.info("statistics_refreshed", "statistics snapshot refreshed", tables=targets)
        return {"tables": targets, "stats_version": new_version, "snapshot": payload}

    def _build_stats_payload_no_tx(self, version: int) -> dict[str, Any]:
        row_counts: dict[str, int] = {}
        for table in self._user_tables():
            row_counts[table] = int(
                self.conn.execute(f"SELECT COUNT(*) AS n FROM {_quote_identifier(table)}").fetchone()["n"]
            )
        try:
            stat_rows = [
                {"table": row["tbl"], "index": row["idx"], "stat": row["stat"]}
                for row in self.conn.execute("SELECT tbl, idx, stat FROM sqlite_stat1 ORDER BY tbl, idx")
            ]
        except sqlite3.Error:
            stat_rows = []
        return {
            "version": version,
            "created_at": utc_now(),
            "row_counts": row_counts,
            "indexes": self.index_snapshot(),
            "sqlite_stat1": stat_rows,
        }

    def stats_snapshot(self, version: int | None = None) -> dict[str, Any]:
        target = self._meta_int("stats_version") if version is None else int(version)
        row = self.conn.execute(
            "SELECT payload_json FROM __sw_stats_snapshots WHERE version = ?", (target,)
        ).fetchone()
        if row is None:
            raise fail("E_SESSION", f"statistics snapshot is missing: version {target}")
        return json.loads(row["payload_json"])

    # ------------------------------------------------------------ sample bootstrap
    def bootstrap_domain(
        self,
        schema_statements: Sequence[str],
        rows_by_table: Mapping[str, tuple[Sequence[str], Sequence[Sequence[Any]]]],
        index_specs: Sequence[Mapping[str, Any]],
        *,
        seed_statements: Sequence[str] | None = None,
        threshold_ms: int = DEFAULT_THRESHOLD_MS,
        threshold_rows: int = DEFAULT_THRESHOLD_ROWS,
    ) -> dict[str, Any]:
        self._ensure_open()
        if self._user_tables():
            raise fail("E_SESSION", "session already contains domain tables")
        if not isinstance(threshold_ms, int) or threshold_ms < 0 or not isinstance(threshold_rows, int) or threshold_rows < 0:
            raise fail("E_DATA", "sample thresholds must be non-negative integers")
        try:
            with self.conn:
                for statement in schema_statements:
                    if not re.match(r"(?is)^create\s+table\s+", statement.strip()):
                        raise fail("E_DATA", "sample schema may contain only CREATE TABLE statements")
                    self.conn.execute(statement)
                for statement in seed_statements or ():
                    if not re.match(r"(?is)^(insert|with)\s+", statement.strip()):
                        raise fail("E_DATA", "sample seed may contain only INSERT statements")
                    self.conn.execute(statement)
                for table, (columns, rows) in rows_by_table.items():
                    valid_columns = set(self._table_columns(table))
                    normalized_columns = [str(column) for column in columns]
                    if any(column not in valid_columns for column in normalized_columns):
                        raise fail("E_DATA", f"sample columns do not exist in table {table}")
                    if rows:
                        column_sql = ", ".join(_quote_identifier(column) for column in normalized_columns)
                        placeholders = ", ".join("?" for _ in normalized_columns)
                        self.conn.executemany(
                            f"INSERT INTO {_quote_identifier(table)} ({column_sql}) VALUES ({placeholders})",
                            [tuple(row) for row in rows],
                        )
                for spec in index_specs:
                    table = str(spec["table"])
                    columns = list(spec["columns"])
                    name = str(spec["name"])
                    unique = bool(spec.get("unique", False))
                    column_sql = ", ".join(_quote_identifier(column) for column in columns)
                    unique_sql = "UNIQUE " if unique else ""
                    self.conn.execute(
                        f"CREATE {unique_sql}INDEX {_quote_identifier(name)} "
                        f"ON {_quote_identifier(table)} ({column_sql})"
                    )
                self._refresh_schema_meta_no_tx()
                self._refresh_index_meta_no_tx()
                self._set_initial_version_no_tx("data_version", 1)
                self._set_initial_version_no_tx("index_version", 1)
                self._set_initial_version_no_tx("stats_version", 1)
                self._set_meta_no_tx("slow_ms_threshold", str(threshold_ms))
                self._set_meta_no_tx("slow_scan_threshold", str(threshold_rows))
                self._set_meta_no_tx("updated_at", utc_now())
                self.conn.execute("ANALYZE")
                payload = self._build_stats_payload_no_tx(1)
                self.conn.execute(
                    "INSERT OR REPLACE INTO __sw_stats_snapshots(version, payload_json, created_at) VALUES (1, ?, ?)",
                    (_stable_json(payload), payload["created_at"]),
                )
        except sqlite3.Error as exc:
            raise fail("E_DATA", f"sample bootstrap failed: {exc}") from exc
        self.logger.info(
            "samples_loaded",
            "deterministic samples loaded",
            tables=list(rows_by_table),
            indexes=[spec["name"] for spec in index_specs],
            versions=self.version_info(),
        )
        return {"versions": self.version_info(), "statistics": payload, "thresholds": self.thresholds()}

    # ------------------------------------------------------------- query ledger
    def register_query(self, query: Query) -> None:
        now = utc_now()
        with self.conn:
            self.conn.execute(
                "INSERT INTO __sw_queries(query_id, canonical_sql, tables_json, first_seen, last_seen, execution_count) "
                "VALUES (?, ?, ?, ?, ?, 0) "
                "ON CONFLICT(query_id) DO UPDATE SET canonical_sql = excluded.canonical_sql, "
                "tables_json = excluded.tables_json, last_seen = excluded.last_seen",
                (query.query_id, query.canonical_sql, _stable_json(query.table_names), now, now),
            )

    def query_record(self, query_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM __sw_queries WHERE query_id = ?", (query_id,)).fetchone()
        return dict(row) if row is not None else None

    def query_by_id(self, query_id: str) -> Query:
        row = self.query_record(query_id)
        if row is None:
            raise fail("E_REPLAY", f"query identifier is missing: {query_id}", {"query_id": query_id})
        return parse_select(row["canonical_sql"])

    def _count_pipeline_rows(self, query: Query) -> int:
        from_sql = query.tables[0].sql()
        if len(query.tables) > 1:
            from_sql += " JOIN " + query.tables[1].sql()
        if query.join_conditions:
            from_sql += " ON " + " AND ".join(item.sql() for item in query.join_conditions)
        where_sql = " WHERE " + " AND ".join(item.sql() for item in query.predicates) if query.predicates else ""
        row = self.conn.execute(f"SELECT COUNT(*) AS n FROM {from_sql}{where_sql}").fetchone()
        return int(row["n"])

    def execute_observed(
        self,
        query_or_sql: Query | str,
        *,
        sql: str | None = None,
        kind: str = "original",
        candidate_id: str | None = None,
        timeout_ms: int | float | None = None,
        record_query: bool = True,
    ) -> Execution:
        self._ensure_open()
        query = parse_select(query_or_sql) if isinstance(query_or_sql, str) else query_or_sql
        executed_sql = sql or query.canonical_sql
        if timeout_ms is not None:
            if not isinstance(timeout_ms, (int, float)) or isinstance(timeout_ms, bool) or timeout_ms <= 0:
                raise fail("E_SESSION", "timeout_ms must be a positive number of milliseconds")
        if record_query:
            self.register_query(query)
        versions = self.version_info()
        execution_id = "x_" + uuid.uuid4().hex
        started_ns = time.perf_counter_ns()
        deadline_ns = None if timeout_ms is None else started_ns + int(float(timeout_ms) * 1_000_000)
        interrupted = False
        rows: list[Any] = []
        outcome = "success"
        error_code: str | None = None
        error_message: str | None = None

        def progress_handler() -> int:
            nonlocal interrupted
            if deadline_ns is not None and time.perf_counter_ns() >= deadline_ns:
                interrupted = True
                return 1
            return 0

        self.conn.set_progress_handler(progress_handler, 1000)
        try:
            cursor = self.conn.execute(executed_sql)
            rows = cursor.fetchall()
        except sqlite3.OperationalError as exc:
            if interrupted or "interrupted" in str(exc).lower():
                outcome = "timeout"
                error_code = "E_TIMEOUT"
                error_message = f"query interrupted after monotonic timeout: {timeout_ms} ms"
            else:
                outcome = "error"
                error_code = "E_DATA"
                error_message = str(exc)
        except sqlite3.Error as exc:
            outcome = "error"
            error_code = "E_DATA"
            error_message = str(exc)
        finally:
            ended_ns = time.perf_counter_ns()
            self.conn.set_progress_handler(None, 0)

        elapsed_ms = max(0.0, (ended_ns - started_ns) / 1_000_000.0)
        if timeout_ms is not None and elapsed_ms > float(timeout_ms) and outcome == "success":
            outcome = "timeout"
            error_code = "E_TIMEOUT"
            error_message = f"query exceeded monotonic timeout: {timeout_ms} ms"
            rows = []
        result_rows = len(rows) if outcome == "success" else 0
        scanned_rows = 0
        if outcome == "success":
            try:
                scanned_rows = self._count_pipeline_rows(query)
            except sqlite3.Error as exc:
                self.logger.warning(
                    "scan_count_failed",
                    "query succeeded but scan-row observation failed",
                    query_id=query.query_id,
                    error=str(exc),
                )
        created_at = utc_now()
        execution = Execution(
            execution_id=execution_id,
            query_id=query.query_id,
            canonical_sql=query.canonical_sql,
            executed_sql=executed_sql,
            kind=kind,
            candidate_id=candidate_id,
            started_mono_ms=started_ns / 1_000_000.0,
            ended_mono_ms=ended_ns / 1_000_000.0,
            elapsed_ms=elapsed_ms,
            scanned_rows=scanned_rows,
            result_rows=result_rows,
            outcome=outcome,
            error_code=error_code,
            error_message=error_message,
            timeout_ms=float(timeout_ms) if timeout_ms is not None else None,
            data_version=versions["data_version"],
            index_version=versions["index_version"],
            stats_version=versions["stats_version"],
            created_at=created_at,
        )
        with self.conn:
            self.conn.execute("UPDATE __sw_queries SET execution_count = execution_count + 1, last_seen = ? WHERE query_id = ?", (created_at, query.query_id))
            self.conn.execute(
                "INSERT INTO __sw_executions("
                "execution_id, query_id, canonical_sql, executed_sql, kind, candidate_id, "
                "started_mono_ms, ended_mono_ms, elapsed_ms, scanned_rows, result_rows, outcome, "
                "error_code, error_message, timeout_ms, data_version, index_version, stats_version, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    execution.execution_id,
                    execution.query_id,
                    execution.canonical_sql,
                    execution.executed_sql,
                    execution.kind,
                    execution.candidate_id,
                    execution.started_mono_ms,
                    execution.ended_mono_ms,
                    execution.elapsed_ms,
                    execution.scanned_rows,
                    execution.result_rows,
                    execution.outcome,
                    execution.error_code,
                    execution.error_message,
                    execution.timeout_ms,
                    execution.data_version,
                    execution.index_version,
                    execution.stats_version,
                    execution.created_at,
                ),
            )
        if outcome == "timeout":
            self.logger.error(
                "query_timeout",
                error_message or "query timed out",
                code="E_TIMEOUT",
                details={"query_id": query.query_id, "execution_id": execution_id, "elapsed_ms": elapsed_ms, "timeout_ms": timeout_ms},
            )
        elif outcome == "error":
            self.logger.error(
                "query_data_error",
                error_message or "query execution failed",
                code="E_DATA",
                details={"query_id": query.query_id, "execution_id": execution_id},
            )
        else:
            self.logger.info(
                "query_executed",
                "query execution observed",
                query_id=query.query_id,
                execution_id=execution_id,
                kind=kind,
                candidate_id=candidate_id,
                elapsed_ms=elapsed_ms,
                scanned_rows=scanned_rows,
                result_rows=result_rows,
            )
        return execution

    def execute_select(
        self,
        sql: str,
        *,
        timeout_ms: int | float | None = None,
        kind: str = "original",
    ) -> dict[str, Any]:
        query = parse_select(sql)
        execution = self.execute_observed(query, kind=kind, timeout_ms=timeout_ms)
        if execution.outcome == "timeout":
            raise fail(
                "E_TIMEOUT",
                execution.error_message or "query timed out",
                {"query_id": execution.query_id, "execution_id": execution.execution_id, "elapsed_ms": execution.elapsed_ms, "timeout_ms": execution.timeout_ms},
            )
        if execution.outcome == "error":
            raise fail(
                "E_DATA",
                execution.error_message or "query execution failed",
                {"query_id": execution.query_id, "execution_id": execution.execution_id},
            )
        return execution.to_dict()

    def execution_history(
        self,
        query_id: str | None = None,
        *,
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if query_id is not None:
            clauses.append("query_id = ?")
            params.append(query_id)
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        return [
            dict(row)
            for row in self.conn.execute(f"SELECT * FROM __sw_executions{where} ORDER BY seq", params)
        ]

    def slow_queries(self, threshold_ms: int | None = None, threshold_rows: int | None = None) -> list[dict[str, Any]]:
        defaults = self.thresholds()
        ms_limit = defaults["threshold_ms"] if threshold_ms is None else threshold_ms
        row_limit = defaults["threshold_rows"] if threshold_rows is None else threshold_rows
        if not isinstance(ms_limit, int) or isinstance(ms_limit, bool) or ms_limit < 0:
            raise fail("E_SESSION", "threshold_ms must be a non-negative integer")
        if not isinstance(row_limit, int) or isinstance(row_limit, bool) or row_limit < 0:
            raise fail("E_SESSION", "threshold_rows must be a non-negative integer")
        rows = self.conn.execute(
            "SELECT e.* FROM __sw_executions e "
            "JOIN (SELECT query_id, MAX(seq) AS max_seq FROM __sw_executions "
            "      WHERE kind IN ('original', 'diagnosis_original', 'replay_original') "
            "      GROUP BY query_id) latest "
            "ON e.query_id = latest.query_id AND e.seq = latest.max_seq "
            "ORDER BY e.query_id"
        ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["is_slow"] = is_slow(
                float(item["elapsed_ms"]), int(item["scanned_rows"]), ms_limit, row_limit
            )
            item["threshold_ms"] = ms_limit
            item["threshold_rows"] = row_limit
            if item["is_slow"]:
                result.append(item)
        return result

    def classify_history(
        self, threshold_ms: int, threshold_rows: int, *, query_id: str | None = None
    ) -> list[dict[str, Any]]:
        rows = self.execution_history(query_id=query_id)
        for row in rows:
            row["threshold_ms"] = threshold_ms
            row["threshold_rows"] = threshold_rows
            row["is_slow"] = is_slow(
                float(row["elapsed_ms"]), int(row["scanned_rows"]), threshold_ms, threshold_rows
            )
        return rows

    # ------------------------------------------------------------- invalidation
    def _affected_query_ids_no_tx(self, tables: Sequence[str]) -> list[str]:
        targets = sorted({str(table).lower() for table in tables})
        affected: list[str] = []
        for row in self.conn.execute("SELECT query_id, tables_json FROM __sw_queries ORDER BY query_id"):
            dependencies = {str(item).lower() for item in json.loads(row["tables_json"])}
            if not targets or dependencies.intersection(targets):
                affected.append(str(row["query_id"]))
        return sorted(set(affected))

    def _record_invalidation_no_tx(
        self,
        event_type: str,
        source_version_field: str,
        old_value: int,
        new_value: int,
        affected_tables: Sequence[str],
        reason: str,
    ) -> dict[str, Any]:
        affected_query_ids = self._affected_query_ids_no_tx(affected_tables)
        normalized_tables = sorted({str(table).lower() for table in affected_tables})
        now = utc_now()
        cursor = self.conn.execute(
            "INSERT INTO __sw_change_events("
            "event_type, source_version_field, old_value, new_value, affected_query_ids_json, "
            "affected_tables_json, reason, created_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event_type,
                source_version_field,
                old_value,
                new_value,
                _stable_json(affected_query_ids),
                _stable_json(normalized_tables),
                reason,
                now,
            ),
        )
        event_id = int(cursor.lastrowid)
        if affected_query_ids:
            placeholders = ", ".join("?" for _ in affected_query_ids)
            self.conn.execute(
                f"UPDATE __sw_conclusions SET is_stale = 1, stale_event_id = ?, stale_reason = ?, stale_at = ?, "
                f"updated_at = ? WHERE query_id IN ({placeholders})",
                (event_id, reason, now, now, *affected_query_ids),
            )
        event = {
            "event_id": event_id,
            "event_type": event_type,
            "source_version_field": source_version_field,
            "old_value": old_value,
            "new_value": new_value,
            "affected_query_ids": affected_query_ids,
            "affected_tables": normalized_tables,
            "reason": reason,
            "created_at": now,
        }
        self.logger.info("version_invalidation", "version change propagated to conclusions", change_event=event)
        return event

    def change_events(self) -> list[dict[str, Any]]:
        result = []
        for row in self.conn.execute("SELECT * FROM __sw_change_events ORDER BY id"):
            item = dict(row)
            item["affected_query_ids"] = json.loads(item.pop("affected_query_ids_json"))
            item["affected_tables"] = json.loads(item.pop("affected_tables_json"))
            result.append(item)
        return result

    # --------------------------------------------------------------- conclusions
    def upsert_candidate(self, candidate: Mapping[str, Any]) -> None:
        now = utc_now()
        with self.conn:
            self.conn.execute(
                "INSERT INTO __sw_candidates(candidate_id, query_id, kind, rewritten_sql, rationale, "
                "referenced_indexes_json, index_version, stats_version, status, elapsed_ms, result_rows, "
                "execution_id, reason, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'generated', NULL, NULL, NULL, NULL, ?, ?) "
                "ON CONFLICT(candidate_id) DO UPDATE SET rewritten_sql = excluded.rewritten_sql, "
                "rationale = excluded.rationale, referenced_indexes_json = excluded.referenced_indexes_json, "
                "index_version = excluded.index_version, stats_version = excluded.stats_version, "
                "status = 'generated', elapsed_ms = NULL, result_rows = NULL, execution_id = NULL, reason = NULL, "
                "updated_at = excluded.updated_at",
                (
                    candidate["candidate_id"],
                    candidate["query_id"],
                    candidate["kind"],
                    candidate["sql"],
                    candidate["rationale"],
                    _stable_json(candidate.get("referenced_indexes", [])),
                    int(candidate["index_version"]),
                    int(candidate["stats_version"]),
                    now,
                    now,
                ),
            )

    def record_candidate_result(
        self,
        candidate_id: str,
        *,
        status: str,
        elapsed_ms: float | None = None,
        result_rows: int | None = None,
        execution_id: str | None = None,
        error_code: str | None = None,
        reason: str | None = None,
    ) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE __sw_candidates SET status = ?, elapsed_ms = ?, result_rows = ?, execution_id = ?, "
                "error_code = ?, reason = ?, updated_at = ? WHERE candidate_id = ?",
                (status, elapsed_ms, result_rows, execution_id, error_code, reason, utc_now(), candidate_id),
            )

    def save_conclusion(self, conclusion: Mapping[str, Any]) -> None:
        now = utc_now()
        with self.conn:
            self.conn.execute(
                "INSERT INTO __sw_conclusions("
                "query_id, conclusion_id, original_execution_id, original_elapsed_ms, original_scanned_rows, "
                "original_result_rows, candidate_ids_json, optimal_candidate_id, optimal_elapsed_ms, speedup, "
                "threshold_ms, threshold_rows, data_version, index_version, stats_version, is_stale, "
                "stale_event_id, stale_reason, stale_at, created_at, updated_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL, NULL, NULL, ?, ?) "
                "ON CONFLICT(query_id) DO UPDATE SET conclusion_id = excluded.conclusion_id, "
                "original_execution_id = excluded.original_execution_id, "
                "original_elapsed_ms = excluded.original_elapsed_ms, "
                "original_scanned_rows = excluded.original_scanned_rows, "
                "original_result_rows = excluded.original_result_rows, "
                "candidate_ids_json = excluded.candidate_ids_json, "
                "optimal_candidate_id = excluded.optimal_candidate_id, "
                "optimal_elapsed_ms = excluded.optimal_elapsed_ms, speedup = excluded.speedup, "
                "threshold_ms = excluded.threshold_ms, threshold_rows = excluded.threshold_rows, "
                "data_version = excluded.data_version, index_version = excluded.index_version, "
                "stats_version = excluded.stats_version, is_stale = 0, stale_event_id = NULL, "
                "stale_reason = NULL, stale_at = NULL, created_at = excluded.created_at, updated_at = excluded.updated_at",
                (
                    conclusion["query_id"],
                    conclusion["conclusion_id"],
                    conclusion["original_execution_id"],
                    float(conclusion["original_elapsed_ms"]),
                    int(conclusion["original_scanned_rows"]),
                    int(conclusion["original_result_rows"]),
                    _stable_json(list(conclusion["candidate_ids"])),
                    conclusion.get("optimal_candidate_id"),
                    float(conclusion["optimal_elapsed_ms"]) if conclusion.get("optimal_elapsed_ms") is not None else None,
                    float(conclusion["speedup"]) if conclusion.get("speedup") is not None else None,
                    int(conclusion["threshold_ms"]),
                    int(conclusion["threshold_rows"]),
                    int(conclusion["data_version"]),
                    int(conclusion["index_version"]),
                    int(conclusion["stats_version"]),
                    now,
                    now,
                ),
            )

    def conclusion(self, query_id: str, *, include_stale: bool = True) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM __sw_conclusions WHERE query_id = ?", (query_id,)).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["candidate_ids"] = json.loads(item.pop("candidate_ids_json"))
        item["is_stale"] = bool(item["is_stale"])
        if not include_stale and item["is_stale"]:
            return None
        candidates = []
        for candidate_id in item["candidate_ids"]:
            candidate_row = self.conn.execute(
                "SELECT * FROM __sw_candidates WHERE candidate_id = ?", (candidate_id,)
            ).fetchone()
            if candidate_row is None:
                candidates.append({"candidate_id": candidate_id, "missing": True})
                continue
            candidate = dict(candidate_row)
            candidate["referenced_indexes"] = json.loads(candidate.pop("referenced_indexes_json"))
            candidate["missing"] = False
            candidates.append(candidate)
        item["candidates"] = candidates
        return item

    def conclusions(self, *, include_stale: bool = True) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT query_id FROM __sw_conclusions ORDER BY query_id").fetchall()
        result = []
        for row in rows:
            conclusion = self.conclusion(str(row["query_id"]), include_stale=include_stale)
            if conclusion is not None:
                result.append(conclusion)
        return result

    def current_best(self, query_id: str | None = None) -> dict[str, Any] | list[dict[str, Any]] | None:
        if query_id is not None:
            conclusion = self.conclusion(query_id, include_stale=False)
            return conclusion
        return [item for item in self.conclusions(include_stale=False)]

    # -------------------------------------------------------------- save/snapshot
    def save_as(self, path: str | Path, name: str | None = None) -> "Session":
        self._ensure_open()
        target = Path(path).resolve()
        if target == self.path:
            raise fail("E_SESSION", "save-as target must differ from the current session file")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        destination = sqlite3.connect(str(target))
        try:
            self.conn.commit()
            self.conn.backup(destination)
            destination.commit()
            if name is not None:
                destination.execute("UPDATE __sw_meta SET value = ? WHERE key = 'session_name'", (name,))
                destination.commit()
        finally:
            destination.close()
        self.logger.info("session_saved_as", "session copied", target=str(target), new_name=name)
        return Session.open(target, logger=JsonlLogger(target.with_name(target.name + ".log.jsonl")))

    def save_snapshot(self, path: str | Path) -> Path:
        self._ensure_open()
        target = Path(path).resolve()
        if target == self.path:
            raise fail("E_SESSION", "snapshot target must differ from the current session file")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        destination = sqlite3.connect(str(target))
        try:
            self.conn.commit()
            self.conn.backup(destination)
            destination.commit()
        finally:
            destination.close()
        self.logger.info("snapshot_saved", "session snapshot saved", snapshot_path=str(target), versions=self.version_info())
        return target

    def _snapshot_versions(self, path: Path) -> dict[str, Any]:
        connection = sqlite3.connect(str(path))
        connection.row_factory = sqlite3.Row
        try:
            rows = {row["key"]: row["value"] for row in connection.execute("SELECT key, value FROM __sw_meta")}
            missing = [field for field in REQUIRED_META if field not in rows]
            if missing:
                raise fail("E_VERSION", "snapshot metadata is incomplete", {"missing_fields": missing})
            return {field: int(rows[field]) for field in VERSION_FIELDS} | {
                "schema_hash": rows["schema_hash"],
                "index_set_hash": rows["index_set_hash"],
            }
        except sqlite3.Error as exc:
            raise fail("E_VERSION", f"snapshot is unreadable: {exc}") from exc
        finally:
            connection.close()

    def restore_snapshot(self, snapshot_path: str | Path) -> dict[str, Any]:
        """Wholesale restore a snapshot; report version differences as E_VERSION.

        The replacement happens before E_VERSION is raised, so callers can inspect
        the restored state and then replay the restored conclusions.
        """

        self._ensure_open()
        snapshot = Path(snapshot_path).resolve()
        if snapshot == self.path:
            raise fail("E_SESSION", "snapshot path equals the active session path")
        if not snapshot.exists():
            raise fail("E_VERSION", f"snapshot does not exist: {snapshot}")
        current = self._snapshot_versions(self.path)
        incoming = self._snapshot_versions(snapshot)
        field_order = (
            "schema_hash",
            "data_version",
            "index_set_hash",
            "index_version",
            "stats_version",
        )
        differences = [
            {"field": field, "current": current[field], "snapshot": incoming[field]}
            for field in field_order
            if current[field] != incoming[field]
        ]
        source = sqlite3.connect(str(snapshot))
        try:
            self.conn.commit()
            source.backup(self.conn)
            self.conn.commit()
        except sqlite3.Error as exc:
            raise fail("E_VERSION", f"snapshot replacement failed: {exc}") from exc
        finally:
            source.close()
        self._validate_open()
        self.logger.info(
            "snapshot_restored",
            "snapshot replaced whole session state",
            snapshot_path=str(snapshot),
            differences=differences,
            versions=self.version_info(),
        )
        if differences:
            raise fail(
                "E_VERSION",
                "snapshot versions differ; session was replaced wholesale",
                {"differences": differences, "replaced": True, "snapshot": str(snapshot)},
            )
        return {"differences": [], "replaced": True, "snapshot": str(snapshot), "versions": self.version_info()}

    # ------------------------------------------------------------------ replay log
    def save_replay_validation(self, report: Mapping[str, Any]) -> str:
        validation_id = "v_" + uuid.uuid4().hex
        now = utc_now()
        with self.conn:
            self.conn.execute(
                "INSERT INTO __sw_replay_validations(validation_id, created_at, match, report_json) VALUES (?, ?, ?, ?)",
                (validation_id, now, 1 if report.get("matched") else 0, _stable_json(report)),
            )
        return validation_id
