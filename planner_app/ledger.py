"""Crash-safe SQLite ledger for plan versions, edits, latency and verification."""

from __future__ import annotations

import datetime as _datetime
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .canonical import canonical_bytes, digest, sha256_bytes
from .errors import E_LEDGER, E_REPLAY, AppError


LEDGER_FORMAT_VERSION = 1


def _utc_now() -> str:
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _from_json(value: str | None, default: Any) -> Any:
    if value is None:
        return default
    return json.loads(value)


class Ledger:
    """A small connection-per-operation ledger.

    Connection-per-operation keeps background verification safe to run from a
    Tkinter worker thread while the UI thread performs cache-only reads.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)

    @contextmanager
    def connection(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(self.path), timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        if write:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> int:
        try:
            with self.connection(write=True) as connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS meta (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS plan_batches (
                        batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        workspace_id TEXT NOT NULL,
                        query_name TEXT NOT NULL,
                        epoch INTEGER NOT NULL,
                        cursor INTEGER NOT NULL,
                        batch_kind TEXT NOT NULL,
                        parameter_version INTEGER NOT NULL,
                        tables_version INTEGER NOT NULL,
                        sql_version INTEGER NOT NULL,
                        edits_version INTEGER NOT NULL,
                        scale_version INTEGER NOT NULL,
                        table_digest TEXT NOT NULL,
                        sql_digest TEXT NOT NULL,
                        scale_digest TEXT NOT NULL,
                        context_digest TEXT NOT NULL,
                        edit_seq_start INTEGER NOT NULL,
                        edit_seq_end INTEGER NOT NULL,
                        plan_summary TEXT NOT NULL,
                        plan_bytes BLOB NOT NULL,
                        node_count INTEGER NOT NULL,
                        estimated_rows REAL NOT NULL,
                        affected_json TEXT NOT NULL,
                        propagation_json TEXT NOT NULL,
                        edits_json TEXT NOT NULL,
                        incremental_ms REAL NOT NULL,
                        full_ms REAL,
                        verify_status TEXT NOT NULL,
                        verify_json TEXT,
                        progress_cursor INTEGER NOT NULL,
                        stale INTEGER NOT NULL DEFAULT 0,
                        content_hash TEXT NOT NULL,
                        created_mono_ns INTEGER NOT NULL,
                        created_at TEXT NOT NULL,
                        UNIQUE(workspace_id, cursor)
                    );
                    CREATE INDEX IF NOT EXISTS idx_batches_query
                        ON plan_batches(workspace_id, query_name, context_digest, epoch, cursor);
                    CREATE TABLE IF NOT EXISTS edit_attempts (
                        attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        workspace_id TEXT NOT NULL,
                        cursor INTEGER NOT NULL,
                        query_name TEXT NOT NULL,
                        edits_json TEXT NOT NULL,
                        parameter_version INTEGER NOT NULL,
                        next_edit_seq INTEGER NOT NULL,
                        status TEXT NOT NULL,
                        code TEXT,
                        detail_json TEXT,
                        batch_id INTEGER,
                        started_mono_ns INTEGER NOT NULL,
                        finished_mono_ns INTEGER,
                        started_at TEXT NOT NULL,
                        finished_at TEXT
                    );
                    CREATE TABLE IF NOT EXISTS latency_records (
                        latency_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        workspace_id TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        query_name TEXT,
                        batch_id INTEGER,
                        attempt_id INTEGER,
                        code TEXT NOT NULL,
                        within_budget INTEGER NOT NULL,
                        budget_ms REAL NOT NULL,
                        elapsed_ms REAL NOT NULL,
                        mono_start_ns INTEGER NOT NULL,
                        mono_end_ns INTEGER NOT NULL,
                        truth_source TEXT NOT NULL,
                        details_json TEXT NOT NULL,
                        content_hash TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_latency_workspace
                        ON latency_records(workspace_id, latency_id);
                    CREATE TABLE IF NOT EXISTS verification_records (
                        verification_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        workspace_id TEXT NOT NULL,
                        batch_id INTEGER,
                        mode TEXT NOT NULL,
                        status TEXT NOT NULL,
                        elapsed_ms REAL NOT NULL,
                        expected_summary TEXT NOT NULL,
                        actual_summary TEXT NOT NULL,
                        first_diff_json TEXT,
                        mono_start_ns INTEGER NOT NULL,
                        mono_end_ns INTEGER NOT NULL,
                        content_hash TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS parameter_events (
                        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        workspace_id TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        parameter_version INTEGER NOT NULL,
                        versions_json TEXT NOT NULL,
                        reason TEXT NOT NULL,
                        details_json TEXT NOT NULL,
                        mono_ns INTEGER NOT NULL,
                        content_hash TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_parameter_events
                        ON parameter_events(workspace_id, event_id);
                    """
                )
                row = connection.execute("PRAGMA user_version").fetchone()
                current = int(row[0]) if row else 0
                if current == 0:
                    connection.execute(f"PRAGMA user_version = {LEDGER_FORMAT_VERSION}")
                elif current != LEDGER_FORMAT_VERSION:
                    raise AppError(
                        E_LEDGER,
                        f"unsupported ledger user_version {current}",
                    )
                meta = connection.execute(
                    "SELECT value FROM meta WHERE key = 'ledger_format_version'"
                ).fetchone()
                if meta is None:
                    connection.execute(
                        "INSERT INTO meta(key, value) VALUES('ledger_format_version', ?)",
                        (str(LEDGER_FORMAT_VERSION),),
                    )
                elif int(meta["value"]) != LEDGER_FORMAT_VERSION:
                    raise AppError(E_LEDGER, f"unsupported ledger format {meta['value']}")
                connection.commit()
            return LEDGER_FORMAT_VERSION
        except AppError:
            raise
        except sqlite3.Error as error:
            raise AppError(E_LEDGER, f"cannot initialize ledger: {error}") from error

    @staticmethod
    def _batch_content_hash(record: Mapping[str, Any]) -> str:
        affected = record.get("affected")
        if affected is None:
            affected = _from_json(record.get("affected_json"), [])
        propagation = record.get("propagation")
        if propagation is None:
            propagation = _from_json(record.get("propagation_json"), [])
        edits = record.get("edits")
        if edits is None:
            edits = _from_json(record.get("edits_json"), [])
        immutable = {
            "workspace_id": record.get("workspace_id"),
            "query_name": record.get("query_name"),
            "epoch": record.get("epoch"),
            "cursor": record.get("cursor"),
            "batch_kind": record.get("batch_kind"),
            "parameter_version": record.get("parameter_version"),
            "tables_version": record.get("tables_version"),
            "sql_version": record.get("sql_version"),
            "edits_version": record.get("edits_version"),
            "scale_version": record.get("scale_version"),
            "table_digest": record.get("table_digest"),
            "sql_digest": record.get("sql_digest"),
            "scale_digest": record.get("scale_digest"),
            "context_digest": record.get("context_digest"),
            "edit_seq_start": record.get("edit_seq_start"),
            "edit_seq_end": record.get("edit_seq_end"),
            "plan_summary": record.get("plan_summary"),
            "node_count": record.get("node_count"),
            "estimated_rows": record.get("estimated_rows"),
            "affected": affected,
            "propagation": propagation,
            "edits": edits,
            "progress_cursor": record.get("progress_cursor"),
        }
        plan_bytes = record.get("plan_bytes")
        if isinstance(plan_bytes, memoryview):
            plan_bytes = bytes(plan_bytes)
        immutable["plan_bytes_sha256"] = sha256_bytes(bytes(plan_bytes or b""))
        return digest(immutable)

    def _prepare_batch(self, record: Mapping[str, Any]) -> dict[str, Any]:
        prepared = dict(record)
        prepared["affected_json"] = _json(record.get("affected", []))
        prepared["propagation_json"] = _json(record.get("propagation", []))
        prepared["edits_json"] = _json(record.get("edits", []))
        prepared["verify_json"] = _json(record.get("verify", {})) if record.get("verify") is not None else None
        prepared["plan_bytes"] = sqlite3.Binary(bytes(record["plan_bytes"]))
        prepared["created_mono_ns"] = int(record.get("created_mono_ns", time.monotonic_ns()))
        prepared["created_at"] = str(record.get("created_at", _utc_now()))
        prepared["stale"] = int(record.get("stale", 0))
        prepared["content_hash"] = self._batch_content_hash(record)
        return prepared

    @staticmethod
    def _insert_batch(connection: sqlite3.Connection, record: Mapping[str, Any]) -> int:
        columns = (
            "workspace_id",
            "query_name",
            "epoch",
            "cursor",
            "batch_kind",
            "parameter_version",
            "tables_version",
            "sql_version",
            "edits_version",
            "scale_version",
            "table_digest",
            "sql_digest",
            "scale_digest",
            "context_digest",
            "edit_seq_start",
            "edit_seq_end",
            "plan_summary",
            "plan_bytes",
            "node_count",
            "estimated_rows",
            "affected_json",
            "propagation_json",
            "edits_json",
            "incremental_ms",
            "full_ms",
            "verify_status",
            "verify_json",
            "progress_cursor",
            "stale",
            "content_hash",
            "created_mono_ns",
            "created_at",
        )
        placeholders = ",".join("?" for _ in columns)
        cursor = connection.execute(
            f"INSERT INTO plan_batches({','.join(columns)}) VALUES({placeholders})",
            tuple(record.get(column) for column in columns),
        )
        return int(cursor.lastrowid)

    @staticmethod
    def _insert_latency(
        connection: sqlite3.Connection,
        record: Mapping[str, Any],
        *,
        batch_id: int | None = None,
    ) -> int:
        payload = dict(record)
        payload.setdefault("details", {})
        content_hash = digest(
            {
                key: value
                for key, value in payload.items()
                if key not in {"details", "content_hash", "created_at"}
            }
        )
        cursor = connection.execute(
            """
            INSERT INTO latency_records(
                workspace_id, kind, query_name, batch_id, attempt_id, code,
                within_budget, budget_ms, elapsed_ms, mono_start_ns, mono_end_ns,
                truth_source, details_json, content_hash, created_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                payload["workspace_id"],
                payload["kind"],
                payload.get("query_name"),
                batch_id if batch_id is not None else payload.get("batch_id"),
                payload.get("attempt_id"),
                payload["code"],
                int(bool(payload["within_budget"])),
                float(payload["budget_ms"]),
                float(payload["elapsed_ms"]),
                int(payload["mono_start_ns"]),
                int(payload["mono_end_ns"]),
                str(payload.get("truth_source", "ledger:latency_records")),
                _json(payload.get("details", {})),
                content_hash,
                str(payload.get("created_at", _utc_now())),
            ),
        )
        return int(cursor.lastrowid)

    @staticmethod
    def _insert_parameter_event(connection: sqlite3.Connection, event: Mapping[str, Any]) -> int:
        payload = dict(event)
        versions_json = _json(payload.get("versions", {}))
        details_json = _json(payload.get("details", {}))
        content_hash = digest(
            {
                "workspace_id": payload["workspace_id"],
                "event_type": payload["event_type"],
                "parameter_version": int(payload["parameter_version"]),
                "versions": payload.get("versions", {}),
                "reason": payload["reason"],
                "details": payload.get("details", {}),
            }
        )
        cursor = connection.execute(
            """
            INSERT INTO parameter_events(
                workspace_id, event_type, parameter_version, versions_json,
                reason, details_json, mono_ns, content_hash, created_at
            ) VALUES(?,?,?,?,?,?,?,?,?)
            """,
            (
                payload["workspace_id"],
                payload["event_type"],
                int(payload["parameter_version"]),
                versions_json,
                payload["reason"],
                details_json,
                int(payload.get("mono_ns", time.monotonic_ns())),
                content_hash,
                str(payload.get("created_at", _utc_now())),
            ),
        )
        return int(cursor.lastrowid)

    def commit_batches(
        self,
        batches: Sequence[Mapping[str, Any]],
        *,
        latencies: Sequence[Mapping[str, Any]] = (),
        parameter_event: Mapping[str, Any] | None = None,
        attempt_id: int | None = None,
        stale_before_parameter_version: tuple[str, int] | tuple[str, str, int] | None = None,
    ) -> list[int]:
        if not batches:
            raise ValueError("at least one batch is required")
        prepared = [self._prepare_batch(batch) for batch in batches]
        try:
            with self.connection(write=True) as connection:
                connection.execute("BEGIN IMMEDIATE")
                batch_ids: list[int] = []
                for record in prepared:
                    batch_ids.append(self._insert_batch(connection, record))
                for latency_index, latency in enumerate(latencies):
                    payload = dict(latency)
                    selected_batch_id = payload.get("batch_id")
                    batch_index = payload.pop("batch_index", None)
                    if selected_batch_id is None and batch_index is not None:
                        selected_batch_id = batch_ids[int(batch_index)]
                    elif selected_batch_id is None and len(batch_ids) == 1:
                        selected_batch_id = batch_ids[0]
                    self._insert_latency(connection, payload, batch_id=selected_batch_id)
                if parameter_event is not None:
                    self._insert_parameter_event(connection, parameter_event)
                if attempt_id is not None:
                    connection.execute(
                        "UPDATE edit_attempts SET status='COMMITTED', batch_id=?, finished_mono_ns=?, finished_at=? WHERE attempt_id=?",
                        (
                            batch_ids[0] if batch_ids else None,
                            time.monotonic_ns(),
                            _utc_now(),
                            int(attempt_id),
                        ),
                    )
                if stale_before_parameter_version is not None:
                    if len(stale_before_parameter_version) == 2:
                        workspace_id, cutoff = stale_before_parameter_version
                        connection.execute(
                            "UPDATE plan_batches SET stale=1 WHERE workspace_id=? AND parameter_version < ?",
                            (workspace_id, int(cutoff)),
                        )
                    else:
                        workspace_id, query_name, cutoff = stale_before_parameter_version
                        connection.execute(
                            "UPDATE plan_batches SET stale=1 WHERE workspace_id=? AND query_name=? AND parameter_version < ?",
                            (workspace_id, query_name, int(cutoff)),
                        )
                connection.commit()
                return batch_ids
        except sqlite3.Error as error:
            raise AppError(E_LEDGER, f"cannot commit plan batch: {error}") from error

    def begin_attempt(
        self,
        *,
        workspace_id: str,
        cursor: int,
        query_name: str,
        edits: Sequence[Mapping[str, Any]],
        parameter_version: int,
        next_edit_seq: int,
    ) -> int:
        try:
            with self.connection(write=True) as connection:
                connection.execute("BEGIN IMMEDIATE")
                cursor_row = connection.execute(
                    "INSERT INTO edit_attempts(workspace_id,cursor,query_name,edits_json,parameter_version,next_edit_seq,status,started_mono_ns,started_at) VALUES(?,?,?,?,?,?,'RUNNING',?,?)",
                    (
                        workspace_id,
                        int(cursor),
                        query_name,
                        _json(list(edits)),
                        int(parameter_version),
                        int(next_edit_seq),
                        time.monotonic_ns(),
                        _utc_now(),
                    ),
                )
                connection.commit()
                return int(cursor_row.lastrowid)
        except sqlite3.Error as error:
            raise AppError(E_LEDGER, f"cannot record edit intent: {error}") from error

    def fail_attempt(self, attempt_id: int, code: str, detail: Mapping[str, Any]) -> None:
        try:
            with self.connection(write=True) as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "UPDATE edit_attempts SET status='FAILED', code=?, detail_json=?, finished_mono_ns=?, finished_at=? WHERE attempt_id=?",
                    (code, _json(dict(detail)), time.monotonic_ns(), _utc_now(), int(attempt_id)),
                )
                connection.commit()
        except sqlite3.Error as error:
            raise AppError(E_LEDGER, f"cannot finalise failed edit intent: {error}") from error

    def recover_unknown_attempts(self, workspace_id: str) -> list[dict[str, Any]]:
        try:
            with self.connection(write=True) as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "UPDATE edit_attempts SET status='UNKNOWN', code='E_UNKNOWN', finished_mono_ns=?, finished_at=? WHERE workspace_id=? AND status='RUNNING'",
                    (time.monotonic_ns(), _utc_now(), workspace_id),
                )
                rows = connection.execute(
                    "SELECT * FROM edit_attempts WHERE workspace_id=? AND status='UNKNOWN' ORDER BY attempt_id",
                    (workspace_id,),
                ).fetchall()
                connection.commit()
            return [self._attempt_row(row) for row in rows]
        except sqlite3.Error as error:
            raise AppError(E_LEDGER, f"cannot recover edit intents: {error}") from error

    @staticmethod
    def _attempt_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "attempt_id": int(row["attempt_id"]),
            "workspace_id": row["workspace_id"],
            "cursor": int(row["cursor"]),
            "query_name": row["query_name"],
            "edits": _from_json(row["edits_json"], []),
            "parameter_version": int(row["parameter_version"]),
            "next_edit_seq": int(row["next_edit_seq"]),
            "status": row["status"],
            "code": row["code"],
            "batch_id": row["batch_id"],
            "started_mono_ns": int(row["started_mono_ns"]),
            "finished_mono_ns": row["finished_mono_ns"],
            "requires_rerun": row["status"] == "UNKNOWN",
        }

    def next_cursor(self, workspace_id: str) -> int:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(cursor), 0) AS maximum FROM plan_batches WHERE workspace_id=?",
                (workspace_id,),
            ).fetchone()
            return int(row["maximum"]) + 1

    def latest_cursor(self, workspace_id: str) -> int:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(cursor), 0) AS maximum FROM plan_batches WHERE workspace_id=?",
                (workspace_id,),
            ).fetchone()
            return int(row["maximum"])

    def max_epoch(self, workspace_id: str) -> int:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(epoch), 0) AS maximum FROM plan_batches WHERE workspace_id=?",
                (workspace_id,),
            ).fetchone()
            return int(row["maximum"])

    def max_parameter_version(self, workspace_id: str) -> int:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT MAX(parameter_version) AS batches FROM plan_batches WHERE workspace_id=?",
                (workspace_id,),
            ).fetchone()
            events = connection.execute(
                "SELECT MAX(parameter_version) AS events FROM parameter_events WHERE workspace_id=?",
                (workspace_id,),
            ).fetchone()
            return max(int(rows["batches"] or 0), int(events["events"] or 0))

    def latest_parameter_event(self, workspace_id: str) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM parameter_events WHERE workspace_id=? ORDER BY event_id DESC LIMIT 1",
                (workspace_id,),
            ).fetchone()
            if row is None:
                return None
            return {
                "event_id": int(row["event_id"]),
                "event_type": row["event_type"],
                "parameter_version": int(row["parameter_version"]),
                "versions": _from_json(row["versions_json"], {}),
                "reason": row["reason"],
                "details": _from_json(row["details_json"], {}),
                "mono_ns": int(row["mono_ns"]),
            }

    @staticmethod
    def _batch_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "batch_id": int(row["batch_id"]),
            "workspace_id": row["workspace_id"],
            "query_name": row["query_name"],
            "epoch": int(row["epoch"]),
            "cursor": int(row["cursor"]),
            "batch_kind": row["batch_kind"],
            "parameter_version": int(row["parameter_version"]),
            "tables_version": int(row["tables_version"]),
            "sql_version": int(row["sql_version"]),
            "edits_version": int(row["edits_version"]),
            "scale_version": int(row["scale_version"]),
            "table_digest": row["table_digest"],
            "sql_digest": row["sql_digest"],
            "scale_digest": row["scale_digest"],
            "context_digest": row["context_digest"],
            "edit_seq_start": int(row["edit_seq_start"]),
            "edit_seq_end": int(row["edit_seq_end"]),
            "plan_summary": row["plan_summary"],
            "plan_bytes": bytes(row["plan_bytes"]),
            "node_count": int(row["node_count"]),
            "estimated_rows": float(row["estimated_rows"]),
            "affected": _from_json(row["affected_json"], []),
            "propagation": _from_json(row["propagation_json"], []),
            "edits": _from_json(row["edits_json"], []),
            "incremental_ms": float(row["incremental_ms"]),
            "full_ms": None if row["full_ms"] is None else float(row["full_ms"]),
            "verify_status": row["verify_status"],
            "verify": _from_json(row["verify_json"], {}),
            "progress_cursor": int(row["progress_cursor"]),
            "stale": bool(row["stale"]),
            "content_hash": row["content_hash"],
            "created_mono_ns": int(row["created_mono_ns"]),
            "created_at": row["created_at"],
        }

    def batches_for_query(
        self,
        workspace_id: str,
        query_name: str,
        context_digest: str,
    ) -> list[dict[str, Any]]:
        with self.connection() as connection:
            epoch_row = connection.execute(
                "SELECT MAX(epoch) AS epoch FROM plan_batches WHERE workspace_id=? AND query_name=? AND context_digest=?",
                (workspace_id, query_name, context_digest),
            ).fetchone()
            epoch = None if epoch_row is None else epoch_row["epoch"]
            if epoch is None:
                return []
            rows = connection.execute(
                "SELECT * FROM plan_batches WHERE workspace_id=? AND query_name=? AND context_digest=? AND epoch=? ORDER BY cursor, batch_id",
                (workspace_id, query_name, context_digest, int(epoch)),
            ).fetchall()
        result = [self._batch_row(row) for row in rows]
        for record in result:
            expected = self._batch_content_hash(record)
            if expected != record["content_hash"]:
                raise AppError(
                    E_REPLAY,
                    "ledger batch content hash mismatch",
                    details={
                        "batch_id": record["batch_id"],
                        "query_name": record["query_name"],
                        "edit_seq": record["edit_seq_start"],
                        "node_id": "$",
                        "basis": "ledger batch content_hash",
                    },
                )
        return result

    def all_batches_up_to(self, workspace_id: str, cursor: int) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM plan_batches WHERE workspace_id=? AND cursor<=? ORDER BY cursor, batch_id",
                (workspace_id, int(cursor)),
            ).fetchall()
        return [self._batch_row(row) for row in rows]

    def batch_by_id(self, batch_id: int) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute("SELECT * FROM plan_batches WHERE batch_id=?", (int(batch_id),)).fetchone()
            return None if row is None else self._batch_row(row)

    def record_latency(self, record: Mapping[str, Any]) -> int:
        try:
            with self.connection(write=True) as connection:
                connection.execute("BEGIN IMMEDIATE")
                latency_id = self._insert_latency(connection, record)
                connection.commit()
                return latency_id
        except sqlite3.Error as error:
            raise AppError(E_LEDGER, f"cannot record latency: {error}") from error

    def record_verification(
        self,
        *,
        workspace_id: str,
        batch_id: int | None,
        mode: str,
        status: str,
        elapsed_ms: float,
        expected_summary: str,
        actual_summary: str,
        first_diff: Mapping[str, Any] | None,
        mono_start_ns: int,
        mono_end_ns: int,
        update_batch: bool = True,
    ) -> int:
        first_diff_json = _json(dict(first_diff)) if first_diff is not None else None
        content_hash = digest(
            {
                "workspace_id": workspace_id,
                "batch_id": batch_id,
                "mode": mode,
                "status": status,
                "elapsed_ms": float(elapsed_ms),
                "expected_summary": expected_summary,
                "actual_summary": actual_summary,
                "first_diff": dict(first_diff) if first_diff is not None else None,
                "mono_start_ns": int(mono_start_ns),
                "mono_end_ns": int(mono_end_ns),
            }
        )
        try:
            with self.connection(write=True) as connection:
                connection.execute("BEGIN IMMEDIATE")
                cursor = connection.execute(
                    """
                    INSERT INTO verification_records(
                        workspace_id,batch_id,mode,status,elapsed_ms,expected_summary,
                        actual_summary,first_diff_json,mono_start_ns,mono_end_ns,
                        content_hash,created_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        workspace_id,
                        batch_id,
                        mode,
                        status,
                        float(elapsed_ms),
                        expected_summary,
                        actual_summary,
                        first_diff_json,
                        int(mono_start_ns),
                        int(mono_end_ns),
                        content_hash,
                        _utc_now(),
                    ),
                )
                if batch_id is not None and update_batch:
                    connection.execute(
                        "UPDATE plan_batches SET verify_status=?, verify_json=?, full_ms=? WHERE batch_id=?",
                        (status, first_diff_json or _json({"status": status}), float(elapsed_ms), int(batch_id)),
                    )
                connection.commit()
                return int(cursor.lastrowid)
        except sqlite3.Error as error:
            raise AppError(E_LEDGER, f"cannot record verification: {error}") from error

    def recent_latency(self, workspace_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM latency_records WHERE workspace_id=? ORDER BY latency_id DESC LIMIT ?",
                (workspace_id, int(limit)),
            ).fetchall()
        return [
            {
                "latency_id": int(row["latency_id"]),
                "kind": row["kind"],
                "query_name": row["query_name"],
                "batch_id": row["batch_id"],
                "code": row["code"],
                "within_budget": bool(row["within_budget"]),
                "budget_ms": float(row["budget_ms"]),
                "elapsed_ms": float(row["elapsed_ms"]),
                "mono_start_ns": int(row["mono_start_ns"]),
                "mono_end_ns": int(row["mono_end_ns"]),
                "truth_source": row["truth_source"],
                "details": _from_json(row["details_json"], {}),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def recent_verifications(self, workspace_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM verification_records WHERE workspace_id=? ORDER BY verification_id DESC LIMIT ?",
                (workspace_id, int(limit)),
            ).fetchall()
        return [
            {
                "verification_id": int(row["verification_id"]),
                "batch_id": row["batch_id"],
                "mode": row["mode"],
                "status": row["status"],
                "elapsed_ms": float(row["elapsed_ms"]),
                "expected_summary": row["expected_summary"],
                "actual_summary": row["actual_summary"],
                "first_diff": _from_json(row["first_diff_json"], None),
                "mono_start_ns": int(row["mono_start_ns"]),
                "mono_end_ns": int(row["mono_end_ns"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def count_stale_batches(self, workspace_id: str) -> int:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM plan_batches WHERE workspace_id=? AND stale=1",
                (workspace_id,),
            ).fetchone()
            return int(row["count"])

    def attempts(self, workspace_id: str) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM edit_attempts WHERE workspace_id=? ORDER BY attempt_id",
                (workspace_id,),
            ).fetchall()
        return [self._attempt_row(row) for row in rows]
