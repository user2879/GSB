"""SQLite metadata, page files, checkpoints, and the recovery ledger."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .content import (
    canonical_json_bytes,
    deterministic_page_bytes,
    digest_bytes,
    digest_json,
    page_token,
)
from .errors import BusinessError, append_diagnostic, monotonic_ms, utc_now_iso
from .models import PageDecl, PreparedEntry, ReplayBudget, SampleSpec


GENESIS_HASH = "0" * 64


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS app_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pages (
    page_id TEXT PRIMARY KEY,
    size_bytes INTEGER NOT NULL,
    initial_version INTEGER NOT NULL,
    initial_digest TEXT NOT NULL,
    declaration_order INTEGER NOT NULL,
    page_table_version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS budget_params (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    global_slots INTEGER NOT NULL,
    per_tx_quota INTEGER NOT NULL,
    page_byte_budget INTEGER NOT NULL,
    checkpoint_interval INTEGER NOT NULL,
    params_version INTEGER NOT NULL,
    params_digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS wal_entries (
    lsn INTEGER PRIMARY KEY,
    txid TEXT NOT NULL,
    page_id TEXT NOT NULL REFERENCES pages(page_id),
    content_digest TEXT NOT NULL,
    commit_flag INTEGER NOT NULL,
    persist_flag INTEGER NOT NULL,
    target_version INTEGER NOT NULL,
    generated_digest TEXT NOT NULL,
    visible INTEGER NOT NULL,
    tx_group INTEGER NOT NULL,
    loaded_at_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS page_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    page_id TEXT NOT NULL REFERENCES pages(page_id),
    version INTEGER NOT NULL,
    lsn INTEGER NOT NULL,
    txid TEXT NOT NULL,
    digest TEXT NOT NULL,
    content_path TEXT NOT NULL,
    visible INTEGER NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('base', 'wal')),
    batch_id INTEGER,
    committed_ms INTEGER NOT NULL,
    UNIQUE (page_id, version)
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_page_versions_wal_lsn
    ON page_versions(lsn) WHERE kind = 'wal';
CREATE INDEX IF NOT EXISTS ix_page_versions_page_version
    ON page_versions(page_id, version);
CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    interval INTEGER NOT NULL,
    ledger_cursor INTEGER NOT NULL,
    page_count_since_checkpoint INTEGER NOT NULL,
    state_digest TEXT NOT NULL,
    state_json TEXT NOT NULL,
    created_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS replay_runs (
    run_id TEXT PRIMARY KEY,
    cursor_start INTEGER NOT NULL,
    cursor_end INTEGER,
    status TEXT NOT NULL,
    budget_json TEXT NOT NULL,
    started_ms INTEGER NOT NULL,
    finished_ms INTEGER,
    note TEXT
);
CREATE TABLE IF NOT EXISTS replay_batches (
    batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    start_lsn INTEGER NOT NULL,
    end_lsn INTEGER NOT NULL,
    page_count INTEGER NOT NULL,
    bytes_consumed INTEGER NOT NULL,
    ledger_cursor INTEGER NOT NULL,
    checkpoint_id INTEGER,
    committed_ms INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    ledger_hash TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS replay_tasks (
    task_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    txid TEXT NOT NULL,
    page_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    lsn_start INTEGER NOT NULL,
    lsn_end INTEGER NOT NULL,
    entry_count INTEGER NOT NULL,
    planned_bytes INTEGER NOT NULL,
    state TEXT NOT NULL,
    queued_ms INTEGER NOT NULL,
    wait_started_ms INTEGER,
    wait_ended_ms INTEGER,
    started_ms INTEGER,
    finished_ms INTEGER,
    committed_ms INTEGER,
    consumed_bytes INTEGER NOT NULL DEFAULT 0,
    batch_id INTEGER,
    detail_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_replay_tasks_run_order
    ON replay_tasks(run_id, queued_ms, task_id);
CREATE TABLE IF NOT EXISTS ledger_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    cursor INTEGER NOT NULL,
    batch_id INTEGER,
    task_id TEXT,
    prev_hash TEXT NOT NULL,
    event_hash TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshot_meta (
    snapshot_id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    format_version INTEGER NOT NULL,
    content_digest TEXT NOT NULL,
    log_digest TEXT NOT NULL,
    pages_digest TEXT NOT NULL,
    budget_digest TEXT NOT NULL,
    ledger_cursor INTEGER NOT NULL,
    created_ms INTEGER NOT NULL
);
"""


class Storage:
    """Owns one workspace and exposes explicit, mostly small transactions."""

    def __init__(self, workspace: str | os.PathLike[str]):
        self.workspace = Path(workspace).resolve()
        self.db_path = self.workspace / "state.sqlite3"
        self.pages_dir = self.workspace / "pages"
        self.snapshots_dir = self.workspace / "snapshots"
        self.logs_dir = self.workspace / "logs"
        self.inputs_dir = self.workspace / "inputs"
        self.diagnostic_log = self.logs_dir / "app.ndjson"
        self.ensure_dirs()

    def ensure_dirs(self) -> None:
        for path in (
            self.workspace,
            self.pages_dir,
            self.snapshots_dir,
            self.logs_dir,
            self.inputs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def path_for(self, relative_path: str) -> Path:
        return self.workspace / Path(relative_path)

    def connect(self, readonly: bool = False) -> sqlite3.Connection:
        if readonly:
            uri = f"{self.db_path.as_uri()}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=30.0)
        else:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 30000")
        if not readonly:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = FULL")
        return conn

    @contextmanager
    def read_connection(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect(readonly=True)
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self, conn: sqlite3.Connection | None = None) -> Iterator[sqlite3.Connection]:
        owns = conn is None
        connection = conn or self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.execute("COMMIT")
        except BaseException:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            if owns:
                connection.close()

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(SCHEMA_SQL)
        conn.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES('format_version', '1')"
        )
        conn.commit()

    def is_initialized(self) -> bool:
        if not self.db_path.exists():
            return False
        try:
            with self.read_connection() as conn:
                row = conn.execute(
                    "SELECT value FROM schema_meta WHERE key='format_version'"
                ).fetchone()
            return row is not None
        except sqlite3.Error:
            return False

    def remove_state(self, remove_snapshots: bool = False, preserve_logs: bool = False) -> None:
        for suffix in ("", "-wal", "-shm"):
            target = Path(str(self.db_path) + suffix)
            if target.exists():
                target.unlink()
        removable_dirs = [self.pages_dir, self.inputs_dir]
        if not preserve_logs:
            removable_dirs.append(self.logs_dir)
        for dirname in removable_dirs:
            if dirname.exists():
                shutil.rmtree(dirname)
        if remove_snapshots and self.snapshots_dir.exists():
            shutil.rmtree(self.snapshots_dir)
        self.ensure_dirs()

    def content_path_for(self, page_id: str, version: int) -> str:
        return f"pages/{page_token(page_id)}/v{version:08d}.bin"

    def write_page_file(self, relative_path: str, data: bytes) -> Path:
        target = self.path_for(relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp")
        with temp.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, target)
        return target

    def _set_meta(self, conn: sqlite3.Connection, key: str, value: Any) -> None:
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
        conn.execute(
            "INSERT INTO app_meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, text),
        )

    def get_meta(self, key: str, default: Any = None) -> Any:
        with self.read_connection() as conn:
            row = conn.execute("SELECT value FROM app_meta WHERE key=?", (key,)).fetchone()
        if row is None:
            return default
        return row["value"]

    def get_meta_int(self, key: str, default: int = 0) -> int:
        value = self.get_meta(key)
        return int(value) if value is not None else default

    def _last_event_hash(self, conn: sqlite3.Connection) -> str:
        row = conn.execute(
            "SELECT event_hash FROM ledger_events ORDER BY event_id DESC LIMIT 1"
        ).fetchone()
        return row["event_hash"] if row else GENESIS_HASH

    @staticmethod
    def _event_hash(
        event_type: str,
        cursor: int,
        batch_id: int | None,
        task_id: str | None,
        prev_hash: str,
        payload_json: str,
        created_ms: int,
    ) -> str:
        return digest_json(
            {
                "event_type": event_type,
                "cursor": cursor,
                "batch_id": batch_id,
                "task_id": task_id,
                "prev_hash": prev_hash,
                "payload_json": payload_json,
                "created_ms": created_ms,
            }
        )

    def _append_ledger_event(
        self,
        conn: sqlite3.Connection,
        event_type: str,
        cursor: int,
        payload: dict[str, Any],
        batch_id: int | None = None,
        task_id: str | None = None,
    ) -> tuple[int, str, str]:
        created_ms = monotonic_ms()
        prev_hash = self._last_event_hash(conn)
        payload_json = canonical_json_bytes(payload).decode("utf-8")
        event_hash = self._event_hash(
            event_type, cursor, batch_id, task_id, prev_hash, payload_json, created_ms
        )
        cursor_obj = conn.execute(
            "INSERT INTO ledger_events(event_type, cursor, batch_id, task_id, prev_hash, "
            "event_hash, payload_json, created_ms) VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event_type,
                cursor,
                batch_id,
                task_id,
                prev_hash,
                event_hash,
                payload_json,
                created_ms,
            ),
        )
        return int(cursor_obj.lastrowid), event_hash, prev_hash

    def load_fresh(self, sample: SampleSpec) -> None:
        """Replace workspace state with a fully validated sample."""

        self.remove_state(remove_snapshots=False, preserve_logs=True)
        schema_conn = self.connect()
        try:
            self._create_schema(schema_conn)
        finally:
            schema_conn.close()
        (self.inputs_dir / "wal.ndjson").write_text(sample.wal_text, encoding="utf-8", newline="\n")
        sample_document = {
            "format_version": 1,
            "source_sample": sample.source_sample,
            "pages": [
                {
                    "page_id": page.page_id,
                    "size_bytes": page.size_bytes,
                    "initial_version": page.initial_version,
                }
                for page in sample.pages
            ],
            "budget": sample.budget.as_dict(),
            "log_file": "wal.ndjson",
        }
        (self.inputs_dir / "sample.json").write_text(
            json.dumps(sample_document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        conn = self.connect()
        try:
            with self.transaction(conn):
                now = monotonic_ms()
                self._set_meta(conn, "format_version", 1)
                self._set_meta(conn, "ledger_cursor", sample.initial_cursor)
                self._set_meta(conn, "initial_cursor", sample.initial_cursor)
                self._set_meta(conn, "crash_point_lsn", sample.crash_point_lsn)
                self._set_meta(conn, "wal_last_lsn", sample.last_lsn)
                self._set_meta(conn, "active_wal_digest", sample.wal_digest)
                self._set_meta(conn, "active_pages_digest", sample.pages_digest)
                self._set_meta(conn, "active_budget_digest", sample.budget_digest)
                self._set_meta(conn, "active_sample_digest", sample.sample_digest)
                self._set_meta(conn, "source_sample", sample.source_sample)
                self._set_meta(conn, "log_file", sample.log_file)
                self._set_meta(conn, "page_table_version", 1)
                self._set_meta(conn, "run_counter", 0)
                self._set_meta(conn, "loaded_at", utc_now_iso())

                conn.execute(
                    "INSERT INTO budget_params(singleton, global_slots, per_tx_quota, "
                    "page_byte_budget, checkpoint_interval, params_version, params_digest) "
                    "VALUES(1, ?, ?, ?, ?, 1, ?)",
                    (
                        sample.budget.global_slots,
                        sample.budget.per_tx_quota,
                        sample.budget.page_byte_budget,
                        sample.budget.checkpoint_interval,
                        sample.budget_digest,
                    ),
                )

                for order, page in enumerate(sample.pages):
                    base = deterministic_page_bytes(page.page_id, page.size_bytes, page.initial_version, 0)
                    base_digest = digest_bytes(base)
                    content_path = self.content_path_for(page.page_id, page.initial_version)
                    self.write_page_file(content_path, base)
                    conn.execute(
                        "INSERT INTO pages(page_id, size_bytes, initial_version, initial_digest, "
                        "declaration_order, page_table_version) VALUES(?, ?, ?, ?, ?, 1)",
                        (page.page_id, page.size_bytes, page.initial_version, base_digest, order),
                    )
                    conn.execute(
                        "INSERT INTO page_versions(page_id, version, lsn, txid, digest, "
                        "content_path, visible, kind, committed_ms) VALUES(?, ?, 0, 'INIT', ?, ?, 1, 'base', ?)",
                        (page.page_id, page.initial_version, base_digest, content_path, now),
                    )

                tx_group = 0
                current_tx: str | None = None
                for entry in sample.entries:
                    if entry.txid != current_tx:
                        tx_group += 1
                        current_tx = entry.txid
                    conn.execute(
                        "INSERT INTO wal_entries(lsn, txid, page_id, content_digest, commit_flag, "
                        "persist_flag, target_version, generated_digest, visible, tx_group, loaded_at_ms) "
                        "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            entry.lsn,
                            entry.txid,
                            entry.page_id,
                            entry.content_digest,
                            1 if entry.commit else 0,
                            1 if entry.persist else 0,
                            entry.target_version,
                            entry.generated_digest,
                            1 if entry.visible else 0,
                            tx_group,
                            now,
                        ),
                    )

                self._append_ledger_event(
                    conn,
                    "LOAD",
                    sample.initial_cursor,
                    {
                        "source_sample": sample.source_sample,
                        "log_file": sample.log_file,
                        "wal_digest": sample.wal_digest,
                        "pages_digest": sample.pages_digest,
                        "budget_digest": sample.budget_digest,
                        "entry_count": len(sample.entries),
                        "crash_point_lsn": sample.crash_point_lsn,
                    },
                )
        finally:
            conn.close()

    def budget(self) -> ReplayBudget:
        with self.read_connection() as conn:
            row = conn.execute("SELECT * FROM budget_params WHERE singleton=1").fetchone()
        if row is None:
            raise BusinessError("E_STATE", "workspace has no budget parameters")
        return ReplayBudget(
            global_slots=row["global_slots"],
            per_tx_quota=row["per_tx_quota"],
            page_byte_budget=row["page_byte_budget"],
            checkpoint_interval=row["checkpoint_interval"],
        )

    def ledger_budget(self) -> tuple[ReplayBudget, str]:
        """Return the budget recorded by the ledger, not merely current rows.

        The latest run's serialized budget is authoritative for recovery.  If
        the mutable parameter table no longer matches it, recovery stops with
        E_REPLAY rather than silently changing semantics.
        """

        with self.read_connection() as conn:
            row = conn.execute("SELECT * FROM budget_params WHERE singleton=1").fetchone()
            if row is None:
                raise BusinessError("E_STATE", "workspace has no budget parameters")
            table_budget = ReplayBudget(
                global_slots=row["global_slots"],
                per_tx_quota=row["per_tx_quota"],
                page_byte_budget=row["page_byte_budget"],
                checkpoint_interval=row["checkpoint_interval"],
            )
            active_digest = conn.execute(
                "SELECT value FROM app_meta WHERE key='active_budget_digest'"
            ).fetchone()
            computed_digest = digest_json(table_budget.as_dict())
            if active_digest is None or active_digest["value"] != computed_digest:
                raise BusinessError(
                    "E_REPLAY",
                    "budget parameter table no longer matches the active ledger digest",
                    {
                        "first_mismatch_lsn": 0,
                        "lsn": 0,
                        "page_id": None,
                        "field": "budget_params_digest",
                        "expected": active_digest["value"] if active_digest else None,
                        "actual": computed_digest,
                        "ledger_record": {"event_type": "LOAD", "cursor": 0},
                    },
                )
            prior = conn.execute(
                "SELECT run_id, budget_json FROM replay_runs ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
        if prior is None:
            return table_budget, "budget_params_table"
        try:
            recorded = json.loads(prior["budget_json"])
        except json.JSONDecodeError:
            raise BusinessError(
                "E_REPLAY",
                "recorded run budget is not valid JSON",
                {
                    "first_mismatch_lsn": 0,
                    "lsn": 0,
                    "page_id": None,
                    "field": "budget_json",
                    "run_id": prior["run_id"],
                    "ledger_record": {"event_type": "REPLAY_RUN", "run_id": prior["run_id"]},
                },
            )
        if recorded != table_budget.as_dict():
            raise BusinessError(
                "E_REPLAY",
                "current budget differs from the ledger-recorded recovery budget",
                {
                    "first_mismatch_lsn": 0,
                    "lsn": 0,
                    "page_id": None,
                    "field": "budget_params",
                    "expected": recorded,
                    "actual": table_budget.as_dict(),
                    "ledger_record": {"event_type": "REPLAY_RUN", "run_id": prior["run_id"]},
                },
            )
        return table_budget, f"replay_runs:{prior['run_id']}"

    def page_declarations(self) -> list[PageDecl]:
        with self.read_connection() as conn:
            rows = conn.execute(
                "SELECT page_id, size_bytes, initial_version FROM pages ORDER BY page_id"
            ).fetchall()
        return [
            PageDecl(page_id=row["page_id"], size_bytes=row["size_bytes"], initial_version=row["initial_version"])
            for row in rows
        ]

    def prepared_entries(self, after_lsn: int = 0, until_lsn: int | None = None) -> list[PreparedEntry]:
        query = "SELECT * FROM wal_entries WHERE lsn > ?"
        params: list[Any] = [after_lsn]
        if until_lsn is not None:
            query += " AND lsn <= ?"
            params.append(until_lsn)
        query += " ORDER BY lsn"
        with self.read_connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            PreparedEntry(
                lsn=row["lsn"],
                txid=row["txid"],
                page_id=row["page_id"],
                content_digest=row["content_digest"],
                commit=bool(row["commit_flag"]),
                persist=bool(row["persist_flag"]),
                target_version=row["target_version"],
                generated_digest=row["generated_digest"],
                visible=bool(row["visible"]),
            )
            for row in rows
        ]

    def all_prepared_entries(self) -> list[PreparedEntry]:
        return self.prepared_entries(0)

    def ledger_cursor(self) -> int:
        return self.get_meta_int("ledger_cursor", 0)

    def next_run_id(self) -> str:
        with self.transaction() as conn:
            row = conn.execute("SELECT value FROM app_meta WHERE key='run_counter'").fetchone()
            counter = int(row["value"]) + 1 if row else 1
            self._set_meta(conn, "run_counter", counter)
        return f"run-{counter:04d}"

    def start_run(self, run_id: str, cursor_start: int, budget: ReplayBudget) -> None:
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO replay_runs(run_id, cursor_start, cursor_end, status, budget_json, "
                "started_ms, finished_ms, note) VALUES(?, ?, NULL, 'running', ?, ?, NULL, NULL)",
                (run_id, cursor_start, json.dumps(budget.as_dict(), sort_keys=True), monotonic_ms()),
            )

    def insert_tasks(self, tasks: Iterable[dict[str, Any]]) -> None:
        with self.transaction() as conn:
            for task in tasks:
                conn.execute(
                    "INSERT INTO replay_tasks(task_id, run_id, txid, page_id, chunk_index, "
                    "lsn_start, lsn_end, entry_count, planned_bytes, state, queued_ms, "
                    "wait_started_ms, wait_ended_ms, started_ms, finished_ms, committed_ms, "
                    "consumed_bytes, batch_id, detail_json) "
                    "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, NULL, NULL, NULL, NULL, NULL, 0, NULL, ?)",
                    (
                        task["task_key"],
                        task["run_id"],
                        task["txid"],
                        task["page_id"],
                        task["chunk_index"],
                        task["lsn_start"],
                        task["lsn_end"],
                        task["entry_count"],
                        task["planned_bytes"],
                        monotonic_ms(),
                        json.dumps(task.get("detail", {}), ensure_ascii=False, sort_keys=True),
                    ),
                )

    def task_started(
        self, task_id: str, wait_started_ms: int | None, wait_ended_ms: int | None
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE replay_tasks SET state='running', wait_started_ms=COALESCE(wait_started_ms, ?), "
                "wait_ended_ms=?, started_ms=? WHERE task_id=?",
                (wait_started_ms, wait_ended_ms, monotonic_ms(), task_id),
            )

    def task_waiting(self, task_id: str, wait_started_ms: int) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE replay_tasks SET wait_started_ms=COALESCE(wait_started_ms, ?) WHERE task_id=?",
                (wait_started_ms, task_id),
            )

    def task_staged(self, task_id: str) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE replay_tasks SET state='staged', finished_ms=? WHERE task_id=?",
                (monotonic_ms(), task_id),
            )

    def recover_orphan_tasks(self) -> int:
        with self.transaction() as conn:
            rows = conn.execute(
                "SELECT task_id FROM replay_tasks WHERE state IN "
                "('queued', 'running', 'staged', 'partial')"
            ).fetchall()
            if not rows:
                return 0
            task_rows = conn.execute(
                "SELECT task_id, detail_json FROM replay_tasks WHERE state IN "
                "('queued', 'running', 'staged', 'partial')"
            ).fetchall()
            for task_row in task_rows:
                try:
                    detail = json.loads(task_row["detail_json"])
                except json.JSONDecodeError:
                    detail = {}
                detail["recovery_note"] = "not covered by a committed batch; rerun required"
                conn.execute(
                    "UPDATE replay_tasks SET state='unknown', detail_json=? WHERE task_id=?",
                    (
                        json.dumps(detail, ensure_ascii=False, sort_keys=True),
                        task_row["task_id"],
                    ),
                )
            conn.execute(
                "UPDATE replay_runs SET status='interrupted_unknown', finished_ms=?, "
                "note='process ended before a complete run report' WHERE status='running'",
                (monotonic_ms(),),
            )
            self._append_ledger_event(
                conn,
                "RECOVERY_INCOMPLETE",
                self._cursor_in_conn(conn),
                {
                    "unknown_task_ids": sorted(row["task_id"] for row in rows),
                    "note": "only fully committed batches are authoritative; no exit report is claimed",
                },
            )
        return len(rows)

    def _cursor_in_conn(self, conn: sqlite3.Connection) -> int:
        row = conn.execute("SELECT value FROM app_meta WHERE key='ledger_cursor'").fetchone()
        return int(row["value"]) if row else 0

    def visible_page_state(self, conn: sqlite3.Connection | None = None) -> list[dict[str, Any]]:
        owns = conn is None
        connection = conn or self.connect(readonly=True)
        try:
            rows = connection.execute(
                "SELECT p.page_id, p.size_bytes, p.initial_version, "
                "(SELECT v.version FROM page_versions v WHERE v.page_id=p.page_id AND v.visible=1 "
                " ORDER BY v.version DESC LIMIT 1) AS visible_version, "
                "(SELECT v.digest FROM page_versions v WHERE v.page_id=p.page_id AND v.visible=1 "
                " ORDER BY v.version DESC LIMIT 1) AS visible_digest, "
                "(SELECT v.content_path FROM page_versions v WHERE v.page_id=p.page_id AND v.visible=1 "
                " ORDER BY v.version DESC LIMIT 1) AS content_path, "
                "(SELECT MAX(v.version) FROM page_versions v WHERE v.page_id=p.page_id) AS applied_version "
                "FROM pages p ORDER BY p.page_id"
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            if owns:
                connection.close()

    def commit_batch(
        self,
        run_id: str,
        effects: Sequence[dict[str, Any]],
        task_updates: Sequence[dict[str, Any]],
        *,
        is_final: bool,
        checkpoint_interval: int,
    ) -> dict[str, Any]:
        """Atomically commit page versions, task outcomes, checkpoint, cursor and ledger."""

        if not effects:
            raise BusinessError("E_STATE", "cannot commit an empty replay batch")

        # Write versioned files before the SQLite transaction.  A file left by a
        # crash before commit is an unreferenced orphan and is never treated as
        # committed page state.
        for effect in effects:
            entry: PreparedEntry = effect.entry
            self.write_page_file(effect.content_path, effect.data)

        now = monotonic_ms()
        conn = self.connect()
        try:
            with self.transaction(conn):
                start_lsn = effects[0].entry.lsn
                end_lsn = effects[-1].entry.lsn
                cursor = self._cursor_in_conn(conn)
                if start_lsn != cursor + 1:
                    raise BusinessError(
                        "E_REPLAY",
                        "batch is not contiguous with the ledger cursor",
                        {
                            "first_mismatch_lsn": start_lsn,
                            "expected_lsn": cursor + 1,
                            "ledger_cursor": cursor,
                        },
                    )
                for expected_lsn, effect in zip(range(start_lsn, end_lsn + 1), effects):
                    entry = effect.entry
                    if entry.lsn != expected_lsn:
                        raise BusinessError(
                            "E_REPLAY",
                            "batch contains a non-contiguous LSN",
                            {
                                "first_mismatch_lsn": entry.lsn,
                                "expected_lsn": expected_lsn,
                                "field": "lsn",
                            },
                        )
                    max_row = conn.execute(
                        "SELECT MAX(version) AS max_version FROM page_versions WHERE page_id=?",
                        (entry.page_id,),
                    ).fetchone()
                    max_version = max_row["max_version"] if max_row and max_row["max_version"] is not None else -1
                    if entry.target_version <= max_version:
                        raise BusinessError(
                            "E_REPLAY",
                            "page version would not increase monotonically",
                            {
                                "first_mismatch_lsn": entry.lsn,
                                "page_id": entry.page_id,
                                "field": "version",
                                "current_version": max_version,
                                "attempted_version": entry.target_version,
                            },
                        )
                    conn.execute(
                        "INSERT INTO page_versions(page_id, version, lsn, txid, digest, content_path, "
                        "visible, kind, batch_id, committed_ms) VALUES(?, ?, ?, ?, ?, ?, ?, 'wal', ?, ?)",
                        (
                            entry.page_id,
                            entry.target_version,
                            entry.lsn,
                            entry.txid,
                            entry.generated_digest,
                            effect.content_path,
                            1 if entry.visible else 0,
                            None,
                            now,
                        ),
                    )

                page_count = len(effects)
                bytes_consumed = sum(len(effect.data) for effect in effects)
                new_cursor = end_lsn
                checkpoint_state = self.visible_page_state(conn)
                state_payload = {
                    "cursor": new_cursor,
                    "pages": [
                        {
                            "page_id": item["page_id"],
                            "version": item["visible_version"],
                            "digest": item["visible_digest"],
                        }
                        for item in checkpoint_state
                    ],
                }
                state_json = canonical_json_bytes(state_payload).decode("utf-8")
                state_digest = digest_json(state_payload)
                checkpoint_obj = conn.execute(
                    "INSERT INTO checkpoints(run_id, interval, ledger_cursor, "
                    "page_count_since_checkpoint, state_digest, state_json, created_ms) "
                    "VALUES(?, ?, ?, ?, ?, ?, ?)",
                    (run_id, checkpoint_interval, new_cursor, page_count, state_digest, state_json, now),
                )
                checkpoint_id = int(checkpoint_obj.lastrowid)

                payload_core = {
                    "run_id": run_id,
                    "start_lsn": start_lsn,
                    "end_lsn": end_lsn,
                    "page_count": page_count,
                    "bytes_consumed": bytes_consumed,
                    "ledger_cursor": new_cursor,
                    "checkpoint_id": checkpoint_id,
                    "page_versions": [
                        {
                            "lsn": effect.entry.lsn,
                            "page_id": effect.entry.page_id,
                            "version": effect.entry.target_version,
                            "digest": effect.entry.generated_digest,
                            "visible": effect.entry.visible,
                        }
                        for effect in effects
                    ],
                    "task_outcomes": list(task_updates),
                    "budget_consumed": {"bytes": bytes_consumed, "page_versions": page_count},
                }
                payload_json = canonical_json_bytes(payload_core).decode("utf-8")
                batch_obj = conn.execute(
                    "INSERT INTO replay_batches(run_id, start_lsn, end_lsn, page_count, "
                    "bytes_consumed, ledger_cursor, checkpoint_id, committed_ms, payload_json, "
                    "prev_hash, ledger_hash) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, '', '')",
                    (
                        run_id,
                        start_lsn,
                        end_lsn,
                        page_count,
                        bytes_consumed,
                        new_cursor,
                        checkpoint_id,
                        now,
                        payload_json,
                    ),
                )
                batch_id = int(batch_obj.lastrowid)
                conn.execute(
                    "UPDATE page_versions SET batch_id=? WHERE kind='wal' AND lsn BETWEEN ? AND ?",
                    (batch_id, start_lsn, end_lsn),
                )

                for update in task_updates:
                    conn.execute(
                        "UPDATE replay_tasks SET state=?, consumed_bytes=?, batch_id=?, "
                        "committed_ms=CASE WHEN ?='committed' THEN ? ELSE committed_ms END "
                        "WHERE task_id=?",
                        (
                            update["state"],
                            update["consumed_bytes"],
                            batch_id,
                            update["state"],
                            now,
                            update["task_id"],
                        ),
                    )

                self._set_meta(conn, "ledger_cursor", new_cursor)
                event_payload = dict(payload_core)
                event_payload["batch_id"] = batch_id
                event_payload["checkpoint_state_digest"] = state_digest
                _, event_hash, prev_hash = self._append_ledger_event(
                    conn, "REPLAY_BATCH", new_cursor, event_payload, batch_id=batch_id
                )
                conn.execute(
                    "UPDATE replay_batches SET prev_hash=?, ledger_hash=? WHERE batch_id=?",
                    (prev_hash, event_hash, batch_id),
                )
                if is_final:
                    conn.execute(
                        "UPDATE replay_runs SET cursor_end=?, status='completed', finished_ms=?, "
                        "note='all available WAL entries replayed' WHERE run_id=?",
                        (new_cursor, monotonic_ms(), run_id),
                    )
                else:
                    conn.execute(
                        "UPDATE replay_runs SET cursor_end=? WHERE run_id=?",
                        (new_cursor, run_id),
                    )
            return {
                "batch_id": batch_id,
                "start_lsn": start_lsn,
                "end_lsn": end_lsn,
                "page_count": page_count,
                "bytes_consumed": bytes_consumed,
                "checkpoint_id": checkpoint_id,
                "ledger_hash": event_hash,
            }
        finally:
            conn.close()

    def complete_empty_run(self, run_id: str, cursor: int) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE replay_runs SET cursor_end=?, status='completed', finished_ms=?, "
                "note='no entries remained after the ledger cursor' WHERE run_id=?",
                (cursor, monotonic_ms(), run_id),
            )

    def save_checkpoint(self, label: str = "manual") -> dict[str, Any]:
        """Persist an explicit checkpoint at the current committed cursor."""

        now = monotonic_ms()
        conn = self.connect()
        try:
            with self.transaction(conn):
                cursor = self._cursor_in_conn(conn)
                state = self.visible_page_state(conn)
                state_payload = {
                    "cursor": cursor,
                    "pages": [
                        {
                            "page_id": item["page_id"],
                            "version": item["visible_version"],
                            "digest": item["visible_digest"],
                        }
                        for item in state
                    ],
                }
                state_json = canonical_json_bytes(state_payload).decode("utf-8")
                state_digest = digest_json(state_payload)
                obj = conn.execute(
                    "INSERT INTO checkpoints(run_id, interval, ledger_cursor, "
                    "page_count_since_checkpoint, state_digest, state_json, created_ms) "
                    "VALUES(?, 0, ?, 0, ?, ?, ?)",
                    (label, cursor, state_digest, state_json, now),
                )
                checkpoint_id = int(obj.lastrowid)
                self._append_ledger_event(
                    conn,
                    "CHECKPOINT_SAVED",
                    cursor,
                    {
                        "checkpoint_id": checkpoint_id,
                        "label": label,
                        "state_digest": state_digest,
                        "page_count": len(state_payload["pages"]),
                    },
                )
            return {
                "ok": True,
                "checkpoint_id": checkpoint_id,
                "ledger_cursor": cursor,
                "state_digest": state_digest,
                "label": label,
            }
        finally:
            conn.close()

    def fail_run(self, run_id: str, error: BusinessError) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE replay_runs SET status='failed', finished_ms=?, note=? WHERE run_id=?",
                (monotonic_ms(), f"{error.code}: {error.message}", run_id),
            )
            self._append_ledger_event(
                conn,
                "RUN_FAILED",
                self._cursor_in_conn(conn),
                {"run_id": run_id, "code": error.code, "message": error.message, "context": dict(error.context)},
            )

    def status(self) -> dict[str, Any]:
        """Read-only dashboard snapshot; ordering is deterministic."""

        if not self.is_initialized():
            return {"initialized": False, "workspace": str(self.workspace)}
        with self.read_connection() as conn:
            cursor = self._cursor_in_conn(conn)
            pages = self.visible_page_state(conn)
            counts = {
                "pages": conn.execute("SELECT COUNT(*) AS n FROM pages").fetchone()["n"],
                "wal_entries": conn.execute("SELECT COUNT(*) AS n FROM wal_entries").fetchone()["n"],
                "page_versions": conn.execute(
                    "SELECT COUNT(*) AS n FROM page_versions WHERE kind='wal'"
                ).fetchone()["n"],
                "invisible_page_versions": conn.execute(
                    "SELECT COUNT(*) AS n FROM page_versions WHERE kind='wal' AND visible=0"
                ).fetchone()["n"],
                "batches": conn.execute("SELECT COUNT(*) AS n FROM replay_batches").fetchone()["n"],
                "checkpoints": conn.execute("SELECT COUNT(*) AS n FROM checkpoints").fetchone()["n"],
                "unknown_tasks": conn.execute(
                    "SELECT COUNT(*) AS n FROM replay_tasks WHERE state='unknown'"
                ).fetchone()["n"],
                "active_tasks": conn.execute(
                    "SELECT COUNT(*) AS n FROM replay_tasks WHERE state IN "
                    "('queued','running','staged','partial')"
                ).fetchone()["n"],
            }
            last_event = conn.execute(
                "SELECT event_hash FROM ledger_events ORDER BY event_id DESC LIMIT 1"
            ).fetchone()
            last_run = conn.execute(
                "SELECT run_id, status, cursor_start, cursor_end FROM replay_runs "
                "ORDER BY rowid DESC LIMIT 1"
            ).fetchone()
            last_checkpoint = conn.execute(
                "SELECT checkpoint_id, ledger_cursor, state_digest FROM checkpoints "
                "ORDER BY checkpoint_id DESC LIMIT 1"
            ).fetchone()
            budget_row = conn.execute("SELECT * FROM budget_params WHERE singleton=1").fetchone()
            schema_row = conn.execute(
                "SELECT value FROM schema_meta WHERE key='format_version'"
            ).fetchone()
            event_count = conn.execute("SELECT COUNT(*) AS n FROM ledger_events").fetchone()["n"]
            ledger_version_row = conn.execute(
                "SELECT MAX(event_id) AS version FROM ledger_events"
            ).fetchone()
            snapshots = conn.execute(
                "SELECT snapshot_id, file_path, format_version, content_digest, ledger_cursor "
                "FROM snapshot_meta ORDER BY snapshot_id"
            ).fetchall()
            meta = {
                row["key"]: row["value"]
                for row in conn.execute("SELECT key, value FROM app_meta ORDER BY key").fetchall()
            }
        return {
            "initialized": True,
            "workspace": str(self.workspace),
            "cursor": cursor,
            "initial_cursor": int(meta.get("initial_cursor", "0")),
            "crash_point_lsn": int(meta.get("crash_point_lsn", "0")),
            "wal_last_lsn": int(meta.get("wal_last_lsn", "0")),
            "page_table_version": int(meta.get("page_table_version", "1")),
            "schema_format_version": int(schema_row["value"]) if schema_row else None,
            "budget_params_version": budget_row["params_version"] if budget_row else None,
            "ledger_event_count": event_count,
            "ledger_version": int(ledger_version_row["version"] or 0) if ledger_version_row else 0,
            "pages": pages,
            "counts": counts,
            "last_run": dict(last_run) if last_run else None,
            "last_checkpoint": dict(last_checkpoint) if last_checkpoint else None,
            "snapshots": [dict(row) for row in snapshots],
            "ledger_head": last_event["event_hash"] if last_event else None,
            "active_log_digest": meta.get("active_wal_digest"),
            "active_pages_digest": meta.get("active_pages_digest"),
            "active_budget_digest": meta.get("active_budget_digest"),
        }

    def log_entries(self, start_lsn: int | None = None, end_lsn: int | None = None) -> list[dict[str, Any]]:
        start = max(1, int(start_lsn or 1))
        end = int(end_lsn) if end_lsn is not None else 2**63 - 1
        with self.read_connection() as conn:
            rows = conn.execute(
                "SELECT lsn, txid, page_id, content_digest, commit_flag, persist_flag, "
                "target_version, visible, tx_group FROM wal_entries "
                "WHERE lsn BETWEEN ? AND ? ORDER BY lsn",
                (start, end),
            ).fetchall()
        return [dict(row) | {"commit": bool(row["commit_flag"]), "persist": bool(row["persist_flag"]), "visible": bool(row["visible"])} for row in rows]

    def page_versions(self, include_invisible: bool = True) -> list[dict[str, Any]]:
        where = "" if include_invisible else " WHERE visible=1"
        with self.read_connection() as conn:
            rows = conn.execute(
                "SELECT id, page_id, version, lsn, txid, digest, content_path, visible, kind, "
                "batch_id, committed_ms FROM page_versions" + where + " ORDER BY page_id, version"
            ).fetchall()
        return [dict(row) | {"visible": bool(row["visible"])} for row in rows]

    def tasks(self, run_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM replay_tasks"
        params: list[Any] = []
        if run_id:
            query += " WHERE run_id=?"
            params.append(run_id)
        query += " ORDER BY run_id, queued_ms, task_id"
        with self.read_connection() as conn:
            rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def batches(self) -> list[dict[str, Any]]:
        with self.read_connection() as conn:
            rows = conn.execute(
                "SELECT batch_id, run_id, start_lsn, end_lsn, page_count, bytes_consumed, "
                "ledger_cursor, checkpoint_id, committed_ms, prev_hash, ledger_hash "
                "FROM replay_batches ORDER BY batch_id"
            ).fetchall()
        return [dict(row) for row in rows]

    def checkpoint_rows(self) -> list[dict[str, Any]]:
        with self.read_connection() as conn:
            rows = conn.execute(
                "SELECT checkpoint_id, run_id, interval, ledger_cursor, "
                "page_count_since_checkpoint, state_digest, state_json, created_ms "
                "FROM checkpoints ORDER BY checkpoint_id"
            ).fetchall()
        return [dict(row) for row in rows]

    def ledger_rows(self) -> list[dict[str, Any]]:
        with self.read_connection() as conn:
            rows = conn.execute(
                "SELECT event_id, event_type, cursor, batch_id, task_id, prev_hash, event_hash, "
                "payload_json, created_ms FROM ledger_events ORDER BY event_id"
            ).fetchall()
        return [dict(row) for row in rows]

    def verify_ledger(self) -> dict[str, Any]:
        rows = self.ledger_rows()
        previous = GENESIS_HASH
        for row in rows:
            if row["prev_hash"] != previous:
                return {
                    "ok": False,
                    "event_id": row["event_id"],
                    "field": "prev_hash",
                    "expected": previous,
                    "actual": row["prev_hash"],
                }
            expected_hash = self._event_hash(
                row["event_type"],
                row["cursor"],
                row["batch_id"],
                row["task_id"],
                row["prev_hash"],
                row["payload_json"],
                row["created_ms"],
            )
            if expected_hash != row["event_hash"]:
                return {
                    "ok": False,
                    "event_id": row["event_id"],
                    "field": "event_hash",
                    "expected": expected_hash,
                    "actual": row["event_hash"],
                }
            previous = row["event_hash"]
        with self.read_connection() as conn:
            first_wal = conn.execute("SELECT MIN(lsn) AS first_lsn FROM wal_entries").fetchone()
            initial_cursor = (
                int(first_wal["first_lsn"]) - 1
                if first_wal is not None and first_wal["first_lsn"] is not None
                else 0
            )
            cursor_row = conn.execute(
                "SELECT value FROM app_meta WHERE key='ledger_cursor'"
            ).fetchone()
            metadata_cursor = int(cursor_row["value"]) if cursor_row else 0
            batch_rows = conn.execute(
                "SELECT b.*, e.prev_hash AS event_prev_hash, e.event_hash AS event_hash, "
                "e.payload_json AS event_payload_json FROM replay_batches b "
                "LEFT JOIN ledger_events e ON e.batch_id=b.batch_id AND e.event_type='REPLAY_BATCH' "
                "ORDER BY b.batch_id"
            ).fetchall()
            checkpoint_map = {
                row["checkpoint_id"]: dict(row)
                for row in conn.execute("SELECT * FROM checkpoints ORDER BY checkpoint_id").fetchall()
            }
            orphan_events = conn.execute(
                "SELECT e.event_id, e.batch_id FROM ledger_events e "
                "LEFT JOIN replay_batches b ON b.batch_id=e.batch_id "
                "WHERE e.event_type='REPLAY_BATCH' AND b.batch_id IS NULL "
                "ORDER BY e.event_id LIMIT 1"
            ).fetchone()
        if orphan_events is not None:
            return {
                "ok": False,
                "event_id": orphan_events["event_id"],
                "batch_id": orphan_events["batch_id"],
                "field": "orphan_batch_event",
            }
        if batch_rows:
            if batch_rows[-1]["ledger_cursor"] != metadata_cursor:
                return {
                    "ok": False,
                    "batch_id": batch_rows[-1]["batch_id"],
                    "field": "metadata_ledger_cursor",
                    "expected": batch_rows[-1]["ledger_cursor"],
                    "actual": metadata_cursor,
                }
        elif metadata_cursor != initial_cursor:
            return {
                "ok": False,
                "field": "batches_missing",
                "expected": "one or more committed batches",
                "actual": metadata_cursor,
            }
        previous_end = initial_cursor
        for row in batch_rows:
            if row["event_payload_json"] is None:
                return {"ok": False, "batch_id": row["batch_id"], "field": "batch_ledger_event"}
            if row["event_hash"] != row["ledger_hash"] or row["event_prev_hash"] != row["prev_hash"]:
                return {
                    "ok": False,
                    "batch_id": row["batch_id"],
                    "field": "batch_ledger_hash",
                    "expected": row["event_hash"],
                    "actual": row["ledger_hash"],
                }
            if row["start_lsn"] != previous_end + 1:
                return {
                    "ok": False,
                    "batch_id": row["batch_id"],
                    "field": "batch_start_lsn",
                    "expected": previous_end + 1,
                    "actual": row["start_lsn"],
                }
            if row["page_count"] != row["end_lsn"] - row["start_lsn"] + 1:
                return {"ok": False, "batch_id": row["batch_id"], "field": "batch_page_count"}
            if row["ledger_cursor"] != row["end_lsn"]:
                return {
                    "ok": False,
                    "batch_id": row["batch_id"],
                    "field": "batch_ledger_cursor",
                    "expected": row["end_lsn"],
                    "actual": row["ledger_cursor"],
                }
            try:
                payload = json.loads(row["event_payload_json"])
            except json.JSONDecodeError:
                return {"ok": False, "batch_id": row["batch_id"], "field": "batch_event_payload"}
            payload_checks = {
                "run_id": row["run_id"],
                "start_lsn": row["start_lsn"],
                "end_lsn": row["end_lsn"],
                "page_count": row["page_count"],
                "bytes_consumed": row["bytes_consumed"],
                "ledger_cursor": row["ledger_cursor"],
                "checkpoint_id": row["checkpoint_id"],
            }
            for field, expected in payload_checks.items():
                if payload.get(field) != expected:
                    return {
                        "ok": False,
                        "batch_id": row["batch_id"],
                        "field": f"batch_payload_{field}",
                        "expected": expected,
                        "actual": payload.get(field),
                    }
            if payload.get("budget_consumed", {}).get("bytes") != row["bytes_consumed"]:
                return {"ok": False, "batch_id": row["batch_id"], "field": "batch_payload_budget"}
            if len(payload.get("page_versions", [])) != row["page_count"]:
                return {"ok": False, "batch_id": row["batch_id"], "field": "batch_payload_pages"}
            checkpoint = checkpoint_map.get(row["checkpoint_id"])
            if checkpoint is None or checkpoint["ledger_cursor"] != row["ledger_cursor"]:
                return {"ok": False, "batch_id": row["batch_id"], "field": "batch_checkpoint"}
            if payload.get("checkpoint_state_digest") != checkpoint["state_digest"]:
                return {"ok": False, "batch_id": row["batch_id"], "field": "batch_checkpoint_digest"}
            previous_end = row["end_lsn"]
        return {"ok": True, "event_count": len(rows), "head": previous}

    def record_error(self, error: BusinessError, **extra: Any) -> None:
        try:
            append_diagnostic(self.diagnostic_log, {"level": "ERROR", **error.as_dict(), **extra})
        except OSError:
            pass

    def dump_tables(self) -> dict[str, Any]:
        """Return compact stable rows used by the status CLI and tests."""

        return {
            "status": self.status(),
            "batches": self.batches(),
            "tasks": self.tasks(),
            "checkpoints": self.checkpoint_rows(),
            "ledger": self.ledger_rows(),
        }
