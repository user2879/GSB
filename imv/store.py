from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import BusinessError, fail
from .model import Definitions, definition_summary, rows_from_payload
from .util import canonical_json_text, digest_obj, wall_ms


SCHEMA_VERSION = 1
LEDGER_FORMAT_VERSION = 1
SNAPSHOT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class CommitReceipt:
    batch_no: int
    cursor: int
    ledger_format_version: int
    content_digest: str
    snapshot_version: int
    snapshot_digest: str


@dataclass(frozen=True)
class SnapshotAudit:
    snapshot_version: int
    ledger_format_version: int
    stored_digest: str
    current_digest: str
    matches: bool
    recovered: bool
    error: BusinessError | None


@dataclass(frozen=True)
class PersistedState:
    param_version: int
    cursor: int
    batch_counter: int
    base_rows_payload: dict[str, Any]
    base_digest: str
    changes_digest: str
    results: dict[str, dict[str, Any]]
    leases: dict[str, dict[str, Any]]
    snapshot_version: int
    snapshot_digest: str


class LedgerStore:
    """SQLite ledger whose batch write is one transaction and one snapshot cut."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            str(self.path),
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute("PRAGMA busy_timeout=30000")
        self._create_schema()

    def _create_schema(self) -> None:
        statements = [
            """CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS param_versions (
                version INTEGER PRIMARY KEY,
                fingerprint TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                created_wall_ms INTEGER NOT NULL,
                is_current INTEGER NOT NULL DEFAULT 0
            )""",
            """CREATE TABLE IF NOT EXISTS view_defs (
                param_version INTEGER NOT NULL,
                view_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                digest TEXT NOT NULL,
                is_current INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (param_version, view_id)
            )""",
            """CREATE TABLE IF NOT EXISTS view_results (
                param_version INTEGER NOT NULL,
                view_id TEXT NOT NULL,
                result_version INTEGER NOT NULL,
                cursor INTEGER NOT NULL,
                result_json TEXT NOT NULL,
                result_digest TEXT NOT NULL,
                stale INTEGER NOT NULL DEFAULT 0,
                inconsistent INTEGER NOT NULL DEFAULT 0,
                updated_wall_ms INTEGER NOT NULL,
                PRIMARY KEY (param_version, view_id)
            )""",
            """CREATE TABLE IF NOT EXISTS leases (
                param_version INTEGER NOT NULL,
                view_id TEXT NOT NULL,
                refresh_version INTEGER NOT NULL,
                state TEXT NOT NULL,
                stale INTEGER NOT NULL DEFAULT 0,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (param_version, view_id)
            )""",
            """CREATE TABLE IF NOT EXISTS scheduler_jobs (
                job_id TEXT PRIMARY KEY,
                batch_id TEXT,
                view_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                param_version INTEGER NOT NULL,
                status TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS batches (
                batch_no INTEGER PRIMARY KEY,
                ledger_format_version INTEGER NOT NULL,
                param_version INTEGER NOT NULL,
                first_seq INTEGER NOT NULL,
                last_seq INTEGER NOT NULL,
                cursor INTEGER NOT NULL,
                summary_json TEXT NOT NULL,
                content_digest TEXT NOT NULL,
                snapshot_version INTEGER NOT NULL,
                committed_wall_ms INTEGER NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS snapshots (
                snapshot_version INTEGER PRIMARY KEY,
                ledger_format_version INTEGER NOT NULL,
                content_digest TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_wall_ms INTEGER NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS rebuild_traces (
                trace_id INTEGER PRIMARY KEY AUTOINCREMENT,
                view_id TEXT NOT NULL,
                param_version INTEGER NOT NULL,
                refresh_version INTEGER NOT NULL,
                reason TEXT NOT NULL,
                status TEXT NOT NULL,
                old_digest TEXT,
                new_digest TEXT,
                started_mono_ms INTEGER NOT NULL,
                finished_mono_ms INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                created_wall_ms INTEGER NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS definition_replacements (
                replacement_id INTEGER PRIMARY KEY AUTOINCREMENT,
                old_param_version INTEGER,
                new_param_version INTEGER NOT NULL,
                summary_json TEXT NOT NULL,
                content_digest TEXT NOT NULL,
                created_wall_ms INTEGER NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS authority_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                param_version INTEGER NOT NULL,
                cursor INTEGER NOT NULL,
                status TEXT NOT NULL,
                report_json TEXT NOT NULL,
                content_digest TEXT NOT NULL,
                created_wall_ms INTEGER NOT NULL
            )""",
            """CREATE INDEX IF NOT EXISTS idx_view_results_current
                ON view_results(param_version, stale, inconsistent)""",
            """CREATE INDEX IF NOT EXISTS idx_jobs_batch ON scheduler_jobs(batch_id)""",
            """CREATE INDEX IF NOT EXISTS idx_rebuild_view ON rebuild_traces(view_id, param_version)""",
        ]
        with self._lock:
            for statement in statements:
                self._connection.execute(statement)

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def _tx(self) -> None:
        self._connection.execute("BEGIN IMMEDIATE")

    def _commit(self) -> None:
        self._connection.execute("COMMIT")

    def _rollback(self) -> None:
        try:
            self._connection.execute("ROLLBACK")
        except sqlite3.Error:
            pass

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        with self._lock:
            row = self._connection.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return default if row is None else str(row["value"])

    def _set_meta(self, key: str, value: Any) -> None:
        self._connection.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(value)),
        )

    def is_initialized(self) -> bool:
        return self.get_meta("param_version") is not None

    @property
    def current_param_version(self) -> int | None:
        value = self.get_meta("param_version")
        return None if value is None else int(value)

    @property
    def current_fingerprint(self) -> str | None:
        return self.get_meta("definitions_fingerprint")

    def load_current_view_defs(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT view_id,payload_json,digest FROM view_defs WHERE is_current=1 ORDER BY view_id"
            ).fetchall()
        return {
            row["view_id"]: {"payload": json.loads(row["payload_json"]), "digest": row["digest"]}
            for row in rows
        }

    def current_param_summary(self) -> dict[str, Any] | None:
        version = self.current_param_version
        if version is None:
            return None
        with self._lock:
            row = self._connection.execute(
                "SELECT summary_json,fingerprint FROM param_versions WHERE version=?", (version,)
            ).fetchone()
        if row is None:
            return None
        return {"version": version, "fingerprint": row["fingerprint"], "summary": json.loads(row["summary_json"])}

    def _insert_definition_rows(self, definitions: Definitions, version: int, current: int) -> None:
        summary_json = canonical_json_text(definition_summary(definitions))
        self._connection.execute(
            "INSERT INTO param_versions(version,fingerprint,summary_json,created_wall_ms,is_current) VALUES(?,?,?,?,?)",
            (version, definitions.fingerprint, summary_json, wall_ms(), current),
        )
        for view_id in sorted(definitions.views):
            view = definitions.views[view_id]
            self._connection.execute(
                "INSERT INTO view_defs(param_version,view_id,payload_json,digest,is_current) VALUES(?,?,?,?,?)",
                (version, view_id, canonical_json_text(view.payload), view.digest, current),
            )

    def _upsert_results(self, param_version: int, results: dict[str, dict[str, Any]]) -> None:
        for view_id in sorted(results):
            result = results[view_id]
            self._connection.execute(
                """INSERT INTO view_results(
                    param_version,view_id,result_version,cursor,result_json,result_digest,
                    stale,inconsistent,updated_wall_ms
                ) VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(param_version,view_id) DO UPDATE SET
                    result_version=excluded.result_version,
                    cursor=excluded.cursor,
                    result_json=excluded.result_json,
                    result_digest=excluded.result_digest,
                    stale=excluded.stale,
                    inconsistent=excluded.inconsistent,
                    updated_wall_ms=excluded.updated_wall_ms""",
                (
                    param_version,
                    view_id,
                    int(result["result_version"]),
                    int(result["cursor"]),
                    result["result_json"],
                    result["result_digest"],
                    1 if result.get("stale") else 0,
                    1 if result.get("inconsistent") else 0,
                    int(result.get("updated_wall_ms", wall_ms())),
                ),
            )

    def _upsert_leases(self, param_version: int, leases: dict[str, dict[str, Any]]) -> None:
        for view_id in sorted(leases):
            lease = leases[view_id]
            self._connection.execute(
                """INSERT INTO leases(param_version,view_id,refresh_version,state,stale,payload_json)
                VALUES(?,?,?,?,?,?)
                ON CONFLICT(param_version,view_id) DO UPDATE SET
                    refresh_version=excluded.refresh_version,
                    state=excluded.state,
                    stale=excluded.stale,
                    payload_json=excluded.payload_json""",
                (
                    param_version,
                    view_id,
                    int(lease["refresh_version"]),
                    lease["state"],
                    1 if lease.get("stale") else 0,
                    canonical_json_text(lease),
                ),
            )

    def _insert_jobs(self, jobs: list[dict[str, Any]]) -> None:
        for job in jobs:
            self._connection.execute(
                """INSERT INTO scheduler_jobs(job_id,batch_id,view_id,kind,param_version,status,payload_json)
                VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(job_id) DO UPDATE SET status=excluded.status,payload_json=excluded.payload_json""",
                (
                    job["job_id"],
                    job.get("batch_id"),
                    job["view_id"],
                    job["kind"],
                    int(job["param_version"]),
                    job["status"],
                    canonical_json_text(job),
                ),
            )

    def _build_snapshot_payload(self, snapshot_version: int) -> dict[str, Any]:
        param_version = self.current_param_version
        if param_version is None:
            fail("E_SNAPSHOT", "ledger has no current parameter version", location={"path": "$.meta.param_version"})
        cursor = int(self.get_meta("cursor", "0") or 0)
        batch_counter = int(self.get_meta("batch_counter", "0") or 0)
        base_initial = json.loads(self.get_meta("base_initial_json", "null") or "null")
        base_current = json.loads(self.get_meta("base_current_json", "null") or "null")
        result_rows = self._connection.execute(
            """SELECT view_id,result_version,cursor,result_json,result_digest,stale,inconsistent,updated_wall_ms
            FROM view_results WHERE param_version=? ORDER BY view_id""",
            (param_version,),
        ).fetchall()
        lease_rows = self._connection.execute(
            "SELECT view_id,payload_json FROM leases WHERE param_version=? ORDER BY view_id",
            (param_version,),
        ).fetchall()
        trace_rows = self._connection.execute(
            "SELECT payload_json FROM rebuild_traces WHERE param_version=? ORDER BY trace_id",
            (param_version,),
        ).fetchall()
        return {
            "format_version": SNAPSHOT_FORMAT_VERSION,
            "snapshot_version": snapshot_version,
            "ledger_format_version": LEDGER_FORMAT_VERSION,
            "param_version": param_version,
            "cursor": cursor,
            "batch_counter": batch_counter,
            "definitions_fingerprint": self.get_meta("definitions_fingerprint"),
            "base_initial": base_initial,
            "base_current": base_current,
            "base_initial_digest": self.get_meta("base_initial_digest"),
            "base_digest": self.get_meta("base_digest"),
            "changes_digest": self.get_meta("changes_digest"),
            "results": {
                row["view_id"]: {
                    "result_version": row["result_version"],
                    "cursor": row["cursor"],
                    "result_json": row["result_json"],
                    "result_digest": row["result_digest"],
                    "stale": bool(row["stale"]),
                    "inconsistent": bool(row["inconsistent"]),
                    "updated_wall_ms": row["updated_wall_ms"],
                }
                for row in result_rows
            },
            "leases": {row["view_id"]: json.loads(row["payload_json"]) for row in lease_rows},
            "rebuild_traces": [json.loads(row["payload_json"]) for row in trace_rows],
        }

    def _write_snapshot(self) -> tuple[int, str]:
        current = int(self.get_meta("snapshot_version", "0") or 0)
        next_version = current + 1
        payload = self._build_snapshot_payload(next_version)
        payload_json = canonical_json_text(payload)
        digest = digest_obj(payload)
        self._connection.execute(
            "INSERT INTO snapshots(snapshot_version,ledger_format_version,content_digest,payload_json,created_wall_ms) VALUES(?,?,?,?,?)",
            (next_version, LEDGER_FORMAT_VERSION, digest, payload_json, wall_ms()),
        )
        self._set_meta("snapshot_version", next_version)
        self._set_meta("snapshot_digest", digest)
        self._set_meta("snapshot_format_version", SNAPSHOT_FORMAT_VERSION)
        return next_version, digest

    def initialize_session(
        self,
        definitions: Definitions,
        base_payload: dict[str, Any],
        base_digest: str,
        changes_digest: str,
        results: dict[str, dict[str, Any]],
        leases: dict[str, dict[str, Any]],
        jobs: list[dict[str, Any]],
        summary: dict[str, Any],
    ) -> CommitReceipt:
        with self._lock:
            if self.is_initialized():
                fail("E_STATE", "ledger session is already initialized")
            self._tx()
            try:
                self._set_meta("schema_version", SCHEMA_VERSION)
                self._set_meta("ledger_format_version", LEDGER_FORMAT_VERSION)
                self._set_meta("snapshot_format_version", SNAPSHOT_FORMAT_VERSION)
                self._set_meta("param_version", 1)
                self._set_meta("definitions_fingerprint", definitions.fingerprint)
                self._set_meta("cursor", 0)
                self._set_meta("batch_counter", 0)
                self._set_meta("snapshot_version", 0)
                base_json = canonical_json_text(base_payload)
                self._set_meta("base_initial_json", base_json)
                self._set_meta("base_current_json", base_json)
                self._set_meta("base_initial_digest", base_digest)
                self._set_meta("base_digest", base_digest)
                self._set_meta("changes_digest", changes_digest)
                self._insert_definition_rows(definitions, 1, 1)
                self._upsert_results(1, results)
                self._upsert_leases(1, leases)
                self._insert_jobs(jobs)
                snapshot_version, snapshot_digest = self._write_snapshot()
                ledger_entry = {
                    "ledger_format_version": LEDGER_FORMAT_VERSION,
                    "batch_no": 0,
                    "param_version": 1,
                    "first_seq": 0,
                    "last_seq": 0,
                    "cursor": 0,
                    "summary": summary,
                    "view_digests": {view_id: results[view_id]["result_digest"] for view_id in sorted(results)},
                    "leases": leases,
                    "jobs": jobs,
                    "snapshot_version": snapshot_version,
                }
                content_digest = digest_obj(ledger_entry)
                self._connection.execute(
                    """INSERT INTO batches(
                        batch_no,ledger_format_version,param_version,first_seq,last_seq,cursor,
                        summary_json,content_digest,snapshot_version,committed_wall_ms
                    ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (
                        0,
                        LEDGER_FORMAT_VERSION,
                        1,
                        0,
                        0,
                        0,
                        canonical_json_text(summary),
                        content_digest,
                        snapshot_version,
                        wall_ms(),
                    ),
                )
                self._commit()
            except BaseException:
                self._rollback()
                raise
        return CommitReceipt(0, 0, LEDGER_FORMAT_VERSION, content_digest, snapshot_version, snapshot_digest)

    def commit_batch(
        self,
        *,
        batch_no: int,
        first_seq: int,
        last_seq: int,
        cursor: int,
        param_version: int,
        base_payload: dict[str, Any],
        results: dict[str, dict[str, Any]],
        leases: dict[str, dict[str, Any]],
        jobs: list[dict[str, Any]],
        summary: dict[str, Any],
    ) -> CommitReceipt:
        with self._lock:
            self._tx()
            try:
                self._upsert_results(param_version, results)
                self._upsert_leases(param_version, leases)
                self._insert_jobs(jobs)
                base_json = canonical_json_text(base_payload)
                self._set_meta("cursor", cursor)
                self._set_meta("batch_counter", batch_no)
                self._set_meta("base_current_json", base_json)
                self._set_meta("base_digest", digest_obj(base_payload))
                snapshot_version, snapshot_digest = self._write_snapshot()
                ledger_entry = {
                    "ledger_format_version": LEDGER_FORMAT_VERSION,
                    "batch_no": batch_no,
                    "param_version": param_version,
                    "first_seq": first_seq,
                    "last_seq": last_seq,
                    "cursor": cursor,
                    "summary": summary,
                    "view_digests": {view_id: results[view_id]["result_digest"] for view_id in sorted(results)},
                    "leases": leases,
                    "jobs": jobs,
                    "snapshot_version": snapshot_version,
                }
                content_digest = digest_obj(ledger_entry)
                self._connection.execute(
                    """INSERT INTO batches(
                        batch_no,ledger_format_version,param_version,first_seq,last_seq,cursor,
                        summary_json,content_digest,snapshot_version,committed_wall_ms
                    ) VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (
                        batch_no,
                        LEDGER_FORMAT_VERSION,
                        param_version,
                        first_seq,
                        last_seq,
                        cursor,
                        canonical_json_text(summary),
                        content_digest,
                        snapshot_version,
                        wall_ms(),
                    ),
                )
                self._commit()
            except BaseException:
                self._rollback()
                raise
        return CommitReceipt(batch_no, cursor, LEDGER_FORMAT_VERSION, content_digest, snapshot_version, snapshot_digest)

    def replace_definitions(
        self,
        *,
        old_param_version: int,
        new_param_version: int,
        definitions: Definitions,
        cursor: int,
        base_payload: dict[str, Any],
        results: dict[str, dict[str, Any]],
        leases: dict[str, dict[str, Any]],
        jobs: list[dict[str, Any]],
        replacement_summary: dict[str, Any],
    ) -> CommitReceipt:
        with self._lock:
            self._tx()
            try:
                self._connection.execute("UPDATE param_versions SET is_current=0 WHERE is_current=1")
                self._connection.execute("UPDATE view_defs SET is_current=0 WHERE is_current=1")
                self._connection.execute("UPDATE view_results SET stale=1 WHERE param_version=?", (old_param_version,))
                self._connection.execute("UPDATE leases SET stale=1 WHERE param_version=?", (old_param_version,))
                self._insert_definition_rows(definitions, new_param_version, 1)
                self._set_meta("param_version", new_param_version)
                self._set_meta("definitions_fingerprint", definitions.fingerprint)
                self._set_meta("cursor", cursor)
                self._set_meta("base_current_json", canonical_json_text(base_payload))
                self._set_meta("base_digest", digest_obj(base_payload))
                self._upsert_results(new_param_version, results)
                self._upsert_leases(new_param_version, leases)
                self._insert_jobs(jobs)
                replacement_digest = digest_obj(replacement_summary)
                self._connection.execute(
                    """INSERT INTO definition_replacements(
                        old_param_version,new_param_version,summary_json,content_digest,created_wall_ms
                    ) VALUES(?,?,?,?,?)""",
                    (old_param_version, new_param_version, canonical_json_text(replacement_summary), replacement_digest, wall_ms()),
                )
                snapshot_version, snapshot_digest = self._write_snapshot()
                ledger_entry = {
                    "ledger_format_version": LEDGER_FORMAT_VERSION,
                    "event": "definition_replacement",
                    "old_param_version": old_param_version,
                    "new_param_version": new_param_version,
                    "summary": replacement_summary,
                    "snapshot_version": snapshot_version,
                }
                content_digest = digest_obj(ledger_entry)
                self._commit()
            except BaseException:
                self._rollback()
                raise
        return CommitReceipt(
            -1,
            cursor,
            LEDGER_FORMAT_VERSION,
            content_digest,
            snapshot_version,
            snapshot_digest,
        )

    def persist_inconsistency(
        self,
        *,
        param_version: int,
        view_id: str,
        result: dict[str, Any],
        lease: dict[str, Any],
    ) -> CommitReceipt:
        with self._lock:
            self._tx()
            try:
                self._upsert_results(param_version, {view_id: result})
                self._upsert_leases(param_version, {view_id: lease})
                snapshot_version, snapshot_digest = self._write_snapshot()
                entry = {
                    "ledger_format_version": LEDGER_FORMAT_VERSION,
                    "event": "lease_inconsistent",
                    "param_version": param_version,
                    "view_id": view_id,
                    "lease": lease,
                    "snapshot_version": snapshot_version,
                }
                content_digest = digest_obj(entry)
                self._commit()
            except BaseException:
                self._rollback()
                raise
        return CommitReceipt(-1, int(self.get_meta("cursor", "0") or 0), LEDGER_FORMAT_VERSION, content_digest, snapshot_version, snapshot_digest)

    def commit_rebuild(
        self,
        *,
        param_version: int,
        view_id: str,
        result: dict[str, Any],
        lease: dict[str, Any],
        job: dict[str, Any] | None,
        trace: dict[str, Any],
    ) -> CommitReceipt:
        with self._lock:
            self._tx()
            try:
                self._upsert_results(param_version, {view_id: result})
                self._upsert_leases(param_version, {view_id: lease})
                if job is not None:
                    self._insert_jobs([job])
                trace_json = canonical_json_text(trace)
                self._connection.execute(
                    """INSERT INTO rebuild_traces(
                        view_id,param_version,refresh_version,reason,status,old_digest,new_digest,
                        started_mono_ms,finished_mono_ms,payload_json,created_wall_ms
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        view_id,
                        param_version,
                        int(trace["refresh_version"]),
                        trace["reason"],
                        trace["status"],
                        trace.get("old_digest"),
                        trace.get("new_digest"),
                        int(trace["started_mono_ms"]),
                        int(trace["finished_mono_ms"]),
                        trace_json,
                        wall_ms(),
                    ),
                )
                snapshot_version, snapshot_digest = self._write_snapshot()
                entry = {
                    "ledger_format_version": LEDGER_FORMAT_VERSION,
                    "event": "view_rebuild",
                    "param_version": param_version,
                    "view_id": view_id,
                    "result_digest": result["result_digest"],
                    "lease": lease,
                    "trace": trace,
                    "job": job,
                    "snapshot_version": snapshot_version,
                }
                content_digest = digest_obj(entry)
                self._commit()
            except BaseException:
                self._rollback()
                raise
        return CommitReceipt(-1, int(self.get_meta("cursor", "0") or 0), LEDGER_FORMAT_VERSION, content_digest, snapshot_version, snapshot_digest)

    def load_state(self) -> PersistedState:
        with self._lock:
            param_version = self.current_param_version
            if param_version is None:
                fail("E_STATE", "ledger is not initialized")
            result_rows = self._connection.execute(
                "SELECT * FROM view_results WHERE param_version=? ORDER BY view_id", (param_version,)
            ).fetchall()
            lease_rows = self._connection.execute(
                "SELECT view_id,payload_json FROM leases WHERE param_version=? ORDER BY view_id", (param_version,)
            ).fetchall()
            base_payload = json.loads(self.get_meta("base_current_json", "null") or "null")
            return PersistedState(
                param_version=param_version,
                cursor=int(self.get_meta("cursor", "0") or 0),
                batch_counter=int(self.get_meta("batch_counter", "0") or 0),
                base_rows_payload=base_payload,
                base_digest=self.get_meta("base_digest", "") or "",
                changes_digest=self.get_meta("changes_digest", "") or "",
                results={
                    row["view_id"]: {
                        "result_version": row["result_version"],
                        "cursor": row["cursor"],
                        "result_json": row["result_json"],
                        "result_digest": row["result_digest"],
                        "stale": bool(row["stale"]),
                        "inconsistent": bool(row["inconsistent"]),
                        "updated_wall_ms": row["updated_wall_ms"],
                    }
                    for row in result_rows
                },
                leases={row["view_id"]: json.loads(row["payload_json"]) for row in lease_rows},
                snapshot_version=int(self.get_meta("snapshot_version", "0") or 0),
                snapshot_digest=self.get_meta("snapshot_digest", "") or "",
            )

    def snapshot_notice(self) -> dict[str, Any]:
        with self._lock:
            return {
                "snapshot_version": int(self.get_meta("snapshot_version", "0") or 0),
                "snapshot_digest": self.get_meta("snapshot_digest", ""),
                "snapshot_format_version": int(self.get_meta("snapshot_format_version", "0") or 0),
                "ledger_format_version": int(self.get_meta("ledger_format_version", "0") or 0),
            }

    def validate_and_restore_snapshot(self) -> SnapshotAudit:
        with self._lock:
            snapshot_version = int(self.get_meta("snapshot_version", "0") or 0)
            ledger_format = int(self.get_meta("ledger_format_version", "0") or 0)
            snapshot_format = int(self.get_meta("snapshot_format_version", "0") or 0)
            if snapshot_format != SNAPSHOT_FORMAT_VERSION or ledger_format != LEDGER_FORMAT_VERSION:
                error = BusinessError(
                    "E_SNAPSHOT",
                    "snapshot format or ledger format version is not supported",
                    {"path": "$.snapshot.format_version"},
                    {"snapshot_format_version": snapshot_format, "ledger_format_version": ledger_format},
                )
                return SnapshotAudit(snapshot_version, ledger_format, "", "", False, False, error)
            row = self._connection.execute(
                "SELECT * FROM snapshots WHERE snapshot_version=?", (snapshot_version,)
            ).fetchone()
            if row is None:
                error = BusinessError("E_SNAPSHOT", "latest snapshot row is missing", {"path": "$.snapshots", "snapshot_version": snapshot_version})
                return SnapshotAudit(snapshot_version, ledger_format, "", "", False, False, error)
            try:
                payload = json.loads(row["payload_json"])
            except json.JSONDecodeError as exc:
                error = BusinessError(
                    "E_SNAPSHOT",
                    "snapshot payload is not valid JSON",
                    {"path": "$.snapshot.payload_json", "snapshot_version": snapshot_version},
                    {"error": str(exc)},
                )
                return SnapshotAudit(snapshot_version, ledger_format, row["content_digest"], "", False, False, error)
            computed_snapshot_digest = digest_obj(payload)
            stored_digest = str(row["content_digest"])
            meta_digest = self.get_meta("snapshot_digest", "") or ""
            current_payload = self._build_snapshot_payload(snapshot_version)
            current_digest = digest_obj(current_payload)
            valid_snapshot = (
                computed_snapshot_digest == stored_digest == meta_digest
                and int(payload.get("snapshot_version", -1)) == snapshot_version
                and int(payload.get("format_version", -1)) == SNAPSHOT_FORMAT_VERSION
            )
            if not valid_snapshot:
                error = BusinessError(
                    "E_SNAPSHOT",
                    "snapshot digest or format is invalid and cannot be restored",
                    {"path": "$.snapshots", "snapshot_version": snapshot_version, "field": "content_digest"},
                    {"computed": computed_snapshot_digest, "stored": stored_digest, "meta": meta_digest},
                )
                return SnapshotAudit(snapshot_version, ledger_format, stored_digest, current_digest, False, False, error)
            if current_digest == stored_digest:
                return SnapshotAudit(snapshot_version, ledger_format, stored_digest, current_digest, True, False, None)

            error = BusinessError(
                "E_SNAPSHOT",
                "current state digest differs from the latest snapshot; replacing results, leases and cursor atomically",
                {"path": "$.snapshots", "snapshot_version": snapshot_version, "field": "content_digest"},
                {"expected": stored_digest, "actual": current_digest},
            )
            self._restore_snapshot_payload(payload, stored_digest)
            return SnapshotAudit(snapshot_version, ledger_format, stored_digest, current_digest, False, True, error)

    def _restore_snapshot_payload(self, payload: dict[str, Any], stored_digest: str) -> None:
        param_version = int(payload["param_version"])
        self._tx()
        try:
            self._connection.execute("DELETE FROM view_results WHERE param_version=?", (param_version,))
            self._connection.execute("DELETE FROM leases WHERE param_version=?", (param_version,))
            self._connection.execute("DELETE FROM rebuild_traces WHERE param_version=?", (param_version,))
            self._set_meta("param_version", param_version)
            if payload.get("definitions_fingerprint"):
                self._set_meta("definitions_fingerprint", payload["definitions_fingerprint"])
            self._set_meta("cursor", int(payload.get("cursor", 0)))
            self._set_meta("batch_counter", int(payload.get("batch_counter", 0)))
            self._set_meta("snapshot_version", int(payload["snapshot_version"]))
            self._set_meta("snapshot_digest", stored_digest)
            self._set_meta("snapshot_format_version", SNAPSHOT_FORMAT_VERSION)
            self._set_meta("base_initial_json", canonical_json_text(payload.get("base_initial")))
            self._set_meta("base_current_json", canonical_json_text(payload.get("base_current")))
            self._set_meta("base_initial_digest", payload.get("base_initial_digest", payload.get("base_digest", "")))
            self._set_meta("base_digest", payload.get("base_digest", ""))
            self._set_meta("changes_digest", payload.get("changes_digest", ""))
            self._upsert_results(param_version, payload.get("results", {}))
            self._upsert_leases(param_version, payload.get("leases", {}))
            for trace in payload.get("rebuild_traces", []):
                self._connection.execute(
                    """INSERT INTO rebuild_traces(
                        view_id,param_version,refresh_version,reason,status,old_digest,new_digest,
                        started_mono_ms,finished_mono_ms,payload_json,created_wall_ms
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        trace["view_id"],
                        param_version,
                        int(trace["refresh_version"]),
                        trace["reason"],
                        trace["status"],
                        trace.get("old_digest"),
                        trace.get("new_digest"),
                        int(trace["started_mono_ms"]),
                        int(trace["finished_mono_ms"]),
                        canonical_json_text(trace),
                        wall_ms(),
                    ),
                )
            self._commit()
        except BaseException:
            self._rollback()
            raise

    def list_batches(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM batches ORDER BY batch_no DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            {
                "batch_no": row["batch_no"],
                "ledger_format_version": row["ledger_format_version"],
                "param_version": row["param_version"],
                "first_seq": row["first_seq"],
                "last_seq": row["last_seq"],
                "cursor": row["cursor"],
                "summary": json.loads(row["summary_json"]),
                "content_digest": row["content_digest"],
                "snapshot_version": row["snapshot_version"],
                "committed_wall_ms": row["committed_wall_ms"],
            }
            for row in rows
        ]

    def list_scheduler_jobs(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload_json FROM scheduler_jobs ORDER BY rowid DESC LIMIT ?", (limit,)
            ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def list_rebuild_traces(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload_json FROM rebuild_traces ORDER BY trace_id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def list_replacements(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM definition_replacements ORDER BY replacement_id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            {
                "old_param_version": row["old_param_version"],
                "new_param_version": row["new_param_version"],
                "summary": json.loads(row["summary_json"]),
                "content_digest": row["content_digest"],
                "created_wall_ms": row["created_wall_ms"],
            }
            for row in rows
        ]

    def record_authority_run(self, *, param_version: int, cursor: int, status: str, report: dict[str, Any]) -> str:
        digest = digest_obj(report)
        with self._lock:
            self._tx()
            try:
                self._connection.execute(
                    "INSERT INTO authority_runs(param_version,cursor,status,report_json,content_digest,created_wall_ms) VALUES(?,?,?,?,?,?)",
                    (param_version, cursor, status, canonical_json_text(report), digest, wall_ms()),
                )
                self._commit()
            except BaseException:
                self._rollback()
                raise
        return digest

    def latest_snapshot_payload(self) -> dict[str, Any]:
        version = int(self.get_meta("snapshot_version", "0") or 0)
        with self._lock:
            row = self._connection.execute(
                "SELECT payload_json FROM snapshots WHERE snapshot_version=?", (version,)
            ).fetchone()
        if row is None:
            fail("E_SNAPSHOT", "latest snapshot payload is unavailable", location={"snapshot_version": version})
        return json.loads(row["payload_json"])

    def tamper_result_digest(self, view_id: str | None = None) -> dict[str, Any]:
        param_version = self.current_param_version
        if param_version is None:
            fail("E_STATE", "ledger is not initialized")
        with self._lock:
            if view_id is None:
                row = self._connection.execute(
                    "SELECT view_id FROM view_results WHERE param_version=? ORDER BY view_id LIMIT 1",
                    (param_version,),
                ).fetchone()
                if row is None:
                    fail("E_SNAPSHOT", "no incremental result is available to tamper")
                view_id = row["view_id"]
            row = self._connection.execute(
                "SELECT result_digest FROM view_results WHERE param_version=? AND view_id=?",
                (param_version, view_id),
            ).fetchone()
            if row is None:
                fail("E_SNAPSHOT", "requested incremental result does not exist", location={"view_id": view_id})
            old_digest = row["result_digest"]
            tampered = "0" * 64
            self._connection.execute(
                "UPDATE view_results SET result_digest=? WHERE param_version=? AND view_id=?",
                (tampered, param_version, view_id),
            )
        return {"view_id": view_id, "old_digest": old_digest, "tampered_digest": tampered}

    def verify_base_payload(self, payload: dict[str, Any]) -> None:
        try:
            table_name, row_id_column, columns, rows = rows_from_payload(payload)
        except (KeyError, TypeError, ValueError) as exc:
            fail("E_DATA", "base row snapshot is structurally corrupt", location={"path": "$.base_current"}, details={"error": str(exc)})
        column_map = {column.name: column for column in columns}
        if row_id_column not in column_map:
            fail("E_DATA", "base row snapshot lost its row id column", location={"path": "$.base_current.row_id_column"})
        for row_id in sorted(rows, key=lambda item: (type(item).__name__, canonical_json_text(item))):
            row = rows[row_id]
            if row_id_column not in row or row[row_id_column] != row_id:
                fail("E_DATA", "base row identifier checksum mismatch", location={"row_id": row_id, "field": row_id_column})
            for column in columns:
                value = row.get(column.name)
                if value is None:
                    if not column.nullable:
                        fail("E_DATA", "corrupt base row contains null in a non-nullable column", location={"row_id": row_id, "field": column.name, "column": column.name})
                    continue
                if column.type == "integer" and (isinstance(value, bool) or not isinstance(value, int)):
                    fail("E_DATA", "corrupt base row has a non-integer value", location={"row_id": row_id, "field": column.name, "column": column.name})
                if column.type == "float" and (isinstance(value, bool) or not isinstance(value, (int, float))):
                    fail("E_DATA", "corrupt base row has a non-numeric value", location={"row_id": row_id, "field": column.name, "column": column.name})
                if column.type == "text" and not isinstance(value, str):
                    fail("E_DATA", "corrupt base row has a non-text value", location={"row_id": row_id, "field": column.name, "column": column.name})
                if column.type == "boolean" and not isinstance(value, bool):
                    fail("E_DATA", "corrupt base row has a non-boolean value", location={"row_id": row_id, "field": column.name, "column": column.name})
