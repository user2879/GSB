from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .errors import BusinessError
from .models import (
    CommitRecord,
    EditScript,
    SchedulerParams,
    TraceEvent,
    canonical_json,
)


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rows (
    row_key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    data_version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS commits (
    run_id TEXT NOT NULL,
    commit_seq INTEGER NOT NULL,
    session_id TEXT NOT NULL,
    data_version INTEGER NOT NULL,
    virtual_ms INTEGER NOT NULL,
    writes_json TEXT NOT NULL,
    PRIMARY KEY (run_id, commit_seq)
);
CREATE TABLE IF NOT EXISTS ledger (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    trace_event_seq INTEGER NOT NULL,
    script_id TEXT NOT NULL,
    source_version INTEGER NOT NULL,
    normalized_version INTEGER NOT NULL,
    session_id TEXT,
    op_no INTEGER,
    event_type TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    virtual_ms INTEGER NOT NULL,
    monotonic_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ledger_run_idx ON ledger(run_id, trace_event_seq);
"""

SNAPSHOT_SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    script_id TEXT NOT NULL,
    source_version INTEGER NOT NULL,
    normalized_version INTEGER NOT NULL,
    parameter_version INTEGER NOT NULL,
    script_digest TEXT NOT NULL,
    params_digest TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
"""


@dataclass
class LedgerRecord:
    event_id: int
    run_id: str
    trace_event_seq: int
    script_id: str
    source_version: int
    normalized_version: int
    session_id: str | None
    op_no: int | None
    event_type: str
    payload: dict[str, Any]
    virtual_ms: int
    monotonic_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "run_id": self.run_id,
            "trace_event_seq": self.trace_event_seq,
            "script_id": self.script_id,
            "source_version": self.source_version,
            "normalized_version": self.normalized_version,
            "session_id": self.session_id,
            "op_no": self.op_no,
            "event_type": self.event_type,
            "payload": self.payload,
            "virtual_ms": self.virtual_ms,
            "monotonic_ms": self.monotonic_ms,
        }


class Storage:
    """SQLite persistence for committed rows, commits and the event ledger."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Storage":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _meta_get(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def _meta_set(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def global_data_version(self) -> int:
        return int(self._meta_get("data_version", "0") or 0)

    def reset_for_run(
        self,
        script: EditScript,
        params: SchedulerParams,
        run_id: str,
    ) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM rows")
            self.conn.execute("DELETE FROM commits")
            self.conn.execute("DELETE FROM ledger WHERE run_id=?", (run_id,))
            for row_key in sorted(script.initial_rows):
                self.conn.execute(
                    "INSERT INTO rows(row_key,value_json,data_version) VALUES(?,?,0)",
                    (row_key, canonical_json(script.initial_rows[row_key])),
                )
            self._meta_set("data_version", "0")
            self._set_context(script, params, run_id, outcomes=[])

    def _set_context(
        self,
        script: EditScript,
        params: SchedulerParams,
        run_id: str,
        *,
        outcomes: Iterable[dict[str, Any]],
    ) -> None:
        self._meta_set("current_script_id", script.script_id)
        self._meta_set("current_script_source_version", str(script.source_version))
        self._meta_set("current_script_normalized_version", str(script.normalized_version))
        self._meta_set("current_script_digest", script.semantic_digest)
        self._meta_set("current_script_json", canonical_json(script.normalized_for_snapshot()))
        self._meta_set("current_params_version", str(params.parameter_version))
        self._meta_set("current_params_json", canonical_json(params.to_dict()))
        self._meta_set("current_params_digest", params.digest)
        self._meta_set("current_run_id", run_id)
        self._meta_set("current_sessions_json", canonical_json([s.session_id for s in script.sessions]))
        self._meta_set("current_outcomes_json", canonical_json(list(outcomes)))

    def update_context_outcomes(self, script: EditScript, params: SchedulerParams, run_id: str, outcomes: list[dict[str, Any]]) -> None:
        with self.conn:
            self._set_context(script, params, run_id, outcomes=outcomes)

    def get_context(self) -> dict[str, Any]:
        def load_json(key: str, default: Any) -> Any:
            raw = self._meta_get(key)
            if raw is None:
                return default
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return default

        return {
            "script_id": self._meta_get("current_script_id"),
            "source_version": int(self._meta_get("current_script_source_version", "0") or 0),
            "normalized_version": int(self._meta_get("current_script_normalized_version", "0") or 0),
            "script_digest": self._meta_get("current_script_digest"),
            "script": load_json("current_script_json", None),
            "parameter_version": int(self._meta_get("current_params_version", "0") or 0),
            "params": load_json("current_params_json", {}),
            "params_digest": self._meta_get("current_params_digest"),
            "run_id": self._meta_get("current_run_id"),
            "sessions": load_json("current_sessions_json", []),
            "outcomes": load_json("current_outcomes_json", []),
            "data_version": self.global_data_version(),
        }

    def read_row(self, row_key: str) -> tuple[Any, int]:
        row = self.conn.execute(
            "SELECT value_json,data_version FROM rows WHERE row_key=?",
            (row_key,),
        ).fetchone()
        if row is None:
            return None, 0
        return json.loads(row["value_json"]), int(row["data_version"])

    def all_rows(self) -> dict[str, dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT row_key,value_json,data_version FROM rows ORDER BY row_key"
        ).fetchall()
        return {
            row["row_key"]: {
                "value": json.loads(row["value_json"]),
                "version": int(row["data_version"]),
            }
            for row in rows
        }

    def apply_commit(
        self,
        *,
        run_id: str,
        session_id: str,
        virtual_ms: int,
        writes: list[dict[str, Any]],
    ) -> tuple[int | None, CommitRecord | None]:
        if not writes:
            return None, None
        with self.conn:
            current_version = self.global_data_version()
            data_version = current_version + 1
            commit_seq = int(
                self.conn.execute(
                    "SELECT COALESCE(MAX(commit_seq),0) FROM commits WHERE run_id=?",
                    (run_id,),
                ).fetchone()[0]
            ) + 1
            for write in writes:
                self.conn.execute(
                    """
                    INSERT INTO rows(row_key,value_json,data_version) VALUES(?,?,?)
                    ON CONFLICT(row_key) DO UPDATE SET
                        value_json=excluded.value_json,
                        data_version=excluded.data_version
                    """,
                    (write["row_key"], canonical_json(write["value"]), data_version),
                )
            self._meta_set("data_version", str(data_version))
            self.conn.execute(
                """
                INSERT INTO commits(run_id,commit_seq,session_id,data_version,virtual_ms,writes_json)
                VALUES(?,?,?,?,?,?)
                """,
                (
                    run_id,
                    commit_seq,
                    session_id,
                    data_version,
                    virtual_ms,
                    canonical_json(writes),
                ),
            )
        commit = CommitRecord(
            commit_seq=commit_seq,
            session_id=session_id,
            data_version=data_version,
            virtual_ms=virtual_ms,
            writes=writes,
        )
        return data_version, commit

    def commits_for_run(self, run_id: str) -> list[CommitRecord]:
        rows = self.conn.execute(
            "SELECT commit_seq,session_id,data_version,virtual_ms,writes_json FROM commits WHERE run_id=? ORDER BY commit_seq",
            (run_id,),
        ).fetchall()
        return [
            CommitRecord(
                commit_seq=int(row["commit_seq"]),
                session_id=row["session_id"],
                data_version=int(row["data_version"]),
                virtual_ms=int(row["virtual_ms"]),
                writes=json.loads(row["writes_json"]),
            )
            for row in rows
        ]

    def append_trace(
        self,
        *,
        run_id: str,
        script: EditScript,
        trace: Iterable[TraceEvent],
    ) -> None:
        with self.conn:
            for event in trace:
                self.conn.execute(
                    """
                    INSERT INTO ledger(
                        run_id,trace_event_seq,script_id,source_version,normalized_version,
                        session_id,op_no,event_type,payload_json,virtual_ms,monotonic_ms
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        run_id,
                        event.event_seq,
                        script.script_id,
                        script.source_version,
                        script.normalized_version,
                        event.session_id,
                        event.op_no,
                        event.event_type,
                        canonical_json(event.payload),
                        event.virtual_ms,
                        event.monotonic_ms,
                    ),
                )

    def read_ledger(self, *, run_id: str | None = None) -> list[LedgerRecord]:
        if run_id is None:
            rows = self.conn.execute("SELECT * FROM ledger ORDER BY event_id").fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM ledger WHERE run_id=? ORDER BY event_id",
                (run_id,),
            ).fetchall()
        records: list[LedgerRecord] = []
        for row in rows:
            source_version = int(row["source_version"])
            if source_version not in (1, 2, 3):
                raise BusinessError(
                    "E_SCRIPT",
                    f"ledger record uses unsupported source_version {source_version}",
                    path=f"$.ledger[{row['event_id']}].source_version",
                    session_id=row["session_id"],
                    details={"event_id": row["event_id"]},
                )
            try:
                payload = json.loads(row["payload_json"])
            except json.JSONDecodeError as exc:
                raise BusinessError(
                    "E_SCRIPT",
                    "ledger payload is not valid JSON",
                    path=f"$.ledger[{row['event_id']}].payload_json",
                    session_id=row["session_id"],
                    details={"event_id": row["event_id"], "reason": exc.msg},
                ) from exc
            event_type = row["event_type"]
            if not event_type:
                event_type = str(payload.get("kind", "legacy.event"))
            records.append(
                LedgerRecord(
                    event_id=int(row["event_id"]),
                    run_id=row["run_id"],
                    trace_event_seq=int(row["trace_event_seq"]),
                    script_id=row["script_id"],
                    source_version=source_version,
                    normalized_version=int(row["normalized_version"]),
                    session_id=row["session_id"],
                    op_no=None if row["op_no"] is None else int(row["op_no"]),
                    event_type=event_type,
                    payload=payload,
                    virtual_ms=int(row["virtual_ms"]),
                    monotonic_ms=int(row["monotonic_ms"]),
                )
            )
        return records

    def replace_state_from_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Replace all result/trace state atomically; never merge row-by-row."""
        run_id = snapshot["run_id"]
        with self.conn:
            self.conn.execute("DELETE FROM rows")
            self.conn.execute("DELETE FROM commits")
            self.conn.execute("DELETE FROM ledger WHERE run_id=?", (run_id,))
            for row_key, row in sorted(snapshot["rows"].items()):
                self.conn.execute(
                    "INSERT INTO rows(row_key,value_json,data_version) VALUES(?,?,?)",
                    (row_key, canonical_json(row["value"]), int(row["version"])),
                )
            for commit in snapshot["commits"]:
                self.conn.execute(
                    """
                    INSERT INTO commits(run_id,commit_seq,session_id,data_version,virtual_ms,writes_json)
                    VALUES(?,?,?,?,?,?)
                    """,
                    (
                        run_id,
                        int(commit["commit_seq"]),
                        commit["session_id"],
                        int(commit["data_version"]),
                        int(commit["virtual_ms"]),
                        canonical_json(commit["writes"]),
                    ),
                )
            self._meta_set("data_version", str(int(snapshot["global_data_version"])))
            script = snapshot["script"]
            params = SchedulerParams(
                global_concurrency_limit=int(snapshot["params"]["global_concurrency_limit"]),
                clock_quantum_ms=int(snapshot["params"]["clock_quantum_ms"]),
                parameter_version=int(snapshot["params"]["parameter_version"]),
            )
            # Build a transient EditScript only to reuse the context writer.
            from .scripts import normalize_document

            normalized = normalize_document(script)
            edit_script = EditScript(
                script_id=normalized.script_id,
                source_version=int(snapshot["source_version"]),
                normalized_version=int(snapshot["normalized_version"]),
                initial_rows=normalized.initial_rows,
                sessions=normalized.sessions,
                description=normalized.description,
                upgrade_changes=tuple(snapshot.get("upgrade_changes", ())),
            )
            self._set_context(
                edit_script,
                params,
                run_id,
                outcomes=snapshot["outcomes"],
            )
            for event in snapshot["trace"]:
                self.conn.execute(
                    """
                    INSERT INTO ledger(
                        run_id,trace_event_seq,script_id,source_version,normalized_version,
                        session_id,op_no,event_type,payload_json,virtual_ms,monotonic_ms
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        run_id,
                        int(event["event_seq"]),
                        edit_script.script_id,
                        edit_script.source_version,
                        edit_script.normalized_version,
                        event.get("session_id"),
                        event.get("op_no"),
                        event["event_type"],
                        canonical_json(event.get("payload", {})),
                        int(event["virtual_ms"]),
                        int(event.get("monotonic_ms", 0)),
                    ),
                )

    def integrity_check(self) -> str:
        return str(self.conn.execute("PRAGMA integrity_check").fetchone()[0])


class SnapshotStore:
    """SQLite-backed snapshot catalog; each payload is atomically replaced by id."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SNAPSHOT_SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "SnapshotStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def save(self, snapshot: dict[str, Any]) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO snapshots(
                    snapshot_id,created_at,script_id,source_version,normalized_version,
                    parameter_version,script_digest,params_digest,payload_json
                ) VALUES(?,?,?,?,?,?,?,?,?)
                ON CONFLICT(snapshot_id) DO UPDATE SET
                    created_at=excluded.created_at,
                    script_id=excluded.script_id,
                    source_version=excluded.source_version,
                    normalized_version=excluded.normalized_version,
                    parameter_version=excluded.parameter_version,
                    script_digest=excluded.script_digest,
                    params_digest=excluded.params_digest,
                    payload_json=excluded.payload_json
                """,
                (
                    snapshot["snapshot_id"],
                    snapshot["created_at"],
                    snapshot["script_id"],
                    int(snapshot["source_version"]),
                    int(snapshot["normalized_version"]),
                    int(snapshot["params"]["parameter_version"]),
                    snapshot["script_digest"],
                    snapshot["params_digest"],
                    canonical_json(snapshot),
                ),
            )

    def load(self, snapshot_id: str) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT payload_json FROM snapshots WHERE snapshot_id=?",
            (snapshot_id,),
        ).fetchone()
        if row is None:
            raise BusinessError(
                "E_SNAPSHOT",
                f"snapshot not found: {snapshot_id}",
                path=f"$.snapshots.{snapshot_id}",
                details={"snapshot_id": snapshot_id},
            )
        try:
            snapshot = json.loads(row["payload_json"])
        except json.JSONDecodeError as exc:
            raise BusinessError(
                "E_SNAPSHOT",
                "snapshot payload is not valid JSON",
                path=f"$.snapshots.{snapshot_id}",
                details={"snapshot_id": snapshot_id, "reason": exc.msg},
            ) from exc
        required = {
            "snapshot_id",
            "created_at",
            "script_id",
            "source_version",
            "normalized_version",
            "script",
            "script_digest",
            "params",
            "parameter_version",
            "params_digest",
            "lock_state",
            "trace",
            "outcomes",
            "rows",
            "commits",
            "global_data_version",
            "run_id",
        }
        missing = sorted(required - set(snapshot))
        if missing:
            raise BusinessError(
                "E_SNAPSHOT",
                f"snapshot payload is missing fields: {', '.join(missing)}",
                path=f"$.snapshots.{snapshot_id}",
                details={"missing": missing},
            )
        return snapshot

    def list_ids(self) -> list[str]:
        rows = self.conn.execute("SELECT snapshot_id FROM snapshots ORDER BY snapshot_id").fetchall()
        return [row["snapshot_id"] for row in rows]
