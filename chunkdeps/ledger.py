"""SQLite ledger and filesystem layout for the workbench."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .graph import SCHEMA_GRAPH, DependencyGraph
from .util import resolve_relative


SCHEMA_CORE = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=FULL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS schema_meta(
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions(
    session_id TEXT PRIMARY KEY,
    object_key TEXT NOT NULL,
    chunk_size INTEGER NOT NULL,
    total_size INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL,
    digest_algorithm TEXT NOT NULL,
    tags_json TEXT NOT NULL,
    state TEXT NOT NULL,
    bitmap TEXT NOT NULL,
    registered_count INTEGER NOT NULL DEFAULT 0,
    created_mono_ms INTEGER NOT NULL,
    updated_mono_ms INTEGER NOT NULL,
    completed INTEGER NOT NULL DEFAULT 0,
    commit_unknown INTEGER NOT NULL DEFAULT 0,
    recovery_status TEXT NOT NULL DEFAULT 'none',
    expected_base_version INTEGER,
    version INTEGER,
    content_digest TEXT,
    object_path TEXT,
    abort_reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_object_key ON sessions(object_key);
CREATE INDEX IF NOT EXISTS idx_sessions_state ON sessions(state);

CREATE TABLE IF NOT EXISTS session_events(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    from_state TEXT NOT NULL,
    to_state TEXT NOT NULL,
    mono_ms INTEGER NOT NULL,
    bitmap TEXT NOT NULL,
    reason TEXT NOT NULL,
    commit_unknown INTEGER NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);
CREATE INDEX IF NOT EXISTS idx_session_events_session ON session_events(session_id, id);

CREATE TABLE IF NOT EXISTS chunks(
    session_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    digest TEXT NOT NULL,
    size INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    registered_mono_ms INTEGER NOT NULL,
    PRIMARY KEY(session_id, chunk_index),
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS object_versions(
    object_key TEXT NOT NULL,
    version INTEGER NOT NULL,
    session_id TEXT NOT NULL,
    metadata_digest TEXT NOT NULL,
    manifest_digest TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    chunk_size INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL,
    chunk_set_digest TEXT NOT NULL,
    digest_algorithm TEXT NOT NULL,
    tags_json TEXT NOT NULL,
    status TEXT NOT NULL,
    visible INTEGER NOT NULL,
    object_path TEXT NOT NULL,
    created_mono_ms INTEGER NOT NULL,
    replaced_mono_ms INTEGER,
    PRIMARY KEY(object_key, version)
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_object_versions_one_active
    ON object_versions(object_key) WHERE status='active';
CREATE INDEX IF NOT EXISTS idx_object_versions_key_version
    ON object_versions(object_key, version);

CREATE TABLE IF NOT EXISTS audit_runs(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL UNIQUE,
    mono_ms INTEGER NOT NULL,
    passed INTEGER NOT NULL,
    summary_json TEXT NOT NULL,
    conclusion_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scheduler_events(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    queue_ms INTEGER NOT NULL,
    start_mono_ms INTEGER NOT NULL,
    end_mono_ms INTEGER NOT NULL,
    global_limit INTEGER NOT NULL,
    session_limit INTEGER NOT NULL,
    injection TEXT,
    created_mono_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_scheduler_session ON scheduler_events(session_id, id);
"""


class Ledger:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.data_dir = self.root / "data"
        self.chunks_dir = self.data_dir / "chunks"
        self.objects_dir = self.data_dir / "objects"
        self.temp_dir = self.data_dir / "tmp"
        self.logs_dir = self.root / "logs"
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.objects_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "ledger.sqlite3"
        self.snapshot_path = self.data_dir / "snapshot.json"
        self.conn = sqlite3.connect(
            self.db_path,
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA_CORE)
        self.conn.executescript(SCHEMA_GRAPH)
        self.conn.execute(
            "INSERT OR REPLACE INTO schema_meta(key, value) VALUES('schema_version','1')"
        )
        self.graph = DependencyGraph(self.conn)

    def close(self) -> None:
        self.conn.close()

    def relative(self, path: str | Path) -> str:
        return Path(path).resolve().relative_to(self.root).as_posix()

    def resolve(self, relative: str) -> Path:
        return resolve_relative(self.root, relative)

    def session_row(self, session_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()

    def object_version_row(self, object_key: str, version: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM object_versions WHERE object_key=? AND version=?",
            (object_key, version),
        ).fetchone()

    def active_version_row(self, object_key: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM object_versions WHERE object_key=? AND status='active'",
            (object_key,),
        ).fetchone()

    def chunk_rows(self, session_id: str) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT * FROM chunks WHERE session_id=? ORDER BY chunk_index",
                (session_id,),
            )
        )

    def chunk_row(self, session_id: str, chunk_index: int) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM chunks WHERE session_id=? AND chunk_index=?",
            (session_id, chunk_index),
        ).fetchone()

    def latest_audit(self) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM audit_runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return {
            "run_id": row["run_id"],
            "mono_ms": row["mono_ms"],
            "passed": bool(row["passed"]),
            "summary": json.loads(row["summary_json"]),
            "conclusion": json.loads(row["conclusion_json"]),
        }

    def next_object_version(self, object_key: str) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(version), 0) AS version FROM object_versions WHERE object_key=?",
            (object_key,),
        ).fetchone()
        assert row is not None
        return int(row["version"]) + 1

    def record_schedule(
        self,
        *,
        session_id: str,
        chunk_index: int,
        queue_ms: int,
        start_mono_ms: int,
        end_mono_ms: int,
        global_limit: int,
        session_limit: int,
        injection: str | None,
    ) -> dict[str, Any]:
        cursor = self.conn.execute(
            """
            INSERT INTO scheduler_events(
                session_id, chunk_index, queue_ms, start_mono_ms,
                end_mono_ms, global_limit, session_limit, injection, created_mono_ms
            ) VALUES(?,?,?,?,?,?,?,?,?)
            """,
            (
                session_id,
                chunk_index,
                queue_ms,
                start_mono_ms,
                end_mono_ms,
                global_limit,
                session_limit,
                injection,
                start_mono_ms,
            ),
        )
        return {
            "schedule_id": cursor.lastrowid,
            "session_id": session_id,
            "chunk_index": chunk_index,
            "queue_ms": queue_ms,
            "start_mono_ms": start_mono_ms,
            "end_mono_ms": end_mono_ms,
            "duration_ms": end_mono_ms - start_mono_ms,
            "global_limit": global_limit,
            "session_limit": session_limit,
            "injection": injection,
        }

    def schedule_events(self, session_id: str | None = None) -> list[dict[str, Any]]:
        if session_id:
            rows = self.conn.execute(
                "SELECT * FROM scheduler_events WHERE session_id=? ORDER BY id", (session_id,)
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM scheduler_events ORDER BY id").fetchall()
        return [
            {
                "schedule_id": row["id"],
                "session_id": row["session_id"],
                "chunk_index": row["chunk_index"],
                "queue_ms": row["queue_ms"],
                "start_mono_ms": row["start_mono_ms"],
                "end_mono_ms": row["end_mono_ms"],
                "duration_ms": row["end_mono_ms"] - row["start_mono_ms"],
                "global_limit": row["global_limit"],
                "session_limit": row["session_limit"],
                "injection": row["injection"],
            }
            for row in rows
        ]

    def all_tables_snapshot(self) -> dict[str, Any]:
        def rows(table: str, order: str) -> list[dict[str, Any]]:
            return [dict(row) for row in self.conn.execute(f"SELECT * FROM {table} ORDER BY {order}")]

        graph = self.graph.snapshot()
        return {
            "sessions": rows("sessions", "session_id"),
            "session_events": rows("session_events", "id"),
            "chunks": rows("chunks", "session_id, chunk_index"),
            "object_versions": rows("object_versions", "object_key, version"),
            "audit_runs": rows("audit_runs", "id"),
            "scheduler_events": rows("scheduler_events", "id"),
            "graph_nodes": graph["nodes"],
            "graph_edges": graph["edges"],
            "graph_version": graph["graph_version"],
        }


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None

