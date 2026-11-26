from __future__ import annotations

import sqlite3
from pathlib import Path


def connect(path: str | Path) -> sqlite3.Connection:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(target), isolation_level=None, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def connect_readonly(path: str | Path) -> sqlite3.Connection:
    target = Path(path).resolve()
    connection = sqlite3.connect(
        f"file:{target}?mode=ro", uri=True, isolation_level=None, timeout=10
    )
    connection.row_factory = sqlite3.Row
    return connection


def ensure_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS migration_batches (
            batch_id TEXT PRIMARY KEY,
            source_path TEXT NOT NULL,
            source_sha TEXT NOT NULL,
            target_path TEXT NOT NULL,
            total_records INTEGER NOT NULL,
            checkpoint INTEGER NOT NULL,
            status TEXT NOT NULL,
            started_mono_ms INTEGER NOT NULL,
            updated_mono_ms INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS migration_records (
            batch_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            pod_id TEXT NOT NULL,
            source_format_version INTEGER NOT NULL,
            source_line INTEGER NOT NULL,
            source_sha TEXT NOT NULL,
            target_json TEXT NOT NULL,
            target_sha TEXT NOT NULL,
            PRIMARY KEY (batch_id, ordinal),
            FOREIGN KEY (batch_id) REFERENCES migration_batches(batch_id)
        );
        CREATE TABLE IF NOT EXISTS ledger_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS ledger_entries (
            pod_id TEXT PRIMARY KEY,
            node_name TEXT NOT NULL,
            total_score INTEGER NOT NULL,
            veto_code INTEGER NOT NULL,
            plugin_scores_json TEXT NOT NULL,
            plugin_contributions_json TEXT NOT NULL,
            error_bound INTEGER NOT NULL,
            parameter_version INTEGER NOT NULL,
            scale_factor INTEGER NOT NULL,
            format_versions_json TEXT NOT NULL,
            status TEXT NOT NULL,
            conclusion_json TEXT NOT NULL,
            conclusion_digest TEXT NOT NULL,
            cursor INTEGER NOT NULL,
            session_id TEXT,
            updated_mono_ms INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS ledger_snapshots (
            session_id TEXT PRIMARY KEY,
            content_digest TEXT NOT NULL,
            format_version_set_json TEXT NOT NULL,
            parameter_version INTEGER NOT NULL,
            scale_factor INTEGER NOT NULL,
            cursor INTEGER NOT NULL,
            created_mono_ms INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS ledger_snapshot_entries (
            session_id TEXT NOT NULL,
            pod_id TEXT NOT NULL,
            content_digest TEXT NOT NULL,
            status TEXT NOT NULL,
            PRIMARY KEY (session_id, pod_id),
            FOREIGN KEY (session_id) REFERENCES ledger_snapshots(session_id)
        );
        CREATE TABLE IF NOT EXISTS ledger_replacement_events (
            session_id TEXT NOT NULL,
            sequence INTEGER NOT NULL,
            action TEXT NOT NULL,
            pod_id TEXT NOT NULL,
            old_digest TEXT,
            new_digest TEXT,
            PRIMARY KEY (session_id, sequence)
        );
        """
    )
