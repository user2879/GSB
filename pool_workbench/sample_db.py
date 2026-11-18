"""Deterministic sample database construction and schema verification.

The sample library contains plain business tables only.  Deterministic task
durations are produced by the ``wb_sleep`` scalar function registered on every
pooled connection, so no real network or random timing is involved.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Callable

from .errors import BusinessError, ErrorCode

INVENTORY_TABLE = "inventory"
OPS_LOG_TABLE = "ops_log"
ALLOWED_TABLES = (INVENTORY_TABLE, OPS_LOG_TABLE)

_SCHEMA_SQL = (
    """
    CREATE TABLE inventory (
        item_id INTEGER PRIMARY KEY,
        item_name TEXT NOT NULL,
        qty INTEGER NOT NULL,
        updated_ms INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE ops_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        workload TEXT NOT NULL,
        task_id TEXT NOT NULL,
        note TEXT NOT NULL
    )
    """,
)

_SEED_INVENTORY = (
    (1, "alpha", 100, 0),
    (2, "bravo", 200, 0),
    (3, "charlie", 300, 0),
    (4, "delta", 400, 0),
)


def version_sidecar_path(db_path: str | Path) -> Path:
    return Path(str(db_path) + ".version.json")


def build_sample_db(db_path: str | Path, sample_data_version: str = "1") -> dict:
    """(Re)build the sample database deterministically and return row counts."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(str(db_path))
    try:
        for statement in _SCHEMA_SQL:
            conn.execute(statement)
        conn.executemany(
            "INSERT INTO inventory (item_id, item_name, qty, updated_ms) VALUES (?, ?, ?, ?)",
            _SEED_INVENTORY,
        )
        conn.commit()
    finally:
        conn.close()
    meta = {
        "sample_data_version": sample_data_version,
        "tables": list(ALLOWED_TABLES),
        "row_counts": {
            INVENTORY_TABLE: len(_SEED_INVENTORY),
            OPS_LOG_TABLE: 0,
        },
    }
    version_sidecar_path(db_path).write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    return meta


def read_sample_data_version(db_path: str | Path) -> str:
    sidecar = version_sidecar_path(db_path)
    if not sidecar.exists():
        raise BusinessError(
            ErrorCode.E_SCHEMA,
            "sample database version sidecar is missing; rebuild the sample library",
            [str(sidecar)],
        )
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    version = meta.get("sample_data_version")
    if not isinstance(version, str) or not version:
        raise BusinessError(
            ErrorCode.E_SCHEMA,
            "sample database version sidecar has no valid sample_data_version",
            [str(sidecar)],
        )
    return version


def verify_schema(db_path: str | Path, allowed_tables: tuple[str, ...] = ALLOWED_TABLES) -> None:
    """Raise E_SCHEMA when any required sample table is missing."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise BusinessError(
            ErrorCode.E_SCHEMA,
            f"sample database file does not exist: {db_path}",
            [str(db_path)],
        )
    conn = sqlite3.connect(str(db_path))
    try:
        present = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    finally:
        conn.close()
    missing = [table for table in allowed_tables if table not in present]
    if missing:
        raise BusinessError(
            ErrorCode.E_SCHEMA,
            "sample library is missing required table(s): " + ", ".join(missing),
            [f"schema.table.{table}" for table in missing],
        )


def _make_sleep_function(abort_event: threading.Event) -> Callable[[int], int]:
    def wb_sleep(milliseconds: int) -> int:
        waited = 0
        step = 2
        while waited < int(milliseconds):
            if abort_event.is_set():
                raise sqlite3.OperationalError("wb_sleep interrupted at statement boundary")
            remaining = int(milliseconds) - waited
            threading.Event().wait(min(step, remaining) / 1000.0)
            waited += min(step, remaining)
        return int(milliseconds)

    return wb_sleep


def open_connection(
    db_path: str | Path,
    read_only: bool,
    abort_event: threading.Event,
) -> sqlite3.Connection:
    """Open a pooled connection and register deterministic helpers on it."""
    absolute = str(Path(db_path).resolve())
    if read_only:
        uri = "file:" + absolute.replace("\\", "/") + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False, timeout=5.0)
    else:
        conn = sqlite3.connect(absolute, check_same_thread=False, timeout=5.0)
    conn.create_function("wb_sleep", 1, _make_sleep_function(abort_event))
    return conn


def clone_database(src_path: str | Path, dst_path: str | Path) -> None:
    """Byte-level logical copy used by replay so the sample tables stay untouched."""
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if dst_path.exists():
        dst_path.unlink()
    src = sqlite3.connect(str(src_path))
    try:
        dst = sqlite3.connect(str(dst_path))
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()
