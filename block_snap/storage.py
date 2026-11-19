"""SQLite metadata store plus content-addressed block repository.

Layout under a workspace directory::

    meta/block_snap.sqlite   all manifests, snapshots, jobs, audit verdicts
    blocks/<hash>            content-addressed block data (write once)
    backups/                 read-only manifest backups
    tmp/                     rollback staging area
    volumes/                 raw volume images
"""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Optional

from .errors import DomainError, E_CHAIN
from .volume import block_digest

SCHEMA = """
CREATE TABLE IF NOT EXISTS volumes (
    name        TEXT PRIMARY KEY,
    path        TEXT NOT NULL,
    block_size  INTEGER NOT NULL,
    block_count INTEGER NOT NULL,
    seed        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS manifests (
    volume_name     TEXT PRIMARY KEY,
    format_version  INTEGER NOT NULL,
    content_digest  TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
    volume_name      TEXT NOT NULL,
    snapshot_id      TEXT NOT NULL,
    name             TEXT NOT NULL,
    version_no       INTEGER NOT NULL,
    parent_id        TEXT,
    bitmap           BLOB NOT NULL,
    created_at       TEXT NOT NULL,
    PRIMARY KEY (volume_name, snapshot_id),
    UNIQUE (volume_name, name)
);
CREATE TABLE IF NOT EXISTS block_refs (
    volume_name  TEXT NOT NULL,
    snapshot_id  TEXT NOT NULL,
    block_no     INTEGER NOT NULL,
    block_hash   TEXT NOT NULL,
    PRIMARY KEY (volume_name, snapshot_id, block_no)
);
CREATE TABLE IF NOT EXISTS current_state (
    volume_name TEXT NOT NULL,
    block_no    INTEGER NOT NULL,
    block_hash  TEXT NOT NULL,
    PRIMARY KEY (volume_name, block_no)
);
CREATE TABLE IF NOT EXISTS rollback_jobs (
    job_id            TEXT PRIMARY KEY,
    volume_name       TEXT NOT NULL,
    target_snapshot   TEXT NOT NULL,
    manifest_version  INTEGER NOT NULL,
    state             TEXT NOT NULL,
    staging_path      TEXT,
    path_used         TEXT,
    verified_count    INTEGER NOT NULL DEFAULT 0,
    total_count       INTEGER NOT NULL DEFAULT 0,
    reason            TEXT,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_verdicts (
    volume_name      TEXT PRIMARY KEY,
    manifest_version INTEGER NOT NULL,
    ok               INTEGER NOT NULL,
    payload          TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
"""

JOB_RUNNING = "RUNNING"
JOB_COMMITTED = "COMMITTED"
JOB_UNKNOWN = "UNKNOWN"


class BlockStore:
    """Content-addressed, immutable block directory."""

    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)

    def path_for(self, block_hash: str) -> str:
        return os.path.join(self.root, block_hash)

    def put(self, data: bytes) -> str:
        block_hash = block_digest(data)
        path = self.path_for(block_hash)
        if not os.path.exists(path):
            tmp = path + ".part"
            with open(tmp, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        return block_hash

    def get(self, block_hash: str) -> bytes:
        path = self.path_for(block_hash)
        if not os.path.exists(path):
            raise DomainError(E_CHAIN, f"missing block object {block_hash[:12]}...", field="block_store")
        with open(path, "rb") as handle:
            return handle.read()

    def exists(self, block_hash: str) -> bool:
        return os.path.exists(self.path_for(block_hash))


class MetadataStore:
    """Thin SQLite wrapper used by the snapshot manager."""

    def __init__(self, db_path: str) -> None:
        absolute_path = os.path.abspath(db_path)
        self.db_path = absolute_path
        os.makedirs(os.path.dirname(absolute_path), exist_ok=True)
        self.conn = sqlite3.connect(absolute_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def commit(self) -> None:
        self.conn.commit()

    # -- volume ---------------------------------------------------------
    def upsert_volume(self, name: str, path: str, block_size: int, block_count: int, seed: str, when: str) -> None:
        self.conn.execute(
            "INSERT INTO volumes(name, path, block_size, block_count, seed, created_at) "
            "VALUES (?,?,?,?,?,?) ON CONFLICT(name) DO UPDATE SET "
            "path=excluded.path, block_size=excluded.block_size, "
            "block_count=excluded.block_count, seed=excluded.seed",
            (name, path, block_size, block_count, seed, when),
        )

    def get_volume(self, name: str) -> Optional[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM volumes WHERE name=?", (name,)).fetchone()

    # -- manifest -------------------------------------------------------
    def get_manifest(self, volume_name: str) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM manifests WHERE volume_name=?", (volume_name,)
        ).fetchone()

    def put_manifest(self, volume_name: str, format_version: int, content_digest: str, when: str) -> None:
        self.conn.execute(
            "INSERT INTO manifests(volume_name, format_version, content_digest, updated_at) "
            "VALUES (?,?,?,?) ON CONFLICT(volume_name) DO UPDATE SET "
            "format_version=excluded.format_version, content_digest=excluded.content_digest, "
            "updated_at=excluded.updated_at",
            (volume_name, format_version, content_digest, when),
        )

    # -- snapshots ------------------------------------------------------
    def snapshot_exists(self, volume_name: str, name: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM snapshots WHERE volume_name=? AND name=?", (volume_name, name)
        ).fetchone()
        return row is not None

    def list_snapshots(self, volume_name: str) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT * FROM snapshots WHERE volume_name=? ORDER BY version_no ASC",
                (volume_name,),
            ).fetchall()
        )

    def get_snapshot(self, volume_name: str, snapshot_id: str) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM snapshots WHERE volume_name=? AND snapshot_id=?",
            (volume_name, snapshot_id),
        ).fetchone()

    def max_version(self, volume_name: str) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(version_no), 0) AS v FROM snapshots WHERE volume_name=?",
            (volume_name,),
        ).fetchone()
        return int(row["v"])

    def insert_snapshot(
        self, volume_name: str, snapshot_id: str, name: str, version_no: int,
        parent_id: Optional[str], bitmap: bytes, when: str,
    ) -> None:
        self.conn.execute(
            "INSERT INTO snapshots(volume_name, snapshot_id, name, version_no, parent_id, bitmap, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (volume_name, snapshot_id, name, version_no, parent_id, bitmap, when),
        )

    def delete_all_snapshots(self, volume_name: str) -> int:
        count = self.conn.execute(
            "SELECT COUNT(*) AS c FROM snapshots WHERE volume_name=?", (volume_name,)
        ).fetchone()["c"]
        self.conn.execute("DELETE FROM block_refs WHERE volume_name=?", (volume_name,))
        self.conn.execute("DELETE FROM snapshots WHERE volume_name=?", (volume_name,))
        return int(count)

    # -- block refs -----------------------------------------------------
    def insert_ref(self, volume_name: str, snapshot_id: str, block_no: int, block_hash: str) -> None:
        self.conn.execute(
            "INSERT INTO block_refs(volume_name, snapshot_id, block_no, block_hash) VALUES (?,?,?,?)",
            (volume_name, snapshot_id, block_no, block_hash),
        )

    def get_refs(self, volume_name: str, snapshot_id: str) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT * FROM block_refs WHERE volume_name=? AND snapshot_id=? ORDER BY block_no ASC",
                (volume_name, snapshot_id),
            ).fetchall()
        )

    def ref_map(self, volume_name: str, snapshot_id: str) -> dict[int, str]:
        return {int(row["block_no"]): row["block_hash"] for row in self.get_refs(volume_name, snapshot_id)}

    # -- current live state --------------------------------------------
    def current_map(self, volume_name: str) -> dict[int, str]:
        rows = self.conn.execute(
            "SELECT block_no, block_hash FROM current_state WHERE volume_name=?", (volume_name,)
        ).fetchall()
        return {int(row["block_no"]): row["block_hash"] for row in rows}

    def replace_current_state(self, volume_name: str, block_map: dict[int, str]) -> None:
        self.conn.execute("DELETE FROM current_state WHERE volume_name=?", (volume_name,))
        self.conn.executemany(
            "INSERT INTO current_state(volume_name, block_no, block_hash) VALUES (?,?,?)",
            [(volume_name, block_no, block_hash) for block_no, block_hash in sorted(block_map.items())],
        )

    def patch_current_state(self, volume_name: str, block_map: dict[int, str]) -> None:
        self.conn.executemany(
            "INSERT INTO current_state(volume_name, block_no, block_hash) VALUES (?,?,?) "
            "ON CONFLICT(volume_name, block_no) DO UPDATE SET block_hash=excluded.block_hash",
            [(volume_name, block_no, block_hash) for block_no, block_hash in sorted(block_map.items())],
        )

    # -- rollback jobs --------------------------------------------------
    def upsert_job(self, job: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT INTO rollback_jobs(job_id, volume_name, target_snapshot, manifest_version, state, "
            "staging_path, path_used, verified_count, total_count, reason, created_at, updated_at) "
            "VALUES (:job_id,:volume_name,:target_snapshot,:manifest_version,:state,:staging_path,"
            ":path_used,:verified_count,:total_count,:reason,:created_at,:updated_at) "
            "ON CONFLICT(job_id) DO UPDATE SET state=excluded.state, staging_path=excluded.staging_path, "
            "path_used=excluded.path_used, verified_count=excluded.verified_count, "
            "total_count=excluded.total_count, reason=excluded.reason, updated_at=excluded.updated_at",
            job,
        )

    def get_job(self, job_id: str) -> Optional[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM rollback_jobs WHERE job_id=?", (job_id,)).fetchone()

    def list_jobs(self, volume_name: str) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT * FROM rollback_jobs WHERE volume_name=? ORDER BY created_at", (volume_name,)
            ).fetchall()
        )

    # -- audit ----------------------------------------------------------
    def put_audit(self, volume_name: str, manifest_version: int, ok: bool, payload: dict[str, Any], when: str) -> None:
        self.conn.execute(
            "INSERT INTO audit_verdicts(volume_name, manifest_version, ok, payload, updated_at) "
            "VALUES (?,?,?,?,?) ON CONFLICT(volume_name) DO UPDATE SET "
            "manifest_version=excluded.manifest_version, ok=excluded.ok, "
            "payload=excluded.payload, updated_at=excluded.updated_at",
            (volume_name, manifest_version, 1 if ok else 0, json.dumps(payload, sort_keys=True), when),
        )

    def get_audit(self, volume_name: str) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM audit_verdicts WHERE volume_name=?", (volume_name,)
        ).fetchone()
