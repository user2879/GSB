from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .canonical import canonical_bytes, digest_value
from .errors import ReplayError

LEDGER_FORMAT_VERSION = 1

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS batch_plan (
    batch_id TEXT PRIMARY KEY,
    ordinal INTEGER NOT NULL,
    domain_id TEXT NOT NULL,
    batch_seq INTEGER NOT NULL,
    raw_digest TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    format_version INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS batch_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    domain_id TEXT NOT NULL,
    batch_seq INTEGER NOT NULL,
    status TEXT NOT NULL,
    format_version INTEGER NOT NULL,
    config_version INTEGER NOT NULL DEFAULT 1,
    payload_digest TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    committed_monotonic_ms INTEGER NOT NULL,
    committed_utc TEXT NOT NULL,
    UNIQUE(batch_id, config_version)
);
CREATE TABLE IF NOT EXISTS invalidations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_version INTEGER NOT NULL,
    change_kind TEXT NOT NULL,
    changed_ref TEXT,
    affected_shards_json TEXT NOT NULL,
    affected_domains_json TEXT NOT NULL,
    format_version INTEGER NOT NULL,
    digest TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS ui_reads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_version INTEGER NOT NULL,
    cache_digest TEXT NOT NULL,
    elapsed_ms REAL NOT NULL,
    bytes_read INTEGER NOT NULL,
    monotonic_start_ns INTEGER,
    format_version INTEGER NOT NULL,
    digest TEXT NOT NULL,
    created_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cursors (
    name TEXT PRIMARY KEY,
    ordinal INTEGER NOT NULL,
    last_ref TEXT,
    config_version INTEGER,
    format_version INTEGER NOT NULL,
    digest TEXT NOT NULL,
    updated_monotonic_ms INTEGER NOT NULL,
    updated_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS stage_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    config_version INTEGER NOT NULL,
    batch_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    domain_id TEXT,
    shard_id TEXT,
    field_name TEXT NOT NULL,
    item_key TEXT NOT NULL,
    canonical_json TEXT NOT NULL,
    digest TEXT NOT NULL,
    UNIQUE(config_version, item_key)
);
CREATE TABLE IF NOT EXISTS recovery_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT,
    event_type TEXT NOT NULL,
    format_version INTEGER NOT NULL,
    digest TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id TEXT PRIMARY KEY,
    format_version INTEGER NOT NULL,
    config_version INTEGER NOT NULL,
    path TEXT NOT NULL,
    file_sha256 TEXT NOT NULL,
    digest TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_utc TEXT NOT NULL
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def initialize(self) -> None:
        with self.connect() as conn:
            current = conn.execute("PRAGMA user_version").fetchone()[0]
            if current > LEDGER_FORMAT_VERSION:
                raise ReplayError("ledger format version is newer than supported", path=str(self.path), user_version=current)
            conn.executescript(SCHEMA_SQL)
            conn.execute(f"PRAGMA user_version={LEDGER_FORMAT_VERSION}")
            conn.execute(
                "INSERT OR IGNORE INTO meta(key,value) VALUES('ledger_format_version', ?)",
                (str(LEDGER_FORMAT_VERSION),),
            )

    def user_version(self) -> int:
        with self.connect() as conn:
            return int(conn.execute("PRAGMA user_version").fetchone()[0])

    def init_batch_plan(self, batches: Iterable[dict[str, Any]]) -> None:
        rows = []
        for ordinal, batch in enumerate(sorted(batches, key=lambda b: int(b.get("stream_ordinal", 0))), start=1):
            rows.append(
                (
                    batch["batch_id"],
                    int(batch.get("stream_ordinal", ordinal)),
                    batch["domain_id"],
                    int(batch["batch_seq"]),
                    str(batch.get("batch_digest", digest_value(batch))),
                )
            )
        with self.connect() as conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO batch_plan(batch_id, ordinal, domain_id, batch_seq, raw_digest, status, format_version)
                VALUES(?,?,?,?,?,'pending',1)
                """,
                rows,
            )

    def pending_batches(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT p.* FROM batch_plan p
                LEFT JOIN batch_outcomes o ON o.batch_id=p.batch_id
                WHERE o.batch_id IS NULL OR p.status IN ('pending','unknown','running')
                ORDER BY p.ordinal
                """
            ).fetchall()
            return [dict(row) for row in rows]

    def recover_unknown(self) -> list[dict[str, Any]]:
        unknown: list[dict[str, Any]] = []
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT p.* FROM batch_plan p
                LEFT JOIN batch_outcomes o ON o.batch_id=p.batch_id
                WHERE o.batch_id IS NULL
                ORDER BY p.ordinal
                """
            ).fetchall()
            for row in rows:
                batch = dict(row)
                payload = {"batch_id": batch["batch_id"], "ordinal": batch["ordinal"], "reason": "no_committed_outcome_after_recovery"}
                digest = digest_value(payload)
                conn.execute("UPDATE batch_plan SET status='unknown' WHERE batch_id=?", (batch["batch_id"],))
                conn.execute(
                    "INSERT INTO recovery_events(batch_id,event_type,format_version,digest,payload_json,created_utc) VALUES(?,?,1,?,?,?)",
                    (batch["batch_id"], "unknown_batch", digest, canonical_bytes(payload).decode(), utc_now()),
                )
                unknown.append(batch)
        return unknown

    def commit_batch_outcome(
        self,
        *,
        batch_id: str,
        ordinal: int,
        domain_id: str,
        batch_seq: int,
        outcome: dict[str, Any],
        config_version: int,
        monotonic_ms_value: int,
    ) -> str:
        payload = dict(outcome)
        payload.setdefault("batch_id", batch_id)
        payload.setdefault("status", "committed")
        digest = digest_value(payload)
        payload_json = canonical_bytes(payload).decode()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO batch_outcomes
                (batch_id,ordinal,domain_id,batch_seq,status,format_version,config_version,payload_digest,payload_json,committed_monotonic_ms,committed_utc)
                VALUES(?,?,?,?, 'committed',1,?,?,?,?,?)
                ON CONFLICT(batch_id,config_version) DO UPDATE SET
                    payload_digest=excluded.payload_digest,
                    payload_json=excluded.payload_json,
                    committed_monotonic_ms=excluded.committed_monotonic_ms,
                    committed_utc=excluded.committed_utc
                """,
                (batch_id, ordinal, domain_id, batch_seq, config_version, digest, payload_json, monotonic_ms_value, utc_now()),
            )
            conn.execute("UPDATE batch_plan SET status='committed' WHERE batch_id=?", (batch_id,))
            cursor_payload = {"name": "batch_stream", "ordinal": ordinal, "last_ref": batch_id, "config_version": config_version}
            conn.execute(
                """
                INSERT OR REPLACE INTO cursors(name,ordinal,last_ref,config_version,format_version,digest,updated_monotonic_ms,updated_utc)
                VALUES('batch_stream',?,?,?,1,?,?,?)
                """,
                (ordinal, batch_id, config_version, digest_value(cursor_payload), monotonic_ms_value, utc_now()),
            )
        return digest

    def insert_uncommitted_batch_before_kill(
        self,
        *,
        batch_id: str,
        ordinal: int,
        domain_id: str,
        batch_seq: int,
        outcome: dict[str, Any],
        config_version: int,
        monotonic_ms_value: int,
        marker_path: Path,
    ) -> None:
        """Open a transaction, write DML and marker, then wait for SIGKILL."""

        conn = self.connect()
        conn.execute("BEGIN IMMEDIATE")
        payload = dict(outcome)
        payload.setdefault("batch_id", batch_id)
        payload["status"] = "uncommitted_crash_fixture"
        digest = digest_value(payload)
        conn.execute(
            """
            INSERT INTO batch_outcomes
            (batch_id,ordinal,domain_id,batch_seq,status,format_version,config_version,payload_digest,payload_json,committed_monotonic_ms,committed_utc)
            VALUES(?,?,?,?, 'uncommitted',1,?,?,?,?,?)
            """,
            (batch_id, ordinal, domain_id, batch_seq, config_version, digest, canonical_bytes(payload).decode(), monotonic_ms_value, utc_now()),
        )
        conn.execute("UPDATE batch_plan SET status='running' WHERE batch_id=?", (batch_id,))
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(json.dumps({"event": "uncommitted_dml_visible_to_this_connection", "batch_id": batch_id}) + "\n", encoding="utf-8")
        # Parent sends SIGKILL while this sleep is in progress. No commit/rollback runs.
        import time

        for _ in range(600):
            time.sleep(0.1)
        conn.commit()  # pragma: no cover

    def record_ui_read(self, *, config_version: int, cache_digest: str, elapsed_ms: float, bytes_read: int, start_ns: int | None, end_ns: int | None) -> int:
        payload = {
            "config_version": config_version,
            "cache_digest": cache_digest,
            "elapsed_ms": elapsed_ms,
            "bytes_read": bytes_read,
        }
        digest = digest_value(payload)
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO ui_reads(config_version,cache_digest,elapsed_ms,bytes_read,monotonic_start_ns,format_version,digest,created_utc)
                VALUES(?,?,?,?,?,1,?,?)
                """,
                (config_version, cache_digest, elapsed_ms, bytes_read, start_ns, digest, utc_now()),
            )
            read_id = int(cursor.lastrowid)
            cursor_payload = {"name": "last_ui_read", "read_id": read_id, "elapsed_ms": elapsed_ms}
            conn.execute(
                """
                INSERT OR REPLACE INTO cursors(name,ordinal,last_ref,config_version,format_version,digest,updated_monotonic_ms,updated_utc)
                VALUES('last_ui_read',?,?,?,1,?,?,?)
                """,
                (read_id, f"ui_read:{read_id}", config_version, digest_value(cursor_payload), end_ns // 1_000_000 if end_ns else 0, utc_now()),
            )
            return read_id

    def commit_invalidation_and_items(
        self,
        *,
        propagation: dict[str, Any],
        items: list[dict[str, Any]],
        monotonic_ms_value: int,
        batch_outcomes: list[dict[str, Any]] | None = None,
        ui_read: dict[str, Any] | None = None,
    ) -> None:
        config_version = int(propagation["config_version"])
        digest = propagation.get("propagation_digest") or digest_value(propagation)
        with self.connect() as conn:
            for outcome_row in batch_outcomes or []:
                outcome = dict(outcome_row["outcome"])
                outcome.setdefault("batch_id", outcome_row["batch_id"])
                outcome_digest = digest_value(outcome)
                conn.execute(
                    """
                    INSERT INTO batch_outcomes
                    (batch_id,ordinal,domain_id,batch_seq,status,format_version,config_version,payload_digest,payload_json,committed_monotonic_ms,committed_utc)
                    VALUES(?,?,?,?,?,1,?,?,?,?,?)
                    ON CONFLICT(batch_id,config_version) DO UPDATE SET
                        payload_digest=excluded.payload_digest,
                        payload_json=excluded.payload_json,
                        committed_utc=excluded.committed_utc
                    """,
                    (
                        outcome_row["batch_id"],
                        int(outcome_row["ordinal"]),
                        outcome_row["domain_id"],
                        int(outcome_row["batch_seq"]),
                        outcome_row.get("status", "committed_revision"),
                        config_version,
                        outcome_digest,
                        canonical_bytes(outcome).decode(),
                        monotonic_ms_value,
                        utc_now(),
                    ),
                )
            conn.execute(
                """
                INSERT INTO invalidations(config_version,change_kind,changed_ref,affected_shards_json,affected_domains_json,format_version,digest,payload_json,created_utc)
                VALUES(?,?,?,?,?,1,?,?,?)
                """,
                (
                    config_version,
                    propagation["change"]["kind"],
                    propagation["change"].get("field") or propagation["change"].get("domain_id"),
                    canonical_bytes(propagation.get("affected_shards", [])).decode(),
                    canonical_bytes(propagation.get("affected_domains", [])).decode(),
                    digest,
                    canonical_bytes(propagation).decode(),
                    utc_now(),
                ),
            )
            for item in items:
                item_digest = item.get("digest") or digest_value(item["value"])
                canonical = canonical_bytes(item["value"]).decode()
                conn.execute(
                    """
                    INSERT INTO stage_items(config_version,batch_id,stage,domain_id,shard_id,field_name,item_key,canonical_json,digest)
                    VALUES(?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(config_version,item_key) DO UPDATE SET
                        batch_id=excluded.batch_id,
                        stage=excluded.stage,
                        domain_id=excluded.domain_id,
                        shard_id=excluded.shard_id,
                        field_name=excluded.field_name,
                        canonical_json=excluded.canonical_json,
                        digest=excluded.digest
                    """,
                    (
                        config_version,
                        item["batch_id"],
                        item["stage"],
                        item.get("domain_id"),
                        item.get("shard_id"),
                        item.get("field_name", "canonical"),
                        item["item_key"],
                        canonical,
                        item_digest,
                    ),
                )
            ui_read_id: int | None = None
            if ui_read is not None:
                read_payload = {
                    "config_version": int(ui_read["config_version"]),
                    "cache_digest": str(ui_read["cache_digest"]),
                    "elapsed_ms": float(ui_read["elapsed_ms"]),
                    "bytes_read": int(ui_read["bytes_read"]),
                }
                read_digest = digest_value(read_payload)
                read_cursor = conn.execute(
                    """
                    INSERT INTO ui_reads(config_version,cache_digest,elapsed_ms,bytes_read,monotonic_start_ns,format_version,digest,created_utc)
                    VALUES(?,?,?,?,?,1,?,?)
                    """,
                    (
                        int(ui_read["config_version"]),
                        str(ui_read["cache_digest"]),
                        float(ui_read["elapsed_ms"]),
                        int(ui_read["bytes_read"]),
                        ui_read.get("start_ns"),
                        read_digest,
                        utc_now(),
                    ),
                )
                ui_read_id = int(read_cursor.lastrowid)
                read_cursor_payload = {"name": "last_ui_read", "read_id": ui_read_id, "elapsed_ms": ui_read["elapsed_ms"]}
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cursors(name,ordinal,last_ref,config_version,format_version,digest,updated_monotonic_ms,updated_utc)
                    VALUES('last_ui_read',?,?,?,1,?,?,?)
                    """,
                    (
                        ui_read_id,
                        f"ui_read:{ui_read_id}",
                        int(ui_read["config_version"]),
                        digest_value(read_cursor_payload),
                        int(ui_read.get("end_ns", 0)) // 1_000_000 if ui_read.get("end_ns") else monotonic_ms_value,
                        utc_now(),
                    ),
                )
            cursor_payload = {"name": "conclusion", "config_version": config_version, "items": len(items), "propagation_digest": digest}
            conn.execute(
                """
                INSERT OR REPLACE INTO cursors(name,ordinal,last_ref,config_version,format_version,digest,updated_monotonic_ms,updated_utc)
                VALUES('conclusion',?,?,?,1,?,?,?)
                """,
                (config_version, f"v{config_version}:{len(items)}items", config_version, digest_value(cursor_payload), monotonic_ms_value, utc_now()),
            )
            if ui_read is not None:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO cursors(name,ordinal,last_ref,config_version,format_version,digest,updated_monotonic_ms,updated_utc)
                    VALUES('last_combined_commit',?,?,?,1,?,?,?)
                    """,
                    (
                        ui_read_id or 0,
                        f"v{config_version}:read:{ui_read_id}",
                        config_version,
                        digest_value({"config_version": config_version, "read_id": ui_read_id, "invalidation_digest": digest}),
                        monotonic_ms_value,
                        utc_now(),
                    ),
                )

    def list_snapshots(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM snapshots ORDER BY snapshot_id").fetchall()
            return [dict(row) for row in rows]

    def latest_config_version_with_items(self) -> int | None:
        with self.connect() as conn:
            row = conn.execute("SELECT MAX(config_version) AS v FROM stage_items").fetchone()
            return int(row["v"]) if row and row["v"] is not None else None

    def stage_items(self, config_version: int | None = None) -> list[dict[str, Any]]:
        with self.connect() as conn:
            if config_version is None:
                config_version = self.latest_config_version_with_items()
            rows = conn.execute(
                "SELECT * FROM stage_items WHERE config_version=? ORDER BY id",
                (config_version,),
            ).fetchall()
            return [dict(row) for row in rows]

    def batch_outcomes(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM batch_outcomes ORDER BY ordinal, config_version").fetchall()
            return [dict(row) for row in rows]

    def latest_batch_outcomes(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT o.* FROM batch_outcomes o
                JOIN (
                    SELECT batch_id, MAX(config_version) AS max_version
                    FROM batch_outcomes GROUP BY batch_id
                ) latest ON latest.batch_id=o.batch_id AND latest.max_version=o.config_version
                ORDER BY o.ordinal
                """
            ).fetchall()
            return [dict(row) for row in rows]

    def ui_reads(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM ui_reads ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
            return [dict(row) for row in rows]

    def recovery_events(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM recovery_events ORDER BY id").fetchall()
            return [dict(row) for row in rows]

    def create_snapshot(self, snapshot_dir: Path, config_version: int) -> dict[str, Any]:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_id = f"snapshot-v{config_version}-{monotonic_compact()}"
        db_path = snapshot_dir / f"{snapshot_id}.sqlite"
        with self.connect() as source:
            target = sqlite3.connect(db_path)
            try:
                source.backup(target)
            finally:
                target.close()
        file_bytes = db_path.read_bytes()
        file_sha = __import__("hashlib").sha256(file_bytes).hexdigest()
        metadata = {
            "snapshot_id": snapshot_id,
            "format_version": LEDGER_FORMAT_VERSION,
            "config_version": config_version,
            "path": str(db_path),
            "file_sha256": file_sha,
            "created_utc": utc_now(),
        }
        metadata["digest"] = digest_value({key: value for key, value in metadata.items() if key != "digest"})
        with self.connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO snapshots(snapshot_id,format_version,config_version,path,file_sha256,digest,payload_json,created_utc) VALUES(?,?,?,?,?,?,?,?)",
                (snapshot_id, 1, config_version, str(db_path), file_sha, metadata["digest"], canonical_bytes(metadata).decode(), metadata["created_utc"]),
            )
        (snapshot_dir / f"{snapshot_id}.json").write_bytes(canonical_bytes(metadata))
        return metadata


def monotonic_compact() -> str:
    from .canonical import monotonic_ns

    return format(monotonic_ns(), "x")
