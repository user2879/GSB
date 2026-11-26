from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from . import LEDGER_FORMAT_VERSION
from .canonical import canonical_bytes, digest_json, sha256_hex
from .watermark import monotonic_ms

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS shard_outcomes (
    shard_id TEXT PRIMARY KEY,
    node_id TEXT,
    status TEXT NOT NULL,
    sample_rows INTEGER,
    participated_rows INTEGER,
    sample_digest TEXT,
    aggregate_digest TEXT,
    aggregate_json TEXT,
    caliber_version TEXT,
    quota_version INTEGER,
    lease_id TEXT,
    lease_version INTEGER,
    resource_units INTEGER,
    used_units_before INTEGER,
    used_units_after INTEGER,
    global_water_q INTEGER,
    node_slot_water_q INTEGER,
    node_lease_water_q INTEGER,
    fixed_scale INTEGER,
    quantization_json TEXT,
    error_json TEXT,
    event_seq INTEGER,
    wait_started_ms INTEGER,
    wait_ended_ms INTEGER,
    wait_duration_ms INTEGER,
    started_ms INTEGER,
    finished_ms INTEGER,
    content_digest TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS lease_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_seq INTEGER,
    lease_id TEXT,
    shard_id TEXT,
    node_id TEXT,
    kind TEXT NOT NULL,
    from_version INTEGER,
    to_version INTEGER,
    at_monotonic_ms INTEGER NOT NULL,
    expires_monotonic_ms INTEGER,
    detail_json TEXT,
    content_digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS watermarks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_seq INTEGER,
    scope_type TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    used_units INTEGER,
    active_shards INTEGER,
    active_leases INTEGER,
    denominator INTEGER,
    scale INTEGER,
    water_q INTEGER,
    report_json TEXT,
    at_monotonic_ms INTEGER NOT NULL,
    content_digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS event_journal (
    event_seq INTEGER PRIMARY KEY,
    event_hash TEXT NOT NULL,
    event_json TEXT NOT NULL,
    decision_json TEXT NOT NULL,
    affected_json TEXT,
    at_monotonic_ms INTEGER NOT NULL,
    content_digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS progress (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    last_event_seq INTEGER NOT NULL,
    status TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS recovery_records (
    shard_id TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    detail_json TEXT,
    at_monotonic_ms INTEGER NOT NULL,
    content_digest TEXT NOT NULL
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def row_digest(table: str, payload: dict[str, Any]) -> str:
    return sha256_hex(canonical_bytes({"table": table, "row": payload}))


class Ledger:
    def __init__(self, path: str | Path, conn: sqlite3.Connection, replaced: bool = False, previous: dict[str, Any] | None = None) -> None:
        self.path = Path(path)
        self.conn = conn
        self.lock = threading.RLock()
        self.replaced = replaced
        self.previous = previous or {}

    @classmethod
    def open_session(cls, path: str | Path, spec_digest: str) -> "Ledger":
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        previous: dict[str, Any] = {}
        replaced = False
        existed = path.exists()
        if path.exists():
            try:
                probe = sqlite3.connect(str(path))
                probe.row_factory = sqlite3.Row
                cls._create_schema(probe)
                rows = probe.execute("SELECT key, value FROM meta").fetchall()
                previous = {row["key"]: row["value"] for row in rows}
                probe.close()
                if previous.get("format_version") != str(LEDGER_FORMAT_VERSION) or previous.get("spec_digest") != spec_digest:
                    replaced = True
            except sqlite3.DatabaseError:
                replaced = True
            if replaced:
                for suffix in ("", "-wal", "-shm", "-journal"):
                    candidate = Path(str(path) + suffix)
                    try:
                        candidate.unlink()
                    except FileNotFoundError:
                        pass
        conn = sqlite3.connect(str(path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=DELETE")
        cls._create_schema(conn)
        ledger = cls(path, conn, replaced=replaced, previous=previous)
        if not existed or replaced:
            ledger.initialize_meta(spec_digest)
        else:
            ledger.ensure_meta(spec_digest)
        return ledger

    @classmethod
    def open_read_only(cls, path: str | Path) -> "Ledger":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(str(path))
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return cls(path, conn)

    @staticmethod
    def _create_schema(conn: sqlite3.Connection) -> None:
        conn.executescript(SCHEMA_SQL)
        conn.commit()

    def initialize_meta(self, spec_digest: str) -> None:
        with self.transaction():
            self.conn.execute(
                "INSERT INTO meta(key,value) VALUES('format_version',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(LEDGER_FORMAT_VERSION),),
            )
            self.conn.execute(
                "INSERT INTO meta(key,value) VALUES('spec_digest',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (spec_digest,),
            )
            self.conn.execute(
                "INSERT INTO meta(key,value) VALUES('opened_at',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (utc_now(),),
            )
            self.set_progress_locked(0, "opened")

    def ensure_meta(self, spec_digest: str) -> None:
        with self.transaction():
            self.conn.execute(
                "INSERT INTO meta(key,value) VALUES('format_version',?) ON CONFLICT(key) DO NOTHING",
                (str(LEDGER_FORMAT_VERSION),),
            )
            self.conn.execute(
                "INSERT INTO meta(key,value) VALUES('spec_digest',?) ON CONFLICT(key) DO NOTHING",
                (spec_digest,),
            )
            if self.progress() is None:
                self.set_progress_locked(0, "reopened")

    @contextmanager
    def transaction(self) -> Iterator["Ledger"]:
        with self.lock:
            try:
                with self.conn:
                    yield self
                    self.update_root_digest_locked()
            except Exception:
                # sqlite connection context rolls back; explicit rollback is harmless.
                if self.conn.in_transaction:
                    self.conn.rollback()
                raise

    def meta(self) -> dict[str, str]:
        rows = self.conn.execute("SELECT key,value FROM meta").fetchall()
        return {row["key"]: row["value"] for row in rows}

    def set_meta_locked(self, key: str, value: Any) -> None:
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(value)),
        )

    def insert_lease_event_locked(self, event: dict[str, Any]) -> int:
        payload = dict(event)
        at_ms = int(payload.setdefault("at_monotonic_ms", monotonic_ms()))
        digest = row_digest("lease_events", payload)
        cur = self.conn.execute(
            """INSERT INTO lease_events(event_seq,lease_id,shard_id,node_id,kind,from_version,to_version,
               at_monotonic_ms,expires_monotonic_ms,detail_json,content_digest)
               VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (
                payload.get("event_seq"), payload.get("lease_id"), payload.get("shard_id"),
                payload.get("node_id"), payload["kind"], payload.get("from_version"),
                payload.get("to_version"), at_ms, payload.get("expires_monotonic_ms"),
                json_dumps(payload.get("detail", {})), digest,
            ),
        )
        return int(cur.lastrowid)

    def insert_watermark_locked(self, water: dict[str, Any]) -> int:
        payload = dict(water)
        payload.setdefault("at_monotonic_ms", monotonic_ms())
        digest = row_digest("watermarks", payload)
        cur = self.conn.execute(
            """INSERT INTO watermarks(event_seq,scope_type,scope_id,used_units,active_shards,active_leases,
               denominator,scale,water_q,report_json,at_monotonic_ms,content_digest)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                payload.get("event_seq"), payload["scope_type"], payload["scope_id"],
                payload.get("used_units"), payload.get("active_shards"), payload.get("active_leases"),
                payload.get("denominator"), payload.get("scale"), payload.get("water_q"),
                json_dumps(payload.get("report", {})), payload["at_monotonic_ms"], digest,
            ),
        )
        return int(cur.lastrowid)

    def upsert_outcome_locked(self, outcome: dict[str, Any]) -> None:
        payload = dict(outcome)
        payload.setdefault("updated_at", utc_now())
        payload.setdefault("finished_ms", monotonic_ms())
        digest = row_digest("shard_outcomes", payload)
        columns = [
            "shard_id", "node_id", "status", "sample_rows", "participated_rows", "sample_digest",
            "aggregate_digest", "aggregate_json", "caliber_version", "quota_version", "lease_id",
            "lease_version", "resource_units", "used_units_before", "used_units_after",
            "global_water_q", "node_slot_water_q", "node_lease_water_q", "fixed_scale",
            "quantization_json", "error_json", "event_seq", "wait_started_ms", "wait_ended_ms",
            "wait_duration_ms", "started_ms", "finished_ms", "content_digest", "updated_at",
        ]
        values = {
            "shard_id": payload["shard_id"],
            "node_id": payload.get("node_id"),
            "status": payload["status"],
            "sample_rows": payload.get("sample_rows"),
            "participated_rows": payload.get("participated_rows"),
            "sample_digest": payload.get("sample_digest"),
            "aggregate_digest": payload.get("aggregate_digest"),
            "aggregate_json": payload.get("aggregate_json") if isinstance(payload.get("aggregate_json"), str) else json_dumps(payload.get("aggregate_json")),
            "caliber_version": payload.get("caliber_version"),
            "quota_version": payload.get("quota_version"),
            "lease_id": payload.get("lease_id"),
            "lease_version": payload.get("lease_version"),
            "resource_units": payload.get("resource_units"),
            "used_units_before": payload.get("used_units_before"),
            "used_units_after": payload.get("used_units_after"),
            "global_water_q": payload.get("global_water_q"),
            "node_slot_water_q": payload.get("node_slot_water_q"),
            "node_lease_water_q": payload.get("node_lease_water_q"),
            "fixed_scale": payload.get("fixed_scale"),
            "quantization_json": json_dumps(payload.get("quantization")),
            "error_json": json_dumps(payload.get("error")),
            "event_seq": payload.get("event_seq"),
            "wait_started_ms": payload.get("wait_started_ms"),
            "wait_ended_ms": payload.get("wait_ended_ms"),
            "wait_duration_ms": payload.get("wait_duration_ms"),
            "started_ms": payload.get("started_ms"),
            "finished_ms": payload.get("finished_ms"),
            "content_digest": digest,
            "updated_at": payload["updated_at"],
        }
        placeholders = ",".join("?" for _ in columns)
        updates = ",".join(f"{name}=excluded.{name}" for name in columns if name != "shard_id")
        self.conn.execute(
            f"INSERT INTO shard_outcomes ({','.join(columns)}) VALUES ({placeholders}) "
            f"ON CONFLICT(shard_id) DO UPDATE SET {updates}",
            [values[name] for name in columns],
        )

    def journal_event_locked(self, seq: int, event_hash: str, event: dict[str, Any], decision: dict[str, Any], affected: Any = None) -> None:
        at_ms = monotonic_ms()
        payload = {
            "event_seq": seq,
            "event_hash": event_hash,
            "event": event,
            "decision": decision,
            "affected": affected,
            "at_monotonic_ms": at_ms,
        }
        digest = row_digest("event_journal", payload)
        self.conn.execute(
            """INSERT INTO event_journal(event_seq,event_hash,event_json,decision_json,affected_json,
               at_monotonic_ms,content_digest) VALUES(?,?,?,?,?,?,?)
               ON CONFLICT(event_seq) DO UPDATE SET event_hash=excluded.event_hash,
                 event_json=excluded.event_json, decision_json=excluded.decision_json,
                 affected_json=excluded.affected_json, at_monotonic_ms=excluded.at_monotonic_ms,
                 content_digest=excluded.content_digest""",
            (seq, event_hash, json_dumps(event), json_dumps(decision), json_dumps(affected), at_ms, digest),
        )

    def set_progress_locked(self, last_event_seq: int, status: str) -> None:
        now = utc_now()
        at_ms = monotonic_ms()
        digest = row_digest("progress", {"last_event_seq": last_event_seq, "status": status, "updated_at": now, "at_monotonic_ms": at_ms})
        self.conn.execute(
            """INSERT INTO progress(id,last_event_seq,status,content_digest,updated_at)
               VALUES(1,?,?,?,?) ON CONFLICT(id) DO UPDATE SET last_event_seq=excluded.last_event_seq,
               status=excluded.status, content_digest=excluded.content_digest, updated_at=excluded.updated_at""",
            (last_event_seq, status, digest, now),
        )

    def upsert_recovery_locked(self, shard_id: str, state: str, detail: dict[str, Any] | None = None) -> None:
        at_ms = monotonic_ms()
        digest = row_digest("recovery_records", {"shard_id": shard_id, "state": state, "detail": detail or {}, "at": at_ms})
        self.conn.execute(
            """INSERT INTO recovery_records(shard_id,state,detail_json,at_monotonic_ms,content_digest)
               VALUES(?,?,?,?,?) ON CONFLICT(shard_id) DO UPDATE SET state=excluded.state,
               detail_json=excluded.detail_json, at_monotonic_ms=excluded.at_monotonic_ms,
               content_digest=excluded.content_digest""",
            (shard_id, state, json_dumps(detail or {}), at_ms, digest),
        )

    def update_root_digest_locked(self) -> str:
        digest = self.compute_root_digest()
        self.set_meta_locked("content_digest", digest)
        return digest

    def compute_root_digest(self) -> str:
        tables = [
            "meta", "shard_outcomes", "lease_events", "watermarks",
            "event_journal", "progress", "recovery_records",
        ]
        payload: dict[str, Any] = {
            "format_version": LEDGER_FORMAT_VERSION,
            "spec_digest": self.meta().get("spec_digest"),
        }
        for table in tables:
            rows = self.conn.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
            table_rows = [dict(row) for row in rows]
            if table == "meta":
                table_rows = [row for row in table_rows if row.get("key") != "content_digest"]
            payload[table] = table_rows
        return digest_json(payload)

    def get_outcome(self, shard_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM shard_outcomes WHERE shard_id=?", (shard_id,)).fetchone()
        return dict(row) if row else None

    def outcomes(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.conn.execute("SELECT * FROM shard_outcomes ORDER BY shard_id")]

    def lease_events(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.conn.execute("SELECT * FROM lease_events ORDER BY id")]

    def journal(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.conn.execute("SELECT * FROM event_journal ORDER BY event_seq")]

    def progress(self) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM progress WHERE id=1").fetchone()
        return dict(row) if row else None

    def recovery_records(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.conn.execute("SELECT * FROM recovery_records ORDER BY shard_id")]

    def close(self) -> None:
        self.conn.close()

    def backup_to(self, destination: str | Path) -> Path:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        target = sqlite3.connect(str(destination))
        try:
            self.conn.backup(target)
        finally:
            target.close()
        return destination


def json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
