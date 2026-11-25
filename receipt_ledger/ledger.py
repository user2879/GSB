from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import LEDGER_FORMAT_VERSION, SNAPSHOT_FORMAT_VERSION
from .canonical import canonical_bytes, canonical_digest, monotonic_ms
from .errors import SnapshotError
from .models import Batch, SpecBundle


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY,
    worker_id TEXT NOT NULL,
    arrival_seq INTEGER NOT NULL,
    input_json TEXT NOT NULL,
    request_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS batches (
    batch_id TEXT PRIMARY KEY,
    worker_id TEXT NOT NULL,
    batch_index INTEGER NOT NULL,
    arrival_seqs_json TEXT NOT NULL,
    request_ids_json TEXT NOT NULL,
    status TEXT NOT NULL,
    membership_digest TEXT NOT NULL,
    param_version INTEGER,
    param_digest TEXT,
    worker_snapshot_json TEXT,
    output_digest TEXT,
    frozen_output_digest TEXT,
    result_json TEXT,
    propagation_json TEXT,
    lease_id TEXT,
    lease_version INTEGER,
    lease_status TEXT,
    wait_started_ms INTEGER,
    wait_ended_ms INTEGER,
    wait_ms INTEGER,
    updated_monotonic_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS assignments (
    request_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    worker_id TEXT NOT NULL,
    arrival_seq INTEGER NOT NULL,
    rank INTEGER NOT NULL,
    assignment_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS receipts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT NOT NULL,
    revision INTEGER NOT NULL,
    source_format TEXT NOT NULL,
    normalized_json TEXT NOT NULL,
    output_digest TEXT NOT NULL,
    frozen INTEGER NOT NULL DEFAULT 1,
    monotonic_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS leases (
    lease_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    worker_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    granted_monotonic_ms INTEGER NOT NULL,
    expires_monotonic_ms INTEGER NOT NULL,
    ttl_ms INTEGER NOT NULL,
    status TEXT NOT NULL,
    release_reason TEXT,
    released_monotonic_ms INTEGER
);
CREATE TABLE IF NOT EXISTS invalidations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    epoch TEXT NOT NULL,
    worker_id TEXT,
    batch_id TEXT,
    request_id TEXT,
    node_type TEXT NOT NULL,
    dependency_level TEXT NOT NULL,
    cause TEXT NOT NULL,
    basis_json TEXT NOT NULL,
    monotonic_ms INTEGER NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_invalidation_idem
    ON invalidations(epoch, node_type, COALESCE(batch_id, ''), COALESCE(request_id, ''), cause);
CREATE TABLE IF NOT EXISTS progress (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_monotonic_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id TEXT PRIMARY KEY,
    format_version TEXT NOT NULL,
    ledger_version INTEGER NOT NULL,
    content_digest TEXT NOT NULL,
    path TEXT NOT NULL,
    created_wall TEXT NOT NULL,
    created_monotonic_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    monotonic_ms INTEGER NOT NULL,
    level TEXT NOT NULL,
    code TEXT NOT NULL,
    message TEXT NOT NULL,
    details_json TEXT NOT NULL
);
"""


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {key: row[key] for key in row.keys()}


def normalize_tables_for_digest(tables: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    normalized: dict[str, list[dict[str, Any]]] = {}
    for table, raw_rows in sorted(tables.items()):
        rows: list[dict[str, Any]] = []
        for raw_row in raw_rows or []:
            row = dict(raw_row)
            if table == "meta" and row.get("key") == "content_digest":
                continue
            if table == "snapshots":
                row.pop("content_digest", None)
            rows.append(row)
        normalized[table] = sorted(rows, key=lambda item: canonical_bytes(item))
    return normalized


def table_digest_payload(conn: sqlite3.Connection) -> dict[str, list[dict[str, Any]]]:
    tables = [
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]
    payload: dict[str, list[dict[str, Any]]] = {}
    for table in tables:
        rows = [_row_to_dict(row) for row in conn.execute(f'SELECT * FROM "{table}"')]
        payload[table] = rows
    return payload


def digest_payload(payload: dict[str, Any]) -> str:
    return canonical_digest(normalize_tables_for_digest(payload))


def database_digest(conn: sqlite3.Connection) -> str:
    return digest_payload(table_digest_payload(conn))


class Ledger:
    def __init__(self, path: str | Path = ".runtime/ledger.sqlite3", *, reset: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if reset and self.path.exists():
            self.path.unlink()
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA_SQL)
        self.conn.commit()
        self._initialize_meta()

    def _initialize_meta(self) -> None:
        with self.lock:
            with self.conn:
                self.conn.execute(
                    "INSERT OR IGNORE INTO meta(key,value) VALUES('ledger_format_version', ?)",
                    (str(LEDGER_FORMAT_VERSION),),
                )
                self.conn.execute("INSERT OR IGNORE INTO meta(key,value) VALUES('content_digest','')")
            self.refresh_digest()

    def close(self) -> None:
        with self.lock:
            self.conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def refresh_digest(self) -> str:
        with self.lock:
            digest = database_digest(self.conn)
            with self.conn:
                self.conn.execute(
                    "INSERT INTO meta(key,value) VALUES('content_digest',?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (digest,),
                )
            return digest

    def stored_digest(self) -> str:
        row = self.conn.execute("SELECT value FROM meta WHERE key='content_digest'").fetchone()
        return str(row["value"]) if row else ""

    def format_version(self) -> int:
        row = self.conn.execute("SELECT value FROM meta WHERE key='ledger_format_version'").fetchone()
        try:
            return int(row["value"]) if row else 0
        except (TypeError, ValueError):
            return 0

    def verify_digest(self) -> str:
        actual = database_digest(self.conn)
        stored = self.stored_digest()
        if not stored or actual != stored:
            raise SnapshotError(
                "账本内容摘要不符，拒绝以被篡改内容继续恢复",
                code_name="E_SNAPSHOT_CONTENT_DIGEST",
                stored_digest=stored,
                actual_digest=actual,
                db_path=str(self.path),
            )
        return actual

    def log_event(self, code: str, message: str, level: str = "INFO", **details: Any) -> None:
        with self.lock:
            with self.conn:
                self.conn.execute(
                    "INSERT INTO events(monotonic_ms,level,code,message,details_json) VALUES(?,?,?,?,?)",
                    (monotonic_ms(), level, code, message, json.dumps(details, ensure_ascii=False, sort_keys=True)),
                )
            self.refresh_digest()

    def initialize_bundle(self, bundle: SpecBundle) -> None:
        with self.lock:
            with self.conn:
                for request in bundle.requests:
                    self.conn.execute(
                        "INSERT INTO requests(request_id,worker_id,arrival_seq,input_json,request_json) "
                        "VALUES(?,?,?,?,?) ON CONFLICT(request_id) DO UPDATE SET "
                        "worker_id=excluded.worker_id, arrival_seq=excluded.arrival_seq, "
                        "input_json=excluded.input_json, request_json=excluded.request_json",
                        (
                            request.request_id,
                            request.worker_id,
                            request.arrival_seq,
                            json.dumps(list(request.vector), sort_keys=True),
                            json.dumps(request.to_json(), ensure_ascii=False, sort_keys=True),
                        ),
                    )
                self.conn.execute(
                    "INSERT INTO meta(key,value) VALUES('workers_json',?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (json.dumps([w.to_json() for w in bundle.workers], ensure_ascii=False, sort_keys=True),),
                )
                self.conn.execute(
                    "INSERT INTO meta(key,value) VALUES('rules_json',?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (json.dumps(bundle.rules.__dict__, ensure_ascii=False, sort_keys=True),),
                )
                self.conn.execute(
                    "INSERT INTO progress(key,value_json,updated_monotonic_ms) VALUES('spec_initialized',?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_monotonic_ms=excluded.updated_monotonic_ms",
                    (json.dumps({"request_count": len(bundle.requests)}, sort_keys=True), monotonic_ms()),
                )
            self.refresh_digest()

    def record_dispatch(self, batch: Batch, lease: Any) -> None:
        now = monotonic_ms()
        wait_started = getattr(lease, "wait_started_monotonic_ms", None)
        wait_ended = getattr(lease, "wait_ended_monotonic_ms", None)
        wait_ms = None
        if wait_started is not None and wait_ended is not None:
            wait_ms = int(wait_ended - wait_started)
        membership = canonical_digest(batch.material())
        with self.lock:
            with self.conn:
                self.conn.execute(
                    "INSERT INTO batches(batch_id,worker_id,batch_index,arrival_seqs_json,request_ids_json,"
                    "status,membership_digest,lease_id,lease_version,lease_status,wait_started_ms,wait_ended_ms,"
                    "wait_ms,updated_monotonic_ms) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(batch_id) DO UPDATE SET status='running', lease_id=excluded.lease_id, "
                    "lease_version=excluded.lease_version, lease_status=excluded.lease_status, "
                    "wait_started_ms=excluded.wait_started_ms, wait_ended_ms=excluded.wait_ended_ms, "
                    "wait_ms=excluded.wait_ms, updated_monotonic_ms=excluded.updated_monotonic_ms",
                    (
                        batch.batch_id,
                        batch.worker_id,
                        batch.batch_index,
                        json.dumps(list(batch.arrival_seqs)),
                        json.dumps(list(batch.request_ids)),
                        "running",
                        membership,
                        lease.lease_id,
                        lease.version,
                        lease.status,
                        wait_started,
                        wait_ended,
                        wait_ms,
                        now,
                    ),
                )
                self.conn.execute(
                    "INSERT OR REPLACE INTO leases(lease_id,batch_id,worker_id,version,granted_monotonic_ms,"
                    "expires_monotonic_ms,ttl_ms,status,release_reason,released_monotonic_ms) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (
                        lease.lease_id,
                        lease.batch_id,
                        lease.worker_id,
                        lease.version,
                        lease.granted_monotonic_ms,
                        lease.expires_monotonic_ms,
                        lease.ttl_ms,
                        lease.status,
                        lease.release_reason,
                        lease.released_monotonic_ms,
                    ),
                )
            self.refresh_digest()

    def commit_batch_result(
        self,
        batch: Batch,
        lease: Any,
        result: dict[str, Any],
        *,
        source_format: str = "r3",
        normalized_receipt: dict[str, Any] | None = None,
        propagation_rows: list[dict[str, Any]] | None = None,
    ) -> int:
        now = monotonic_ms()
        with self.lock:
            previous = self.conn.execute(
                "SELECT frozen_output_digest, COUNT(*) AS receipt_count FROM batches b "
                "LEFT JOIN receipts r ON r.batch_id=b.batch_id WHERE b.batch_id=?",
                (batch.batch_id,),
            ).fetchone()
            frozen_digest = previous["frozen_output_digest"] if previous and previous["frozen_output_digest"] else result["output_digest"]
            revision = int(previous["receipt_count"] or 0) + 1 if previous else 1
            worker_snapshot = result.get("worker_snapshot_json")
            if worker_snapshot is None:
                worker_snapshot = json.dumps(result.get("worker_snapshot", {}), ensure_ascii=False, sort_keys=True)
            wait_started = getattr(lease, "wait_started_monotonic_ms", None)
            wait_ended = getattr(lease, "wait_ended_monotonic_ms", None)
            wait_ms = getattr(lease, "wait_ms", None)
            if wait_ms is None and wait_started is not None and wait_ended is not None:
                wait_ms = wait_ended - wait_started

            with self.conn:
                self.conn.execute(
                    "INSERT INTO batches(batch_id,worker_id,batch_index,arrival_seqs_json,request_ids_json,status,"
                    "membership_digest,param_version,param_digest,worker_snapshot_json,output_digest,"
                    "frozen_output_digest,result_json,propagation_json,lease_id,lease_version,lease_status,"
                    "wait_started_ms,wait_ended_ms,wait_ms,updated_monotonic_ms) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(batch_id) DO UPDATE SET status='committed', param_version=excluded.param_version, "
                    "param_digest=excluded.param_digest, worker_snapshot_json=excluded.worker_snapshot_json, "
                    "output_digest=excluded.output_digest, result_json=excluded.result_json, "
                    "frozen_output_digest=COALESCE(batches.frozen_output_digest, excluded.frozen_output_digest), "
                    "propagation_json=excluded.propagation_json, lease_id=excluded.lease_id, "
                    "lease_version=excluded.lease_version, lease_status=excluded.lease_status, "
                    "wait_started_ms=excluded.wait_started_ms, wait_ended_ms=excluded.wait_ended_ms, "
                    "wait_ms=excluded.wait_ms, updated_monotonic_ms=excluded.updated_monotonic_ms",
                    (
                        batch.batch_id,
                        batch.worker_id,
                        batch.batch_index,
                        json.dumps(list(batch.arrival_seqs)),
                        json.dumps(list(batch.request_ids)),
                        "committed",
                        canonical_digest(batch.material()),
                        result.get("param_version"),
                        result.get("param_digest"),
                        worker_snapshot,
                        result["output_digest"],
                        frozen_digest,
                        json.dumps(result, ensure_ascii=False, sort_keys=True),
                        json.dumps(propagation_rows or [], ensure_ascii=False, sort_keys=True),
                        lease.lease_id,
                        lease.version,
                        lease.release_reason or lease.status,
                        wait_started,
                        wait_ended,
                        wait_ms,
                        now,
                    ),
                )
                for rank, request_id in enumerate(batch.request_ids):
                    arrival_seq = batch.arrival_seqs[rank]
                    assignment = {
                        "batch_id": batch.batch_id,
                        "worker_id": batch.worker_id,
                        "request_id": request_id,
                        "arrival_seq": arrival_seq,
                        "rank": rank,
                    }
                    self.conn.execute(
                        "INSERT INTO assignments(request_id,batch_id,worker_id,arrival_seq,rank,assignment_json) "
                        "VALUES(?,?,?,?,?,?) ON CONFLICT(request_id) DO UPDATE SET batch_id=excluded.batch_id, "
                        "worker_id=excluded.worker_id, arrival_seq=excluded.arrival_seq, rank=excluded.rank, "
                        "assignment_json=excluded.assignment_json",
                        (request_id, batch.batch_id, batch.worker_id, arrival_seq, rank, json.dumps(assignment, sort_keys=True)),
                    )
                self.conn.execute(
                    "INSERT INTO receipts(batch_id,revision,source_format,normalized_json,output_digest,frozen,monotonic_ms) "
                    "VALUES(?,?,?,?,?,1,?)",
                    (
                        batch.batch_id,
                        revision,
                        source_format,
                        json.dumps(normalized_receipt or result, ensure_ascii=False, sort_keys=True),
                        result["output_digest"],
                        now,
                    ),
                )
                self.conn.execute(
                    "UPDATE leases SET version=?, status=?, release_reason=?, released_monotonic_ms=? WHERE lease_id=?",
                    (lease.version, lease.status, lease.release_reason or "completed", lease.released_monotonic_ms or now, lease.lease_id),
                )
                for row in propagation_rows or []:
                    self.conn.execute(
                        "INSERT OR IGNORE INTO invalidations(epoch,worker_id,batch_id,request_id,node_type,"
                        "dependency_level,cause,basis_json,monotonic_ms) VALUES(?,?,?,?,?,?,?,?,?)",
                        (
                            row.get("epoch", "commit"),
                            row.get("worker_id"),
                            row.get("batch_id"),
                            row.get("request_id"),
                            row.get("node_type", "batch"),
                            row.get("dependency_level", "direct"),
                            row.get("cause", "committed"),
                            json.dumps(row.get("basis", {}), ensure_ascii=False, sort_keys=True),
                            now,
                        ),
                    )
                cursor = {
                    "last_committed_batch_id": batch.batch_id,
                    "worker_id": batch.worker_id,
                    "batch_index": batch.batch_index,
                    "revision": revision,
                    "monotonic_ms": now,
                }
                self.conn.execute(
                    "INSERT INTO progress(key,value_json,updated_monotonic_ms) VALUES('commit_cursor',?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_monotonic_ms=excluded.updated_monotonic_ms",
                    (json.dumps(cursor, sort_keys=True), now),
                )
            self.refresh_digest()
            return revision

    def save_invalidation_rows(self, epoch: str, rows: list[dict[str, Any]]) -> None:
        now = monotonic_ms()
        with self.lock:
            with self.conn:
                for row in rows:
                    self.conn.execute(
                        "INSERT OR IGNORE INTO invalidations(epoch,worker_id,batch_id,request_id,node_type,"
                        "dependency_level,cause,basis_json,monotonic_ms) VALUES(?,?,?,?,?,?,?,?,?)",
                        (
                            epoch,
                            row.get("worker_id"),
                            row.get("batch_id"),
                            row.get("request_id"),
                            row.get("node_type", "batch"),
                            row.get("dependency_level", "direct"),
                            row.get("cause", ""),
                            json.dumps(row.get("basis", {}), ensure_ascii=False, sort_keys=True),
                            now,
                        ),
                    )
                self.conn.execute(
                    "INSERT INTO progress(key,value_json,updated_monotonic_ms) VALUES('invalidation_cursor',?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_monotonic_ms=excluded.updated_monotonic_ms",
                    (json.dumps({"epoch": epoch, "row_count": len(rows)}, sort_keys=True), now),
                )
            self.refresh_digest()

    def mark_interrupted_batches_unknown(self, reason: str = "forced_worker_termination") -> list[str]:
        now = monotonic_ms()
        with self.lock:
            rows = self.conn.execute("SELECT batch_id FROM batches WHERE status='running' ORDER BY batch_id").fetchall()
            batch_ids = [row["batch_id"] for row in rows]
            with self.conn:
                self.conn.execute(
                    "UPDATE batches SET status='unknown', updated_monotonic_ms=? WHERE status='running'",
                    (now,),
                )
                self.conn.execute(
                    "UPDATE leases SET status='unknown', release_reason=?, released_monotonic_ms=? WHERE status='active'",
                    (reason, now),
                )
                self.conn.execute(
                    "INSERT INTO events(monotonic_ms,level,code,message,details_json) VALUES(?,?,?,?,?)",
                    (now, "WARNING", "E_RECOVERY_UNKNOWN", "未提交批次标记为 unknown，必须重跑；不生成强制终止退出报告", json.dumps({"batch_ids": batch_ids, "reason": reason}, sort_keys=True)),
                )
                self.conn.execute(
                    "INSERT INTO progress(key,value_json,updated_monotonic_ms) VALUES('recovery_cursor',?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_monotonic_ms=excluded.updated_monotonic_ms",
                    (json.dumps({"unknown_batch_ids": batch_ids, "reason": reason}, sort_keys=True), now),
                )
            self.refresh_digest()
            return batch_ids

    def unknown_batches(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM batches WHERE status='unknown' ORDER BY worker_id,batch_index").fetchall()
        return [_row_to_dict(row) for row in rows]

    def list_batches(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM batches ORDER BY worker_id,batch_index,batch_id").fetchall()
        return [_row_to_dict(row) for row in rows]

    def list_invalidations(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM invalidations ORDER BY worker_id,batch_id,request_id,id").fetchall()
        return [_row_to_dict(row) for row in rows]

    def snapshot_rows(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM snapshots ORDER BY created_monotonic_ms,snapshot_id").fetchall()
        return [_row_to_dict(row) for row in rows]

    def export_snapshot(self, path: str | Path, snapshot_id: str | None = None) -> dict[str, Any]:
        self.verify_digest()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        snapshot_id = snapshot_id or f"snapshot_{monotonic_ms()}_{target.stem}"
        now_mono = monotonic_ms()
        created_wall = datetime.now(timezone.utc).isoformat()
        with self.lock:
            with self.conn:
                self.conn.execute(
                    "INSERT OR REPLACE INTO snapshots(snapshot_id,format_version,ledger_version,content_digest,path,"
                    "created_wall,created_monotonic_ms) VALUES(?,?,?,?,?,?,?)",
                    (
                        snapshot_id,
                        SNAPSHOT_FORMAT_VERSION,
                        LEDGER_FORMAT_VERSION,
                        "",
                        str(target),
                        created_wall,
                        now_mono,
                    ),
                )
            self.refresh_digest()
            digest = self.stored_digest()
            self.conn.execute("UPDATE snapshots SET content_digest=? WHERE snapshot_id=?", (digest, snapshot_id))
            self.conn.commit()
            payload = {
                "snapshot_format": SNAPSHOT_FORMAT_VERSION,
                "snapshot_id": snapshot_id,
                "ledger_format_version": LEDGER_FORMAT_VERSION,
                "content_digest": digest,
                "exported_wall": created_wall,
                "tables": table_digest_payload(self.conn),
            }
            tmp = target.with_suffix(target.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
            tmp.replace(target)
            return payload

    @staticmethod
    def verify_snapshot_file(path: str | Path) -> dict[str, Any]:
        source = Path(path)
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SnapshotError("快照无法读取或不是合法 JSON", path=str(source)) from exc
        if payload.get("snapshot_format") != SNAPSHOT_FORMAT_VERSION:
            raise SnapshotError("无法识别的会话快照版本", path=str(source), snapshot_format=payload.get("snapshot_format"))
        if not isinstance(payload.get("ledger_format_version"), int):
            raise SnapshotError("快照缺少整数账本格式版本", path=str(source))
        if payload["ledger_format_version"] != LEDGER_FORMAT_VERSION:
            raise SnapshotError(
                "账本格式版本不受本程序支持",
                path=str(source),
                ledger_format_version=payload["ledger_format_version"],
                supported_version=LEDGER_FORMAT_VERSION,
            )
        actual = digest_payload(payload.get("tables", {}))
        if actual != payload.get("content_digest"):
            raise SnapshotError(
                "会话快照内容摘要不符，拒绝整体替换",
                path=str(source),
                expected_digest=payload.get("content_digest"),
                actual_digest=actual,
            )
        return payload

    @classmethod
    def restore_snapshot_to_path(cls, snapshot_path: str | Path, db_path: str | Path) -> "Ledger":
        payload = cls.verify_snapshot_file(snapshot_path)
        target = Path(db_path)
        tmp_path = target.with_name(target.name + ".restore-tmp")
        if tmp_path.exists():
            tmp_path.unlink()
        restored = cls(tmp_path, reset=True)
        try:
            tables = payload["tables"]
            restored.conn.execute("PRAGMA foreign_keys=OFF")
            with restored.conn:
                for table, rows in tables.items():
                    if table.startswith("sqlite_"):
                        continue
                    if not rows:
                        continue
                    columns = list(rows[0].keys())
                    quoted = ",".join(f'"{column}"' for column in columns)
                    placeholders = ",".join("?" for _ in columns)
                    sql = f'INSERT OR REPLACE INTO "{table}" ({quoted}) VALUES ({placeholders})'
                    restored.conn.executemany(sql, [[row.get(column) for column in columns] for row in rows])
                restored.conn.execute(
                    "INSERT INTO meta(key,value) VALUES('ledger_format_version',?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (str(payload["ledger_format_version"]),),
                )
            restored.refresh_digest()
            restored.verify_digest()
            restored.close()
            target.parent.mkdir(parents=True, exist_ok=True)
            Path(tmp_path).replace(target)
        except Exception:
            restored.close()
            if tmp_path.exists():
                tmp_path.unlink()
            raise
        return cls(target)

    def restore_from_snapshot(self, snapshot_path: str | Path) -> None:
        db_path = self.path
        self.close()
        restored = self.restore_snapshot_to_path(snapshot_path, db_path)
        self.conn = restored.conn
        self.lock = restored.lock
        self.verify_digest()

    def ui_state(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version(),
            "content_digest": self.stored_digest(),
            "batches": self.list_batches(),
            "unknown_batches": self.unknown_batches(),
            "invalidations": self.list_invalidations(),
            "snapshots": self.snapshot_rows(),
            "progress": [dict(row) for row in self.conn.execute("SELECT * FROM progress ORDER BY key")],
        }
