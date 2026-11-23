"""SQLite ledger with transactional layer commits and crash recovery views."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .errors import DiagnosticError, replay_error
from .inference import LayerBatchResult
from .util import canonical_json_bytes, digest_value, wall_time_ms


LEDGER_FORMAT_VERSION = 1


class Ledger:
    """A single-writer repository safe for scheduler worker callbacks."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.path,
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._initialize()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _initialize(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS batches (
                    plan_version INTEGER NOT NULL,
                    batch_id TEXT NOT NULL,
                    sample_ids_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    queue_start_ms INTEGER,
                    worker_start_ms INTEGER,
                    end_ms INTEGER,
                    worker_name TEXT,
                    error_json TEXT,
                    created_ms INTEGER NOT NULL,
                    updated_ms INTEGER NOT NULL,
                    PRIMARY KEY (plan_version, batch_id)
                );
                CREATE TABLE IF NOT EXISTS leases (
                    lease_id TEXT PRIMARY KEY,
                    plan_version INTEGER NOT NULL,
                    batch_id TEXT NOT NULL,
                    layer_id TEXT NOT NULL,
                    block_start INTEGER NOT NULL,
                    block_count INTEGER NOT NULL,
                    granted_ms INTEGER NOT NULL,
                    expires_ms INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    released_ms INTEGER,
                    reason TEXT,
                    updated_ms INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS layer_results (
                    plan_version INTEGER NOT NULL,
                    batch_id TEXT NOT NULL,
                    layer_id TEXT NOT NULL,
                    layer_version INTEGER NOT NULL,
                    sample_ids_json TEXT NOT NULL,
                    float_digest TEXT NOT NULL,
                    quant_digest TEXT NOT NULL,
                    per_sample_json TEXT NOT NULL,
                    errors_json TEXT NOT NULL,
                    error_bound_json TEXT NOT NULL,
                    saturation_count INTEGER NOT NULL,
                    occupancy_bytes INTEGER NOT NULL,
                    occupancy_blocks INTEGER NOT NULL,
                    lease_id TEXT,
                    block_start INTEGER,
                    block_count INTEGER,
                    input_digest TEXT NOT NULL,
                    valid INTEGER NOT NULL DEFAULT 1,
                    committed_ms INTEGER NOT NULL,
                    PRIMARY KEY (plan_version, batch_id, layer_id)
                );
                CREATE TABLE IF NOT EXISTS progress (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    plan_version INTEGER NOT NULL,
                    batch_id TEXT NOT NULL,
                    layer_id TEXT NOT NULL,
                    committed_ms INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS invalidations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plan_version INTEGER NOT NULL,
                    changed_json TEXT NOT NULL,
                    affected_json TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    created_ms INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS attributions (
                    plan_version INTEGER NOT NULL,
                    layer_id TEXT NOT NULL,
                    attribution_json TEXT NOT NULL,
                    PRIMARY KEY (plan_version, layer_id)
                );
                CREATE TABLE IF NOT EXISTS lease_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event TEXT NOT NULL,
                    plan_version INTEGER NOT NULL,
                    batch_id TEXT NOT NULL,
                    layer_id TEXT NOT NULL,
                    lease_id TEXT NOT NULL,
                    block_start INTEGER NOT NULL,
                    block_count INTEGER NOT NULL,
                    at_ms INTEGER NOT NULL,
                    reason TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_results_valid
                    ON layer_results(plan_version, valid, batch_id, layer_id);
                CREATE INDEX IF NOT EXISTS idx_leases_batch
                    ON leases(plan_version, batch_id, layer_id);
                """
            )
            found = self._conn.execute(
                "SELECT value FROM meta WHERE key='ledger_format_version'"
            ).fetchone()
            if found is None:
                self._conn.execute("BEGIN IMMEDIATE")
                try:
                    self._conn.execute(
                        "INSERT INTO meta(key,value) VALUES('ledger_format_version',?)",
                        (str(LEDGER_FORMAT_VERSION),),
                    )
                    self._digest = self._compute_content_digest_locked()
                    self._set_meta_locked("content_digest", self._digest)
                    self._conn.commit()
                except BaseException:
                    self._conn.rollback()
                    raise
            else:
                if int(found["value"]) != LEDGER_FORMAT_VERSION:
                    raise replay_error(
                        "unsupported ledger format version",
                        expected=LEDGER_FORMAT_VERSION,
                        actual=int(found["value"]),
                    )

    def _set_meta_locked(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def _meta_locked(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def _rows_locked(self, table: str, order_by: str) -> list[list[Any]]:
        rows = self._conn.execute(f"SELECT * FROM {table} ORDER BY {order_by}").fetchall()
        return [list(row) for row in rows]

    def _compute_content_digest_locked(self) -> str:
        payload = {
            "format_version": LEDGER_FORMAT_VERSION,
            "batches": self._rows_locked("batches", "plan_version,batch_id"),
            "leases": self._rows_locked("leases", "lease_id"),
            "layer_results": self._rows_locked(
                "layer_results", "plan_version,batch_id,layer_id"
            ),
            "progress": self._rows_locked("progress", "sequence"),
            "invalidations": self._rows_locked("invalidations", "id"),
            "attributions": self._rows_locked(
                "attributions", "plan_version,layer_id"
            ),
            "lease_events": self._rows_locked("lease_events", "id"),
        }
        return digest_value(payload)

    def _update_digest_locked(self) -> str:
        digest = self._compute_content_digest_locked()
        self._set_meta_locked("content_digest", digest)
        return digest

    def content_digest(self) -> str:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM meta WHERE key='content_digest'"
            ).fetchone()
            return "" if row is None else str(row["value"])

    def verify_integrity(self) -> dict[str, Any]:
        with self._lock:
            format_row = self._conn.execute(
                "SELECT value FROM meta WHERE key='ledger_format_version'"
            ).fetchone()
            expected = self._meta_locked("content_digest")
            actual = self._compute_content_digest_locked()
            if format_row is None or int(format_row["value"]) != LEDGER_FORMAT_VERSION:
                raise replay_error(
                    "ledger format cannot be aligned",
                    expected_format=LEDGER_FORMAT_VERSION,
                    actual_format=None if format_row is None else int(format_row["value"]),
                )
            if expected != actual:
                raise replay_error(
                    "ledger content digest mismatch",
                    field="content_digest",
                    expected=expected,
                    actual=actual,
                )
            return {
                "ok": True,
                "format_version": LEDGER_FORMAT_VERSION,
                "content_digest": actual,
            }

    def _decode_json_fields(self, row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        for key in (
            "sample_ids_json",
            "per_sample_json",
            "errors_json",
            "error_bound_json",
        ):
            if key in item and item[key] is not None:
                item[key.removesuffix("_json")] = json.loads(item.pop(key))
        return item

    def begin_batch(
        self,
        plan_version: int,
        batch_id: str,
        sample_ids: Sequence[str],
        leases: Sequence[Any],
        *,
        queue_start_ms: int,
    ) -> None:
        now = wall_time_ms()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                existing = self._conn.execute(
                    "SELECT status FROM batches WHERE plan_version=? AND batch_id=?",
                    (plan_version, batch_id),
                ).fetchone()
                if existing is not None and existing["status"] == "committed":
                    raise replay_error(
                        "batch is already committed",
                        field="batch_id",
                        batch_id=batch_id,
                    )
                self._conn.execute(
                    "INSERT INTO batches(plan_version,batch_id,sample_ids_json,status,queue_start_ms,created_ms,updated_ms) "
                    "VALUES(?,?,?,'running',?,?,?) "
                    "ON CONFLICT(plan_version,batch_id) DO UPDATE SET "
                    "sample_ids_json=excluded.sample_ids_json,status='running',"
                    "queue_start_ms=excluded.queue_start_ms,updated_ms=excluded.updated_ms",
                    (
                        plan_version,
                        batch_id,
                        json.dumps(list(sample_ids), separators=(",", ":")),
                        queue_start_ms,
                        now,
                        now,
                    ),
                )
                self._record_leases_locked(leases, plan_version=plan_version, now=now)
                self._update_digest_locked()
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise

    def _record_leases_locked(
        self, leases: Iterable[Any], *, plan_version: int, now: int
    ) -> None:
        for lease in leases:
            start = lease.block_start
            count = lease.block_count
            self._conn.execute(
                "INSERT INTO leases(lease_id,plan_version,batch_id,layer_id,block_start,block_count,"
                "granted_ms,expires_ms,status,released_ms,reason,updated_ms) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(lease_id) DO UPDATE SET status=excluded.status,"
                "released_ms=excluded.released_ms,reason=excluded.reason,updated_ms=excluded.updated_ms",
                (
                    lease.lease_id,
                    plan_version,
                    lease.batch_id,
                    lease.layer_id,
                    start,
                    count,
                    lease.granted_ms,
                    lease.expires_ms,
                    lease.status,
                    lease.released_ms,
                    lease.reason,
                    now,
                ),
            )
            self._conn.execute(
                "INSERT INTO lease_events(event,plan_version,batch_id,layer_id,lease_id,"
                "block_start,block_count,at_ms,reason) VALUES('grant',?,?,?,?,?,?,?,NULL)",
                (
                    plan_version,
                    lease.batch_id,
                    lease.layer_id,
                    lease.lease_id,
                    start,
                    count,
                    lease.granted_ms,
                ),
            )

    def mark_worker_started(
        self, plan_version: int, batch_id: str, *, worker_start_ms: int, worker_name: str
    ) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "UPDATE batches SET worker_start_ms=?,worker_name=?,updated_ms=? "
                    "WHERE plan_version=? AND batch_id=?",
                    (worker_start_ms, worker_name, wall_time_ms(), plan_version, batch_id),
                )
                self._update_digest_locked()
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise

    def commit_layer_result(
        self,
        result: LayerBatchResult,
        *,
        committed_ms: int | None = None,
    ) -> str:
        now = wall_time_ms() if committed_ms is None else committed_ms
        payload = result.ledger_payload()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "INSERT INTO layer_results(plan_version,batch_id,layer_id,layer_version,"
                    "sample_ids_json,float_digest,quant_digest,per_sample_json,errors_json,"
                    "error_bound_json,saturation_count,occupancy_bytes,occupancy_blocks,"
                    "lease_id,block_start,block_count,input_digest,valid,committed_ms) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1,?) "
                    "ON CONFLICT(plan_version,batch_id,layer_id) DO UPDATE SET "
                    "layer_version=excluded.layer_version,sample_ids_json=excluded.sample_ids_json,"
                    "float_digest=excluded.float_digest,quant_digest=excluded.quant_digest,"
                    "per_sample_json=excluded.per_sample_json,errors_json=excluded.errors_json,"
                    "error_bound_json=excluded.error_bound_json,saturation_count=excluded.saturation_count,"
                    "occupancy_bytes=excluded.occupancy_bytes,occupancy_blocks=excluded.occupancy_blocks,"
                    "lease_id=excluded.lease_id,block_start=excluded.block_start,block_count=excluded.block_count,"
                    "input_digest=excluded.input_digest,valid=1,committed_ms=excluded.committed_ms",
                    (
                        result.plan_version,
                        result.batch_id,
                        result.layer_id,
                        result.layer_version,
                        json.dumps(list(result.sample_ids), separators=(",", ":")),
                        result.float_digest,
                        result.quant_digest,
                        json.dumps(
                            {key: result.per_sample[key] for key in sorted(result.per_sample)},
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        json.dumps(result.errors, sort_keys=True, separators=(",", ":")),
                        json.dumps(result.error_bound, sort_keys=True, separators=(",", ":")),
                        result.saturation_count,
                        result.occupancy_bytes,
                        result.occupancy_blocks,
                        result.lease_id,
                        result.block_start,
                        result.block_count,
                        result.input_digest,
                        now,
                    ),
                )
                self._conn.execute(
                    "INSERT INTO progress(plan_version,batch_id,layer_id,committed_ms) "
                    "VALUES(?,?,?,?)",
                    (result.plan_version, result.batch_id, result.layer_id, now),
                )
                digest = self._update_digest_locked()
                self._conn.commit()
                return digest
            except BaseException:
                self._conn.rollback()
                raise

    def finish_batch(
        self,
        plan_version: int,
        batch_id: str,
        leases: Sequence[Any],
        *,
        end_ms: int,
        error: str | None = None,
    ) -> str:
        now = wall_time_ms()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                status = "failed" if error is not None else "committed"
                self._conn.execute(
                    "UPDATE batches SET status=?,end_ms=?,error_json=?,updated_ms=? "
                    "WHERE plan_version=? AND batch_id=?",
                    (
                        status,
                        end_ms,
                        None if error is None else json.dumps({"message": error}, sort_keys=True),
                        now,
                        plan_version,
                        batch_id,
                    ),
                )
                for lease in leases:
                    self._conn.execute(
                        "UPDATE leases SET status=?,released_ms=?,reason=?,updated_ms=? "
                        "WHERE lease_id=?",
                        (
                            lease.status,
                            lease.released_ms,
                            lease.reason,
                            now,
                            lease.lease_id,
                        ),
                    )
                    if lease.released_ms is not None and lease.reason is not None:
                        event = "expire" if lease.status == "expired" else "release"
                        if lease.status == "revoked":
                            event = "revoke"
                        self._conn.execute(
                            "INSERT INTO lease_events(event,plan_version,batch_id,layer_id,lease_id,"
                            "block_start,block_count,at_ms,reason) VALUES(?,?,?,?,?,?,?,?,?)",
                            (
                                event,
                                plan_version,
                                lease.batch_id,
                                lease.layer_id,
                                lease.lease_id,
                                lease.block_start,
                                lease.block_count,
                                lease.released_ms,
                                lease.reason,
                            ),
                        )
                digest = self._update_digest_locked()
                self._conn.commit()
                return digest
            except BaseException:
                self._conn.rollback()
                raise

    def record_invalidation(self, plan_version: int, report: Mapping[str, Any]) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "INSERT INTO invalidations(plan_version,changed_json,affected_json,evidence_json,created_ms) "
                    "VALUES(?,?,?,?,?)",
                    (
                        plan_version,
                        json.dumps(report.get("changed", []), sort_keys=True, separators=(",", ":")),
                        json.dumps(report.get("invalidated", []), sort_keys=True, separators=(",", ":")),
                        json.dumps(report, sort_keys=True, separators=(",", ":")),
                        wall_time_ms(),
                    ),
                )
                self._update_digest_locked()
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise

    def record_attribution(self, plan_version: int, attribution: Mapping[str, Any]) -> None:
        payload = json.dumps(attribution, sort_keys=True, separators=(",", ":"))
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                for layer in attribution.get("layers", []):
                    self._conn.execute(
                        "INSERT INTO attributions(plan_version,layer_id,attribution_json) VALUES(?,?,?) "
                        "ON CONFLICT(plan_version,layer_id) DO UPDATE SET attribution_json=excluded.attribution_json",
                        (plan_version, layer["layer_id"], payload),
                    )
                self._update_digest_locked()
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise

    def record_held_leases(self, plan_version: int, leases: Iterable[Any]) -> None:
        now = wall_time_ms()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._record_leases_locked(leases, plan_version=plan_version, now=now)
                self._update_digest_locked()
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise

    def record_version_replacement(
        self,
        *,
        old_plan_version: int,
        new_plan_version: int,
        revoked: Sequence[Mapping[str, Any]],
        new_leases: Sequence[Mapping[str, Any]],
        basis: str,
    ) -> None:
        """Persist E_REVOKE events and a whole replacement lease view atomically."""

        now = wall_time_ms()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                for record in revoked:
                    block_range = record.get("released_block_range") or record.get("block_range")
                    if block_range is None:
                        continue
                    self._conn.execute(
                        "UPDATE leases SET status='revoked',released_ms=?,reason=?,updated_ms=? "
                        "WHERE lease_id=?",
                        (
                            record.get("released_ms", now),
                            "E_REVOKE:" + basis,
                            now,
                            record["lease_id"],
                        ),
                    )
                    self._conn.execute(
                        "INSERT INTO lease_events(event,plan_version,batch_id,layer_id,lease_id,"
                        "block_start,block_count,at_ms,reason) VALUES('revoke',?,?,?,?,?,?,?,?)",
                        (
                            old_plan_version,
                            record["batch_id"],
                            record["layer_id"],
                            record["lease_id"],
                            int(block_range[0]),
                            int(block_range[1]) - int(block_range[0]),
                            int(record.get("released_ms") or now),
                            "E_REVOKE:" + basis,
                        ),
                    )
                for record in new_leases:
                    block_range = record["block_range"]
                    self._conn.execute(
                        "INSERT INTO leases(lease_id,plan_version,batch_id,layer_id,block_start,block_count,"
                        "granted_ms,expires_ms,status,released_ms,reason,updated_ms) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?) "
                        "ON CONFLICT(lease_id) DO UPDATE SET status=excluded.status,"
                        "released_ms=excluded.released_ms,reason=excluded.reason,updated_ms=excluded.updated_ms",
                        (
                            record["lease_id"],
                            new_plan_version,
                            record["batch_id"],
                            record["layer_id"],
                            int(block_range[0]),
                            int(block_range[1]) - int(block_range[0]),
                            int(record["granted_ms"]),
                            int(record["expires_ms"]),
                            str(record.get("status", "assigned")),
                            record.get("released_ms"),
                            record.get("reason"),
                            now,
                        ),
                    )
                    self._conn.execute(
                        "INSERT INTO lease_events(event,plan_version,batch_id,layer_id,lease_id,"
                        "block_start,block_count,at_ms,reason) VALUES('assign',?,?,?,?,?,?,?,?)",
                        (
                            new_plan_version,
                            record["batch_id"],
                            record["layer_id"],
                            record["lease_id"],
                            int(block_range[0]),
                            int(block_range[1]) - int(block_range[0]),
                            int(record["granted_ms"]),
                            "version_ledger_replacement",
                        ),
                    )
                self._update_digest_locked()
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise

    def mark_unknown_uncommitted(self) -> dict[str, Any]:
        """Mark queued/running batches and their leases unknown after a crash."""

        now = wall_time_ms()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                running = self._conn.execute(
                    "SELECT plan_version,batch_id FROM batches WHERE status IN ('queued','running','failed') "
                    "ORDER BY plan_version,batch_id"
                ).fetchall()
                unknown_batches: list[dict[str, Any]] = []
                unknown_leases: list[dict[str, Any]] = []
                for row in running:
                    plan_version = int(row["plan_version"])
                    batch_id = str(row["batch_id"])
                    self._conn.execute(
                        "UPDATE batches SET status='unknown',updated_ms=? WHERE plan_version=? AND batch_id=?",
                        (now, plan_version, batch_id),
                    )
                    self._conn.execute(
                        "UPDATE layer_results SET valid=0 WHERE plan_version=? AND batch_id=?",
                        (plan_version, batch_id),
                    )
                    leases = self._conn.execute(
                        "SELECT * FROM leases WHERE plan_version=? AND batch_id=? "
                        "AND status NOT IN ('released','expired','revoked','unknown') "
                        "ORDER BY lease_id",
                        (plan_version, batch_id),
                    ).fetchall()
                    for lease_row in leases:
                        self._conn.execute(
                            "UPDATE leases SET status='unknown',reason=?,updated_ms=? WHERE lease_id=?",
                            ("E_REPLAY_UNCOMMITTED_BATCH", now, lease_row["lease_id"]),
                        )
                        self._conn.execute(
                            "INSERT INTO lease_events(event,plan_version,batch_id,layer_id,lease_id,"
                            "block_start,block_count,at_ms,reason) VALUES('unknown',?,?,?,?,?,?,?,?)",
                            (
                                plan_version,
                                batch_id,
                                lease_row["layer_id"],
                                lease_row["lease_id"],
                                lease_row["block_start"],
                                lease_row["block_count"],
                                now,
                                "E_REPLAY_UNCOMMITTED_BATCH",
                            ),
                        )
                        unknown_leases.append(
                            {
                                "lease_id": lease_row["lease_id"],
                                "block_range": [
                                    lease_row["block_start"],
                                    lease_row["block_start"] + lease_row["block_count"],
                                ],
                                "status": "unknown",
                                "reason": "E_REPLAY_UNCOMMITTED_BATCH",
                            }
                        )
                    unknown_batches.append(
                        {
                            "plan_version": plan_version,
                            "batch_id": batch_id,
                            "status": "unknown",
                            "requires_rerun": True,
                        }
                    )
                self._update_digest_locked()
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
            cursor = self._cursor_locked()
            committed_batches = [
                dict(row)
                for row in self._conn.execute(
                    "SELECT plan_version,batch_id,sample_ids_json,status,end_ms FROM batches "
                    "WHERE status='committed' ORDER BY plan_version,batch_id"
                ).fetchall()
            ]
            for batch in committed_batches:
                batch["sample_ids"] = json.loads(batch.pop("sample_ids_json"))
            return {
                "unknown_batches": unknown_batches,
                "unknown_leases": unknown_leases,
                "committed_batches": committed_batches,
                "cursor": cursor,
                "exit_report_available": False,
            }

    def _cursor_locked(self) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT sequence,plan_version,batch_id,layer_id,committed_ms FROM progress "
            "ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        return None if row is None else dict(row)

    def recovery_view(self) -> dict[str, Any]:
        with self._lock:
            committed_batches = [
                dict(row)
                for row in self._conn.execute(
                    "SELECT plan_version,batch_id,sample_ids_json,status,end_ms FROM batches "
                    "WHERE status='committed' ORDER BY plan_version,batch_id"
                ).fetchall()
            ]
            for batch in committed_batches:
                batch["sample_ids"] = json.loads(batch.pop("sample_ids_json"))
            return {
                "committed_batches": committed_batches,
                "cursor": self._cursor_locked(),
                "completed_layer_count": int(
                    self._conn.execute(
                        "SELECT COUNT(*) AS n FROM layer_results WHERE valid=1"
                    ).fetchone()["n"]
                ),
                "content_digest": self.content_digest(),
            }

    def load_committed_results(self, plan_version: int) -> dict[tuple[str, str], dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM layer_results WHERE plan_version=? AND valid=1 "
                "ORDER BY batch_id,layer_id",
                (plan_version,),
            ).fetchall()
            result: dict[tuple[str, str], dict[str, Any]] = {}
            for row in rows:
                item = self._decode_json_fields(row)
                result[(str(item["batch_id"]), str(item["layer_id"]))] = item
            return result

    def load_invalidations(self, plan_version: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if plan_version is None:
                rows = self._conn.execute(
                    "SELECT * FROM invalidations ORDER BY id"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM invalidations WHERE plan_version=? ORDER BY id",
                    (plan_version,),
                ).fetchall()
            return [
                {
                    "id": row["id"],
                    "plan_version": row["plan_version"],
                    "changed": json.loads(row["changed_json"]),
                    "invalidated": json.loads(row["affected_json"]),
                    "evidence": json.loads(row["evidence_json"]),
                    "created_ms": row["created_ms"],
                }
                for row in rows
            ]

    def load_leases(self, plan_version: int | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if plan_version is None:
                rows = self._conn.execute("SELECT * FROM leases ORDER BY lease_id").fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM leases WHERE plan_version=? ORDER BY lease_id",
                    (plan_version,),
                ).fetchall()
            return [dict(row) for row in rows]

    def load_attribution(self, plan_version: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT attribution_json FROM attributions WHERE plan_version=? "
                "ORDER BY layer_id LIMIT 1",
                (plan_version,),
            ).fetchone()
            return None if row is None else json.loads(row["attribution_json"])

    def table_counts(self) -> dict[str, int]:
        tables = (
            "batches",
            "leases",
            "layer_results",
            "progress",
            "invalidations",
            "attributions",
            "lease_events",
        )
        with self._lock:
            return {
                table: int(
                    self._conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
                )
                for table in tables
            }
