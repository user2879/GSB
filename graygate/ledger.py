from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .errors import GatewayError
from .util import (
    CLOCK_BASIS,
    FORMAT_VERSION,
    TRUTH_SOURCE,
    canonical_json,
    content_digest,
    monotonic_ms,
    sha256_text,
    utc_epoch_ms,
)


GENESIS_DIGEST = "sha256:" + ("0" * 64)


@dataclass(frozen=True)
class HitRecord:
    ledger_version: int
    request_seq: int
    rule_id: str
    stage_id: str
    stage_weight_bp: int
    hash_bucket: int
    side: str
    backend_version: str
    batch_seq: int
    output_digest: str
    elapsed_us: int
    client_key: str
    feature: str
    sample_json: str
    format_version: int
    content_digest: str
    created_at_ms: int

    @property
    def sample(self) -> list[float]:
        value = json.loads(self.sample_json)
        return [float(item) for item in value]

    def canonical_without_digest(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "ledger_version": self.ledger_version,
            "request_seq": self.request_seq,
            "rule_id": self.rule_id,
            "stage_id": self.stage_id,
            "stage_weight_bp": self.stage_weight_bp,
            "hash_bucket": self.hash_bucket,
            "side": self.side,
            "backend_version": self.backend_version,
            "batch_seq": self.batch_seq,
            "output_digest": self.output_digest,
            "elapsed_us": self.elapsed_us,
            "client_key": self.client_key,
            "feature": self.feature,
            "sample_json": self.sample_json,
            "created_at_ms": self.created_at_ms,
        }


class LedgerStore:
    """SQLite ledger with short write transactions and WAL read concurrency."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.RLock()
        self._sequence_lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.path), check_same_thread=False, isolation_level=None, timeout=30
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._init_schema()
        row = self._read_one(
            "SELECT COALESCE(MAX(request_seq), 0) AS value FROM hits", ()
        )
        self._next_request_seq = int(row["value"]) + 1
        self.recovered_unknown = self._mark_interrupted_rebuilds_unknown()

    def _init_schema(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS ledger_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS hits (
            ledger_version INTEGER PRIMARY KEY,
            request_seq INTEGER NOT NULL UNIQUE,
            rule_id TEXT NOT NULL,
            stage_id TEXT NOT NULL,
            stage_weight_bp INTEGER NOT NULL,
            hash_bucket INTEGER NOT NULL,
            side TEXT NOT NULL,
            backend_version TEXT NOT NULL,
            batch_seq INTEGER NOT NULL,
            output_digest TEXT NOT NULL,
            elapsed_us INTEGER NOT NULL,
            client_key TEXT NOT NULL,
            feature TEXT NOT NULL,
            sample_json TEXT NOT NULL,
            format_version INTEGER NOT NULL,
            content_digest TEXT NOT NULL,
            created_at_ms INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS hits_rule_stage_idx
            ON hits(rule_id, stage_id, ledger_version);
        CREATE TABLE IF NOT EXISTS rebuilds (
            rebuild_id TEXT PRIMARY KEY,
            source_ledger_version INTEGER NOT NULL,
            observed_ledger_version INTEGER,
            status TEXT NOT NULL,
            started_ms INTEGER NOT NULL,
            finished_ms INTEGER,
            elapsed_ms INTEGER,
            excluded_hits INTEGER,
            budget_ms INTEGER NOT NULL,
            snapshot_id TEXT,
            error_code TEXT,
            detail_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS rebuilds_status_idx ON rebuilds(status);
        CREATE TABLE IF NOT EXISTS snapshots (
            snapshot_row_id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_id TEXT NOT NULL UNIQUE,
            rebuild_id TEXT NOT NULL,
            source_ledger_version INTEGER NOT NULL,
            observed_ledger_version INTEGER NOT NULL,
            excluded_hits INTEGER NOT NULL,
            format_version INTEGER NOT NULL,
            content_digest TEXT NOT NULL,
            body_json TEXT NOT NULL,
            spec_digest TEXT NOT NULL,
            spec_json TEXT NOT NULL,
            aggregation_path TEXT NOT NULL,
            rebuild_elapsed_ms INTEGER NOT NULL,
            budget_ms INTEGER NOT NULL,
            latency_exceeded INTEGER NOT NULL,
            created_at_ms INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS snapshots_source_idx
            ON snapshots(source_ledger_version, snapshot_row_id);
        CREATE TABLE IF NOT EXISTS latency_observations (
            observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL,
            ref_id TEXT NOT NULL,
            start_ms INTEGER NOT NULL,
            end_ms INTEGER NOT NULL,
            elapsed_ms INTEGER NOT NULL,
            budget_ms INTEGER NOT NULL,
            exceeded INTEGER NOT NULL,
            over_by_ms INTEGER NOT NULL,
            error_code TEXT,
            impact_scope TEXT NOT NULL,
            clock_basis TEXT NOT NULL,
            truth_source TEXT NOT NULL,
            created_at_ms INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS latency_kind_idx
            ON latency_observations(kind, elapsed_ms);
        CREATE TABLE IF NOT EXISTS consistency_checks (
            check_id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_id TEXT NOT NULL,
            check_kind TEXT NOT NULL,
            ok INTEGER NOT NULL,
            code TEXT NOT NULL,
            rule_id TEXT NOT NULL,
            stage_id TEXT NOT NULL,
            field_name TEXT NOT NULL,
            detail_json TEXT NOT NULL,
            checked_at_ms INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS consistency_snapshot_idx
            ON consistency_checks(snapshot_id, check_kind, check_id);
        """
        with self._write_lock:
            self._conn.executescript(schema)
            self._conn.execute(
                "INSERT OR IGNORE INTO ledger_meta(key, value) VALUES('format_version', ?)",
                (str(FORMAT_VERSION),),
            )
            self._conn.execute(
                "INSERT OR IGNORE INTO ledger_meta(key, value) VALUES('ledger_digest', ?)",
                (GENESIS_DIGEST,),
            )

    def _mark_interrupted_rebuilds_unknown(self) -> list[dict[str, Any]]:
        now = monotonic_ms()
        with self._write_lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                rows = self._conn.execute(
                    "SELECT * FROM rebuilds WHERE status = 'RUNNING' ORDER BY started_ms, rebuild_id"
                ).fetchall()
                for row in rows:
                    detail = {
                        "previous_status": "RUNNING",
                        "reason": "process_terminated_before_atomic_snapshot_commit",
                        "exit_report_claimed": False,
                    }
                    self._conn.execute(
                        """
                        UPDATE rebuilds
                           SET status='UNKNOWN', finished_ms=?, error_code='E_RECOVERY',
                               detail_json=?
                         WHERE rebuild_id=?
                        """,
                        (now, canonical_json(detail), row["rebuild_id"]),
                    )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        return [dict(row) | {"status": "UNKNOWN"} for row in rows]

    def close(self) -> None:
        with self._write_lock:
            self._conn.close()

    def _read_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.path), timeout=30, isolation_level=None
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _read_one(self, sql: str, params: Iterable[Any]) -> sqlite3.Row:
        connection = self._read_connection()
        try:
            row = connection.execute(sql, tuple(params)).fetchone()
            assert row is not None
            return row
        finally:
            connection.close()

    def _read_all(self, sql: str, params: Iterable[Any]) -> list[sqlite3.Row]:
        connection = self._read_connection()
        try:
            return list(connection.execute(sql, tuple(params)).fetchall())
        finally:
            connection.close()

    def reserve_request_seq(self) -> int:
        with self._sequence_lock:
            value = self._next_request_seq
            self._next_request_seq += 1
            return value

    def current_version(self) -> int:
        row = self._read_one(
            "SELECT COALESCE(MAX(ledger_version), 0) AS value FROM hits", ()
        )
        return int(row["value"])

    def max_request_seq(self) -> int:
        row = self._read_one(
            "SELECT COALESCE(MAX(request_seq), 0) AS value FROM hits", ()
        )
        return int(row["value"])

    def ledger_digest(self) -> str:
        row = self._read_one(
            "SELECT value FROM ledger_meta WHERE key='ledger_digest'", ()
        )
        return str(row["value"])

    def append_hit(
        self,
        *,
        request_seq: int,
        rule_id: str,
        stage_id: str,
        stage_weight_bp: int,
        hash_bucket: int,
        side: str,
        backend_version: str,
        batch_seq: int,
        output_digest: str,
        elapsed_us: int,
        client_key: str,
        feature: str,
        sample: list[float],
    ) -> HitRecord:
        created_at_ms = utc_epoch_ms()
        with self._write_lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT COALESCE(MAX(ledger_version), 0) AS value FROM hits"
                ).fetchone()
                ledger_version = int(row["value"]) + 1
                previous = self._conn.execute(
                    "SELECT value FROM ledger_meta WHERE key='ledger_digest'"
                ).fetchone()
                previous_digest = str(previous["value"])
                base = {
                    "format_version": FORMAT_VERSION,
                    "ledger_version": ledger_version,
                    "request_seq": request_seq,
                    "rule_id": rule_id,
                    "stage_id": stage_id,
                    "stage_weight_bp": stage_weight_bp,
                    "hash_bucket": hash_bucket,
                    "side": side,
                    "backend_version": backend_version,
                    "batch_seq": batch_seq,
                    "output_digest": output_digest,
                    "elapsed_us": int(elapsed_us),
                    "client_key": client_key,
                    "feature": feature,
                    "sample_json": canonical_json(sample),
                    "created_at_ms": created_at_ms,
                }
                digest = sha256_text(previous_digest + "\n" + canonical_json(base))
                self._conn.execute(
                    """
                    INSERT INTO hits(
                        ledger_version, request_seq, rule_id, stage_id,
                        stage_weight_bp, hash_bucket, side, backend_version,
                        batch_seq, output_digest, elapsed_us, client_key, feature,
                        sample_json, format_version, content_digest, created_at_ms
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        ledger_version,
                        request_seq,
                        rule_id,
                        stage_id,
                        stage_weight_bp,
                        hash_bucket,
                        side,
                        backend_version,
                        batch_seq,
                        output_digest,
                        int(elapsed_us),
                        client_key,
                        feature,
                        base["sample_json"],
                        FORMAT_VERSION,
                        digest,
                        created_at_ms,
                    ),
                )
                self._conn.execute(
                    "UPDATE ledger_meta SET value=? WHERE key='ledger_digest'",
                    (digest,),
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        return HitRecord(
            ledger_version=ledger_version,
            request_seq=request_seq,
            rule_id=rule_id,
            stage_id=stage_id,
            stage_weight_bp=stage_weight_bp,
            hash_bucket=hash_bucket,
            side=side,
            backend_version=backend_version,
            batch_seq=batch_seq,
            output_digest=output_digest,
            elapsed_us=int(elapsed_us),
            client_key=client_key,
            feature=feature,
            sample_json=base["sample_json"],
            format_version=FORMAT_VERSION,
            content_digest=digest,
            created_at_ms=created_at_ms,
        )

    @staticmethod
    def _hit_from_row(row: sqlite3.Row) -> HitRecord:
        return HitRecord(
            ledger_version=int(row["ledger_version"]),
            request_seq=int(row["request_seq"]),
            rule_id=str(row["rule_id"]),
            stage_id=str(row["stage_id"]),
            stage_weight_bp=int(row["stage_weight_bp"]),
            hash_bucket=int(row["hash_bucket"]),
            side=str(row["side"]),
            backend_version=str(row["backend_version"]),
            batch_seq=int(row["batch_seq"]),
            output_digest=str(row["output_digest"]),
            elapsed_us=int(row["elapsed_us"]),
            client_key=str(row["client_key"]),
            feature=str(row["feature"]),
            sample_json=str(row["sample_json"]),
            format_version=int(row["format_version"]),
            content_digest=str(row["content_digest"]),
            created_at_ms=int(row["created_at_ms"]),
        )

    def hits_upto(self, ledger_version: int) -> list[HitRecord]:
        rows = self._read_all(
            "SELECT * FROM hits WHERE ledger_version <= ? ORDER BY ledger_version",
            (ledger_version,),
        )
        return [self._hit_from_row(row) for row in rows]

    def all_hits(self) -> list[HitRecord]:
        rows = self._read_all("SELECT * FROM hits ORDER BY ledger_version", ())
        return [self._hit_from_row(row) for row in rows]

    def hit_at_version(self, ledger_version: int) -> HitRecord | None:
        rows = self._read_all(
            "SELECT * FROM hits WHERE ledger_version = ?", (ledger_version,)
        )
        return self._hit_from_row(rows[0]) if rows else None

    def count_hits_after(self, ledger_version: int) -> int:
        row = self._read_one(
            "SELECT COUNT(*) AS value FROM hits WHERE ledger_version > ?",
            (ledger_version,),
        )
        return int(row["value"])

    def count_hits_between(self, low_exclusive: int, high_inclusive: int) -> int:
        row = self._read_one(
            "SELECT COUNT(*) AS value FROM hits WHERE ledger_version > ? AND ledger_version <= ?",
            (low_exclusive, high_inclusive),
        )
        return int(row["value"])

    def snapshot_bound(self, source_ledger_version: int) -> tuple[int, int]:
        """Read the completion bound and excluded count in one read snapshot."""
        connection = self._read_connection()
        try:
            connection.execute("BEGIN")
            row = connection.execute(
                "SELECT COALESCE(MAX(ledger_version), 0) AS value FROM hits"
            ).fetchone()
            observed = int(row["value"])
            excluded_row = connection.execute(
                """
                SELECT COUNT(*) AS value FROM hits
                 WHERE ledger_version > ? AND ledger_version <= ?
                """,
                (source_ledger_version, observed),
            ).fetchone()
            connection.execute("COMMIT")
        finally:
            connection.close()
        return observed, int(excluded_row["value"])

    def create_rebuild(
        self,
        *,
        rebuild_id: str,
        source_ledger_version: int,
        started_ms: int,
        budget_ms: int,
    ) -> None:
        with self._write_lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    """
                    INSERT INTO rebuilds(
                        rebuild_id, source_ledger_version, status, started_ms, budget_ms
                    ) VALUES(?,?,'RUNNING',?,?)
                    """,
                    (rebuild_id, source_ledger_version, started_ms, budget_ms),
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def complete_rebuild(
        self,
        *,
        rebuild_id: str,
        snapshot: dict[str, Any],
        observed_ledger_version: int,
        excluded_hits: int,
        elapsed_ms: int,
        latency_exceeded: bool,
    ) -> None:
        with self._write_lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    """
                    INSERT INTO snapshots(
                        snapshot_id, rebuild_id, source_ledger_version,
                        observed_ledger_version, excluded_hits, format_version,
                        content_digest, body_json, spec_digest, spec_json,
                        aggregation_path, rebuild_elapsed_ms, budget_ms,
                        latency_exceeded, created_at_ms
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        snapshot["snapshot_id"],
                        rebuild_id,
                        snapshot["source_ledger_version"],
                        observed_ledger_version,
                        excluded_hits,
                        snapshot["format_version"],
                        snapshot["content_digest"],
                        snapshot["body_json"],
                        snapshot["spec_digest"],
                        snapshot["spec_json"],
                        snapshot["aggregation_path"],
                        elapsed_ms,
                        snapshot["budget_ms"],
                        1 if latency_exceeded else 0,
                        snapshot["created_at_ms"],
                    ),
                )
                cursor = self._conn.execute(
                    """
                    UPDATE rebuilds
                       SET status='COMPLETE', observed_ledger_version=?,
                           finished_ms=?, elapsed_ms=?, excluded_hits=?,
                           snapshot_id=?, error_code=?, detail_json=?
                     WHERE rebuild_id=? AND status='RUNNING'
                    """,
                    (
                        observed_ledger_version,
                        monotonic_ms(),
                        elapsed_ms,
                        excluded_hits,
                        snapshot["snapshot_id"],
                        "E_LATENCY" if latency_exceeded else None,
                        canonical_json(
                            {
                                "aggregation_path": snapshot["aggregation_path"],
                                "latency_exceeded": bool(latency_exceeded),
                                "source_ledger_version": snapshot[
                                    "source_ledger_version"
                                ],
                            }
                        ),
                        rebuild_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError(f"rebuild {rebuild_id!r} is not RUNNING")
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def fail_rebuild(
        self, *, rebuild_id: str, error_code: str, detail: dict[str, Any]
    ) -> None:
        with self._write_lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    """
                    UPDATE rebuilds
                       SET status='FAILED', finished_ms=?, error_code=?, detail_json=?
                     WHERE rebuild_id=? AND status='RUNNING'
                    """,
                    (monotonic_ms(), error_code, canonical_json(detail), rebuild_id),
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def get_rebuild(self, rebuild_id: str) -> dict[str, Any] | None:
        rows = self._read_all(
            "SELECT * FROM rebuilds WHERE rebuild_id=?", (rebuild_id,)
        )
        return dict(rows[0]) if rows else None

    def list_rebuilds(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._read_all(
            "SELECT * FROM rebuilds ORDER BY started_ms DESC, rebuild_id DESC LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in rows]

    def save_consistency(
        self,
        *,
        snapshot_id: str,
        check_kind: str,
        ok: bool,
        code: str,
        rule_id: str,
        stage_id: str,
        field_name: str,
        detail: dict[str, Any],
    ) -> None:
        with self._write_lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    """
                    INSERT INTO consistency_checks(
                        snapshot_id, check_kind, ok, code, rule_id, stage_id,
                        field_name, detail_json, checked_at_ms
                    ) VALUES(?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        snapshot_id,
                        check_kind,
                        1 if ok else 0,
                        code,
                        rule_id,
                        stage_id,
                        field_name,
                        canonical_json(detail),
                        utc_epoch_ms(),
                    ),
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def latest_consistency(
        self, snapshot_id: str, check_kind: str
    ) -> dict[str, Any] | None:
        rows = self._read_all(
            """
            SELECT * FROM consistency_checks
             WHERE snapshot_id=? AND check_kind=?
             ORDER BY check_id DESC LIMIT 1
            """,
            (snapshot_id, check_kind),
        )
        return dict(rows[0]) if rows else None

    def record_latency(
        self,
        *,
        kind: str,
        ref_id: str,
        start_ms: int,
        end_ms: int,
        budget_ms: int,
        impact_scope: str,
        error_code: str | None = None,
    ) -> dict[str, Any]:
        elapsed_ms = max(0, end_ms - start_ms)
        exceeded = elapsed_ms > budget_ms
        over_by_ms = max(0, elapsed_ms - budget_ms)
        code = "E_LATENCY" if exceeded else error_code
        with self._write_lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                cursor = self._conn.execute(
                    """
                    INSERT INTO latency_observations(
                        kind, ref_id, start_ms, end_ms, elapsed_ms, budget_ms,
                        exceeded, over_by_ms, error_code, impact_scope,
                        clock_basis, truth_source, created_at_ms
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        kind,
                        ref_id,
                        start_ms,
                        end_ms,
                        elapsed_ms,
                        budget_ms,
                        1 if exceeded else 0,
                        over_by_ms,
                        code,
                        impact_scope,
                        CLOCK_BASIS,
                        TRUTH_SOURCE,
                        utc_epoch_ms(),
                    ),
                )
                observation_id = int(cursor.lastrowid)
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        return {
            "observation_id": observation_id,
            "kind": kind,
            "ref_id": ref_id,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "elapsed_ms": elapsed_ms,
            "budget_ms": budget_ms,
            "exceeded": exceeded,
            "over_by_ms": over_by_ms,
            "error_code": code,
            "impact_scope": impact_scope,
            "clock_basis": CLOCK_BASIS,
            "truth_source": TRUTH_SOURCE,
        }

    def latency_values(self, kind: str) -> list[int]:
        rows = self._read_all(
            "SELECT elapsed_ms FROM latency_observations WHERE kind=? ORDER BY observation_id",
            (kind,),
        )
        return [int(row["elapsed_ms"]) for row in rows]

    def list_latency(self, kind: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        if kind is None:
            rows = self._read_all(
                "SELECT * FROM latency_observations ORDER BY observation_id DESC LIMIT ?",
                (limit,),
            )
        else:
            rows = self._read_all(
                "SELECT * FROM latency_observations WHERE kind=? ORDER BY observation_id DESC LIMIT ?",
                (kind, limit),
            )
        return [dict(row) for row in rows]

    @staticmethod
    def _snapshot_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["body"] = json.loads(result["body_json"])
        return result

    def get_snapshot(self, snapshot_id: str) -> dict[str, Any] | None:
        rows = self._read_all(
            "SELECT * FROM snapshots WHERE snapshot_id=?", (snapshot_id,)
        )
        return self._snapshot_row_to_dict(rows[0]) if rows else None

    def latest_snapshot(self) -> dict[str, Any] | None:
        rows = self._read_all(
            "SELECT * FROM snapshots ORDER BY snapshot_row_id DESC LIMIT 1", ()
        )
        return self._snapshot_row_to_dict(rows[0]) if rows else None

    def list_snapshots(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._read_all(
            "SELECT * FROM snapshots ORDER BY snapshot_row_id DESC LIMIT ?", (limit,)
        )
        return [self._snapshot_row_to_dict(row) for row in rows]

    def verify_ledger_chain(self) -> tuple[bool, int | None, str]:
        previous = GENESIS_DIGEST
        for hit in self.all_hits():
            expected = sha256_text(
                previous + "\n" + canonical_json(hit.canonical_without_digest())
            )
            if expected != hit.content_digest:
                return False, hit.ledger_version, "content_digest"
            previous = hit.content_digest
        if previous != self.ledger_digest():
            return False, None, "ledger_digest"
        return True, None, "-"
