from __future__ import annotations

import json
import os
import shutil
import sqlite3
import stat
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .canonical import canonical_json
from .compute import (
    FORMAT_VERSION,
    build_ledger_digest,
    build_view,
    record_from_sql_row,
    record_to_sql_params,
    snapshot_payload,
    verify_record_digest,
)
from .errors import DiagnosticError
from .model import ConfigSpec, SubscriptionSample
from .structured_log import JsonlLogger, NullLogger

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ledger_meta (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    format_version INTEGER NOT NULL,
    ledger_version INTEGER NOT NULL,
    config_set_version INTEGER NOT NULL,
    derivation_version INTEGER NOT NULL,
    change_cursor INTEGER NOT NULL,
    content_digest TEXT NOT NULL,
    initialized_mono_ns INTEGER NOT NULL,
    last_mutation_mono_ns INTEGER NOT NULL,
    initial_spec_json TEXT NOT NULL,
    initial_clients_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ledger_keys (
    key TEXT PRIMARY KEY,
    raw_value INTEGER NOT NULL,
    value INTEGER,
    version INTEGER NOT NULL,
    refs_json TEXT NOT NULL,
    formula_json TEXT NOT NULL,
    ttl_ms INTEGER NOT NULL,
    expires_mono_ns INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('active', 'invalid', 'expired')),
    basis_json TEXT NOT NULL,
    content_digest TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS change_journal (
    journal_id INTEGER PRIMARY KEY AUTOINCREMENT,
    operation_id TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    source_sequence INTEGER,
    status TEXT NOT NULL CHECK (status IN ('pending', 'committed', 'unknown')),
    payload_json TEXT NOT NULL,
    trace_json TEXT,
    started_mono_ns INTEGER NOT NULL,
    committed_mono_ns INTEGER,
    base_ledger_version INTEGER NOT NULL,
    result_ledger_version INTEGER,
    base_set_version INTEGER NOT NULL,
    result_set_version INTEGER,
    base_derivation_version INTEGER NOT NULL,
    result_derivation_version INTEGER
);

CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ledger_version INTEGER NOT NULL,
    config_set_version INTEGER NOT NULL,
    derivation_version INTEGER NOT NULL,
    change_cursor INTEGER NOT NULL,
    content_digest TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_mono_ns INTEGER NOT NULL,
    reason TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS views (
    view_version INTEGER PRIMARY KEY,
    config_set_version INTEGER NOT NULL,
    derivation_version INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('current', 'stale')),
    view_digest TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_mono_ns INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS stale_traces (
    journal_id INTEGER PRIMARY KEY,
    stale_since_view INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_change_status ON change_journal(status, journal_id);
CREATE INDEX IF NOT EXISTS idx_snapshot_version ON snapshots(ledger_version DESC, snapshot_id DESC);
"""


class Ledger:
    def __init__(
        self,
        db_path: Path | str,
        *,
        logger: JsonlLogger | NullLogger | None = None,
    ) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.logger = logger or NullLogger()
        self.conn = sqlite3.connect(str(self.path), timeout=30.0, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.execute("PRAGMA busy_timeout=30000")
        self.conn.executescript(SCHEMA_SQL)
        self.open_events: list[dict[str, Any]] = []
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.conn.execute("PRAGMA wal_checkpoint(FULL)")
        except sqlite3.Error:
            pass
        self.conn.close()

    @contextmanager
    def transaction(self):
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        else:
            self.conn.execute("COMMIT")

    def has_state(self) -> bool:
        row = self.conn.execute("SELECT COUNT(*) AS n FROM ledger_meta").fetchone()
        return bool(row and row["n"])

    def meta(self) -> dict[str, Any]:
        row = self.conn.execute("SELECT * FROM ledger_meta WHERE id = 1").fetchone()
        if row is None:
            raise DiagnosticError("E_STATE", "账本尚未初始化", field="ledger_meta")
        return dict(row)

    def records(self) -> dict[str, dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM ledger_keys ORDER BY key").fetchall()
        return {row["key"]: record_from_sql_row(row) for row in rows}

    def raw_key_rows(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self.conn.execute("SELECT * FROM ledger_keys ORDER BY key")]

    def journal_rows(self, status: str | None = None) -> list[dict[str, Any]]:
        base_sql = (
            "SELECT cj.*, st.stale_since_view FROM change_journal cj "
            "LEFT JOIN stale_traces st ON st.journal_id = cj.journal_id "
        )
        if status:
            rows = self.conn.execute(
                base_sql + "WHERE cj.status = ? ORDER BY cj.journal_id",
                (status,),
            ).fetchall()
        else:
            rows = self.conn.execute(base_sql + "ORDER BY cj.journal_id").fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item["payload_json"])
            item["trace"] = json.loads(item["trace_json"]) if item["trace_json"] else None
            result.append(item)
        return result

    def get_journal_by_operation(self, operation_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT cj.*, st.stale_since_view FROM change_journal cj "
            "LEFT JOIN stale_traces st ON st.journal_id = cj.journal_id "
            "WHERE cj.operation_id = ?",
            (operation_id,),
        ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["payload"] = json.loads(item["payload_json"])
        item["trace"] = json.loads(item["trace_json"]) if item["trace_json"] else None
        return item

    def unresolved_unknown(self) -> list[dict[str, Any]]:
        return self.journal_rows("unknown")

    def snapshots(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM snapshots ORDER BY ledger_version DESC, snapshot_id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def views(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM views ORDER BY view_version DESC LIMIT ?",
            (limit,),
        ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["payload"] = json.loads(item["payload_json"])
            result.append(item)
        return result

    def current_view(self) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM views WHERE status = 'current' ORDER BY view_version DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["payload_json"])

    def initialize(
        self,
        spec: ConfigSpec,
        clients: SubscriptionSample,
        records: Mapping[str, Mapping[str, Any]],
        now_mono_ns: int,
    ) -> dict[str, Any]:
        if self.has_state():
            raise DiagnosticError("E_STATE", "账本已初始化，拒绝覆盖", field="ledger_meta")
        records_copy = {key: dict(value) for key, value in records.items()}
        meta: dict[str, Any] = {
            "format_version": FORMAT_VERSION,
            "ledger_version": 1,
            "config_set_version": 1,
            "derivation_version": spec.derivation_version,
            "change_cursor": 0,
            "content_digest": "",
            "initialized_mono_ns": now_mono_ns,
            "last_mutation_mono_ns": now_mono_ns,
            "initial_spec_json": canonical_json(spec.to_obj()),
            "initial_clients_json": canonical_json(clients.to_obj()),
        }
        meta["content_digest"] = build_ledger_digest(meta, records_copy)
        view = build_view(
            records_copy,
            view_version=1,
            config_set_version=meta["config_set_version"],
            derivation_version=meta["derivation_version"],
            created_mono_ns=now_mono_ns,
            previous_view=None,
        )
        snap = snapshot_payload(meta, records_copy, view)
        with self.transaction():
            self.conn.execute(
                """
                INSERT INTO ledger_meta(
                    id, format_version, ledger_version, config_set_version,
                    derivation_version, change_cursor, content_digest,
                    initialized_mono_ns, last_mutation_mono_ns,
                    initial_spec_json, initial_clients_json
                ) VALUES(1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    meta["format_version"],
                    meta["ledger_version"],
                    meta["config_set_version"],
                    meta["derivation_version"],
                    meta["change_cursor"],
                    meta["content_digest"],
                    meta["initialized_mono_ns"],
                    meta["last_mutation_mono_ns"],
                    meta["initial_spec_json"],
                    meta["initial_clients_json"],
                ),
            )
            for key in sorted(records_copy):
                self._upsert_record(records_copy[key])
            self._insert_current_view(view)
            self._insert_snapshot(meta, snap, now_mono_ns, "initial")
        self.logger.emit(
            "INFO_INIT",
            "ledger_initialized",
            message="配置账本初始装载完成",
            ledger_path=str(self.path),
            ledger_version=meta["ledger_version"],
            content_digest=meta["content_digest"],
            key_count=len(records_copy),
        )
        return {
            "ledger_version": meta["ledger_version"],
            "config_set_version": meta["config_set_version"],
            "derivation_version": meta["derivation_version"],
            "content_digest": meta["content_digest"],
            "view_version": view["view_version"],
            "view_digest": view["view_digest"],
        }

    def _upsert_record(self, record: Mapping[str, Any]) -> None:
        params = record_to_sql_params(record)
        self.conn.execute(
            """
            INSERT INTO ledger_keys(
                key, raw_value, value, version, refs_json, formula_json,
                ttl_ms, expires_mono_ns, status, basis_json, content_digest
            ) VALUES(
                :key, :raw_value, :value, :version, :refs_json, :formula_json,
                :ttl_ms, :expires_mono_ns, :status, :basis_json, :content_digest
            )
            ON CONFLICT(key) DO UPDATE SET
                raw_value = excluded.raw_value,
                value = excluded.value,
                version = excluded.version,
                refs_json = excluded.refs_json,
                formula_json = excluded.formula_json,
                ttl_ms = excluded.ttl_ms,
                expires_mono_ns = excluded.expires_mono_ns,
                status = excluded.status,
                basis_json = excluded.basis_json,
                content_digest = excluded.content_digest
            """,
            params,
        )

    def _write_meta(self, meta: Mapping[str, Any]) -> None:
        self.conn.execute(
            """
            UPDATE ledger_meta SET
                format_version = ?, ledger_version = ?, config_set_version = ?,
                derivation_version = ?, change_cursor = ?, content_digest = ?,
                initialized_mono_ns = ?, last_mutation_mono_ns = ?,
                initial_spec_json = ?, initial_clients_json = ?
            WHERE id = 1
            """,
            (
                meta["format_version"],
                meta["ledger_version"],
                meta["config_set_version"],
                meta["derivation_version"],
                meta["change_cursor"],
                meta["content_digest"],
                meta["initialized_mono_ns"],
                meta["last_mutation_mono_ns"],
                meta["initial_spec_json"],
                meta["initial_clients_json"],
            ),
        )

    def _insert_current_view(self, view: Mapping[str, Any]) -> None:
        self.conn.execute("UPDATE views SET status = 'stale' WHERE status = 'current'")
        # A trace belongs to the view that was current when it was committed.
        # Every later whole-view replacement renders the previous trace stale.
        self.conn.execute(
            "INSERT OR IGNORE INTO stale_traces(journal_id, stale_since_view) "
            "SELECT journal_id, ? FROM change_journal WHERE status = 'committed'",
            (view["view_version"],),
        )
        self.conn.execute(
            """
            INSERT INTO views(
                view_version, config_set_version, derivation_version,
                status, view_digest, payload_json, created_mono_ns
            ) VALUES(?, ?, ?, 'current', ?, ?, ?)
            """,
            (
                view["view_version"],
                view["config_set_version"],
                view["derivation_version"],
                view["view_digest"],
                canonical_json(view),
                view["created_mono_ns"],
            ),
        )

    def _insert_snapshot(self, meta: Mapping[str, Any], payload: Mapping[str, Any], now_ns: int, reason: str) -> None:
        self.conn.execute(
            """
            INSERT INTO snapshots(
                ledger_version, config_set_version, derivation_version,
                change_cursor, content_digest, payload_json,
                created_mono_ns, reason
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                meta["ledger_version"],
                meta["config_set_version"],
                meta["derivation_version"],
                meta["change_cursor"],
                meta["content_digest"],
                canonical_json(payload),
                now_ns,
                reason,
            ),
        )

    def prepare_journal(
        self,
        *,
        operation_id: str,
        kind: str,
        source_sequence: int | None,
        payload: Mapping[str, Any],
        meta: Mapping[str, Any],
        started_mono_ns: int,
    ) -> tuple[int, bool]:
        """Commit an intent row before the atomic ledger transaction.

        Returns (journal_id, is_new).  A committed operation is idempotent.  An
        unknown operation from a forced termination is allowed to be rerun with
        the same journal id, which keeps cursor semantics simple.
        """

        existing = self.get_journal_by_operation(operation_id)
        if existing is not None:
            if existing["status"] == "committed":
                return int(existing["journal_id"]), False
            if existing["status"] == "pending":
                raise DiagnosticError(
                    "E_STATE",
                    "存在未完成事务，需先恢复并要求重跑",
                    key=operation_id,
                    field="change_journal.status",
                    context={"journal_id": existing["journal_id"]},
                )
            journal_id = int(existing["journal_id"])
            with self.transaction():
                self.conn.execute(
                    """
                    UPDATE change_journal SET
                        status = 'pending', payload_json = ?, trace_json = NULL,
                        started_mono_ns = ?, committed_mono_ns = NULL,
                        base_ledger_version = ?, result_ledger_version = NULL,
                        base_set_version = ?, result_set_version = NULL,
                        base_derivation_version = ?, result_derivation_version = NULL
                    WHERE journal_id = ?
                    """,
                    (
                        canonical_json(payload),
                        started_mono_ns,
                        meta["ledger_version"],
                        meta["config_set_version"],
                        meta["derivation_version"],
                        journal_id,
                    ),
                )
            return journal_id, True

        with self.transaction():
            cursor = self.conn.execute(
                "INSERT INTO change_journal("
                "operation_id, kind, source_sequence, status, payload_json, trace_json, "
                "started_mono_ns, base_ledger_version, base_set_version, base_derivation_version"
                ") VALUES(?, ?, ?, 'pending', ?, NULL, ?, ?, ?, ?)",
                (
                    operation_id,
                    kind,
                    source_sequence,
                    canonical_json(payload),
                    started_mono_ns,
                    meta["ledger_version"],
                    meta["config_set_version"],
                    meta["derivation_version"],
                ),
            )
            return int(cursor.lastrowid), True

    def mark_journal_committed(
        self,
        journal_id: int,
        meta: Mapping[str, Any],
        trace: Mapping[str, Any],
        committed_mono_ns: int,
    ) -> None:
        self.conn.execute(
            """
            UPDATE change_journal SET
                status = 'committed', trace_json = ?, committed_mono_ns = ?,
                result_ledger_version = ?, result_set_version = ?,
                result_derivation_version = ?
            WHERE journal_id = ?
            """,
            (
                canonical_json(trace),
                committed_mono_ns,
                meta["ledger_version"],
                meta["config_set_version"],
                meta["derivation_version"],
                journal_id,
            ),
        )

    def commit_state(
        self,
        *,
        meta: Mapping[str, Any],
        changed_records: Iterable[Mapping[str, Any]],
        view: Mapping[str, Any],
        journal_id: int,
        trace: Mapping[str, Any],
        committed_mono_ns: int,
        snapshot_reason: str,
    ) -> None:
        for record in sorted(changed_records, key=lambda item: item["key"]):
            self._upsert_record(record)
        self._write_meta(meta)
        self.mark_journal_committed(journal_id, meta, trace, committed_mono_ns)
        all_records = self.records()
        self._insert_current_view(view)
        snap = snapshot_payload(meta, all_records, view)
        self._insert_snapshot(meta, snap, committed_mono_ns, snapshot_reason)

    def open_existing(self) -> list[dict[str, Any]]:
        if not self.has_state():
            return self.open_events

        pending = self.journal_rows("pending")
        if pending:
            ids = [item["journal_id"] for item in pending]
            with self.transaction():
                self.conn.execute(
                    "UPDATE change_journal SET status = 'unknown' WHERE status = 'pending'"
                )
            event = {
                "code": "E_RECOVERY",
                "severity": "WARNING",
                "event": "unclean_shutdown_detected",
                "message": "检测到提交中或提交前终止的事务；只能按已提交账本恢复，原操作必须重跑",
                "journal_ids": ids,
                "exit_report_available": False,
                "requires_rerun": True,
            }
            self.open_events.append(event)
            self.logger.emit(
                event["code"],
                event["event"],
                severity=event["severity"],
                message=event["message"],
                journal_ids=ids,
                exit_report_available=False,
                requires_rerun=True,
            )

        mismatch = self._state_mismatch()
        if mismatch:
            event = self._repair_from_snapshot(mismatch)
            self.open_events.append(event)
        return self.open_events

    def _state_mismatch(self) -> dict[str, Any] | None:
        integrity = self.conn.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            return {"reason": "sqlite_integrity_check", "detail": integrity}
        meta = self.meta()
        records = self.records()
        if meta["format_version"] != FORMAT_VERSION:
            return {
                "reason": "format_version_mismatch",
                "expected": FORMAT_VERSION,
                "actual": meta["format_version"],
            }
        bad_keys = [key for key, record in records.items() if not verify_record_digest(record)]
        actual_digest = build_ledger_digest(meta, records)
        if bad_keys or actual_digest != meta["content_digest"]:
            return {
                "reason": "content_digest_mismatch",
                "stored": meta["content_digest"],
                "actual": actual_digest,
                "bad_keys": sorted(bad_keys),
            }
        return None

    def _readonly_backup(self) -> Path:
        self.conn.execute("PRAGMA wal_checkpoint(FULL)")
        backup_dir = self.path.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        target = backup_dir / f"{self.path.stem}.corrupt.{stamp}.{time.monotonic_ns()}.sqlite3"
        shutil.copy2(self.path, target)
        os.chmod(target, stat.S_IREAD)
        return target

    def _repair_from_snapshot(self, mismatch: Mapping[str, Any]) -> dict[str, Any]:
        backup = self._readonly_backup()
        selected: tuple[dict[str, Any], dict[str, Any]] | None = None
        rejected: list[dict[str, Any]] = []
        for row in self.snapshots(limit=1000):
            try:
                payload = json.loads(row["payload_json"])
                snap_meta = dict(payload["meta"])
                records = {
                    item["key"]: dict(item) for item in payload["records"]
                }
                if snap_meta["format_version"] != FORMAT_VERSION:
                    raise ValueError("snapshot format version mismatch")
                computed = build_ledger_digest(snap_meta, records)
                if computed != row["content_digest"] or computed != snap_meta["content_digest"]:
                    raise ValueError("snapshot digest mismatch")
                if any(not verify_record_digest(record) for record in records.values()):
                    raise ValueError("snapshot key digest mismatch")
                selected = (row, payload)
                break
            except Exception as exc:  # a corrupt snapshot must not block an older valid one
                rejected.append({"snapshot_id": row["snapshot_id"], "reason": str(exc)})
        if selected is None:
            event = {
                "code": "E_SNAPSHOT",
                "severity": "ERROR",
                "event": "snapshot_recovery_failed",
                "message": "账本版本或摘要不一致，且没有可验证的快照可供整体替换",
                "mismatch": dict(mismatch),
                "backup": str(backup),
                "rejected_snapshots": rejected,
            }
            self.logger.emit(
                event["code"],
                event["event"],
                severity=event["severity"],
                message=event["message"],
                mismatch=dict(mismatch),
                backup=str(backup),
                rejected_snapshots=rejected,
            )
            raise DiagnosticError(
                "E_SNAPSHOT",
                event["message"],
                field="ledger_meta.content_digest",
                context={
                    "mismatch": dict(mismatch),
                    "backup": str(backup),
                    "rejected_snapshots": rejected,
                },
            )

        row, payload = selected
        snap_meta = dict(payload["meta"])
        records = {item["key"]: dict(item) for item in payload["records"]}
        view = payload["view"]
        with self.transaction():
            self.conn.execute("DELETE FROM ledger_keys")
            for key in sorted(records):
                self._upsert_record(records[key])
            self._write_meta(snap_meta)
            self.conn.execute("DELETE FROM views")
            self.conn.execute(
                """
                INSERT INTO views(
                    view_version, config_set_version, derivation_version,
                    status, view_digest, payload_json, created_mono_ns
                ) VALUES(?, ?, ?, 'current', ?, ?, ?)
                """,
                (
                    view["view_version"],
                    view["config_set_version"],
                    view["derivation_version"],
                    view["view_digest"],
                    canonical_json(view),
                    view["created_mono_ns"],
                ),
            )
        post = self._state_mismatch()
        if post:
            raise DiagnosticError(
                "E_SNAPSHOT",
                "快照整体替换后摘要仍不一致",
                field="ledger_meta.content_digest",
                context={"mismatch": post, "backup": str(backup)},
            )
        event = {
            "code": "E_SNAPSHOT",
            "severity": "WARNING",
            "event": "snapshot_mismatch_recovered",
            "message": "账本格式版本或内容摘要不一致；已保留只读备份并整体替换为已验证快照",
            "mismatch": dict(mismatch),
            "backup": str(backup),
            "snapshot_id": row["snapshot_id"],
            "restored_ledger_version": snap_meta["ledger_version"],
            "mutation_mode": "whole_snapshot_replace",
        }
        self.logger.emit(
            event["code"],
            event["event"],
            severity=event["severity"],
            message=event["message"],
            mismatch=dict(mismatch),
            backup=str(backup),
            snapshot_id=row["snapshot_id"],
            restored_ledger_version=snap_meta["ledger_version"],
            mutation_mode="whole_snapshot_replace",
            rejected_snapshots=rejected,
        )
        return event

    def commit_cursor(self) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(journal_id), 0) AS cursor FROM change_journal WHERE status = 'committed'"
        ).fetchone()
        return int(row["cursor"] if row else 0)

    def max_committed_source_sequence(self) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(source_sequence), 0) AS seq FROM change_journal "
            "WHERE status = 'committed' AND source_sequence IS NOT NULL"
        ).fetchone()
        return int(row["seq"] if row else 0)
