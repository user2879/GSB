"""SQLite snapshot and convergence-ledger persistence."""

from __future__ import annotations

import json
import os
import sqlite3
import stat
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .canonical import canonical_json, content_digest
from .config import EvaluationConfig
from .errors import snapshot_error
from .models import FIELD_ORDER, TargetSet


LEDGER_FORMAT_VERSION = 1


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def make_batch_key(snapshot_digest: str, event_cursor: int) -> str:
    return content_digest({"snapshot_digest": snapshot_digest, "event_cursor": event_cursor})


def _field_sort_key(field_path: str) -> tuple[str, int, str]:
    workload, _, field = field_path.partition("/")
    return workload, FIELD_ORDER.index(field) if field in FIELD_ORDER else len(FIELD_ORDER), field


class LedgerStore:
    """One connection to the local audit ledger.

    Normal instances are read/write. The Tkinter view uses ``read_only=True`` so
    refreshing the UI cannot create tables, update metadata, or write anything.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        backup_dir: str | Path | None = None,
        read_only: bool = False,
    ) -> None:
        self.path = Path(path)
        self.backup_dir = Path(backup_dir) if backup_dir else self.path.parent / "declaration-backups"
        self.read_only = read_only
        if not read_only:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(
                str(self.path),
                timeout=30,
                isolation_level=None,
                check_same_thread=False,
            )
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA foreign_keys = ON")
            self.conn.execute("PRAGMA journal_mode = WAL")
            self.conn.execute("PRAGMA synchronous = FULL")
            self._ensure_schema()
        else:
            if not self.path.exists():
                raise snapshot_error(
                    f"ledger does not exist: {self.path}", item=str(self.path), field="<ledger>"
                )
            uri_path = quote(str(self.path.resolve()).replace("\\", "/"), safe="/:")
            self.conn = sqlite3.connect(
                f"file:{uri_path}?mode=ro",
                uri=True,
                timeout=30,
                isolation_level=None,
                check_same_thread=False,
            )
            self.conn.row_factory = sqlite3.Row
            self._verify_schema_meta()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "LedgerStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _ensure_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS ledger_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS snapshot (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                declaration_format_versions TEXT NOT NULL,
                declaration_digest TEXT NOT NULL,
                target_digest TEXT NOT NULL,
                defaults_version TEXT NOT NULL,
                scale_version TEXT NOT NULL,
                scale_exp INTEGER NOT NULL,
                tolerance_num INTEGER NOT NULL,
                tolerance_den INTEGER NOT NULL,
                target_json TEXT NOT NULL,
                raw_declaration_json TEXT NOT NULL,
                snapshot_digest TEXT NOT NULL,
                updated_at_ms INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS replacements (
                replacement_id INTEGER PRIMARY KEY AUTOINCREMENT,
                reason TEXT NOT NULL,
                old_snapshot_digest TEXT,
                new_snapshot_digest TEXT NOT NULL,
                diff_json TEXT NOT NULL,
                replaced_at_ms INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS batches (
                batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_key TEXT NOT NULL UNIQUE,
                snapshot_digest TEXT NOT NULL,
                ledger_version INTEGER NOT NULL,
                event_cursor INTEGER NOT NULL,
                first_event_seq INTEGER NOT NULL,
                batch_kind TEXT NOT NULL,
                format_versions_json TEXT NOT NULL,
                defaults_version TEXT NOT NULL,
                scale_version TEXT NOT NULL,
                scale_exp INTEGER NOT NULL,
                workload_metrics_json TEXT NOT NULL,
                trace_json TEXT NOT NULL,
                content_digest TEXT NOT NULL,
                created_at_ms INTEGER NOT NULL,
                UNIQUE(snapshot_digest, event_cursor)
            );

            CREATE TABLE IF NOT EXISTS decisions (
                batch_id INTEGER NOT NULL REFERENCES batches(batch_id) ON DELETE CASCADE,
                workload TEXT NOT NULL,
                field TEXT NOT NULL,
                drifted INTEGER NOT NULL,
                target_json TEXT NOT NULL,
                observed_json TEXT NOT NULL,
                observed_present INTEGER NOT NULL,
                basis TEXT NOT NULL,
                reason TEXT NOT NULL,
                field_version INTEGER NOT NULL,
                metric_json TEXT NOT NULL,
                PRIMARY KEY (batch_id, workload, field)
            );

            CREATE TABLE IF NOT EXISTS inflight (
                attempt_id TEXT PRIMARY KEY,
                batch_key TEXT NOT NULL,
                workloads_json TEXT NOT NULL,
                started_at_ms INTEGER NOT NULL,
                plan_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS unknown_workloads (
                workload TEXT NOT NULL,
                attempt_id TEXT NOT NULL,
                batch_key TEXT NOT NULL,
                reason TEXT NOT NULL,
                marked_at_ms INTEGER NOT NULL,
                PRIMARY KEY (workload, attempt_id)
            );
            """
        )
        row = self.conn.execute(
            "SELECT value FROM ledger_meta WHERE key = 'ledger_format_version'"
        ).fetchone()
        if row is None:
            self.conn.execute(
                "INSERT INTO ledger_meta(key, value) VALUES ('ledger_format_version', ?)",
                (str(LEDGER_FORMAT_VERSION),),
            )
        elif int(row["value"]) != LEDGER_FORMAT_VERSION:
            raise snapshot_error(
                "incompatible ledger format version",
                item=str(self.path),
                field="ledger_format_version",
                details={"expected": LEDGER_FORMAT_VERSION, "actual": row["value"]},
            )

    def _verify_schema_meta(self) -> None:
        try:
            row = self.conn.execute(
                "SELECT value FROM ledger_meta WHERE key = 'ledger_format_version'"
            ).fetchone()
        except sqlite3.DatabaseError as exc:
            raise snapshot_error(
                f"invalid ledger database: {exc}", item=str(self.path), field="<ledger>"
            ) from exc
        if row is None or int(row["value"]) != LEDGER_FORMAT_VERSION:
            raise snapshot_error(
                "incompatible or missing ledger format version",
                item=str(self.path),
                field="ledger_format_version",
                details={"expected": LEDGER_FORMAT_VERSION},
            )

    def _snapshot_payload(
        self,
        *,
        declaration_format_versions: list[int],
        declaration_digest: str,
        target_digest: str,
        defaults_version: str,
        scale_version: str,
        scale_exp: int,
        tolerance_num: int,
        tolerance_den: int,
        target: Any,
        raw_declaration: Any,
    ) -> dict[str, Any]:
        return {
            "ledger_format_version": LEDGER_FORMAT_VERSION,
            "declaration_format_versions": declaration_format_versions,
            "declaration_digest": declaration_digest,
            "target_digest": target_digest,
            "defaults_version": defaults_version,
            "scale_version": scale_version,
            "scale_exp": scale_exp,
            "tolerance_num": tolerance_num,
            "tolerance_den": tolerance_den,
            "target": target,
            "raw_declaration": raw_declaration,
        }

    def read_snapshot(self) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM snapshot WHERE id = 1").fetchone()
        if row is None:
            return None
        try:
            target = json.loads(row["target_json"])
            raw_declaration = json.loads(row["raw_declaration_json"])
        except json.JSONDecodeError as exc:
            raise snapshot_error(
                "snapshot contains invalid JSON", item="snapshot", field="<json>"
            ) from exc
        payload = self._snapshot_payload(
            declaration_format_versions=json.loads(row["declaration_format_versions"]),
            declaration_digest=row["declaration_digest"],
            target_digest=row["target_digest"],
            defaults_version=row["defaults_version"],
            scale_version=row["scale_version"],
            scale_exp=row["scale_exp"],
            tolerance_num=row["tolerance_num"],
            tolerance_den=row["tolerance_den"],
            target=target,
            raw_declaration=raw_declaration,
        )
        actual_digest = content_digest(payload)
        if actual_digest != row["snapshot_digest"]:
            raise snapshot_error(
                "snapshot content digest mismatch",
                item="snapshot",
                field="snapshot_digest",
                details={"expected": row["snapshot_digest"], "actual": actual_digest},
            )
        return {
            "snapshot_digest": row["snapshot_digest"],
            "ledger_format_version": LEDGER_FORMAT_VERSION,
            "declaration_format_versions": payload["declaration_format_versions"],
            "declaration_digest": row["declaration_digest"],
            "target_digest": row["target_digest"],
            "defaults_version": row["defaults_version"],
            "scale_version": row["scale_version"],
            "scale_exp": row["scale_exp"],
            "tolerance_num": row["tolerance_num"],
            "tolerance_den": row["tolerance_den"],
            "target": target,
            "raw_declaration": raw_declaration,
            "updated_at_ms": row["updated_at_ms"],
        }

    def get_snapshot_target(self) -> TargetSet | None:
        snapshot = self.read_snapshot()
        return TargetSet.from_dict(snapshot["target"]) if snapshot else None

    def _write_old_declaration_backup(self, snapshot: dict[str, Any]) -> str:
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        digest = snapshot["declaration_digest"]
        path = self.backup_dir / f"{digest}.json"
        if not path.exists():
            path.write_text(
                json.dumps(snapshot["raw_declaration"], ensure_ascii=False, sort_keys=True, indent=2)
                + "\n",
                encoding="utf-8",
            )
        try:
            os.chmod(path, stat.S_IREAD)
        except OSError:
            pass
        return str(path)

    def install_snapshot(
        self,
        target: TargetSet,
        raw_declaration: dict[str, Any],
        *,
        scale_version: str,
        scale_exp: int,
        tolerance_num: int = 0,
        tolerance_den: int = 1,
    ) -> dict[str, Any]:
        if self.read_only:
            raise snapshot_error("cannot replace a snapshot through a read-only ledger")
        previous = self.read_snapshot()
        format_versions = sorted(target.declaration_format_versions)
        target_json = target.to_dict()
        raw_json = deepcopy(raw_declaration)
        payload = self._snapshot_payload(
            declaration_format_versions=format_versions,
            declaration_digest=target.declaration_digest,
            target_digest=target.target_digest,
            defaults_version=target.defaults_version,
            scale_version=scale_version,
            scale_exp=scale_exp,
            tolerance_num=tolerance_num,
            tolerance_den=tolerance_den,
            target=target_json,
            raw_declaration=raw_json,
        )
        new_digest = content_digest(payload)

        reasons: list[str] = []
        if previous is None:
            reasons.append("initial")
        else:
            if previous["declaration_format_versions"] != format_versions:
                reasons.append("declaration_format_version_changed")
            if previous["declaration_digest"] != target.declaration_digest:
                reasons.append("declaration_digest_changed")
            if previous["defaults_version"] != target.defaults_version:
                reasons.append("defaults_version_changed")
            if previous["scale_version"] != scale_version:
                reasons.append("scale_version_changed")
            if previous["scale_exp"] != scale_exp:
                reasons.append("scale_factor_changed")
            if (
                previous["tolerance_num"] != tolerance_num
                or previous["tolerance_den"] != tolerance_den
            ):
                reasons.append("tolerance_changed")
            if previous["target_digest"] != target.target_digest and not reasons:
                reasons.append("target_digest_changed")

        if previous is not None and previous["snapshot_digest"] == new_digest:
            return {
                "replaced": False,
                "reasons": [],
                "diff": {"added": [], "removed": [], "upgraded": [], "details": []},
                "backup_path": None,
                "snapshot_digest": new_digest,
            }

        old_target = previous["target"] if previous else None
        diff = self._target_diff(old_target, target_json)
        backup_path = self._write_old_declaration_backup(previous) if previous else None

        self.conn.execute("BEGIN IMMEDIATE")
        try:
            if previous is not None:
                self.conn.execute(
                    """
                    INSERT INTO replacements(
                        reason, old_snapshot_digest, new_snapshot_digest,
                        diff_json, replaced_at_ms
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        ",".join(reasons),
                        previous["snapshot_digest"],
                        new_digest,
                        canonical_json(diff),
                        _now_ms(),
                    ),
                )
                self.conn.execute("DELETE FROM decisions")
                self.conn.execute("DELETE FROM batches")
                self.conn.execute("DELETE FROM inflight")
                self.conn.execute("DELETE FROM unknown_workloads")

            self.conn.execute(
                """
                INSERT INTO snapshot(
                    id, declaration_format_versions, declaration_digest, target_digest,
                    defaults_version, scale_version, scale_exp, tolerance_num,
                    tolerance_den, target_json, raw_declaration_json,
                    snapshot_digest, updated_at_ms
                ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    declaration_format_versions = excluded.declaration_format_versions,
                    declaration_digest = excluded.declaration_digest,
                    target_digest = excluded.target_digest,
                    defaults_version = excluded.defaults_version,
                    scale_version = excluded.scale_version,
                    scale_exp = excluded.scale_exp,
                    tolerance_num = excluded.tolerance_num,
                    tolerance_den = excluded.tolerance_den,
                    target_json = excluded.target_json,
                    raw_declaration_json = excluded.raw_declaration_json,
                    snapshot_digest = excluded.snapshot_digest,
                    updated_at_ms = excluded.updated_at_ms
                """,
                (
                    canonical_json(format_versions),
                    target.declaration_digest,
                    target.target_digest,
                    target.defaults_version,
                    scale_version,
                    scale_exp,
                    tolerance_num,
                    tolerance_den,
                    canonical_json(target_json),
                    canonical_json(raw_json),
                    new_digest,
                    _now_ms(),
                ),
            )
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise

        return {
            "replaced": True,
            "reasons": reasons,
            "diff": diff,
            "backup_path": backup_path,
            "snapshot_digest": new_digest,
        }

    def _target_diff(self, old_target: dict[str, Any] | None, new_target: dict[str, Any]) -> dict[str, Any]:
        old_workloads = old_target.get("workloads", {}) if old_target else {}
        new_workloads = new_target.get("workloads", {})
        added: list[str] = []
        removed: list[str] = []
        upgraded: list[str] = []
        details: list[dict[str, Any]] = []

        for workload in sorted(set(old_workloads) | set(new_workloads)):
            old = old_workloads.get(workload)
            new = new_workloads.get(workload)
            for field in FIELD_ORDER:
                field_path = f"{workload}/{field}"
                if old is None:
                    added.append(field_path)
                    details.append({"field": field_path, "kind": "added", "workload": workload})
                elif new is None:
                    removed.append(field_path)
                    details.append({"field": field_path, "kind": "removed", "workload": workload})
                else:
                    old_value = old["fields"][field]
                    new_value = new["fields"][field]
                    old_provenance = old["provenance"][field]
                    new_provenance = new["provenance"][field]
                    if old_value != new_value or old_provenance != new_provenance:
                        upgraded.append(field_path)
                        details.append(
                            {
                                "field": field_path,
                                "kind": "upgraded",
                                "workload": workload,
                                "old_value": old_value,
                                "new_value": new_value,
                                "value_changed": old_value != new_value,
                                "provenance_changed": old_provenance != new_provenance,
                            }
                        )
        added.sort(key=_field_sort_key)
        removed.sort(key=_field_sort_key)
        upgraded.sort(key=_field_sort_key)
        details.sort(key=lambda item: _field_sort_key(item["field"]))
        return {"added": added, "removed": removed, "upgraded": upgraded, "details": details}

    def begin_attempt(
        self,
        *,
        batch_key: str,
        workloads: list[str],
        plan: dict[str, Any],
    ) -> str:
        if self.read_only:
            raise snapshot_error("cannot record an attempt through a read-only ledger")
        attempt_id = f"{batch_key[:16]}-{uuid.uuid4().hex}"
        self.conn.execute(
            """
            INSERT INTO inflight(attempt_id, batch_key, workloads_json, started_at_ms, plan_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                attempt_id,
                batch_key,
                canonical_json(sorted(set(workloads))),
                _now_ms(),
                canonical_json(plan),
            ),
        )
        return attempt_id

    def recover_inflight(self) -> list[dict[str, Any]]:
        if self.read_only:
            raise snapshot_error("cannot recover an inflight batch through a read-only ledger")
        rows = self.conn.execute("SELECT * FROM inflight ORDER BY started_at_ms, attempt_id").fetchall()
        recovered: list[dict[str, Any]] = []
        if not rows:
            return recovered
        timestamp = _now_ms()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            for row in rows:
                workloads = json.loads(row["workloads_json"])
                for workload in workloads:
                    item = {
                        "workload": workload,
                        "attempt_id": row["attempt_id"],
                        "batch_key": row["batch_key"],
                        "reason": "uncommitted_batch_after_forced_termination",
                        "marked_at_ms": timestamp,
                    }
                    self.conn.execute(
                        """
                        INSERT OR REPLACE INTO unknown_workloads(
                            workload, attempt_id, batch_key, reason, marked_at_ms
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            item["workload"],
                            item["attempt_id"],
                            item["batch_key"],
                            item["reason"],
                            timestamp,
                        ),
                    )
                    recovered.append(item)
            self.conn.execute("DELETE FROM inflight")
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        return recovered

    def list_unknown_workloads(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM unknown_workloads ORDER BY workload, attempt_id"
        ).fetchall()
        return [dict(row) for row in rows]

    def latest_batch(self) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM batches ORDER BY event_cursor DESC, batch_id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return self._materialize_batch(row)

    def _batch_payload_from_row(
        self,
        row: sqlite3.Row,
        decisions: list[dict[str, Any]],
        workload_metrics: dict[str, Any],
        trace: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "ledger_format_version": LEDGER_FORMAT_VERSION,
            "snapshot_digest": row["snapshot_digest"],
            "event_cursor": row["event_cursor"],
            "first_event_seq": row["first_event_seq"],
            "batch_kind": row["batch_kind"],
            "declaration_format_versions": json.loads(row["format_versions_json"]),
            "defaults_version": row["defaults_version"],
            "scale_version": row["scale_version"],
            "scale_exp": row["scale_exp"],
            "decisions": decisions,
            "workload_metrics": workload_metrics,
            "trace": trace,
        }

    def _materialize_batch(self, row: sqlite3.Row) -> dict[str, Any]:
        decision_rows = self.conn.execute(
            "SELECT * FROM decisions WHERE batch_id = ? ORDER BY workload, field",
            (row["batch_id"],),
        ).fetchall()
        decisions = [
            {
                "workload": decision["workload"],
                "field": decision["field"],
                "target": json.loads(decision["target_json"]),
                "observed": json.loads(decision["observed_json"]),
                "observed_present": bool(decision["observed_present"]),
                "drift": bool(decision["drifted"]),
                "basis": decision["basis"],
                "reason": decision["reason"],
                "field_version": decision["field_version"],
                "fixed_metric": json.loads(decision["metric_json"]),
            }
            for decision in decision_rows
        ]
        decisions.sort(key=lambda d: (d["workload"], FIELD_ORDER.index(d["field"])))
        workload_metrics = json.loads(row["workload_metrics_json"])
        trace = json.loads(row["trace_json"])
        payload = self._batch_payload_from_row(row, decisions, workload_metrics, trace)
        return {
            "batch_id": row["batch_id"],
            "batch_key": row["batch_key"],
            "snapshot_digest": row["snapshot_digest"],
            "ledger_format_version": LEDGER_FORMAT_VERSION,
            "event_cursor": row["event_cursor"],
            "first_event_seq": row["first_event_seq"],
            "batch_kind": row["batch_kind"],
            "declaration_format_versions": json.loads(row["format_versions_json"]),
            "defaults_version": row["defaults_version"],
            "scale_version": row["scale_version"],
            "scale_exp": row["scale_exp"],
            "workload_metrics": workload_metrics,
            "trace": trace,
            "decisions": decisions,
            "content_digest": row["content_digest"],
            "created_at_ms": row["created_at_ms"],
            "content_payload": payload,
        }

    def read_batches(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM batches ORDER BY event_cursor, batch_id"
        ).fetchall()
        return [self._materialize_batch(row) for row in rows]

    def verify_integrity(self) -> dict[str, Any]:
        self._verify_schema_meta()
        snapshot = self.read_snapshot()
        batches = self.read_batches()
        for batch in batches:
            actual = content_digest(batch["content_payload"])
            if actual != batch["content_digest"]:
                raise snapshot_error(
                    "ledger batch content digest mismatch",
                    item=batch["batch_key"],
                    field="content_digest",
                    details={
                        "batch_id": batch["batch_id"],
                        "expected": batch["content_digest"],
                        "actual": actual,
                    },
                )
        return {
            "ok": True,
            "ledger_format_version": LEDGER_FORMAT_VERSION,
            "snapshot_digest": snapshot["snapshot_digest"] if snapshot else None,
            "batches_checked": len(batches),
            "decisions_checked": sum(len(batch["decisions"]) for batch in batches),
        }

    def commit_batch(
        self,
        *,
        event_cursor: int,
        first_event_seq: int,
        batch_kind: str,
        decisions: list[dict[str, Any]],
        workload_metrics: dict[str, Any],
        trace: dict[str, Any],
        attempt_id: str | None,
    ) -> dict[str, Any]:
        if self.read_only:
            raise snapshot_error("cannot commit a batch through a read-only ledger")
        decisions = sorted(
            decisions,
            key=lambda decision: (
                decision["workload"],
                FIELD_ORDER.index(decision["field"]),
            ),
        )
        snapshot = self.read_snapshot()
        if snapshot is None:
            raise snapshot_error("cannot commit a batch before a snapshot is installed")

        existing = self.conn.execute(
            "SELECT * FROM batches WHERE snapshot_digest = ? AND event_cursor = ?",
            (snapshot["snapshot_digest"], event_cursor),
        ).fetchone()
        if existing is not None:
            materialized = self._materialize_batch(existing)
            if canonical_json(
                {
                    "decisions": materialized["decisions"],
                    "workload_metrics": materialized["workload_metrics"],
                }
            ) != canonical_json(
                {"decisions": decisions, "workload_metrics": workload_metrics}
            ):
                raise snapshot_error(
                    "an existing batch has different decisive content",
                    item=materialized["batch_key"],
                    field="event_cursor",
                    details={"event_cursor": event_cursor},
                )
            return {**materialized, "idempotent": True}

        batch_key = make_batch_key(snapshot["snapshot_digest"], event_cursor)
        header_row = {
            "snapshot_digest": snapshot["snapshot_digest"],
            "event_cursor": event_cursor,
            "first_event_seq": first_event_seq,
            "batch_kind": batch_kind,
            "format_versions_json": canonical_json(snapshot["declaration_format_versions"]),
            "defaults_version": snapshot["defaults_version"],
            "scale_version": snapshot["scale_version"],
            "scale_exp": snapshot["scale_exp"],
        }
        payload = self._batch_payload_from_row(
            header_row, decisions, workload_metrics, trace  # type: ignore[arg-type]
        )
        digest = content_digest(payload)

        self.conn.execute("BEGIN IMMEDIATE")
        try:
            current = self.conn.execute("SELECT snapshot_digest FROM snapshot WHERE id = 1").fetchone()
            if current is None or current["snapshot_digest"] != snapshot["snapshot_digest"]:
                raise snapshot_error("snapshot changed while a batch was being judged")
            cursor = self.conn.execute(
                "INSERT INTO batches("
                "batch_key, snapshot_digest, ledger_version, event_cursor, first_event_seq, batch_kind,"
                "format_versions_json, defaults_version, scale_version, scale_exp,"
                "workload_metrics_json, trace_json, content_digest, created_at_ms"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    batch_key,
                    snapshot["snapshot_digest"],
                    LEDGER_FORMAT_VERSION,
                    event_cursor,
                    first_event_seq,
                    batch_kind,
                    canonical_json(snapshot["declaration_format_versions"]),
                    snapshot["defaults_version"],
                    snapshot["scale_version"],
                    snapshot["scale_exp"],
                    canonical_json(workload_metrics),
                    canonical_json(trace),
                    digest,
                    _now_ms(),
                ),
            )
            batch_id = int(cursor.lastrowid)
            for decision in decisions:
                self.conn.execute(
                    """
                    INSERT INTO decisions(
                        batch_id, workload, field, drifted, target_json, observed_json,
                        observed_present, basis, reason, field_version, metric_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        batch_id,
                        decision["workload"],
                        decision["field"],
                        int(decision["drift"]),
                        canonical_json(decision["target"]),
                        canonical_json(decision["observed"]),
                        int(decision["observed_present"]),
                        decision["basis"],
                        decision["reason"],
                        decision["field_version"],
                        canonical_json(decision["fixed_metric"]),
                    ),
                )
            if attempt_id:
                self.conn.execute("DELETE FROM inflight WHERE attempt_id = ?", (attempt_id,))
            workloads = sorted({decision["workload"] for decision in decisions})
            if workloads:
                placeholders = ",".join("?" for _ in workloads)
                self.conn.execute(
                    f"DELETE FROM unknown_workloads WHERE workload IN ({placeholders})",
                    workloads,
                )
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise

        return self._materialize_batch(
            self.conn.execute("SELECT * FROM batches WHERE batch_id = ?", (batch_id,)).fetchone()
        )

    def replacement_history(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM replacements ORDER BY replacement_id"
        ).fetchall()
        return [
            {
                **dict(row),
                "diff": json.loads(row["diff_json"]),
            }
            for row in rows
        ]

    def ledger_summary(self) -> dict[str, Any]:
        snapshot = self.read_snapshot()
        batches = self.read_batches()
        return {
            "ledger_path": str(self.path),
            "ledger_format_version": LEDGER_FORMAT_VERSION,
            "snapshot": None
            if snapshot is None
            else {
                key: snapshot[key]
                for key in (
                    "snapshot_digest",
                    "declaration_format_versions",
                    "declaration_digest",
                    "target_digest",
                    "defaults_version",
                    "scale_version",
                    "scale_exp",
                    "tolerance_num",
                    "tolerance_den",
                    "updated_at_ms",
                )
            },
            "batch_count": len(batches),
            "latest_event_cursor": batches[-1]["event_cursor"] if batches else None,
            "unknown_workloads": self.list_unknown_workloads(),
        }
