from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable

from .errors import LedgerError, SnapshotError
from .util import canonical_sha256, canonical_text

FORMAT_VERSION = 1


class Ledger:
    """SQLite-backed arbitration ledger with one transaction per view commit."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path), timeout=30)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self._create_schema()
        self._initialize_meta()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    @staticmethod
    def _now_ms() -> int:
        return int(time.time() * 1000)

    def _create_schema(self) -> None:
        with self.connection:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS ledger_meta (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    format_version INTEGER NOT NULL,
                    content_digest TEXT NOT NULL,
                    environment_set_version INTEGER,
                    source_priority_version INTEGER,
                    scale_policy_version TEXT,
                    arbitration_policy_version TEXT,
                    updated_at_ms INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS views (
                    view_id TEXT PRIMARY KEY,
                    current_version INTEGER NOT NULL,
                    fingerprint TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_sha256 TEXT NOT NULL,
                    updated_at_ms INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS view_runs (
                    view_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    fingerprint TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    result_sha256 TEXT NOT NULL,
                    dependency_manifest_json TEXT NOT NULL,
                    stale_reason TEXT,
                    created_at_ms INTEGER NOT NULL,
                    PRIMARY KEY (view_id, version)
                );

                CREATE TABLE IF NOT EXISTS view_keys (
                    view_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    key TEXT NOT NULL,
                    stale INTEGER NOT NULL,
                    result_json TEXT NOT NULL,
                    result_sha256 TEXT NOT NULL,
                    dependency_json TEXT NOT NULL,
                    PRIMARY KEY (view_id, version, key)
                );

                CREATE TABLE IF NOT EXISTS stale_findings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    view_id TEXT NOT NULL,
                    old_version INTEGER NOT NULL,
                    new_version INTEGER NOT NULL,
                    key TEXT NOT NULL,
                    old_result_sha256 TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    trace_json TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_view_keys_view_version
                    ON view_keys(view_id, version, stale);
                CREATE INDEX IF NOT EXISTS idx_stale_findings_view
                    ON stale_findings(view_id, new_version);
                """
            )

    def _initialize_meta(self) -> None:
        row = self.connection.execute("SELECT * FROM ledger_meta WHERE singleton = 1").fetchone()
        if row is not None:
            if int(row["format_version"]) != FORMAT_VERSION:
                raise SnapshotError(
                    "unsupported ledger format version",
                    details={
                        "path": str(self.path),
                        "expected_format_version": FORMAT_VERSION,
                        "actual_format_version": int(row["format_version"]),
                    },
                )
            return
        digest = self._compute_content_digest(self.connection)
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO ledger_meta(
                    singleton, format_version, content_digest,
                    environment_set_version, source_priority_version,
                    scale_policy_version, arbitration_policy_version, updated_at_ms
                ) VALUES (1, ?, ?, NULL, NULL, NULL, NULL, ?)
                """,
                (FORMAT_VERSION, digest, self._now_ms()),
            )

    def _compute_content_digest(self, connection: sqlite3.Connection) -> str:
        records: list[dict[str, Any]] = []
        for row in connection.execute(
            """
            SELECT view_id, version, fingerprint, status, result_json, result_sha256,
                   dependency_manifest_json, stale_reason, created_at_ms
            FROM view_runs ORDER BY view_id, version
            """
        ):
            records.append(
                {
                    "table": "view_runs",
                    "view_id": row["view_id"],
                    "version": row["version"],
                    "fingerprint": row["fingerprint"],
                    "status": row["status"],
                    "result_json": row["result_json"],
                    "result_sha256": row["result_sha256"],
                    "dependency_manifest_json": row["dependency_manifest_json"],
                    "stale_reason": row["stale_reason"],
                    "created_at_ms": row["created_at_ms"],
                }
            )
        for row in connection.execute(
            """
            SELECT view_id, version, key, stale, result_json, result_sha256, dependency_json
            FROM view_keys ORDER BY view_id, version, key
            """
        ):
            records.append(
                {
                    "table": "view_keys",
                    "view_id": row["view_id"],
                    "version": row["version"],
                    "key": row["key"],
                    "stale": row["stale"],
                    "result_json": row["result_json"],
                    "result_sha256": row["result_sha256"],
                    "dependency_json": row["dependency_json"],
                }
            )
        for row in connection.execute(
            """
            SELECT view_id, old_version, new_version, key, old_result_sha256,
                   reason, trace_json, created_at_ms
            FROM stale_findings ORDER BY view_id, new_version, key, id
            """
        ):
            records.append(
                {
                    "table": "stale_findings",
                    "view_id": row["view_id"],
                    "old_version": row["old_version"],
                    "new_version": row["new_version"],
                    "key": row["key"],
                    "old_result_sha256": row["old_result_sha256"],
                    "reason": row["reason"],
                    "trace_json": row["trace_json"],
                    "created_at_ms": row["created_at_ms"],
                }
            )
        for row in connection.execute(
            "SELECT view_id, current_version, fingerprint, status, result_sha256, updated_at_ms FROM views ORDER BY view_id"
        ):
            records.append(
                {
                    "table": "views",
                    "view_id": row["view_id"],
                    "current_version": row["current_version"],
                    "fingerprint": row["fingerprint"],
                    "status": row["status"],
                    "result_sha256": row["result_sha256"],
                    "updated_at_ms": row["updated_at_ms"],
                }
            )
        return canonical_sha256(records)

    def verify_integrity(self, expected_digest: str | None = None) -> str:
        meta = self.meta()
        if int(meta["format_version"]) != FORMAT_VERSION:
            raise SnapshotError(
                "unsupported ledger format version",
                details={
                    "path": str(self.path),
                    "expected_format_version": FORMAT_VERSION,
                    "actual_format_version": int(meta["format_version"]),
                },
            )
        actual = self._compute_content_digest(self.connection)
        if expected_digest is not None and expected_digest != actual:
            raise SnapshotError(
                "injected ledger content digest mismatch",
                details={
                    "path": str(self.path),
                    "expected": expected_digest,
                    "actual": actual,
                    "stored": meta["content_digest"],
                    "injected": True,
                },
            )
        if actual != meta["content_digest"]:
            raise SnapshotError(
                "ledger content digest mismatch",
                details={
                    "path": str(self.path),
                    "expected": meta["content_digest"],
                    "actual": actual,
                    "injected": False,
                },
            )
        return actual

    def meta(self) -> dict[str, Any]:
        row = self.connection.execute("SELECT * FROM ledger_meta WHERE singleton = 1").fetchone()
        if row is None:
            raise LedgerError("ledger metadata is missing", details={"path": str(self.path)})
        return dict(row)

    def current_view(self, view_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT v.view_id, v.current_version, v.fingerprint, v.status, v.result_sha256,
                   r.result_json, r.dependency_manifest_json, r.created_at_ms
            FROM views AS v
            JOIN view_runs AS r ON r.view_id = v.view_id AND r.version = v.current_version
            WHERE v.view_id = ?
            """,
            (view_id,),
        ).fetchone()
        if row is None:
            return None
        value = dict(row)
        value["result"] = json.loads(value.pop("result_json"))
        value["dependencies"] = json.loads(value.pop("dependency_manifest_json"))
        return value

    def all_current_views(self) -> list[dict[str, Any]]:
        rows = self.connection.execute("SELECT view_id FROM views ORDER BY view_id").fetchall()
        return [current for row in rows if (current := self.current_view(row["view_id"])) is not None]

    def view_history(self, view_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT view_id, version, fingerprint, status, result_sha256, stale_reason, created_at_ms
            FROM view_runs WHERE view_id = ? ORDER BY version DESC
            """,
            (view_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def stale_findings(self, view_id: str | None = None, version: int | None = None) -> list[dict[str, Any]]:
        clauses: list[str] = []
        args: list[Any] = []
        if view_id is not None:
            clauses.append("view_id = ?")
            args.append(view_id)
        if version is not None:
            clauses.append("new_version = ?")
            args.append(version)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self.connection.execute(
            f"SELECT * FROM stale_findings{where} ORDER BY view_id, new_version, key, id",
            args,
        ).fetchall()
        results = []
        for row in rows:
            value = dict(row)
            value["trace"] = json.loads(value.pop("trace_json"))
            results.append(value)
        return results

    def key_rows(self, view_id: str, version: int) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT key, stale, result_json, result_sha256, dependency_json
            FROM view_keys WHERE view_id = ? AND version = ? ORDER BY key
            """,
            (view_id, version),
        ).fetchall()
        values = []
        for row in rows:
            value = dict(row)
            value["result"] = json.loads(value.pop("result_json"))
            value["dependency"] = json.loads(value.pop("dependency_json"))
            values.append(value)
        return values

    def commit_view(
        self,
        *,
        view_id: str,
        version: int,
        fingerprint: str,
        result: dict[str, Any],
        key_results: dict[str, dict[str, Any]],
        dependencies: dict[str, dict[str, Any]],
        stale_events: dict[str, dict[str, Any]],
        environment_set_version: int,
        source_priority_version: int,
        scale_policy_version: str,
        arbitration_policy_version: str,
    ) -> str:
        """Commit one complete view and all its key records atomically."""
        result_json = canonical_text(result)
        result_sha = canonical_sha256(result)
        dependency_manifest_json = canonical_text(dependencies)
        now = self._now_ms()
        previous = self.current_view(view_id)
        old_version = int(previous["current_version"]) if previous else None
        old_key_hashes = {
            row["key"]: row["result_sha256"] for row in self.key_rows(view_id, old_version)
        } if old_version is not None else {}

        try:
            self.connection.execute("BEGIN IMMEDIATE")
            self.connection.execute(
                "UPDATE view_runs SET status = 'stale', stale_reason = ? WHERE view_id = ? AND status = 'complete'",
                (f"superseded_by_version_{version}", view_id),
            )
            if old_version is not None:
                self.connection.execute(
                    "UPDATE view_keys SET stale = 1 WHERE view_id = ? AND version = ?",
                    (view_id, old_version),
                )
            self.connection.execute(
                """
                INSERT INTO view_runs(
                    view_id, version, fingerprint, status, result_json, result_sha256,
                    dependency_manifest_json, stale_reason, created_at_ms
                ) VALUES (?, ?, ?, 'complete', ?, ?, ?, NULL, ?)
                """,
                (view_id, version, fingerprint, result_json, result_sha, dependency_manifest_json, now),
            )
            for key in sorted(key_results):
                key_json = canonical_text(key_results[key])
                key_sha = canonical_sha256(key_results[key])
                dependency_json = canonical_text(dependencies[key])
                self.connection.execute(
                    """
                    INSERT INTO view_keys(
                        view_id, version, key, stale, result_json, result_sha256, dependency_json
                    ) VALUES (?, ?, ?, 0, ?, ?, ?)
                    """,
                    (view_id, version, key, key_json, key_sha, dependency_json),
                )
            for key in sorted(stale_events):
                trace = stale_events[key]
                self.connection.execute(
                    """
                    INSERT INTO stale_findings(
                        view_id, old_version, new_version, key, old_result_sha256,
                        reason, trace_json, created_at_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        view_id,
                        int(trace.get("old_version", old_version or version)),
                        version,
                        key,
                        str(trace.get("old_result_sha256", old_key_hashes.get(key, ""))),
                        str(trace.get("reason", "dependency_changed")),
                        canonical_text(trace),
                        now,
                    ),
                )
            self.connection.execute(
                """
                INSERT INTO views(view_id, current_version, fingerprint, status, result_sha256, updated_at_ms)
                VALUES (?, ?, ?, 'complete', ?, ?)
                ON CONFLICT(view_id) DO UPDATE SET
                    current_version = excluded.current_version,
                    fingerprint = excluded.fingerprint,
                    status = excluded.status,
                    result_sha256 = excluded.result_sha256,
                    updated_at_ms = excluded.updated_at_ms
                """,
                (view_id, version, fingerprint, result_sha, now),
            )
            digest = self._compute_content_digest(self.connection)
            self.connection.execute(
                """
                UPDATE ledger_meta SET
                    content_digest = ?,
                    environment_set_version = ?,
                    source_priority_version = ?,
                    scale_policy_version = ?,
                    arbitration_policy_version = ?,
                    updated_at_ms = ?
                WHERE singleton = 1
                """,
                (
                    digest,
                    environment_set_version,
                    source_priority_version,
                    scale_policy_version,
                    arbitration_policy_version,
                    now,
                ),
            )
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise
        return digest

    def reset_all(self) -> str:
        """Whole-ledger replacement used by snapshot restoration."""
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            for table in ("stale_findings", "view_keys", "view_runs", "views"):
                self.connection.execute(f"DELETE FROM {table}")
            digest = self._compute_content_digest(self.connection)
            self.connection.execute(
                """
                UPDATE ledger_meta SET
                    content_digest = ?,
                    environment_set_version = NULL,
                    source_priority_version = NULL,
                    scale_policy_version = NULL,
                    arbitration_policy_version = NULL,
                    updated_at_ms = ?
                WHERE singleton = 1
                """,
                (digest, self._now_ms()),
            )
            self.connection.commit()
        except BaseException:
            self.connection.rollback()
            raise
        return digest

    def set_context(
        self,
        *,
        environment_set_version: int,
        source_priority_version: int,
        scale_policy_version: str,
        arbitration_policy_version: str,
    ) -> None:
        with self.connection:
            self.connection.execute(
                """
                UPDATE ledger_meta SET
                    environment_set_version = ?,
                    source_priority_version = ?,
                    scale_policy_version = ?,
                    arbitration_policy_version = ?,
                    updated_at_ms = ?
                WHERE singleton = 1
                """,
                (
                    environment_set_version,
                    source_priority_version,
                    scale_policy_version,
                    arbitration_policy_version,
                    self._now_ms(),
                ),
            )

    def integrity_snapshot(self) -> dict[str, Any]:
        meta = self.meta()
        return {
            "format_version": int(meta["format_version"]),
            "content_digest": meta["content_digest"],
            "environment_set_version": meta["environment_set_version"],
            "source_priority_version": meta["source_priority_version"],
            "scale_policy_version": meta["scale_policy_version"],
            "arbitration_policy_version": meta["arbitration_policy_version"],
            "views": self.all_current_views(),
        }
