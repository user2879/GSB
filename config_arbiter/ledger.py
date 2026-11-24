from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .arbitration import KeyResult, ViewSummary
from .canonical import canonical_json, sha256_hex
from .errors import ReplayError

LEDGER_FORMAT_VERSION = 1


@dataclass(frozen=True)
class LedgerOpenReport:
    path: str
    replaced: bool
    differences: tuple[dict[str, str], ...]
    format_version: int
    content_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "replaced": self.replaced,
            "differences": list(self.differences),
            "format_version": self.format_version,
            "content_digest": self.content_digest,
        }


@dataclass(frozen=True)
class LedgerViewState:
    view_id: str
    version: int
    config_digest: str
    active: bool
    view_config_json: str
    environment_snapshot_json: str


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = self._open()
        self.open_report = self._validate_or_replace()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=5.0,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def _open(self) -> sqlite3.Connection:
        return self._connect()

    def _tables_present(self, connection: sqlite3.Connection) -> bool:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = {row[0] for row in rows}
        return {"meta", "views", "results", "propagation", "replacements"} <= names

    def _initialize_fresh(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS views (
                view_id TEXT PRIMARY KEY,
                version INTEGER NOT NULL,
                config_digest TEXT NOT NULL,
                view_config_json TEXT NOT NULL DEFAULT '{}',
                environment_snapshot_json TEXT NOT NULL DEFAULT '{}',
                active INTEGER NOT NULL CHECK (active IN (0, 1))
            );
            CREATE TABLE IF NOT EXISTS results (
                view_id TEXT NOT NULL,
                key TEXT NOT NULL,
                view_version INTEGER NOT NULL,
                status TEXT NOT NULL,
                summary_units INTEGER,
                summary_digest TEXT NOT NULL,
                adopted_json TEXT NOT NULL,
                rejected_json TEXT NOT NULL,
                bound_json TEXT NOT NULL,
                attribution_json TEXT NOT NULL,
                slice_versions_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                result_digest TEXT NOT NULL,
                is_stale INTEGER NOT NULL CHECK (is_stale IN (0, 1)),
                committed_monotonic_ms INTEGER NOT NULL,
                PRIMARY KEY (view_id, key, view_version)
            );
            CREATE TABLE IF NOT EXISTS propagation (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                view_id TEXT NOT NULL,
                previous_version INTEGER NOT NULL,
                new_version INTEGER NOT NULL,
                affected_keys_json TEXT NOT NULL,
                reasons_json TEXT NOT NULL,
                created_monotonic_ms INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS replacements (
                sequence INTEGER PRIMARY KEY,
                field TEXT NOT NULL,
                previous TEXT NOT NULL,
                current TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS results_current_idx
                ON results(view_id, is_stale, key);
            """
        )
        self.connection.execute("PRAGMA user_version = 1")
        self.connection.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('format_version', ?)",
            (str(LEDGER_FORMAT_VERSION),),
        )
        digest = self._compute_content_digest(self.connection)
        self.connection.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('content_digest', ?)",
            (digest,),
        )

    def _validate_snapshot(self) -> list[dict[str, str]]:
        differences: list[dict[str, str]] = []
        if not self._tables_present(self.connection):
            differences.append(
                {
                    "field": "schema",
                    "previous": "missing_or_incomplete",
                    "current": "config-arbiter-ledger-v1",
                }
            )
            return differences
        user_version = int(self.connection.execute("PRAGMA user_version").fetchone()[0])
        if user_version != LEDGER_FORMAT_VERSION:
            differences.append(
                {
                    "field": "format_version",
                    "previous": str(user_version),
                    "current": str(LEDGER_FORMAT_VERSION),
                }
            )
        meta_version_row = self.connection.execute(
            "SELECT value FROM meta WHERE key='format_version'"
        ).fetchone()
        meta_version = meta_version_row[0] if meta_version_row else "missing"
        if meta_version != str(LEDGER_FORMAT_VERSION):
            differences.append(
                {
                    "field": "meta.format_version",
                    "previous": str(meta_version),
                    "current": str(LEDGER_FORMAT_VERSION),
                }
            )
        digest_row = self.connection.execute(
            "SELECT value FROM meta WHERE key='content_digest'"
        ).fetchone()
        stored_digest = digest_row[0] if digest_row else "missing"
        computed_digest = self._compute_content_digest(self.connection)
        if stored_digest != computed_digest:
            differences.append(
                {
                    "field": "content_digest",
                    "previous": str(stored_digest),
                    "current": computed_digest,
                }
            )
        return differences

    def _next_replacement_path(self) -> Path:
        index = 1
        while True:
            candidate = self.path.with_name(f"{self.path.name}.replaced.{index}")
            if not candidate.exists():
                return candidate
            index += 1

    def _replace_invalid(
        self, differences: list[dict[str, str]]
    ) -> tuple[bool, tuple[dict[str, str], ...]]:
        self.connection.close()
        destination_base = self._next_replacement_path()
        for suffix in ("", "-wal", "-shm"):
            source = Path(str(self.path) + suffix)
            if source.exists():
                destination = Path(str(destination_base) + suffix)
                try:
                    os.replace(source, destination)
                except OSError:
                    if source.exists():
                        source.unlink()
        self.connection = self._connect()
        self._initialize_fresh()
        stable = tuple(
            {
                "sequence": str(index),
                "field": difference["field"],
                "previous": difference["previous"],
                "current": difference["current"],
            }
            for index, difference in enumerate(differences, start=1)
        )
        for item in stable:
            self.connection.execute(
                "INSERT INTO replacements(sequence, field, previous, current) VALUES(?,?,?,?)",
                (
                    int(item["sequence"]),
                    item["field"],
                    item["previous"],
                    item["current"],
                ),
            )
        self._refresh_content_digest(self.connection)
        self.connection.commit()
        return True, stable

    def _validate_or_replace(self) -> LedgerOpenReport:
        # A database created by this constructor is either brand new (no schema),
        # or must pass both integer format and content-digest checks.
        try:
            differences = self._validate_snapshot()
        except (sqlite3.DatabaseError, ValueError, TypeError) as exc:
            differences = [
                {
                    "field": "schema",
                    "previous": f"unreadable:{type(exc).__name__}",
                    "current": "config-arbiter-ledger-v1",
                }
            ]
        replaced = False
        stable_differences: tuple[dict[str, str], ...] = ()
        if differences:
            raw_tables = self.connection.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
            ).fetchone()[0]
            if (
                differences
                == [
                    {
                        "field": "schema",
                        "previous": "missing_or_incomplete",
                        "current": "config-arbiter-ledger-v1",
                    }
                ]
                and raw_tables == 0
            ):
                self._initialize_fresh()
            else:
                replaced, stable_differences = self._replace_invalid(differences)
        else:
            # Ensure the maintenance metadata is present without changing content rows.
            self.connection.execute("PRAGMA user_version = 1")
        format_version = self.get_format_version()
        content_digest = self.get_content_digest()
        return LedgerOpenReport(
            path=str(self.path),
            replaced=replaced,
            differences=stable_differences,
            format_version=format_version,
            content_digest=content_digest,
        )

    def _compute_content_digest(self, connection: sqlite3.Connection) -> str:
        manifest: dict[str, Any] = {
            "schema": "config-arbiter-ledger-v1",
            "format_version": LEDGER_FORMAT_VERSION,
        }
        if self._tables_present(connection):
            manifest["views"] = [
                dict(row)
                for row in connection.execute(
                    "SELECT view_id, version, config_digest, view_config_json, "
                    "environment_snapshot_json, active "
                    "FROM views ORDER BY view_id"
                )
            ]
            manifest["results"] = [
                dict(row)
                for row in connection.execute(
                    "SELECT view_id, key, view_version, status, summary_units, "
                    "summary_digest, adopted_json, rejected_json, bound_json, "
                    "attribution_json, slice_versions_json, result_json, "
                    "result_digest, is_stale, committed_monotonic_ms "
                    "FROM results ORDER BY view_id, key, view_version"
                )
            ]
            manifest["propagation"] = [
                dict(row)
                for row in connection.execute(
                    "SELECT sequence, view_id, previous_version, new_version, "
                    "affected_keys_json, reasons_json, created_monotonic_ms "
                    "FROM propagation ORDER BY sequence"
                )
            ]
            manifest["replacements"] = [
                dict(row)
                for row in connection.execute(
                    "SELECT sequence, field, previous, current "
                    "FROM replacements ORDER BY sequence"
                )
            ]
        return sha256_hex(manifest)

    def _refresh_content_digest(self, connection: sqlite3.Connection) -> str:
        digest = self._compute_content_digest(connection)
        connection.execute(
            "INSERT INTO meta(key, value) VALUES('content_digest', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (digest,),
        )
        return digest

    def get_format_version(self) -> int:
        row = self.connection.execute(
            "SELECT value FROM meta WHERE key='format_version'"
        ).fetchone()
        if row is None:
            return -1
        try:
            return int(row[0])
        except (TypeError, ValueError):
            return -1

    def get_content_digest(self) -> str:
        row = self.connection.execute(
            "SELECT value FROM meta WHERE key='content_digest'"
        ).fetchone()
        return str(row[0]) if row else "missing"

    def view_state(self, view_id: str) -> LedgerViewState | None:
        row = self.connection.execute(
            "SELECT view_id, version, config_digest, view_config_json, "
            "environment_snapshot_json, active FROM views WHERE view_id=?",
            (view_id,),
        ).fetchone()
        if row is None:
            return None
        return LedgerViewState(
            view_id=row["view_id"],
            version=int(row["version"]),
            config_digest=row["config_digest"],
            active=bool(row["active"]),
            view_config_json=row["view_config_json"],
            environment_snapshot_json=row["environment_snapshot_json"],
        )

    def view_states(self) -> dict[str, LedgerViewState]:
        rows = self.connection.execute(
            "SELECT view_id, version, config_digest, view_config_json, "
            "environment_snapshot_json, active FROM views ORDER BY view_id"
        ).fetchall()
        return {
            row["view_id"]: LedgerViewState(
                view_id=row["view_id"],
                version=int(row["version"]),
                config_digest=row["config_digest"],
                active=bool(row["active"]),
                view_config_json=row["view_config_json"],
                environment_snapshot_json=row["environment_snapshot_json"],
            )
            for row in rows
        }

    def _insert_propagation(
        self,
        connection: sqlite3.Connection,
        *,
        view_id: str,
        previous_version: int,
        new_version: int,
        affected_keys: Iterable[str],
        reasons: Iterable[dict[str, Any]],
        created_monotonic_ms: int,
    ) -> None:
        connection.execute(
            "INSERT INTO propagation(view_id, previous_version, new_version, "
            "affected_keys_json, reasons_json, created_monotonic_ms) VALUES(?,?,?,?,?,?)",
            (
                view_id,
                previous_version,
                new_version,
                canonical_json(sorted(set(affected_keys))),
                canonical_json(list(reasons)),
                created_monotonic_ms,
            ),
        )

    def commit_view(
        self,
        summary: ViewSummary,
        *,
        config_digest: str,
        view_config_json: str = "{}",
        environment_snapshot_json: str = "{}",
        propagation: dict[str, Any] | None = None,
        active: bool = True,
    ) -> str:
        connection = self.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                "INSERT INTO views(view_id, version, config_digest, view_config_json, "
                "environment_snapshot_json, active) VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(view_id) DO UPDATE SET version=excluded.version, "
                "config_digest=excluded.config_digest, "
                "view_config_json=excluded.view_config_json, "
                "environment_snapshot_json=excluded.environment_snapshot_json, "
                "active=excluded.active",
                (
                    summary.view_id,
                    summary.version,
                    config_digest,
                    view_config_json,
                    environment_snapshot_json,
                    1 if active else 0,
                ),
            )
            connection.execute(
                "UPDATE results SET is_stale=1 WHERE view_id=?", (summary.view_id,)
            )
            if active:
                committed_ms = time.monotonic_ns() // 1_000_000
                for result in summary.key_results:
                    connection.execute(
                        "INSERT INTO results(view_id, key, view_version, status, summary_units, "
                        "summary_digest, adopted_json, rejected_json, bound_json, "
                        "attribution_json, slice_versions_json, result_json, result_digest, "
                        "is_stale, committed_monotonic_ms) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,0,?) "
                        "ON CONFLICT(view_id, key, view_version) DO UPDATE SET "
                        "status=excluded.status, summary_units=excluded.summary_units, "
                        "summary_digest=excluded.summary_digest, adopted_json=excluded.adopted_json, "
                        "rejected_json=excluded.rejected_json, bound_json=excluded.bound_json, "
                        "attribution_json=excluded.attribution_json, "
                        "slice_versions_json=excluded.slice_versions_json, "
                        "result_json=excluded.result_json, result_digest=excluded.result_digest, "
                        "is_stale=0, committed_monotonic_ms=excluded.committed_monotonic_ms",
                        (
                            result.view_id,
                            result.key,
                            summary.version,
                            result.status,
                            result.summary_units,
                            result.summary_digest,
                            result.adopted_json,
                            result.rejected_json,
                            result.bound_json,
                            result.attribution_json,
                            result.slice_versions_json,
                            result.result_json,
                            result.result_digest,
                            committed_ms,
                        ),
                    )
            if propagation is not None:
                self._insert_propagation(
                    connection,
                    view_id=summary.view_id,
                    previous_version=int(propagation.get("previous_version", summary.version)),
                    new_version=int(propagation.get("new_version", summary.version)),
                    affected_keys=propagation.get("affected_keys", []),
                    reasons=propagation.get("reasons", []),
                    created_monotonic_ms=time.monotonic_ns() // 1_000_000,
                )
            digest = self._refresh_content_digest(connection)
            connection.commit()
            return digest
        except Exception:
            connection.rollback()
            raise

    def mark_view_removed(
        self,
        *,
        view_id: str,
        previous_version: int,
        removed_version: int,
        config_digest: str,
        affected_keys: Iterable[str],
        reasons: Iterable[dict[str, Any]],
    ) -> str:
        connection = self.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                "INSERT INTO views(view_id, version, config_digest, active) VALUES(?,?,?,0) "
                "ON CONFLICT(view_id) DO UPDATE SET version=excluded.version, "
                "config_digest=excluded.config_digest, active=0",
                (view_id, removed_version, config_digest),
            )
            connection.execute("UPDATE results SET is_stale=1 WHERE view_id=?", (view_id,))
            self._insert_propagation(
                connection,
                view_id=view_id,
                previous_version=previous_version,
                new_version=removed_version,
                affected_keys=affected_keys,
                reasons=reasons,
                created_monotonic_ms=time.monotonic_ns() // 1_000_000,
            )
            digest = self._refresh_content_digest(connection)
            connection.commit()
            return digest
        except Exception:
            connection.rollback()
            raise

    def current_results(self, view_id: str | None = None) -> list[sqlite3.Row]:
        if view_id is None:
            return self.connection.execute(
                "SELECT * FROM results WHERE is_stale=0 ORDER BY view_id, key"
            ).fetchall()
        return self.connection.execute(
            "SELECT * FROM results WHERE is_stale=0 AND view_id=? ORDER BY key",
            (view_id,),
        ).fetchall()

    def all_results(self) -> list[sqlite3.Row]:
        return self.connection.execute(
            "SELECT * FROM results ORDER BY view_id, key, view_version"
        ).fetchall()

    def propagations(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT sequence, view_id, previous_version, new_version, "
            "affected_keys_json, reasons_json, created_monotonic_ms "
            "FROM propagation ORDER BY sequence"
        ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            result.append(
                {
                    "sequence": int(row["sequence"]),
                    "view_id": row["view_id"],
                    "previous_version": int(row["previous_version"]),
                    "new_version": int(row["new_version"]),
                    "affected_keys": json_load(row["affected_keys_json"]),
                    "reasons": json_load(row["reasons_json"]),
                    "created_monotonic_ms": int(row["created_monotonic_ms"]),
                }
            )
        return result

    def replay(
        self,
        expected: Mapping[tuple[str, str], KeyResult],
        view_versions: Mapping[str, int] | None = None,
    ) -> dict[str, Any]:
        actual_rows = self.current_results()
        actual = {(row["view_id"], row["key"]): row for row in actual_rows}
        all_keys = sorted(set(actual) | set(expected))
        compared_fields = 0
        for key in all_keys:
            actual_row = actual.get(key)
            expected_result = expected.get(key)
            verify_key = f"{key[0]}/{key[1]}"
            if actual_row is None:
                raise ReplayError(
                    view_id=key[0],
                    key=verify_key,
                    field="result.exists",
                    message="ledger is missing a committed result",
                    expected="present",
                    actual="absent",
                )
            if expected_result is None:
                raise ReplayError(
                    view_id=key[0],
                    key=verify_key,
                    field="result.exists",
                    message="ledger contains a current result absent from replay computation",
                    expected="absent",
                    actual="present",
                )
            field_checks: list[tuple[str, Any, Any]] = []
            if view_versions is not None and key[0] in view_versions:
                field_checks.append(
                    (
                        "view_version",
                        int(actual_row["view_version"]),
                        int(view_versions[key[0]]),
                    )
                )
            field_checks.extend([
                ("status", actual_row["status"], expected_result.status),
                ("summary_units", actual_row["summary_units"], expected_result.summary_units),
                ("summary_digest", actual_row["summary_digest"], expected_result.summary_digest),
                ("adopted", actual_row["adopted_json"], expected_result.adopted_json),
                ("rejected", actual_row["rejected_json"], expected_result.rejected_json),
                ("error_bound", actual_row["bound_json"], expected_result.bound_json),
                ("attribution", actual_row["attribution_json"], expected_result.attribution_json),
                ("slice_versions", actual_row["slice_versions_json"], expected_result.slice_versions_json),
                ("result_digest", actual_row["result_digest"], expected_result.result_digest),
            ])
            for field_name, actual_value, expected_value in field_checks:
                compared_fields += 1
                if actual_value != expected_value:
                    raise ReplayError(
                        view_id=key[0],
                        key=verify_key,
                        field=field_name,
                        message=f"first replay mismatch is field {field_name!r}",
                        expected=expected_value,
                        actual=actual_value,
                    )
        return {
            "status": "OK",
            "view_key_count": len(all_keys),
            "field_comparisons": compared_fields,
            "content_digest": self.get_content_digest(),
        }

    def snapshot_report(self) -> dict[str, Any]:
        views = {
            row["view_id"]: {
                "version": int(row["version"]),
                "active": bool(row["active"]),
                "config_digest": row["config_digest"],
            }
            for row in self.connection.execute(
                "SELECT view_id, version, config_digest, view_config_json, "
                "environment_snapshot_json, active FROM views ORDER BY view_id"
            )
        }
        counts = {
            table: int(self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in ("views", "results", "propagation", "replacements")
        }
        return {
            "format_version": self.get_format_version(),
            "content_digest": self.get_content_digest(),
            "user_version": int(self.connection.execute("PRAGMA user_version").fetchone()[0]),
            "views": views,
            "counts": counts,
            "open_report": self.open_report.to_dict(),
        }

    def close(self) -> None:
        try:
            self.connection.close()
        except sqlite3.Error:
            pass

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def json_load(value: str) -> Any:
    import json

    return json.loads(value)
