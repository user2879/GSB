from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping

from .canonical import canonical_json, digest_value, to_plain
from .errors import E_LEDGER, WorkbenchError
from .model import VersionState
from .errors import utc_timestamp


LEDGER_FORMAT_VERSION = 1


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ledger_format_version INTEGER NOT NULL,
    status TEXT NOT NULL,
    is_current INTEGER NOT NULL CHECK (is_current IN (0, 1)),
    started_at TEXT NOT NULL,
    completed_at TEXT,
    environment_version INTEGER NOT NULL,
    priority_version INTEGER NOT NULL,
    scale_version INTEGER NOT NULL,
    environment_fingerprint TEXT NOT NULL,
    priority_fingerprint TEXT NOT NULL,
    scale_fingerprint TEXT NOT NULL,
    spec_digest TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    expected_envs_json TEXT NOT NULL,
    stats_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS runs_current_idx ON runs(is_current, run_id DESC);

CREATE TABLE IF NOT EXISTS environment_records (
    run_id INTEGER NOT NULL,
    env_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    chain_digest TEXT NOT NULL,
    environment_version INTEGER NOT NULL,
    priority_version INTEGER NOT NULL,
    scale_version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, env_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS environment_progress (
    run_id INTEGER NOT NULL,
    env_id TEXT NOT NULL,
    status TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (run_id, env_id),
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TRIGGER IF NOT EXISTS environment_records_readonly_update
BEFORE UPDATE ON environment_records
BEGIN
    SELECT RAISE(ABORT, 'environment conclusions are append-only');
END;

CREATE TRIGGER IF NOT EXISTS environment_records_readonly_delete
BEFORE DELETE ON environment_records
BEGIN
    SELECT RAISE(ABORT, 'environment conclusions are append-only');
END;
"""


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.connection = sqlite3.connect(self.path)
            self.connection.row_factory = sqlite3.Row
            self.connection.execute("PRAGMA foreign_keys = ON")
            self.connection.execute("PRAGMA journal_mode = WAL")
            self.connection.execute("PRAGMA synchronous = FULL")
            with self.connection:
                self.connection.executescript(SCHEMA)
                self._ensure_meta()
        except sqlite3.DatabaseError as exc:
            raise WorkbenchError(E_LEDGER, f"cannot open ledger: {exc}", path="$") from exc
        self._validate_format()
        self.verify_record_digests()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _ensure_meta(self) -> None:
        existing = self.connection.execute(
            "SELECT value FROM meta WHERE key = 'ledger_format_version'"
        ).fetchone()
        if existing is None:
            self.connection.execute(
                "INSERT INTO meta(key, value) VALUES ('ledger_format_version', ?)",
                (str(LEDGER_FORMAT_VERSION),),
            )
        if self.connection.execute("SELECT 1 FROM meta WHERE key = 'version_state'").fetchone() is None:
            self.connection.execute(
                "INSERT INTO meta(key, value) VALUES ('version_state', '')"
            )
        if self.connection.execute("SELECT 1 FROM meta WHERE key = 'ledger_content_digest'").fetchone() is None:
            self.connection.execute(
                "INSERT INTO meta(key, value) VALUES ('ledger_content_digest', '')"
            )

    def _validate_format(self) -> None:
        row = self.connection.execute(
            "SELECT value FROM meta WHERE key = 'ledger_format_version'"
        ).fetchone()
        if row is None or int(row["value"]) != LEDGER_FORMAT_VERSION:
            raise WorkbenchError(
                E_LEDGER,
                "ledger format is not supported",
                path="$.ledger_format_version",
                details={"expected": LEDGER_FORMAT_VERSION, "actual": None if row is None else row["value"]},
            )

    def verify_record_digests(self) -> None:
        rows = self.connection.execute(
            "SELECT run_id, env_id, payload_json, content_digest FROM environment_records"
        ).fetchall()
        for row in rows:
            actual = digest_value(json.loads(row["payload_json"]))
            if actual != row["content_digest"]:
                raise WorkbenchError(
                    E_LEDGER,
                    "ledger content digest mismatch",
                    environment=row["env_id"],
                    path="$",
                    details={"run_id": row["run_id"], "expected": row["content_digest"], "actual": actual},
                )

    def _meta(self, key: str) -> str:
        row = self.connection.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return "" if row is None else str(row["value"])

    def stored_version_state(self) -> VersionState | None:
        text = self._meta("version_state")
        if not text:
            return None
        try:
            return VersionState.from_payload(json.loads(text))
        except (ValueError, KeyError, TypeError) as exc:
            raise WorkbenchError(E_LEDGER, "stored version state is corrupt", path="$.version_state") from exc

    def plan_versions(self, fingerprints: Mapping[str, str]) -> VersionState:
        stored = self.stored_version_state()
        if stored is None:
            return VersionState(
                environment_version=1,
                priority_version=1,
                scale_version=1,
                environment_fingerprint=fingerprints["environment"],
                priority_fingerprint=fingerprints["priority"],
                scale_fingerprint=fingerprints["scale"],
                spec_digest=fingerprints["content"],
            )
        return VersionState(
            environment_version=stored.environment_version + (
                stored.environment_fingerprint != fingerprints["environment"]
            ),
            priority_version=stored.priority_version + (
                stored.priority_fingerprint != fingerprints["priority"]
            ),
            scale_version=stored.scale_version + (
                stored.scale_fingerprint != fingerprints["scale"]
            ),
            environment_fingerprint=fingerprints["environment"],
            priority_fingerprint=fingerprints["priority"],
            scale_fingerprint=fingerprints["scale"],
            spec_digest=fingerprints["content"],
        )

    def apply_pending_versions(self, versions: VersionState) -> dict[str, Any]:
        """Persist a changed declaration version and retire the old current view.

        Environment records themselves are append-only; only the run-level
        freshness marker changes.  If this method is called while nothing has
        changed it is a no-op.
        """

        stored = self.stored_version_state()
        changed = stored is None or stored.payload() != versions.payload()
        if not changed:
            return {"changed": False, "invalidated_run_ids": []}
        invalidated = [
            int(row["run_id"])
            for row in self.connection.execute(
                "SELECT run_id FROM runs WHERE is_current = 1 ORDER BY run_id"
            ).fetchall()
        ]
        with self.connection:
            self.connection.execute(
                "INSERT INTO meta(key, value) VALUES ('version_state', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (canonical_json(versions.payload()),),
            )
            self.connection.execute("UPDATE runs SET is_current = 0 WHERE is_current = 1")
        return {"changed": True, "invalidated_run_ids": invalidated}

    def update_run_stats(self, run_id: int, stats: Mapping[str, Any]) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE runs SET stats_json = ? WHERE run_id = ?",
                (canonical_json(stats), run_id),
            )

    def current_run(self) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM runs WHERE is_current = 1 ORDER BY run_id DESC LIMIT 1"
        ).fetchone()

    def get_run(self, run_id: int) -> sqlite3.Row | None:
        return self.connection.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()

    def latest_completed_run(self) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM runs WHERE status = 'complete' ORDER BY run_id DESC LIMIT 1"
        ).fetchone()

    def begin_run(
        self,
        versions: VersionState,
        expected_envs: Iterable[str],
        stats: Mapping[str, Any],
    ) -> int:
        expected = sorted(expected_envs)
        now = utc_timestamp()
        with self.connection:
            self.connection.execute(
                "INSERT INTO meta(key, value) VALUES ('version_state', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (canonical_json(versions.payload()),),
            )
            self.connection.execute("UPDATE runs SET is_current = 0 WHERE is_current = 1")
            cursor = self.connection.execute(
                """
                INSERT INTO runs(
                    ledger_format_version, status, is_current, started_at,
                    environment_version, priority_version, scale_version,
                    environment_fingerprint, priority_fingerprint, scale_fingerprint,
                    spec_digest, content_digest, expected_envs_json, stats_json
                ) VALUES (?, 'running', 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    LEDGER_FORMAT_VERSION,
                    now,
                    versions.environment_version,
                    versions.priority_version,
                    versions.scale_version,
                    versions.environment_fingerprint,
                    versions.priority_fingerprint,
                    versions.scale_fingerprint,
                    versions.spec_digest,
                    versions.spec_digest,
                    canonical_json(expected),
                    canonical_json(stats),
                ),
            )
            run_id = int(cursor.lastrowid)
            for env_id in expected:
                self.connection.execute(
                    "INSERT INTO environment_progress(run_id, env_id, status, updated_at) VALUES (?, ?, 'pending', ?)",
                    (run_id, env_id, now),
                )
            return run_id

    def mark_computing(self, run_id: int, env_id: str) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE environment_progress SET status = 'computing', updated_at = ? WHERE run_id = ? AND env_id = ?",
                (utc_timestamp(), run_id, env_id),
            )

    def commit_environment(
        self,
        run_id: int,
        env_id: str,
        result: Mapping[str, Any],
        versions: VersionState,
        *,
        sequence: int,
    ) -> str:
        payload_json = canonical_json(result)
        content_digest = digest_value(result)
        chain_digest = str(result.get("source_chain_digest", ""))
        now = utc_timestamp()
        try:
            with self.connection:
                self.connection.execute(
                    """
                    INSERT INTO environment_records(
                        run_id, env_id, sequence, payload_json, content_digest,
                        chain_digest, environment_version, priority_version,
                        scale_version, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        env_id,
                        sequence,
                        payload_json,
                        content_digest,
                        chain_digest,
                        versions.environment_version,
                        versions.priority_version,
                        versions.scale_version,
                        now,
                    ),
                )
                self.connection.execute(
                    "UPDATE environment_progress SET status = 'committed', updated_at = ? WHERE run_id = ? AND env_id = ?",
                    (now, run_id, env_id),
                )
        except sqlite3.IntegrityError as exc:
            raise WorkbenchError(
                E_LEDGER,
                f"environment {env_id!r} already has a committed record in this run",
                environment=env_id,
                path="$",
                details={"run_id": run_id},
            ) from exc
        return content_digest

    def mark_environment_failed(self, run_id: int, env_id: str, message: str) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE environment_progress SET status = 'failed', updated_at = ? WHERE run_id = ? AND env_id = ?",
                (utc_timestamp(), run_id, env_id),
            )

    def finalize_run(self, run_id: int) -> dict[str, Any]:
        run = self.get_run(run_id)
        if run is None:
            raise WorkbenchError(E_LEDGER, "run does not exist", path="$", details={"run_id": run_id})
        rows = self.connection.execute(
            "SELECT env_id, content_digest, chain_digest FROM environment_records WHERE run_id = ? ORDER BY env_id",
            (run_id,),
        ).fetchall()
        completed = [str(row["env_id"]) for row in rows]
        expected = sorted(json.loads(run["expected_envs_json"]))
        status = "complete" if completed == expected else "incomplete"
        content_digest = digest_value(
            [
                {
                    "env_id": row["env_id"],
                    "content_digest": row["content_digest"],
                    "chain_digest": row["chain_digest"],
                }
                for row in rows
            ]
        )
        now = utc_timestamp()
        with self.connection:
            self.connection.execute(
                "UPDATE runs SET status = ?, completed_at = ?, content_digest = ? WHERE run_id = ?",
                (status, now, content_digest, run_id),
            )
            self.connection.execute(
                "UPDATE meta SET value = ? WHERE key = 'ledger_content_digest'",
                (content_digest,),
            )
        return {
            "run_id": run_id,
            "status": status,
            "completed_environments": completed,
            "content_digest": content_digest,
        }

    def fetch_run_environments(self, run_id: int) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT env_id, sequence, payload_json, content_digest, chain_digest FROM environment_records "
            "WHERE run_id = ? ORDER BY sequence, env_id",
            (run_id,),
        ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            actual = digest_value(payload)
            if actual != row["content_digest"]:
                raise WorkbenchError(
                    E_LEDGER,
                    "ledger record digest mismatch while reading",
                    environment=row["env_id"],
                    path="$",
                    details={"run_id": run_id},
                )
            result.append(
                {
                    "env_id": row["env_id"],
                    "sequence": row["sequence"],
                    "content_digest": row["content_digest"],
                    "chain_digest": row["chain_digest"],
                    "payload": payload,
                }
            )
        return result

    def current_view(self) -> dict[str, Any]:
        run = self.current_run()
        if run is None:
            return {
                "run": None,
                "completed_environments": [],
                "unknown_environments": [],
                "environments": [],
            }
        records = self.fetch_run_environments(int(run["run_id"]))
        completed = sorted(record["env_id"] for record in records)
        expected = sorted(json.loads(run["expected_envs_json"]))
        unknown_rows = self.connection.execute(
            "SELECT env_id, status FROM environment_progress WHERE run_id = ? AND status != 'committed' ORDER BY env_id",
            (run["run_id"],),
        ).fetchall()
        unknown = sorted({str(row["env_id"]) for row in unknown_rows} | (set(expected) - set(completed)))
        return {
            "run": self.run_payload(run),
            "completed_environments": completed,
            "unknown_environments": unknown,
            "environments": records,
        }

    def run_payload(self, run: sqlite3.Row) -> dict[str, Any]:
        return {
            "run_id": run["run_id"],
            "status": run["status"],
            "is_current": bool(run["is_current"]),
            "started_at": run["started_at"],
            "completed_at": run["completed_at"],
            "environment_version": run["environment_version"],
            "priority_version": run["priority_version"],
            "scale_version": run["scale_version"],
            "spec_digest": run["spec_digest"],
            "content_digest": run["content_digest"],
            "stats": json.loads(run["stats_json"]),
        }

    def list_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM runs ORDER BY run_id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self.run_payload(row) for row in rows]

    def recover_incomplete(self) -> dict[str, Any]:
        runs = self.connection.execute(
            "SELECT * FROM runs WHERE status = 'running' ORDER BY run_id"
        ).fetchall()
        recovered: list[dict[str, Any]] = []
        now = utc_timestamp()
        for run in runs:
            run_id = int(run["run_id"])
            expected = sorted(json.loads(run["expected_envs_json"]))
            committed_rows = self.connection.execute(
                "SELECT env_id FROM environment_records WHERE run_id = ? ORDER BY env_id", (run_id,)
            ).fetchall()
            committed = [str(row["env_id"]) for row in committed_rows]
            unknown = sorted(set(expected) - set(committed))
            content_rows = self.connection.execute(
                "SELECT env_id, content_digest, chain_digest FROM environment_records WHERE run_id = ? ORDER BY env_id",
                (run_id,),
            ).fetchall()
            partial_digest = digest_value(
                [
                    {
                        "env_id": row["env_id"],
                        "content_digest": row["content_digest"],
                        "chain_digest": row["chain_digest"],
                    }
                    for row in content_rows
                ]
            )
            with self.connection:
                for env_id in unknown:
                    self.connection.execute(
                        "INSERT INTO environment_progress(run_id, env_id, status, updated_at) VALUES (?, ?, 'unknown', ?) "
                        "ON CONFLICT(run_id, env_id) DO UPDATE SET status = 'unknown', updated_at = excluded.updated_at",
                        (run_id, env_id, now),
                    )
                self.connection.execute(
                    "UPDATE runs SET status = 'interrupted', completed_at = ?, content_digest = ? WHERE run_id = ?",
                    (now, partial_digest, run_id),
                )
            recovered.append(
                {
                    "run_id": run_id,
                    "completed_environments": committed,
                    "unknown_environments": unknown,
                }
            )
        return {
            "recovered_runs": recovered,
            "note": "Only fully committed environment records were recovered; interrupted environments require a rerun and no forced-exit report is claimed.",
        }
