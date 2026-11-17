"""SQLite-backed session snapshots, whole-state restore and replay checks."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .codec import parse_library, parse_record
from .errors import FormatError, ReplayError, StateError, TrackerError
from .metrics import record_metric_digest, replay_record
from .models import AGGREGATION_VERSION, Library
from .util import canonical_json, sha256_json

SNAPSHOT_SCHEMA_VERSION = 1


class SnapshotStore:
    def __init__(self, workspace: Path, database_path: Path | None = None) -> None:
        self.workspace = workspace.resolve()
        self.database_path = (
            database_path.resolve()
            if database_path is not None
            else self.workspace / "snapshots.sqlite3"
        )
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS snapshots (
                    snapshot_id INTEGER PRIMARY KEY,
                    label TEXT NOT NULL,
                    library_id TEXT NOT NULL,
                    library_json TEXT NOT NULL,
                    library_digest TEXT NOT NULL,
                    format_versions_json TEXT NOT NULL,
                    source_format_versions_json TEXT NOT NULL,
                    aggregation_versions_json TEXT NOT NULL,
                    checkpoint_path TEXT NOT NULL,
                    checkpoint_position INTEGER NOT NULL,
                    checkpoint_json TEXT,
                    run_count INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS snapshot_runs (
                    snapshot_id INTEGER NOT NULL,
                    run_id TEXT NOT NULL,
                    run_version INTEGER NOT NULL,
                    is_current INTEGER NOT NULL,
                    record_format_version INTEGER NOT NULL,
                    source_format_version INTEGER NOT NULL,
                    aggregation_name TEXT NOT NULL,
                    aggregation_version TEXT NOT NULL,
                    content_digest TEXT NOT NULL,
                    metric_digest TEXT,
                    record_json TEXT NOT NULL,
                    PRIMARY KEY (snapshot_id, run_id, run_version),
                    FOREIGN KEY (snapshot_id) REFERENCES snapshots(snapshot_id)
                        ON DELETE CASCADE
                );
                """
            )
            connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
                ("schema_version", str(SNAPSHOT_SCHEMA_VERSION)),
            )

    def capture(
        self,
        library: Library,
        checkpoint_view: dict[str, Any] | None,
        *,
        label: str,
    ) -> dict[str, Any]:
        if not isinstance(label, str) or not label.strip():
            raise StateError("snapshot label must be a non-empty string")
        envelope = library.to_envelope()
        library_json = canonical_json(envelope)
        library_digest = sha256_json(envelope)
        checkpoint_path = ""
        checkpoint_position = 0
        checkpoint_json: str | None = None
        if checkpoint_view is not None:
            checkpoint_path = str(checkpoint_view["path"])
            checkpoint_content = checkpoint_view["content"]
            checkpoint_position = int(checkpoint_content.get("position", 0))
            checkpoint_json = canonical_json(checkpoint_content)

        records: list[tuple[Any, ...]] = []
        current_records = library.sorted_runs()
        for current in current_records:
            entries = [(True, current)]
            entries.extend(
                (False, historical) for historical in library.history_for(current.run_id)
            )
            for is_current, record in entries:
                records.append(
                    (
                        record.run_id,
                        record.run_version,
                        int(is_current),
                        record.format_version,
                        record.provenance.source_format_version,
                        record.aggregation.name,
                        record.aggregation.version,
                        record.content_digest(),
                        record_metric_digest(record),
                        canonical_json(record.to_normalized_dict()),
                    )
                )

        with self._connect() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(snapshot_id), 0) + 1 FROM snapshots"
            ).fetchone()
            snapshot_id = int(row[0])
            connection.execute(
                """
                INSERT INTO snapshots(
                    snapshot_id, label, library_id, library_json, library_digest,
                    format_versions_json, source_format_versions_json,
                    aggregation_versions_json, checkpoint_path,
                    checkpoint_position, checkpoint_json, run_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    label.strip(),
                    library.library_id,
                    library_json,
                    library_digest,
                    canonical_json(library.format_versions()),
                    canonical_json(library.source_format_versions()),
                    canonical_json(library.aggregation_versions()),
                    checkpoint_path,
                    checkpoint_position,
                    checkpoint_json,
                    len(current_records),
                ),
            )
            connection.executemany(
                """
                INSERT INTO snapshot_runs(
                    snapshot_id, run_id, run_version, is_current,
                    record_format_version, source_format_version,
                    aggregation_name, aggregation_version, content_digest,
                    metric_digest, record_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [(snapshot_id, *record) for record in records],
            )
        return self.get(snapshot_id)

    def list(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM snapshots ORDER BY snapshot_id ASC"
            ).fetchall()
        return [self._summary_from_row(row) for row in rows]

    def get(self, snapshot_id: int) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM snapshots WHERE snapshot_id = ?", (snapshot_id,)
            ).fetchone()
        if row is None:
            raise StateError(
                "snapshot does not exist",
                snapshot_id=snapshot_id,
                path=str(self.database_path),
            )
        return self._summary_from_row(row)

    def load_library(self, snapshot_id: int) -> Library:
        row = self._raw_snapshot(snapshot_id)
        try:
            raw = json.loads(row["library_json"])
        except json.JSONDecodeError as exc:
            raise FormatError(
                f"snapshots.sqlite3.snapshot[{snapshot_id}].library_json",
                f"invalid snapshot library JSON: {exc.msg}",
            ) from exc
        return parse_library(raw, path=f"snapshot[{snapshot_id}].library")

    def checkpoint_artifact(self, snapshot_id: int) -> dict[str, Any]:
        row = self._raw_snapshot(snapshot_id)
        content = None
        if row["checkpoint_json"] is not None:
            try:
                content = json.loads(row["checkpoint_json"])
            except json.JSONDecodeError as exc:
                raise FormatError(
                    f"snapshots.sqlite3.snapshot[{snapshot_id}].checkpoint_json",
                    f"invalid checkpoint JSON: {exc.msg}",
                ) from exc
        return {
            "path": row["checkpoint_path"],
            "position": int(row["checkpoint_position"]),
            "content": content,
        }

    def diff_against_disk(
        self, snapshot_id: int, current_library: Library
    ) -> dict[str, Any]:
        snapshot = self.get(snapshot_id)
        snapshot_digests = self.current_content_digests(snapshot_id)
        disk_digests = current_library.content_digests()
        added = sorted(set(snapshot_digests) - set(disk_digests))
        removed = sorted(set(disk_digests) - set(snapshot_digests))
        changed = []
        unchanged = []
        snapshot_versions = self.current_versions(snapshot_id)
        disk_versions = {
            record.run_id: record.run_version for record in current_library.sorted_runs()
        }
        for run_id in sorted(set(snapshot_digests) & set(disk_digests)):
            if snapshot_digests[run_id] != disk_digests[run_id]:
                changed.append(
                    {
                        "run_id": run_id,
                        "disk_run_version": disk_versions[run_id],
                        "snapshot_run_version": snapshot_versions[run_id],
                        "disk_digest": disk_digests[run_id],
                        "snapshot_digest": snapshot_digests[run_id],
                    }
                )
            else:
                unchanged.append(run_id)

        snapshot_all = self.all_content_digests(snapshot_id)
        disk_all = {
            record.key: record.content_digest() for record in current_library.all_records()
        }
        history_changes: list[dict[str, Any]] = []
        for key in sorted(set(snapshot_all) | set(disk_all)):
            if snapshot_all.get(key) == disk_all.get(key):
                continue
            run_id, version_text = key.rsplit("@v", 1)
            if run_id in snapshot_digests and int(version_text) == snapshot_versions.get(run_id):
                continue
            if run_id in disk_digests and int(version_text) == disk_versions.get(run_id):
                continue
            history_changes.append(
                {
                    "key": key,
                    "run_id": run_id,
                    "run_version": int(version_text),
                    "disk_digest": disk_all.get(key),
                    "snapshot_digest": snapshot_all.get(key),
                }
            )
        return {
            "snapshot_id": snapshot_id,
            "library_id": snapshot["library_id"],
            "mode": "whole_replace",
            "added_from_snapshot": added,
            "removed_from_current": removed,
            "changed": changed,
            "unchanged": unchanged,
            "history_changes": history_changes,
            "is_different": bool(added or removed or changed or history_changes),
        }

    def current_content_digests(self, snapshot_id: int) -> dict[str, str]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT run_id, content_digest FROM snapshot_runs
                WHERE snapshot_id = ? AND is_current = 1 ORDER BY run_id
                """,
                (snapshot_id,),
            ).fetchall()
        return {row["run_id"]: row["content_digest"] for row in rows}

    def current_versions(self, snapshot_id: int) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT run_id, run_version FROM snapshot_runs
                WHERE snapshot_id = ? AND is_current = 1 ORDER BY run_id
                """,
                (snapshot_id,),
            ).fetchall()
        return {row["run_id"]: int(row["run_version"]) for row in rows}

    def all_content_digests(self, snapshot_id: int) -> dict[str, str]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT run_id, run_version, content_digest FROM snapshot_runs
                WHERE snapshot_id = ? ORDER BY run_id, run_version
                """,
                (snapshot_id,),
            ).fetchall()
        return {
            f"{row['run_id']}@v{int(row['run_version'])}": row["content_digest"]
            for row in rows
        }

    def replay(self, snapshot_id: int) -> dict[str, Any]:
        snapshot = self.get(snapshot_id)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM snapshot_runs
                WHERE snapshot_id = ? AND is_current = 1 ORDER BY run_id
                """,
                (snapshot_id,),
            ).fetchall()
        if not rows:
            raise ReplayError(
                "<snapshot>",
                f"snapshot_id={snapshot_id}",
                "snapshot contains no current run records",
            )

        results: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        for row in rows:
            run_id = row["run_id"]
            run_version = int(row["run_version"])
            raw_version = (
                f"run_version={run_version},format_version={row['record_format_version']},"
                f"aggregation_version={row['aggregation_version']}"
            )
            try:
                raw = json.loads(row["record_json"])
                if not isinstance(raw, dict):
                    raise ReplayError(run_id, raw_version, "snapshot record is not an object")
                aggregation = raw.get("aggregation")
                if not isinstance(aggregation, dict):
                    raise ReplayError(
                        run_id,
                        raw_version,
                        "aggregation version is missing from the snapshot record",
                    )
                aggregation_version = aggregation.get("version")
                if aggregation_version != AGGREGATION_VERSION:
                    raise ReplayError(
                        run_id,
                        raw_version,
                        f"aggregation version {aggregation_version!r} cannot be reproduced by this build",
                    )
                digest = row["metric_digest"]
                if not digest:
                    raise ReplayError(
                        run_id,
                        raw_version,
                        "metric summary is missing from the snapshot",
                    )
                record = parse_record(raw, path=f"snapshot[{snapshot_id}].runs.{run_id}")
                results.append(replay_record(record, digest))
            except ReplayError as exc:
                failures.append(
                    {
                        "run_id": exc.details.get("run_id", run_id),
                        "version": exc.details.get("version", raw_version),
                        "basis": exc.details.get("basis", exc.message),
                        "details": exc.details,
                    }
                )
            except (TrackerError, json.JSONDecodeError, TypeError, ValueError) as exc:
                details = getattr(exc, "details", {})
                failures.append(
                    {
                        "run_id": run_id,
                        "version": raw_version,
                        "basis": str(exc),
                        "details": details,
                    }
                )
        if failures:
            first = failures[0]
            first_details = {
                key: value
                for key, value in first.get("details", {}).items()
                if key not in {"run_id", "version", "basis"}
            }
            raise ReplayError(
                first["run_id"],
                first["version"],
                first["basis"],
                failures=failures,
                passed_runs=results,
                **first_details,
            )
        return {
            "snapshot_id": snapshot_id,
            "snapshot_label": snapshot["label"],
            "status": "pass",
            "mode": "regenerate_and_compare",
            "results": results,
        }

    def _raw_snapshot(self, snapshot_id: int) -> sqlite3.Row:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM snapshots WHERE snapshot_id = ?", (snapshot_id,)
            ).fetchone()
        if row is None:
            raise StateError(
                "snapshot does not exist",
                snapshot_id=snapshot_id,
                path=str(self.database_path),
            )
        return row

    def _summary_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "snapshot_id": int(row["snapshot_id"]),
            "label": row["label"],
            "library_id": row["library_id"],
            "library_digest": row["library_digest"],
            "format_versions": json.loads(row["format_versions_json"]),
            "source_format_versions": json.loads(row["source_format_versions_json"]),
            "aggregation_versions": json.loads(row["aggregation_versions_json"]),
            "checkpoint_path": row["checkpoint_path"],
            "checkpoint_position": int(row["checkpoint_position"]),
            "run_count": int(row["run_count"]),
        }
