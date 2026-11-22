from __future__ import annotations

import json
import hashlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .canonical import canonical_json, sha256_canonical
from .codecs import Evidence, decode_record
from .config import SourceConfig, WorkbenchConfig
from .errors import format_error


@dataclass(frozen=True)
class SourceDataset:
    config: SourceConfig
    metadata: dict[str, str]
    evidence: tuple[Evidence, ...]
    content_sha256: str
    format_versions: tuple[int, ...]
    stored_content_sha256: str

    def query_ids(self) -> tuple[str, ...]:
        return tuple(sorted({row.query_id for row in self.evidence}))

    def rows_for(self, query_id: str) -> tuple[Evidence, ...]:
        return tuple(row for row in self.evidence if row.query_id == query_id)

    def summary(self) -> dict[str, Any]:
        return {
            "source_id": self.config.id,
            "source_type": self.config.type,
            "format_version": self.config.format_version,
            "format_versions": list(self.format_versions),
            "sample_rows": self.config.sample_rows,
            "sample_period_ms": self.config.sample_period_ms,
            "elapsed_semantics": self.config.elapsed_semantics,
            "scan_semantics": self.config.scan_semantics,
            "content_sha256": self.content_sha256,
            "config_fingerprint_sha256": self.config.fingerprint_sha256(),
        }


def _metadata_mismatch(source: SourceConfig, metadata: dict[str, str]) -> None:
    expected = {
        "source_id": source.id,
        "source_type": source.type,
        "format_version": str(source.format_version),
        "sample_rows": str(source.sample_rows),
        "sample_period_ms": str(source.sample_period_ms),
        "elapsed_semantics": source.elapsed_semantics,
        "scan_semantics": source.scan_semantics,
    }
    for key in (
        "source_id",
        "source_type",
        "format_version",
        "sample_rows",
        "sample_period_ms",
        "elapsed_semantics",
        "scan_semantics",
    ):
        if metadata.get(key) != expected[key]:
            raise format_error(
                f"source metadata conflicts with current declaration at {key}",
                source_id=source.id,
                row_no=None,
                path=f"metadata.{key}",
            )


def read_source_dataset(source: SourceConfig) -> SourceDataset:
    path = source.resolved_data_path
    if not path.is_file():
        raise format_error(
            "source SQLite file does not exist; run the generator first",
            source_id=source.id,
            row_no=None,
            path="$.data_path",
            details={"file": str(path)},
        )
    try:
        connection = sqlite3.connect(str(path))
        connection.row_factory = sqlite3.Row
        try:
            metadata_rows = connection.execute(
                "SELECT key, value FROM metadata ORDER BY key"
            ).fetchall()
            raw_rows = connection.execute(
                "SELECT row_no, format_version, payload, payload_sha256 "
                "FROM records ORDER BY row_no"
            ).fetchall()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise format_error(
            f"cannot read source SQLite: {exc}",
            source_id=source.id,
            row_no=None,
            path="$.data_path",
        ) from exc

    metadata = {str(row["key"]): str(row["value"]) for row in metadata_rows}
    _metadata_mismatch(source, metadata)
    if len(raw_rows) != source.sample_rows:
        raise format_error(
            f"record count mismatch: expected {source.sample_rows}, got {len(raw_rows)}",
            source_id=source.id,
            row_no=None,
            path="records",
        )

    evidence: list[Evidence] = []
    digest_records: list[list[Any]] = []
    for position, row in enumerate(raw_rows, start=1):
        row_no = int(row["row_no"])
        if row_no != position:
            raise format_error(
                "record row numbers must be contiguous and ordered",
                source_id=source.id,
                row_no=row_no,
                path="records.row_no",
            )
        payload = bytes(row["payload"])
        payload_sha = str(row["payload_sha256"])
        actual_payload_sha = hashlib.sha256(payload).hexdigest()
        if payload_sha != actual_payload_sha:
            raise format_error(
                "record payload digest mismatch",
                source_id=source.id,
                row_no=row_no,
                path="records.payload_sha256",
            )
        evidence_row = decode_record(source, row_no, payload)
        stored_version = int(row["format_version"])
        if stored_version != evidence_row.record_version:
            raise format_error(
                "format_version column conflicts with the encoded envelope",
                source_id=source.id,
                row_no=row_no,
                path="records.format_version",
            )
        evidence.append(evidence_row)
        digest_records.append([row_no, stored_version, payload_sha])

    digest_metadata = {
        key: value for key, value in metadata.items() if key != "content_sha256"
    }
    computed_content_sha = sha256_canonical(
        {"metadata": digest_metadata, "records": digest_records}
    )
    stored_content_sha = metadata.get("content_sha256", "")
    if stored_content_sha != computed_content_sha:
        raise format_error(
            "source content digest mismatch",
            source_id=source.id,
            row_no=None,
            path="metadata.content_sha256",
        )

    return SourceDataset(
        config=source,
        metadata=metadata,
        evidence=tuple(evidence),
        content_sha256=computed_content_sha,
        format_versions=tuple(sorted({row.record_version for row in evidence})),
        stored_content_sha256=stored_content_sha,
    )


def read_all_sources(config: WorkbenchConfig) -> list[SourceDataset]:
    return [read_source_dataset(source) for source in config.sorted_sources()]


LEDGER_SCHEMA = """
CREATE TABLE IF NOT EXISTS plan_state (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    plan_version INTEGER NOT NULL,
    source_fingerprints_json TEXT NOT NULL,
    policy_sha256 TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS current_view (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    plan_version INTEGER NOT NULL,
    view_json TEXT NOT NULL,
    view_sha256 TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS source_snapshots (
    plan_version INTEGER NOT NULL,
    source_id TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    format_versions_json TEXT NOT NULL,
    fingerprint_json TEXT NOT NULL,
    PRIMARY KEY (plan_version, source_id)
);
CREATE TABLE IF NOT EXISTS evidence (
    plan_version INTEGER NOT NULL,
    source_id TEXT NOT NULL,
    row_no INTEGER NOT NULL,
    query_id TEXT NOT NULL,
    source_type TEXT NOT NULL,
    record_version INTEGER NOT NULL,
    source_format_version INTEGER NOT NULL,
    elapsed_raw INTEGER NOT NULL,
    scanned_raw INTEGER NOT NULL,
    elapsed_value INTEGER NOT NULL,
    scanned_value INTEGER NOT NULL,
    elapsed_semantics TEXT NOT NULL,
    scan_semantics TEXT NOT NULL,
    sample_period_ms INTEGER NOT NULL,
    original_fields_json TEXT NOT NULL,
    padded_fields_json TEXT NOT NULL,
    PRIMARY KEY (plan_version, source_id, row_no)
);
CREATE TABLE IF NOT EXISTS merged_conclusions (
    plan_version INTEGER NOT NULL,
    query_id TEXT NOT NULL,
    metric TEXT NOT NULL,
    adopted_source TEXT NOT NULL,
    adopted_value INTEGER NOT NULL,
    comparisons_json TEXT NOT NULL,
    basis_json TEXT NOT NULL,
    PRIMARY KEY (plan_version, query_id, metric)
);
CREATE TABLE IF NOT EXISTS migration_results (
    plan_version INTEGER NOT NULL,
    pair_id TEXT NOT NULL,
    query_id TEXT NOT NULL,
    status TEXT NOT NULL,
    result_json TEXT NOT NULL,
    PRIMARY KEY (plan_version, pair_id, query_id)
);
CREATE TABLE IF NOT EXISTS attribution_results (
    plan_version INTEGER NOT NULL,
    pair_id TEXT NOT NULL,
    query_id TEXT NOT NULL,
    result_json TEXT NOT NULL,
    PRIMARY KEY (plan_version, pair_id, query_id)
);
CREATE TABLE IF NOT EXISTS invalidation_marks (
    plan_version INTEGER NOT NULL,
    query_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    PRIMARY KEY (plan_version, query_id)
);
CREATE TABLE IF NOT EXISTS invalidation_history (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_version INTEGER NOT NULL,
    trace_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS run_metrics (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_version INTEGER NOT NULL,
    metrics_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    event_json TEXT NOT NULL
);
"""


class Ledger:
    def __init__(self, workspace: Path | str):
        self.workspace = Path(workspace).resolve(strict=False)
        self.path = self.workspace / "ledger" / "ledger.sqlite3"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def initialize(self) -> None:
        connection = self._connect()
        try:
            connection.executescript(LEDGER_SCHEMA)
            connection.commit()
        finally:
            connection.close()

    def get_plan_state(self) -> dict[str, Any] | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT plan_version, source_fingerprints_json, policy_sha256 "
                "FROM plan_state WHERE singleton=1"
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        return {
            "plan_version": int(row["plan_version"]),
            "source_fingerprints": json.loads(row["source_fingerprints_json"]),
            "policy_sha256": str(row["policy_sha256"]),
        }

    def set_plan_state(
        self,
        plan_version: int,
        source_fingerprints: dict[str, dict[str, Any]],
        policy_sha256: str,
    ) -> None:
        connection = self._connect()
        try:
            connection.execute(
                "INSERT INTO plan_state(singleton, plan_version, "
                "source_fingerprints_json, policy_sha256) VALUES (1, ?, ?, ?) "
                "ON CONFLICT(singleton) DO UPDATE SET "
                "plan_version=excluded.plan_version, "
                "source_fingerprints_json=excluded.source_fingerprints_json, "
                "policy_sha256=excluded.policy_sha256",
                (
                    plan_version,
                    canonical_json(source_fingerprints),
                    policy_sha256,
                ),
            )
            connection.commit()
        finally:
            connection.close()

    def get_current_view(self) -> dict[str, Any] | None:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT plan_version, view_json, view_sha256 FROM current_view "
                "WHERE singleton=1"
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return None
        return {
            "plan_version": int(row["plan_version"]),
            "view": json.loads(row["view_json"]),
            "view_sha256": str(row["view_sha256"]),
        }

    def replace_view(
        self,
        view: dict[str, Any],
        datasets: Iterable[SourceDataset],
        plan_version: int,
        run_metrics: dict[str, Any] | None = None,
    ) -> str:
        view_json = canonical_json(view)
        view_sha = sha256_canonical(view)
        dataset_list = list(datasets)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            for table in (
                "source_snapshots",
                "evidence",
                "merged_conclusions",
                "migration_results",
                "attribution_results",
                "invalidation_marks",
            ):
                connection.execute(
                    f"DELETE FROM {table} WHERE plan_version <> ?", (plan_version,)
                )
                connection.execute(
                    f"DELETE FROM {table} WHERE plan_version = ?", (plan_version,)
                )

            for dataset in dataset_list:
                summary = dataset.summary()
                connection.execute(
                    "INSERT INTO source_snapshots(plan_version, source_id, "
                    "content_sha256, format_versions_json, fingerprint_json) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        plan_version,
                        dataset.config.id,
                        dataset.content_sha256,
                        canonical_json(list(dataset.format_versions)),
                        canonical_json(dataset.config.config_fingerprint()),
                    ),
                )
                connection.executemany(
                    "INSERT INTO evidence(plan_version, source_id, row_no, query_id, "
                    "source_type, record_version, source_format_version, elapsed_raw, "
                    "scanned_raw, elapsed_value, scanned_value, elapsed_semantics, "
                    "scan_semantics, sample_period_ms, original_fields_json, "
                    "padded_fields_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            plan_version,
                            row.source_id,
                            row.row_no,
                            row.query_id,
                            row.source_type,
                            row.record_version,
                            row.source_format_version,
                            row.elapsed_raw,
                            row.scanned_raw,
                            row.elapsed_value,
                            row.scanned_value,
                            row.elapsed_semantics,
                            row.scan_semantics,
                            row.sample_period_ms,
                            canonical_json(list(row.original_fields)),
                            canonical_json(list(row.padded_fields)),
                        )
                        for row in dataset.evidence
                    ],
                )

            for query in view.get("queries", []):
                query_id = query["query_id"]
                for metric in ("elapsed_ms", "scanned_rows"):
                    item = query["metrics"][metric]
                    adopted = item["adopted"]
                    connection.execute(
                        "INSERT INTO merged_conclusions(plan_version, query_id, metric, "
                        "adopted_source, adopted_value, comparisons_json, basis_json) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            plan_version,
                            query_id,
                            metric,
                            adopted["source_id"],
                            adopted["value"],
                            canonical_json(item["comparisons"]),
                            canonical_json(item["basis"]),
                        ),
                    )

            for pair in view.get("migration", {}).get("pairs", []):
                pair_id = pair["pair_id"]
                for result in pair["results"]:
                    connection.execute(
                        "INSERT INTO migration_results(plan_version, pair_id, query_id, "
                        "status, result_json) VALUES (?, ?, ?, ?, ?)",
                        (
                            plan_version,
                            pair_id,
                            result["query_id"],
                            result["status"],
                            canonical_json(result),
                        ),
                    )

            for result in view.get("attribution", {}).get("results", []):
                pair_id = result["pair_id"]
                query_id = result["query_id"]
                connection.execute(
                    "INSERT INTO attribution_results(plan_version, pair_id, query_id, "
                    "result_json) VALUES (?, ?, ?, ?)",
                    (plan_version, pair_id, query_id, canonical_json(result)),
                )

            invalidation = view.get("invalidation", {})
            for query_id in invalidation.get("affected_queries", []):
                connection.execute(
                    "INSERT INTO invalidation_marks(plan_version, query_id, reason) "
                    "VALUES (?, ?, ?)",
                    (plan_version, query_id, canonical_json(invalidation.get("reasons", []))),
                )
            connection.execute(
                "INSERT INTO invalidation_history(plan_version, trace_json) VALUES (?, ?)",
                (plan_version, canonical_json(invalidation)),
            )

            connection.execute(
                "INSERT INTO current_view(singleton, plan_version, view_json, view_sha256) "
                "VALUES (1, ?, ?, ?) ON CONFLICT(singleton) DO UPDATE SET "
                "plan_version=excluded.plan_version, view_json=excluded.view_json, "
                "view_sha256=excluded.view_sha256",
                (plan_version, view_json, view_sha),
            )
            if run_metrics is not None:
                connection.execute(
                    "INSERT INTO run_metrics(plan_version, metrics_json) VALUES (?, ?)",
                    (plan_version, canonical_json(run_metrics)),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return view_sha

    def append_event(self, event: dict[str, Any]) -> None:
        connection = self._connect()
        try:
            connection.execute(
                "INSERT INTO events(event_json) VALUES (?)",
                (canonical_json(event),),
            )
            connection.commit()
        finally:
            connection.close()

    def list_events(self, limit: int = 20) -> list[dict[str, Any]]:
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT event_json FROM events ORDER BY sequence DESC LIMIT ?",
                (limit,),
            ).fetchall()
        finally:
            connection.close()
        return [json.loads(row["event_json"]) for row in rows]
