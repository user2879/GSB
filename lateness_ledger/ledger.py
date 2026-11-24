"""SQLite ledger, per-window atomic commits, crash recovery and snapshot replacement."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .canonical import canonical_json, digest_object
from .engine import DisplacementRun, PlannedWindow, RunWriter
from .errors import LedgerError, SnapshotError
from .simulate import Event
from .spec import LEDGER_SPEC_VERSION, SessionBundle
from .watermark import WatermarkPoint

STATE_TABLES = (
    "watermark_points",
    "late_evidence",
    "window_displacements",
    "displacement_runs",
    "windows",
    "run_plans",
)


def _open_connection(path: str | Path, *, readonly: bool = False) -> sqlite3.Connection:
    file_path = Path(path)
    if readonly:
        uri = f"file:{file_path.as_posix()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
    else:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(file_path), isolation_level=None)
    connection.row_factory = sqlite3.Row
    if not readonly:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
    return connection


@contextmanager
def _transaction(connection: sqlite3.Connection) -> Iterator[None]:
    connection.execute("BEGIN IMMEDIATE")
    try:
        yield
    except Exception:
        connection.execute("ROLLBACK")
        raise
    else:
        connection.execute("COMMIT")


def _set_meta(connection: sqlite3.Connection, key: str, value: Any) -> None:
    connection.execute(
        "INSERT INTO meta(key,value) VALUES(?,?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, canonical_json(value)),
    )


def _delete_meta(connection: sqlite3.Connection, key: str) -> None:
    connection.execute("DELETE FROM meta WHERE key=?", (key,))


def read_meta(connection: sqlite3.Connection) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for row in connection.execute("SELECT key,value FROM meta ORDER BY key"):
        try:
            result[str(row["key"])] = json.loads(str(row["value"]))
        except json.JSONDecodeError:
            result[str(row["key"])] = str(row["value"])
    return result


def initialize_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta(
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS run_plans(
            window_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            rule_id TEXT NOT NULL,
            partition_label TEXT NOT NULL,
            partition_json TEXT NOT NULL,
            start_ms INTEGER NOT NULL,
            nominal_end_ms INTEGER NOT NULL,
            actual_end_ms INTEGER NOT NULL,
            parent_window_id TEXT,
            rule_depth INTEGER NOT NULL,
            status TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS watermark_points(
            point_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            event_seq INTEGER NOT NULL,
            event_id TEXT NOT NULL,
            phase TEXT NOT NULL,
            partition_label TEXT NOT NULL,
            partition_json TEXT NOT NULL,
            partition_watermark INTEGER NOT NULL,
            global_watermark INTEGER NOT NULL,
            strategy TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_watermark_points_order
            ON watermark_points(point_id);
        CREATE TABLE IF NOT EXISTS windows(
            window_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            format_version INTEGER NOT NULL,
            rules_version TEXT NOT NULL,
            scale INTEGER NOT NULL,
            rule_id TEXT NOT NULL,
            partition_label TEXT NOT NULL,
            partition_json TEXT NOT NULL,
            start_ms INTEGER NOT NULL,
            nominal_end_ms INTEGER NOT NULL,
            end_ms INTEGER NOT NULL,
            finalization_order INTEGER,
            trigger_seq INTEGER,
            trigger_event_id TEXT,
            strategy TEXT NOT NULL,
            version INTEGER NOT NULL,
            status TEXT NOT NULL,
            digest TEXT NOT NULL,
            content_json TEXT,
            progress_cursor INTEGER,
            parent_window_id TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_windows_order
            ON windows(finalization_order, window_id);
        CREATE TABLE IF NOT EXISTS late_evidence(
            window_id TEXT NOT NULL,
            event_seq INTEGER NOT NULL,
            event_id TEXT NOT NULL,
            decision TEXT NOT NULL,
            rule TEXT NOT NULL,
            measure_fixed INTEGER NOT NULL,
            absolute_measure_fixed INTEGER NOT NULL,
            remaining_event_count INTEGER NOT NULL,
            remaining_absolute_measure_fixed INTEGER NOT NULL,
            rejected_cumulative_absolute_fixed INTEGER NOT NULL,
            evidence_json TEXT NOT NULL,
            PRIMARY KEY(window_id,event_seq)
        );
        CREATE TABLE IF NOT EXISTS displacement_runs(
            displacement_id TEXT PRIMARY KEY,
            root_window_id TEXT NOT NULL,
            reason TEXT NOT NULL,
            event_seq INTEGER NOT NULL,
            payload_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS window_displacements(
            window_id TEXT NOT NULL,
            displacement_id TEXT NOT NULL,
            changed INTEGER NOT NULL,
            version_before INTEGER NOT NULL,
            version_after INTEGER NOT NULL,
            path_json TEXT NOT NULL,
            basis_json TEXT NOT NULL,
            PRIMARY KEY(window_id,displacement_id)
        );
        CREATE TABLE IF NOT EXISTS snapshot_records(
            snapshot_id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            created_at TEXT NOT NULL,
            format_version INTEGER NOT NULL,
            spec_digest TEXT NOT NULL,
            content_digest TEXT NOT NULL
        );
        """
    )


def _clear_state(connection: sqlite3.Connection) -> None:
    for table in STATE_TABLES:
        connection.execute(f"DELETE FROM {table}")
    for key in (
        "format_version",
        "session_id",
        "scale",
        "strategy",
        "rules_version",
        "spec_digest",
        "spec_json",
        "run_id",
        "run_status",
        "event_cursor",
        "events_generated",
        "summary_json",
        "last_error",
        "unknown_count",
        "last_restore",
    ):
        _delete_meta(connection, key)


class LedgerWriter(RunWriter):
    def __init__(
        self,
        path: str | Path,
        *,
        crash_after_commits: int | None = None,
        reset_existing: bool = False,
    ) -> None:
        self.path = Path(path)
        self.connection = _open_connection(self.path)
        initialize_schema(self.connection)
        self.crash_after_commits = crash_after_commits
        self.reset_existing = reset_existing
        self.committed_windows = 0
        self.run_id = ""
        self.run_initialized = False

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "LedgerWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self.close()

    def _check_existing(self, *, reset: bool) -> None:
        meta = read_meta(self.connection)
        if not meta:
            return
        status = meta.get("run_status")
        if reset:
            return
        if status in {"active", "incomplete_recovered"}:
            raise LedgerError(
                "ledger contains an incomplete run; recover it or rerun with --reset",
                details={
                    "ledger": str(self.path),
                    "run_status": status,
                    "run_id": meta.get("run_id"),
                },
            )
        if status == "complete" or any(
            self.connection.execute(
                "SELECT 1 FROM windows WHERE status != 'unknown' LIMIT 1"
            )
        ):
            raise LedgerError(
                "ledger already contains conclusions; use --reset to replace the run",
                details={"ledger": str(self.path), "run_status": status},
            )

    def start_run(
        self,
        *,
        bundle: SessionBundle,
        run_id: str,
        events: list[Event],
        plans: list[PlannedWindow],
        initial_watermarks: list[WatermarkPoint],
    ) -> None:
        self._check_existing(reset=self.reset_existing)
        self.run_id = run_id
        with _transaction(self.connection):
            _clear_state(self.connection)
            _set_meta(self.connection, "format_version", LEDGER_SPEC_VERSION)
            _set_meta(self.connection, "session_id", bundle.session_id)
            _set_meta(self.connection, "scale", bundle.scale)
            _set_meta(self.connection, "strategy", bundle.strategy.to_dict())
            _set_meta(self.connection, "rules_version", bundle.rules_version)
            _set_meta(self.connection, "spec_digest", bundle.spec_digest)
            _set_meta(self.connection, "spec_json", bundle.public_spec())
            _set_meta(self.connection, "run_id", run_id)
            _set_meta(self.connection, "run_status", "active")
            _set_meta(self.connection, "event_cursor", -1)
            _set_meta(self.connection, "events_generated", len(events))
            for plan in plans:
                self.connection.execute(
                    "INSERT INTO run_plans("
                    "window_id,run_id,rule_id,partition_label,partition_json,start_ms,"
                    "nominal_end_ms,actual_end_ms,parent_window_id,rule_depth,status"
                    ") VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        plan.window_id,
                        run_id,
                        plan.rule.rule_id,
                        plan.partition_label,
                        canonical_json(plan.partition),
                        plan.start_ms,
                        plan.nominal_end_ms,
                        plan.end_ms,
                        plan.parent_window_id,
                        plan.rule_depth,
                        "planned",
                    ),
                )
            self._insert_watermarks(initial_watermarks)
        self.run_initialized = True

    def _insert_watermarks(self, points: list[WatermarkPoint]) -> None:
        self.connection.executemany(
            "INSERT INTO watermark_points("
            "run_id,event_seq,event_id,phase,partition_label,partition_json,"
            "partition_watermark,global_watermark,strategy"
            ") VALUES(?,?,?,?,?,?,?,?,?)",
            [
                (
                    self.run_id,
                    point.event_seq,
                    point.event_id,
                    point.phase,
                    point.partition_label,
                    point.partition_json,
                    point.partition_watermark,
                    point.global_watermark,
                    point.strategy,
                )
                for point in points
            ],
        )

    def record_watermarks(self, points: list[WatermarkPoint]) -> None:
        if not points:
            return
        with _transaction(self.connection):
            self._insert_watermarks(points)

    def commit_window(self, window: dict[str, Any]) -> None:
        payload_json = canonical_json(
            {key: value for key, value in window.items() if key != "digest"}
        )
        late = window.get("late_decisions", {})
        accepted = late.get("accepted", [])
        discarded = late.get("discarded", [])
        evidence = list(accepted) + list(discarded)
        with _transaction(self.connection):
            self.connection.execute(
                "INSERT INTO windows("
                "window_id,run_id,format_version,rules_version,scale,rule_id,"
                "partition_label,partition_json,start_ms,nominal_end_ms,end_ms,"
                "finalization_order,trigger_seq,trigger_event_id,strategy,version,"
                "status,digest,content_json,progress_cursor,parent_window_id"
                ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(window_id) DO UPDATE SET "
                "version=excluded.version,digest=excluded.digest,"
                "content_json=excluded.content_json,progress_cursor=excluded.progress_cursor,"
                "status=excluded.status",
                (
                    window["window_id"],
                    self.run_id,
                    int(window["format_version"]),
                    str(window["rules_version"]),
                    int(window["scale"]),
                    str(window["rule_id"]),
                    str(window["partition_label"]),
                    canonical_json(window["partition"]),
                    int(window["start_ms"]),
                    int(window["nominal_end_ms"]),
                    int(window["end_ms"]),
                    int(window["finalization_order"]),
                    int(window["trigger_seq"]),
                    str(window["trigger_event_id"]),
                    str(window["strategy"]),
                    int(window["version"]),
                    str(window["status"]),
                    str(window["digest"]),
                    payload_json,
                    int(window["progress_cursor"]),
                    window.get("parent_window_id"),
                ),
            )
            self.connection.execute(
                "UPDATE run_plans SET status=? WHERE window_id=?",
                (str(window["status"]), window["window_id"]),
            )
            self.connection.execute(
                "DELETE FROM late_evidence WHERE window_id=?", (window["window_id"],)
            )
            for item in evidence:
                self.connection.execute(
                    "INSERT INTO late_evidence("
                    "window_id,event_seq,event_id,decision,rule,measure_fixed,"
                    "absolute_measure_fixed,remaining_event_count,"
                    "remaining_absolute_measure_fixed,rejected_cumulative_absolute_fixed,"
                    "evidence_json"
                    ") VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        window["window_id"],
                        int(item["event_seq"]),
                        str(item["event_id"]),
                        str(item["decision"]),
                        str(item["rule"]),
                        int(item["measure_fixed"]),
                        int(item["absolute_measure_fixed"]),
                        int(item["remaining_event_count"]),
                        int(item["remaining_absolute_measure_fixed"]),
                        int(item["rejected_cumulative_absolute_fixed"]),
                        canonical_json(item),
                    ),
                )
            old_cursor = read_meta(self.connection).get("event_cursor", -1)
            _set_meta(
                self.connection,
                "event_cursor",
                max(int(old_cursor), int(window["progress_cursor"])),
            )
        self.committed_windows += 1
        if (
            self.crash_after_commits is not None
            and self.committed_windows >= self.crash_after_commits
        ):
            os._exit(137)

    def record_displacement(self, displacement: DisplacementRun) -> None:
        payload = displacement.to_dict()
        with _transaction(self.connection):
            self.connection.execute(
                "INSERT OR REPLACE INTO displacement_runs("
                "displacement_id,root_window_id,reason,event_seq,payload_json"
                ") VALUES(?,?,?,?,?)",
                (
                    displacement.displacement_id,
                    displacement.root_window_id,
                    displacement.reason,
                    displacement.event_seq,
                    canonical_json(payload),
                ),
            )
            for step in displacement.affected:
                self.connection.execute(
                    "INSERT OR REPLACE INTO window_displacements("
                    "window_id,displacement_id,changed,version_before,version_after,"
                    "path_json,basis_json"
                    ") VALUES(?,?,?,?,?,?,?)",
                    (
                        step.window_id,
                        displacement.displacement_id,
                        int(step.changed),
                        step.version_before,
                        step.version_after,
                        canonical_json(step.path),
                        canonical_json(step.basis),
                    ),
                )

    def complete_run(self, summary: dict[str, Any]) -> None:
        with _transaction(self.connection):
            _set_meta(self.connection, "run_status", "complete")
            _set_meta(self.connection, "summary_json", summary)
            _set_meta(self.connection, "event_cursor", summary["event_cursor"])

    def fail_run(self, error: Any) -> None:
        with _transaction(self.connection):
            _set_meta(self.connection, "run_status", "failed")
            _set_meta(
                self.connection,
                "last_error",
                error.to_dict() if hasattr(error, "to_dict") else {"message": str(error)},
            )


def _readonly_connection(path: str | Path) -> sqlite3.Connection:
    file_path = Path(path)
    if not file_path.exists():
        raise LedgerError(
            "ledger file does not exist", details={"ledger": str(file_path)}
        )
    connection = _open_connection(file_path, readonly=True)
    return connection


class LedgerReader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def connection(self) -> sqlite3.Connection:
        return _readonly_connection(self.path)

    def meta(self) -> dict[str, Any]:
        with self.connection() as connection:
            return read_meta(connection)

    def windows(self, *, include_unknown: bool = True) -> list[dict[str, Any]]:
        with self.connection() as connection:
            where = "" if include_unknown else "WHERE status='finalized'"
            rows = connection.execute(
                f"SELECT * FROM windows {where} ORDER BY "
                "CASE WHEN finalization_order IS NULL THEN 1 ELSE 0 END,"
                "finalization_order, window_id"
            ).fetchall()
            result: list[dict[str, Any]] = []
            for row in rows:
                item = dict(row)
                item["partition"] = json.loads(item.pop("partition_json"))
                content_json = item.pop("content_json")
                item["content"] = json.loads(content_json) if content_json else None
                result.append(item)
            return result

    def windows_by_id(self) -> dict[str, dict[str, Any]]:
        return {item["window_id"]: item for item in self.windows()}

    def watermark_points(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM watermark_points ORDER BY point_id"
            ).fetchall()
            result: list[dict[str, Any]] = []
            for row in rows:
                item = dict(row)
                item.pop("point_id", None)
                item["partition"] = json.loads(item.pop("partition_json"))
                result.append(item)
            return result

    def plans(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM run_plans ORDER BY window_id"
            ).fetchall()
            return [dict(row) for row in rows]

    def late_evidence(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM late_evidence "
                "ORDER BY window_id,event_seq"
            ).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item["evidence"] = json.loads(item.pop("evidence_json"))
                result.append(item)
            return result

    def displacements(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM displacement_runs "
                "ORDER BY displacement_id"
            ).fetchall()
            return [json.loads(str(row["payload_json"])) for row in rows]

    def snapshots(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            if not _table_exists(connection, "snapshot_records"):
                return []
            rows = connection.execute(
                "SELECT * FROM snapshot_records ORDER BY created_at,snapshot_id"
            ).fetchall()
            return [dict(row) for row in rows]

    def status_report(self) -> dict[str, Any]:
        with self.connection() as connection:
            meta = read_meta(connection)
            finalized = [
                dict(row)
                for row in connection.execute(
                    "SELECT window_id,rule_id,partition_label,start_ms,end_ms,"
                    "finalization_order,trigger_seq,version,status,digest "
                    "FROM windows WHERE status='finalized' "
                    "ORDER BY finalization_order,window_id"
                )
            ]
            unknown = [
                dict(row)
                for row in connection.execute(
                    "SELECT window_id,rule_id,partition_label,start_ms,end_ms,status "
                    "FROM windows WHERE status='unknown' ORDER BY window_id"
                )
            ]
            plans = [
                dict(row)
                for row in connection.execute(
                    "SELECT window_id,status FROM run_plans "
                    "WHERE status!='finalized' ORDER BY window_id"
                )
            ]
            snapshots = []
            if _table_exists(connection, "snapshot_records"):
                snapshots = [
                    dict(row)
                    for row in connection.execute(
                        "SELECT * FROM snapshot_records ORDER BY created_at,snapshot_id"
                    )
                ]
            return {
                "meta": meta,
                "finalized": finalized,
                "unknown": unknown,
                "unfinished_plans": plans,
                "snapshots": snapshots,
            }


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        is not None
    )


def recover_unknown_windows(path: str | Path) -> list[dict[str, Any]]:
    """Mark every planned-but-not-committed window as unknown after a crash."""

    ledger_path = Path(path)
    if not ledger_path.exists():
        raise LedgerError(
            "ledger file does not exist", details={"ledger": str(ledger_path)}
        )
    connection = _open_connection(path)
    initialize_schema(connection)
    try:
        with _transaction(connection):
            meta = read_meta(connection)
            if meta.get("run_status") == "active":
                run_id = str(meta.get("run_id", ""))
                rows = connection.execute(
                    "SELECT * FROM run_plans WHERE status!='finalized' "
                    "ORDER BY window_id"
                ).fetchall()
                for row in rows:
                    window_id = str(row["window_id"])
                    if connection.execute(
                        "SELECT 1 FROM windows WHERE window_id=?", (window_id,)
                    ).fetchone():
                        continue
                    connection.execute(
                        "INSERT INTO windows("
                        "window_id,run_id,format_version,rules_version,scale,rule_id,"
                        "partition_label,partition_json,start_ms,nominal_end_ms,end_ms,"
                        "finalization_order,trigger_seq,trigger_event_id,strategy,version,"
                        "status,digest,content_json,progress_cursor,parent_window_id"
                        ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            window_id,
                            run_id,
                            LEDGER_SPEC_VERSION,
                            str(meta.get("rules_version", "")),
                            int(meta.get("scale", 0)),
                            str(row["rule_id"]),
                            str(row["partition_label"]),
                            str(row["partition_json"]),
                            int(row["start_ms"]),
                            int(row["nominal_end_ms"]),
                            int(row["actual_end_ms"]),
                            None,
                            None,
                            None,
                            str(meta.get("strategy", {}).get("kind", "unknown"))
                            if isinstance(meta.get("strategy"), dict)
                            else "unknown",
                            -1,
                            "unknown",
                            "UNKNOWN",
                            None,
                            None,
                            row["parent_window_id"],
                        ),
                    )
                    connection.execute(
                        "UPDATE run_plans SET status='unknown' WHERE window_id=?",
                        (window_id,),
                    )
                _set_meta(connection, "run_status", "incomplete_recovered")
                _set_meta(connection, "unknown_count", len(rows))
            rows = connection.execute(
                "SELECT window_id,rule_id,partition_label,start_ms,end_ms,status "
                "FROM windows WHERE status='unknown' ORDER BY window_id"
            ).fetchall()
            return [dict(row) for row in rows]
    finally:
        connection.close()


def _content_digest(connection: sqlite3.Connection) -> str:
    windows = [
        {
            "window_id": row["window_id"],
            "digest": row["digest"],
            "version": row["version"],
            "status": row["status"],
        }
        for row in connection.execute(
            "SELECT window_id,digest,version,status FROM windows ORDER BY window_id"
        )
    ]
    watermarks = [
        {
            "partition_label": row["partition_label"],
            "partition_watermark": row["partition_watermark"],
            "global_watermark": row["global_watermark"],
            "event_seq": row["event_seq"],
            "phase": row["phase"],
        }
        for row in connection.execute(
            "SELECT partition_label,partition_watermark,global_watermark,event_seq,phase "
            "FROM watermark_points ORDER BY point_id DESC LIMIT 1000"
        )
    ]
    meta = read_meta(connection)
    return digest_object(
        {
            "windows": windows,
            "watermarks": watermarks,
            "format_version": meta.get("format_version"),
            "spec_digest": meta.get("spec_digest"),
            "event_cursor": meta.get("event_cursor"),
        }
    )


def create_snapshot(
    ledger_path: str | Path, snapshot_path: str | Path, *, snapshot_id: str | None = None
) -> dict[str, Any]:
    source_path = Path(ledger_path)
    destination_path = Path(snapshot_path)
    if source_path.resolve() == destination_path.resolve():
        raise SnapshotError("snapshot destination cannot be the live ledger")
    if not source_path.exists():
        raise SnapshotError("ledger file does not exist", details={"ledger": str(source_path)})
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists():
        destination_path.unlink()
    source = _open_connection(source_path)
    initialize_schema(source)
    source.execute("PRAGMA wal_checkpoint(FULL)")
    meta = read_meta(source)
    format_version = int(meta.get("format_version", LEDGER_SPEC_VERSION))
    if format_version != LEDGER_SPEC_VERSION:
        source.close()
        raise SnapshotError(
            "unsupported ledger format version",
            details={"format_version": format_version, "expected": LEDGER_SPEC_VERSION},
        )
    content_digest = _content_digest(source)
    snapshot_identifier = snapshot_id or (
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        + "-"
        + content_digest[:12]
    )
    destination = sqlite3.connect(str(destination_path), isolation_level=None)
    try:
        source.backup(destination)
        initialize_schema(destination)
        with _transaction(destination):
            _set_meta(destination, "snapshot_id", snapshot_identifier)
            _set_meta(destination, "snapshot_content_digest", content_digest)
    finally:
        destination.close()
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with _transaction(source):
        source.execute(
            "INSERT OR REPLACE INTO snapshot_records("
            "snapshot_id,path,created_at,format_version,spec_digest,content_digest"
            ") VALUES(?,?,?,?,?,?)",
            (
                snapshot_identifier,
                str(destination_path.resolve()),
                created_at,
                format_version,
                str(meta.get("spec_digest", "")),
                content_digest,
            ),
        )
    source.close()
    return {
        "snapshot_id": snapshot_identifier,
        "path": str(destination_path.resolve()),
        "created_at": created_at,
        "format_version": format_version,
        "spec_digest": meta.get("spec_digest"),
        "content_digest": content_digest,
    }


def restore_snapshot(ledger_path: str | Path, snapshot_path: str | Path) -> dict[str, Any]:
    destination_path = Path(ledger_path)
    source_path = Path(snapshot_path)
    if not source_path.exists():
        raise SnapshotError(
            "snapshot file does not exist", details={"snapshot": str(source_path)}
        )
    try:
        snapshot = _open_connection(source_path, readonly=True)
        snapshot_meta = read_meta(snapshot)
        snapshot.close()
    except sqlite3.Error as exc:
        raise SnapshotError(
            "cannot open snapshot database",
            details={"snapshot": str(source_path), "cause": str(exc)},
        ) from exc
    if int(snapshot_meta.get("format_version", -1)) != LEDGER_SPEC_VERSION:
        raise SnapshotError(
            "unsupported snapshot format version",
            details={
                "format_version": snapshot_meta.get("format_version"),
                "expected": LEDGER_SPEC_VERSION,
            },
        )

    live = _open_connection(destination_path)
    initialize_schema(live)
    current_meta = read_meta(live)
    current_windows = {
        row["window_id"]: dict(row)
        for row in live.execute(
            "SELECT window_id,scale,rules_version,format_version,digest FROM windows"
        )
    }
    snapshot_attach = sqlite3.connect(str(source_path), isolation_level=None)
    snapshot_attach.close()
    try:
        live.execute("ATTACH DATABASE ? AS snap", (str(source_path),))
        snapshot_windows = {
            row["window_id"]: dict(row)
            for row in live.execute(
                "SELECT window_id,scale,rules_version,format_version,digest "
                "FROM snap.windows"
            )
        }
        added = sorted(set(snapshot_windows) - set(current_windows))
        removed = sorted(set(current_windows) - set(snapshot_windows))
        upgraded = sorted(
            window_id
            for window_id in set(snapshot_windows) & set(current_windows)
            if int(snapshot_windows[window_id]["scale"])
            != int(current_windows[window_id]["scale"])
            or str(snapshot_windows[window_id]["rules_version"])
            != str(current_windows[window_id]["rules_version"])
            or int(snapshot_windows[window_id]["format_version"])
            != int(current_windows[window_id]["format_version"])
        )
        current_digest = _content_digest(live)
        snapshot_digest = str(
            snapshot_meta.get("snapshot_content_digest")
            or _content_digest_from_attached(live)
        )
        if int(current_meta.get("format_version", -1)) != int(
            snapshot_meta.get("format_version", -1)
        ):
            reason = "format_version_mismatch"
        elif str(current_meta.get("spec_digest", "")) != str(
            snapshot_meta.get("spec_digest", "")
        ):
            reason = "spec_digest_mismatch"
        elif current_digest != snapshot_digest:
            reason = "content_digest_mismatch"
        else:
            reason = "explicit_whole_snapshot_replace"

        with _transaction(live):
            for table in STATE_TABLES:
                live.execute(f"DELETE FROM {table}")
            for table in STATE_TABLES:
                live.execute(f"INSERT INTO {table} SELECT * FROM snap.{table}")
            live.execute("DELETE FROM meta")
            live.execute("INSERT INTO meta(key,value) SELECT key,value FROM snap.meta")
            restore_report = {
                "snapshot_path": str(source_path.resolve()),
                "snapshot_id": snapshot_meta.get("snapshot_id"),
                "reason": reason,
                "replacement": "whole_window_conclusion_and_watermark_state",
                "added": added,
                "removed": removed,
                "scale_upgraded": upgraded,
                "current_content_digest": current_digest,
                "snapshot_content_digest": snapshot_digest,
            }
            _set_meta(live, "last_restore", restore_report)
            record = (
                f"restore-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-"
                f"{digest_object(restore_report)[:12]}"
            )
            live.execute(
                "INSERT OR REPLACE INTO snapshot_records("
                "snapshot_id,path,created_at,format_version,spec_digest,content_digest"
                ") VALUES(?,?,?,?,?,?)",
                (
                    record,
                    str(source_path.resolve()),
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    int(snapshot_meta.get("format_version", LEDGER_SPEC_VERSION)),
                    str(snapshot_meta.get("spec_digest", "")),
                    snapshot_digest,
                ),
            )
        return restore_report
    except sqlite3.Error as exc:
        raise SnapshotError(
            "snapshot replacement failed",
            details={"ledger": str(destination_path), "snapshot": str(source_path), "cause": str(exc)},
        ) from exc
    finally:
        try:
            live.execute("DETACH DATABASE snap")
        except sqlite3.Error:
            pass
        live.close()


def _content_digest_from_attached(connection: sqlite3.Connection) -> str:
    windows = [
        {
            "window_id": row["window_id"],
            "digest": row["digest"],
            "version": row["version"],
            "status": row["status"],
        }
        for row in connection.execute(
            "SELECT window_id,digest,version,status FROM snap.windows ORDER BY window_id"
        )
    ]
    watermarks = [
        {
            "partition_label": row["partition_label"],
            "partition_watermark": row["partition_watermark"],
            "global_watermark": row["global_watermark"],
            "event_seq": row["event_seq"],
            "phase": row["phase"],
        }
        for row in connection.execute(
            "SELECT partition_label,partition_watermark,global_watermark,event_seq,phase "
            "FROM snap.watermark_points ORDER BY point_id DESC LIMIT 1000"
        )
    ]
    meta = {
        str(row["key"]): json.loads(str(row["value"]))
        for row in connection.execute("SELECT key,value FROM snap.meta")
    }
    return digest_object(
        {
            "windows": windows,
            "watermarks": watermarks,
            "format_version": meta.get("format_version"),
            "spec_digest": meta.get("spec_digest"),
            "event_cursor": meta.get("event_cursor"),
        }
    )
