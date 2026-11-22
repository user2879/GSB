"""SQLite ledger, per-topic transactions, parameter versions, and replay."""

from __future__ import annotations

import json
import sqlite3
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from .canonical import canonical_json, sha256_obj, utc_now
from .config import Declarations, declarations_from_document
from .errors import LabError, io_error, state_error
from .records import NormalizedRecord, load_normalized_records, model_digest as compute_model_digest
from .semantics import analyze


SCHEMA = """
PRAGMA foreign_keys = OFF;
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS parameter_versions (
    version INTEGER PRIMARY KEY,
    fingerprint TEXT NOT NULL,
    declarations_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS executions (
    execution_id TEXT PRIMARY KEY,
    ordinal INTEGER NOT NULL,
    parameter_version INTEGER NOT NULL,
    parameter_fingerprint TEXT NOT NULL,
    declarations_json TEXT NOT NULL,
    format_versions_json TEXT NOT NULL,
    topics_json TEXT NOT NULL,
    model_digest TEXT NOT NULL,
    status TEXT NOT NULL,
    current_state INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    content_digest TEXT
);
CREATE TABLE IF NOT EXISTS batches (
    batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id TEXT NOT NULL,
    parameter_version INTEGER NOT NULL,
    topic TEXT NOT NULL,
    topic_index INTEGER NOT NULL,
    record_count INTEGER NOT NULL,
    batch_digest TEXT NOT NULL,
    committed_at TEXT NOT NULL,
    stale INTEGER NOT NULL DEFAULT 0,
    UNIQUE(execution_id, topic)
);
CREATE TABLE IF NOT EXISTS records (
    record_id INTEGER PRIMARY KEY AUTOINCREMENT,
    execution_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    message_seq INTEGER NOT NULL,
    message_id TEXT NOT NULL,
    consumer_group TEXT NOT NULL,
    source_format TEXT NOT NULL,
    source_line INTEGER NOT NULL,
    record_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS records_exec_topic ON records(execution_id, topic, message_seq);
CREATE TABLE IF NOT EXISTS conclusions (
    execution_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    message_seq INTEGER NOT NULL,
    message_id TEXT NOT NULL,
    verdict TEXT NOT NULL,
    conclusion_json TEXT NOT NULL,
    PRIMARY KEY(execution_id, topic, message_seq, message_id)
);
CREATE TABLE IF NOT EXISTS group_conclusions (
    execution_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    message_seq INTEGER NOT NULL,
    message_id TEXT NOT NULL,
    consumer_group TEXT NOT NULL,
    verdict TEXT NOT NULL,
    conclusion_json TEXT NOT NULL,
    PRIMARY KEY(execution_id, topic, message_seq, message_id, consumer_group)
);
CREATE TABLE IF NOT EXISTS conflicts (
    execution_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    message_seq INTEGER NOT NULL,
    message_id TEXT NOT NULL,
    conflict_type TEXT NOT NULL,
    conflict_json TEXT NOT NULL,
    PRIMARY KEY(execution_id, topic, message_seq, message_id, conflict_type)
);
CREATE TABLE IF NOT EXISTS group_stats (
    execution_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    consumer_group TEXT NOT NULL,
    message_count INTEGER NOT NULL,
    exact_once_count INTEGER NOT NULL,
    exact_fixed INTEGER NOT NULL,
    retry_count INTEGER NOT NULL,
    retry_fixed INTEGER NOT NULL,
    conflict_count INTEGER NOT NULL,
    scale_factor INTEGER NOT NULL,
    single_error_lt_units INTEGER NOT NULL,
    comparison_error_lt_units INTEGER NOT NULL,
    stats_json TEXT NOT NULL,
    PRIMARY KEY(execution_id, topic, consumer_group)
);
CREATE TABLE IF NOT EXISTS topic_stats (
    execution_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    fixed_exact_once_ratio INTEGER NOT NULL,
    error_affected INTEGER NOT NULL,
    stats_json TEXT NOT NULL,
    PRIMARY KEY(execution_id, topic)
);
CREATE TABLE IF NOT EXISTS view_diffs (
    execution_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    topic TEXT NOT NULL,
    message_seq INTEGER NOT NULL,
    message_id TEXT NOT NULL,
    diff_json TEXT NOT NULL,
    PRIMARY KEY(execution_id, topic, message_seq, message_id)
);
CREATE TABLE IF NOT EXISTS cursors (
    execution_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    batch_id INTEGER NOT NULL,
    last_message_seq INTEGER NOT NULL,
    record_count INTEGER NOT NULL,
    completed_at TEXT NOT NULL,
    PRIMARY KEY(execution_id, topic)
);
"""


def _parse_json(text: str) -> Any:
    return json.loads(text)


def _state_for_digest(state: dict[str, Any]) -> dict[str, Any]:
    value = deepcopy(state)
    value.pop("execution_id", None)
    value.pop("execution_status", None)
    value.pop("started_at", None)
    value.pop("completed_at", None)
    value.pop("content_digest", None)
    value.pop("view_diff", None)
    for batch in value.get("batches", []):
        batch.pop("committed_at", None)
        batch.pop("stale", None)
    for cursor in value.get("cursors", []):
        cursor.pop("completed_at", None)
    return value


def state_content_digest(state: dict[str, Any]) -> str:
    return sha256_obj(_state_for_digest(state))


class Ledger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.connection = sqlite3.connect(str(self.path), timeout=30.0)
        except sqlite3.Error as exc:
            raise io_error("could not open SQLite ledger", path=str(self.path), reason=str(exc))
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode = DELETE")
        self.connection.execute("PRAGMA synchronous = FULL")
        self.connection.executescript(SCHEMA)
        if self._meta_get("ledger_schema_version") is None:
            with self.connection:
                self._meta_set("ledger_schema_version", "1")
                self._meta_set("execution_ordinal", "0")

    def close(self) -> None:
        try:
            self.connection.close()
        except sqlite3.Error:
            pass

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def _meta_get(self, key: str) -> str | None:
        row = self.connection.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def _meta_set(self, key: str, value: str) -> None:
        self.connection.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def current_execution_id(self) -> str | None:
        return self._meta_get("current_execution_id")

    def current_execution(self) -> dict[str, Any] | None:
        execution_id = self.current_execution_id()
        if execution_id is None:
            return None
        row = self.connection.execute(
            "SELECT * FROM executions WHERE execution_id=?", (execution_id,)
        ).fetchone()
        return None if row is None else dict(row)

    def _parameter_version_for(
        self,
        declarations: Declarations,
        previous_execution: dict[str, Any] | None,
    ) -> int:
        fingerprint = declarations.parameter_fingerprint()
        if previous_execution is not None and previous_execution["parameter_fingerprint"] == fingerprint:
            return int(previous_execution["parameter_version"])
        row = self.connection.execute("SELECT COALESCE(MAX(version), 0) AS max_version FROM parameter_versions").fetchone()
        version = int(row["max_version"]) + 1
        self.connection.execute(
            "INSERT INTO parameter_versions(version, fingerprint, declarations_json, created_at) VALUES(?,?,?,?)",
            (
                version,
                fingerprint,
                canonical_json(declarations.parameter_document()),
                utc_now(),
            ),
        )
        return version

    def _conclusions_for_execution(self, execution_id: str) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT conclusion_json FROM conclusions WHERE execution_id=? "
            "ORDER BY topic, message_seq, message_id",
            (execution_id,),
        ).fetchall()
        return [_parse_json(row["conclusion_json"]) for row in rows]

    def _view_diff(
        self,
        old_rows: Iterable[dict[str, Any]],
        new_rows: Iterable[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        def index(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, int, str], dict[str, Any]]:
            result: dict[tuple[str, int, str], dict[str, Any]] = {}
            for row in rows:
                result[(row["topic"], int(row["message_seq"]), row["message_id"])] = row
            return result

        old = index(old_rows)
        new = index(new_rows)
        diffs: list[dict[str, Any]] = []
        for key in sorted(set(old) | set(new)):
            topic, seq, message_id = key
            before = old.get(key)
            after = new.get(key)
            if before is None:
                diffs.append(
                    {
                        "kind": "added",
                        "topic": topic,
                        "message_seq": seq,
                        "message_id": message_id,
                        "old_verdict": None,
                        "new_verdict": after["verdict"],
                        "changed_fields": [],
                    }
                )
            elif after is None:
                diffs.append(
                    {
                        "kind": "deleted",
                        "topic": topic,
                        "message_seq": seq,
                        "message_id": message_id,
                        "old_verdict": before["verdict"],
                        "new_verdict": None,
                        "changed_fields": [],
                    }
                )
            elif canonical_json(before) != canonical_json(after):
                changed_fields = sorted(
                    key
                    for key in set(before) | set(after)
                    if canonical_json(before.get(key)) != canonical_json(after.get(key))
                )
                diffs.append(
                    {
                        "kind": "changed",
                        "topic": topic,
                        "message_seq": seq,
                        "message_id": message_id,
                        "old_verdict": before.get("verdict"),
                        "new_verdict": after.get("verdict"),
                        "changed_fields": changed_fields,
                    }
                )
        order = {"added": 0, "deleted": 1, "changed": 2}
        diffs.sort(
            key=lambda row: (
                order.get(row["kind"], 99),
                row["topic"],
                row["message_seq"],
                row["message_id"],
            )
        )
        return diffs

    def start_execution(
        self,
        declarations: Declarations,
        model_digest: str,
        new_conclusions: Iterable[dict[str, Any]],
    ) -> dict[str, Any]:
        previous = self.current_execution()
        old_rows: list[dict[str, Any]] = []
        if previous is not None:
            old_rows = self._conclusions_for_execution(previous["execution_id"])
        new_rows = list(new_conclusions)
        with self.connection:
            parameter_version = self._parameter_version_for(declarations, previous)
            ordinal = int(self._meta_get("execution_ordinal") or "0") + 1
            execution_id = f"exec-{ordinal:06d}-v{parameter_version}"
            self.connection.execute("UPDATE batches SET stale=1 WHERE stale=0")
            self.connection.execute("UPDATE executions SET current_state=0 WHERE current_state=1")
            self._meta_set("execution_ordinal", str(ordinal))
            self.connection.execute(
                "INSERT INTO executions("
                "execution_id, ordinal, parameter_version, parameter_fingerprint, declarations_json, "
                "format_versions_json, topics_json, model_digest, status, current_state, started_at"
                ") VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    execution_id,
                    ordinal,
                    parameter_version,
                    declarations.parameter_fingerprint(),
                    canonical_json(declarations.parameter_document()),
                    canonical_json(list(declarations.format_versions)),
                    canonical_json(list(declarations.topic_names)),
                    model_digest,
                    "running",
                    1,
                    utc_now(),
                ),
            )
            self._meta_set("current_execution_id", execution_id)
            diff_rows = self._view_diff(old_rows, new_rows)
            for row in diff_rows:
                self.connection.execute(
                    "INSERT INTO view_diffs(execution_id, kind, topic, message_seq, message_id, diff_json) "
                    "VALUES(?,?,?,?,?,?)",
                    (
                        execution_id,
                        row["kind"],
                        row["topic"],
                        row["message_seq"],
                        row["message_id"],
                        canonical_json(row),
                    ),
                )
        return {
            "execution_id": execution_id,
            "parameter_version": parameter_version,
            "previous_execution_id": None if previous is None else previous["execution_id"],
            "previous_parameter_version": None if previous is None else previous["parameter_version"],
            "parameter_changed": previous is None
            or previous["parameter_fingerprint"] != declarations.parameter_fingerprint(),
            "view_diff": diff_rows,
        }

    def commit_topic(
        self,
        execution_id: str,
        topic: str,
        topic_index: int,
        records: Iterable[NormalizedRecord],
        conclusions: Iterable[dict[str, Any]],
        group_conclusions: Iterable[dict[str, Any]],
        conflicts: Iterable[dict[str, Any]],
        group_stats: Iterable[dict[str, Any]],
        topic_stat: dict[str, Any] | None,
    ) -> dict[str, Any]:
        record_rows = sorted(
            (record.provenance_dict() for record in records),
            key=lambda row: (
                row["message_seq"],
                row["message_id"],
                row["consumer_group"],
                row["source_format"],
                row["source_line"],
            ),
        )
        conclusion_rows = sorted(
            list(conclusions), key=lambda row: (row["message_seq"], row["message_id"])
        )
        group_conclusion_rows = sorted(
            list(group_conclusions),
            key=lambda row: (row["message_seq"], row["message_id"], row["consumer_group"]),
        )
        conflict_rows = sorted(
            list(conflicts),
            key=lambda row: (row["message_seq"], row["conflict_type"], row["message_id"]),
        )
        group_stats_rows = sorted(
            list(group_stats), key=lambda row: (row["consumer_group"],)
        )
        last_message_seq = max((row["message_seq"] for row in record_rows), default=0)
        payload = {
            "topic": topic,
            "records": record_rows,
            "conclusions": conclusion_rows,
            "group_conclusions": group_conclusion_rows,
            "conflicts": conflict_rows,
            "group_stats": group_stats_rows,
            "topic_stat": topic_stat,
            "last_message_seq": last_message_seq,
        }
        batch_digest = sha256_obj(payload)
        execution = self.current_execution()
        if execution is None or execution["execution_id"] != execution_id:
            raise state_error("execution is not current", execution_id=execution_id)
        if execution["status"] != "running":
            raise state_error("execution is not running", execution_id=execution_id, status=execution["status"])
        parameter_version = int(execution["parameter_version"])
        now = utc_now()
        with self.connection:
            batch_cursor = self.connection.execute(
                "INSERT INTO batches("
                "execution_id, parameter_version, topic, topic_index, record_count, batch_digest, committed_at, stale"
                ") VALUES(?,?,?,?,?,?,?,0)",
                (execution_id, parameter_version, topic, topic_index, len(record_rows), batch_digest, now),
            )
            batch_id = int(batch_cursor.lastrowid)
            for row in record_rows:
                self.connection.execute(
                    "INSERT INTO records("
                    "execution_id, topic, message_seq, message_id, consumer_group, source_format, source_line, record_json"
                    ") VALUES(?,?,?,?,?,?,?,?)",
                    (
                        execution_id,
                        topic,
                        row["message_seq"],
                        row["message_id"],
                        row["consumer_group"],
                        row["source_format"],
                        row["source_line"],
                        canonical_json(row),
                    ),
                )
            for row in conclusion_rows:
                self.connection.execute(
                    "INSERT INTO conclusions(execution_id, topic, message_seq, message_id, verdict, conclusion_json) "
                    "VALUES(?,?,?,?,?,?)",
                    (
                        execution_id,
                        topic,
                        row["message_seq"],
                        row["message_id"],
                        row["verdict"],
                        canonical_json(row),
                    ),
                )
            for row in group_conclusion_rows:
                self.connection.execute(
                    "INSERT INTO group_conclusions("
                    "execution_id, topic, message_seq, message_id, consumer_group, verdict, conclusion_json"
                    ") VALUES(?,?,?,?,?,?,?)",
                    (
                        execution_id,
                        topic,
                        row["message_seq"],
                        row["message_id"],
                        row["consumer_group"],
                        row["verdict"],
                        canonical_json(row),
                    ),
                )
            for row in conflict_rows:
                self.connection.execute(
                    "INSERT INTO conflicts(execution_id, topic, message_seq, message_id, conflict_type, conflict_json) "
                    "VALUES(?,?,?,?,?,?)",
                    (
                        execution_id,
                        topic,
                        row["message_seq"],
                        row["message_id"],
                        row["conflict_type"],
                        canonical_json(row),
                    ),
                )
            for row in group_stats_rows:
                exact = row["exact_once_ratio"]
                retry = row["retry_rate"]
                self.connection.execute(
                    "INSERT INTO group_stats("
                    "execution_id, topic, consumer_group, message_count, exact_once_count, exact_fixed, "
                    "retry_count, retry_fixed, conflict_count, scale_factor, single_error_lt_units, "
                    "comparison_error_lt_units, stats_json"
                    ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        execution_id,
                        topic,
                        row["consumer_group"],
                        int(row["message_count"]),
                        int(row["exact_once_count"]),
                        int(exact["fixed"]),
                        int(row["retry_count"]),
                        int(retry["fixed"]),
                        int(row["semantic_conflict_count"]),
                        int(row["scale_factor"]),
                        int(row["single_error_lt_units"]),
                        int(row["comparison_error_lt_units"]),
                        canonical_json(row),
                    ),
                )
            if topic_stat is not None:
                self.connection.execute(
                    "INSERT INTO topic_stats("
                    "execution_id, topic, fixed_exact_once_ratio, error_affected, stats_json"
                    ") VALUES(?,?,?,?,?)",
                    (
                        execution_id,
                        topic,
                        int(topic_stat["fixed_exact_once_ratio"]),
                        1 if topic_stat["error_affected"] else 0,
                        canonical_json(topic_stat),
                    ),
                )
            self.connection.execute(
                "INSERT INTO cursors(execution_id, topic, batch_id, last_message_seq, record_count, completed_at) "
                "VALUES(?,?,?,?,?,?)",
                (execution_id, topic, batch_id, last_message_seq, len(record_rows), now),
            )
        return {
            "topic": topic,
            "batch_id": batch_id,
            "record_count": len(record_rows),
            "last_message_seq": last_message_seq,
            "batch_digest": batch_digest,
        }

    def finish_execution(self, execution_id: str) -> dict[str, Any]:
        execution = self.current_execution()
        if execution is None or execution["execution_id"] != execution_id:
            raise state_error("execution is not current", execution_id=execution_id)
        if execution["status"] != "running":
            raise state_error("execution is not running", execution_id=execution_id, status=execution["status"])
        state = self.export_state(execution_id)
        digest = state_content_digest(state)
        now = utc_now()
        with self.connection:
            self.connection.execute(
                "UPDATE executions SET status='complete', completed_at=?, content_digest=? WHERE execution_id=?",
                (now, digest, execution_id),
            )
            self._meta_set("current_content_digest", digest)
        return {"execution_id": execution_id, "status": "complete", "content_digest": digest}

    def export_state(self, execution_id: str | None = None) -> dict[str, Any] | None:
        execution_id = execution_id or self.current_execution_id()
        if execution_id is None:
            return None
        execution = self.connection.execute(
            "SELECT * FROM executions WHERE execution_id=?", (execution_id,)
        ).fetchone()
        if execution is None:
            return None
        execution = dict(execution)
        record_rows = self.connection.execute(
            "SELECT record_json FROM records WHERE execution_id=? "
            "ORDER BY topic, message_seq, message_id, consumer_group, source_format, source_line",
            (execution_id,),
        ).fetchall()
        conclusions = self.connection.execute(
            "SELECT conclusion_json FROM conclusions WHERE execution_id=? "
            "ORDER BY topic, message_seq, message_id",
            (execution_id,),
        ).fetchall()
        group_conclusions = self.connection.execute(
            "SELECT conclusion_json FROM group_conclusions WHERE execution_id=? "
            "ORDER BY topic, message_seq, consumer_group, message_id",
            (execution_id,),
        ).fetchall()
        conflicts = self.connection.execute(
            "SELECT conflict_json FROM conflicts WHERE execution_id=? "
            "ORDER BY message_seq, conflict_type, topic, message_id",
            (execution_id,),
        ).fetchall()
        group_stats = self.connection.execute(
            "SELECT stats_json FROM group_stats WHERE execution_id=? ORDER BY topic, consumer_group",
            (execution_id,),
        ).fetchall()
        topic_stats = self.connection.execute(
            "SELECT stats_json FROM topic_stats WHERE execution_id=? ORDER BY topic",
            (execution_id,),
        ).fetchall()
        view_diff = self.connection.execute(
            "SELECT diff_json FROM view_diffs WHERE execution_id=? "
            "ORDER BY CASE kind WHEN 'added' THEN 0 WHEN 'deleted' THEN 1 ELSE 2 END, topic, message_seq, message_id",
            (execution_id,),
        ).fetchall()
        batches = self.connection.execute(
            "SELECT topic, topic_index, record_count, batch_digest, committed_at, stale "
            "FROM batches WHERE execution_id=? ORDER BY topic",
            (execution_id,),
        ).fetchall()
        cursors = self.connection.execute(
            "SELECT topic, last_message_seq, record_count, completed_at FROM cursors "
            "WHERE execution_id=? ORDER BY topic",
            (execution_id,),
        ).fetchall()
        state = {
            "snapshot_schema_version": 1,
            "execution_id": execution_id,
            "execution_status": execution["status"],
            "parameter_version": int(execution["parameter_version"]),
            "parameter_fingerprint": execution["parameter_fingerprint"],
            "format_versions": _parse_json(execution["format_versions_json"]),
            "topics": _parse_json(execution["topics_json"]),
            "declarations": _parse_json(execution["declarations_json"]),
            "model_digest": execution["model_digest"],
            "started_at": execution["started_at"],
            "completed_at": execution["completed_at"],
            "content_digest": execution["content_digest"],
            "records": [_parse_json(row["record_json"]) for row in record_rows],
            "conclusions": [_parse_json(row["conclusion_json"]) for row in conclusions],
            "group_conclusions": [_parse_json(row["conclusion_json"]) for row in group_conclusions],
            "conflicts": [_parse_json(row["conflict_json"]) for row in conflicts],
            "group_stats": [_parse_json(row["stats_json"]) for row in group_stats],
            "topic_stats": [_parse_json(row["stats_json"]) for row in topic_stats],
            "view_diff": [_parse_json(row["diff_json"]) for row in view_diff],
            "batches": [
                {
                    "topic": row["topic"],
                    "topic_index": int(row["topic_index"]),
                    "record_count": int(row["record_count"]),
                    "batch_digest": row["batch_digest"],
                    "committed_at": row["committed_at"],
                    "stale": bool(row["stale"]),
                }
                for row in batches
            ],
            "cursors": [
                {
                    "topic": row["topic"],
                    "last_message_seq": int(row["last_message_seq"]),
                    "record_count": int(row["record_count"]),
                    "completed_at": row["completed_at"],
                }
                for row in cursors
            ],
        }
        return state

    def _first_diff(
        self,
        expected: list[dict[str, Any]],
        actual: list[dict[str, Any]],
        *,
        key_fields: tuple[str, ...],
        label: str,
    ) -> dict[str, Any] | None:
        def index(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
            result: dict[tuple[Any, ...], dict[str, Any]] = {}
            for row in rows:
                key = tuple(row[field] for field in key_fields)
                result[key] = row
            return result

        left = index(expected)
        right = index(actual)
        for key in sorted(set(left) | set(right), key=lambda item: tuple(str(part) for part in item)):
            before = left.get(key)
            after = right.get(key)
            if before is None or after is None:
                row = before if before is not None else after or {}
                return {
                    "section": label,
                    "message_id": row.get("message_id", "<non-message-row>"),
                    "field": "row",
                    "left": before,
                    "right": after,
                }
            if canonical_json(before) != canonical_json(after):
                for field in sorted(set(before) | set(after)):
                    if canonical_json(before.get(field)) != canonical_json(after.get(field)):
                        return {
                            "section": label,
                            "message_id": before.get("message_id", "<non-message-row>"),
                            "field": field,
                            "left": before.get(field),
                            "right": after.get(field),
                        }
        return None

    def recover(self, records_path: str | Path | None = None) -> dict[str, Any]:
        state = self.export_state()
        if state is None:
            raise state_error("ledger has no current execution")
        declarations = declarations_from_document(state["declarations"], source_path=None)
        records = [NormalizedRecord.from_dict(row, include_line=True) for row in state["records"]]
        replayed_model_digest = compute_model_digest(records)
        if state["execution_status"] == "complete" and replayed_model_digest != state["model_digest"]:
            raise LabError(
                "E_REPLAY",
                "ledger record sequence does not reproduce the stored normalized-model digest",
                {
                    "first_mismatch": {
                        "section": "normalized_model",
                        "message_id": "<normalized-model>",
                        "field": "model_digest",
                        "left": state["model_digest"],
                        "right": replayed_model_digest,
                    },
                    "basis": "canonical records stored in the ledger under the stored parameter version",
                    "parameter_version": state["parameter_version"],
                },
            )
        replay = analyze(records, declarations)
        mismatch = None
        mismatch = mismatch or self._first_diff(
            state["conclusions"], replay["conclusions"], key_fields=("topic", "message_seq", "message_id"), label="conclusions"
        )
        mismatch = mismatch or self._first_diff(
            state["group_conclusions"],
            replay["group_conclusions"],
            key_fields=("topic", "message_seq", "message_id", "consumer_group"),
            label="group_conclusions",
        )
        mismatch = mismatch or self._first_diff(
            state["conflicts"],
            replay["conflicts"],
            key_fields=("topic", "message_seq", "message_id", "conflict_type"),
            label="conflicts",
        )
        expected_group_stats = state["group_stats"]
        actual_group_stats = replay["stats"]["by_topic_group"]
        mismatch = mismatch or self._first_diff(
            expected_group_stats,
            actual_group_stats,
            key_fields=("topic", "consumer_group"),
            label="group_stats",
        )
        mismatch = mismatch or self._first_diff(
            state["topic_stats"],
            replay["stats"]["topic_ranking"],
            key_fields=("topic",),
            label="topic_stats",
        )
        if mismatch is not None:
            raise LabError(
                "E_REPLAY",
                "ledger replay diverged from committed conclusions",
                {
                    "first_mismatch": mismatch,
                    "basis": "re-derivation from ledger records under stored parameter version",
                    "parameter_version": state["parameter_version"],
                },
            )

        committed_topics = sorted(batch["topic"] for batch in state["batches"])
        all_topics = sorted(state["topics"])
        missing_topics = [topic for topic in all_topics if topic not in set(committed_topics)]
        unknown_messages: list[dict[str, Any]] = []
        if missing_topics and records_path is not None:
            raw_records = load_normalized_records(records_path, declarations)
            for record in sorted(
                raw_records,
                key=lambda item: (
                    item.topic,
                    item.message_seq,
                    item.message_id,
                    item.consumer_group,
                    item.source_format,
                    item.line_no,
                ),
            ):
                if record.topic in missing_topics:
                    unknown_messages.append(
                        {
                            "topic": record.topic,
                            "message_seq": record.message_seq,
                            "message_id": record.message_id,
                            "consumer_group": record.consumer_group,
                            "status": "unknown_uncommitted_topic",
                        }
                    )
        return {
            "status": "complete" if state["execution_status"] == "complete" else "incomplete",
            "execution_status": state["execution_status"],
            "parameter_version": state["parameter_version"],
            "replay_ok": True,
            "replayed_message_count": len(replay["conclusions"]),
            "committed_topics": committed_topics,
            "missing_topics": missing_topics,
            "unknown_messages": unknown_messages,
            "rerun_required_topics": missing_topics,
            "exit_report_available": state["execution_status"] == "complete",
            "content_digest": state["content_digest"],
        }
