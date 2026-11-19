from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

from .canonical import canonical_json
from .errors import MeshError, diag
from .models import AttemptTrace, RequestResult, RoundSummary


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS revisions (
    revision_version INTEGER NOT NULL,
    revision_digest TEXT NOT NULL,
    workspace_id TEXT NOT NULL,
    canonical_text TEXT NOT NULL,
    revision_document_json TEXT NOT NULL,
    services_json TEXT NOT NULL,
    defaults_json TEXT NOT NULL,
    dependency_graph_version TEXT NOT NULL,
    instance_fingerprint TEXT NOT NULL,
    created_mono_ms INTEGER NOT NULL,
    PRIMARY KEY (revision_version, revision_digest, instance_fingerprint)
);
CREATE TABLE IF NOT EXISTS rounds (
    round_id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL,
    revision_version INTEGER NOT NULL,
    revision_digest TEXT NOT NULL,
    dependency_graph_version TEXT NOT NULL,
    instance_fingerprint TEXT NOT NULL,
    status TEXT NOT NULL,
    stale INTEGER NOT NULL DEFAULT 0,
    created_mono_ms INTEGER NOT NULL,
    summary_json TEXT NOT NULL,
    change_json TEXT,
    snapshot_path TEXT,
    restored_from TEXT
);
CREATE TABLE IF NOT EXISTS requests (
    round_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    request_index INTEGER NOT NULL,
    method TEXT NOT NULL,
    path TEXT NOT NULL,
    rule_id TEXT,
    service_id TEXT,
    outcome TEXT NOT NULL,
    reason TEXT NOT NULL,
    attempt_count INTEGER NOT NULL,
    instance_sequence_json TEXT NOT NULL,
    budget_consumed_ms INTEGER NOT NULL,
    wall_elapsed_ms INTEGER NOT NULL,
    unknown INTEGER NOT NULL,
    rerun_required INTEGER NOT NULL,
    detail TEXT NOT NULL,
    PRIMARY KEY (round_id, request_id),
    FOREIGN KEY (round_id) REFERENCES rounds(round_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS attempts (
    round_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    attempt_no INTEGER NOT NULL,
    rule_id TEXT NOT NULL,
    service_id TEXT NOT NULL,
    instance_id TEXT NOT NULL,
    outcome TEXT NOT NULL,
    reason TEXT NOT NULL,
    started_mono_ms INTEGER NOT NULL,
    ended_mono_ms INTEGER NOT NULL,
    elapsed_ms INTEGER NOT NULL,
    status_code INTEGER,
    detail TEXT NOT NULL,
    PRIMARY KEY (round_id, request_id, attempt_no),
    FOREIGN KEY (round_id) REFERENCES rounds(round_id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS current_requests (
    request_id TEXT PRIMARY KEY,
    round_id TEXT NOT NULL,
    request_index INTEGER NOT NULL,
    method TEXT NOT NULL,
    path TEXT NOT NULL,
    rule_id TEXT,
    service_id TEXT,
    outcome TEXT NOT NULL,
    reason TEXT NOT NULL,
    attempt_count INTEGER NOT NULL,
    instance_sequence_json TEXT NOT NULL,
    budget_consumed_ms INTEGER NOT NULL,
    wall_elapsed_ms INTEGER NOT NULL,
    unknown INTEGER NOT NULL,
    rerun_required INTEGER NOT NULL,
    detail TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS current_attempts (
    request_id TEXT NOT NULL,
    attempt_no INTEGER NOT NULL,
    round_id TEXT NOT NULL,
    rule_id TEXT NOT NULL,
    service_id TEXT NOT NULL,
    instance_id TEXT NOT NULL,
    outcome TEXT NOT NULL,
    reason TEXT NOT NULL,
    started_mono_ms INTEGER NOT NULL,
    ended_mono_ms INTEGER NOT NULL,
    elapsed_ms INTEGER NOT NULL,
    status_code INTEGER,
    detail TEXT NOT NULL,
    PRIMARY KEY (request_id, attempt_no)
);
CREATE TABLE IF NOT EXISTS state (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rounds_revision ON rounds(revision_version, stale);
CREATE INDEX IF NOT EXISTS idx_requests_round ON requests(round_id, request_index);
CREATE INDEX IF NOT EXISTS idx_attempts_round ON attempts(round_id, request_id, attempt_no);
"""


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path) if str(path) != ":memory:" else Path(":memory:")
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(
            str(path),
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA busy_timeout=30000")
        self.connection.executescript(SCHEMA_SQL)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self.connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            yield connection
        except Exception:
            connection.execute("ROLLBACK")
            raise
        else:
            connection.execute("COMMIT")

    @staticmethod
    def _mono_ms() -> int:
        return time.monotonic_ns() // 1_000_000

    def current_state(self) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT value FROM state WHERE key='current_revision'"
        ).fetchone()
        if row is None:
            return None
        value = json.loads(row["value"])
        if not isinstance(value, dict):
            return None
        return value

    def current_round_id(self) -> str | None:
        row = self.connection.execute(
            "SELECT value FROM state WHERE key='current_round_id'"
        ).fetchone()
        return None if row is None else str(json.loads(row["value"]))

    def assert_monotonic(
        self,
        revision_version: int,
        revision_digest: str,
        instance_fp: str,
    ) -> None:
        state = self.current_state()
        if state is None:
            return
        current_version = int(state.get("revision_version", 0))
        if revision_version < current_version:
            raise MeshError(
                "E_VERSION",
                "修订版本号必须单调递增",
                diagnostics=[
                    diag(
                        "E_VERSION",
                        "revision",
                        "新修订版本小于当前版本",
                        revision_version,
                        current_version=current_version,
                    )
                ],
            )
        if revision_version == current_version:
            if revision_digest != state.get("revision_digest"):
                raise MeshError(
                    "E_VERSION",
                    "同一版本号的规则内容发生变化",
                    diagnostics=[
                        diag(
                            "E_VERSION",
                            "revision",
                            "内容摘要变化时必须递增 revision",
                            revision_digest,
                            current_digest=state.get("revision_digest"),
                        )
                    ],
                )
            if instance_fp != state.get("instance_fingerprint"):
                raise MeshError(
                    "E_VERSION",
                    "同一版本号的实例集合发生变化",
                    diagnostics=[
                        diag(
                            "E_VERSION",
                            "services.instances",
                            "实例集合变化时必须递增 revision",
                            instance_fp,
                            current_instance_fingerprint=state.get("instance_fingerprint"),
                        )
                    ],
                )

    def _insert_current_materialized(
        self,
        connection: sqlite3.Connection,
        round_id: str,
        results: Iterable[RequestResult],
    ) -> None:
        connection.execute("DELETE FROM current_requests")
        connection.execute("DELETE FROM current_attempts")
        for result in sorted(results, key=lambda item: (item.request_index, item.request_id)):
            connection.execute(
                """
                INSERT INTO current_requests(
                    request_id, round_id, request_index, method, path, rule_id,
                    service_id, outcome, reason, attempt_count, instance_sequence_json,
                    budget_consumed_ms, wall_elapsed_ms, unknown, rerun_required, detail
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.request_id,
                    round_id,
                    result.request_index,
                    result.method,
                    result.path,
                    result.rule_id,
                    result.service_id,
                    result.outcome,
                    result.reason,
                    result.attempt_count,
                    canonical_json(list(result.instance_sequence)),
                    result.budget_consumed_ms,
                    result.wall_elapsed_ms,
                    int(result.unknown),
                    int(result.rerun_required),
                    result.detail,
                ),
            )
            for trace in sorted(result.traces, key=lambda item: item.attempt_no):
                connection.execute(
                    """
                    INSERT INTO current_attempts(
                        request_id, attempt_no, round_id, rule_id, service_id,
                        instance_id, outcome, reason, started_mono_ms, ended_mono_ms,
                        elapsed_ms, status_code, detail
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trace.request_id,
                        trace.attempt_no,
                        round_id,
                        trace.rule_id,
                        trace.service_id,
                        trace.instance_id,
                        trace.outcome,
                        trace.reason,
                        trace.started_mono_ms,
                        trace.ended_mono_ms,
                        trace.elapsed_ms,
                        trace.status_code,
                        trace.detail,
                    ),
                )

    def commit_round(
        self,
        summary: RoundSummary,
        *,
        revision,
        services,
        defaults: dict[str, Any],
        snapshot_path: str | None,
    ) -> None:
        results = summary.results
        with self.transaction() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO revisions(
                    revision_version, revision_digest, workspace_id, canonical_text,
                    revision_document_json, services_json, defaults_json,
                    dependency_graph_version, instance_fingerprint, created_mono_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    revision.revision_version,
                    revision.digest,
                    revision.workspace_id,
                    revision.canonical_text,
                    canonical_json(revision.source_document),
                    canonical_json([service.as_dict() for service in services]),
                    canonical_json(defaults),
                    revision.dependency_graph.version,
                    summary.instance_fingerprint,
                    self._mono_ms(),
                ),
            )
            connection.execute("UPDATE rounds SET stale=1 WHERE stale=0")
            connection.execute(
                """
                INSERT INTO rounds(
                    round_id, workspace_id, revision_version, revision_digest,
                    dependency_graph_version, instance_fingerprint, status, stale,
                    created_mono_ms, summary_json, change_json, snapshot_path, restored_from
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, NULL)
                """,
                (
                    summary.round_id,
                    revision.workspace_id,
                    summary.revision_version,
                    summary.revision_digest,
                    summary.dependency_graph_version,
                    summary.instance_fingerprint,
                    "completed",
                    self._mono_ms(),
                    canonical_json(summary.as_dict()),
                    canonical_json(summary.change.as_dict()) if summary.change else None,
                    snapshot_path,
                ),
            )
            for result in sorted(results, key=lambda item: (item.request_index, item.request_id)):
                connection.execute(
                    """
                    INSERT INTO requests(
                        round_id, request_id, request_index, method, path, rule_id,
                        service_id, outcome, reason, attempt_count, instance_sequence_json,
                        budget_consumed_ms, wall_elapsed_ms, unknown, rerun_required, detail
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        summary.round_id,
                        result.request_id,
                        result.request_index,
                        result.method,
                        result.path,
                        result.rule_id,
                        result.service_id,
                        result.outcome,
                        result.reason,
                        result.attempt_count,
                        canonical_json(list(result.instance_sequence)),
                        result.budget_consumed_ms,
                        result.wall_elapsed_ms,
                        int(result.unknown),
                        int(result.rerun_required),
                        result.detail,
                    ),
                )
                for trace in sorted(result.traces, key=lambda item: item.attempt_no):
                    connection.execute(
                        """
                        INSERT INTO attempts(
                            round_id, request_id, attempt_no, rule_id, service_id,
                            instance_id, outcome, reason, started_mono_ms, ended_mono_ms,
                            elapsed_ms, status_code, detail
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            summary.round_id,
                            trace.request_id,
                            trace.attempt_no,
                            trace.rule_id,
                            trace.service_id,
                            trace.instance_id,
                            trace.outcome,
                            trace.reason,
                            trace.started_mono_ms,
                            trace.ended_mono_ms,
                            trace.elapsed_ms,
                            trace.status_code,
                            trace.detail,
                        ),
                    )
            self._insert_current_materialized(connection, summary.round_id, results)
            state_value = {
                "workspace_id": revision.workspace_id,
                "revision_version": summary.revision_version,
                "revision_digest": summary.revision_digest,
                "dependency_graph_version": summary.dependency_graph_version,
                "instance_fingerprint": summary.instance_fingerprint,
                "round_id": summary.round_id,
            }
            connection.execute(
                "INSERT OR REPLACE INTO state(key, value) VALUES ('current_revision', ?)",
                (canonical_json(state_value),),
            )
            connection.execute(
                "INSERT OR REPLACE INTO state(key, value) VALUES ('current_round_id', ?)",
                (canonical_json(summary.round_id),),
            )

    def replace_with_snapshot(
        self,
        *,
        snapshot_id: str,
        revision,
        services,
        defaults: dict[str, Any],
        instance_fp: str,
        results: Iterable[RequestResult],
        status: str,
        snapshot_path: str,
    ) -> str:
        round_id = f"restore:{snapshot_id}"
        result_tuple = tuple(
            sorted(results, key=lambda item: (item.request_index, item.request_id))
        )
        summary = RoundSummary(
            round_id=round_id,
            revision_version=revision.revision_version,
            revision_digest=revision.digest,
            dependency_graph_version=revision.dependency_graph.version,
            instance_fingerprint=instance_fp,
            results=result_tuple,
            total_requests=len(result_tuple),
            success=sum(result.outcome == "success" for result in result_tuple),
            failure=sum(result.outcome == "failure" for result in result_tuple),
            timeout=sum(result.outcome == "timeout" for result in result_tuple),
            circuit_open=sum(result.outcome == "circuit_open" for result in result_tuple),
            unknown=sum(result.unknown for result in result_tuple),
            max_global_active=0,
            max_service_active={},
            snapshot_path=snapshot_path,
        )
        with self.transaction() as connection:
            connection.execute(
                "DELETE FROM attempts WHERE round_id=?",
                (round_id,),
            )
            connection.execute(
                "DELETE FROM requests WHERE round_id=?",
                (round_id,),
            )
            connection.execute("DELETE FROM rounds WHERE round_id=?", (round_id,))
            connection.execute(
                """
                INSERT OR IGNORE INTO revisions(
                    revision_version, revision_digest, workspace_id, canonical_text,
                    revision_document_json, services_json, defaults_json,
                    dependency_graph_version, instance_fingerprint, created_mono_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    revision.revision_version,
                    revision.digest,
                    revision.workspace_id,
                    revision.canonical_text,
                    canonical_json(revision.source_document),
                    canonical_json([service.as_dict() for service in services]),
                    canonical_json(defaults),
                    revision.dependency_graph.version,
                    instance_fp,
                    self._mono_ms(),
                ),
            )
            connection.execute("UPDATE rounds SET stale=1 WHERE stale=0")
            connection.execute(
                """
                INSERT INTO rounds(
                    round_id, workspace_id, revision_version, revision_digest,
                    dependency_graph_version, instance_fingerprint, status, stale,
                    created_mono_ms, summary_json, change_json, snapshot_path, restored_from
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, NULL, ?, ?)
                """,
                (
                    round_id,
                    revision.workspace_id,
                    revision.revision_version,
                    revision.digest,
                    revision.dependency_graph.version,
                    instance_fp,
                    status,
                    self._mono_ms(),
                    canonical_json(summary.as_dict()),
                    snapshot_path,
                    snapshot_id,
                ),
            )
            for result in result_tuple:
                connection.execute(
                    """
                    INSERT INTO requests(
                        round_id, request_id, request_index, method, path, rule_id,
                        service_id, outcome, reason, attempt_count, instance_sequence_json,
                        budget_consumed_ms, wall_elapsed_ms, unknown, rerun_required, detail
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        round_id,
                        result.request_id,
                        result.request_index,
                        result.method,
                        result.path,
                        result.rule_id,
                        result.service_id,
                        result.outcome,
                        result.reason,
                        result.attempt_count,
                        canonical_json(list(result.instance_sequence)),
                        result.budget_consumed_ms,
                        result.wall_elapsed_ms,
                        int(result.unknown),
                        int(result.rerun_required),
                        result.detail,
                    ),
                )
                for trace in sorted(result.traces, key=lambda item: item.attempt_no):
                    connection.execute(
                        """
                        INSERT INTO attempts(
                            round_id, request_id, attempt_no, rule_id, service_id,
                            instance_id, outcome, reason, started_mono_ms, ended_mono_ms,
                            elapsed_ms, status_code, detail
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            round_id,
                            trace.request_id,
                            trace.attempt_no,
                            trace.rule_id,
                            trace.service_id,
                            trace.instance_id,
                            trace.outcome,
                            trace.reason,
                            trace.started_mono_ms,
                            trace.ended_mono_ms,
                            trace.elapsed_ms,
                            trace.status_code,
                            trace.detail,
                        ),
                    )
            self._insert_current_materialized(connection, round_id, result_tuple)
            state_value = {
                "workspace_id": revision.workspace_id,
                "revision_version": revision.revision_version,
                "revision_digest": revision.digest,
                "dependency_graph_version": revision.dependency_graph.version,
                "instance_fingerprint": instance_fp,
                "round_id": round_id,
            }
            connection.execute(
                "INSERT OR REPLACE INTO state(key, value) VALUES ('current_revision', ?)",
                (canonical_json(state_value),),
            )
            connection.execute(
                "INSERT OR REPLACE INTO state(key, value) VALUES ('current_round_id', ?)",
                (canonical_json(round_id),),
            )
        return round_id

    def current_view(self) -> dict[str, Any] | None:
        round_id = self.current_round_id()
        if round_id is None:
            return None
        round_row = self.connection.execute(
            "SELECT * FROM rounds WHERE round_id=?", (round_id,)
        ).fetchone()
        if round_row is None:
            return None
        requests = []
        for row in self.connection.execute(
            "SELECT * FROM current_requests ORDER BY request_index, request_id"
        ):
            request_payload = dict(row)
            request_payload["instance_sequence"] = json.loads(
                request_payload.pop("instance_sequence_json")
            )
            request_payload["unknown"] = bool(request_payload["unknown"])
            request_payload["rerun_required"] = bool(request_payload["rerun_required"])
            requests.append(request_payload)
        attempts = []
        for row in self.connection.execute(
            "SELECT * FROM current_attempts ORDER BY request_id, attempt_no"
        ):
            attempts.append(dict(row))
        return {
            "round": dict(round_row),
            "requests": requests,
            "attempts": attempts,
        }

    def list_rounds(self) -> tuple[dict[str, Any], ...]:
        rows = []
        for row in self.connection.execute(
            "SELECT round_id, revision_version, stale, status, created_mono_ms, snapshot_path "
            "FROM rounds ORDER BY revision_version, round_id"
        ):
            rows.append(dict(row))
        return tuple(rows)

    def round_rows(self, round_id: str) -> dict[str, Any]:
        round_row = self.connection.execute(
            "SELECT * FROM rounds WHERE round_id=?", (round_id,)
        ).fetchone()
        if round_row is None:
            raise MeshError("E_LEDGER", "账本中不存在该轮次")
        requests = [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM requests WHERE round_id=? ORDER BY request_index, request_id",
                (round_id,),
            )
        ]
        attempts = [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM attempts WHERE round_id=? ORDER BY request_id, attempt_no",
                (round_id,),
            )
        ]
        return {"round": dict(round_row), "requests": requests, "attempts": attempts}

    def current_revision_documents(self) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]] | None:
        state = self.current_state()
        if state is None:
            return None
        row = self.connection.execute(
            """
            SELECT revision_document_json, defaults_json, services_json
            FROM revisions
            WHERE revision_version=? AND revision_digest=? AND instance_fingerprint=?
            ORDER BY created_mono_ms DESC LIMIT 1
            """,
            (
                int(state["revision_version"]),
                str(state["revision_digest"]),
                str(state["instance_fingerprint"]),
            ),
        ).fetchone()
        if row is None:
            return None
        return (
            json.loads(row["revision_document_json"]),
            json.loads(row["defaults_json"]),
            json.loads(row["services_json"]),
        )

    def previous_revision_documents(self):
        return self.current_revision_documents()
