"""SQLite ledger with commit-per-layer transactions and replay verification."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping
from uuid import uuid4

from .deps import build_dependency_graph
from .errors import LedgerError, ReplayError
from .models import (
    DiagnosticRun,
    InvalidationStep,
    LayerBatchCommit,
    LayerResult,
    Plan,
    SampleSet,
)
from .plan import plan_from_dict, sample_set_from_dict
from .state import run_from_snapshot_payload
from .util import canonical_digest, canonical_json


LEDGER_FORMAT_VERSION = 3


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


SCHEMA = """
CREATE TABLE IF NOT EXISTS ledger_meta (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    format_version INTEGER NOT NULL,
    content_digest TEXT,
    plan_version INTEGER,
    plan_digest TEXT,
    state_json TEXT,
    updated_at TEXT NOT NULL
);
INSERT OR IGNORE INTO ledger_meta
    (id, format_version, content_digest, plan_version, plan_digest, state_json, updated_at)
VALUES (1, 3, NULL, NULL, NULL, NULL, '');

CREATE TABLE IF NOT EXISTS plan_versions (
    plan_version INTEGER PRIMARY KEY,
    plan_digest TEXT NOT NULL,
    plan_json TEXT NOT NULL,
    samples_digest TEXT NOT NULL,
    samples_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    mode TEXT NOT NULL,
    plan_version INTEGER NOT NULL,
    plan_digest TEXT NOT NULL,
    plan_json TEXT NOT NULL,
    samples_digest TEXT NOT NULL,
    samples_json TEXT NOT NULL,
    expected_layers_json TEXT NOT NULL,
    changed_sources_json TEXT NOT NULL,
    invalidated_json TEXT NOT NULL,
    base_state_json TEXT,
    status TEXT NOT NULL,
    cursor TEXT NOT NULL,
    completed_layers INTEGER NOT NULL,
    total_layers INTEGER NOT NULL,
    progress_digest TEXT,
    started_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS layer_commits (
    session_id TEXT NOT NULL,
    layer_index INTEGER NOT NULL,
    layer_id TEXT NOT NULL,
    plan_version INTEGER NOT NULL,
    layer_version INTEGER NOT NULL,
    output_digest TEXT NOT NULL,
    float_digest TEXT NOT NULL,
    block_ids_json TEXT NOT NULL,
    request_bytes INTEGER NOT NULL,
    allocation_mode TEXT NOT NULL,
    error_json TEXT NOT NULL,
    result_json TEXT NOT NULL,
    committed_at TEXT NOT NULL,
    PRIMARY KEY (session_id, layer_id)
);

CREATE TABLE IF NOT EXISTS batch_commits (
    session_id TEXT NOT NULL,
    layer_index INTEGER NOT NULL,
    layer_id TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    sample_ids_json TEXT NOT NULL,
    block_ids_json TEXT NOT NULL,
    request_bytes INTEGER NOT NULL,
    peak_blocks INTEGER NOT NULL,
    fragmentation_blocks INTEGER NOT NULL,
    output_digest TEXT NOT NULL,
    float_digest TEXT NOT NULL,
    committed_at TEXT NOT NULL,
    PRIMARY KEY (session_id, layer_id, batch_id)
);

CREATE TABLE IF NOT EXISTS invalidation_events (
    session_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    layer_id TEXT NOT NULL,
    trigger_source TEXT NOT NULL,
    via TEXT,
    depth INTEGER NOT NULL,
    reason TEXT NOT NULL,
    path_json TEXT NOT NULL,
    PRIMARY KEY (session_id, layer_id)
);

CREATE TABLE IF NOT EXISTS unknown_layers (
    session_id TEXT NOT NULL,
    layer_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    marked_at TEXT NOT NULL,
    PRIMARY KEY (session_id, layer_id)
);

CREATE TABLE IF NOT EXISTS ledger_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    code TEXT NOT NULL,
    details_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS current_layers (
    plan_version INTEGER NOT NULL,
    layer_index INTEGER NOT NULL,
    layer_id TEXT NOT NULL,
    layer_version INTEGER NOT NULL,
    precision TEXT NOT NULL,
    scale INTEGER NOT NULL,
    output_digest TEXT NOT NULL,
    float_digest TEXT NOT NULL,
    block_ids_json TEXT NOT NULL,
    error_json TEXT NOT NULL,
    result_json TEXT NOT NULL,
    PRIMARY KEY (layer_id)
);

CREATE TABLE IF NOT EXISTS current_batches (
    batch_id TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS current_invalidations (
    seq INTEGER PRIMARY KEY,
    layer_id TEXT NOT NULL,
    trigger_source TEXT NOT NULL,
    via TEXT,
    depth INTEGER NOT NULL,
    reason TEXT NOT NULL,
    path_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS current_attribution (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id TEXT PRIMARY KEY,
    plan_version INTEGER NOT NULL,
    plan_digest TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    path TEXT NOT NULL,
    label TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path), timeout=30.0)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA busy_timeout=30000")
        self.connection.executescript(SCHEMA)
        self._ensure_column("batch_commits", "float_digest", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("sessions", "base_state_json", "TEXT")
        row = self.connection.execute("SELECT format_version FROM ledger_meta WHERE id=1").fetchone()
        if row is None:
            raise LedgerError("ledger meta row is missing")
        if int(row["format_version"]) != LEDGER_FORMAT_VERSION:
            raise LedgerError(
                "unsupported ledger format version",
                expected=LEDGER_FORMAT_VERSION,
                actual=int(row["format_version"]),
            )
        self.connection.execute(f"PRAGMA user_version={LEDGER_FORMAT_VERSION}")
        self.connection.commit()

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        existing = {
            str(row["name"])
            for row in self.connection.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in existing:
            self.connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield self.connection
        except BaseException:
            self.connection.rollback()
            raise
        else:
            self.connection.commit()

    def event(self, code: str, details: Mapping[str, Any], session_id: str | None = None) -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO ledger_events(session_id, code, details_json, created_at) VALUES (?, ?, ?, ?)",
                (session_id, code, canonical_json(dict(details)), _utc_now()),
            )

    def start_session(
        self,
        plan: Plan,
        samples: SampleSet,
        *,
        session_id: str | None = None,
        mode: str = "full",
        changed_sources: Mapping[str, str] | None = None,
        invalidated: Iterable[str] | None = None,
        base_run: DiagnosticRun | None = None,
    ) -> str:
        graph = build_dependency_graph(plan)
        expected = list(graph.topo)
        identifier = session_id or f"session-{uuid4().hex[:12]}"
        now = _utc_now()
        changed = {str(key): str(value) for key, value in (changed_sources or {}).items()}
        invalidated_list = list(invalidated or expected)
        try:
            with self.transaction() as connection:
                connection.execute(
                    """INSERT INTO plan_versions
                    (plan_version, plan_digest, plan_json, samples_digest, samples_json, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(plan_version) DO UPDATE SET
                        plan_digest=excluded.plan_digest,
                        plan_json=excluded.plan_json,
                        samples_digest=excluded.samples_digest,
                        samples_json=excluded.samples_json""",
                    (
                        plan.plan_version,
                        plan.digest,
                        canonical_json(plan.to_dict()),
                        samples.digest,
                        canonical_json(samples.to_dict()),
                        now,
                    ),
                )
                connection.execute(
                    """INSERT INTO sessions
                    (session_id, mode, plan_version, plan_digest, plan_json,
                     samples_digest, samples_json, expected_layers_json,
                     changed_sources_json, invalidated_json, base_state_json, status, cursor,
                     completed_layers, total_layers, progress_digest, started_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', '', 0, ?, NULL, ?, ?)""",
                    (
                        identifier,
                        mode,
                        plan.plan_version,
                        plan.digest,
                        canonical_json(plan.to_dict()),
                        samples.digest,
                        canonical_json(samples.to_dict()),
                        canonical_json(expected),
                        canonical_json(changed),
                        canonical_json(invalidated_list),
                        None if base_run is None else canonical_json(base_run.to_snapshot_dict()),
                        len(expected),
                        now,
                        now,
                    ),
                )
        except sqlite3.IntegrityError as error:
            raise LedgerError(
                "session id already exists", session_id=identifier
            ) from error
        return identifier

    def commit_layer(
        self,
        session_id: str,
        layer_index: int,
        result: LayerResult,
        batch_commits: Iterable[LayerBatchCommit],
    ) -> None:
        now = _utc_now()
        error_payload = {
            "error_bound": result.error_bound,
            "max_abs_error": result.max_abs_error,
            "error_components": result.error_components,
            "dominant_segment": result.dominant_segment,
        }
        result_json = canonical_json(result.to_dict())
        batch_list = list(batch_commits)
        progress_payload = {
            "layer_id": result.layer_id,
            "result": result.to_dict(),
            "batches": [batch.to_dict() for batch in batch_list],
        }
        progress_digest = canonical_digest(progress_payload)
        with self.transaction() as connection:
            session_row = connection.execute(
                "SELECT plan_version FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()
            if session_row is None:
                raise LedgerError("session does not exist", session_id=session_id)
            connection.execute(
                """INSERT INTO layer_commits
                (session_id, layer_index, layer_id, plan_version, layer_version,
                 output_digest, float_digest, block_ids_json, request_bytes,
                 allocation_mode, error_json, result_json, committed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, layer_id) DO UPDATE SET
                    layer_index=excluded.layer_index,
                    plan_version=excluded.plan_version,
                    layer_version=excluded.layer_version,
                    output_digest=excluded.output_digest,
                    float_digest=excluded.float_digest,
                    block_ids_json=excluded.block_ids_json,
                    request_bytes=excluded.request_bytes,
                    allocation_mode=excluded.allocation_mode,
                    error_json=excluded.error_json,
                    result_json=excluded.result_json,
                    committed_at=excluded.committed_at""",
                (
                    session_id,
                    layer_index,
                    result.layer_id,
                    int(session_row["plan_version"]),
                    result.version,
                    result.output_digest,
                    result.float_digest,
                    canonical_json(list(result.block_ids)),
                    result.request_bytes,
                    result.allocation_mode,
                    canonical_json(error_payload),
                    result_json,
                    now,
                ),
            )
            connection.execute(
                "DELETE FROM batch_commits WHERE session_id=? AND layer_id=?",
                (session_id, result.layer_id),
            )
            for batch in batch_list:
                connection.execute(
                    """INSERT INTO batch_commits
                    (session_id, layer_index, layer_id, batch_id, sample_ids_json,
                     block_ids_json, request_bytes, peak_blocks, fragmentation_blocks,
                     output_digest, float_digest, committed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        session_id,
                        layer_index,
                        result.layer_id,
                        batch.batch_id,
                        canonical_json(list(batch.sample_ids)),
                        canonical_json(list(batch.block_ids)),
                        batch.request_bytes,
                        batch.peak_blocks,
                        batch.fragmentation_blocks,
                        batch.output_digest,
                        batch.float_digest,
                        now,
                    ),
                )
            connection.execute(
                "DELETE FROM unknown_layers WHERE session_id=? AND layer_id=?",
                (session_id, result.layer_id),
            )
            count = connection.execute(
                "SELECT COUNT(*) AS value FROM layer_commits WHERE session_id=?",
                (session_id,),
            ).fetchone()["value"]
            connection.execute(
                """UPDATE sessions SET status='running', cursor=?, completed_layers=?,
                   progress_digest=?, updated_at=? WHERE session_id=?""",
                (result.layer_id, int(count), progress_digest, now, session_id),
            )

    def record_invalidation(
        self,
        session_id: str,
        trace: Iterable[InvalidationStep],
        changed_sources: Mapping[str, str] | None = None,
    ) -> None:
        trace_list = list(trace)
        with self.transaction() as connection:
            connection.execute(
                "DELETE FROM invalidation_events WHERE session_id=?", (session_id,)
            )
            for sequence, step in enumerate(trace_list):
                connection.execute(
                    """INSERT INTO invalidation_events
                    (session_id, seq, layer_id, trigger_source, via, depth, reason, path_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        session_id,
                        sequence,
                        step.layer_id,
                        step.trigger_source,
                        step.via,
                        step.depth,
                        step.reason,
                        canonical_json(list(step.path)),
                    ),
                )
            if changed_sources is not None:
                connection.execute(
                    "UPDATE sessions SET changed_sources_json=?, updated_at=? WHERE session_id=?",
                    (canonical_json(dict(changed_sources)), _utc_now(), session_id),
                )

    def complete_session(self, session_id: str, run: DiagnosticRun) -> None:
        now = _utc_now()
        with self.transaction() as connection:
            session = connection.execute(
                "SELECT expected_layers_json FROM sessions WHERE session_id=?", (session_id,)
            ).fetchone()
            if session is None:
                raise LedgerError("session does not exist", session_id=session_id)
            expected = list(__import__("json").loads(session["expected_layers_json"]))
            count = connection.execute(
                "SELECT COUNT(*) AS value FROM layer_commits WHERE session_id=?",
                (session_id,),
            ).fetchone()["value"]
            if int(count) != len(expected):
                raise LedgerError(
                    "cannot complete session with uncommitted layer batches",
                    session_id=session_id,
                    expected_layers=len(expected),
                    committed_layers=int(count),
                )
            state_payload = run.to_snapshot_dict()
            connection.execute(
                """UPDATE sessions SET status='complete', cursor=?, completed_layers=?,
                   total_layers=?, updated_at=? WHERE session_id=?""",
                (expected[-1] if expected else "", len(expected), len(expected), now, session_id),
            )
            self._replace_current_state_tx(
                connection,
                run,
                state_payload,
            )

    def _replace_current_state_tx(
        self,
        connection: sqlite3.Connection,
        run: DiagnosticRun,
        state_payload: Mapping[str, Any] | None = None,
    ) -> None:
        payload = dict(state_payload or run.to_snapshot_dict())
        now = _utc_now()
        connection.execute("DELETE FROM current_layers")
        connection.execute("DELETE FROM current_batches")
        connection.execute("DELETE FROM current_invalidations")
        connection.execute("DELETE FROM current_attribution")
        for index, layer_id in enumerate(run.layers):
            result = run.layers[layer_id]
            connection.execute(
                """INSERT INTO current_layers
                (plan_version, layer_index, layer_id, layer_version, precision, scale,
                 output_digest, float_digest, block_ids_json, error_json, result_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run.plan.plan_version,
                    index,
                    layer_id,
                    result.version,
                    result.precision,
                    result.scale,
                    result.output_digest,
                    result.float_digest,
                    canonical_json(list(result.block_ids)),
                    canonical_json(
                        {
                            "error_bound": result.error_bound,
                            "max_abs_error": result.max_abs_error,
                            "components": result.error_components,
                            "dominant_segment": result.dominant_segment,
                        }
                    ),
                    canonical_json(result.to_dict()),
                ),
            )
        for batch in run.batches:
            connection.execute(
                "INSERT INTO current_batches(batch_id, payload_json) VALUES (?, ?)",
                (batch.batch_id, canonical_json(batch.to_dict())),
            )
        for sequence, step in enumerate(run.invalidation):
            connection.execute(
                """INSERT INTO current_invalidations
                (seq, layer_id, trigger_source, via, depth, reason, path_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    sequence,
                    step.layer_id,
                    step.trigger_source,
                    step.via,
                    step.depth,
                    step.reason,
                    canonical_json(list(step.path)),
                ),
            )
        connection.execute(
            "INSERT INTO current_attribution(id, payload_json) VALUES (1, ?)",
            (canonical_json(run.attribution),),
        )
        connection.execute(
            """UPDATE ledger_meta SET content_digest=?, plan_version=?, plan_digest=?,
               state_json=?, updated_at=? WHERE id=1""",
            (
                run.content_digest,
                run.plan.plan_version,
                run.plan.digest,
                canonical_json(payload),
                now,
            ),
        )

    def replace_current_state(self, run: DiagnosticRun, *, session_id: str | None = None) -> None:
        with self.transaction() as connection:
            self._replace_current_state_tx(connection, run)
            connection.execute(
                "INSERT INTO ledger_events(session_id, code, details_json, created_at) VALUES (?, ?, ?, ?)",
                (
                    session_id,
                    "STATE_WHOLESALE_REPLACE",
                    canonical_json(
                        {
                            "plan_version": run.plan.plan_version,
                            "content_digest": run.content_digest,
                        }
                    ),
                    _utc_now(),
                ),
            )

    def load_current_state_payload(self) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT state_json FROM ledger_meta WHERE id=1"
        ).fetchone()
        if row is None or row["state_json"] is None:
            return None
        import json

        return dict(json.loads(row["state_json"]))

    def load_current_run(self) -> DiagnosticRun | None:
        payload = self.load_current_state_payload()
        return None if payload is None else run_from_snapshot_payload(payload)

    def current_meta(self) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT format_version, content_digest, plan_version, plan_digest, updated_at FROM ledger_meta WHERE id=1"
        ).fetchone()
        assert row is not None
        return {key: row[key] for key in row.keys()}

    def session(self, session_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT * FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
        if row is None:
            raise LedgerError("session does not exist", session_id=session_id)
        return {key: row[key] for key in row.keys()}

    def session_plan_samples(self, session_id: str) -> tuple[Plan, SampleSet]:
        row = self.session(session_id)
        import json

        return (
            plan_from_dict(json.loads(row["plan_json"])),
            sample_set_from_dict(json.loads(row["samples_json"])),
        )

    def session_base_run(self, session_id: str) -> DiagnosticRun | None:
        import json

        row = self.session(session_id)
        if row.get("base_state_json") is None:
            return None
        return run_from_snapshot_payload(json.loads(row["base_state_json"]))

    def session_changed_sources(self, session_id: str) -> dict[str, str]:
        import json

        row = self.session(session_id)
        return {
            str(key): str(value)
            for key, value in json.loads(row["changed_sources_json"]).items()
        }

    def expected_layers(self, session_id: str) -> tuple[str, ...]:
        import json

        row = self.session(session_id)
        return tuple(str(item) for item in json.loads(row["expected_layers_json"]))

    def committed_layer_ids(self, session_id: str) -> tuple[str, ...]:
        rows = self.connection.execute(
            "SELECT layer_id FROM layer_commits WHERE session_id=? ORDER BY layer_index",
            (session_id,),
        ).fetchall()
        return tuple(str(row["layer_id"]) for row in rows)

    def committed_layer_results(self, session_id: str) -> tuple[LayerResult, ...]:
        rows = self.connection.execute(
            "SELECT result_json FROM layer_commits WHERE session_id=? ORDER BY layer_index",
            (session_id,),
        ).fetchall()
        import json

        return tuple(LayerResult.from_dict(json.loads(row["result_json"])) for row in rows)

    def recover_session(self, session_id: str) -> dict[str, Any]:
        session = self.session(session_id)
        expected = self.expected_layers(session_id)
        committed = self.committed_layer_ids(session_id)
        committed_set = set(committed)
        unknown = [layer_id for layer_id in expected if layer_id not in committed_set]
        now = _utc_now()
        with self.transaction() as connection:
            for layer_id in unknown:
                connection.execute(
                    """INSERT INTO unknown_layers(session_id, layer_id, reason, marked_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(session_id, layer_id) DO UPDATE SET reason=excluded.reason,
                    marked_at=excluded.marked_at""",
                    (session_id, layer_id, "uncommitted_layer_batch", now),
                )
            status = "complete" if session["status"] == "complete" else "needs_rerun"
            connection.execute(
                "UPDATE sessions SET status=?, updated_at=? WHERE session_id=?",
                (status, now, session_id),
            )
            connection.execute(
                "INSERT INTO ledger_events(session_id, code, details_json, created_at) VALUES (?, ?, ?, ?)",
                (
                    session_id,
                    "SESSION_RECOVERY_SCAN" if status != "complete" else "SESSION_COMPLETE_SCAN",
                    canonical_json(
                        {
                            "committed_layers": list(committed),
                            "unknown_layers": unknown,
                            "exit_report_available": status == "complete",
                        }
                    ),
                    now,
                ),
            )
        complete = status == "complete"
        return {
            "ok": complete,
            "code": None if complete else "E_RECOVERY",
            "session_id": session_id,
            "status": status,
            "committed_layers": list(committed),
            "unknown_layers": unknown,
            "unknown_batches": [
                {"layer_id": layer_id, "batch_ids": "all"} for layer_id in unknown
            ],
            "exit_report_available": complete,
            "action_required": None if complete else "rerun_uncommitted_layers",
            "message": (
                "session complete"
                if complete
                else "only fully committed layer batches are trusted; uncommitted batches are unknown and must be rerun"
            ),
        }

    def verify_session_commits(self, session_id: str, reference: DiagnosticRun) -> dict[str, Any]:
        session = self.session(session_id)
        if str(session["plan_digest"]) != reference.plan.digest:
            raise ReplayError(
                "recorded plan digest differs from replay plan",
                layer_id="<plan>",
                field="plan_digest",
                recorded=session["plan_digest"],
                replayed=reference.plan.digest,
                basis={
                    "session_id": session_id,
                    "recorded_plan_version": session["plan_version"],
                    "replayed_plan_version": reference.plan.plan_version,
                },
            )
        rows = self.connection.execute(
            "SELECT * FROM layer_commits WHERE session_id=? ORDER BY layer_index",
            (session_id,),
        ).fetchall()
        import json

        checked: list[str] = []
        for row in rows:
            layer_id = str(row["layer_id"])
            if int(row["plan_version"]) != reference.plan.plan_version:
                raise ReplayError(
                    "recorded layer plan version differs from replay plan",
                    layer_id=layer_id,
                    field="plan_version",
                    recorded=int(row["plan_version"]),
                    replayed=reference.plan.plan_version,
                    basis={"session_id": session_id},
                )
            if layer_id not in reference.layers:
                raise ReplayError(
                    "recorded layer is absent from replay",
                    layer_id=layer_id,
                    field="layer",
                    recorded="present",
                    replayed="missing",
                    basis={"session_id": session_id},
                )
            result = reference.layers[layer_id]
            recorded_result = json.loads(row["result_json"])
            recorded_spec = recorded_result.get("spec", {})
            for spec_field in (
                "id",
                "precision",
                "scale",
                "input_dim",
                "output_dim",
                "param_count",
            ):
                recorded_value = recorded_spec.get(spec_field)
                replayed_value = result.spec.get(spec_field)
                if recorded_value != replayed_value:
                    raise ReplayError(
                        "recorded layer specification differs from replay plan",
                        layer_id=layer_id,
                        field=f"spec.{spec_field}",
                        recorded=recorded_value,
                        replayed=replayed_value,
                        basis={
                            "session_id": session_id,
                            "recorded_plan_version": session["plan_version"],
                            "recorded_layer_version": int(row["layer_version"]),
                        },
                    )
            recorded = {
                "output_digest": row["output_digest"],
                "float_digest": row["float_digest"],
                "block_ids": tuple(json.loads(row["block_ids_json"])),
                "request_bytes": int(row["request_bytes"]),
                "allocation_mode": row["allocation_mode"],
            }
            current = {
                "output_digest": result.output_digest,
                "float_digest": result.float_digest,
                "block_ids": result.block_ids,
                "request_bytes": result.request_bytes,
                "allocation_mode": result.allocation_mode,
            }
            recorded_error = json.loads(row["error_json"])
            current_error = {
                "error_bound": result.error_bound,
                "max_abs_error": result.max_abs_error,
                "error_components": result.error_components,
                "dominant_segment": result.dominant_segment,
            }
            for field_name in (
                "output_digest",
                "float_digest",
                "block_ids",
                "request_bytes",
                "allocation_mode",
            ):
                if recorded[field_name] != current[field_name]:
                    raise ReplayError(
                        "replayed layer conclusion differs from committed ledger",
                        layer_id=layer_id,
                        field=field_name,
                        recorded=recorded[field_name],
                        replayed=current[field_name],
                        basis={
                            "session_id": session_id,
                            "recorded_plan_version": session["plan_version"],
                            "recorded_layer_version": int(row["layer_version"]),
                            "commit_order": int(row["layer_index"]),
                        },
                    )
            for field_name in recorded_error:
                if recorded_error[field_name] != current_error[field_name]:
                    raise ReplayError(
                        "replayed error attribution differs from committed ledger",
                        layer_id=layer_id,
                        field=field_name,
                        recorded=recorded_error[field_name],
                        replayed=current_error[field_name],
                        basis={
                            "session_id": session_id,
                            "recorded_plan_version": session["plan_version"],
                            "recorded_layer_version": int(row["layer_version"]),
                            "commit_order": int(row["layer_index"]),
                        },
                    )
            batch_rows = self.connection.execute(
                "SELECT * FROM batch_commits WHERE session_id=? AND layer_id=? ORDER BY batch_id",
                (session_id, layer_id),
            ).fetchall()
            expected_batches = {
                f"b{index:04d}": sample_ids
                for index, sample_ids in enumerate(
                    [
                        reference.samples.ordered_ids[start : start + reference.plan.max_batch_size]
                        for start in range(0, len(reference.samples.ordered_ids), reference.plan.max_batch_size)
                    ]
                )
            }
            if len(batch_rows) != len(expected_batches):
                raise ReplayError(
                    "committed batch count differs from replay",
                    layer_id=layer_id,
                    field="batch_count",
                    recorded=len(batch_rows),
                    replayed=len(expected_batches),
                    basis={"session_id": session_id},
                )
            for batch_row in batch_rows:
                batch_id = str(batch_row["batch_id"])
                sample_ids = tuple(json.loads(batch_row["sample_ids_json"]))
                from .models import digest_sample_subset

                expected_digest = digest_sample_subset(
                    result.sample_output_digests, sample_ids
                )
                expected_float_digest = digest_sample_subset(
                    result.sample_float_digests, sample_ids
                )
                expected_sample_ids = expected_batches.get(batch_id, ())
                if sample_ids != tuple(expected_sample_ids):
                    raise ReplayError(
                        "replayed batch sample membership differs from committed ledger",
                        layer_id=layer_id,
                        field=f"batch[{batch_id}].sample_ids",
                        recorded=list(sample_ids),
                        replayed=list(expected_sample_ids),
                        basis={"session_id": session_id, "batch_id": batch_id},
                    )
                recorded_batch_blocks = tuple(json.loads(batch_row["block_ids_json"]))
                if recorded_batch_blocks != result.block_ids:
                    raise ReplayError(
                        "replayed batch block order differs from committed ledger",
                        layer_id=layer_id,
                        field=f"batch[{batch_id}].block_ids",
                        recorded=list(recorded_batch_blocks),
                        replayed=list(result.block_ids),
                        basis={"session_id": session_id, "batch_id": batch_id},
                    )
                from .engine import layer_request_bytes

                reference_batch = next(
                    (batch for batch in reference.batches if batch.batch_id == batch_id),
                    None,
                )
                expected_batch_values = {
                    "request_bytes": layer_request_bytes(
                        reference.plan.layer_map[layer_id], len(sample_ids)
                    ),
                    "peak_blocks": None if reference_batch is None else reference_batch.peak_blocks,
                    "fragmentation_blocks": None
                    if reference_batch is None
                    else reference_batch.fragmentation_blocks,
                }
                for field_name, expected_value in expected_batch_values.items():
                    replayed_value = expected_value
                    recorded_value = int(batch_row[field_name])
                    if recorded_value != replayed_value:
                        raise ReplayError(
                            "replayed batch memory accounting differs from committed ledger",
                            layer_id=layer_id,
                            field=f"batch[{batch_id}].{field_name}",
                            recorded=recorded_value,
                            replayed=replayed_value,
                            basis={"session_id": session_id, "batch_id": batch_id},
                        )
                if str(batch_row["output_digest"]) != expected_digest:
                    raise ReplayError(
                        "replayed batch output digest differs from committed ledger",
                        layer_id=layer_id,
                        field=f"batch[{batch_id}].output_digest",
                        recorded=batch_row["output_digest"],
                        replayed=expected_digest,
                        basis={"session_id": session_id, "batch_id": batch_id},
                    )
                if str(batch_row["float_digest"]) != expected_float_digest:
                    raise ReplayError(
                        "replayed float batch output digest differs from committed ledger",
                        layer_id=layer_id,
                        field=f"batch[{batch_id}].float_digest",
                        recorded=batch_row["float_digest"],
                        replayed=expected_float_digest,
                        basis={"session_id": session_id, "batch_id": batch_id},
                    )
            checked.append(layer_id)
        return {
            "ok": True,
            "session_id": session_id,
            "checked_layers": checked,
            "plan_version": int(session["plan_version"]),
            "plan_digest": session["plan_digest"],
            "basis": "committed layer_commits + batch_commits + recorded plan/scale",
        }

    def register_snapshot(
        self,
        *,
        snapshot_id: str,
        plan_version: int,
        plan_digest: str,
        content_digest: str,
        path: str,
        label: str,
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                """INSERT INTO snapshots
                (snapshot_id, plan_version, plan_digest, content_digest, path, label, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(snapshot_id) DO UPDATE SET
                    plan_version=excluded.plan_version,
                    plan_digest=excluded.plan_digest,
                    content_digest=excluded.content_digest,
                    path=excluded.path,
                    label=excluded.label""",
                (
                    snapshot_id,
                    plan_version,
                    plan_digest,
                    content_digest,
                    path,
                    label,
                    _utc_now(),
                ),
            )

    def list_snapshots(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM snapshots ORDER BY plan_version, snapshot_id"
        ).fetchall()
        return [{key: row[key] for key in row.keys()} for row in rows]

    def mark_session_running(self, session_id: str) -> None:
        with self.transaction() as connection:
            connection.execute(
                "UPDATE sessions SET status='running', updated_at=? WHERE session_id=?",
                (_utc_now(), session_id),
            )

    def list_sessions(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT session_id, status, plan_version, cursor, completed_layers, total_layers, updated_at FROM sessions ORDER BY started_at DESC"
        ).fetchall()
        return [{key: row[key] for key in row.keys()} for row in rows]
