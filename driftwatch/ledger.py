from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from .config import parse_agent_manifest, parse_criteria, parse_snapshot_chain
from .errors import DriftError
from .models import (
    METRICS,
    AffectedItem,
    AgentSpec,
    AttemptRecord,
    Criteria,
    DriftResult,
    MergedResult,
    Observation,
    SnapshotChain,
    agents_version,
    canonical_json,
    declaration_digest,
    digest_json,
)
from .propagation import PropagationResult, affected_key_payload, derive_affected

LEDGER_FORMAT_VERSION = 1


def now_ms() -> int:
    return time.monotonic_ns() // 1_000_000


@dataclass
class ReconcileResult:
    generation: int
    replaced: bool
    previous_generation: int | None
    affected: list[AffectedItem]
    differences: list[dict[str, object]]
    snapshot_chain_version: str
    agent_set_version: str
    scale_version: str
    declaration_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "replaced": self.replaced,
            "previous_generation": self.previous_generation,
            "affected": [item.to_dict() for item in self.affected],
            "differences": self.differences,
            "snapshot_chain_version": self.snapshot_chain_version,
            "agent_set_version": self.agent_set_version,
            "scale_version": self.scale_version,
            "declaration_digest": self.declaration_digest,
        }



def _semantic_view_digests(
    merged_view: list[dict[str, Any]], drift_view: list[dict[str, Any]]
) -> tuple[str, str, str]:
    merged_semantic: list[dict[str, Any]] = []
    for row in merged_view:
        item = {
            key: value
            for key, value in row.items()
            if key not in {"batch_id", "generation", "stale"}
        }
        item["sources"] = [
            {key: value for key, value in source.items() if key != "monotonic_ms"}
            for source in item.get("sources", [])
        ]
        merged_semantic.append(item)
    drift_semantic = [
        {
            key: value
            for key, value in row.items()
            if key not in {"batch_id", "generation", "stale"}
        }
        for row in drift_view
    ]
    merged_digest = digest_json(merged_semantic)
    drift_digest = digest_json(drift_semantic)
    conclusion_digest = digest_json({"merged": merged_semantic, "drift": drift_semantic})
    return merged_digest, drift_digest, conclusion_digest


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = str(Path(path))
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA busy_timeout=30000")
        self._initialize_schema()

    def close(self) -> None:
        with self._lock:
            self.connection.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                yield self.connection
            except BaseException:
                self.connection.execute("ROLLBACK")
                raise
            else:
                self.connection.execute("COMMIT")

    def _initialize_schema(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS ledger_info (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            format_version INTEGER NOT NULL,
            content_digest TEXT NOT NULL,
            updated_ms INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS generations (
            generation INTEGER PRIMARY KEY AUTOINCREMENT,
            active INTEGER NOT NULL,
            created_ms INTEGER NOT NULL,
            declaration_digest TEXT NOT NULL,
            snapshot_chain_version TEXT NOT NULL,
            agent_set_version TEXT NOT NULL,
            scale_version TEXT NOT NULL,
            snapshot_json TEXT NOT NULL,
            agents_json TEXT NOT NULL,
            criteria_json TEXT NOT NULL,
            merged_view_digest TEXT,
            conclusion_digest TEXT
        );
        CREATE UNIQUE INDEX IF NOT EXISTS one_active_generation
            ON generations(active) WHERE active = 1;
        CREATE TABLE IF NOT EXISTS affected_items (
            generation INTEGER NOT NULL REFERENCES generations(generation),
            snapshot_seq INTEGER NOT NULL,
            host_id TEXT NOT NULL,
            metric_id TEXT NOT NULL,
            reasons_json TEXT NOT NULL,
            stale_applied INTEGER NOT NULL,
            PRIMARY KEY (generation, snapshot_seq, host_id, metric_id)
        );
        CREATE TABLE IF NOT EXISTS generation_diffs (
            generation INTEGER NOT NULL REFERENCES generations(generation),
            ordinal INTEGER NOT NULL,
            kind TEXT NOT NULL,
            snapshot_seq INTEGER,
            host_id TEXT,
            metric_id TEXT,
            agent_id TEXT,
            old_value TEXT,
            new_value TEXT,
            basis TEXT NOT NULL,
            PRIMARY KEY (generation, ordinal)
        );
        CREATE TABLE IF NOT EXISTS session_snapshots (
            generation INTEGER PRIMARY KEY REFERENCES generations(generation),
            snapshot_chain_version TEXT NOT NULL,
            agent_set_version TEXT NOT NULL,
            scale_version TEXT NOT NULL,
            merged_view_json TEXT NOT NULL,
            merged_view_digest TEXT NOT NULL,
            drift_view_json TEXT NOT NULL,
            drift_view_digest TEXT NOT NULL,
            conclusion_digest TEXT NOT NULL,
            updated_ms INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS batches (
            batch_id TEXT PRIMARY KEY,
            generation INTEGER NOT NULL REFERENCES generations(generation),
            snapshot_seq INTEGER NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('RUNNING','COMMITTED','ABORTED')),
            started_ms INTEGER NOT NULL,
            committed_ms INTEGER,
            cursor_json TEXT,
            planned_json TEXT NOT NULL,
            observations_digest TEXT,
            conclusions_digest TEXT,
            content_digest TEXT,
            abort_reason TEXT
        );
        CREATE INDEX IF NOT EXISTS batch_generation_seq ON batches(generation, snapshot_seq, status);
        CREATE TABLE IF NOT EXISTS observations (
            batch_id TEXT NOT NULL REFERENCES batches(batch_id),
            agent_id TEXT NOT NULL,
            host_id TEXT NOT NULL,
            metric_id TEXT NOT NULL,
            observation_seq INTEGER NOT NULL,
            value INTEGER NOT NULL,
            aggregation_version TEXT NOT NULL,
            monotonic_ms INTEGER NOT NULL,
            is_authority INTEGER NOT NULL,
            deviation INTEGER,
            PRIMARY KEY (batch_id, agent_id, host_id, metric_id)
        );
        CREATE TABLE IF NOT EXISTS attempts (
            batch_id TEXT NOT NULL REFERENCES batches(batch_id),
            task_key TEXT NOT NULL,
            attempt_no INTEGER NOT NULL,
            is_retry INTEGER NOT NULL,
            agent_id TEXT NOT NULL,
            snapshot_seq INTEGER NOT NULL,
            host_id TEXT NOT NULL,
            metric_id TEXT NOT NULL,
            observation_seq INTEGER NOT NULL,
            wait_start_ms INTEGER NOT NULL,
            wait_end_ms INTEGER NOT NULL,
            run_start_ms INTEGER NOT NULL,
            run_end_ms INTEGER NOT NULL,
            status TEXT NOT NULL,
            error_code TEXT,
            message TEXT NOT NULL,
            value INTEGER,
            PRIMARY KEY (batch_id, task_key, attempt_no)
        );
        CREATE TABLE IF NOT EXISTS merged_rows (
            batch_id TEXT NOT NULL REFERENCES batches(batch_id),
            generation INTEGER NOT NULL,
            snapshot_seq INTEGER NOT NULL,
            host_id TEXT NOT NULL,
            metric_id TEXT NOT NULL,
            status TEXT NOT NULL,
            authority_agent TEXT,
            authority_observation_seq INTEGER,
            authority_value INTEGER,
            aggregation_version TEXT,
            source_count INTEGER NOT NULL,
            sparse INTEGER NOT NULL,
            sources_json TEXT NOT NULL,
            max_abs_deviation INTEGER NOT NULL,
            error_code TEXT,
            stale INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (batch_id, host_id, metric_id)
        );
        CREATE INDEX IF NOT EXISTS merged_view ON merged_rows(generation, snapshot_seq, host_id, metric_id);
        CREATE TABLE IF NOT EXISTS drift_rows (
            batch_id TEXT NOT NULL REFERENCES batches(batch_id),
            generation INTEGER NOT NULL,
            snapshot_seq INTEGER NOT NULL,
            host_id TEXT NOT NULL,
            metric_id TEXT NOT NULL,
            status TEXT NOT NULL,
            decision TEXT NOT NULL,
            expected INTEGER,
            observed INTEGER,
            tolerance_bps INTEGER,
            drift_signed_fixed INTEGER,
            drift_excess_fixed INTEGER,
            drift_ratio_bps INTEGER,
            allowed_fixed INTEGER,
            error_bound_json TEXT,
            error_affected INTEGER NOT NULL,
            boundary_rule TEXT NOT NULL,
            saturated INTEGER NOT NULL,
            error_code TEXT,
            reason TEXT NOT NULL,
            result_json TEXT NOT NULL,
            stale INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (batch_id, host_id, metric_id)
        );
        CREATE INDEX IF NOT EXISTS drift_view ON drift_rows(generation, snapshot_seq, host_id, metric_id);
        CREATE TABLE IF NOT EXISTS cursors (
            generation INTEGER PRIMARY KEY REFERENCES generations(generation),
            last_committed_snapshot_seq INTEGER,
            last_batch_id TEXT,
            updated_ms INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS recovery_unknown (
            generation INTEGER NOT NULL REFERENCES generations(generation),
            batch_id TEXT NOT NULL,
            snapshot_seq INTEGER NOT NULL,
            host_id TEXT NOT NULL,
            metric_id TEXT NOT NULL,
            marked_ms INTEGER NOT NULL,
            reason TEXT NOT NULL,
            required_recollect INTEGER NOT NULL DEFAULT 1,
            recollected_batch_id TEXT,
            PRIMARY KEY (generation, snapshot_seq, host_id, metric_id)
        );
        """
        self.connection.executescript(schema)
        row = self.connection.execute("SELECT * FROM ledger_info WHERE singleton=1").fetchone()
        if row is None:
            self.connection.execute(
                "INSERT INTO ledger_info(singleton, format_version, content_digest, updated_ms) VALUES(1,?,?,?)",
                (LEDGER_FORMAT_VERSION, "", now_ms()),
            )
            with self._transaction() as conn:
                self._update_ledger_digest(conn)
        else:
            if int(row["format_version"]) != LEDGER_FORMAT_VERSION:
                raise DriftError(
                    "E_REPLAY",
                    "unsupported SQLite ledger format version",
                    details={
                        "ledger_format_version": int(row["format_version"]),
                        "supported": LEDGER_FORMAT_VERSION,
                    },
                )
            expected = self._compute_ledger_digest()
            if row["content_digest"] != expected:
                raise DriftError(
                    "E_REPLAY",
                    "SQLite ledger content digest mismatch",
                    details={"field": "ledger.content_digest", "expected": expected, "actual": row["content_digest"]},
                )

    def _compute_ledger_digest(self) -> str:
        rows = self.connection.execute(
            "SELECT batch_id, content_digest FROM batches WHERE status='COMMITTED' ORDER BY batch_id"
        ).fetchall()
        return digest_json(
            {
                "format_version": LEDGER_FORMAT_VERSION,
                "batches": [{"batch_id": row["batch_id"], "content_digest": row["content_digest"]} for row in rows],
            }
        )

    def _update_ledger_digest(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            "SELECT batch_id, content_digest FROM batches WHERE status='COMMITTED' ORDER BY batch_id"
        ).fetchall()
        digest = digest_json(
            {
                "format_version": LEDGER_FORMAT_VERSION,
                "batches": [{"batch_id": row["batch_id"], "content_digest": row["content_digest"]} for row in rows],
            }
        )
        conn.execute(
            "UPDATE ledger_info SET content_digest=?, updated_ms=? WHERE singleton=1",
            (digest, now_ms()),
        )

    def ledger_info(self) -> dict[str, Any]:
        with self._lock:
            row = self.connection.execute("SELECT * FROM ledger_info WHERE singleton=1").fetchone()
            assert row is not None
            return {
                "format_version": int(row["format_version"]),
                "content_digest": row["content_digest"],
                "updated_ms": int(row["updated_ms"]),
            }

    def active_generation(self) -> int | None:
        with self._lock:
            row = self.connection.execute("SELECT generation FROM generations WHERE active=1").fetchone()
            return int(row["generation"]) if row else None

    def reconcile_session(
        self,
        chain: SnapshotChain,
        agents: tuple[AgentSpec, ...],
        criteria: Criteria,
    ) -> ReconcileResult:
        current = {
            "declaration_digest": declaration_digest(chain, agents, criteria),
            "snapshot_chain_version": chain.version(),
            "agent_set_version": agents_version(agents),
            "scale_version": criteria.scale_version(),
        }
        with self._lock:
            active_row = self.connection.execute("SELECT * FROM generations WHERE active=1").fetchone()
            if active_row is not None and all(active_row[key] == current[key] for key in current):
                mismatches = self._session_snapshot_mismatches(active_row, current)
                if not mismatches:
                    return ReconcileResult(
                        generation=int(active_row["generation"]),
                        replaced=False,
                        previous_generation=None,
                        affected=[],
                        differences=[],
                        **current,
                    )
            else:
                mismatches = []

            previous_generation = int(active_row["generation"]) if active_row is not None else None
            propagation = PropagationResult(affected=[], differences=[])
            if active_row is not None:
                old_chain, old_agents, old_criteria = self._parse_generation_declarations(active_row)
                propagation = derive_affected(
                    old_chain,
                    chain,
                    old_agents,
                    agents,
                    old_criteria,
                    criteria,
                )
                if mismatches:
                    propagation.differences = sorted(
                        propagation.differences
                        + [
                            {
                                "kind": "session_snapshot_mismatch",
                                "field": mismatch["field"],
                                "old": mismatch["actual"],
                                "new": mismatch["expected"],
                                "basis": "persisted session digest or version is inconsistent with declarations",
                            }
                            for mismatch in mismatches
                        ],
                        key=lambda item: (
                            str(item.get("kind")),
                            str(item.get("field", "")),
                            str(item.get("snapshot_seq", "")),
                        ),
                    )

            with self._transaction() as conn:
                if active_row is not None:
                    conn.execute("UPDATE generations SET active=0 WHERE generation=?", (previous_generation,))
                    for item in propagation.affected:
                        conn.execute(
                            "UPDATE merged_rows SET stale=1 WHERE generation=? AND snapshot_seq=? AND host_id=? AND metric_id=?",
                            (previous_generation, item.snapshot_seq, item.host_id, item.metric_id),
                        )
                        conn.execute(
                            "UPDATE drift_rows SET stale=1 WHERE generation=? AND snapshot_seq=? AND host_id=? AND metric_id=?",
                            (previous_generation, item.snapshot_seq, item.host_id, item.metric_id),
                        )
                empty_merged_digest = digest_json([])
                empty_conclusion_digest = digest_json({"merged": [], "drift": []})
                cursor = conn.execute(
                    """
                    INSERT INTO generations(
                        active, created_ms, declaration_digest, snapshot_chain_version,
                        agent_set_version, scale_version, snapshot_json, agents_json,
                        criteria_json, merged_view_digest, conclusion_digest
                    ) VALUES(1,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        now_ms(),
                        current["declaration_digest"],
                        current["snapshot_chain_version"],
                        current["agent_set_version"],
                        current["scale_version"],
                        canonical_json(chain.to_dict()),
                        canonical_json([agent.to_dict() for agent in agents]),
                        canonical_json(criteria.to_dict()),
                        empty_merged_digest,
                        empty_conclusion_digest,
                    ),
                )
                generation = int(cursor.lastrowid)
                for ordinal, item in enumerate(propagation.affected):
                    conn.execute(
                        "INSERT INTO affected_items(generation, snapshot_seq, host_id, metric_id, reasons_json, stale_applied) VALUES(?,?,?,?,?,?)",
                        (
                            generation,
                            item.snapshot_seq,
                            item.host_id,
                            item.metric_id,
                            canonical_json(list(item.reasons)),
                            1 if previous_generation is not None else 0,
                        ),
                    )
                for ordinal, difference in enumerate(propagation.differences):
                    conn.execute(
                        """
                        INSERT INTO generation_diffs(
                            generation, ordinal, kind, snapshot_seq, host_id, metric_id,
                            agent_id, old_value, new_value, basis
                        ) VALUES(?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            generation,
                            ordinal,
                            str(difference.get("kind")),
                            difference.get("snapshot_seq"),
                            difference.get("host_id"),
                            difference.get("metric_id"),
                            difference.get("agent_id"),
                            canonical_json(difference.get("old")) if "old" in difference else None,
                            canonical_json(difference.get("new")) if "new" in difference else None,
                            str(difference.get("basis", "")),
                        ),
                    )
                conn.execute(
                    """
                    INSERT INTO session_snapshots(
                        generation, snapshot_chain_version, agent_set_version, scale_version,
                        merged_view_json, merged_view_digest, drift_view_json, drift_view_digest,
                        conclusion_digest, updated_ms
                    ) VALUES(?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        generation,
                        current["snapshot_chain_version"],
                        current["agent_set_version"],
                        current["scale_version"],
                        canonical_json([]),
                        empty_merged_digest,
                        canonical_json([]),
                        digest_json([]),
                        empty_conclusion_digest,
                        now_ms(),
                    ),
                )
                conn.execute(
                    "INSERT INTO cursors(generation, last_committed_snapshot_seq, last_batch_id, updated_ms) VALUES(?,?,?,?)",
                    (generation, None, None, now_ms()),
                )
            return ReconcileResult(
                generation=generation,
                replaced=previous_generation is not None,
                previous_generation=previous_generation,
                affected=propagation.affected,
                differences=propagation.differences,
                **current,
            )

    def _session_snapshot_mismatches(
        self, generation_row: sqlite3.Row, current: dict[str, str]
    ) -> list[dict[str, object]]:
        generation = int(generation_row["generation"])
        row = self.connection.execute(
            "SELECT * FROM session_snapshots WHERE generation=?", (generation,)
        ).fetchone()
        if row is None:
            return [{"field": "session_snapshot.presence", "expected": "present", "actual": "missing"}]
        mismatches: list[dict[str, object]] = []
        for field in ("snapshot_chain_version", "agent_set_version", "scale_version"):
            if row[field] != current[field]:
                mismatches.append({"field": f"session_snapshot.{field}", "expected": current[field], "actual": row[field]})
        merged_view = json.loads(row["merged_view_json"])
        drift_view = json.loads(row["drift_view_json"])
        expected_merged, expected_drift, expected_conclusion = _semantic_view_digests(merged_view, drift_view)
        checks = (
            ("session_snapshot.merged_view_digest", expected_merged, row["merged_view_digest"]),
            ("session_snapshot.drift_view_digest", expected_drift, row["drift_view_digest"]),
            ("session_snapshot.conclusion_digest", expected_conclusion, row["conclusion_digest"]),
            ("generation.conclusion_digest", expected_conclusion, generation_row["conclusion_digest"]),
            ("generation.merged_view_digest", expected_merged, generation_row["merged_view_digest"]),
        )
        for field, expected, actual in checks:
            if expected != actual:
                mismatches.append({"field": field, "expected": expected, "actual": actual})
        return sorted(mismatches, key=lambda item: str(item["field"]))

    def _parse_generation_declarations(
        self, row: sqlite3.Row
    ) -> tuple[SnapshotChain, tuple[AgentSpec, ...], Criteria]:
        chain = parse_snapshot_chain(json.loads(row["snapshot_json"]))
        criteria = parse_criteria(json.loads(row["criteria_json"]))
        agents = parse_agent_manifest({"agents": json.loads(row["agents_json"])}, chain, criteria)
        return chain, agents, criteria

    def generation_declarations(self, generation: int) -> tuple[SnapshotChain, tuple[AgentSpec, ...], Criteria]:
        with self._lock:
            row = self.connection.execute("SELECT * FROM generations WHERE generation=?", (generation,)).fetchone()
            if row is None:
                raise DriftError("E_REPLAY", "generation does not exist", details={"generation": generation})
            return self._parse_generation_declarations(row)

    def begin_batch(
        self,
        generation: int,
        snapshot_seq: int,
        planned_items: Iterable[dict[str, Any]],
        *,
        batch_id: str | None = None,
    ) -> str:
        batch_id = batch_id or f"g{generation}-s{snapshot_seq}-{now_ms()}-{uuid.uuid4().hex[:10]}"
        planned = sorted(
            planned_items,
            key=lambda item: (int(item["snapshot_seq"]), str(item["host_id"]), str(item["metric_id"])),
        )
        with self._transaction() as conn:
            generation_row = conn.execute(
                "SELECT active FROM generations WHERE generation=?", (generation,)
            ).fetchone()
            if generation_row is None or int(generation_row["active"]) != 1:
                raise DriftError("E_REPLAY", "cannot start a batch in an inactive generation", details={"generation": generation})
            conn.execute(
                "INSERT INTO batches(batch_id, generation, snapshot_seq, status, started_ms, planned_json) VALUES(?,?,?,?,?,?)",
                (batch_id, generation, snapshot_seq, "RUNNING", now_ms(), canonical_json(planned)),
            )
        return batch_id

    def commit_batch(
        self,
        batch_id: str,
        *,
        generation: int,
        snapshot_seq: int,
        observations: list[Observation],
        merged_results: list[MergedResult],
        drift_results: list[DriftResult],
        attempts: list[AttemptRecord],
        cursor: dict[str, Any],
    ) -> dict[str, Any]:
        observations = sorted(
            observations,
            key=lambda item: (item.host_id, item.metric_id, item.agent_id, item.observation_seq),
        )
        merged_results = sorted(merged_results, key=lambda item: (item.host_id, item.metric_id))
        drift_results = sorted(drift_results, key=lambda item: (item.host_id, item.metric_id))
        attempts = sorted(attempts, key=lambda item: (item.task_key, item.attempt_no))
        observations_payload = [item.to_dict() for item in observations]
        merged_payload = [item.to_dict() for item in merged_results]
        drift_payload = [item.to_dict() for item in drift_results]
        attempts_payload = [item.to_dict() for item in attempts]
        observations_digest = digest_json(observations_payload)
        conclusions_digest = digest_json({"merged": merged_payload, "drift": drift_payload})
        content_digest = digest_json(
            {
                "observations": observations_payload,
                "merged": merged_payload,
                "drift": drift_payload,
                "attempts": attempts_payload,
                "cursor": cursor,
            }
        )
        merged_by_key = {(item.host_id, item.metric_id): item for item in merged_results}
        with self._transaction() as conn:
            batch = conn.execute("SELECT * FROM batches WHERE batch_id=?", (batch_id,)).fetchone()
            if batch is None:
                raise DriftError("E_REPLAY", "batch does not exist", details={"batch_id": batch_id})
            if batch["status"] != "RUNNING":
                raise DriftError(
                    "E_REPLAY",
                    "batch is not open for commit",
                    details={"batch_id": batch_id, "status": batch["status"]},
                )
            for observation in observations:
                merged = merged_by_key.get((observation.host_id, observation.metric_id))
                deviation = None
                if merged and not merged.sparse:
                    deviation = observation.value - int(merged.authority_value)
                conn.execute(
                    """
                    INSERT INTO observations(
                        batch_id, agent_id, host_id, metric_id, observation_seq, value,
                        aggregation_version, monotonic_ms, is_authority, deviation
                    ) VALUES(?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        batch_id,
                        observation.agent_id,
                        observation.host_id,
                        observation.metric_id,
                        observation.observation_seq,
                        observation.value,
                        observation.aggregation_version,
                        observation.monotonic_ms,
                        1
                        if merged
                        and merged.authority_agent == observation.agent_id
                        and merged.authority_observation_seq == observation.observation_seq
                        else 0,
                        deviation,
                    ),
                )
            for attempt in attempts:
                conn.execute(
                    """
                    INSERT INTO attempts(
                        batch_id, task_key, attempt_no, is_retry, agent_id, snapshot_seq,
                        host_id, metric_id, observation_seq, wait_start_ms, wait_end_ms,
                        run_start_ms, run_end_ms, status, error_code, message, value
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        attempt.batch_id,
                        attempt.task_key,
                        attempt.attempt_no,
                        int(attempt.is_retry),
                        attempt.agent_id,
                        attempt.snapshot_seq,
                        attempt.host_id,
                        attempt.metric_id,
                        attempt.observation_seq,
                        attempt.wait_start_ms,
                        attempt.wait_end_ms,
                        attempt.run_start_ms,
                        attempt.run_end_ms,
                        attempt.status,
                        attempt.error_code,
                        attempt.message,
                        attempt.value,
                    ),
                )
            for merged in merged_results:
                conn.execute(
                    """
                    INSERT INTO merged_rows(
                        batch_id, generation, snapshot_seq, host_id, metric_id, status,
                        authority_agent, authority_observation_seq, authority_value,
                        aggregation_version, source_count, sparse, sources_json,
                        max_abs_deviation, error_code, stale
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)
                    """,
                    (
                        batch_id,
                        generation,
                        merged.snapshot_seq,
                        merged.host_id,
                        merged.metric_id,
                        merged.status,
                        merged.authority_agent,
                        merged.authority_observation_seq,
                        merged.authority_value,
                        merged.aggregation_version,
                        merged.source_count,
                        int(merged.sparse),
                        canonical_json(merged.sources),
                        merged.max_abs_deviation,
                        merged.error_code,
                    ),
                )
            for drift in drift_results:
                conn.execute(
                    """
                    INSERT INTO drift_rows(
                        batch_id, generation, snapshot_seq, host_id, metric_id, status,
                        decision, expected, observed, tolerance_bps, drift_signed_fixed,
                        drift_excess_fixed, drift_ratio_bps, allowed_fixed, error_bound_json,
                        error_affected, boundary_rule, saturated, error_code, reason, result_json, stale
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)
                    """,
                    (
                        batch_id,
                        generation,
                        drift.snapshot_seq,
                        drift.host_id,
                        drift.metric_id,
                        drift.status,
                        drift.decision,
                        drift.expected,
                        drift.observed,
                        drift.tolerance_bps,
                        drift.drift_signed_fixed,
                        drift.drift_excess_fixed,
                        drift.drift_ratio_bps,
                        drift.allowed_fixed,
                        canonical_json(drift.error_bound.to_dict()) if drift.error_bound else None,
                        int(drift.error_affected),
                        drift.boundary_rule,
                        int(drift.saturated),
                        drift.error_code,
                        drift.reason,
                        canonical_json(drift.to_dict()),
                    ),
                )
            conn.execute(
                """
                UPDATE batches SET status='COMMITTED', committed_ms=?, cursor_json=?,
                    observations_digest=?, conclusions_digest=?, content_digest=?
                WHERE batch_id=?
                """,
                (
                    now_ms(),
                    canonical_json(cursor),
                    observations_digest,
                    conclusions_digest,
                    content_digest,
                    batch_id,
                ),
            )
            conn.execute(
                """
                INSERT INTO cursors(generation, last_committed_snapshot_seq, last_batch_id, updated_ms)
                VALUES(?,?,?,?)
                ON CONFLICT(generation) DO UPDATE SET
                    last_committed_snapshot_seq=excluded.last_committed_snapshot_seq,
                    last_batch_id=excluded.last_batch_id,
                    updated_ms=excluded.updated_ms
                """,
                (generation, snapshot_seq, batch_id, now_ms()),
            )
            for merged in merged_results:
                conn.execute(
                    """
                    UPDATE recovery_unknown SET required_recollect=0, recollected_batch_id=?
                    WHERE generation=? AND snapshot_seq=? AND host_id=? AND metric_id=?
                    """,
                    (batch_id, generation, merged.snapshot_seq, merged.host_id, merged.metric_id),
                )
            self._refresh_session_snapshot(conn, generation)
            self._update_ledger_digest(conn)
        return {
            "batch_id": batch_id,
            "observations_digest": observations_digest,
            "conclusions_digest": conclusions_digest,
            "content_digest": content_digest,
            "cursor": cursor,
        }

    def abort_batch(self, batch_id: str, reason: str) -> list[dict[str, Any]]:
        with self._transaction() as conn:
            batch = conn.execute("SELECT * FROM batches WHERE batch_id=?", (batch_id,)).fetchone()
            if batch is None:
                raise DriftError("E_REPLAY", "batch does not exist", details={"batch_id": batch_id})
            if batch["status"] == "COMMITTED":
                raise DriftError(
                    "E_REPLAY",
                    "cannot abort a committed batch",
                    details={"batch_id": batch_id},
                )
            if batch["status"] == "RUNNING":
                conn.execute(
                    "UPDATE batches SET status='ABORTED', abort_reason=?, committed_ms=? WHERE batch_id=?",
                    (reason, now_ms(), batch_id),
                )
            return self._register_unknown_for_batch(conn, batch)

    def _register_unknown_for_batch(
        self, conn: sqlite3.Connection, batch: sqlite3.Row
    ) -> list[dict[str, Any]]:
        planned = json.loads(batch["planned_json"])
        unknown: list[dict[str, Any]] = []
        for item in planned:
            conn.execute(
                """
                INSERT INTO recovery_unknown(
                    generation, batch_id, snapshot_seq, host_id, metric_id,
                    marked_ms, reason, required_recollect
                ) VALUES(?,?,?,?,?,?,?,1)
                ON CONFLICT(generation, snapshot_seq, host_id, metric_id) DO UPDATE SET
                    batch_id=excluded.batch_id,
                    marked_ms=excluded.marked_ms,
                    reason=excluded.reason,
                    required_recollect=1,
                    recollected_batch_id=NULL
                """,
                (
                    int(batch["generation"]),
                    batch["batch_id"],
                    int(item["snapshot_seq"]),
                    str(item["host_id"]),
                    str(item["metric_id"]),
                    now_ms(),
                    "batch was not committed",
                ),
            )
            unknown.append(
                {
                    "generation": int(batch["generation"]),
                    "batch_id": batch["batch_id"],
                    "snapshot_seq": int(item["snapshot_seq"]),
                    "host_id": str(item["host_id"]),
                    "metric_id": str(item["metric_id"]),
                    "required_recollect": True,
                }
            )
        return sorted(unknown, key=lambda item: (item["generation"], item["snapshot_seq"], item["host_id"], item["metric_id"]))

    def recover_incomplete(self) -> list[dict[str, Any]]:
        recovered: list[dict[str, Any]] = []
        with self._transaction() as conn:
            rows = conn.execute("SELECT * FROM batches WHERE status='RUNNING' ORDER BY generation, snapshot_seq, started_ms").fetchall()
            for batch in rows:
                conn.execute(
                    "UPDATE batches SET status='ABORTED', abort_reason=?, committed_ms=? WHERE batch_id=?",
                    ("process interrupted before atomic commit", now_ms(), batch["batch_id"]),
                )
                recovered.extend(self._register_unknown_for_batch(conn, batch))
        return sorted(recovered, key=lambda item: (item["generation"], item["snapshot_seq"], item["host_id"], item["metric_id"]))

    def required_recollection(self, generation: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT * FROM recovery_unknown
                WHERE generation=? AND required_recollect=1
                ORDER BY snapshot_seq, host_id, metric_id
                """,
                (generation,),
            ).fetchall()
            return [dict(row) for row in rows]

    def committed_snapshot_seqs(self, generation: int) -> tuple[int, ...]:
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT DISTINCT snapshot_seq FROM batches
                WHERE generation=? AND status='COMMITTED'
                ORDER BY snapshot_seq
                """,
                (generation,),
            ).fetchall()
            return tuple(int(row["snapshot_seq"]) for row in rows)

    def cursor(self, generation: int) -> dict[str, Any]:
        with self._lock:
            row = self.connection.execute("SELECT * FROM cursors WHERE generation=?", (generation,)).fetchone()
            if row is None:
                return {}
            return {
                "last_committed_snapshot_seq": row["last_committed_snapshot_seq"],
                "last_batch_id": row["last_batch_id"],
                "updated_ms": int(row["updated_ms"]),
            }

    def _refresh_session_snapshot(self, conn: sqlite3.Connection, generation: int) -> None:
        merged_rows = conn.execute(
            "SELECT * FROM merged_rows WHERE generation=? ORDER BY snapshot_seq, host_id, metric_id",
            (generation,),
        ).fetchall()
        drift_rows = conn.execute(
            "SELECT * FROM drift_rows WHERE generation=? ORDER BY snapshot_seq, host_id, metric_id",
            (generation,),
        ).fetchall()
        merged_view: list[dict[str, Any]] = []
        for row in merged_rows:
            item = dict(row)
            item["sources"] = json.loads(item.pop("sources_json"))
            item["sparse"] = bool(item["sparse"])
            item["stale"] = bool(item["stale"])
            merged_view.append(item)
        drift_view: list[dict[str, Any]] = []
        for row in drift_rows:
            item = dict(row)
            item["error_bound"] = json.loads(item.pop("error_bound_json")) if item["error_bound_json"] else None
            item.pop("error_bound_json", None)
            item["error_affected"] = bool(item["error_affected"])
            item["saturated"] = bool(item["saturated"])
            item["stale"] = bool(item["stale"])
            item.pop("result_json", None)
            drift_view.append(item)
        merged_json = canonical_json(merged_view)
        drift_json = canonical_json(drift_view)
        merged_digest, drift_digest, conclusion_digest = _semantic_view_digests(merged_view, drift_view)
        row = conn.execute("SELECT * FROM generations WHERE generation=?", (generation,)).fetchone()
        if row is None:
            raise DriftError("E_REPLAY", "generation vanished during commit")
        conn.execute(
            """
            UPDATE session_snapshots SET
                merged_view_json=?, merged_view_digest=?, drift_view_json=?,
                drift_view_digest=?, conclusion_digest=?, updated_ms=?
            WHERE generation=?
            """,
            (merged_json, merged_digest, drift_json, drift_digest, conclusion_digest, now_ms(), generation),
        )
        conn.execute(
            "UPDATE generations SET merged_view_digest=?, conclusion_digest=? WHERE generation=?",
            (merged_digest, conclusion_digest, generation),
        )

    def session_snapshot(self, generation: int) -> dict[str, Any]:
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM session_snapshots WHERE generation=?", (generation,)
            ).fetchone()
            if row is None:
                raise DriftError("E_REPLAY", "session snapshot does not exist", details={"generation": generation})
            return {
                "generation": generation,
                "snapshot_chain_version": row["snapshot_chain_version"],
                "agent_set_version": row["agent_set_version"],
                "scale_version": row["scale_version"],
                "merged_view": json.loads(row["merged_view_json"]),
                "merged_view_digest": row["merged_view_digest"],
                "drift_view": json.loads(row["drift_view_json"]),
                "drift_view_digest": row["drift_view_digest"],
                "conclusion_digest": row["conclusion_digest"],
                "updated_ms": int(row["updated_ms"]),
            }

    def active_view(self) -> dict[str, Any]:
        generation = self.active_generation()
        if generation is None:
            return {"generation": None, "merged_view": [], "drift_view": []}
        snapshot = self.session_snapshot(generation)
        return {
            "generation": generation,
            "merged_view": snapshot["merged_view"],
            "drift_view": snapshot["drift_view"],
            "conclusion_digest": snapshot["conclusion_digest"],
        }

    def affected(self, generation: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT * FROM affected_items WHERE generation=?
                ORDER BY snapshot_seq, host_id, metric_id
                """,
                (generation,),
            ).fetchall()
            return [
                {
                    "snapshot_seq": int(row["snapshot_seq"]),
                    "host_id": row["host_id"],
                    "metric_id": row["metric_id"],
                    "reasons": json.loads(row["reasons_json"]),
                    "stale_applied": bool(row["stale_applied"]),
                }
                for row in rows
            ]

    def differences(self, generation: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT * FROM generation_diffs WHERE generation=? ORDER BY ordinal",
                (generation,),
            ).fetchall()
            return [
                {
                    "kind": row["kind"],
                    "snapshot_seq": row["snapshot_seq"],
                    "host_id": row["host_id"],
                    "metric_id": row["metric_id"],
                    "agent_id": row["agent_id"],
                    "old": json.loads(row["old_value"]) if row["old_value"] is not None else None,
                    "new": json.loads(row["new_value"]) if row["new_value"] is not None else None,
                    "basis": row["basis"],
                }
                for row in rows
            ]

    def committed_batches(self, generation: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.connection.execute(
                """
                SELECT * FROM batches WHERE generation=? AND status='COMMITTED'
                ORDER BY committed_ms, snapshot_seq, batch_id
                """,
                (generation,),
            ).fetchall()
            return [dict(row) for row in rows]

    def batch_results(self, batch_id: str) -> dict[str, list[dict[str, Any]]]:
        with self._lock:
            merged_rows = self.connection.execute(
                "SELECT * FROM merged_rows WHERE batch_id=? ORDER BY host_id, metric_id",
                (batch_id,),
            ).fetchall()
            drift_rows = self.connection.execute(
                "SELECT * FROM drift_rows WHERE batch_id=? ORDER BY host_id, metric_id",
                (batch_id,),
            ).fetchall()
            merged = []
            for row in merged_rows:
                item = dict(row)
                item["sources"] = json.loads(item.pop("sources_json"))
                merged.append(item)
            drift = []
            for row in drift_rows:
                item = dict(row)
                item["error_bound"] = json.loads(item.pop("error_bound_json")) if item["error_bound_json"] else None
                item["result"] = json.loads(item.pop("result_json"))
                drift.append(item)
            return {"merged": merged, "drift": drift}

    def batch_planned(self, batch_id: str) -> list[dict[str, Any]]:
        with self._lock:
            row = self.connection.execute("SELECT planned_json FROM batches WHERE batch_id=?", (batch_id,)).fetchone()
            if row is None:
                return []
            return list(json.loads(row["planned_json"]))

    def status(self) -> dict[str, Any]:
        with self._lock:
            generations = self.connection.execute(
                "SELECT * FROM generations ORDER BY generation"
            ).fetchall()
            batch_counts = {
                row["status"]: int(row["count"])
                for row in self.connection.execute(
                    "SELECT status, COUNT(*) AS count FROM batches GROUP BY status"
                ).fetchall()
            }
            active_generation = self.active_generation()
            if active_generation is None:
                unknown_count = 0
            else:
                unknown_count = int(
                    self.connection.execute(
                        "SELECT COUNT(*) AS count FROM recovery_unknown WHERE required_recollect=1 AND generation=?",
                        (active_generation,),
                    ).fetchone()["count"]
                )
            return {
                "ledger": self.ledger_info(),
                "active_generation": self.active_generation(),
                "generations": [
                    {
                        "generation": int(row["generation"]),
                        "active": bool(row["active"]),
                        "snapshot_chain_version": row["snapshot_chain_version"],
                        "agent_set_version": row["agent_set_version"],
                        "scale_version": row["scale_version"],
                        "declaration_digest": row["declaration_digest"],
                        "conclusion_digest": row["conclusion_digest"],
                    }
                    for row in generations
                ],
                "batch_counts": batch_counts,
                "required_recollections": unknown_count,
            }

    def view_rows_for_replay(self, generation: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self.connection.execute(
                "SELECT * FROM drift_rows WHERE generation=? ORDER BY snapshot_seq, host_id, metric_id",
                (generation,),
            ).fetchall()
            return [dict(row) for row in rows]
