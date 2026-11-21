from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from .canonical import canonical_text, digest_bytes, digest_text
from .errors import ArbiterError
from .pool import BatchOutcome
from .spec import Dataset, Node, relevant_rules


SCHEMA_VERSION = 1
STAT_SCALE_FACTOR = 1000
STAT_ROUNDING = "floor_toward_negative_infinity"
STAT_QUANTIZATION_ERROR_US = 1
STAT_QUANTIZATION_ERROR_NS = 999


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    format_version INTEGER NOT NULL,
    ruleset_version INTEGER NOT NULL,
    rule_versions_json TEXT NOT NULL,
    nodes_digest TEXT NOT NULL,
    spans_digest TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    nodes_json TEXT NOT NULL,
    spans_json TEXT NOT NULL,
    created_mono_ns INTEGER NOT NULL,
    is_current INTEGER NOT NULL DEFAULT 0,
    is_stale INTEGER NOT NULL DEFAULT 1
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_snapshot_one_current
    ON snapshots(is_current) WHERE is_current = 1;
CREATE TABLE IF NOT EXISTS ledger_entries (
    entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id TEXT NOT NULL,
    format_version INTEGER NOT NULL,
    entry_json TEXT NOT NULL,
    entry_digest TEXT NOT NULL,
    UNIQUE(trace_id, entry_digest)
);
CREATE TABLE IF NOT EXISTS snapshot_traces (
    snapshot_id INTEGER NOT NULL,
    trace_id TEXT NOT NULL,
    entry_id INTEGER NOT NULL,
    rule_versions_json TEXT NOT NULL,
    PRIMARY KEY(snapshot_id, trace_id),
    FOREIGN KEY(snapshot_id) REFERENCES snapshots(snapshot_id),
    FOREIGN KEY(entry_id) REFERENCES ledger_entries(entry_id)
);
CREATE TABLE IF NOT EXISTS flips (
    flip_id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER NOT NULL,
    trace_id TEXT NOT NULL,
    change_kind TEXT NOT NULL,
    before_rules_json TEXT,
    after_rules_json TEXT,
    diff_fields_json TEXT NOT NULL,
    before_entry_digest TEXT,
    after_entry_digest TEXT,
    is_stale INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY(snapshot_id) REFERENCES snapshots(snapshot_id)
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    snapshot_id INTEGER NOT NULL,
    status TEXT NOT NULL,
    global_concurrency INTEGER NOT NULL,
    per_node_concurrency INTEGER NOT NULL,
    batch_size INTEGER NOT NULL,
    request_timeout_ms INTEGER NOT NULL,
    retries INTEGER NOT NULL,
    created_mono_ns INTEGER NOT NULL,
    updated_mono_ns INTEGER NOT NULL,
    progress_cursor INTEGER NOT NULL DEFAULT 0,
    batch_count INTEGER NOT NULL,
    concurrency_json TEXT,
    error_json TEXT,
    FOREIGN KEY(snapshot_id) REFERENCES snapshots(snapshot_id)
);
CREATE TABLE IF NOT EXISTS run_batches (
    run_id TEXT NOT NULL,
    batch_index INTEGER NOT NULL,
    node_id TEXT NOT NULL,
    mode TEXT NOT NULL,
    request_json TEXT NOT NULL,
    request_digest TEXT NOT NULL,
    status TEXT NOT NULL,
    response_summary TEXT,
    response_json TEXT,
    attribution_json TEXT,
    processing_us INTEGER,
    attempts INTEGER,
    queue_enter_ns INTEGER,
    queue_exit_ns INTEGER,
    queue_wait_us INTEGER,
    queue_enter_mono_ms INTEGER,
    queue_exit_mono_ms INTEGER,
    queue_wait_ms INTEGER,
    committed_mono_ns INTEGER,
    error_code TEXT,
    was_unknown INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(run_id, batch_index),
    FOREIGN KEY(run_id) REFERENCES runs(run_id)
);
CREATE TABLE IF NOT EXISTS latency_samples (
    sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER NOT NULL,
    run_id TEXT,
    operation TEXT NOT NULL,
    node_id TEXT NOT NULL,
    elapsed_us INTEGER NOT NULL,
    started_mono_ns INTEGER NOT NULL,
    ended_mono_ns INTEGER NOT NULL,
    is_stale INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY(snapshot_id) REFERENCES snapshots(snapshot_id)
);
CREATE TABLE IF NOT EXISTS latency_stats (
    snapshot_id INTEGER NOT NULL,
    operation TEXT NOT NULL,
    node_id TEXT NOT NULL,
    sample_count INTEGER NOT NULL,
    mean_us INTEGER NOT NULL,
    p95_us INTEGER NOT NULL,
    max_us INTEGER NOT NULL,
    scale_factor INTEGER NOT NULL,
    rounding TEXT NOT NULL,
    quantization_error_us INTEGER NOT NULL,
    quantization_error_ns INTEGER NOT NULL,
    is_stale INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(snapshot_id, operation, node_id),
    FOREIGN KEY(snapshot_id) REFERENCES snapshots(snapshot_id)
);
"""


def _row_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def _stats_for_values(values: Sequence[int]) -> tuple[int, int, int, int]:
    ordered = sorted(values)
    count = len(ordered)
    mean_us = sum(ordered) // count
    p95_index = max(0, (95 * count + 99) // 100 - 1)
    return count, mean_us, ordered[p95_index], ordered[-1]


class LedgerDB:
    def __init__(self, path: str) -> None:
        self.path = path
        if path != ":memory:":
            Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, timeout=30.0, isolation_level=None, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = FULL")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        with self._transaction():
            self._conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def _read_one(self, sql: str, params: Sequence[Any] = ()) -> dict[str, Any] | None:
        with self._lock:
            return _row_dict(self._conn.execute(sql, params).fetchone())

    def _read_all(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self._conn.execute(sql, params).fetchall()]

    def _current_snapshot_row(self) -> dict[str, Any] | None:
        row = self._read_one("SELECT * FROM snapshots WHERE is_current = 1")
        return row

    def check_ruleset_evolution(self, dataset: Dataset) -> None:
        current = self._current_snapshot_row()
        if current is None:
            return
        old_nodes = json.loads(current["nodes_json"])
        try:
            old_rule_map = relevant_rules(
                Node(
                    node_id=node["node_id"],
                    host=node["host"],
                    port=node["port"],
                    propagation_rule_version=node["propagation_rule_version"],
                    sample_ratio_bps=node["sample_ratio_bps"],
                    random_seed=node["random_seed"],
                )
                for node in old_nodes
            )
        except (KeyError, TypeError):
            raise ArbiterError("E_SNAPSHOT", "current snapshot contains invalid node rules") from None
        new_rule_map = relevant_rules(dataset.nodes)
        rules_changed = old_rule_map != new_rule_map
        new_nodes_by_id = {node.node_id: node for node in dataset.nodes}
        dataset_node_index = {
            node.node_id: index
            for index, node in enumerate(dataset.nodes)
        }
        for node_id in sorted(set(old_rule_map) & set(new_rule_map)):
            old_version, old_ratio, old_seed = old_rule_map[node_id]
            new_node = new_nodes_by_id[node_id]
            new_version = new_node.propagation_rule_version
            new_ratio = new_node.sample_ratio_bps
            new_seed = new_node.random_seed
            if new_version < old_version or (
                (new_ratio, new_seed) != (old_ratio, old_seed)
                and new_version <= old_version
            ):
                raise ArbiterError(
                    "E_SPEC",
                    "node propagation_rule_version must increase across snapshots when ratio or seed changes",
                    path=f"/nodes/{dataset_node_index[node_id]}/propagation_rule_version",
                    details={
                        "node_id": node_id,
                        "previous_version": old_version,
                        "new_version": new_version,
                        "previous_ratio_bps": old_ratio,
                        "new_ratio_bps": new_ratio,
                    },
                )
        if dataset.ruleset_version < current["ruleset_version"]:
            raise ArbiterError(
                "E_SPEC",
                "ruleset_version must not decrease across snapshots",
                path="/ruleset_version",
                details={
                    "previous_ruleset_version": current["ruleset_version"],
                    "new_ruleset_version": dataset.ruleset_version,
                },
            )
        if rules_changed and dataset.ruleset_version == current["ruleset_version"]:
            raise ArbiterError(
                "E_SPEC",
                "ruleset_version must increase when rule version, ratio, seed, or node rule set changes",
                path="/ruleset_version",
                details={
                    "previous_ruleset_version": current["ruleset_version"],
                    "new_ruleset_version": dataset.ruleset_version,
                },
            )

    def create_snapshot_and_run(
        self,
        dataset: Dataset,
        *,
        run_id: str,
        batches: list[dict[str, Any]],
        concurrency: dict[str, Any],
    ) -> tuple[int, str]:
        self.check_ruleset_evolution(dataset)
        now_ns = time.perf_counter_ns()
        nodes_json = canonical_text(dataset.nodes_payload()["nodes"])
        spans_json = canonical_text(dataset.spans_payload())
        versions_json = canonical_text(list(dataset.versions))
        with self._transaction():
            cursor = self._conn.execute(
                """
                INSERT INTO snapshots(
                    format_version, ruleset_version, rule_versions_json,
                    nodes_digest, spans_digest, content_digest, nodes_json,
                    spans_json, created_mono_ns, is_current, is_stale
                ) VALUES(?,?,?,?,?,?,?,?,?,0,1)
                """,
                (
                    dataset.format_version,
                    dataset.ruleset_version,
                    versions_json,
                    dataset.nodes_digest,
                    dataset.spans_digest,
                    dataset.content_digest,
                    nodes_json,
                    spans_json,
                    now_ns,
                ),
            )
            snapshot_id = int(cursor.lastrowid)
            self._conn.execute(
                """
                INSERT INTO runs(
                    run_id, snapshot_id, status, global_concurrency,
                    per_node_concurrency, batch_size, request_timeout_ms,
                    retries, created_mono_ns, updated_mono_ns, progress_cursor,
                    batch_count, concurrency_json, error_json
                ) VALUES(?,?,?,?,?,?,?,?,?,?,0,?,?,NULL)
                """,
                (
                    run_id,
                    snapshot_id,
                    "running",
                    concurrency["global_concurrency"],
                    concurrency["per_node_concurrency"],
                    concurrency["batch_size"],
                    concurrency["request_timeout_ms"],
                    concurrency["retries"],
                    now_ns,
                    now_ns,
                    len(batches),
                    canonical_text(concurrency),
                ),
            )
            for batch in batches:
                request_json = batch["request_text"]
                self._conn.execute(
                    """
                    INSERT INTO run_batches(
                        run_id, batch_index, node_id, mode, request_json,
                        request_digest, status
                    ) VALUES(?,?,?,?,?,?, 'pending')
                    """,
                    (
                        run_id,
                        batch["batch_index"],
                        batch["node_id"],
                        batch["mode"],
                        request_json,
                        digest_text(request_json),
                    ),
                )
        return snapshot_id, run_id

    def mark_run_running(self, run_id: str) -> None:
        with self._transaction():
            self._conn.execute(
                "UPDATE runs SET status='running', updated_mono_ns=? WHERE run_id=?",
                (time.perf_counter_ns(), run_id),
            )
            self._conn.execute(
                "UPDATE run_batches SET status='running' WHERE run_id=? AND status='pending'",
                (run_id,),
            )

    def fail_run(self, run_id: str, error: ArbiterError) -> None:
        with self._transaction():
            self._conn.execute(
                "UPDATE runs SET status='failed', error_json=?, updated_mono_ns=? WHERE run_id=?",
                (canonical_text(error.as_dict()), time.perf_counter_ns(), run_id),
            )

    def commit_batch(
        self,
        run_id: str,
        outcome: BatchOutcome,
        *,
        operation: str,
    ) -> None:
        batch = outcome.batch
        now_ns = time.perf_counter_ns()
        with self._transaction():
            request_row = self._conn.execute(
                "SELECT request_digest FROM run_batches WHERE run_id=? AND batch_index=?",
                (run_id, batch.batch_index),
            ).fetchone()
            if request_row is None:
                raise ArbiterError("E_RECOVERY", "batch manifest entry is missing")
            if request_row["request_digest"] != batch.request_digest:
                raise ArbiterError(
                    "E_PROTOCOL",
                    "request bytes changed between retries or recovery",
                    details={"batch_index": batch.batch_index},
                )
            if outcome.error is not None:
                error = outcome.error
                self._conn.execute(
                    """
                    UPDATE run_batches SET
                        status='failed', error_code=?, attempts=?,
                        queue_enter_ns=?, queue_exit_ns=?, queue_wait_us=?,
                        queue_enter_mono_ms=?, queue_exit_mono_ms=?, queue_wait_ms=?,
                        committed_mono_ns=?
                    WHERE run_id=? AND batch_index=?
                    """,
                    (
                        error.code,
                        outcome.attempts,
                        outcome.queue_enter_ns,
                        outcome.queue_exit_ns,
                        outcome.queue_wait_us,
                        outcome.queue_enter_ns // 1_000_000,
                        outcome.queue_exit_ns // 1_000_000,
                        outcome.queue_wait_ms,
                        now_ns,
                        run_id,
                        batch.batch_index,
                    ),
                )
                self._conn.execute(
                    "UPDATE runs SET status='failed', error_json=?, updated_mono_ns=? WHERE run_id=?",
                    (canonical_text(error.as_dict()), now_ns, run_id),
                )
                return

            response = outcome.response
            summary = response.get("decision_summary")
            if not isinstance(summary, str):
                raise ArbiterError("E_PROTOCOL", "worker response is missing decision_summary")
            if response.get("batch_id") != batch.request_digest:
                raise ArbiterError(
                    "E_PROTOCOL",
                    "worker batch_id does not match canonical request digest",
                    details={"batch_index": batch.batch_index},
                )
            attribution = response.get("trace_context")
            if attribution is None and response.get("trace_results") is not None:
                attribution = [
                    {
                        "trace_id": item["trace_id"],
                        "root_span_id": item["root_span_id"],
                        "participating_nodes": item["participating_nodes"],
                    }
                    for item in response["trace_results"]
                ]
            self._conn.execute(
                """
                UPDATE run_batches SET
                    status='committed', response_summary=?, response_json=?,
                    attribution_json=?, processing_us=?, attempts=?,
                    queue_enter_ns=?, queue_exit_ns=?, queue_wait_us=?,
                    queue_enter_mono_ms=?, queue_exit_mono_ms=?, queue_wait_ms=?,
                    committed_mono_ns=?, error_code=NULL
                WHERE run_id=? AND batch_index=?
                """,
                (
                    summary,
                    canonical_text(response),
                    canonical_text(attribution) if attribution is not None else None,
                    int(response.get("processing_us", 0)),
                    outcome.attempts,
                    outcome.queue_enter_ns,
                    outcome.queue_exit_ns,
                    outcome.queue_wait_us,
                    outcome.queue_enter_ns // 1_000_000,
                    outcome.queue_exit_ns // 1_000_000,
                    outcome.queue_wait_ms,
                    now_ns,
                    run_id,
                    batch.batch_index,
                ),
            )
            self._conn.execute(
                """
                INSERT INTO latency_samples(
                    snapshot_id, run_id, operation, node_id, elapsed_us,
                    started_mono_ns, ended_mono_ns, is_stale
                ) SELECT snapshot_id, ?, ?, ?, ?, ?, ?, 0 FROM runs WHERE run_id=?
                """,
                (
                    run_id,
                    operation,
                    batch.node_id,
                    max(0, outcome.elapsed_us),
                    max(0, outcome.queue_exit_ns),
                    max(0, outcome.finished_ns),
                    run_id,
                ),
            )
            committed = [
                int(row["batch_index"])
                for row in self._conn.execute(
                    "SELECT batch_index FROM run_batches WHERE run_id=? AND status='committed' ORDER BY batch_index",
                    (run_id,),
                ).fetchall()
            ]
            cursor = 0
            for index in committed:
                if index == cursor:
                    cursor += 1
                else:
                    break
            self._conn.execute(
                "UPDATE runs SET progress_cursor=?, updated_mono_ns=? WHERE run_id=?",
                (cursor, now_ns, run_id),
            )

    def mark_unknown_batches(self, run_id: str) -> list[int]:
        with self._transaction():
            rows = self._conn.execute(
                "SELECT batch_index FROM run_batches WHERE run_id=? AND status IN ('pending','running','failed') ORDER BY batch_index",
                (run_id,),
            ).fetchall()
            indices = [int(row["batch_index"]) for row in rows]
            self._conn.execute(
                """
                UPDATE run_batches SET status='unknown', was_unknown=1
                WHERE run_id=? AND status IN ('pending','running','failed')
                """,
                (run_id,),
            )
            self._conn.execute(
                "UPDATE runs SET status='interrupted', updated_mono_ns=? WHERE run_id=?",
                (time.perf_counter_ns(), run_id),
            )
        return indices

    def finalize_run(
        self,
        *,
        dataset: Dataset,
        run_id: str,
        snapshot_id: int,
        finalize_response: dict[str, Any],
        concurrency: dict[str, Any],
    ) -> dict[str, Any]:
        trace_results = finalize_response.get("trace_results")
        if not isinstance(trace_results, list):
            raise ArbiterError("E_PROTOCOL", "finalize response is missing trace_results")
        trace_results = sorted(trace_results, key=lambda item: item["trace_id"])
        global_summary = finalize_response.get("decision_summary")
        if not isinstance(global_summary, str):
            raise ArbiterError("E_PROTOCOL", "finalize response is missing decision_summary")

        old_snapshot = self._read_one("SELECT * FROM snapshots WHERE is_current=1")
        old_mapping: dict[str, dict[str, Any]] = {}
        if old_snapshot is not None:
            for row in self._read_all(
                """
                SELECT st.trace_id, st.rule_versions_json, le.entry_id, le.entry_json,
                       le.entry_digest
                FROM snapshot_traces st
                JOIN ledger_entries le ON le.entry_id = st.entry_id
                WHERE st.snapshot_id=?
                """,
                (old_snapshot["snapshot_id"],),
            ):
                old_mapping[row["trace_id"]] = row

        nodes_by_id = {node.node_id: node for node in dataset.nodes}
        new_trace_ids: set[str] = set()
        flips: list[dict[str, Any]] = []
        new_entries = 0
        reused_entries = 0
        now_ns = time.perf_counter_ns()
        with self._transaction():
            for trace in trace_results:
                trace_id = trace.get("trace_id")
                if not isinstance(trace_id, str) or trace_id in new_trace_ids:
                    raise ArbiterError("E_PROTOCOL", "finalize response has duplicate or invalid trace_id")
                new_trace_ids.add(trace_id)
                trace_summary = canonical_text(trace)
                decision_digest = digest_text(trace_summary)
                entry_payload = {
                    "decision_digest": decision_digest,
                    "decision_summary": trace_summary,
                    "format_version": 1,
                    "participating_nodes": sorted(trace["participating_nodes"]),
                    "root_span_id": trace["root_span_id"],
                    "rule_difference": trace["rule_difference"],
                    "sampled_node_ids": sorted(trace["sampled_node_ids"]),
                    "sampled_span_ids": sorted(trace["sampled_span_ids"]),
                    "trace_id": trace_id,
                }
                entry_json = canonical_text(entry_payload)
                entry_digest = digest_text(entry_json)
                existing_entry = self._conn.execute(
                    "SELECT entry_id FROM ledger_entries WHERE trace_id=? AND entry_digest=?",
                    (trace_id, entry_digest),
                ).fetchone()
                if existing_entry is not None:
                    entry_id = int(existing_entry["entry_id"])
                    reused_entries += 1
                else:
                    cursor = self._conn.execute(
                        """
                        INSERT INTO ledger_entries(trace_id, format_version, entry_json, entry_digest)
                        VALUES(?,?,?,?)
                        """,
                        (trace_id, 1, entry_json, entry_digest),
                    )
                    entry_id = int(cursor.lastrowid)
                    new_entries += 1

                version_set = sorted(
                    {nodes_by_id[node_id].propagation_rule_version for node_id in trace["participating_nodes"]}
                )
                rules_json = canonical_text(version_set)
                self._conn.execute(
                    """
                    INSERT INTO snapshot_traces(snapshot_id, trace_id, entry_id, rule_versions_json)
                    VALUES(?,?,?,?)
                    """,
                    (snapshot_id, trace_id, entry_id, rules_json),
                )

                before = old_mapping.get(trace_id)
                if old_snapshot is not None and before is not None:
                    before_payload = json.loads(before["entry_json"])
                    diff_fields: list[str] = []
                    for field_name in ("root_span_id", "participating_nodes", "sampled_node_ids"):
                        if before_payload.get(field_name) != entry_payload.get(field_name):
                            diff_fields.append(field_name)
                    if diff_fields:
                        self._conn.execute(
                            """
                            INSERT INTO flips(
                                snapshot_id, trace_id, change_kind, before_rules_json,
                                after_rules_json, diff_fields_json,
                                before_entry_digest, after_entry_digest
                            ) VALUES(?,?,?,?,?,?,?,?)
                            """,
                            (
                                snapshot_id,
                                trace_id,
                                "changed",
                                before["rule_versions_json"],
                                rules_json,
                                canonical_text(diff_fields),
                                before["entry_digest"],
                                entry_digest,
                            ),
                        )
                        flips.append(
                            {
                                "trace_id": trace_id,
                                "change_kind": "changed",
                                "diff_fields": diff_fields,
                            }
                        )
                elif old_snapshot is not None and before is None:
                    self._conn.execute(
                        """
                        INSERT INTO flips(
                            snapshot_id, trace_id, change_kind, before_rules_json,
                            after_rules_json, diff_fields_json,
                            before_entry_digest, after_entry_digest
                        ) VALUES(?,?,?,?,?,?,?,?)
                        """,
                        (
                            snapshot_id,
                            trace_id,
                            "added",
                            None,
                            rules_json,
                            canonical_text(["trace_added"]),
                            None,
                            entry_digest,
                        ),
                    )
                    flips.append(
                        {
                            "trace_id": trace_id,
                            "change_kind": "added",
                            "diff_fields": ["trace_added"],
                        }
                    )

            if old_snapshot is not None:
                for trace_id in sorted(set(old_mapping) - new_trace_ids):
                    before = old_mapping[trace_id]
                    self._conn.execute(
                        """
                        INSERT INTO flips(
                            snapshot_id, trace_id, change_kind, before_rules_json,
                            after_rules_json, diff_fields_json,
                            before_entry_digest, after_entry_digest
                        ) VALUES(?,?,?,?,?,?,?,?)
                        """,
                        (
                            snapshot_id,
                            trace_id,
                            "removed",
                            before["rule_versions_json"],
                            None,
                            canonical_text(["trace_removed"]),
                            before["entry_digest"],
                            None,
                        ),
                    )
                    flips.append(
                        {
                            "trace_id": trace_id,
                            "change_kind": "removed",
                            "diff_fields": ["trace_removed"],
                        }
                    )

            self._conn.execute(
                "UPDATE snapshots SET is_current=0, is_stale=1 WHERE is_current=1"
            )
            self._conn.execute(
                "UPDATE snapshots SET is_current=1, is_stale=0 WHERE snapshot_id=?",
                (snapshot_id,),
            )
            self._conn.execute(
                "UPDATE latency_samples SET is_stale=1 WHERE snapshot_id<>?",
                (snapshot_id,),
            )
            self._conn.execute(
                "UPDATE latency_stats SET is_stale=1 WHERE snapshot_id<>?",
                (snapshot_id,),
            )
            self._conn.execute("UPDATE flips SET is_stale=1 WHERE snapshot_id<>?", (snapshot_id,))
            self._conn.execute(
                """
                UPDATE runs SET status='completed', progress_cursor=batch_count,
                    concurrency_json=?, updated_mono_ns=?, error_json=NULL
                WHERE run_id=?
                """,
                (canonical_text(concurrency), now_ns, run_id),
            )

        self.rebuild_latency_stats(snapshot_id)
        return {
            "flips": flips,
            "new_entries": new_entries,
            "previous_snapshot_id": old_snapshot["snapshot_id"] if old_snapshot else None,
            "reused_entries": reused_entries,
            "service_decision_digest": finalize_response.get("decision_digest"),
            "snapshot_id": snapshot_id,
            "trace_count": len(trace_results),
        }

    def incomplete_run(self) -> dict[str, Any] | None:
        return self._read_one(
            "SELECT * FROM runs WHERE status<>'completed' ORDER BY rowid DESC LIMIT 1"
        )

    def latest_run(self) -> dict[str, Any] | None:
        return self._read_one("SELECT * FROM runs ORDER BY rowid DESC LIMIT 1")

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return self._read_one("SELECT * FROM runs WHERE run_id=?", (run_id,))

    def list_batches(self, run_id: str) -> list[dict[str, Any]]:
        return self._read_all(
            "SELECT * FROM run_batches WHERE run_id=? ORDER BY batch_index",
            (run_id,),
        )

    def get_snapshot(self, snapshot_id: int | None = None) -> dict[str, Any]:
        if snapshot_id is None:
            row = self._current_snapshot_row()
        else:
            row = self._read_one("SELECT * FROM snapshots WHERE snapshot_id=?", (snapshot_id,))
        if row is None:
            raise ArbiterError("E_RECOVERY", "no matching snapshot is available")
        row["nodes"] = json.loads(row.pop("nodes_json"))
        row["spans"] = json.loads(row.pop("spans_json"))
        row["rule_versions"] = json.loads(row.pop("rule_versions_json"))
        return row

    def verify_snapshot_digest(self, snapshot_id: int, *, inject_mismatch: bool = False) -> dict[str, Any]:
        snapshot = self.get_snapshot(snapshot_id)
        actual = digest_bytes(
            canonical_text(
                {
                    "format_version": snapshot["format_version"],
                    "ruleset_version": snapshot["ruleset_version"],
                    "nodes": snapshot["nodes"],
                    "spans": snapshot["spans"],
                }
            ).encode("utf-8")
        )
        expected = snapshot["content_digest"]
        if inject_mismatch or actual != expected:
            raise ArbiterError(
                "E_SNAPSHOT",
                "snapshot content digest does not match its stored payload",
                details={
                    "snapshot_id": snapshot_id,
                    "expected_digest": expected,
                    "actual_digest": actual,
                    "injected": inject_mismatch,
                },
            )
        derived_versions = sorted(
            {int(node["propagation_rule_version"]) for node in snapshot["nodes"]}
        )
        if snapshot["rule_versions"] != derived_versions:
            raise ArbiterError(
                "E_SNAPSHOT",
                "snapshot rule version set does not match its node declarations",
                details={
                    "snapshot_id": snapshot_id,
                    "stored_rule_versions": snapshot["rule_versions"],
                    "derived_rule_versions": derived_versions,
                },
            )
        return snapshot

    def snapshot_traces(
        self,
        snapshot_id: int,
        *,
        include_raw_entry: bool = False,
    ) -> list[dict[str, Any]]:
        rows = self._read_all(
            """
            SELECT st.trace_id, st.rule_versions_json, le.entry_id, le.format_version,
                   le.entry_json, le.entry_digest
            FROM snapshot_traces st
            JOIN ledger_entries le ON le.entry_id=st.entry_id
            WHERE st.snapshot_id=?
            ORDER BY st.trace_id
            """,
            (snapshot_id,),
        )
        for row in rows:
            row["rule_versions"] = json.loads(row.pop("rule_versions_json"))
            raw_entry_json = row.pop("entry_json")
            row["entry"] = json.loads(raw_entry_json)
            if include_raw_entry:
                row["entry_json"] = raw_entry_json
        return rows

    def flips_for_snapshot(self, snapshot_id: int) -> list[dict[str, Any]]:
        rows = self._read_all(
            "SELECT * FROM flips WHERE snapshot_id=? AND is_stale=0 ORDER BY trace_id, change_kind, flip_id",
            (snapshot_id,),
        )
        for row in rows:
            row["diff_fields"] = json.loads(row.pop("diff_fields_json"))
            row["before_rule_versions"] = (
                json.loads(row.pop("before_rules_json")) if row["before_rules_json"] else None
            ) if "before_rules_json" in row else None
            row["after_rule_versions"] = (
                json.loads(row["after_rules_json"]) if row["after_rules_json"] else None
            )
            row.pop("before_rules_json", None)
            row.pop("after_rules_json", None)
        return rows

    def record_latency(
        self,
        *,
        snapshot_id: int,
        operation: str,
        node_id: str | None,
        started_ns: int,
        ended_ns: int,
        rebuild: bool = True,
    ) -> dict[str, Any]:
        elapsed_us = max(0, (ended_ns - started_ns) // 1000)
        with self._transaction():
            self._conn.execute(
                """
                INSERT INTO latency_samples(
                    snapshot_id, run_id, operation, node_id, elapsed_us,
                    started_mono_ns, ended_mono_ns, is_stale
                ) VALUES(?, NULL, ?, ?, ?, ?, ?, 0)
                """,
                (
                    snapshot_id,
                    operation,
                    node_id or "END_TO_END",
                    elapsed_us,
                    started_ns,
                    ended_ns,
                ),
            )
        if rebuild:
            self.rebuild_latency_stats(snapshot_id, operation=operation)
        return {"elapsed_us": elapsed_us, "operation": operation, "node_id": node_id or "END_TO_END"}

    def rebuild_latency_stats(self, snapshot_id: int, *, operation: str | None = None) -> list[dict[str, Any]]:
        params: list[Any] = [snapshot_id]
        condition = ""
        if operation is not None:
            condition = " AND operation=?"
            params.append(operation)
        rows = self._read_all(
            "SELECT operation, node_id, elapsed_us FROM latency_samples WHERE is_stale=0"
            " AND snapshot_id=?" + condition,
            params,
        )
        grouped: dict[tuple[str, str], list[int]] = {}
        for row in rows:
            grouped.setdefault((row["operation"], row["node_id"]), []).append(int(row["elapsed_us"]))

        operations = {key[0] for key in grouped}
        stats_rows: list[dict[str, Any]] = []
        with self._transaction():
            if operation is None:
                self._conn.execute(
                    "DELETE FROM latency_stats WHERE snapshot_id=? AND is_stale=0",
                    (snapshot_id,),
                )
            else:
                self._conn.execute(
                    "DELETE FROM latency_stats WHERE snapshot_id=? AND operation=?",
                    (snapshot_id, operation),
                )
            for op in sorted(operations):
                node_values = {
                    node_id: values
                    for (group_op, node_id), values in grouped.items()
                    if group_op == op and node_id != "END_TO_END"
                }
                groups = dict(node_values)
                if node_values:
                    overall_values = [value for values in node_values.values() for value in values]
                else:
                    overall_values = grouped.get((op, "END_TO_END"), [])
                if overall_values:
                    groups["ALL"] = overall_values
                end_to_end_values = grouped.get((op, "END_TO_END"), [])
                if end_to_end_values:
                    groups["END_TO_END"] = end_to_end_values
                for node_id in sorted(groups):
                    values = groups[node_id]
                    count, mean_us, p95_us, max_us = _stats_for_values(values)
                    self._conn.execute(
                        """
                        INSERT INTO latency_stats(
                            snapshot_id, operation, node_id, sample_count, mean_us,
                            p95_us, max_us, scale_factor, rounding,
                            quantization_error_us, quantization_error_ns, is_stale
                        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,0)
                        """,
                        (
                            snapshot_id,
                            op,
                            node_id,
                            count,
                            mean_us,
                            p95_us,
                            max_us,
                            STAT_SCALE_FACTOR,
                            STAT_ROUNDING,
                            STAT_QUANTIZATION_ERROR_US,
                            STAT_QUANTIZATION_ERROR_NS,
                        ),
                    )
                    stats_rows.append(
                        {
                            "node_id": node_id,
                            "operation": op,
                            "sample_count": count,
                            "mean_us": mean_us,
                            "p95_us": p95_us,
                            "max_us": max_us,
                            "scale_factor": STAT_SCALE_FACTOR,
                            "rounding": STAT_ROUNDING,
                            "quantization_error_us": STAT_QUANTIZATION_ERROR_US,
                            "quantization_error_ns": STAT_QUANTIZATION_ERROR_NS,
                        }
                    )
        return stats_rows

    def latency_stats(self, snapshot_id: int) -> list[dict[str, Any]]:
        return self._read_all(
            """
            SELECT operation, node_id, sample_count, mean_us, p95_us, max_us,
                   scale_factor, rounding, quantization_error_us, quantization_error_ns
            FROM latency_stats WHERE snapshot_id=? AND is_stale=0
            ORDER BY operation, node_id
            """,
            (snapshot_id,),
        )

    def load_view(self) -> dict[str, Any]:
        snapshot = self._current_snapshot_row()
        if snapshot is None:
            raise ArbiterError("E_RECOVERY", "ledger has no current snapshot")
        snapshot_id = int(snapshot["snapshot_id"])
        traces = self.snapshot_traces(snapshot_id, include_raw_entry=True)
        for row in traces:
            raw_entry_json = row.pop("entry_json")
            row["entry_digest_valid"] = digest_text(raw_entry_json) == row["entry_digest"]
        flips = self.flips_for_snapshot(snapshot_id)
        snapshot_payload = {
            "format_version": snapshot["format_version"],
            "ruleset_version": snapshot["ruleset_version"],
            "nodes": json.loads(snapshot["nodes_json"]),
            "spans": json.loads(snapshot["spans_json"]),
        }
        computed_snapshot_digest = digest_text(canonical_text(snapshot_payload))
        stored_rule_versions = json.loads(snapshot["rule_versions_json"])
        derived_rule_versions = sorted(
            {
                int(node["propagation_rule_version"])
                for node in snapshot_payload["nodes"]
            }
        )
        previous = self._read_one(
            """
            SELECT snapshot_id, ruleset_version FROM snapshots
            WHERE snapshot_id<>?
              AND EXISTS(
                  SELECT 1 FROM snapshot_traces st
                  WHERE st.snapshot_id=snapshots.snapshot_id
              )
            ORDER BY snapshot_id DESC LIMIT 1
            """,
            (snapshot_id,),
        )
        return {
            "authority": "cached_ledger_view_only",
            "changed_traces": sorted(
                flips,
                key=lambda item: (item["trace_id"], item["change_kind"], item["flip_id"]),
            ),
            "current_ruleset_version": snapshot["ruleset_version"],
            "flips": flips,
            "latency_stats": self.latency_stats(snapshot_id),
            "previous_ruleset_version": previous["ruleset_version"] if previous else None,
            "snapshot_id": snapshot_id,
            "snapshot_content_digest": snapshot["content_digest"],
            "snapshot_digest_valid": computed_snapshot_digest == snapshot["content_digest"],
            "snapshot_rule_versions_valid": stored_rule_versions == derived_rule_versions,
            "stale_sources_excluded": True,
            "traces": traces,
            "view_mode": "full_replacement",
        }

    def verify_ledger_entries(
        self,
        snapshot_id: int | None = None,
        *,
        error_code: str = "E_REPLAY",
    ) -> dict[str, Any]:
        if snapshot_id is None:
            snapshot = self._current_snapshot_row()
            if snapshot is None:
                raise ArbiterError("E_RECOVERY", "ledger has no current snapshot")
            snapshot_id = int(snapshot["snapshot_id"])
        traces = self.snapshot_traces(snapshot_id, include_raw_entry=True)
        for row in traces:
            actual = digest_text(row["entry_json"])
            if actual != row["entry_digest"]:
                raise ArbiterError(
                    error_code,
                    "ledger entry content digest mismatch",
                    trace_id=row["trace_id"],
                    field="entry_digest",
                    details={"expected": row["entry_digest"], "actual": actual},
                )
        return {"entries_checked": len(traces), "ok": True, "snapshot_id": snapshot_id}
