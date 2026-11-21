from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .canonical import chained_digest, digest_obj
from .errors import FORMAT_VERSION, BusinessError
from .model import Event, LateDecision, SessionSpec, WatermarkTrace, WindowKey, WindowResult


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    format_version INTEGER NOT NULL,
    spec_digest TEXT NOT NULL,
    rules_version INTEGER NOT NULL,
    fixed_scale INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS session_runtime (
    session_id TEXT PRIMARY KEY,
    rules_version INTEGER NOT NULL,
    fixed_scale INTEGER NOT NULL,
    current_view_id TEXT NOT NULL,
    event_cursor INTEGER NOT NULL,
    last_batch_digest TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS view_manifest (
    session_id TEXT NOT NULL,
    view_id TEXT NOT NULL,
    format_version INTEGER NOT NULL,
    rules_version INTEGER NOT NULL,
    fixed_scale INTEGER NOT NULL,
    spec_digest TEXT NOT NULL,
    status TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    previous_view_id TEXT,
    staleness_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (session_id, view_id)
);

CREATE TABLE IF NOT EXISTS sources (
    session_id TEXT NOT NULL,
    view_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    seed INTEGER NOT NULL,
    event_count INTEGER NOT NULL,
    out_of_order_ratio_bp INTEGER NOT NULL,
    event_digest TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    format_version INTEGER NOT NULL,
    content_digest TEXT NOT NULL,
    PRIMARY KEY (session_id, view_id, source_id)
);

CREATE TABLE IF NOT EXISTS rules (
    session_id TEXT NOT NULL,
    view_id TEXT NOT NULL,
    rule_id TEXT NOT NULL,
    rule_version INTEGER NOT NULL,
    width_ms INTEGER NOT NULL,
    lateness_ms INTEGER NOT NULL,
    partition_keys_json TEXT NOT NULL,
    aggregation_functions_json TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    format_version INTEGER NOT NULL,
    content_digest TEXT NOT NULL,
    PRIMARY KEY (session_id, view_id, rule_id)
);

CREATE TABLE IF NOT EXISTS watermark_trace (
    trace_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    view_id TEXT NOT NULL,
    format_version INTEGER NOT NULL,
    event_id TEXT NOT NULL,
    arrival_no INTEGER NOT NULL,
    partition_key TEXT NOT NULL,
    event_time_ms INTEGER NOT NULL,
    watermark_before_ms INTEGER NOT NULL,
    watermark_after_ms INTEGER NOT NULL,
    was_late INTEGER NOT NULL,
    finalized_count INTEGER NOT NULL,
    content_digest TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (session_id, view_id, arrival_no, event_id)
);

CREATE INDEX IF NOT EXISTS ix_watermark_trace_view
    ON watermark_trace(session_id, view_id, partition_key, arrival_no);

CREATE TABLE IF NOT EXISTS window_state (
    session_id TEXT NOT NULL,
    view_id TEXT NOT NULL,
    format_version INTEGER NOT NULL,
    rule_id TEXT NOT NULL,
    rule_version INTEGER NOT NULL,
    partition_key TEXT NOT NULL,
    window_start_ms INTEGER NOT NULL,
    window_end_ms INTEGER NOT NULL,
    status TEXT NOT NULL,
    count INTEGER NOT NULL,
    sum_fixed INTEGER NOT NULL,
    mean_fixed INTEGER NOT NULL,
    sum_exact TEXT NOT NULL,
    quantization_error_sum INTEGER NOT NULL,
    max_single_quantization_error INTEGER NOT NULL,
    late_accepted_count INTEGER NOT NULL,
    late_abs_amount INTEGER NOT NULL,
    boundary_abs_amount INTEGER NOT NULL,
    committed_at TEXT,
    stale INTEGER NOT NULL DEFAULT 0,
    state_digest TEXT NOT NULL,
    PRIMARY KEY (
        session_id, view_id, rule_id, partition_key, window_start_ms, window_end_ms
    )
);

CREATE INDEX IF NOT EXISTS ix_window_state_current
    ON window_state(session_id, view_id, stale, status);

CREATE TABLE IF NOT EXISTS late_decisions (
    decision_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    view_id TEXT NOT NULL,
    format_version INTEGER NOT NULL,
    rule_id TEXT NOT NULL,
    rule_version INTEGER NOT NULL,
    partition_key TEXT NOT NULL,
    event_id TEXT NOT NULL,
    arrival_no INTEGER NOT NULL,
    event_time_ms INTEGER NOT NULL,
    window_start_ms INTEGER NOT NULL,
    window_end_ms INTEGER NOT NULL,
    decision TEXT NOT NULL,
    reason TEXT NOT NULL,
    fixed_value INTEGER NOT NULL,
    budget_count_before INTEGER NOT NULL,
    budget_count_after INTEGER NOT NULL,
    budget_amount_before INTEGER NOT NULL,
    budget_amount_after INTEGER NOT NULL,
    cumulative_late_abs_amount INTEGER NOT NULL,
    was_late INTEGER NOT NULL,
    stale INTEGER NOT NULL DEFAULT 0,
    evidence_digest TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_late_decisions_current
    ON late_decisions(session_id, view_id, stale, decision, partition_key);

CREATE TABLE IF NOT EXISTS error_attribution (
    session_id TEXT NOT NULL,
    view_id TEXT NOT NULL,
    format_version INTEGER NOT NULL,
    rule_id TEXT NOT NULL,
    rule_version INTEGER NOT NULL,
    partition_key TEXT NOT NULL,
    window_start_ms INTEGER NOT NULL,
    window_end_ms INTEGER NOT NULL,
    quantization_truncation INTEGER NOT NULL,
    late_admission INTEGER NOT NULL,
    window_boundary INTEGER NOT NULL,
    sum_bound INTEGER NOT NULL,
    mean_bound INTEGER NOT NULL,
    single_value_quantization INTEGER NOT NULL,
    dominant_segment TEXT NOT NULL,
    dominant_share_bp INTEGER NOT NULL,
    threshold_bp INTEGER NOT NULL,
    quantization_share_bp INTEGER NOT NULL,
    late_share_bp INTEGER NOT NULL,
    boundary_share_bp INTEGER NOT NULL,
    formula_version TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    stale INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    PRIMARY KEY (
        session_id, view_id, rule_id, partition_key, window_start_ms, window_end_ms
    )
);

CREATE TABLE IF NOT EXISTS ambiguity_pairs (
    pair_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    view_id TEXT NOT NULL,
    format_version INTEGER NOT NULL,
    key_a_json TEXT NOT NULL,
    key_b_json TEXT NOT NULL,
    left_mean_fixed INTEGER NOT NULL,
    right_mean_fixed INTEGER NOT NULL,
    combined_mean_bound INTEGER NOT NULL,
    stable_winner_json TEXT NOT NULL,
    error_affected INTEGER NOT NULL,
    stale INTEGER NOT NULL DEFAULT 0,
    evidence_digest TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ledger_batches (
    batch_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    view_id TEXT NOT NULL,
    format_version INTEGER NOT NULL,
    batch_no INTEGER NOT NULL,
    rules_version INTEGER NOT NULL,
    fixed_scale INTEGER NOT NULL,
    event_cursor INTEGER NOT NULL,
    content_digest TEXT NOT NULL,
    previous_digest TEXT NOT NULL,
    payload_digest TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_ledger_batches_view
    ON ledger_batches(session_id, view_id, batch_no);

CREATE TABLE IF NOT EXISTS staleness_events (
    event_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    from_view_id TEXT NOT NULL,
    to_view_id TEXT NOT NULL,
    rules_version INTEGER NOT NULL,
    reason TEXT NOT NULL,
    affected_json TEXT NOT NULL,
    format_version INTEGER NOT NULL,
    content_digest TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    view_id TEXT NOT NULL,
    format_version INTEGER NOT NULL,
    rules_version INTEGER NOT NULL,
    spec_digest TEXT NOT NULL,
    content_digest TEXT NOT NULL,
    ledger_content_digest TEXT NOT NULL,
    event_cursor INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class Ledger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path), timeout=30.0)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = FULL")
        self.connection.executescript(SCHEMA)
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            yield self.connection
        except Exception:
            self.connection.rollback()
            raise
        else:
            self.connection.commit()

    def register_session(self, session_id: str, spec: SessionSpec, view_id: str, event_digests: dict[str, str]) -> None:
        now = utc_now()
        spec_digest = spec.spec_digest
        initial_digest = digest_obj(
            {
                "session_id": session_id,
                "view_id": view_id,
                "spec_digest": spec_digest,
                "format_version": spec.format_version,
            }
        )
        with self.transaction() as conn:
            existing = conn.execute(
                "SELECT current_view_id FROM session_runtime WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if existing is not None:
                if existing["current_view_id"] == view_id:
                    return
                raise ValueError(f"session {session_id!r} is already registered with another view")
            conn.execute(
                "INSERT INTO sessions(session_id, format_version, spec_digest, rules_version, fixed_scale, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    spec.format_version,
                    spec_digest,
                    spec.rules_version,
                    spec.fixed_scale,
                    now,
                    now,
                ),
            )
            conn.execute(
                "INSERT INTO session_runtime(session_id, rules_version, fixed_scale, current_view_id, event_cursor, last_batch_digest, updated_at) VALUES (?, ?, ?, ?, 0, ?, ?)",
                (session_id, spec.rules_version, spec.fixed_scale, view_id, initial_digest, now),
            )
            conn.execute(
                "INSERT INTO view_manifest(session_id, view_id, format_version, rules_version, fixed_scale, spec_digest, status, content_digest, previous_view_id, staleness_reason, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, 'ACTIVE', ?, NULL, NULL, ?, ?)",
                (
                    session_id,
                    view_id,
                    spec.format_version,
                    spec.rules_version,
                    spec.fixed_scale,
                    spec_digest,
                    initial_digest,
                    now,
                    now,
                ),
            )
            self._insert_spec_rows(conn, session_id, view_id, spec, event_digests, now)

    def _insert_spec_rows(
        self,
        conn: sqlite3.Connection,
        session_id: str,
        view_id: str,
        spec: SessionSpec,
        event_digests: dict[str, str],
        now: str,
    ) -> None:
        for source in spec.sources:
            source_payload = source.to_dict()
            conn.execute(
                "INSERT INTO sources(session_id, view_id, source_id, seed, event_count, out_of_order_ratio_bp, event_digest, spec_json, format_version, content_digest) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    view_id,
                    source.source_id,
                    source.seed,
                    source.event_count,
                    source.out_of_order_ratio_bp,
                    event_digests[source.source_id],
                    _json(source_payload),
                    spec.format_version,
                    digest_obj(source_payload),
                ),
            )
        for rule in spec.rules:
            rule_payload = rule.to_dict()
            conn.execute(
                "INSERT INTO rules(session_id, view_id, rule_id, rule_version, width_ms, lateness_ms, partition_keys_json, aggregation_functions_json, spec_json, format_version, content_digest) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    view_id,
                    rule.rule_id,
                    rule.rule_version,
                    rule.width_ms,
                    rule.lateness_ms,
                    _json(list(rule.partition_keys)),
                    _json(list(rule.aggregation_functions)),
                    _json(rule_payload),
                    spec.format_version,
                    digest_obj(rule_payload),
                ),
            )

    def evolve_session(
        self,
        *,
        session_id: str,
        old_view_id: str,
        new_view_id: str,
        new_spec: SessionSpec,
        reason: str,
        affected: Sequence[dict[str, Any]],
        event_digests: dict[str, str],
    ) -> str:
        now = utc_now()
        old_runtime = self.get_runtime(session_id)
        staleness_payload = {
            "session_id": session_id,
            "from_view_id": old_view_id,
            "to_view_id": new_view_id,
            "rules_version": new_spec.rules_version,
            "reason": reason,
            "affected": list(affected),
            "format_version": new_spec.format_version,
        }
        staleness_digest = digest_obj(staleness_payload)
        with self.transaction() as conn:
            conn.execute(
                "UPDATE window_state SET stale = 1 WHERE session_id = ? AND view_id = ?",
                (session_id, old_view_id),
            )
            conn.execute(
                "UPDATE late_decisions SET stale = 1 WHERE session_id = ? AND view_id = ?",
                (session_id, old_view_id),
            )
            conn.execute(
                "UPDATE error_attribution SET stale = 1 WHERE session_id = ? AND view_id = ?",
                (session_id, old_view_id),
            )
            conn.execute(
                "UPDATE ambiguity_pairs SET stale = 1 WHERE session_id = ? AND view_id = ?",
                (session_id, old_view_id),
            )
            conn.execute(
                "UPDATE view_manifest SET status = 'STALE', staleness_reason = ?, updated_at = ? WHERE session_id = ? AND view_id = ?",
                (reason, now, session_id, old_view_id),
            )
            new_view_digest = digest_obj(
                {
                    "session_id": session_id,
                    "view_id": new_view_id,
                    "spec_digest": new_spec.spec_digest,
                    "previous_view_id": old_view_id,
                }
            )
            conn.execute(
                "INSERT INTO view_manifest(session_id, view_id, format_version, rules_version, fixed_scale, spec_digest, status, content_digest, previous_view_id, staleness_reason, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, 'ACTIVE', ?, ?, NULL, ?, ?)",
                (
                    session_id,
                    new_view_id,
                    new_spec.format_version,
                    new_spec.rules_version,
                    new_spec.fixed_scale,
                    new_spec.spec_digest,
                    new_view_digest,
                    old_view_id,
                    now,
                    now,
                ),
            )
            self._insert_spec_rows(conn, session_id, new_view_id, new_spec, event_digests, now)
            conn.execute(
                "UPDATE sessions SET rules_version = ?, fixed_scale = ?, spec_digest = ?, updated_at = ? WHERE session_id = ?",
                (new_spec.rules_version, new_spec.fixed_scale, new_spec.spec_digest, now, session_id),
            )
            conn.execute(
                "UPDATE session_runtime SET rules_version = ?, fixed_scale = ?, current_view_id = ?, event_cursor = 0, last_batch_digest = ?, updated_at = ? WHERE session_id = ?",
                (new_spec.rules_version, new_spec.fixed_scale, new_view_id, new_view_digest, now, session_id),
            )
            conn.execute(
                "INSERT INTO staleness_events(event_id, session_id, from_view_id, to_view_id, rules_version, reason, affected_json, format_version, content_digest, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    digest_obj({"from": old_view_id, "to": new_view_id, "reason": reason})[:32],
                    session_id,
                    old_view_id,
                    new_view_id,
                    new_spec.rules_version,
                    reason,
                    _json(list(affected)),
                    new_spec.format_version,
                    staleness_digest,
                    now,
                ),
            )
        return staleness_digest

    def stage_window(self, session_id: str, view_id: str, key: WindowKey) -> None:
        now = utc_now()
        payload = {
            **key.to_dict(),
            "session_id": session_id,
            "view_id": view_id,
            "status": "PENDING",
            "format_version": FORMAT_VERSION,
        }
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO window_state(session_id, view_id, format_version, rule_id, rule_version, partition_key, window_start_ms, window_end_ms, status, count, sum_fixed, mean_fixed, sum_exact, quantization_error_sum, max_single_quantization_error, late_accepted_count, late_abs_amount, boundary_abs_amount, committed_at, stale, state_digest) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', 0, 0, 0, '0', 0, 0, 0, 0, 0, NULL, 0, ?)",
                (
                    session_id,
                    view_id,
                    FORMAT_VERSION,
                    key.rule_id,
                    key.rule_version,
                    key.partition_key,
                    key.window_start_ms,
                    key.window_end_ms,
                    digest_obj(payload),
                ),
            )

    def record_event(
        self,
        session_id: str,
        view_id: str,
        trace: WatermarkTrace,
        decisions: Sequence[LateDecision],
        event_cursor: int,
    ) -> None:
        now = utc_now()
        trace_payload = {
            **trace.to_dict(),
            "session_id": session_id,
            "view_id": view_id,
            "format_version": FORMAT_VERSION,
        }
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO watermark_trace(trace_id, session_id, view_id, format_version, event_id, arrival_no, partition_key, event_time_ms, watermark_before_ms, watermark_after_ms, was_late, finalized_count, content_digest, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    trace.trace_id,
                    session_id,
                    view_id,
                    FORMAT_VERSION,
                    trace.event_id,
                    trace.arrival_no,
                    trace.partition_key,
                    trace.event_time_ms,
                    trace.watermark_before_ms,
                    trace.watermark_after_ms,
                    int(trace.was_late),
                    trace.finalized_count,
                    digest_obj(trace_payload),
                    now,
                ),
            )
            for decision in decisions:
                decision_payload = {
                    **decision.to_dict(),
                    "session_id": session_id,
                    "view_id": view_id,
                    "format_version": FORMAT_VERSION,
                }
                conn.execute(
                    "INSERT INTO late_decisions(decision_id, session_id, view_id, format_version, rule_id, rule_version, partition_key, event_id, arrival_no, event_time_ms, window_start_ms, window_end_ms, decision, reason, fixed_value, budget_count_before, budget_count_after, budget_amount_before, budget_amount_after, cumulative_late_abs_amount, was_late, stale, evidence_digest, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)",
                    (
                        decision.decision_id,
                        session_id,
                        view_id,
                        FORMAT_VERSION,
                        decision.rule_id,
                        decision.rule_version,
                        decision.partition_key,
                        decision.event_id,
                        decision.arrival_no,
                        decision.event_time_ms,
                        decision.window_start_ms,
                        decision.window_end_ms,
                        decision.decision,
                        decision.reason,
                        decision.fixed_value,
                        decision.budget_count_before,
                        decision.budget_count_after,
                        decision.budget_amount_before,
                        decision.budget_amount_after,
                        decision.cumulative_late_abs_amount,
                        int(decision.was_late),
                        digest_obj(decision_payload),
                        now,
                    ),
                )
            conn.execute(
                "UPDATE session_runtime SET event_cursor = ?, updated_at = ? WHERE session_id = ?",
                (event_cursor, now, session_id),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )

    def commit_window_finalization(
        self,
        *,
        session_id: str,
        view_id: str,
        result: WindowResult,
        event_cursor: int,
        ambiguity_pairs: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        now = utc_now()
        runtime = self.get_runtime(session_id)
        previous_digest = runtime["last_batch_digest"]
        window_payload = result.to_dict()
        payload_digest = digest_obj(
            {
                "window": window_payload,
                "ambiguity_pairs": list(ambiguity_pairs),
                "event_cursor": event_cursor,
            }
        )
        content_digest = chained_digest(previous_digest, payload_digest)
        batch_no_row = self.connection.execute(
            "SELECT COALESCE(MAX(batch_no), 0) + 1 AS next_no FROM ledger_batches WHERE session_id = ? AND view_id = ?",
            (session_id, view_id),
        ).fetchone()
        batch_no = int(batch_no_row["next_no"])
        batch_id = digest_obj(
            {"session_id": session_id, "view_id": view_id, "batch_no": batch_no, "content_digest": content_digest}
        )[:40]
        state_payload = {
            **result.to_dict(),
            "session_id": session_id,
            "view_id": view_id,
            "format_version": FORMAT_VERSION,
        }
        attribution_payload = {
            **result.attribution.to_dict(),
            **result.key.to_dict(),
            "session_id": session_id,
            "view_id": view_id,
            "format_version": FORMAT_VERSION,
        }
        with self.transaction() as conn:
            conn.execute(
                "UPDATE window_state SET format_version = ?, rule_version = ?, status = 'COMMITTED', count = ?, sum_fixed = ?, mean_fixed = ?, sum_exact = ?, quantization_error_sum = ?, max_single_quantization_error = ?, late_accepted_count = ?, late_abs_amount = ?, boundary_abs_amount = ?, committed_at = ?, stale = 0, state_digest = ? WHERE session_id = ? AND view_id = ? AND rule_id = ? AND partition_key = ? AND window_start_ms = ? AND window_end_ms = ?",
                (
                    FORMAT_VERSION,
                    result.key.rule_version,
                    result.count,
                    result.sum_fixed,
                    result.mean_fixed,
                    str(result.sum_exact),
                    result.attribution.quantization_truncation,
                    result.attribution.single_value_quantization,
                    result.late_accepted_count,
                    result.late_accepted_abs_amount,
                    result.boundary_abs_amount,
                    now,
                    digest_obj(state_payload),
                    session_id,
                    view_id,
                    result.key.rule_id,
                    result.key.partition_key,
                    result.key.window_start_ms,
                    result.key.window_end_ms,
                ),
            )
            conn.execute(
                "INSERT OR REPLACE INTO error_attribution(session_id, view_id, format_version, rule_id, rule_version, partition_key, window_start_ms, window_end_ms, quantization_truncation, late_admission, window_boundary, sum_bound, mean_bound, single_value_quantization, dominant_segment, dominant_share_bp, threshold_bp, quantization_share_bp, late_share_bp, boundary_share_bp, formula_version, evidence_json, content_digest, stale, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)",
                (
                    session_id,
                    view_id,
                    FORMAT_VERSION,
                    result.key.rule_id,
                    result.key.rule_version,
                    result.key.partition_key,
                    result.key.window_start_ms,
                    result.key.window_end_ms,
                    result.attribution.quantization_truncation,
                    result.attribution.late_admission,
                    result.attribution.window_boundary,
                    result.attribution.sum_bound,
                    result.attribution.mean_bound,
                    result.attribution.single_value_quantization,
                    result.attribution.dominant_segment,
                    result.attribution.dominant_share_bp,
                    result.attribution.threshold_bp,
                    result.attribution.quantization_share_bp,
                    result.attribution.late_share_bp,
                    result.attribution.boundary_share_bp,
                    result.attribution.formula_version,
                    _json(attribution_payload),
                    digest_obj(attribution_payload),
                    now,
                ),
            )
            for pair in ambiguity_pairs:
                pair_payload = {
                    **pair,
                    "session_id": session_id,
                    "view_id": view_id,
                    "format_version": FORMAT_VERSION,
                }
                pair_id = digest_obj(pair_payload)[:40]
                conn.execute(
                    "INSERT OR IGNORE INTO ambiguity_pairs(pair_id, session_id, view_id, format_version, key_a_json, key_b_json, left_mean_fixed, right_mean_fixed, combined_mean_bound, stable_winner_json, error_affected, stale, evidence_digest, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)",
                    (
                        pair_id,
                        session_id,
                        view_id,
                        FORMAT_VERSION,
                        _json(pair["key_a"]),
                        _json(pair["key_b"]),
                        pair["left_mean_fixed"],
                        pair["right_mean_fixed"],
                        pair["combined_mean_bound"],
                        _json(pair["stable_winner"]),
                        int(pair["error_affected"]),
                        digest_obj(pair_payload),
                        now,
                    ),
                )
            conn.execute(
                "INSERT INTO ledger_batches(batch_id, session_id, view_id, format_version, batch_no, rules_version, fixed_scale, event_cursor, content_digest, previous_digest, payload_digest, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    batch_id,
                    session_id,
                    view_id,
                    FORMAT_VERSION,
                    batch_no,
                    result.key.rule_version,
                    runtime["fixed_scale"],
                    event_cursor,
                    content_digest,
                    previous_digest,
                    payload_digest,
                    now,
                ),
            )
            conn.execute(
                "UPDATE session_runtime SET event_cursor = ?, last_batch_digest = ?, updated_at = ? WHERE session_id = ?",
                (event_cursor, content_digest, now, session_id),
            )
            conn.execute(
                "UPDATE view_manifest SET content_digest = ?, updated_at = ? WHERE session_id = ? AND view_id = ?",
                (content_digest, now, session_id, view_id),
            )
        return {
            "batch_id": batch_id,
            "batch_no": batch_no,
            "content_digest": content_digest,
            "previous_digest": previous_digest,
        }

    def mark_pending_unknown(self, session_id: str, view_id: str) -> list[dict[str, Any]]:
        now = utc_now()
        rows = self.connection.execute(
            "SELECT * FROM window_state WHERE session_id = ? AND view_id = ? AND status = 'PENDING' AND stale = 0 ORDER BY partition_key, window_start_ms, window_end_ms, rule_id",
            (session_id, view_id),
        ).fetchall()
        affected: list[dict[str, Any]] = []
        with self.transaction() as conn:
            for row in rows:
                key = {
                    "rule_id": row["rule_id"],
                    "rule_version": row["rule_version"],
                    "partition_key": row["partition_key"],
                    "window_start_ms": row["window_start_ms"],
                    "window_end_ms": row["window_end_ms"],
                }
                payload = {
                    **key,
                    "session_id": session_id,
                    "view_id": view_id,
                    "status": "UNKNOWN",
                    "format_version": FORMAT_VERSION,
                }
                conn.execute(
                    "UPDATE window_state SET status = 'UNKNOWN', state_digest = ?, committed_at = ? WHERE session_id = ? AND view_id = ? AND rule_id = ? AND partition_key = ? AND window_start_ms = ? AND window_end_ms = ?",
                    (
                        digest_obj(payload),
                        now,
                        session_id,
                        view_id,
                        row["rule_id"],
                        row["partition_key"],
                        row["window_start_ms"],
                        row["window_end_ms"],
                    ),
                )
                affected.append(key)
        return affected

    def record_snapshot(
        self,
        *,
        snapshot_id: str,
        session_id: str,
        view_id: str,
        spec_digest: str,
        content_digest: str,
        ledger_content_digest: str,
        event_cursor: int,
        rules_version: int,
        file_path: str,
    ) -> None:
        now = utc_now()
        self.connection.execute(
            "INSERT OR REPLACE INTO snapshots(snapshot_id, session_id, view_id, format_version, rules_version, spec_digest, content_digest, ledger_content_digest, event_cursor, file_path, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                snapshot_id,
                session_id,
                view_id,
                FORMAT_VERSION,
                rules_version,
                spec_digest,
                content_digest,
                ledger_content_digest,
                event_cursor,
                file_path,
                now,
            ),
        )
        self.connection.commit()

    def get_runtime(self, session_id: str) -> sqlite3.Row:
        row = self.connection.execute(
            "SELECT * FROM session_runtime WHERE session_id = ?", (session_id,)
        ).fetchone()
        if row is None:
            raise KeyError(session_id)
        return row

    def get_view_manifest(self, session_id: str, view_id: str | None = None) -> sqlite3.Row:
        if view_id is None:
            view_id = self.get_runtime(session_id)["current_view_id"]
        row = self.connection.execute(
            "SELECT * FROM view_manifest WHERE session_id = ? AND view_id = ?",
            (session_id, view_id),
        ).fetchone()
        if row is None:
            raise KeyError((session_id, view_id))
        return row

    def verify_content_digest(self, session_id: str, view_id: str | None = None) -> str:
        runtime = self.get_runtime(session_id)
        active_view_id = view_id or runtime["current_view_id"]
        manifest = self.get_view_manifest(session_id, active_view_id)
        initial_payload = {
            "session_id": session_id,
            "view_id": active_view_id,
            "spec_digest": manifest["spec_digest"],
        }
        if manifest["previous_view_id"] is None:
            initial_payload["format_version"] = manifest["format_version"]
        else:
            initial_payload["previous_view_id"] = manifest["previous_view_id"]
        expected = digest_obj(initial_payload)
        batches = self.current_batches(session_id, active_view_id)
        for batch in batches:
            if batch["previous_digest"] != expected:
                raise BusinessError(
                    code="E_SNAPSHOT",
                    message="ledger batch chain is broken",
                    field="ledger.previous_digest",
                    details={
                        "session_id": session_id,
                        "view_id": active_view_id,
                        "batch_no": batch["batch_no"],
                        "expected": expected,
                        "actual": batch["previous_digest"],
                    },
                )
            computed = chained_digest(batch["previous_digest"], batch["payload_digest"])
            if computed != batch["content_digest"]:
                raise BusinessError(
                    code="E_SNAPSHOT",
                    message="ledger batch content digest mismatch",
                    field="ledger.content_digest",
                    details={
                        "session_id": session_id,
                        "view_id": active_view_id,
                        "batch_no": batch["batch_no"],
                        "expected": computed,
                        "actual": batch["content_digest"],
                    },
                )
            expected = computed
        if runtime["current_view_id"] == active_view_id and runtime["last_batch_digest"] != expected:
            raise BusinessError(
                code="E_SNAPSHOT",
                message="ledger runtime digest does not match the batch chain",
                field="ledger.runtime_digest",
                details={
                    "session_id": session_id,
                    "view_id": active_view_id,
                    "expected": expected,
                    "actual": runtime["last_batch_digest"],
                },
            )
        if manifest["content_digest"] != expected:
            raise BusinessError(
                code="E_SNAPSHOT",
                message="ledger view digest does not match the batch chain",
                field="ledger.view_digest",
                details={
                    "session_id": session_id,
                    "view_id": active_view_id,
                    "expected": expected,
                    "actual": manifest["content_digest"],
                },
            )
        return expected

    def list_sessions(self) -> list[sqlite3.Row]:
        return self.connection.execute(
            "SELECT * FROM sessions ORDER BY created_at, session_id"
        ).fetchall()

    def current_windows(self, session_id: str, view_id: str, include_stale: bool = False) -> list[sqlite3.Row]:
        stale_clause = "" if include_stale else " AND stale = 0"
        return self.connection.execute(
            "SELECT * FROM window_state WHERE session_id = ? AND view_id = ? AND status = 'COMMITTED'" + stale_clause + " ORDER BY partition_key, window_start_ms, window_end_ms, rule_id",
            (session_id, view_id),
        ).fetchall()

    def all_windows(self, session_id: str, view_id: str, include_stale: bool = False) -> list[sqlite3.Row]:
        stale_clause = "" if include_stale else " AND stale = 0"
        return self.connection.execute(
            "SELECT * FROM window_state WHERE session_id = ? AND view_id = ?" + stale_clause + " ORDER BY partition_key, window_start_ms, window_end_ms, rule_id",
            (session_id, view_id),
        ).fetchall()

    def current_decisions(self, session_id: str, view_id: str, include_stale: bool = False) -> list[sqlite3.Row]:
        stale_clause = "" if include_stale else " AND stale = 0"
        return self.connection.execute(
            "SELECT * FROM late_decisions WHERE session_id = ? AND view_id = ?" + stale_clause + " ORDER BY arrival_no, partition_key, rule_id, event_id",
            (session_id, view_id),
        ).fetchall()

    def current_attributions(self, session_id: str, view_id: str, include_stale: bool = False) -> list[sqlite3.Row]:
        stale_clause = "" if include_stale else " AND stale = 0"
        return self.connection.execute(
            "SELECT * FROM error_attribution WHERE session_id = ? AND view_id = ?" + stale_clause + " ORDER BY partition_key, window_start_ms, window_end_ms, rule_id",
            (session_id, view_id),
        ).fetchall()

    def current_ambiguities(self, session_id: str, view_id: str, include_stale: bool = False) -> list[sqlite3.Row]:
        stale_clause = "" if include_stale else " AND stale = 0"
        return self.connection.execute(
            "SELECT * FROM ambiguity_pairs WHERE session_id = ? AND view_id = ?" + stale_clause + " ORDER BY pair_id",
            (session_id, view_id),
        ).fetchall()

    def current_batches(self, session_id: str, view_id: str) -> list[sqlite3.Row]:
        return self.connection.execute(
            "SELECT * FROM ledger_batches WHERE session_id = ? AND view_id = ? ORDER BY batch_no",
            (session_id, view_id),
        ).fetchall()

    def watermark_traces(self, session_id: str, view_id: str) -> list[sqlite3.Row]:
        return self.connection.execute(
            "SELECT * FROM watermark_trace WHERE session_id = ? AND view_id = ? ORDER BY arrival_no, event_id",
            (session_id, view_id),
        ).fetchall()

    def final_watermarks(self, session_id: str, view_id: str) -> dict[str, int]:
        rows = self.watermark_traces(session_id, view_id)
        result: dict[str, int] = {}
        for row in rows:
            result[row["partition_key"]] = row["watermark_after_ms"]
        return result

    def staleness_events(self, session_id: str) -> list[sqlite3.Row]:
        return self.connection.execute(
            "SELECT * FROM staleness_events WHERE session_id = ? ORDER BY created_at, event_id",
            (session_id,),
        ).fetchall()

    def count_rows(self, table: str, session_id: str, view_id: str) -> int:
        if table not in {
            "watermark_trace",
            "window_state",
            "late_decisions",
            "error_attribution",
            "ambiguity_pairs",
            "ledger_batches",
        }:
            raise ValueError(table)
        row = self.connection.execute(
            f"SELECT COUNT(*) AS c FROM {table} WHERE session_id = ? AND view_id = ?",
            (session_id, view_id),
        ).fetchone()
        return int(row["c"])


def _json(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def open_readonly(path: str | Path) -> sqlite3.Connection:
    db_path = Path(path).resolve()
    connection = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def affected_window_keys(rows: Iterable[sqlite3.Row]) -> list[dict[str, Any]]:
    affected: list[dict[str, Any]] = []
    for row in rows:
        affected.append(
            {
                "rule_id": row["rule_id"],
                "rule_version": row["rule_version"],
                "partition_key": row["partition_key"],
                "window_start_ms": row["window_start_ms"],
                "window_end_ms": row["window_end_ms"],
            }
        )
    return affected
