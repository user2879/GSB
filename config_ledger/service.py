"""Application service joining validated specs, observation pulls and the ledger."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .errors import RecoveryInfo, ReplayError, SnapshotError, SpecValidationError
from .latency import (
    LatencySample,
    aggregate_stats,
    build_fetch_aggregates,
    evaluate_budget,
)
from .ledger import ActivationResult, CommitResult, Ledger
from .merge import MergeResult, canonical_json, merge_observations
from .observer import ObservationFetcher, monotonic_ms
from .spec import Spec, load_spec, parse_spec_data
from .structured_log import StructuredLogger, utc_now_text


DECISION_FIELD_ORDER = (
    "status",
    "quorum_met",
    "divergent",
    "valid_node_count",
    "total_nodes",
    "expected",
    "quorum_bp",
    "arbitration_rule",
    "candidate",
    "arbitrated_value",
    "effective_value",
    "winner_basis",
    "majority_count",
    "expected_matches",
    "attributions",
    "attribution_details",
    "basis_text",
)
COMPARISON_FIELD_ORDER = (
    "request_status",
    "http_status",
    "error_reason",
    "value",
    "observation_seq",
    "timestamp_ms",
    "duplicate_count",
    "adopted",
    "reason",
    "latest_candidate",
)


class DiagnosticService:
    """Owns the current validated parameter set and its local ledger."""

    def __init__(
        self,
        spec_path: str | Path,
        *,
        ledger_path: str | Path,
        log_path: str | Path | None = None,
    ):
        self.logger = StructuredLogger(log_path)
        self.spec = self._load_or_log(spec_path)
        self.ledger = Ledger(ledger_path, logger=self.logger)
        self._pending_previous: tuple[int, str | None] | None = None
        try:
            self.snapshot = self.ledger.verify_snapshot()
            self.recovery: RecoveryInfo = self.ledger.recover_incomplete_attempts()
            activation = self.ledger.activate_parameter(self.spec)
            self.spec = self.spec.with_parameter_version(activation.current_version)
            if activation.changed:
                self._pending_previous = (
                    activation.previous_version,
                    activation.previous_round_id,
                )
            self.activation = activation
        except Exception:
            self.ledger.close()
            raise

    def _load_or_log(self, spec_path: str | Path) -> Spec:
        try:
            return load_spec(spec_path)
        except SpecValidationError as exc:
            self.logger.error(
                "E_SPEC",
                str(exc),
                source=str(spec_path),
                issues=[issue.as_dict() for issue in exc.issues],
            )
            raise

    def close(self) -> None:
        self.ledger.close()

    def __enter__(self) -> "DiagnosticService":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def reload_spec(self, spec_path: str | Path) -> ActivationResult:
        """Validate fully before replacing in-memory/current-ledger parameters."""

        new_spec = self._load_or_log(spec_path)
        activation = self.ledger.activate_parameter(new_spec)
        # Only after the complete DB transaction succeeds does the in-memory
        # declaration change. A failed activation leaves the old spec visible.
        self.spec = new_spec.with_parameter_version(activation.current_version)
        if activation.changed:
            self._pending_previous = (
                activation.previous_version,
                activation.previous_round_id,
            )
        self.activation = activation
        return activation

    def run_round(self) -> dict[str, Any]:
        """Pull every configured node/key and commit one complete round."""

        window_id = self.spec.observation_window
        attempt_id = self.ledger.begin_attempt(
            parameter_version=self.spec.parameter_version, window_id=window_id
        )
        try:
            fetcher = ObservationFetcher(self.logger)
            batch = fetcher.fetch_all(
                self.spec,
                window_id=window_id,
                parameter_version=self.spec.parameter_version,
            )
            merge_result = merge_observations(self.spec, batch)
            if self._pending_previous is not None:
                previous_version, previous_round_id = self._pending_previous
            else:
                meta = self.ledger.current_meta()
                previous_version = meta["current_parameter_version"]
                previous_round_id = meta["current_round_id"]
                if previous_round_id is None:
                    previous_version, previous_round_id = (
                        self.ledger.last_committed_context()
                    )
            commit = self.ledger.commit_round(
                spec=self.spec,
                attempt_id=attempt_id,
                window_id=window_id,
                batch=batch,
                merge_result=merge_result,
                previous_version=previous_version,
                previous_round_id=previous_round_id,
            )
            self._pending_previous = None
            dashboard = self.ledger.load_dashboard()
            return {
                "status": "COMMITTED",
                "round_id": commit.round_id,
                "content_digest": commit.content_digest,
                "parameter_version": commit.parameter_version,
                "window_id": window_id,
                "attempt_id": attempt_id,
                "budget_summary": {
                    "global_concurrency": batch.global_limit,
                    "peak_global": batch.peak_global,
                    "node_quotas": {
                        node: batch.node_quotas[node] for node in sorted(batch.node_quotas)
                    },
                    "peak_by_node": {
                        node: batch.peak_by_node[node] for node in sorted(batch.peak_by_node)
                    },
                },
                "observations": [outcome.as_dict() for outcome in batch.ordered()],
                "conclusions": list(merge_result.decisions),
                "divergence_keys": list(merge_result.divergence_keys),
                "divergence_comparisons": [
                    row
                    for row in merge_result.comparisons
                    if row["key"] in set(merge_result.divergence_keys)
                ],
                "merge_digest": merge_result.deterministic_digest(),
                "fetch_latency": dashboard["fetch_latency"],
                "ledger_meta": dashboard["meta"],
            }
        except Exception as exc:
            try:
                self.ledger.mark_attempt_failed(attempt_id, f"{type(exc).__name__}: {exc}")
            except Exception:
                pass
            raise

    def query_dashboard(self) -> dict[str, Any]:
        """Return the whole current view and record this query's latency."""

        start_ns = time.perf_counter_ns()
        start_ms = monotonic_ms()
        dashboard = self.ledger.load_dashboard()
        # Stable presentation order is part of the contract.
        dashboard["conclusions"] = sorted(
            dashboard["conclusions"], key=lambda row: row["key"]
        )
        dashboard["comparisons"] = sorted(
            dashboard["comparisons"], key=lambda row: (row["key"], row["node_id"])
        )
        dashboard["fetch_latency"] = sorted(
            dashboard["fetch_latency"],
            key=lambda row: (row["scope"], row.get("node_id") or ""),
        )
        end_ns = time.perf_counter_ns()
        duration_us = max(0, (end_ns - start_ns) // 1000)
        end_ms = monotonic_ms()
        over_budget, overrun_us = evaluate_budget(
            duration_us, self.spec.ui_latency_budget_ms
        )
        round_id = dashboard["meta"]["current_round_id"]
        self.ledger.record_ui_latency(
            parameter_version=self.spec.parameter_version,
            round_id=round_id,
            duration_us=duration_us,
            monotonic_start_ms=start_ms,
            monotonic_end_ms=end_ms,
            over_budget=over_budget,
            overrun_us=overrun_us,
        )
        if over_budget:
            self.logger.error(
                "E_LATBUDGET",
                "UI query exceeded the declared latency budget",
                duration_us=duration_us,
                duration_fixed6_ms=duration_us * 1000,
                budget_ms=self.spec.ui_latency_budget_ms,
                overrun_us=overrun_us,
                overrun_fixed6_ms=overrun_us * 1000,
                parameter_version=self.spec.parameter_version,
                round_id=round_id,
            )
        dashboard["ui_latency"] = self.ledger.ui_latency_summary(
            parameter_version=self.spec.parameter_version, round_id=round_id
        )
        dashboard["ui_latency"]["budget_ms"] = self.spec.ui_latency_budget_ms
        dashboard["ui_latency"]["last_over_budget"] = over_budget
        dashboard["ui_latency"]["last_overrun_us"] = overrun_us
        dashboard["ui_latency"]["all_samples_retained"] = True
        dashboard["ui_latency"]["true_source"] = "ui_latency_samples rows in the SQLite ledger"
        dashboard["query_latency"] = {
            "duration_us": duration_us,
            "duration_fixed6_ms": duration_us * 1000,
            "monotonic_start_ms": start_ms,
            "monotonic_end_ms": end_ms,
            "budget_ms": self.spec.ui_latency_budget_ms,
            "over_budget": over_budget,
            "overrun_us": overrun_us,
            "overrun_fixed6_ms": overrun_us * 1000,
        }
        dashboard["parameter_version"] = self.spec.parameter_version
        dashboard["parameter_fingerprint"] = self.spec.fingerprint
        dashboard["spec"] = self.spec.to_dict()
        return dashboard

    def replay(self) -> dict[str, Any]:
        """Re-fetch the recorded window and compare deterministic round content."""

        bundle = self.ledger.load_current_bundle()
        if bundle is None or bundle.get("spec") is None:
            raise ReplayError("there is no committed current round to replay")
        replay_spec = parse_spec_data(bundle["spec"], source="<ledger-spec>")
        started_at = utc_now_text()
        fetcher = ObservationFetcher(self.logger)
        batch = fetcher.fetch_all(
            replay_spec,
            window_id=bundle["window_id"],
            parameter_version=bundle["parameter_version"],
        )
        replay_merge = merge_observations(replay_spec, batch)
        ledger_merge = MergeResult(
            decisions=tuple(bundle["decisions"]),
            comparisons=tuple(bundle["comparisons"]),
            divergence_keys=tuple(
                sorted(
                    decision["key"]
                    for decision in bundle["decisions"]
                    if decision.get("divergent")
                )
            ),
        )
        first_diff = self._find_first_diff(
            ledger_merge=ledger_merge,
            replay_merge=replay_merge,
            ledger_latency=bundle["latency_aggregates"],
            latency_samples=bundle["latency_samples"],
            spec=replay_spec,
        )
        finished_at = utc_now_text()
        if first_diff is not None:
            self.ledger.record_replay_run(
                round_id=bundle["round_id"],
                parameter_version=bundle["parameter_version"],
                window_id=bundle["window_id"],
                status="MISMATCH",
                first_diff=first_diff,
                started_at=started_at,
                finished_at=finished_at,
            )
            self.logger.error(
                "E_REPLAY",
                "replay differs from the committed ledger; first mismatch is reported",
                round_id=bundle["round_id"],
                parameter_version=bundle["parameter_version"],
                window_id=bundle["window_id"],
                first_diff=first_diff,
            )
            raise ReplayError(
                "replay mismatch at the first reported key/node/field",
                first_diff=first_diff,
            )
        self.ledger.record_replay_run(
            round_id=bundle["round_id"],
            parameter_version=bundle["parameter_version"],
            window_id=bundle["window_id"],
            status="MATCH",
            first_diff=None,
            started_at=started_at,
            finished_at=finished_at,
        )
        live_samples = [
            LatencySample(
                kind="fetch",
                node_id=outcome.node_id,
                label=outcome.key,
                duration_us=outcome.request_latency_us,
                monotonic_start_ms=outcome.request_start_ms,
                monotonic_end_ms=outcome.request_end_ms,
            )
            for outcome in batch.ordered()
        ]
        self.logger.info(
            "REPLAY_OK",
            "committed decisions, divergence comparisons and ledger latency stats match",
            round_id=bundle["round_id"],
            parameter_version=bundle["parameter_version"],
            window_id=bundle["window_id"],
        )
        return {
            "status": "MATCH",
            "round_id": bundle["round_id"],
            "parameter_version": bundle["parameter_version"],
            "window_id": bundle["window_id"],
            "merge_digest": replay_merge.deterministic_digest(),
            "decision_count": len(replay_merge.decisions),
            "divergence_keys": list(replay_merge.divergence_keys),
            "latency_comparison_source": "ledger samples recomputed against ledger aggregates",
            "ledger_fetch_latency": bundle["latency_aggregates"],
            "live_fetch_latency": build_fetch_aggregates(
                live_samples, replay_spec.sorted_node_ids
            ),
        }

    def _find_first_diff(
        self,
        *,
        ledger_merge: MergeResult,
        replay_merge: MergeResult,
        ledger_latency: list[dict[str, Any]],
        latency_samples: list[dict[str, Any]],
        spec: Spec,
    ) -> dict[str, Any] | None:
        ledger_decisions = ledger_merge.decisions_by_key()
        replay_decisions = replay_merge.decisions_by_key()
        for key in sorted(set(ledger_decisions) | set(replay_decisions)):
            old = ledger_decisions.get(key)
            new = replay_decisions.get(key)
            if old is None or new is None:
                return self._diff(
                    key,
                    "*",
                    "presence",
                    old is not None,
                    new is not None,
                )
            for field in DECISION_FIELD_ORDER:
                old_value = old.get(field)
                new_value = new.get(field)
                if canonical_json(old_value) != canonical_json(new_value):
                    return self._diff(key, "*", field, old_value, new_value)

        ledger_comparisons = ledger_merge.comparisons_by_key_node()
        replay_comparisons = replay_merge.comparisons_by_key_node()
        for key, node_id in sorted(set(ledger_comparisons) | set(replay_comparisons)):
            old = ledger_comparisons.get((key, node_id))
            new = replay_comparisons.get((key, node_id))
            if old is None or new is None:
                return self._diff(
                    key,
                    node_id,
                    "presence",
                    old is not None,
                    new is not None,
                )
            for field in COMPARISON_FIELD_ORDER:
                old_value = old.get(field)
                new_value = new.get(field)
                if canonical_json(old_value) != canonical_json(new_value):
                    return self._diff(key, node_id, field, old_value, new_value)

        samples = [
            LatencySample(
                kind=row["kind"],
                node_id=row["node_id"],
                label=row["label"],
                duration_us=int(row["duration_us"]),
                monotonic_start_ms=int(row["monotonic_start_ms"]),
                monotonic_end_ms=int(row["monotonic_end_ms"]),
                over_budget=bool(row["over_budget"]),
                overrun_us=int(row["overrun_us"]),
            )
            for row in latency_samples
        ]
        recomputed_ledger = build_fetch_aggregates(samples, spec.sorted_node_ids)
        ledger_aggregates = {
            (row.get("scope"), row.get("node_id")): row for row in ledger_latency
        }
        recomputed = {
            (row.get("scope"), row.get("node_id")): row for row in recomputed_ledger
        }
        for scope, node_id in sorted(
            set(ledger_aggregates) | set(recomputed), key=lambda item: (item[0] or "", item[1] or "")
        ):
            old = ledger_aggregates.get((scope, node_id))
            new = recomputed.get((scope, node_id))
            if old is None or new is None:
                return self._diff(
                    "<latency>",
                    node_id or "*",
                    f"scope:{scope or '*'}",
                    old,
                    new,
                )
            for field in (
                "sample_count",
                "mean_us",
                "p95_us",
                "max_us",
                "mean_fixed6_ms",
                "p95_fixed6_ms",
                "max_fixed6_ms",
                "quantization_error_bound_us",
                "quantization_error_bound_fixed6_ms",
                "quantization_error_bound_us_halves",
                "p95_method",
                "true_source",
            ):
                if old.get(field) != new.get(field):
                    return self._diff(
                        "<latency>",
                        node_id or "*",
                        field,
                        old.get(field),
                        new.get(field),
                    )
        return None

    @staticmethod
    def _diff(
        key: str, node_id: str, field: str, ledger_value: Any, replay_value: Any
    ) -> dict[str, Any]:
        return {
            "key": key,
            "node_id": node_id,
            "field": field,
            "ledger": ledger_value,
            "replay": replay_value,
            "basis": "stable traversal: sorted keys, key-level fields, sorted node fields, ledger latency aggregates",
        }
