"""Top-level orchestration: versions, pipeline commits, replay and recovery."""

from __future__ import annotations

from dataclasses import dataclass

from .allocator import allocate_minute, assert_allocations_equal
from .errors import ReplayError
from .estimator import (
    FullEstimator,
    IncrementalEstimator,
    assert_estimates_equal,
    estimates_digest,
    run_both_paths,
    select_path,
)
from .latency import LatencyRecorder
from .ledger import Ledger
from .loader import check_series_lengths
from .logging_utils import JsonlLogger
from .models import ClusterSpec, MinuteDecision
from .preview import PreviewCache

_COMPARE_ESTIMATE_FIELDS = (
    "requests",
    "per_replica_capacity",
    "raw_demand",
    "demand",
    "rule_version",
    "scale_action",
    "utilization_bp",
)
_COMPARE_ALLOCATION_FIELDS = (
    "allocated",
    "rejected",
    "trim_reason",
    "trimmed",
    "effective_cap",
    "capacity_requests",
)


@dataclass
class RunReport:
    version: int
    minutes: int
    selection: str
    selection_reason: str
    threshold_boundary: int
    committed: int
    unknown_minutes: list[int]


class ScalingEngine:
    def __init__(self, ledger: Ledger, logger: JsonlLogger,
                 recorder: LatencyRecorder | None = None,
                 preview: PreviewCache | None = None) -> None:
        self.ledger = ledger
        self.logger = logger
        self.recorder = recorder or LatencyRecorder(budget_ms=50, fixed_point_scale=1000,
                                                    logger=logger)
        self._sync_recorder_seq(1)
        self.preview = preview or PreviewCache()

    def _sync_recorder_seq(self, version: int) -> None:
        if self.recorder is not None:
            self.recorder._seq = max(
                self.recorder._seq, self.ledger.next_latency_seq(version) - 1
            )

    def register_declaration(self, spec: ClusterSpec, *, reason: str = "load") -> int:
        check_series_lengths(spec)
        version = self.ledger.latest_version() + 1
        digest = self.ledger.register_version(version, spec.declaration())
        self.logger.info(
            "version_registered",
            version=version,
            declaration_digest=digest,
            reason=reason,
            service_count=len(spec.services),
            minutes=spec.minutes,
        )
        return version

    def _build_minute_decision(
        self,
        spec: ClusterSpec,
        version: int,
        minute: int,
        full_rows,
        incremental_rows,
        selection: str,
        selection_reason: str,
        boundary: int,
    ) -> MinuteDecision:
        assert_estimates_equal(
            (full_rows[minute],), (incremental_rows[minute],)
        )
        full_allocations = allocate_minute(spec, minute, full_rows[minute])
        incremental_allocations = allocate_minute(spec, minute, incremental_rows[minute])
        assert_allocations_equal(full_allocations, incremental_allocations, minute)

        chosen_estimates = incremental_rows[minute] if selection == "incremental" else full_rows[minute]
        chosen_allocations = (
            incremental_allocations if selection == "incremental" else full_allocations
        )
        return MinuteDecision(
            minute=minute,
            version=version,
            estimates=chosen_estimates,
            allocations=chosen_allocations,
            total_allocated=sum(item.allocated for item in chosen_allocations),
            global_budget=spec.global_replica_budget,
            selection=selection,
            selection_reason=selection_reason,
            threshold_boundary=boundary,
            estimates_digest=estimates_digest((chosen_estimates,)),
        )

    def run_pipeline(self, spec: ClusterSpec, version: int | None = None) -> RunReport:
        check_series_lengths(spec)
        version = version or self.ledger.active_version() or self.register_declaration(spec)
        selection, selection_reason, boundary = select_path(spec)
        full_rows, incremental_rows, _ = run_both_paths(spec)

        unknown = self.ledger.unknown_minutes(version, spec.minutes)
        self._sync_recorder_seq(version)
        for minute in range(spec.minutes):
            decision = self._build_minute_decision(
                spec, version, minute, full_rows, incremental_rows,
                selection, selection_reason, boundary,
            )
            if minute not in unknown:
                # Recovery path: never overwrite a fully committed minute.
                self.preview.populate(decision)
                continue
            digest = self.ledger.commit_minute(decision)
            object.__setattr__(decision, "digest", digest)
            self.preview.populate(decision)
            if self.recorder and minute in unknown:
                start, end = self.recorder.measure("preview", "ALL")
                self.preview.query(version, minute)
                sample = self.recorder.commit(start, end, "preview", "ALL")
                self.ledger.insert_latency_sample(version, sample.as_dict())
            self.logger.info(
                "minute_committed",
                version=version,
                minute=minute,
                total_allocated=decision.total_allocated,
                selection=selection,
            )
        self.ledger.complete_version(version)
        return RunReport(
            version=version,
            minutes=spec.minutes,
            selection=selection,
            selection_reason=selection_reason,
            threshold_boundary=boundary,
            committed=spec.minutes,
            unknown_minutes=unknown,
        )

    def authoritative_recompute(self, spec: ClusterSpec, version: int) -> list[MinuteDecision]:
        """Ledger-driven replay of estimate + trim for every committed minute."""

        check_series_lengths(spec)
        self._sync_recorder_seq(version)
        decisions: list[MinuteDecision] = []

        def replay_work() -> list[MinuteDecision]:
            full_rows = FullEstimator().run(spec)
            incremental = IncrementalEstimator(spec.window_size, spec.fixed_point_scale)
            incremental_rows = incremental.run(spec)
            assert_estimates_equal(full_rows, incremental_rows)
            for minute in range(spec.minutes):
                allocations = allocate_minute(spec, minute, full_rows[minute])
                decision = self._build_minute_decision(
                    spec, version, minute, full_rows, incremental_rows,
                    *select_path(spec),
                )
                decisions.append(decision)
                assert_allocations_equal(allocations, decision.allocations, minute)
            return decisions

        start, _ = self.recorder.measure("replay", "ALL")
        result = replay_work()
        sample = self.recorder.commit(start, self.recorder._clock(), "replay", "ALL")
        self.ledger.insert_latency_sample(version, sample.as_dict())
        return result

    def verify_replay(self, spec: ClusterSpec, version: int) -> None:
        """Compare recomputed rows against the committed ledger (E_REPLAY)."""

        check_series_lengths(spec)
        committed = {item.minute: item for item in self.ledger.load_decisions(version)}
        full_rows = FullEstimator().run(spec)
        for minute in sorted(committed):
            stored = committed[minute]
            recomputed_allocations = allocate_minute(spec, minute, full_rows[minute])
            recomputed_map = {item.service_id: item for item in recomputed_allocations}
            stored_estimate_map = {item.service_id: item for item in stored.estimates}
            for estimate in full_rows[minute]:
                stored_estimate = stored_estimate_map.get(estimate.service_id)
                if stored_estimate is None:
                    raise ReplayError(
                        "replay missing service estimate",
                        details={"minute": minute, "service_id": estimate.service_id,
                                 "field": "service_id"},
                    )
                for field_name in _COMPARE_ESTIMATE_FIELDS:
                    if getattr(estimate, field_name) != getattr(stored_estimate, field_name):
                        raise ReplayError(
                            "replay estimate mismatch",
                            details={
                                "minute": minute,
                                "service_id": estimate.service_id,
                                "field": field_name,
                                "recomputed": getattr(estimate, field_name),
                                "ledger": getattr(stored_estimate, field_name),
                                "basis": "declaration_version_load_series",
                            },
                        )
            stored_map = {item.service_id: item for item in stored.allocations}
            for allocation in recomputed_allocations:
                stored_allocation = stored_map.get(allocation.service_id)
                if stored_allocation is None:
                    raise ReplayError(
                        "replay missing service allocation",
                        details={"minute": minute, "service_id": allocation.service_id,
                                 "field": "service_id"},
                    )
                for field_name in _COMPARE_ALLOCATION_FIELDS:
                    if getattr(allocation, field_name) != getattr(stored_allocation, field_name):
                        raise ReplayError(
                            "replay allocation mismatch",
                            details={
                                "minute": minute,
                                "service_id": allocation.service_id,
                                "field": field_name,
                                "recomputed": getattr(allocation, field_name),
                                "ledger": getattr(stored_allocation, field_name),
                                "basis": "ledger_committed_payload",
                            },
                        )
        self.logger.info("replay_verified", version=version, minutes=len(committed))

    def recovery_state(self, spec: ClusterSpec, version: int) -> dict:
        """Only fully committed minutes are recoverable; the rest are unknown."""

        check_series_lengths(spec)
        progress = self.ledger.progress(version)
        committed = 0 if progress is None else progress["committed_minutes"]
        unknown = self.ledger.unknown_minutes(version, spec.minutes)
        state = {
            "version": version,
            "committed_minutes": committed,
            "total_minutes": spec.minutes,
            "unknown_minutes": unknown,
            "status": (progress or {}).get("status", "missing"),
            "action": "resume_from_last_commit_then_rerun_unknown" if unknown else "complete",
        }
        self.logger.info("recovery_inspected", **state)
        return state
