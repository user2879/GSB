"""Post-restore replay verification and first-divergence localization."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from .clock import LogicalClock
from .engine import CallDecision, DiagnosticEngine
from .errors import replay_error
from .models import ChainSpec, TraceStream
from .storage import SnapshotRecord


@dataclass(frozen=True)
class ReplayReport:
    snapshot_id: str | None
    start_cursor: int
    compared_calls: int
    final_cursor: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": True,
            "snapshot_id": self.snapshot_id,
            "start_cursor": self.start_cursor,
            "compared_calls": self.compared_calls,
            "final_cursor": self.final_cursor,
        }


def _classify_incompatibility(
    snapshot: SnapshotRecord, chain: ChainSpec, trace: TraceStream
) -> str | None:
    if snapshot.chain_version != chain.version:
        return "chain_version_changed"
    if snapshot.chain_digest != chain.digest:
        old_budget = {
            service.get("id"): service.get("budget", {})
            for service in snapshot.chain.get("services", [])
        }
        if old_budget != chain.budget_signature():
            return "window_parameters_changed"
        return "chain_content_changed"
    if snapshot.trace_digest != trace.content_digest:
        return "trace_content_changed"
    return None


def _first_divergence(
    expected: CallDecision,
    actual: CallDecision,
) -> dict[str, Any] | None:
    call_fields = (
        "trigger_source",
        "trigger_reason",
        "fast_failed",
        "fast_fail_chain",
    )
    expected_call = expected.as_dict()
    actual_call = actual.as_dict()
    for field in call_fields:
        if expected_call[field] != actual_call[field]:
            return {
                "call_index": expected.call_index,
                "service": "<call>",
                "field": field,
                "expected": expected_call[field],
                "actual": actual_call[field],
            }

    expected_services = {item.service: item for item in expected.services}
    actual_services = {item.service: item for item in actual.services}
    for service_id in sorted(set(expected_services) | set(actual_services)):
        left = expected_services.get(service_id)
        right = actual_services.get(service_id)
        if left is None or right is None:
            return {
                "call_index": expected.call_index,
                "service": service_id,
                "field": "service_result_presence",
                "expected": left is not None,
                "actual": right is not None,
            }
        left_dict = left.as_dict()
        right_dict = right.as_dict()
        for field in (
            "status",
            "consumed",
            "fast_failed",
            "trigger_source",
            "trigger_reason",
            "propagation_depth",
        ):
            if left_dict[field] != right_dict[field]:
                return {
                    "call_index": expected.call_index,
                    "service": service_id,
                    "field": field,
                    "expected": left_dict[field],
                    "actual": right_dict[field],
                }
        for field in (
            "window_samples",
            "failures",
            "budget_total",
            "raw_remaining",
            "remaining",
            "exhausted",
        ):
            if left_dict["budget"][field] != right_dict["budget"][field]:
                return {
                    "call_index": expected.call_index,
                    "service": service_id,
                    "field": f"budget.{field}",
                    "expected": left_dict["budget"][field],
                    "actual": right_dict["budget"][field],
                }
        for field in (
            "state",
            "consecutive_failures",
            "cooldown_remaining_ms",
            "open_count",
        ):
            if left_dict["circuit"][field] != right_dict["circuit"][field]:
                return {
                    "call_index": expected.call_index,
                    "service": service_id,
                    "field": f"circuit.{field}",
                    "expected": left_dict["circuit"][field],
                    "actual": right_dict["circuit"][field],
                }
    return None


def _service_row_divergence(
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
    call_index: int,
) -> dict[str, Any] | None:
    left = {row["service"]: row for row in expected}
    right = {row["service"]: row for row in actual}
    for service_id in sorted(set(left) | set(right)):
        if service_id not in left or service_id not in right:
            return {
                "call_index": call_index,
                "service": service_id,
                "field": "service_presence",
                "expected": service_id in left,
                "actual": service_id in right,
            }
        for field in (
            "window_samples",
            "failures",
            "budget_total",
            "budget_remaining",
            "budget_exhausted",
            "consecutive_failures",
            "circuit_state",
            "cooldown_remaining_ms",
            "open_count",
        ):
            if left[service_id][field] != right[service_id][field]:
                return {
                    "call_index": call_index,
                    "service": service_id,
                    "field": field,
                    "expected": left[service_id][field],
                    "actual": right[service_id][field],
                }
    return None


class ReplayVerifier:
    def __init__(
        self,
        chain: ChainSpec,
        trace: TraceStream,
        snapshot: SnapshotRecord,
    ) -> None:
        self.chain = chain
        self.trace = trace
        self.snapshot = snapshot

    def run(self) -> ReplayReport:
        reason = _classify_incompatibility(self.snapshot, self.chain, self.trace)
        if reason is not None:
            raise replay_error(
                "快照与当前服务链或调用流不可复现",
                kind="incompatible",
                basis=reason,
                snapshot_id=self.snapshot.snapshot_id,
                snapshot_chain_version=self.snapshot.chain_version,
                current_chain_version=self.chain.version,
                snapshot_chain_digest=self.snapshot.chain_digest,
                current_chain_digest=self.chain.digest,
                snapshot_trace_digest=self.snapshot.trace_digest,
                current_trace_digest=self.trace.content_digest,
            )

        snapshot_cursor = self.snapshot.cursor
        if snapshot_cursor < 0 or snapshot_cursor > len(self.trace):
            raise replay_error(
                "快照游标超出调用流范围",
                kind="incompatible",
                basis="snapshot_cursor_out_of_range",
                snapshot_id=self.snapshot.snapshot_id,
                cursor=snapshot_cursor,
                total_calls=len(self.trace),
            )

        baseline_clock = LogicalClock(0)
        baseline = DiagnosticEngine(self.chain, baseline_clock)
        for call in self.trace.calls[:snapshot_cursor]:
            baseline_clock.follow(call.timestamp_ms)
            baseline.process_call(call)

        replay_clock = LogicalClock(self.snapshot.clock_ms)
        restored = DiagnosticEngine(self.chain, replay_clock)
        restored.restore_state(copy.deepcopy(self.snapshot.state))
        if restored.cursor != snapshot_cursor:
            raise replay_error(
                "快照内容游标不一致",
                kind="incompatible",
                basis="snapshot_cursor_internal_mismatch",
                snapshot_id=self.snapshot.snapshot_id,
                expected=snapshot_cursor,
                actual=restored.cursor,
            )

        compared = 0
        for call in self.trace.calls[snapshot_cursor:]:
            baseline_clock.follow(call.timestamp_ms)
            replay_clock.follow(call.timestamp_ms)
            expected_decision = baseline.process_call(call)
            actual_decision = restored.process_call(call)
            divergence = _first_divergence(expected_decision, actual_decision)
            if divergence is None:
                divergence = _service_row_divergence(
                    baseline.service_rows(baseline_clock.now_ms()),
                    restored.service_rows(replay_clock.now_ms()),
                    call.index,
                )
            if divergence is not None:
                raise replay_error(
                    "恢复后重放与不中断推进出现差异",
                    kind="divergence",
                    basis="first_differing_call_or_state_field",
                    snapshot_id=self.snapshot.snapshot_id,
                    **divergence,
                )
            compared += 1

        return ReplayReport(
            snapshot_id=self.snapshot.snapshot_id,
            start_cursor=snapshot_cursor,
            compared_calls=compared,
            final_cursor=restored.cursor,
        )
