from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .canonical import digest_obj


SUPPORTED_AGGREGATIONS = frozenset({"COUNT", "SUM", "MEAN"})


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    seed: int
    event_count: int
    out_of_order_ratio_bp: int
    partition_keys: tuple[str, ...] = ()
    base_time_ms: int = 1_700_000_000_000
    time_step_ms: int = 20
    measure_min: int = -1000
    measure_max: int = 1000
    reorder_delay_events: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "seed": self.seed,
            "event_count": self.event_count,
            "out_of_order_ratio_bp": self.out_of_order_ratio_bp,
            "partition_keys": list(self.partition_keys),
            "base_time_ms": self.base_time_ms,
            "time_step_ms": self.time_step_ms,
            "measure_min": self.measure_min,
            "measure_max": self.measure_max,
            "reorder_delay_events": self.reorder_delay_events,
        }


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    width_ms: int
    lateness_ms: int
    partition_keys: tuple[str, ...]
    aggregation_functions: tuple[str, ...]
    rule_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "width_ms": self.width_ms,
            "lateness_ms": self.lateness_ms,
            "partition_keys": list(self.partition_keys),
            "aggregation_functions": list(self.aggregation_functions),
            "rule_version": self.rule_version,
        }


@dataclass(frozen=True)
class SessionSpec:
    format_version: int
    sources: tuple[SourceSpec, ...]
    rules: tuple[RuleSpec, ...]
    fixed_scale: int
    lateness_budget_count: int
    lateness_budget_amount: int
    dominant_threshold_bp: int = 6000
    order_backward_limit_ms: int = 86_400_000

    @property
    def rules_version(self) -> int:
        return max(rule.rule_version for rule in self.rules)

    @property
    def partition_keys(self) -> tuple[str, ...]:
        return tuple(sorted({key for rule in self.rules for key in rule.partition_keys}))

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(source.source_id for source in self.sources)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "sources": [source.to_dict() for source in self.sources],
            "rules": [rule.to_dict() for rule in self.rules],
            "session": {
                "fixed_scale": self.fixed_scale,
                "lateness_budget_count": self.lateness_budget_count,
                "lateness_budget_amount": self.lateness_budget_amount,
                "dominant_threshold_bp": self.dominant_threshold_bp,
                "order_backward_limit_ms": self.order_backward_limit_ms,
            },
        }

    @property
    def spec_digest(self) -> str:
        return digest_obj(self.to_dict())

    def with_rules_version(self, version: int) -> "SessionSpec":
        return SessionSpec(
            format_version=self.format_version,
            sources=self.sources,
            rules=tuple(
                RuleSpec(
                    rule_id=rule.rule_id,
                    width_ms=rule.width_ms,
                    lateness_ms=rule.lateness_ms,
                    partition_keys=rule.partition_keys,
                    aggregation_functions=rule.aggregation_functions,
                    rule_version=version,
                )
                for rule in self.rules
            ),
            fixed_scale=self.fixed_scale,
            lateness_budget_count=self.lateness_budget_count,
            lateness_budget_amount=self.lateness_budget_amount,
            dominant_threshold_bp=self.dominant_threshold_bp,
            order_backward_limit_ms=self.order_backward_limit_ms,
        )


@dataclass(frozen=True)
class Event:
    event_id: str
    source_id: str
    source_seq: int
    arrival_no: int
    partition_key: str
    event_time_ms: int
    measure: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "source_id": self.source_id,
            "source_seq": self.source_seq,
            "arrival_no": self.arrival_no,
            "partition_key": self.partition_key,
            "event_time_ms": self.event_time_ms,
            "measure": self.measure,
        }


@dataclass(frozen=True)
class WindowKey:
    rule_id: str
    rule_version: int
    partition_key: str
    window_start_ms: int
    window_end_ms: int

    def sort_key(self) -> tuple[Any, ...]:
        return (
            self.partition_key,
            self.window_start_ms,
            self.window_end_ms,
            self.rule_id,
            self.rule_version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "rule_version": self.rule_version,
            "partition_key": self.partition_key,
            "window_start_ms": self.window_start_ms,
            "window_end_ms": self.window_end_ms,
        }


@dataclass
class WindowAccumulator:
    key: WindowKey
    count: int = 0
    sum_exact: int = 0
    quantization_error_sum: int = 0
    max_single_quantization_error: int = 0
    late_accepted_count: int = 0
    late_accepted_abs_amount: int = 0
    boundary_abs_amount: int = 0
    accepted_late_events: list[int] = field(default_factory=list)
    boundary_events: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class ErrorAttribution:
    quantization_truncation: int
    late_admission: int
    window_boundary: int
    sum_bound: int
    mean_bound: int
    single_value_quantization: int
    dominant_segment: str
    dominant_share_bp: int
    threshold_bp: int
    quantization_share_bp: int
    late_share_bp: int
    boundary_share_bp: int
    formula_version: str = "fixed-i64-v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "quantization_truncation": self.quantization_truncation,
            "late_admission": self.late_admission,
            "window_boundary": self.window_boundary,
            "sum_bound": self.sum_bound,
            "mean_bound": self.mean_bound,
            "single_value_quantization": self.single_value_quantization,
            "dominant_segment": self.dominant_segment,
            "dominant_share_bp": self.dominant_share_bp,
            "threshold_bp": self.threshold_bp,
            "quantization_share_bp": self.quantization_share_bp,
            "late_share_bp": self.late_share_bp,
            "boundary_share_bp": self.boundary_share_bp,
            "formula_version": self.formula_version,
        }


@dataclass(frozen=True)
class WindowResult:
    key: WindowKey
    count: int
    sum_fixed: int
    mean_fixed: int
    attribution: ErrorAttribution
    sum_exact: int = 0
    late_accepted_count: int = 0
    late_accepted_abs_amount: int = 0
    boundary_abs_amount: int = 0
    sum_saturation_error: int = 0
    mean_saturation_error: int = 0
    status: str = "COMMITTED"

    def to_dict(self) -> dict[str, Any]:
        payload = {
            **self.key.to_dict(),
            "count": self.count,
            "sum_fixed": self.sum_fixed,
            "mean_fixed": self.mean_fixed,
            "sum_exact": self.sum_exact,
            "late_accepted_count": self.late_accepted_count,
            "late_accepted_abs_amount": self.late_accepted_abs_amount,
            "boundary_abs_amount": self.boundary_abs_amount,
            "sum_saturation_error": self.sum_saturation_error,
            "mean_saturation_error": self.mean_saturation_error,
            "status": self.status,
            "attribution": self.attribution.to_dict(),
        }
        return payload


@dataclass(frozen=True)
class LateDecision:
    decision_id: str
    rule_id: str
    rule_version: int
    partition_key: str
    event_id: str
    arrival_no: int
    event_time_ms: int
    window_start_ms: int
    window_end_ms: int
    decision: str
    reason: str
    fixed_value: int
    budget_count_before: int
    budget_count_after: int
    budget_amount_before: int
    budget_amount_after: int
    cumulative_late_abs_amount: int
    was_late: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "rule_id": self.rule_id,
            "rule_version": self.rule_version,
            "partition_key": self.partition_key,
            "event_id": self.event_id,
            "arrival_no": self.arrival_no,
            "event_time_ms": self.event_time_ms,
            "window_start_ms": self.window_start_ms,
            "window_end_ms": self.window_end_ms,
            "decision": self.decision,
            "reason": self.reason,
            "fixed_value": self.fixed_value,
            "budget_count_before": self.budget_count_before,
            "budget_count_after": self.budget_count_after,
            "budget_amount_before": self.budget_amount_before,
            "budget_amount_after": self.budget_amount_after,
            "cumulative_late_abs_amount": self.cumulative_late_abs_amount,
            "was_late": self.was_late,
        }


@dataclass(frozen=True)
class WatermarkTrace:
    trace_id: str
    event_id: str
    arrival_no: int
    partition_key: str
    event_time_ms: int
    watermark_before_ms: int
    watermark_after_ms: int
    was_late: bool
    finalized_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "event_id": self.event_id,
            "arrival_no": self.arrival_no,
            "partition_key": self.partition_key,
            "event_time_ms": self.event_time_ms,
            "watermark_before_ms": self.watermark_before_ms,
            "watermark_after_ms": self.watermark_after_ms,
            "was_late": self.was_late,
            "finalized_count": self.finalized_count,
        }
