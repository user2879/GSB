"""Deterministic algorithm selection from scale metrics and work estimates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .model import Spec, SpecIndex

PER_QUERY = "per_query"
MATERIALIZED = "materialized"
PATH_LABELS = {
    PER_QUERY: "逐查询沿链回溯",
    MATERIALIZED: "按读点批量物化",
}


@dataclass(frozen=True)
class PlannerThresholds:
    query_count: int = 32
    row_key_count: int = 12
    avg_chain_length: float = 5.0
    tie_band_ratio: float = 0.10
    tie_break_rule: str = "threshold-majority-v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_count": self.query_count,
            "row_key_count": self.row_key_count,
            "avg_chain_length": self.avg_chain_length,
            "tie_band_ratio": self.tie_band_ratio,
            "tie_break_rule": self.tie_break_rule,
        }


@dataclass(frozen=True)
class AlgorithmPlan:
    selected_path: str
    selected_label: str
    reason: str
    reason_codes: tuple[str, ...]
    threshold_influenced: bool
    threshold_crossed_count: int
    default_path: str
    metrics: dict[str, Any]
    thresholds: dict[str, Any]
    boundary_values: dict[str, Any]
    estimates: dict[str, Any]
    tie_band: dict[str, Any]
    forced: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_path": self.selected_path,
            "selected_label": self.selected_label,
            "reason": self.reason,
            "reason_codes": list(self.reason_codes),
            "threshold_influenced": self.threshold_influenced,
            "threshold_crossed_count": self.threshold_crossed_count,
            "default_path": self.default_path,
            "metrics": self.metrics,
            "thresholds": self.thresholds,
            "boundary_values": self.boundary_values,
            "estimates": self.estimates,
            "tie_band": self.tie_band,
            "forced": self.forced,
        }


def _threshold_boundaries(actual: float, threshold: float, *, integer: bool) -> dict[str, Any]:
    triggered = actual >= threshold
    if integer:
        boundary: dict[str, Any] = {
            "inclusive_min": int(threshold),
            "max_untriggered": int(threshold) - 1,
        }
        distance = int(actual) - int(threshold)
    else:
        boundary = {
            "inclusive_min": float(threshold),
            "max_untriggered_lt": float(threshold),
        }
        distance = float(actual) - float(threshold)
    return {
        "actual": actual,
        "threshold": threshold,
        "triggered": triggered,
        "distance_to_threshold": round(distance, 6),
        "boundary": boundary,
    }


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def select_plan(
    spec: Spec,
    index: SpecIndex,
    *,
    thresholds: PlannerThresholds | None = None,
    forced_path: str | None = None,
) -> AlgorithmPlan:
    limits = thresholds or PlannerThresholds()
    if forced_path not in {None, PER_QUERY, MATERIALIZED}:
        raise ValueError(f"unknown forced path: {forced_path!r}")

    query_count = len(spec.queries)
    row_key_count = len(index.row_keys)
    total_versions = len(spec.records)
    avg_chain_length = _safe_ratio(float(total_versions), float(row_key_count))
    unique_pairs = len(index.query_pairs)

    per_query_work = query_count * (avg_chain_length + 1.0) + query_count * 0.5
    materialized_work = (
        unique_pairs * (avg_chain_length + 2.0)
        + unique_pairs * 0.35
        + query_count * 0.20
    )
    work_delta = materialized_work - per_query_work
    largest_work = max(per_query_work, materialized_work, 1.0)
    relative_delta = abs(work_delta) / largest_work
    in_tie_band = relative_delta <= limits.tie_band_ratio

    checks = {
        "query_count": _threshold_boundaries(
            query_count, limits.query_count, integer=True
        ),
        "row_key_count": _threshold_boundaries(
            row_key_count, limits.row_key_count, integer=True
        ),
        "avg_chain_length": _threshold_boundaries(
            avg_chain_length, limits.avg_chain_length, integer=False
        ),
    }
    crossed_count = sum(1 for item in checks.values() if item["triggered"])
    default_path = MATERIALIZED if crossed_count >= 2 else PER_QUERY

    reason_codes: list[str] = []
    if forced_path is not None:
        selected_path = forced_path
        reason_codes.append("FORCED_PATH")
        reason = f"调用方显式指定 {PATH_LABELS[selected_path]}，阈值诊断信息仍完整保留。"
        threshold_influenced = False
    elif in_tie_band:
        selected_path = default_path
        reason_codes.extend(["ESTIMATE_TIE_BAND", "THRESHOLD_MAJORITY"])
        reason = (
            "两条路径预估工作量落在 "
            f"{limits.tie_band_ratio:.0%} 交界带内，按 {limits.tie_break_rule} "
            f"稳定决胜：{crossed_count}/3 个规模阈值跨界，选择 {PATH_LABELS[selected_path]}。"
        )
        threshold_influenced = True
    else:
        selected_path = MATERIALIZED if materialized_work < per_query_work else PER_QUERY
        reason_codes.append("LOWER_ESTIMATED_WORK")
        reason = (
            "预估工作量不处于交界带，选择较低估算值："
            f"per_query={per_query_work:.3f}, materialized={materialized_work:.3f}，"
            f"选择 {PATH_LABELS[selected_path]}。"
        )
        threshold_influenced = False

    estimates = {
        "per_query_work": round(per_query_work, 6),
        "materialized_work": round(materialized_work, 6),
        "signed_delta_materialized_minus_per_query": round(work_delta, 6),
        "absolute_delta": round(abs(work_delta), 6),
        "relative_delta": round(relative_delta, 8),
        "formula": {
            "per_query": "Q*(L+1.0)+Q*0.5",
            "materialized": "U*(L+2.35)+Q*0.20",
            "Q": "query_count",
            "U": "unique (read_point,row_key) work pairs",
            "L": "total_versions/row_key_count",
        },
    }
    metrics = {
        "query_count": query_count,
        "row_key_count": row_key_count,
        "total_version_count": total_versions,
        "avg_chain_length": round(avg_chain_length, 6),
        "unique_read_point_row_pairs": unique_pairs,
        "read_point_count": len(spec.read_points),
    }
    return AlgorithmPlan(
        selected_path=selected_path,
        selected_label=PATH_LABELS[selected_path],
        reason=reason,
        reason_codes=tuple(reason_codes),
        threshold_influenced=threshold_influenced,
        threshold_crossed_count=crossed_count,
        default_path=default_path,
        metrics=metrics,
        thresholds=limits.to_dict(),
        boundary_values=checks,
        estimates=estimates,
        tie_band={
            "threshold_ratio": limits.tie_band_ratio,
            "in_band": in_tie_band,
            "stable_rule": limits.tie_break_rule,
        },
        forced=forced_path is not None,
    )
