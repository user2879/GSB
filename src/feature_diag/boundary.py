"""Near-tie diagnostics for retained dimensions, cross features, and decisions."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np

from .errors import E_BOUNDARY_SENSITIVE


def _abs_correlation(left: np.ndarray, right: np.ndarray, dtype: np.dtype) -> float:
    left_values = np.asarray(left, dtype=dtype)
    right_values = np.asarray(right, dtype=dtype)
    left_centered = left_values - np.mean(left_values, dtype=dtype)
    right_centered = right_values - np.mean(right_values, dtype=dtype)
    denominator = np.sqrt(
        np.sum(left_centered * left_centered, dtype=dtype)
        * np.sum(right_centered * right_centered, dtype=dtype)
    )
    if not np.isfinite(denominator) or float(denominator) == 0.0:
        return 0.0
    value = float(np.sum(left_centered * right_centered, dtype=dtype) / denominator)
    return abs(value)


def _dimension_metric_map(reduction: dict[str, Any]) -> dict[int, float]:
    return {
        int(candidate["dimension"]): float(candidate["metric"])
        for candidate in reduction["candidate_dimensions"]
    }


def _cross_metric_map(design: dict[str, Any], dtype: np.dtype) -> dict[str, dict[str, Any]]:
    target = np.asarray(design["target"], dtype=dtype)
    output: dict[str, dict[str, Any]] = {}
    for feature in design["cross_features"]:
        signal = _abs_correlation(np.asarray(feature["values"]), target, dtype)
        output[str(feature["name"])] = {
            "metric": signal,
            "unique_count": int(feature["unique_count"]),
            "effective_unique_count": int(feature["effective_unique_count"]),
        }
    return output


def _boundary_record(
    *,
    scenario_id: str,
    sample_size: int,
    order_limit: int,
    kind: str,
    path: str | None,
    left_name: str,
    right_name: str,
    reference_left: float,
    reference_right: float,
    tier_left: float,
    tier_right: float,
    neighborhood: float,
    tie_tolerance: float,
    origin: str,
    sample_id: str | None = None,
    extra: dict[str, Any] | None = None,
    reference_decision_override: bool | None = None,
    tier_decision_override: bool | None = None,
) -> dict[str, Any]:
    reference_gap = abs(reference_left - reference_right)
    tier_gap = abs(tier_left - tier_right)
    reference_decision = reference_left > reference_right
    tier_decision = tier_left > tier_right
    if reference_decision_override is not None:
        reference_decision = bool(reference_decision_override)
    if tier_decision_override is not None:
        tier_decision = bool(tier_decision_override)
    value_delta_left = abs(tier_left - reference_left)
    value_delta_right = abs(tier_right - reference_right)
    precision_delta = max(value_delta_left, value_delta_right)
    flipped = reference_decision != tier_decision
    sensitive = bool(flipped and precision_delta > tie_tolerance)
    return {
        "boundary_id": "|".join(
            [
                scenario_id,
                f"n{sample_size}",
                f"o{order_limit}",
                kind,
                path or "-",
                left_name,
                right_name,
                sample_id or "-",
            ]
        ),
        "scenario_id": scenario_id,
        "sample_size": sample_size,
        "order_limit": order_limit,
        "kind": kind,
        "path": path,
        "sample_id": sample_id,
        "left": left_name,
        "right": right_name,
        "reference_precision": "float64",
        "reference_values": {
            "left": float(reference_left),
            "right": float(reference_right),
            "gap": float(reference_gap),
        },
        "float32_values": {"left": float(tier_left), "right": float(tier_right), "gap": float(tier_gap)},
        "tier_value_deltas": {"left": float(value_delta_left), "right": float(value_delta_right)},
        "precision_delta": float(precision_delta),
        "neighborhood": float(neighborhood),
        "tie_tolerance": float(tie_tolerance),
        "decisions": {
            "float32": bool(tier_decision),
            "float64": bool(reference_decision),
        },
        "flip": bool(flipped),
        "flip_origin": origin if flipped else None,
        "business_code": E_BOUNDARY_SENSITIVE if sensitive else None,
        "extra": extra or {},
    }


def diagnose_boundaries(
    *,
    designs: dict[str, dict[str, Any]],
    reductions: dict[str, dict[str, Any]],
    comparisons: dict[str, dict[str, Any]],
    neighborhood: float,
    tie_tolerance: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for task_key in sorted(comparisons):
        comparison = comparisons[task_key]
        scenario_id = comparison["scenario_id"]
        sample_size = int(comparison["sample_size"])
        order_limit = int(comparison["order_limit"])
        path = comparison["path"]
        base_prefix = f"{scenario_id}|n{sample_size}|o{order_limit}|{path}"
        reduction32 = reductions.get(f"{base_prefix}|float32")
        reduction64 = reductions.get(f"{base_prefix}|float64")
        if reduction32 is None or reduction64 is None:
            continue

        metric32 = _dimension_metric_map(reduction32)
        metric64 = _dimension_metric_map(reduction64)
        shared_dimensions = sorted(set(metric32) & set(metric64))
        for left_dimension, right_dimension in combinations(shared_dimensions, 2):
            reference_left = metric64[left_dimension]
            reference_right = metric64[right_dimension]
            if abs(reference_left - reference_right) > neighborhood:
                continue
            records.append(
                _boundary_record(
                    scenario_id=scenario_id,
                    sample_size=sample_size,
                    order_limit=order_limit,
                    kind="reduction_dimension",
                    path=path,
                    left_name=f"dimension_{left_dimension}",
                    right_name=f"dimension_{right_dimension}",
                    reference_left=reference_left,
                    reference_right=reference_right,
                    tier_left=metric32[left_dimension],
                    tier_right=metric32[right_dimension],
                    neighborhood=neighborhood,
                    tie_tolerance=tie_tolerance,
                    origin="projection_accumulation",
                    extra={
                        "candidate_metric": "cumulative_explained_or_retained_variance_ratio",
                        "retained_dimension": int(reduction64["retained_dimension"]),
                    },
                )
            )

        threshold = float(comparison["threshold"])
        for sample in comparison["per_sample"]:
            margin = abs(float(sample["metric_float64"]) - threshold)
            if margin > neighborhood:
                continue
            records.append(
                _boundary_record(
                    scenario_id=scenario_id,
                    sample_size=sample_size,
                    order_limit=order_limit,
                    kind="decision_threshold",
                    path=path,
                    left_name=f"metric:{sample['sample_id']}",
                    right_name="threshold",
                    reference_left=float(sample["metric_float64"]),
                    reference_right=threshold,
                    tier_left=float(sample["metric_float32"]),
                    tier_right=threshold,
                    neighborhood=neighborhood,
                    tie_tolerance=tie_tolerance,
                    origin=(
                        "encoding_rounding"
                        if sample["primary_error_stage"] in {"cross_encoding", "standardization"}
                        else "projection_accumulation"
                    ),
                    sample_id=sample["sample_id"],
                    reference_decision_override=bool(sample["decision_float64"]),
                    tier_decision_override=bool(sample["decision_float32"]),
                    extra={
                        "candidate_metric": "downstream_metric_vs_threshold",
                        "first_exceeded_stage": sample["first_exceeded_stage"],
                        "primary_error_stage": sample["primary_error_stage"],
                    },
                )
            )

    design_prefixes = sorted(
        {
            key.rsplit("|", 1)[0]
            for key in designs
            if key.endswith("|float32") or key.endswith("|float64")
        }
    )
    for design_prefix in design_prefixes:
        design32 = designs.get(f"{design_prefix}|float32")
        design64 = designs.get(f"{design_prefix}|float64")
        if design32 is None or design64 is None:
            continue
        scenario_id = str(design64["scenario_id"])
        sample_size = int(design64["sample_size"])
        order_limit = int(design64["order_limit"])
        cross32 = _cross_metric_map(design32, np.dtype("float32"))
        cross64 = _cross_metric_map(design64, np.dtype("float64"))
        shared_features = sorted(set(cross32) & set(cross64))
        for left_feature, right_feature in combinations(shared_features, 2):
            reference_left = cross64[left_feature]["metric"]
            reference_right = cross64[right_feature]["metric"]
            if abs(reference_left - reference_right) > neighborhood:
                continue
            records.append(
                _boundary_record(
                    scenario_id=scenario_id,
                    sample_size=sample_size,
                    order_limit=order_limit,
                    kind="cross_feature",
                    path=None,
                    left_name=left_feature,
                    right_name=right_feature,
                    reference_left=reference_left,
                    reference_right=reference_right,
                    tier_left=cross32[left_feature]["metric"],
                    tier_right=cross32[right_feature]["metric"],
                    neighborhood=neighborhood,
                    tie_tolerance=tie_tolerance,
                    origin="encoding_rounding",
                    extra={
                        "candidate_metric": "absolute_pearson_correlation_with_target",
                        "left_unique_count": cross64[left_feature]["unique_count"],
                        "right_unique_count": cross64[right_feature]["unique_count"],
                        "left_effective_unique_count": cross64[left_feature]["effective_unique_count"],
                        "right_effective_unique_count": cross64[right_feature]["effective_unique_count"],
                    },
                )
            )

    records.sort(
        key=lambda item: (
            item["scenario_id"],
            item["sample_size"],
            item["order_limit"],
            item["kind"],
            item["path"] or "",
            item["left"],
            item["right"],
            item["sample_id"] or "",
        )
    )
    return records
