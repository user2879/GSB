"""Cross-width trend-sign diagnostics."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from .engine import FitEngine
from .model import ALLOWED_FIXED_POINT_BITS, FitConfig, SeriesRecord


def _sign(value: float | int) -> str:
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def _slope_by_stage(artifact: dict[str, Any]) -> dict[str, float]:
    attribution = artifact["error_attribution"]
    result = {
        "truth": float(attribution["truth_coefficients"][1]),
    }
    for stage in attribution["stage_order"]:
        result[stage] = float(attribution["stages"][stage]["cumulative_stage_value"][1])
    return result


def classify_sign_origin(artifact: dict[str, Any]) -> dict[str, Any]:
    slopes = _slope_by_stage(artifact)
    signs = {stage: _sign(value) for stage, value in slopes.items()}
    causes: list[str] = []
    if signs["truth"] != signs["interpolation"]:
        causes.append("interpolation")
    if signs["interpolation"] != signs["normal_equation_accumulation"]:
        causes.append("normal_equation_quantization")
    reduction = bool(artifact["elimination"]["reductions"])
    if (
        signs["normal_equation_accumulation"] != signs["integer_elimination"]
        or reduction
    ):
        causes.append("elimination_pivot")
    if signs["integer_elimination"] != signs["coefficient_back_substitution"]:
        causes.append("fixed_point_rounding")
    return {
        "series_id": artifact["series_id"],
        "stage_slopes": slopes,
        "stage_signs": signs,
        "causes": causes,
        "final_sign": signs["coefficient_back_substitution"],
        "pivot_reduction_count": len(artifact["elimination"]["reductions"]),
    }


def compare_two_widths(
    primary: dict[str, Any], secondary: dict[str, Any]
) -> dict[str, Any]:
    if primary["series_id"] != secondary["series_id"]:
        raise ValueError("cannot compare different series")
    left = classify_sign_origin(primary)
    right = classify_sign_origin(secondary)
    flipped = left["final_sign"] != right["final_sign"]
    divergence_stage = None
    if flipped:
        left_slopes = left["stage_slopes"]
        right_slopes = right["stage_slopes"]
        for stage in (
            "interpolation",
            "normal_equation_accumulation",
            "integer_elimination",
            "coefficient_back_substitution",
        ):
            if _sign(left_slopes[stage]) != _sign(right_slopes[stage]):
                divergence_stage = stage
                break
    stage_to_cause = {
        "interpolation": "interpolation",
        "normal_equation_accumulation": "normal_equation_quantization",
        "integer_elimination": "elimination_pivot",
        "coefficient_back_substitution": "fixed_point_rounding",
    }
    return {
        "series_id": primary["series_id"],
        "flipped": flipped,
        "divergence_stage": divergence_stage,
        "attributed_cause": stage_to_cause.get(divergence_stage, "none"),
        "primary": left,
        "secondary": right,
    }


def run_width_sign_check(
    workspace: str | Path,
    series: Iterable[SeriesRecord],
    config: FitConfig,
    widths: Iterable[int],
    *,
    mode: str = "concurrent",
) -> dict[str, Any]:
    width_values = []
    for value in widths:
        if value not in ALLOWED_FIXED_POINT_BITS:
            raise ValueError(
                f"width {value} is not one of {list(ALLOWED_FIXED_POINT_BITS)}"
            )
        if value not in width_values:
            width_values.append(int(value))
    if len(width_values) != 2:
        raise ValueError("width sign check requires exactly two distinct widths")
    reports: dict[str, dict[str, Any]] = {}
    variants: dict[str, dict[str, Any]] = {}
    for width in width_values:
        shifted = min(config.quantization_shift, width - 8)
        if shifted < 6:
            shifted = 6
        variant_config = replace(
            config,
            fixed_point_bits=width,
            quantization_shift=shifted,
            comparison_bit_widths=(),
        )
        workspace_path = Path(workspace) / f"width-{width}"
        engine = FitEngine(
            workspace_path,
            variant_config,
            series,
            mode=mode,
        )
        reports[str(width)] = engine.run(
            version=variant_config.version,
            force_rerun=True,
        )
        variants[str(width)] = {
            "quantization_shift": shifted,
            "summary": reports[str(width)]["summary"],
            "failures": reports[str(width)]["failures"],
            "pending": reports[str(width)]["pending"],
        }
    by_width = {
        width: {item["series_id"]: item for item in report["series"]}
        for width, report in reports.items()
    }
    series_ids = sorted(set.intersection(*(set(value) for value in by_width.values())))
    comparisons = [
        compare_two_widths(
            by_width[str(width_values[0])][series_id],
            by_width[str(width_values[1])][series_id],
        )
        for series_id in series_ids
    ]
    return {
        "schema_version": 1,
        "widths": width_values,
        "variants": variants,
        "series_comparisons": comparisons,
        "flipped_series": [
            item["series_id"] for item in comparisons if item["flipped"]
        ],
        "classification_contract": {
            "interpolation": "imputed float path disagrees with observed-only truth sign",
            "normal_equation_quantization": "integer basis/target quantization changes sign before elimination",
            "elimination_pivot": "integer pivots or deterministic order reduction change sign",
            "fixed_point_rounding": "integer back-substitution rounding changes final sign",
        },
    }
