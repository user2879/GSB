"""Fixed-point rescaling, saturation, dequantisation, and error attribution."""

from __future__ import annotations

from typing import Any

from .config import CalibrationConfig
from .keys import attribution_keys, fixed_rescale_keys
from .quantile import round_half_away_from_zero
from .table import FeatureTable


SOURCE_PRIORITY = (
    "saturation_truncation",
    "quantile_discretization",
    "scale_quantization",
    "dequant_rounding",
)


def _group_rows(table: FeatureTable, group_id: str) -> list[Any]:
    return sorted(table.rows_for_group(group_id), key=lambda row: row.row_id)


def fixed_rescale_group(
    config: CalibrationConfig,
    table: FeatureTable,
    column_groups: dict[str, tuple[str, ...]],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    """Quantise every cell in a group/width and retain all clipping evidence."""

    group_id = calibration["group_id"]
    columns = tuple(calibration["columns"])
    bit_width = int(calibration["bit_width"])
    fractional_bits = int(calibration["fractional_bits"])
    qmin = int(calibration["qmin"])
    qmax = int(calibration["qmax"])
    group_rows = _group_rows(table, group_id)
    keys = fixed_rescale_keys(
        config,
        content_fingerprint=calibration["content_fingerprint"],
        calibration_parameter_fingerprint=calibration["parameter_fingerprint"],
        calibration_cache_key=calibration["cache_key"],
        bit_width=bit_width,
    )

    elements: list[dict[str, Any]] = []
    truncation_count = 0
    missing_count = 0
    for row in group_rows:
        for column in columns:
            raw_value = row.values.get(column)
            if raw_value is None:
                missing_count += 1
                elements.append(
                    {
                        "group_id": group_id,
                        "bit_width": bit_width,
                        "row_id": row.row_id,
                        "column": column,
                        "missing": True,
                        "original": None,
                        "scale_factor": calibration["columns_calibration"][column]["scale_factor"],
                        "zero_point": calibration["columns_calibration"][column]["zero_point"],
                        "q_unclipped": None,
                        "q_value": None,
                        "clipped": False,
                        "clipping_direction": "none",
                        "dequantized": None,
                        "absolute_error": None,
                        "relative_error": None,
                        "zero_value_branch": False,
                        "tolerance_pass": None,
                        "is_outlier": None,
                    }
                )
                continue

            value = float(raw_value)
            calibrated = calibration["columns_calibration"][column]
            scale = float(calibrated["scale_factor"])
            zero_point = int(calibrated["zero_point"])
            q_unclipped = round_half_away_from_zero(value / scale) + zero_point
            if q_unclipped < qmin:
                q_value = qmin
                direction = "low"
            elif q_unclipped > qmax:
                q_value = qmax
                direction = "high"
            else:
                q_value = q_unclipped
                direction = "none"
            clipped = direction != "none"
            if clipped:
                truncation_count += 1
            dequantized = (q_value - zero_point) * scale
            if dequantized == 0.0:
                dequantized = 0.0
            elements.append(
                {
                    "group_id": group_id,
                    "bit_width": bit_width,
                    "row_id": row.row_id,
                    "column": column,
                    "missing": False,
                    "original": value,
                    "scale_factor": scale,
                    "zero_point": zero_point,
                    "q_unclipped": q_unclipped,
                    "q_value": q_value,
                    "clipped": clipped,
                    "clipping_direction": direction,
                    "dequantized": dequantized,
                }
            )

    nonmissing_count = len(elements) - missing_count
    return {
        "stage": "fixed_rescale",
        "task_sort_key": f"{group_id}|fixed_rescale|{bit_width:03d}",
        "group_id": group_id,
        "columns": list(columns),
        "row_ids": calibration["row_ids"],
        "bit_width": bit_width,
        "fractional_bits": fractional_bits,
        "active_bit_width": calibration["active_bit_width"],
        "qmin": qmin,
        "qmax": qmax,
        "clipping_strategy": config.clipping_strategy,
        "element_count": len(elements),
        "nonmissing_count": nonmissing_count,
        "missing_count": missing_count,
        "truncation_count": truncation_count,
        "truncation_ratio": (truncation_count / nonmissing_count) if nonmissing_count else 0.0,
        "truncation_ratio_all_cells": (truncation_count / len(elements)) if elements else 0.0,
        "elements": elements,
        "upstream_cache_key": calibration["cache_key"],
        "content_fingerprint": calibration["content_fingerprint"],
        "parameter_fingerprint": keys["parameter_fingerprint"],
        "cache_key": keys["cache_key"],
    }


def _source_shares(counts: dict[str, int], denominator: int) -> dict[str, float]:
    if denominator <= 0:
        return {source: 0.0 for source in SOURCE_PRIORITY}
    return {source: counts.get(source, 0) / denominator for source in SOURCE_PRIORITY}


def _tolerance_decision(
    config: CalibrationConfig,
    *,
    absolute_error: float,
    relative_error: float | None,
) -> tuple[bool, bool, bool, bool]:
    absolute_exceeded = absolute_error > config.tolerance.absolute
    if relative_error is None:
        return absolute_exceeded, False, False, absolute_exceeded
    relative_exceeded = relative_error > config.tolerance.relative
    if config.tolerance.comparison == "both":
        outlier = absolute_exceeded and relative_exceeded
    else:
        outlier = absolute_exceeded or relative_exceeded
    return absolute_exceeded, relative_exceeded, True, outlier


def attribute_group_errors(
    config: CalibrationConfig,
    calibration: dict[str, Any],
    rescaled: dict[str, Any],
) -> dict[str, Any]:
    """Attach four traceable error criteria to each non-missing quantised cell."""

    group_id = rescaled["group_id"]
    bit_width = int(rescaled["bit_width"])
    fractional_bits = int(rescaled["fractional_bits"])
    group_rows = rescaled["elements"]
    keys = attribution_keys(
        config,
        content_fingerprint=rescaled["content_fingerprint"],
        rescale_cache_key=rescaled["cache_key"],
        bit_width=bit_width,
    )

    attributed: list[dict[str, Any]] = []
    outlier_count = 0
    zero_branch_count = 0
    source_counts = {source: 0 for source in SOURCE_PRIORITY}
    outlier_source_counts = {source: 0 for source in SOURCE_PRIORITY}
    max_absolute_error = 0.0
    max_relative_error: float | None = None
    max_zero_branch_absolute_error: float | None = None

    for element in group_rows:
        if element["missing"]:
            attributed.append(dict(element))
            continue
        value = float(element["original"])
        column = element["column"]
        calibrated = calibration["columns_calibration"][column]
        reference = calibration["reference_full_range"][column]
        scale = float(calibrated["scale_factor"])
        zero_point = int(calibrated["zero_point"])
        raw_scale = float(calibrated["scale_raw"])
        raw_zero_point = int(calibrated["zero_point_unclipped"])
        q_unclipped = int(element["q_unclipped"])
        q_value = int(element["q_value"])
        dequantized = float(element["dequantized"])
        dequantized_unclipped = (q_unclipped - zero_point) * scale
        q_raw_scale = round_half_away_from_zero(value / raw_scale) + raw_zero_point
        dequantized_raw_scale = (q_raw_scale - raw_zero_point) * raw_scale
        reference_scale = float(reference["scale_factor"])
        reference_zero_point = int(reference["zero_point"])
        q_reference = round_half_away_from_zero(value / reference_scale) + reference_zero_point
        dequantized_reference = (q_reference - reference_zero_point) * reference_scale

        absolute_error = abs(value - dequantized)
        if value == 0.0:
            relative_error = None
            zero_branch = True
            zero_branch_count += 1
            max_zero_branch_absolute_error = (
                absolute_error
                if max_zero_branch_absolute_error is None
                else max(max_zero_branch_absolute_error, absolute_error)
            )
        else:
            relative_error = absolute_error / abs(value)
            zero_branch = False
            max_relative_error = (
                relative_error if max_relative_error is None else max(max_relative_error, relative_error)
            )
        max_absolute_error = max(max_absolute_error, absolute_error)

        contributions = {
            "quantile_discretization": abs(dequantized_reference - dequantized_unclipped),
            "scale_quantization": abs(dequantized_raw_scale - dequantized_unclipped)
            + abs(raw_zero_point - zero_point) * scale,
            "saturation_truncation": abs(dequantized_unclipped - dequantized),
            "dequant_rounding": abs(value - dequantized_unclipped),
        }
        maximum = max(contributions.values())
        epsilon = max(config.tolerance.tie_epsilon, 1e-15 * max(1.0, abs(value), maximum))
        tied_sources = [
            source
            for source in SOURCE_PRIORITY
            if abs(contributions[source] - maximum) <= epsilon
        ]
        if maximum <= epsilon and absolute_error <= epsilon:
            dominant_source = "none"
            is_tie = False
        else:
            dominant_source = next(
                (source for source in SOURCE_PRIORITY if source in tied_sources),
                "dequant_rounding",
            )
            is_tie = len(tied_sources) > 1

        absolute_exceeded, relative_exceeded, relative_applicable, is_outlier = _tolerance_decision(
            config,
            absolute_error=absolute_error,
            relative_error=relative_error,
        )
        if is_outlier:
            outlier_count += 1
            outlier_source_counts[dominant_source] = outlier_source_counts.get(dominant_source, 0) + 1
        source_counts[dominant_source] = source_counts.get(dominant_source, 0) + 1

        enriched = dict(element)
        enriched.update(
            {
                "dequantized_unclipped": dequantized_unclipped,
                "absolute_error": absolute_error,
                "relative_error": relative_error,
                "relative_error_applicable": relative_applicable,
                "zero_value_branch": zero_branch,
                "zero_relative_branch": config.tolerance.zero_relative_branch if zero_branch else None,
                "absolute_tolerance_exceeded": absolute_exceeded,
                "relative_tolerance_exceeded": relative_exceeded,
                "tolerance_pass": not is_outlier,
                "is_outlier": is_outlier,
                "criterion_values": contributions,
                "dominant_error_source": dominant_source,
                "source_tie": is_tie,
                "tied_sources": tied_sources,
                "source_priority": list(SOURCE_PRIORITY),
                "attribution_trace": {
                    "selected_scale": scale,
                    "selected_zero_point": zero_point,
                    "raw_scale": raw_scale,
                    "raw_zero_point": raw_zero_point,
                    "reference_scale": reference_scale,
                    "reference_zero_point": reference_zero_point,
                    "q_unclipped": q_unclipped,
                    "q_value": q_value,
                    "q_at_raw_scale": q_raw_scale,
                    "q_at_reference_scale": q_reference,
                },
            }
        )
        attributed.append(enriched)

    nonmissing_count = sum(1 for element in attributed if not element["missing"])
    return {
        "stage": "error_attribution",
        "task_sort_key": f"{group_id}|error_attribution|{bit_width:03d}",
        "group_id": group_id,
        "columns": rescaled["columns"],
        "row_ids": rescaled["row_ids"],
        "bit_width": bit_width,
        "fractional_bits": fractional_bits,
        "active_bit_width": rescaled["active_bit_width"],
        "element_count": len(attributed),
        "nonmissing_count": nonmissing_count,
        "missing_count": rescaled["missing_count"],
        "truncation_count": rescaled["truncation_count"],
        "truncation_ratio": rescaled["truncation_ratio"],
        "outlier_count": outlier_count,
        "outlier_ratio": (outlier_count / nonmissing_count) if nonmissing_count else 0.0,
        "zero_value_branch_count": zero_branch_count,
        "zero_relative_branch": config.tolerance.zero_relative_branch,
        "max_absolute_error": max_absolute_error,
        "max_relative_error": max_relative_error,
        "max_zero_branch_absolute_error": max_zero_branch_absolute_error,
        "dominant_source_counts": source_counts,
        "dominant_source_shares": _source_shares(source_counts, nonmissing_count),
        "outlier_source_counts": outlier_source_counts,
        "outlier_source_shares": _source_shares(outlier_source_counts, outlier_count),
        "source_priority": list(SOURCE_PRIORITY),
        "elements": attributed,
        "upstream_cache_key": rescaled["cache_key"],
        "content_fingerprint": rescaled["content_fingerprint"],
        "parameter_fingerprint": keys["parameter_fingerprint"],
        "cache_key": keys["cache_key"],
    }
