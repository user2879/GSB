"""Per-column fixed-point requantization, saturation and error attribution."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .calibration import (
    full_range_quant_params,
    round_half_away_from_zero_array,
)
from .models import TensorConfig

ATTRIBUTION_SOURCES = (
    "quantile_discretization",
    "scale_quantization",
    "saturation_clipping",
    "dequant_rounding",
)
ATTRIBUTION_PRIORITY = (
    "saturation_clipping",
    "dequant_rounding",
    "scale_quantization",
    "quantile_discretization",
)
_INT64_LIMIT = 9.223372036854775e18


def _round_to_int64(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nan_mask = np.isnan(values)
    safe = np.where(nan_mask, 0.0, values)
    safe = np.clip(safe, -_INT64_LIMIT, _INT64_LIMIT)
    rounded = round_half_away_from_zero_array(safe)
    overflow = (~nan_mask) & (
        np.isinf(values) | (np.abs(values) > _INT64_LIMIT)
    )
    return rounded.astype(np.int64), overflow


def _apply_strategy(
    preclip: np.ndarray,
    *,
    code_min: int,
    code_max: int,
    strategy: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    below = preclip < code_min
    above = preclip > code_max
    clipped = below | above
    if strategy == "wrap":
        levels = code_max - code_min + 1
        codes = ((preclip - code_min) % levels) + code_min
        rejected = np.zeros(preclip.shape, dtype=bool)
    else:
        codes = np.clip(preclip, code_min, code_max)
        rejected = clipped if strategy == "reject" else np.zeros(preclip.shape, dtype=bool)
    return codes, clipped, rejected


def reconstruction_from_params(
    values: np.ndarray,
    missing_mask: np.ndarray,
    params: dict[str, Any],
    strategy: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Counterfactual reconstruction used by attribution."""

    active = ~missing_mask
    scale = float(params["quantized_scale"])
    zero_point = float(params["zero_point"])
    code_min = int(params["code_min"])
    code_max = int(params["code_max"])
    codes = np.zeros(values.shape, dtype=np.int64)
    if scale == 0.0:
        return codes, np.zeros(values.shape, dtype=np.float64)
    exact = np.zeros(values.shape, dtype=np.float64)
    exact[active] = values[active] / scale + zero_point
    preclip, _ = _round_to_int64(exact)
    codes, _, _ = _apply_strategy(
        preclip,
        code_min=code_min,
        code_max=code_max,
        strategy="saturate" if strategy == "reject" else strategy,
    )
    dequantized = np.zeros(values.shape, dtype=np.float64)
    dequantized[active] = (codes[active].astype(np.float64) - zero_point) * scale
    return codes, dequantized


def _ideal_scale_reconstruction(
    values: np.ndarray,
    missing_mask: np.ndarray,
    params: dict[str, Any],
) -> np.ndarray:
    """Unclipped dequantization using the unquantized scale and ideal zero point."""

    active = ~missing_mask
    scale = float(params["ideal_scale"])
    zero_point = float(params["ideal_zero_point"])
    result = np.zeros(values.shape, dtype=np.float64)
    if scale == 0.0:
        return result
    exact = np.zeros(values.shape, dtype=np.float64)
    exact[active] = values[active] / scale + zero_point
    rounded, _ = _round_to_int64(exact)
    result[active] = (rounded[active].astype(np.float64) - zero_point) * scale
    return result


def _attribute_errors(
    *,
    values: np.ndarray,
    missing_mask: np.ndarray,
    tensor: TensorConfig,
    calibration: dict[str, Any],
    params: dict[str, Any],
    strategy: str,
    preclip: np.ndarray,
    codes: np.ndarray,
    dequantized: np.ndarray,
    exact_code: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    active = ~missing_mask
    scale = float(params["quantized_scale"])
    absolute_error = np.zeros(values.shape, dtype=np.float64)
    absolute_error[active] = np.abs(dequantized[active] - values[active])

    if scale == 0.0:
        zeros = np.zeros(values.shape, dtype=np.float64)
        raw = {source: zeros.copy() for source in ATTRIBUTION_SOURCES}
        raw["dequant_rounding"] = absolute_error.copy()
    else:
        final_codes = codes.astype(np.float64)
        preclip_float = preclip.astype(np.float64)
        saturation_raw = np.zeros(values.shape, dtype=np.float64)
        saturation_raw[active] = (
            np.abs(final_codes[active] - preclip_float[active]) * scale
        )
        if strategy == "wrap":
            # A wrapped code is intentionally not a saturation error.  Preserve
            # its direction in the element record, but attribute the arithmetic
            # consequence to code-domain wrapping as dequant rounding.
            saturation_raw[:] = 0.0

        rounding_raw = np.zeros(values.shape, dtype=np.float64)
        rounding_raw[active] = (
            np.abs(preclip_float[active] - exact_code[active]) * scale
        )

        scale_raw = np.zeros(values.shape, dtype=np.float64)
        ideal_scale_dequant = _ideal_scale_reconstruction(
            values, missing_mask, params
        )
        unclipped_quant_dequant = np.zeros(values.shape, dtype=np.float64)
        unclipped_quant_dequant[active] = (
            preclip_float[active] - float(params["zero_point"])
        ) * scale
        scale_raw[active] = np.abs(
            unclipped_quant_dequant[active] - ideal_scale_dequant[active]
        )

        full_range_params = full_range_quant_params(
            observed_min=float(calibration["observed_min"]),
            observed_max=float(calibration["observed_max"]),
            tensor=tensor,
            bit_width=int(params["bit_width"]),
        )
        _, full_range_dequant = reconstruction_from_params(
            values, missing_mask, full_range_params, strategy
        )
        quantile_raw = np.zeros(values.shape, dtype=np.float64)
        quantile_raw[active] = np.abs(
            full_range_dequant[active] - dequantized[active]
        )

        raw = {
            "quantile_discretization": quantile_raw,
            "scale_quantization": scale_raw,
            "saturation_clipping": saturation_raw,
            "dequant_rounding": rounding_raw,
        }

    raw_sum = np.zeros(values.shape, dtype=np.float64)
    for source in ATTRIBUTION_SOURCES:
        raw[source] = np.nan_to_num(
            raw[source], nan=0.0, posinf=1.0e308, neginf=0.0
        )
        raw_sum = raw_sum + raw[source]
    raw_sum = np.nan_to_num(raw_sum, nan=0.0, posinf=1.0e308, neginf=0.0)

    shares = {
        source: np.zeros(values.shape, dtype=np.float64)
        for source in ATTRIBUTION_SOURCES
    }
    nonzero = active & (absolute_error > 0.0)
    if np.any(nonzero):
        fallback = nonzero & (raw_sum <= 0.0)
        regular = nonzero & (raw_sum > 0.0)
        for source in ATTRIBUTION_SOURCES:
            shares[source][regular] = (
                raw[source][regular] / raw_sum[regular] * absolute_error[regular]
            )
        if np.any(fallback):
            shares["dequant_rounding"][fallback] = absolute_error[fallback]
        for row in range(values.shape[0]):
            if not nonzero[row]:
                continue
            component_sum = sum(
                float(shares[source][row]) for source in ATTRIBUTION_SOURCES
            )
            correction = float(absolute_error[row] - component_sum)
            if correction != 0.0:
                if raw_sum[row] > 0.0:
                    target = max(
                        ATTRIBUTION_SOURCES,
                        key=lambda source: (
                            float(raw[source][row]),
                            -ATTRIBUTION_PRIORITY.index(source),
                        ),
                    )
                else:
                    target = "dequant_rounding"
                shares[target][row] += correction

    dominant = np.full(values.shape, "none", dtype=object)
    for row in range(values.shape[0]):
        if not active[row] or absolute_error[row] == 0.0:
            continue
        best_source = ATTRIBUTION_PRIORITY[0]
        best_value = -1.0
        for source in ATTRIBUTION_PRIORITY:
            value = float(raw[source][row]) if raw_sum[row] > 0.0 else 0.0
            if value > best_value:
                best_value = value
                best_source = source
        dominant[row] = best_source
    return absolute_error, shares, dominant, raw_sum


def analyze_column_width(
    *,
    values: np.ndarray,
    missing_mask: np.ndarray,
    tensor: TensorConfig,
    calibration: dict[str, Any],
    params: dict[str, Any],
    sample_refs: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Quantize, saturate, dequantize and attribute one column at one width."""

    strategy = tensor.saturation_strategy
    active = ~missing_mask
    rows = int(values.shape[0])
    scale = float(params["quantized_scale"])
    zero_point = int(params["zero_point"])
    code_min = int(params["code_min"])
    code_max = int(params["code_max"])

    exact_code = np.full(values.shape, np.nan, dtype=np.float64)
    preclip = np.zeros(values.shape, dtype=np.int64)
    codes = np.zeros(values.shape, dtype=np.int64)
    clipped = np.zeros(values.shape, dtype=bool)
    rejected = np.zeros(values.shape, dtype=bool)
    domain_overflow = np.zeros(values.shape, dtype=bool)
    dequantized = np.zeros(values.shape, dtype=np.float64)

    if scale > 0.0:
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            exact_code[active] = values[active] / scale + float(zero_point)
        preclip_all, domain_overflow = _round_to_int64(exact_code)
        preclip[active] = preclip_all[active]
        codes_all, clipped_all, rejected_all = _apply_strategy(
            preclip_all,
            code_min=code_min,
            code_max=code_max,
            strategy=strategy,
        )
        codes[active] = codes_all[active]
        clipped[active] = clipped_all[active]
        rejected[active] = rejected_all[active]
        dequantized[active] = (
            codes[active].astype(np.float64) - float(zero_point)
        ) * scale

    absolute_error, shares, dominant, raw_sum = _attribute_errors(
        values=values,
        missing_mask=missing_mask,
        tensor=tensor,
        calibration=calibration,
        params=params,
        strategy=strategy,
        preclip=preclip,
        codes=codes,
        dequantized=dequantized,
        exact_code=exact_code,
    )

    failures: list[dict[str, Any]] = []
    if bool(params["zero_scale"]):
        failures.append(
            {
                "error_code": "E_SCALE_ZERO",
                "field": "column.scale",
                "message": (
                    "zero scale column; policy="
                    f"{tensor.zero_scale_policy}, degraded={bool(params['degraded_scale'])}"
                ),
            }
        )
    rejected_count = int(np.count_nonzero(rejected))
    if strategy == "reject" and rejected_count:
        failures.append(
            {
                "error_code": "E_SATURATION_REJECTED",
                "field": "column.elements",
                "message": f"{rejected_count} element(s) required saturation under reject policy",
            }
        )

    elements: list[dict[str, Any]] = []
    saturated_count = 0
    over_tolerance_count = 0
    zero_reference_count = 0
    max_abs_error = 0.0
    max_relative_error_nonzero = 0.0
    max_effective_relative_error = 0.0
    source_all = {source: 0.0 for source in ATTRIBUTION_SOURCES}
    source_over = {source: 0.0 for source in ATTRIBUTION_SOURCES}
    dominant_over_counts = {source: 0 for source in ATTRIBUTION_SOURCES}
    share_error_max = 0.0
    for row in range(rows):
        ref = sample_refs[row]
        if not bool(active[row]):
            elements.append(
                {
                    "row": row,
                    "index": ref["index"],
                    "injection_ids": ref["injection_ids"],
                    "missing": True,
                    "value": None,
                    "exact_code": None,
                    "preclip_code": None,
                    "quantized_code": None,
                    "clipped": False,
                    "clip_direction": "none",
                    "domain_overflow": False,
                    "dequantized": None,
                    "abs_error": None,
                    "relative_error": None,
                    "effective_relative_error": None,
                    "error_branch": "missing",
                    "over_tolerance": False,
                    "dominant_source": "missing",
                    "attribution": None,
                }
            )
            continue
        value = float(values[row])
        abs_error = float(absolute_error[row])
        if value == 0.0:
            zero_reference_count += 1
            relative_error: float | None = None
            effective_relative_error = abs_error
            error_branch = "absolute_at_zero"
            over_tolerance = abs_error > tensor.tolerance
        else:
            relative_error = abs_error / abs(value)
            effective_relative_error = relative_error
            error_branch = "relative"
            over_tolerance = relative_error > tensor.tolerance
            if relative_error > max_relative_error_nonzero:
                max_relative_error_nonzero = relative_error
        if abs_error > max_abs_error:
            max_abs_error = abs_error
        if effective_relative_error > max_effective_relative_error:
            max_effective_relative_error = effective_relative_error
        if bool(clipped[row]):
            saturated_count += 1
        if over_tolerance:
            over_tolerance_count += 1
        direction = "none"
        if bool(clipped[row]):
            direction = "high" if preclip[row] > code_max else "low"
        share_values = {
            source: float(shares[source][row]) for source in ATTRIBUTION_SOURCES
        }
        share_sum = sum(share_values.values())
        share_delta = abs(share_sum - abs_error)
        share_error_max = max(share_error_max, share_delta)
        source = str(dominant[row])
        for key in ATTRIBUTION_SOURCES:
            source_all[key] += share_values[key]
            if over_tolerance:
                source_over[key] += share_values[key]
        if over_tolerance and source in dominant_over_counts:
            dominant_over_counts[source] += 1
        elements.append(
            {
                "row": row,
                "index": ref["index"],
                "injection_ids": ref["injection_ids"],
                "missing": False,
                "value": value,
                "exact_code": None
                if not math.isfinite(float(exact_code[row]))
                else float(exact_code[row]),
                "preclip_code": int(preclip[row]),
                "quantized_code": int(codes[row]),
                "clipped": bool(clipped[row]),
                "clip_direction": direction,
                "domain_overflow": bool(domain_overflow[row]),
                "dequantized": float(dequantized[row]),
                "abs_error": abs_error,
                "relative_error": relative_error,
                "effective_relative_error": effective_relative_error,
                "error_branch": error_branch,
                "over_tolerance": bool(over_tolerance),
                "dominant_source": source,
                "attribution": {
                    "priority": list(ATTRIBUTION_PRIORITY),
                    "shares": share_values,
                    "share_sum": share_sum,
                    "total_error": abs_error,
                    "share_sum_error": share_delta,
                    "raw_component_sum": float(raw_sum[row]),
                    "normalized_to_total_error": bool(share_delta <= tensor.tolerance),
                },
            }
        )

    active_count = int(np.count_nonzero(active))
    source_share_all = _normalize_source_sums(source_all)
    source_share_over = _normalize_source_sums(source_over)
    summary = {
        "bit_width": int(params["bit_width"]),
        "quantization_levels": int(params["quantization_levels"]),
        "fractional_bits": int(params["fractional_bits"]),
        "analyzed_count": active_count,
        "missing_count": int(np.count_nonzero(missing_mask)),
        "saturated_count": saturated_count,
        "saturation_ratio": saturated_count / active_count if active_count else 0.0,
        "over_tolerance_count": over_tolerance_count,
        "over_tolerance_ratio": (
            over_tolerance_count / active_count if active_count else 0.0
        ),
        "max_abs_error": max_abs_error,
        "max_relative_error": max_relative_error_nonzero,
        "max_relative_error_nonzero": max_relative_error_nonzero,
        "max_effective_relative_error": max_effective_relative_error,
        "zero_reference_count": zero_reference_count,
        "attribution_share_by_source": source_share_all,
        "over_tolerance_attribution_share_by_source": source_share_over,
        "attribution_error_total": float(sum(source_all.values())),
        "over_tolerance_attribution_error_total": float(sum(source_over.values())),
        "dominant_source_counts": dominant_over_counts,
        "attribution_share_error_max": share_error_max,
    }
    result = {
        "tensor_id": tensor.tensor_id,
        "column_index": None,
        "bit_width": int(params["bit_width"]),
        "quant_params": params,
        "summary": summary,
        "elements": elements,
        "failures": failures,
    }
    return result, failures


def _normalize_source_sums(source_sums: dict[str, float]) -> dict[str, float]:
    total = sum(float(value) for value in source_sums.values())
    if total <= 0.0:
        return {source: 0.0 for source in ATTRIBUTION_SOURCES}
    return {
        source: float(source_sums[source]) / total for source in ATTRIBUTION_SOURCES
    }
