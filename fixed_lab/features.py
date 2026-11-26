"""Feature fixed-point quantization and deterministic correlation analysis."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from typing import Any

from .dataset import Dataset
from .fixedpoint import FixedParams, quantize_feature, quantize_fine


@dataclass
class QuantizedDataset:
    params: FixedParams
    sample_ids: list[str]
    columns: list[str]
    x: list[list[int]]
    y: list[int]
    w: list[int]
    feature_records: list[list[dict[str, Any]]]
    residuals: list[list[Decimal]]

    def column_vectors(self) -> list[list[int]]:
        return [[self.x[row][col] for row in range(len(self.sample_ids))] for col in range(len(self.columns))]


def quantize_dataset(dataset: Dataset, params: FixedParams) -> QuantizedDataset:
    sample_ids = dataset.ids
    width = len(dataset.columns)
    x: list[list[int]] = []
    y: list[int] = []
    w: list[int] = []
    records_by_column: list[list[dict[str, Any]]] = [[] for _ in range(width)]
    residuals: list[list[Decimal]] = []

    for sample_id in sample_ids:
        row = dataset.rows[sample_id]
        int_row: list[int] = []
        row_residuals: list[Decimal] = []
        for col in range(width):
            quantized = quantize_feature(row.features[col], params)
            int_row.append(quantized.integer)
            row_residuals.append(quantized.residual)
            record = quantized.as_dict()
            record.update({"sample_id": sample_id, "column_index": col, "column": dataset.columns[col]})
            records_by_column[col].append(record)
        x.append(int_row)
        residuals.append(row_residuals)
        y_integer, _ = quantize_fine(row.target, params)
        w_integer, _ = quantize_fine(row.weight, params)
        y.append(y_integer)
        w.append(w_integer)

    return QuantizedDataset(
        params=params,
        sample_ids=sample_ids,
        columns=list(dataset.columns),
        x=x,
        y=y,
        w=w,
        feature_records=records_by_column,
        residuals=residuals,
    )


def _threshold_ratio(value: Decimal) -> tuple[int, int]:
    numerator, denominator = value.as_integer_ratio()
    return numerator, denominator


def _decimal_sqrt(value: Decimal) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = 60
        return value.sqrt()


def correlation_analysis(qd: QuantizedDataset, strong_threshold: Decimal, collinear_threshold: Decimal) -> dict[str, Any]:
    vectors = qd.column_vectors()
    n = len(qd.sample_ids)
    p = len(qd.columns)
    weights = qd.w
    weight_sum = sum(weights)

    sums_x: list[int] = []
    sums_xx: list[int] = []
    norms: list[Decimal] = []
    half_step = Decimal(qd.params.grid) / Decimal(2 * qd.params.scale)

    for col in range(p):
        vector = vectors[col]
        sx = 0
        sxx = 0
        for idx in range(n):
            sx += weights[idx] * vector[idx]
            sxx += weights[idx] * vector[idx] * vector[idx]
        sums_x.append(sx)
        sums_xx.append(sxx)
        variance_cross = weight_sum * sxx - sx * sx
        if variance_cross > 0:
            rms_sq = Decimal(variance_cross) / Decimal(weight_sum * weight_sum) if weight_sum else Decimal(0)
            rms = _decimal_sqrt(abs(rms_sq)) if rms_sq > 0 else Decimal(0)
            norms.append(rms)
        else:
            norms.append(Decimal(0))

    pairs: list[dict[str, Any]] = []
    strong_pairs: list[dict[str, Any]] = []
    near_pairs: list[dict[str, Any]] = []
    boundary_pairs: list[dict[str, Any]] = []

    for i in range(p):
        for j in range(i + 1, p):
            sxy = 0
            for idx in range(n):
                sxy += weights[idx] * vectors[i][idx] * vectors[j][idx]
            c_cross = weight_sum * sxy - sums_x[i] * sums_x[j]
            v_i = weight_sum * sums_xx[i] - sums_x[i] * sums_x[i]
            v_j = weight_sum * sums_xx[j] - sums_x[j] * sums_x[j]
            variance_product = v_i * v_j

            corr_decimal: Decimal | None = None
            exact_match_strong = False
            exact_match_col = False
            if variance_product > 0:
                corr_decimal = Decimal(c_cross) / _decimal_sqrt(Decimal(variance_product))
                abs_corr = abs(corr_decimal)
                sp, sq = _threshold_ratio(strong_threshold)
                cp, cq = _threshold_ratio(collinear_threshold)
                exact_match_strong = sq * sq * c_cross * c_cross >= sp * sp * variance_product
                exact_match_col = cq * cq * c_cross * c_cross >= cp * cp * variance_product
            else:
                abs_corr = Decimal(0)

            scale_band = Decimal(0)
            if norms[i] > 0 and norms[j] > 0:
                scale_band = half_step / norms[i] + half_step / norms[j]
            ulp_band = Decimal("0.5e-12")
            quant_band = max(scale_band, ulp_band)

            strong_distance = abs(abs_corr - strong_threshold)
            col_distance = abs(abs_corr - collinear_threshold)
            exact_threshold = strong_distance == 0 or col_distance == 0
            within_half_step = strong_distance <= quant_band or col_distance <= quant_band
            boundary = exact_threshold or within_half_step

            is_strong = bool(exact_match_strong)
            strong_decision_basis = "exact_integer_covariance_comparison"
            if not is_strong and within_half_step:
                is_strong = True
                strong_decision_basis = "half_step_policy=include_below_threshold"

            is_near = bool(exact_match_col)
            col_decision_basis = "exact_integer_covariance_comparison"
            if not is_near and col_distance <= quant_band:
                is_near = True
                col_decision_basis = "half_step_policy=include_below_threshold"

            category = "none"
            if is_strong:
                category = "near_collinear" if is_near else "strong"

            pair = {
                "i": i,
                "j": j,
                "columns": [qd.columns[i], qd.columns[j]],
                "covariance_cross_integer": c_cross,
                "abs_covariance_cross_integer": abs(c_cross),
                "variance_i_integer": v_i,
                "variance_j_integer": v_j,
                "correlation": format(corr_decimal, ".12f") if corr_decimal is not None else None,
                "abs_correlation": format(abs_corr, ".12f"),
                "strong_threshold": format(strong_threshold, "f"),
                "collinearity_threshold": format(collinear_threshold, "f"),
                "strong_match": is_strong,
                "near_collinear_match": is_near,
                "category": category,
                "boundary": boundary,
                "boundary_exact_threshold": exact_threshold,
                "within_quant_half_step": within_half_step,
                "half_step_correlation_band": format(quant_band, ".12g"),
                "strong_decision_basis": strong_decision_basis,
                "collinear_decision_basis": col_decision_basis,
                "caliber": "weighted_fixed_point_covariance; exact squared integer threshold comparison",
            }
            pairs.append(pair)
            if is_strong:
                strong_pairs.append(pair)
            if is_near:
                near_pairs.append(pair)
            if boundary:
                boundary_pairs.append(pair)

    order_effect_pairs = [
        {"i": pair["i"], "j": pair["j"], "columns": pair["columns"], "category": pair["category"]}
        for pair in strong_pairs
    ]
    return {
        "caliber": "weighted fixed-point covariance, sample id ascending, no array reduction default order",
        "sample_order": qd.sample_ids,
        "pairs": pairs,
        "strong_pairs": strong_pairs,
        "near_collinear_pairs": near_pairs,
        "boundary_pairs": boundary_pairs,
        "coordinate_order_pairs": order_effect_pairs,
    }
