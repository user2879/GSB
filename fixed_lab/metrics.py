"""Residual and objective statistics in stable sample order."""

from __future__ import annotations

from decimal import Decimal, localcontext
from typing import Any

from .dataset import Dataset
from .fixedpoint import FixedParams


def residual_statistics(
    dataset: Dataset,
    coefficients_fixed: list[int],
    params: FixedParams,
) -> dict[str, Any]:
    scale = Decimal(params.scale)
    betas = [Decimal(value) / scale for value in coefficients_fixed]
    ids = dataset.ids

    with localcontext() as ctx:
        ctx.prec = 50
        weight_sum = Decimal(0)
        residual_sum = Decimal(0)
        abs_sum = Decimal(0)
        weighted_sse = Decimal(0)
        raw_sse = Decimal(0)
        max_abs = Decimal(0)
        target_sum = Decimal(0)
        for sample_id in ids:
            row = dataset.rows[sample_id]
            prediction = Decimal(0)
            for j, beta in enumerate(betas):
                prediction += row.features[j] * beta
            residual = row.target - prediction
            abs_residual = abs(residual)
            weight = row.weight
            weight_sum += weight
            residual_sum += weight * residual
            abs_sum += weight * abs_residual
            weighted_sse += weight * residual * residual
            raw_sse += residual * residual
            max_abs = max(max_abs, abs_residual)
            target_sum += row.target

        count = Decimal(len(ids))
        weighted_mean_residual = residual_sum / weight_sum if weight_sum != 0 else None
        weighted_mae = abs_sum / weight_sum if weight_sum != 0 else None
        weighted_rmse = (weighted_sse / weight_sum).sqrt() if weight_sum > 0 else None
        mean_target = target_sum / count if ids else Decimal(0)
        total_sse = Decimal(0)
        for sample_id in ids:
            residual_target = dataset.rows[sample_id].target - mean_target
            total_sse += residual_target * residual_target
        r2 = Decimal(1) - raw_sse / total_sse if total_sse != 0 else None

    def number(value: Decimal | None) -> float | None:
        return None if value is None else float(value)

    return {
        "sample_order": ids,
        "sample_count": len(ids),
        "weight_sum": number(weight_sum),
        "weighted_mean_residual": number(weighted_mean_residual),
        "weighted_mae": number(weighted_mae),
        "weighted_rmse": number(weighted_rmse),
        "weighted_sse": number(weighted_sse),
        "raw_sse": number(raw_sse),
        "max_abs_residual": number(max_abs),
        "r_squared": number(r2),
        "caliber": "Decimal/50-digit ordered sample accumulation",
    }


def fixed_objective(
    qd,
    coefficients_fixed: list[int],
    alpha: Decimal,
    penalty: str,
) -> dict[str, Any]:
    params: FixedParams = qd.params
    scale = params.scale
    data_term = 0
    for row_index in range(len(qd.sample_ids)):
        numerator = scale * qd.y[row_index]
        for j, xij in enumerate(qd.x[row_index]):
            numerator -= xij * coefficients_fixed[j]
        data_term += qd.w[row_index] * numerator * numerator

    if penalty == "ridge":
        lambda_objective = int((alpha * Decimal(scale) ** 3).to_integral_value())
        penalty_term = lambda_objective * sum(value * value for value in coefficients_fixed)
        objective_fixed = data_term + penalty_term
        return {
            "objective_fixed": objective_fixed,
            "objective_scale": scale**5,
            "objective": float(Decimal(objective_fixed) / Decimal(scale**5)),
            "data_term_fixed": data_term,
            "penalty_term_fixed": penalty_term,
            "objective_caliber": "RSS + alpha*L2^2; fixed numerator at S^5",
        }

    lambda_objective = int((alpha * Decimal(scale) ** 4).to_integral_value())
    penalty_term_x2 = 2 * lambda_objective * sum(abs(value) for value in coefficients_fixed)
    objective_x2 = data_term + penalty_term_x2
    return {
        "objective_fixed_x2": objective_x2,
        "objective_scale_x2": 2 * scale**5,
        "objective": float(Decimal(objective_x2) / Decimal(2 * scale**5)),
        "data_term_fixed": data_term,
        "penalty_term_fixed_x2": penalty_term_x2,
        "objective_caliber": "1/2 RSS + alpha*L1; doubled fixed numerator at S^5",
    }
