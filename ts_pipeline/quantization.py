"""Fixed-point quantization and deterministic error aggregation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Iterable

from .config import QuantizationConfig


class QuantizationError(ValueError):
    pass


@dataclass(frozen=True)
class Quantized:
    bucket: int
    real: float


def quantize_value(value: float, config: QuantizationConfig, *, level_override: int | None = None) -> Quantized:
    if not math.isfinite(value):
        raise QuantizationError("cannot quantize a non-finite value")
    level = config.quantization_level if level_override is None else level_override
    step = Decimal(level) / Decimal(10**config.fixed_decimal_places)
    if step <= 0:
        raise QuantizationError("quantization step must be positive")
    scaled = Decimal(str(value)) / step
    bucket = int(scaled.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    return Quantized(bucket=bucket, real=float(Decimal(bucket) * step))


def _summarize(
    *,
    absolute_error_sum: float,
    squared_error_sum: float,
    percentage_error_sum: float,
    point_count: int,
    percentage_count: int,
    step: float,
) -> dict[str, float | int | None]:
    mae = absolute_error_sum / point_count * step if point_count else None
    rmse = math.sqrt(squared_error_sum / point_count) * step if point_count else None
    mape = percentage_error_sum / percentage_count * 100.0 if percentage_count else None
    return {
        "point_count": point_count,
        "percentage_point_count": percentage_count,
        "absolute_error_sum_fixed": absolute_error_sum,
        "squared_error_sum_fixed": squared_error_sum,
        "percentage_error_sum": percentage_error_sum,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
    }


def evaluate_window(
    predictions: Iterable[float],
    actuals: Iterable[float | None],
    config: QuantizationConfig,
    tolerance: float,
    *,
    level_override: int | None = None,
) -> dict:
    prediction_list = list(predictions)
    actual_list = list(actuals)
    if len(prediction_list) != len(actual_list):
        raise ValueError("predictions and actuals must have equal length")
    step = (
        level_override if level_override is not None else config.quantization_level
    ) / float(10**config.fixed_decimal_places)
    points: list[dict] = []
    absolute_error_sum = 0.0
    squared_error_sum = 0.0
    percentage_error_sum = 0.0
    percentage_count = 0
    first_exceeded: dict | None = None
    missing_actual_steps: list[int] = []
    for step_index, (prediction, actual) in enumerate(zip(prediction_list, actual_list)):
        quantized_prediction = quantize_value(prediction, config, level_override=level_override)
        if actual is None:
            missing_actual_steps.append(step_index)
            points.append(
                {
                    "step": step_index,
                    "prediction": prediction,
                    "prediction_fixed": quantized_prediction.bucket,
                    "prediction_quantized": quantized_prediction.real,
                    "actual": None,
                    "actual_fixed": None,
                    "actual_quantized": None,
                    "absolute_error": None,
                    "squared_error": None,
                    "absolute_percentage_error": None,
                }
            )
            continue
        quantized_actual = quantize_value(actual, config, level_override=level_override)
        error_bucket = quantized_prediction.bucket - quantized_actual.bucket
        absolute_error = abs(error_bucket) * step
        squared_error = float(error_bucket * error_bucket) * step * step
        percentage_error = (
            abs(error_bucket) / abs(quantized_actual.bucket)
            if quantized_actual.bucket != 0
            else None
        )
        absolute_error_sum += abs(error_bucket)
        squared_error_sum += float(error_bucket * error_bucket)
        if percentage_error is not None:
            percentage_error_sum += percentage_error
            percentage_count += 1
        point = {
            "step": step_index,
            "prediction": prediction,
            "prediction_fixed": quantized_prediction.bucket,
            "prediction_quantized": quantized_prediction.real,
            "actual": actual,
            "actual_fixed": quantized_actual.bucket,
            "actual_quantized": quantized_actual.real,
            "error_fixed": error_bucket,
            "absolute_error": absolute_error,
            "squared_error": squared_error,
            "absolute_percentage_error": percentage_error,
        }
        points.append(point)
        if first_exceeded is None and absolute_error > tolerance:
            first_exceeded = {
                "step": step_index,
                "step_number": step_index + 1,
                "fixed_error": error_bucket,
                "absolute_error": absolute_error,
                "squared_error": squared_error,
                "absolute_percentage_error": percentage_error,
                "prediction_fixed": quantized_prediction.bucket,
                "actual_fixed": quantized_actual.bucket,
            }
    point_count = len(points) - len(missing_actual_steps)
    summary = _summarize(
        absolute_error_sum=absolute_error_sum,
        squared_error_sum=squared_error_sum,
        percentage_error_sum=percentage_error_sum,
        point_count=point_count,
        percentage_count=percentage_count,
        step=step,
    )
    summary.update(
        {
            "tolerance": tolerance,
            "tolerance_exceeded": first_exceeded is not None,
            "first_exceeded": first_exceeded,
            "missing_actual_steps": missing_actual_steps,
            "fixed_step": step,
            "points": points,
        }
    )
    return summary


def aggregate_metrics(window_metrics: Iterable[dict]) -> dict:
    ordered = list(window_metrics)
    absolute_error_sum = 0.0
    squared_error_sum = 0.0
    percentage_error_sum = 0.0
    point_count = 0
    percentage_count = 0
    window_count = 0
    tolerance_exceeded_windows = 0
    fixed_steps: set[float] = set()
    for metrics in ordered:
        window_count += 1
        point_count += int(metrics["point_count"])
        percentage_count += int(metrics["percentage_point_count"])
        absolute_error_sum += float(metrics["absolute_error_sum_fixed"])
        squared_error_sum += float(metrics["squared_error_sum_fixed"])
        percentage_error_sum += float(metrics.get("percentage_error_sum", 0.0))
        if metrics.get("tolerance_exceeded"):
            tolerance_exceeded_windows += 1
        fixed_steps.add(float(metrics["fixed_step"]))
    step = next(iter(fixed_steps)) if len(fixed_steps) == 1 else None
    if step is None:
        return {
            "window_count": window_count,
            "point_count": point_count,
            "percentage_point_count": percentage_count,
            "mae": None,
            "rmse": None,
            "mape": None,
            "tolerance_exceeded_windows": tolerance_exceeded_windows,
        }
    mae = absolute_error_sum / point_count * step if point_count else None
    rmse = math.sqrt(squared_error_sum / point_count) * step if point_count else None
    mape = percentage_error_sum / percentage_count * 100.0 if percentage_count else None
    return {
        "window_count": window_count,
        "point_count": point_count,
        "percentage_point_count": percentage_count,
        "absolute_error_sum_fixed": absolute_error_sum,
        "squared_error_sum_fixed": squared_error_sum,
        "percentage_error_sum": percentage_error_sum,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "fixed_step": step,
        "tolerance_exceeded_windows": tolerance_exceeded_windows,
    }
