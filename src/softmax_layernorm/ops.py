from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .errors import BusinessFailure


def sum_float32_in_order(values: np.ndarray) -> np.float32:
    total = np.float32(0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        for value in np.asarray(values, dtype=np.float32).reshape(-1):
            total = np.float32(total + value)
    return total


def sum_float64_in_order(values: np.ndarray) -> np.float64:
    total = np.float64(0.0)
    with np.errstate(over="ignore", invalid="ignore"):
        for value in np.asarray(values, dtype=np.float64).reshape(-1):
            total = np.float64(total + value)
    return total


@dataclass(frozen=True)
class LaneResult:
    values: np.ndarray
    failure: BusinessFailure | None = None
    stats: dict[str, Any] | None = None


def _lane_failure(
    code: str,
    message: str,
    stage: str,
    call_id: str,
    call_sequence: int,
    lane: int,
    row: int | None,
    details: dict[str, Any] | None = None,
) -> BusinessFailure:
    return BusinessFailure(
        code=code,
        message=message,
        stage=stage,
        call_id=call_id,
        call_sequence=call_sequence,
        lane=lane,
        row=row,
        details=details or {},
    )


def softmax_lane(
    input_values: np.ndarray,
    *,
    stabilize: bool,
    call_id: str = "direct",
    call_sequence: int = 0,
    lane: int = 0,
    row: int | None = None,
    row_label: str = "row",
) -> LaneResult:
    """Softmax one contiguous lane using a fixed float32 summation order."""

    values = np.asarray(input_values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        failure = _lane_failure(
            "E_SHAPE_MISMATCH",
            "softmax lane is empty",
            "softmax",
            call_id,
            call_sequence,
            lane,
            row,
        )
        return LaneResult(values=np.empty(0, dtype=np.float32), failure=failure)

    with np.errstate(over="ignore", invalid="ignore"):
        if stabilize:
            maximum = np.max(values)
            exponential_input = values - maximum
            used_max_subtraction = True
        else:
            exponential_input = values
            used_max_subtraction = False

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        exponentials = np.exp(exponential_input).astype(np.float32, copy=False)
        total = sum_float32_in_order(exponentials)
        output = (exponentials / total).astype(np.float32, copy=False)

    failure = None
    if not stabilize and (np.any(np.isinf(exponentials)) or np.isinf(total)):
        failure = _lane_failure(
            "E_EXP_OVERFLOW",
            f"softmax exponent overflow with max subtraction disabled at {row_label} {lane}",
            "softmax",
            call_id,
            call_sequence,
            lane,
            row,
            {
                row_label: lane,
                "max_subtraction": used_max_subtraction,
                "sum": float(total),
            },
        )

    return LaneResult(
        values=output,
        failure=failure,
        stats={"max_subtraction": used_max_subtraction},
    )


def layer_norm_lane(
    input_values: np.ndarray,
    *,
    epsilon: float,
    scale: float = 1.0,
    call_id: str = "direct",
    call_sequence: int = 0,
    lane: int = 0,
    row: int | None = None,
    row_label: str = "row",
) -> LaneResult:
    """LayerNorm one contiguous lane with fixed mean/variance accumulation order."""

    values = np.asarray(input_values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        failure = _lane_failure(
            "E_SHAPE_MISMATCH",
            "layernorm lane is empty",
            "layernorm",
            call_id,
            call_sequence,
            lane,
            row,
        )
        return LaneResult(values=np.empty(0, dtype=np.float32), failure=failure)

    with np.errstate(over="ignore", invalid="ignore"):
        mean = np.float32(sum_float32_in_order(values) / np.float32(values.size))
        deviations = (values - mean).astype(np.float32, copy=False)
        squared = (deviations * deviations).astype(np.float32, copy=False)
        variance = np.float32(sum_float32_in_order(squared) / np.float32(values.size))
        epsilon_f32 = np.float32(epsilon)
        scale_f32 = np.float32(scale)

    failure = None
    if variance == np.float32(0.0) and epsilon_f32 == np.float32(0.0):
        failure = _lane_failure(
            "E_ZERO_VARIANCE",
            f"zero variance with zero epsilon at {row_label} {lane}",
            "layernorm",
            call_id,
            call_sequence,
            lane,
            row,
            {row_label: lane, "mean": float(mean), "variance": float(variance)},
        )

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        denominator = np.sqrt(np.float32(variance + epsilon_f32))
        normalized = ((deviations / denominator) * scale_f32).astype(np.float32, copy=False)

    epsilon_applied = bool(variance == np.float32(0.0) and epsilon_f32 > np.float32(0.0))
    return LaneResult(
        values=normalized,
        failure=failure,
        stats={
            "mean": mean,
            "variance": variance,
            "denominator": denominator,
            "epsilon_applied": epsilon_applied,
            "epsilon_added": bool(epsilon_f32 > np.float32(0.0)),
            row_label: lane,
        },
    )


def validate_epsilon(value: Any, *, allow_zero: bool = False) -> tuple[float | None, str | None]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None, "epsilon must be a real number"
    number = float(value)
    if not math.isfinite(number):
        return None, "epsilon must be finite"
    if number < 0.0 or (number == 0.0 and not allow_zero):
        return None, "epsilon must be positive"
    return number, None
