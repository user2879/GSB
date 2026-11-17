from __future__ import annotations

from typing import Any

import numpy as np

from .cache import CallState, lane_axis, lane_count, lane_values
from .ops import sum_float64_in_order


def _softmax_float64(values: np.ndarray, stabilize: bool) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    shifted = values - np.max(values) if stabilize else values
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        exponentials = np.exp(shifted)
        total = sum_float64_in_order(exponentials)
        return (exponentials / total).astype(np.float64)


def _layer_norm_float64(
    values: np.ndarray, epsilon: float, scale: float
) -> tuple[np.ndarray, np.float64, np.float64]:
    values = np.asarray(values, dtype=np.float64)
    mean = np.float64(sum_float64_in_order(values) / np.float64(values.size))
    deviations = values - mean
    variance = np.float64(
        sum_float64_in_order(deviations * deviations) / np.float64(values.size)
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        normalized = (deviations / np.sqrt(variance + np.float64(epsilon))) * np.float64(scale)
    return normalized.astype(np.float64), mean, variance


def compute_reference_arrays(states: list[CallState], tensor: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    comparisons: dict[str, Any] = {}
    for index, state in enumerate(states):
        if state.spec.reference_precision != "float64":
            continue
        softmax = np.zeros(state.spec.shape, dtype=np.float64)
        layernorm = np.zeros(state.spec.shape, dtype=np.float64)
        mean = np.zeros(state.spec.shape, dtype=np.float64)
        variance = np.zeros(state.spec.shape, dtype=np.float64)
        softmax_axis = lane_axis(state.spec, "softmax")
        layernorm_axis = lane_axis(state.spec, "layernorm")
        for lane in range(lane_count(state.spec, "softmax")):
            values = _softmax_float64(
                lane_values(tensor, softmax_axis, lane), state.spec.stabilize
            )
            if softmax_axis == 1:
                softmax[lane, :] = values
            else:
                softmax[:, lane] = values
        for lane in range(lane_count(state.spec, "layernorm")):
            values, lane_mean, lane_variance = _layer_norm_float64(
                lane_values(tensor, layernorm_axis, lane),
                state.spec.epsilon,
                state.spec.layernorm_scale,
            )
            if layernorm_axis == 1:
                layernorm[lane, :] = values
                mean[lane, :] = lane_mean
                variance[lane, :] = lane_variance
            else:
                layernorm[:, lane] = values
                mean[:, lane] = lane_mean
                variance[:, lane] = lane_variance
        prefix = f"call_{index}"
        arrays[f"{prefix}_softmax_float64"] = softmax
        arrays[f"{prefix}_layernorm_float64"] = layernorm
        arrays[f"{prefix}_mean_float64"] = mean
        arrays[f"{prefix}_variance_float64"] = variance

        def max_abs_difference(left: np.ndarray, right: np.ndarray) -> float | str:
            with np.errstate(invalid="ignore"):
                difference = np.abs(
                    np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
                )
            if np.all(np.isnan(difference)):
                return "NaN"
            if np.any(np.isnan(difference)):
                return "NaN"
            return float(np.max(difference))

        comparisons[state.spec.call_id] = {
            "reference_precision": "float64",
            "softmax_max_abs_difference": max_abs_difference(
                state.softmax_result, softmax
            ),
            "layernorm_max_abs_difference": max_abs_difference(
                state.layernorm_result, layernorm
            ),
            "mean_max_abs_difference": max_abs_difference(state.layernorm_mean, mean),
            "variance_max_abs_difference": max_abs_difference(
                state.layernorm_variance, variance
            ),
        }
    return arrays, comparisons
