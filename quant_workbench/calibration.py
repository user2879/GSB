"""Calibration-sample error evaluation and deterministic soft-label correction."""

from __future__ import annotations

import math
import threading
from typing import Any

import numpy as np

from .errors import E_LAYER_INVALID, E_SHAPE_MISMATCH, Issue
from .models import CalibrationSet, WeightLayer
from .quantization import DeadlineExceeded, _check_budget, dequantized_matrix
from .util import clean_float, stage_cache_key


def _forward(weights: np.ndarray, activation: np.ndarray) -> np.ndarray:
    if weights.ndim == 1:
        return np.asarray([np.dot(weights.reshape(-1), activation)], dtype=np.float64)
    rows = weights.reshape(-1, weights.shape[-1])
    return np.asarray(rows @ activation, dtype=np.float64).reshape(-1)


def _explicit_sum(values: list[float], order: list[int]) -> float:
    total = 0.0
    for index in order:
        total += values[index]
    return total


def _stats(deviations: list[float]) -> tuple[float, float, float]:
    if not deviations:
        return 0.0, 0.0, 0.0
    absolute_total = 0.0
    squared_total = 0.0
    maximum = 0.0
    for value in deviations:
        absolute = abs(value)
        absolute_total += absolute
        squared_total += value * value
        if absolute > maximum:
            maximum = absolute
    count = len(deviations)
    return maximum, squared_total / count, absolute_total / count


def evaluate_layer(
    weight: WeightLayer,
    quantized: dict[str, Any],
    calibration: CalibrationSet,
    *,
    temperature: float,
    command_index: int | None = None,
    deadline: float | None = None,
    cancel_event: threading.Event | None = None,
) -> tuple[dict[str, Any] | None, list[Issue]]:
    """Evaluate one layer against every calibration sample.

    A sample's correction for each output element is
    ``-(quantized-original) * softmax(original / temperature)``.  The correction
    addends are accumulated in ascending output-index order, and that order is
    persisted in the result as an auditable trace.
    """

    issues: list[Issue] = []
    if temperature <= 0 or not math.isfinite(temperature):
        return None, [
            Issue(
                "E_CONFIG_INVALID",
                f"invalid distillation temperature {temperature!r}",
                command_index=command_index,
                layer=weight.layer,
            )
        ]
    reconstructed_weights = dequantized_matrix(quantized)
    sample_results: list[dict[str, Any]] = []
    all_pre_deviations: list[float] = []
    all_post_deviations: list[float] = []

    for sample in calibration.samples:
        _check_budget(deadline, cancel_event)
        activation = sample.activations.get(weight.layer)
        if activation is None:
            issues.append(
                Issue(
                    E_LAYER_INVALID,
                    f"calibration sample {sample.sample_id!r} has no activation for layer {weight.layer!r}",
                    command_index=command_index,
                    line_number=sample.line_number,
                    layer=weight.layer,
                    sample_id=sample.sample_id,
                )
            )
            continue
        expected = int(weight.shape[-1])
        if int(activation.size) != expected:
            issues.append(
                Issue(
                    E_SHAPE_MISMATCH,
                    f"calibration sample {sample.sample_id!r} activation length {activation.size} for {weight.layer!r} does not match {expected}",
                    command_index=command_index,
                    line_number=sample.line_number,
                    layer=weight.layer,
                    sample_id=sample.sample_id,
                    details={"expected_length": expected, "actual_length": int(activation.size)},
                )
            )
            continue
        try:
            original_output = _forward(weight.weights, activation)
            reconstructed_output = _forward(reconstructed_weights, activation)
        except ValueError as exc:
            issues.append(
                Issue(
                    E_SHAPE_MISMATCH,
                    f"calibration forward failed for {weight.layer!r}: {exc}",
                    command_index=command_index,
                    line_number=sample.line_number,
                    layer=weight.layer,
                    sample_id=sample.sample_id,
                )
            )
            continue

        count = int(original_output.size)
        maximum = float(np.max(original_output))
        exponentials = np.exp((original_output - maximum) / float(temperature))
        denominator = float(np.sum(exponentials))
        soft_weights = exponentials / denominator
        pre_deviation = reconstructed_output - original_output
        correction_addends = [
            float(-pre_deviation[index] * soft_weights[index]) for index in range(count)
        ]
        accumulation_order = list(range(count))
        trace: list[dict[str, Any]] = []
        running = 0.0
        for index in accumulation_order:
            running += correction_addends[index]
            trace.append(
                {
                    "order": index,
                    "addend": clean_float(correction_addends[index]),
                    "running_total": clean_float(running),
                }
            )
        correction_total = _explicit_sum(correction_addends, accumulation_order)
        corrected_output = np.zeros(count, dtype=np.float64)
        for index in accumulation_order:
            corrected_output[index] = reconstructed_output[index] + correction_addends[index]
        post_deviation = corrected_output - original_output

        pre_values = [float(value) for value in pre_deviation.tolist()]
        post_values = [float(value) for value in post_deviation.tolist()]
        all_pre_deviations.extend(pre_values)
        all_post_deviations.extend(post_values)
        pre_max, pre_mse, pre_mae = _stats(pre_values)
        post_max, post_mse, post_mae = _stats(post_values)
        sample_results.append(
            {
                "sample_id": sample.sample_id,
                "line_number": sample.line_number,
                "original_output": [clean_float(value) for value in original_output.tolist()],
                "quantized_output": [clean_float(value) for value in reconstructed_output.tolist()],
                "pre_deviation": [clean_float(value) for value in pre_values],
                "soft_label_weights": [clean_float(value) for value in soft_weights.tolist()],
                "correction_addends": [clean_float(value) for value in correction_addends],
                "accumulation_order": accumulation_order,
                "correction_trace": trace,
                "correction_total": clean_float(correction_total),
                "corrected_output": [clean_float(value) for value in corrected_output.tolist()],
                "post_deviation": [clean_float(value) for value in post_values],
                "pre_max_abs_error": clean_float(pre_max),
                "pre_mse": clean_float(pre_mse),
                "pre_mean_abs_error": clean_float(pre_mae),
                "post_max_abs_error": clean_float(post_max),
                "post_mse": clean_float(post_mse),
                "post_mean_abs_error": clean_float(post_mae),
            }
        )

    if not sample_results:
        return None, issues

    pre_max, pre_mse, pre_mae = _stats(all_pre_deviations)
    post_max, post_mse, post_mae = _stats(all_post_deviations)
    cache_key = stage_cache_key(
        "evaluation",
        quantized["quantization_cache_key"],
        calibration.content_fingerprint,
        temperature,
    )
    payload = {
        "layer": weight.layer,
        "version": weight.version,
        "order": weight.order,
        "content_fingerprint": weight.content_fingerprint,
        "quantization_cache_key": quantized["quantization_cache_key"],
        "calibration_fingerprint": calibration.content_fingerprint,
        "calibration_sample_ids": calibration.sample_ids,
        "temperature": float(temperature),
        "cache_key": cache_key,
        "evaluation_cache_key": cache_key,
        "config_fields": ["distillation_temperature", "calibration_samples"],
        "sample_order": calibration.sample_ids,
        "samples": sample_results,
        "sample_count": len(sample_results),
        "max_abs_error": clean_float(pre_max),
        "mse": clean_float(pre_mse),
        "mean_abs_error": clean_float(pre_mae),
        "corrected_max_abs_error": clean_float(post_max),
        "corrected_mse": clean_float(post_mse),
        "corrected_mean_abs_error": clean_float(post_mae),
        "correction_accumulation_order": calibration.sample_ids,
    }
    return payload, issues


def deployment_entry(
    weight: WeightLayer,
    quantized: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    cache_key = stage_cache_key(
        "deployment",
        quantized["quantization_cache_key"],
        evaluation["evaluation_cache_key"],
        weight.content_fingerprint,
    )
    return {
        "layer": weight.layer,
        "version": weight.version,
        "order": weight.order,
        "artifact_name": f"{weight.layer}.q{quantized['parameters']['bits']}.json",
        "shape": list(weight.shape),
        "point_count": weight.point_count,
        "bits": quantized["parameters"]["bits"],
        "group_size": quantized["parameters"]["group_size"],
        "fractional_bits": quantized["parameters"]["fractional_bits"],
        "scheme": quantized["parameters"]["scheme"],
        "quantization_cache_key": quantized["quantization_cache_key"],
        "evaluation_cache_key": evaluation["evaluation_cache_key"],
        "content_fingerprint": weight.content_fingerprint,
        "calibration_fingerprint": evaluation["calibration_fingerprint"],
        "calibration_sample_ids": evaluation["calibration_sample_ids"],
        "cache_key": cache_key,
        "deployment_cache_key": cache_key,
        "artifact_version": cache_key[:16],
    }
