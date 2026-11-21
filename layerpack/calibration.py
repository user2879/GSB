"""Deterministic calibration evaluation and soft-label distillation correction."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from .config import CompressionConfig
from .fingerprint import (
    normalize_float,
    ordered_dot,
    ordered_error_stats,
    ordered_sum,
    sha256_json,
)
from .model import CalibrationSample, CalibrationResult, Layer, QuantResult


def compute_calibration_key(
    *,
    layer: Layer,
    quant: QuantResult,
    sample_fingerprints: Iterable[dict[str, str]],
    upstream_sample_fingerprints: Iterable[dict[str, str]],
    upstream_dequant_keys: Iterable[str],
    config: CompressionConfig,
) -> str:
    return sha256_json(
        {
            "kind": "calibration-v1",
            "layer": layer.name,
            "dequant_key": quant.dequant_key,
            "dequantized_sha256": quant.dequantized_sha256,
            "samples": list(sample_fingerprints),
            "upstream_samples": list(upstream_sample_fingerprints),
            "upstream_dequant_keys": list(upstream_dequant_keys),
            "distillation": {
                "temperature": config.distillation_temperature,
                "soft_label_weight": config.soft_label_weight,
                "correction_gain": config.correction_gain,
            },
        }
    )


def _layer_output(weights: np.ndarray, sample_input: tuple[float, ...]) -> list[float]:
    values = np.asarray(sample_input, dtype=np.float64)
    if weights.ndim == 1:
        return [ordered_dot(weights, values)]
    rows = weights.reshape(-1, weights.shape[-1])
    output: list[float] = []
    for row_index in range(rows.shape[0]):
        output.append(ordered_dot(rows[row_index], values))
    return output


def _softmax(values: list[float], temperature: float) -> list[float]:
    scaled = [float(value) / temperature for value in values]
    maximum = scaled[0]
    for item in scaled[1:]:
        if item > maximum:
            maximum = item
    exponentials = [math.exp(item - maximum) for item in scaled]
    denominator = ordered_sum(exponentials)
    return [item / denominator for item in exponentials]


def evaluate_layer(
    *,
    layer: Layer,
    quant: QuantResult,
    samples: list[CalibrationSample],
    config: CompressionConfig,
    key: str | None = None,
) -> CalibrationResult:
    """Evaluate original-vs-dequantized outputs and apply soft-label correction."""

    applicable = sorted(
        [sample for sample in samples if sample.applies_to(layer.name)],
        key=lambda sample: sample.sample_id,
    )
    sample_results: list[dict[str, Any]] = []
    temperature = config.distillation_temperature
    configured_weight = config.soft_label_weight
    gain = config.correction_gain
    all_disabled = True

    if not applicable:
        empty = {
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "sum_squared_error": 0.0,
            "mse": 0.0,
            "element_count": 0,
        }
        summary = {
            "sample_count": 0,
            "sample_ids": [],
            "before": dict(empty),
            "after": dict(empty),
            "correction": dict(empty),
            "mse_improvement": 0.0,
            "soft_label_disabled": True,
            "soft_label_disabled_reason": "no_applicable_samples",
            "temperature": temperature,
            "soft_label_weight": configured_weight,
            "correction_gain": gain,
            "operation_order": "sample_id_then_original_c_order",
        }
        if key is None:
            key = sha256_json(
                {"kind": "calibration-empty-v1", "layer": layer.name}
            )
        return CalibrationResult(
            layer_name=layer.name,
            samples=[],
            summary=summary,
            key=key,
        )

    for sample in applicable:
        original = _layer_output(layer.weights, sample.inputs)
        quantized = _layer_output(quant.dequantized, sample.inputs)
        before = ordered_error_stats(
            np.asarray(original, dtype=np.float64)
            - np.asarray(quantized, dtype=np.float64)
        )
        probabilities = _softmax(original, temperature)
        applied_weights = [configured_weight * value for value in probabilities]
        disabled = configured_weight == 0.0 or all(
            value == 0.0 for value in applied_weights
        )
        if disabled:
            corrected = list(quantized)
            disabled_reason = "soft_label_weight_zero"
        else:
            all_disabled = False
            corrected = [
                quantized[index]
                + gain
                * applied_weights[index]
                * (original[index] - quantized[index])
                for index in range(len(original))
            ]
            disabled_reason = None

        correction = [
            corrected[index] - quantized[index] for index in range(len(original))
        ]
        after = ordered_error_stats(
            np.asarray(original, dtype=np.float64)
            - np.asarray(corrected, dtype=np.float64)
        )
        correction_stats = ordered_error_stats(np.asarray(correction, dtype=np.float64))
        sample_results.append(
            {
                "sample_id": sample.sample_id,
                "before": before,
                "after": after,
                "correction": correction_stats,
                "soft_label_weight_mean": normalize_float(
                    ordered_sum(applied_weights) / len(applied_weights)
                ),
                "soft_label_weight_max": normalize_float(max(applied_weights)),
                "correction_disabled": disabled,
                "disabled_reason": disabled_reason,
                "output_element_count": len(original),
            }
        )

    if configured_weight == 0.0:
        all_disabled = True

    before_sse = ordered_sum([float(item["before"]["sum_squared_error"]) for item in sample_results])
    after_sse = ordered_sum([float(item["after"]["sum_squared_error"]) for item in sample_results])
    correction_sse = ordered_sum(
        [float(item["correction"]["sum_squared_error"]) for item in sample_results]
    )
    before_abs = ordered_sum(
        [float(item["before"]["mean_abs_error"]) * int(item["before"]["element_count"]) for item in sample_results]
    )
    after_abs = ordered_sum(
        [float(item["after"]["mean_abs_error"]) * int(item["after"]["element_count"]) for item in sample_results]
    )
    correction_abs = ordered_sum(
        [
            float(item["correction"]["mean_abs_error"])
            * int(item["correction"]["element_count"])
            for item in sample_results
        ]
    )
    element_count = sum(int(item["before"]["element_count"]) for item in sample_results)
    maximum_before = 0.0
    maximum_after = 0.0
    maximum_correction = 0.0
    for item in sample_results:
        maximum_before = max(maximum_before, float(item["before"]["max_abs_error"]))
        maximum_after = max(maximum_after, float(item["after"]["max_abs_error"]))
        maximum_correction = max(
            maximum_correction, float(item["correction"]["max_abs_error"])
        )

    summary: dict[str, Any] = {
        "sample_count": len(sample_results),
        "sample_ids": [item["sample_id"] for item in sample_results],
        "before": {
            "max_abs_error": normalize_float(maximum_before),
            "mean_abs_error": normalize_float(before_abs / element_count)
            if element_count
            else 0.0,
            "sum_squared_error": normalize_float(before_sse),
            "mse": normalize_float(before_sse / element_count) if element_count else 0.0,
            "element_count": element_count,
        },
        "after": {
            "max_abs_error": normalize_float(maximum_after),
            "mean_abs_error": normalize_float(after_abs / element_count)
            if element_count
            else 0.0,
            "sum_squared_error": normalize_float(after_sse),
            "mse": normalize_float(after_sse / element_count) if element_count else 0.0,
            "element_count": element_count,
        },
        "correction": {
            "max_abs_error": normalize_float(maximum_correction),
            "mean_abs_error": normalize_float(correction_abs / element_count)
            if element_count
            else 0.0,
            "sum_squared_error": normalize_float(correction_sse),
            "mse": normalize_float(correction_sse / element_count)
            if element_count
            else 0.0,
            "element_count": element_count,
        },
        "mse_improvement": normalize_float(
            (before_sse - after_sse) / element_count if element_count else 0.0
        ),
        "soft_label_disabled": all_disabled,
        "soft_label_disabled_reason": "soft_label_weight_zero" if all_disabled else None,
        "temperature": temperature,
        "soft_label_weight": configured_weight,
        "correction_gain": gain,
        "operation_order": "sample_id_then_original_c_order",
    }
    if key is None:
        key = sha256_json(
            {
                "kind": "calibration-result-placeholder",
                "layer": layer.name,
                "samples": summary["sample_ids"],
            }
        )
    return CalibrationResult(
        layer_name=layer.name,
        samples=sample_results,
        summary=summary,
        key=key,
    )
