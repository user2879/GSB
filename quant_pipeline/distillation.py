"""Calibration-set output comparison and temperature-scaled soft-label correction."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import DistillationConfig
from .errors import E_LAYER_DISTILLATION, PipelineError
from .inputs import LayerCalibrationView, WeightLayer
from .util import softmax


def _apply(shape: tuple[int, ...], weights: np.ndarray, vector: np.ndarray) -> np.ndarray:
    if len(shape) == 1:
        return np.asarray(weights, dtype=np.float64).reshape(-1) * vector
    matrix = np.asarray(weights, dtype=np.float64).reshape((-1, shape[-1]))
    return matrix @ vector


def distill_layer(
    layer: WeightLayer,
    dequantized_weights: np.ndarray,
    calibration_view: LayerCalibrationView,
    config: DistillationConfig,
) -> dict[str, Any]:
    try:
        if not np.isfinite(config.temperature) or config.temperature <= 0:
            raise ValueError("distillation temperature must be positive and finite")
        if not (0.0 <= config.soft_label_weight <= 1.0):
            raise ValueError("soft_label_weight must lie in [0, 1]")
        original_weights = np.asarray(layer.weights, dtype=np.float64).reshape(-1)
        quantized_weights = np.asarray(dequantized_weights, dtype=np.float64).reshape(-1)
        samples: list[dict[str, Any]] = []
        bias_before_values: list[float] = []
        bias_after_values: list[float] = []
        correction_values: list[float] = []

        for sample_id, input_vector, reference in zip(
            calibration_view.sample_ids,
            calibration_view.inputs,
            calibration_view.references,
            strict=True,
        ):
            original_output = _apply(layer.shape, original_weights, input_vector)
            quantized_output = _apply(layer.shape, quantized_weights, input_vector)
            soft_original = softmax(original_output / config.temperature)
            soft_quantized = softmax(quantized_output / config.temperature)
            soft_reference = softmax(np.asarray(reference, dtype=np.float64) / config.temperature)
            correction = (
                config.soft_label_weight
                * config.temperature
                * (soft_reference - soft_quantized)
            )
            corrected_output = quantized_output + correction
            bias_before = quantized_output - original_output
            bias_after = corrected_output - original_output
            bias_before_values.extend(np.abs(bias_before).tolist())
            bias_after_values.extend(np.abs(bias_after).tolist())
            correction_values.extend(np.abs(correction).tolist())
            samples.append(
                {
                    "sample_id": sample_id,
                    "original_output": original_output.tolist(),
                    "quantized_output": quantized_output.tolist(),
                    "reference_output": np.asarray(reference, dtype=np.float64).tolist(),
                    "corrected_output": corrected_output.tolist(),
                    "correction": correction.tolist(),
                    "bias_before_to_original": bias_before.tolist(),
                    "bias_after_to_original": bias_after.tolist(),
                    "error_before_to_reference": (quantized_output - reference).tolist(),
                    "error_after_to_reference": (corrected_output - reference).tolist(),
                    "soft_original": soft_original.tolist(),
                    "soft_quantized": soft_quantized.tolist(),
                    "soft_reference": soft_reference.tolist(),
                    "mean_absolute_bias_before": float(np.mean(np.abs(bias_before))),
                    "mean_absolute_bias_after": float(np.mean(np.abs(bias_after))),
                    "mean_absolute_correction": float(np.mean(np.abs(correction))),
                }
            )

        return {
            "stage": "distillation",
            "layer": layer.name,
            "temperature": config.temperature,
            "soft_label_weight": config.soft_label_weight,
            "correction_rule": "quantized + soft_label_weight * temperature * (softmax(reference/temperature) - softmax(quantized/temperature))",
            "sample_ids": list(calibration_view.sample_ids),
            "samples": samples,
            "summary": {
                "sample_count": len(samples),
                "mean_absolute_bias_before": float(np.mean(bias_before_values))
                if bias_before_values
                else 0.0,
                "mean_absolute_bias_after": float(np.mean(bias_after_values))
                if bias_after_values
                else 0.0,
                "max_absolute_bias_before": float(np.max(bias_before_values))
                if bias_before_values
                else 0.0,
                "max_absolute_bias_after": float(np.max(bias_after_values))
                if bias_after_values
                else 0.0,
                "mean_absolute_correction": float(np.mean(correction_values))
                if correction_values
                else 0.0,
            },
        }
    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError(
            E_LAYER_DISTILLATION,
            f"distillation correction failed for layer {layer.name!r}: {exc}",
            stage="distillation",
            layer=layer.name,
            exception_type=type(exc).__name__,
        ) from exc

