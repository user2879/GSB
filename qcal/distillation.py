"""Calibration-set error evaluation and deterministic light distillation."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .codes import E_INTERNAL, E_SAMPLE_PARSE
from .io_utils import combined_fingerprint, sha256_json
from .models import FailureRecord, QuantizationResult, WeightPack


class DistillationError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def softmax_with_temperature(values: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise DistillationError(E_SAMPLE_PARSE, "distillation temperature must be positive and finite")
    scaled = values / temperature
    shifted = scaled - float(np.max(scaled))
    exponentials = np.exp(shifted)
    denominator = float(np.sum(exponentials))
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise DistillationError(E_SAMPLE_PARSE, "softmax denominator is invalid")
    return exponentials / denominator


def distill_layer(
    pack: WeightPack,
    quantization: QuantizationResult,
    samples: list[dict[str, Any]],
    distill_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    temperature = float(distill_cfg["temperature"])
    soft_label_weight = float(distill_cfg["soft_label_weight"])
    if not math.isfinite(soft_label_weight) or soft_label_weight < 0.0:
        raise DistillationError(E_SAMPLE_PARSE, "soft_label_weight must be finite and non-negative")
    if len(pack.shape) != 2:
        raise DistillationError(E_SAMPLE_PARSE, "distillation requires a 2-D matrix layer")
    input_dimension = int(pack.shape[1])
    original = np.asarray(pack.weights, dtype=np.float64)
    dequantized = np.asarray(quantization.dequantized, dtype=np.float64).reshape(pack.shape)
    records: list[dict[str, Any]] = []
    for sample in sorted(samples, key=lambda item: str(item["id"])):
        input_vector = np.asarray(sample["input"], dtype=np.float64)
        if input_vector.size != input_dimension:
            raise DistillationError(
                E_SAMPLE_PARSE,
                f"sample {sample['id']} has input width {input_vector.size}, expected {input_dimension}",
            )
        original_output = original @ input_vector
        quantized_output = dequantized @ input_vector
        soft_labels = softmax_with_temperature(original_output, temperature)
        confidence = float(np.max(soft_labels))
        correction = soft_label_weight * confidence * (original_output - quantized_output)
        corrected_output = quantized_output + correction
        before_error = original_output - quantized_output
        after_error = original_output - corrected_output
        reference = sample.get("reference", {})
        supplied_reference = reference.get(pack.layer) if isinstance(reference, dict) else None
        reference_match = None
        if isinstance(supplied_reference, list) and len(supplied_reference) == original_output.size:
            supplied = np.asarray(supplied_reference, dtype=np.float64)
            reference_match = bool(np.allclose(supplied, original_output, rtol=0.0, atol=1e-12))
        records.append(
            {
                "sample_id": str(sample["id"]),
                "layer": pack.layer,
                "temperature": temperature,
                "soft_label_weight": soft_label_weight,
                "soft_labels": [float(value) for value in soft_labels.tolist()],
                "confidence": confidence,
                "original_output": [float(value) for value in original_output.tolist()],
                "quantized_output": [float(value) for value in quantized_output.tolist()],
                "correction": [float(value) for value in correction.tolist()],
                "corrected_output": [float(value) for value in corrected_output.tolist()],
                "before_max_abs_deviation": float(np.max(np.abs(before_error))),
                "after_max_abs_deviation": float(np.max(np.abs(after_error))),
                "before_mse": float(np.mean(before_error * before_error)),
                "after_mse": float(np.mean(after_error * after_error)),
                "reference_supplied_match": reference_match,
            }
        )
    records.sort(key=lambda item: (item["sample_id"], item["layer"]))
    return records


def distill_param_fingerprint(distill_cfg: dict[str, Any]) -> str:
    return sha256_json(
        {
            "temperature": float(distill_cfg["temperature"]),
            "soft_label_weight": float(distill_cfg["soft_label_weight"]),
            "order": "sample_id_then_layer",
        }
    )


def distill_cache_key(
    quantization_key: str,
    sample_fingerprint: str,
    parameter_fingerprint: str,
    exchange_fingerprint: str,
) -> str:
    return combined_fingerprint(
        "distillation_correction",
        quantization_key,
        sample_fingerprint,
        parameter_fingerprint,
        exchange_fingerprint,
    )


def distillation_failure(layer: str, error: Exception) -> FailureRecord:
    code = getattr(error, "code", E_INTERNAL)
    return FailureRecord(
        layer=layer,
        stage="distillation_correction",
        code=str(code),
        message=str(error),
    )
