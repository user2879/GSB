from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .deterministic import ModelParams
from .quantization import (
    INT64_MAX,
    INT64_MIN,
    dequantize_value,
    fixed_linear_trace,
    quantize_array,
)


@dataclass(frozen=True)
class FloatForward:
    layer_outputs: tuple[np.ndarray, ...]
    output: np.ndarray


@dataclass(frozen=True)
class FixedForward:
    input_q: np.ndarray
    layer_outputs: tuple[np.ndarray, ...]
    trace: tuple[dict[str, object], ...]
    output_q: np.ndarray
    output_float: np.ndarray
    input_saturations: int
    total_saturations: int


def float_forward(model: ModelParams, inputs: np.ndarray) -> FloatForward:
    current = np.asarray(inputs, dtype=np.float32)
    layer_outputs: list[np.ndarray] = []
    final_index = len(model.weights) - 1
    for index, (weight, bias) in enumerate(zip(model.weights, model.biases)):
        current = np.asarray(weight.T @ current + bias, dtype=np.float32)
        if index != final_index:
            current = np.maximum(current, np.float32(0.0))
        current = np.ascontiguousarray(current, dtype=np.float32)
        layer_outputs.append(current)
    return FloatForward(tuple(layer_outputs), layer_outputs[-1])


def quantized_model_weights(model: ModelParams, power: int) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], dict[str, int]]:
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    saturations: dict[str, int] = {}
    for index, (weight, bias) in enumerate(zip(model.weights, model.biases)):
        quantized_weight = quantize_array(weight, power)
        quantized_bias = quantize_array(bias, power)
        weights.append(quantized_weight.values)
        biases.append(quantized_bias.values)
        saturations[f"layers[{index}].weight"] = quantized_weight.saturations
        saturations[f"layers[{index}].bias"] = quantized_bias.saturations
    return tuple(weights), tuple(biases), saturations


def fixed_forward(
    model: ModelParams,
    inputs: np.ndarray,
    power: int,
    *,
    quantized_weights: tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]] | None = None,
) -> FixedForward:
    if quantized_weights is None:
        weights_q, biases_q, _ = quantized_model_weights(model, power)
    else:
        weights_q, biases_q = quantized_weights

    input_quant = quantize_array(inputs, power)
    current = input_quant.values
    layer_outputs: list[np.ndarray] = []
    trace: list[dict[str, object]] = []
    total_saturations = input_quant.saturations
    final_index = len(model.weights) - 1
    for index, (weight_q, bias_q) in enumerate(zip(weights_q, biases_q)):
        layer_trace = fixed_linear_trace(current, weight_q, bias_q, power, relu=index != final_index)
        current = np.asarray(layer_trace["output"], dtype=np.int64)
        layer_outputs.append(current)
        total_saturations += int(layer_trace["saturation_count"])
        trace.append(layer_trace)
    output_q = layer_outputs[-1]
    output_float = np.asarray([dequantize_value(int(value), power) for value in output_q], dtype=np.float64)
    return FixedForward(
        input_q=input_quant.values,
        layer_outputs=tuple(layer_outputs),
        trace=tuple(trace),
        output_q=output_q,
        output_float=output_float,
        input_saturations=input_quant.saturations,
        total_saturations=total_saturations,
    )


INT32_LIKE_BOUND = 2_147_483_648.0


def _scale_parts_float(power: int) -> tuple[float, float]:
    if power >= 0:
        return float(10**power), 1.0
    return 1.0, float(10 ** (-power))


def analyze_sample(
    model: ModelParams,
    sample: np.ndarray,
    power: int,
    *,
    quantized_weights: tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]] | None = None,
) -> dict[str, Any]:
    reference = float_forward(model, sample)
    fixed = fixed_forward(model, sample, power, quantized_weights=quantized_weights)
    true_inputs: list[np.ndarray] = []
    for index in range(len(model.weights)):
        true_inputs.append(np.asarray(sample if index == 0 else reference.layer_outputs[index - 1], dtype=np.float64))

    prior_bound = 0.0
    layer_results: list[dict[str, Any]] = []
    numerator, denominator = _scale_parts_float(power)
    for index, (weight, float_output, true_input) in enumerate(
        zip(model.weights, reference.layer_outputs, true_inputs)
    ):
        input_error_units = prior_bound + 0.5
        weight_abs = np.abs(weight.astype(np.float64))
        input_abs = np.abs(true_input)
        cross = weight_abs * input_error_units
        weight_side = (0.5 * input_abs)[:, None]
        product_side = (input_error_units * 0.5) * (denominator / numerator)
        raw_bound = float(np.sum(cross + weight_side + product_side) + 1.0)
        bound_int = int(math.ceil(raw_bound))

        fixed_layer = fixed.layer_outputs[index]
        ideal = quantize_array(float_output, power, lower=INT64_MIN, upper=INT64_MAX)
        diff = np.abs(fixed_layer.astype(np.int64) - ideal.values.astype(np.int64))
        observed_max = int(np.max(diff)) if diff.size else 0
        absolute_sum = int(np.sum(diff, dtype=np.int64)) if diff.size else 0
        saturation_count = int(fixed.trace[index]["saturation_count"])
        bound_valid = saturation_count == 0
        if not bound_valid:
            bound_int = INT64_MAX
        elif bound_int < observed_max:
            bound_valid = False
            bound_int = observed_max + 1
        layer_results.append(
            {
                "layer": index,
                "shape": list(fixed_layer.shape),
                "observed_max_abs_error_int": observed_max,
                "observed_absolute_error_sum_int": absolute_sum,
                "ideal_reference_q": ideal.as_list,
                "fixed_q": [int(value) for value in fixed_layer.tolist()],
                "error_bound_int": bound_int,
                "error_bound_valid": bound_valid,
                "rounding_quantization_limit_int_milli": 500,
                "rounding_quantization_limit_units": 0.5,
                "saturation_count": saturation_count,
            }
        )
        prior_bound = float(bound_int) if bound_int != INT64_MAX else INT32_LIKE_BOUND

    dominant = min(
        layer_results,
        key=lambda item: (-item["observed_max_abs_error_int"], item["layer"]),
    )
    total_abs = sum(item["observed_absolute_error_sum_int"] for item in layer_results)
    cumulative = 0
    for item in layer_results:
        cumulative += item["observed_absolute_error_sum_int"]
        item["error_share_ppm"] = int(round(1_000_000 * item["observed_absolute_error_sum_int"] / total_abs)) if total_abs else 0
        item["cumulative_error_share_ppm"] = int(round(1_000_000 * cumulative / total_abs)) if total_abs else 0

    output_diff = np.abs(fixed.output_q.astype(np.int64) - quantize_array(reference.output, power, lower=INT64_MIN, upper=INT64_MAX).values)
    return {
        "float_output": [float(value) for value in reference.output.tolist()],
        "fixed_output_q": [int(value) for value in fixed.output_q.tolist()],
        "fixed_output_float": [float(value) for value in fixed.output_float.tolist()],
        "output_max_abs_error_int": int(np.max(output_diff)) if output_diff.size else 0,
        "dominant_layer": int(dominant["layer"]),
        "layer_diagnostics": layer_results,
        "input_saturations": fixed.input_saturations,
        "activation_saturations": fixed.total_saturations - fixed.input_saturations,
        "total_saturations": fixed.total_saturations,
        "output_quantization_limit_int_milli": 500,
    }


def summarize_layer_attribution(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not samples:
        return []
    layer_count = len(samples[0]["layer_diagnostics"])
    aggregate: list[dict[str, Any]] = []
    total_abs = sum(
        int(layer["observed_absolute_error_sum_int"])
        for sample in samples
        for layer in sample["layer_diagnostics"]
    )
    cumulative = 0
    for layer_index in range(layer_count):
        observed_max = max(
            int(sample["layer_diagnostics"][layer_index]["observed_max_abs_error_int"]) for sample in samples
        )
        observed_sum = sum(
            int(sample["layer_diagnostics"][layer_index]["observed_absolute_error_sum_int"]) for sample in samples
        )
        bound_max = max(
            int(sample["layer_diagnostics"][layer_index]["error_bound_int"]) for sample in samples
        )
        saturations = sum(
            int(sample["layer_diagnostics"][layer_index]["saturation_count"]) for sample in samples
        )
        valid = all(bool(sample["layer_diagnostics"][layer_index]["error_bound_valid"]) for sample in samples)
        cumulative += observed_sum
        aggregate.append(
            {
                "layer": layer_index,
                "observed_max_abs_error_int": observed_max,
                "observed_absolute_error_sum_int": observed_sum,
                "error_bound_int": bound_max,
                "error_bound_valid": valid,
                "saturation_count": saturations,
                "error_share_ppm": int(round(1_000_000 * observed_sum / total_abs)) if total_abs else 0,
                "cumulative_error_share_ppm": int(round(1_000_000 * cumulative / total_abs)) if total_abs else 0,
                "rounding_quantization_limit_int_milli": 500,
            }
        )

    dominant_counts: dict[int, int] = {}
    for sample in samples:
        layer = int(sample["dominant_layer"])
        dominant_counts[layer] = dominant_counts.get(layer, 0) + 1
    for item in aggregate:
        item["dominant_sample_count"] = dominant_counts.get(item["layer"], 0)
    return aggregate


def output_digest(outputs: Iterable[np.ndarray], precision: str) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps({"precision": precision, "dtype": "float32" if precision == "float32" else "int64"}, sort_keys=True).encode("utf-8")
    )
    for output in outputs:
        if precision == "float32":
            normalized = np.ascontiguousarray(output, dtype="<f4")
        else:
            normalized = np.ascontiguousarray(output, dtype="<i8")
        digest.update(b"\0")
        digest.update(normalized.tobytes(order="C"))
    return digest.hexdigest()


def sample_input_digest(sample: np.ndarray) -> str:
    normalized = np.ascontiguousarray(sample, dtype="<f4")
    return hashlib.sha256(normalized.tobytes(order="C")).hexdigest()
