"""Lightweight deterministic distillation-style correction.

The reference network is a fixed three-layer MLP (ReLU between layers).
Quantized forward passes reuse per-layer dequantized weights. Correction adds
a per-output-channel bias and a scalar gain per layer, updated in a fixed
order (bias then gain) for a configured number of steps. The updates are pure
floating-point arithmetic with no thread-dependent reduction, so serial and
parallel executions are bit-identical.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np


@dataclass
class LayerCorrection:
    layer: str
    bias: list[float]
    gain: float
    before_deviation: float
    after_deviation: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DistillReport:
    temperature: float
    steps: int
    layer_reports: list[dict[str, Any]]
    sample_deviations_before: list[float]
    sample_deviations_after: list[float]
    flagged_samples: list[dict[str, Any]]
    referenced_sample_indices: list[int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _forward(weights: list[np.ndarray], inputs: np.ndarray,
             biases: list[np.ndarray], gains: list[float]) -> list[np.ndarray]:
    activations = [inputs]
    hidden = inputs
    for idx, weight in enumerate(weights):
        hidden = hidden @ weight
        hidden = hidden * gains[idx] + biases[idx]
        if idx < len(weights) - 1:
            hidden = _relu(hidden)
        activations.append(hidden)
    return activations


def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    scaled = x / temperature
    scaled = scaled - np.max(scaled, axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / np.sum(exp, axis=1, keepdims=True)


def correct(weights_full: list[np.ndarray],
            weights_quant: list[np.ndarray],
            inputs: np.ndarray,
            reference_outputs: np.ndarray,
            layer_names: list[str],
            temperature: float,
            steps: int,
            soft_label_weight: float,
            learning_rate: float,
            saturation_counts: list[int],
            ) -> DistillReport:
    layer_count = len(weights_full)
    biases = [np.zeros((w.shape[1],), dtype=np.float64) for w in weights_full]
    gains = [1.0 for _ in weights_full]

    target_soft = _softmax(reference_outputs, temperature)
    quant_activations = _forward(weights_quant, inputs, biases, gains)
    before_output = quant_activations[-1]
    before_dev = np.linalg.norm(before_output - reference_outputs, axis=1)

    layer_before: list[float] = []
    full_activations = _forward(
        weights_full, inputs,
        [np.zeros_like(b) for b in biases], [1.0] * layer_count,
    )
    for idx in range(layer_count):
        layer_before.append(float(np.mean(
            np.abs(quant_activations[idx + 1] - full_activations[idx + 1]))))

    for _ in range(steps):
        activations = _forward(weights_quant, inputs, biases, gains)
        quant_soft = _softmax(activations[-1], temperature)
        final_signal = (
            soft_label_weight * (quant_soft - target_soft)
            + (1.0 - soft_label_weight)
            * (activations[-1] - reference_outputs)
        )
        for idx in range(layer_count):
            layer_input = activations[idx]
            if idx < layer_count - 1:
                layer_input = _relu(layer_input)
            layer_output = activations[idx + 1]
            # Each hidden layer is pulled toward its full-precision activation;
            # the final layer additionally carries the soft-label signal.
            layer_target = full_activations[idx + 1]
            if idx == layer_count - 1:
                signal = (
                    soft_label_weight * (quant_soft - target_soft)
                    + (1.0 - soft_label_weight)
                    * (layer_output - layer_target)
                )
            else:
                signal = layer_output - layer_target
            # Fixed update order: bias first, then gain; deterministic means.
            biases[idx] = biases[idx] - learning_rate * np.mean(signal, axis=0)
            denom = max(float(np.mean(layer_output ** 2)), 1e-12)
            gains[idx] = gains[idx] - learning_rate * float(
                np.mean(signal * layer_output)) / denom

    after_activations = _forward(weights_quant, inputs, biases, gains)
    after_output = after_activations[-1]
    after_dev = np.linalg.norm(after_output - reference_outputs, axis=1)

    layer_reports: list[dict[str, Any]] = []
    for idx, name in enumerate(layer_names):
        layer_after = float(np.mean(
            np.abs(after_activations[idx + 1] - full_activations[idx + 1])))
        correction = LayerCorrection(
            layer=name,
            bias=[float(x) for x in biases[idx].tolist()],
            gain=float(gains[idx]),
            before_deviation=layer_before[idx],
            after_deviation=layer_after,
        )
        layer_reports.append(correction.to_dict())

    flagged: list[dict[str, Any]] = []
    for sample_idx in np.where(after_dev > before_dev)[0]:
        idx = int(sample_idx)
        if sum(saturation_counts) > 0:
            reason = "saturation_concentration"
        elif temperature >= max(abs(np.mean(reference_outputs)), 1.0) * 2:
            reason = "temperature_setting"
        else:
            reason = "weight_distribution_skew"
        flagged.append({
            "sample_index": idx,
            "before": float(before_dev[idx]),
            "after": float(after_dev[idx]),
            "reason": reason,
        })

    return DistillReport(
        temperature=float(temperature),
        steps=int(steps),
        layer_reports=layer_reports,
        sample_deviations_before=[float(x) for x in before_dev.tolist()],
        sample_deviations_after=[float(x) for x in after_dev.tolist()],
        flagged_samples=flagged,
        referenced_sample_indices=list(range(inputs.shape[0])),
    )
