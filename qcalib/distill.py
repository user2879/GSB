"""Calibration-driven lightweight distillation correction and stage attribution."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .hashing import fingerprint
from .models import DistillConfig, LayerQuantConfig, WeightLayer

STAGE_CALIBRATION = "quantile_calibration"
STAGE_REQUANT = "fixed_point_requant"
STAGE_SATURATION = "saturation_truncation"
STAGE_DISTILL = "distillation_correction"
STAGES = (STAGE_CALIBRATION, STAGE_REQUANT, STAGE_SATURATION, STAGE_DISTILL)


@dataclass
class DistillResult:
    pre_deviation: float
    post_deviation: float
    delta_deviation: float
    increased: bool
    increase_judgement: dict[str, float]
    correction_norm: float
    correction_max_abs: float
    corrected_weights: list[float]
    per_sample_pre: list[float]
    per_sample_post: list[float]
    fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pre_deviation": self.pre_deviation,
            "post_deviation": self.post_deviation,
            "delta_deviation": self.delta_deviation,
            "increased": self.increased,
            "increase_judgement": self.increase_judgement,
            "correction_norm": self.correction_norm,
            "correction_max_abs": self.correction_max_abs,
            "corrected_weights": self.corrected_weights,
            "per_sample_pre": self.per_sample_pre,
            "per_sample_post": self.per_sample_post,
            "fingerprint": self.fingerprint,
        }


def forward(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    # Deterministic matmul; layer weights are (out_dim, in_dim).
    squeezed = weights.ndim == 1
    matrix = weights.reshape(1, -1) if squeezed else weights
    output = inputs @ matrix.T
    return output[:, 0] if squeezed else output


def output_deviation(predicted: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean((predicted - reference) ** 2)))


def per_sample_deviation(predicted: np.ndarray,
                         reference: np.ndarray) -> list[float]:
    return [float(np.sqrt(np.mean((predicted[i] - reference[i]) ** 2)))
            for i in range(predicted.shape[0])]


def softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    squeeze = values.ndim == 1
    values = np.atleast_2d(values)
    scaled = values / temperature
    scaled -= np.max(scaled, axis=1, keepdims=True)
    exp_values = np.exp(scaled)
    result = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    return result[0] if squeeze else result


def _soft_grad(inputs: np.ndarray, student_out: np.ndarray,
               teacher_out: np.ndarray, temperature: float) -> np.ndarray:
    soft_s = softmax(student_out, temperature)
    soft_t = softmax(teacher_out, temperature)
    count = inputs.shape[0]
    return (temperature ** 2) * ((soft_s - soft_t).T @ inputs) / count


def _hard_grad(inputs: np.ndarray, student_out: np.ndarray,
               teacher_out: np.ndarray) -> np.ndarray:
    count = inputs.shape[0]
    return ((student_out - teacher_out).T @ inputs) / count


def distill_layer(layer: WeightLayer, quant_weights: np.ndarray,
                  original_output: np.ndarray, sample_inputs: np.ndarray,
                  cfg: DistillConfig) -> DistillResult:
    weights = quant_weights.astype(np.float64).copy()
    student_out = forward(sample_inputs, weights)
    pre = output_deviation(student_out, original_output)
    per_sample_pre = per_sample_deviation(student_out, original_output)

    for _ in range(cfg.steps):
        student_out = forward(sample_inputs, weights)
        # Fixed operation order: hard residual first, then soft residual,
        # sum, then the single weight update.
        hard = _hard_grad(sample_inputs, student_out, original_output)
        soft = _soft_grad(sample_inputs, student_out, original_output,
                          cfg.temperature)
        gradient = (1.0 - cfg.soft_weight) * hard + cfg.soft_weight * soft
        weights = weights - cfg.learning_rate * gradient

    corrected_out = forward(sample_inputs, weights)
    post = output_deviation(corrected_out, original_output)
    per_sample_post = per_sample_deviation(corrected_out, original_output)
    correction = weights - quant_weights
    increased = bool(post > pre)
    result = DistillResult(
        pre_deviation=pre,
        post_deviation=post,
        delta_deviation=post - pre,
        increased=increased,
        increase_judgement={
            "pre_rmse": pre,
            "post_rmse": post,
            "post_minus_pre": post - pre,
            "relative_change": (post - pre) / pre if pre > 0.0 else 0.0,
        },
        correction_norm=float(np.linalg.norm(correction)),
        correction_max_abs=float(np.max(np.abs(correction))),
        corrected_weights=[float(x) for x in weights.ravel()],
        per_sample_pre=per_sample_pre,
        per_sample_post=per_sample_post,
        fingerprint="",
    )
    result = DistillResult(**{**result.__dict__, "fingerprint": fingerprint(
        {"pre": pre, "post": post, "weights": result.corrected_weights,
         "steps": cfg.steps, "temperature": cfg.temperature,
         "soft_weight": cfg.soft_weight,
         "learning_rate": cfg.learning_rate}, prefix="dst-")})
    return result


def reference_config(base: LayerQuantConfig) -> LayerQuantConfig:
    """High-bit-width, coarse-grained reference path (int16, whole layer)."""
    return LayerQuantConfig(
        name=base.name,
        bits=16,
        level="int16",
        group_size=0,
        q_lo=base.q_lo,
        q_hi=base.q_hi,
        frac_bits=max(base.frac_bits, 8),
        scheme=base.scheme,
        rounding=base.rounding,
        saturation=base.saturation,
    )


def stage_attribution(original_output: np.ndarray,
                      calibration_output: np.ndarray,
                      requant_output: np.ndarray,
                      saturated_output: np.ndarray,
                      corrected_output: np.ndarray,
                      reference_output: np.ndarray,
                      tolerance: float) -> dict[str, Any]:
    deviations = {
        STAGE_CALIBRATION: output_deviation(calibration_output, original_output),
        STAGE_REQUANT: output_deviation(requant_output, original_output),
        STAGE_SATURATION: output_deviation(saturated_output, original_output),
        STAGE_DISTILL: output_deviation(corrected_output, original_output),
    }
    reference_rmse = output_deviation(reference_output, original_output)
    first_breach = next((stage for stage in STAGES
                         if deviations[stage] > tolerance), None)
    return {
        "stage_rmse": deviations,
        "reference_rmse": reference_rmse,
        "tolerance": tolerance,
        "first_stage_over_tolerance": first_breach,
        "stage_order": list(STAGES),
    }
