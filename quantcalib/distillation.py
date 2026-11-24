"""Calibration-set response, error attribution and light distillation."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Sequence

from .model import CalibrationSet, DistillationConfig, Layer, LayerQuantConfig
from .quantization import dequantize_integer, fixed_point_round, quantize_pre_saturation, quantize_unsat
from .utils import clean_float


ATTRIBUTION_PRIORITY = (
    "saturation_truncation",
    "percentile_discretization",
    "scale_quantization",
    "dequant_rounding",
)


def _unit_interval(*parts: Any) -> float:
    digest = hashlib.sha256("\0".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _basis_matrix(
    layer: Layer, group_index: int, start: int, end: int, output_dim: int
) -> list[list[float]]:
    width = end - start
    normalizer = math.sqrt(float(width)) if width else 1.0
    matrix: list[list[float]] = []
    for global_index in range(start, end):
        row: list[float] = []
        for output_index in range(output_dim):
            coefficient = 0.5 + 2.0 * _unit_interval(
                layer.name, layer.content_version, group_index, global_index, output_index
            )
            row.append(coefficient / normalizer)
        matrix.append(row)
    return matrix


def _project(
    *,
    deltas: Sequence[float],
    activation: Sequence[float],
    basis: Sequence[Sequence[float]],
    output_dim: int,
) -> list[float]:
    result = [0.0 for _ in range(output_dim)]
    input_dim = len(activation)
    for offset, delta in enumerate(deltas):
        if delta == 0.0:
            continue
        x_value = float(activation[offset % input_dim])
        row = basis[offset]
        for output_index in range(output_dim):
            result[output_index] += float(delta) * x_value * row[output_index]
    return [clean_float(value, 17) for value in result]


def _softmax(values: Sequence[float], temperature: float) -> list[float]:
    scaled = [float(value) / float(temperature) for value in values]
    maximum = max(scaled) if scaled else 0.0
    exponents = [math.exp(value - maximum) for value in scaled]
    total = sum(exponents)
    if total == 0.0 or not math.isfinite(total):
        return [1.0 / len(values) for _ in values]
    return [value / total for value in exponents]


def distill_vectors(
    raw: Sequence[float], quantized: Sequence[float], cfg: DistillationConfig
) -> tuple[list[float], dict[str, Any]]:
    current = [clean_float(value, 17) for value in quantized]
    teacher = _softmax(raw, cfg.temperature)
    alphas: list[float] = []
    agreements: list[float] = []
    for _ in range(cfg.steps):
        student = _softmax(current, cfg.temperature)
        agreement = sum(left * right for left, right in zip(teacher, student))
        alpha = cfg.soft_label_weight * (cfg.temperature / (1.0 + cfg.temperature)) * agreement
        alphas.append(clean_float(alpha, 17))
        agreements.append(clean_float(agreement, 17))
        current = [
            clean_float(value + alpha * (float(target) - value), 17)
            for value, target in zip(current, raw)
        ]
    return current, {
        "temperature": clean_float(cfg.temperature),
        "steps": cfg.steps,
        "soft_label_weight": clean_float(cfg.soft_label_weight),
        "teacher_soft_targets": [clean_float(value, 17) for value in teacher],
        "agreement_sequence": agreements,
        "alpha_sequence": alphas,
        "operation_order": "softmax teacher once; per step: student softmax, agreement dot product, alpha, synchronous element update",
    }


def _error_metrics(truth: float, actual: float, tolerance: float) -> dict[str, Any]:
    error = clean_float(float(actual) - float(truth), 17)
    absolute = abs(error)
    zero_truth = truth == 0.0
    if zero_truth:
        branch = "absolute_zero_truth"
        relative = None
        exceeded = absolute > tolerance
        decision_metric = clean_float(absolute, 17)
    else:
        branch = "relative_nonzero_truth"
        relative = clean_float(absolute / abs(float(truth)), 17)
        exceeded = relative > tolerance
        decision_metric = relative
    return {
        "error": error,
        "absolute_error": clean_float(absolute, 17),
        "relative_error": relative,
        "truth_zero": zero_truth,
        "decision_branch": branch,
        "decision_metric": decision_metric,
        "exceeded": exceeded,
    }


def _choose_attribution(scores: dict[str, float]) -> tuple[str, float, list[str]]:
    maximum = max((abs(float(value)) for value in scores.values()), default=0.0)
    candidates = [name for name in ATTRIBUTION_PRIORITY if abs(float(scores[name])) == maximum]
    source = candidates[0] if candidates else ATTRIBUTION_PRIORITY[0]
    return source, clean_float(maximum, 17), candidates


def analyze_group(
    *,
    layer: Layer,
    cfg: LayerQuantConfig,
    calibration: CalibrationSet,
    calibration_group: dict[str, Any],
    quantization_group: dict[str, Any],
    distill_cfg: DistillationConfig,
) -> dict[str, Any]:
    start = int(calibration_group["start"])
    end = int(calibration_group["end"])
    group_index = int(calibration_group["group_index"])
    raw_weights = [clean_float(value) for value in layer.weights[start:end]]
    dequantized = [float(value) for value in quantization_group["dequantized"]]
    output_dim = calibration.output_dim
    basis = _basis_matrix(layer, group_index, start, end, output_dim)

    effective_scale = float(calibration_group["scale"])
    effective_zero = int(calibration_group["zero"])
    base_scale = float(calibration_group["scale_base"])
    base_zero = int(calibration_group["zero_base"])
    full_scale = float(calibration_group["full_range_scale"])
    full_zero = int(calibration_group["full_range_zero"])
    qmin = int(calibration_group["qmin"])
    qmax = int(calibration_group["qmax"])

    deltas: dict[str, list[float]] = {
        "raw": raw_weights,
        "percentile_discretization": [],
        "scale_quantization": [],
        "saturation_truncation": [],
        "dequant_rounding": [],
    }
    for value, final_value in zip(raw_weights, dequantized):
        _, pct_effective_unsat = quantize_unsat(
            value, scale=effective_scale, zero=effective_zero
        )
        _, pct_base_unsat = quantize_unsat(value, scale=base_scale, zero=base_zero)
        _, full_effective_unsat = quantize_unsat(value, scale=full_scale, zero=full_zero)
        _, pre_integer, unsat_dequant = quantize_pre_saturation(
            value, scale=effective_scale, zero=effective_zero
        )
        sat_integer = min(qmax, max(qmin, pre_integer))
        saturation_dequant = dequantize_integer(
            sat_integer, scale=effective_scale, zero=effective_zero
        )
        final_fixed = fixed_point_round(saturation_dequant, cfg.fractional_bits)
        deltas["percentile_discretization"].append(
            clean_float(pct_effective_unsat - full_effective_unsat, 17)
        )
        deltas["scale_quantization"].append(
            clean_float(pct_effective_unsat - pct_base_unsat, 17)
        )
        deltas["saturation_truncation"].append(
            clean_float(saturation_dequant - pct_effective_unsat, 17)
        )
        deltas["dequant_rounding"].append(clean_float(final_fixed - saturation_dequant, 17))

    sample_records: list[dict[str, Any]] = []
    pre_exceeded_count = 0
    post_exceeded_count = 0
    attribution_counts = {name: 0 for name in ATTRIBUTION_PRIORITY}
    zero_truth_count = 0

    for sample in calibration.samples:
        activation = sample.input_vector
        raw_output = _project(
            deltas=raw_weights,
            activation=activation,
            basis=basis,
            output_dim=output_dim,
        )
        quant_output = _project(
            deltas=dequantized,
            activation=activation,
            basis=basis,
            output_dim=output_dim,
        )
        corrected_output, distill_meta = distill_vectors(raw_output, quant_output, distill_cfg)
        source_contributions = {
            name: _project(
                deltas=deltas[name],
                activation=activation,
                basis=basis,
                output_dim=output_dim,
            )
            for name in ATTRIBUTION_PRIORITY
        }
        for output_index in range(output_dim):
            before = _error_metrics(
                raw_output[output_index], quant_output[output_index], distill_cfg.tolerance
            )
            after = _error_metrics(
                raw_output[output_index], corrected_output[output_index], distill_cfg.tolerance
            )
            scores = {
                name: abs(float(source_contributions[name][output_index]))
                for name in ATTRIBUTION_PRIORITY
            }
            source, magnitude, tied_sources = _choose_attribution(scores)
            if before["exceeded"]:
                pre_exceeded_count += 1
                attribution_counts[source] += 1
            if after["exceeded"]:
                post_exceeded_count += 1
            if before["truth_zero"]:
                zero_truth_count += 1
            sample_records.append(
                {
                    "sample_id": sample.sample_id,
                    "output_index": output_index,
                    "raw_value": raw_output[output_index],
                    "quantized_value": quant_output[output_index],
                    "corrected_value": corrected_output[output_index],
                    "reference_output": clean_float(sample.reference_output[output_index]),
                    "before": before,
                    "after": after,
                    "attribution_source": source,
                    "attribution_magnitude": magnitude,
                    "attribution_scores": {name: clean_float(value, 17) for name, value in scores.items()},
                    "attribution_ties": tied_sources,
                    "attribution_priority": list(ATTRIBUTION_PRIORITY),
                }
            )

    analysis = {
        "kind": "group_error_analysis",
        "layer": layer.name,
        "content_version": layer.content_version,
        "content_fingerprint": layer.content_fingerprint,
        "group_index": group_index,
        "start": start,
        "end": end,
        "calibration_sample_ids": list(calibration.sample_ids),
        "calibration_fingerprint": calibration.fingerprint,
        "records": sample_records,
        "pre_exceeded_count": pre_exceeded_count,
        "post_exceeded_count": post_exceeded_count,
        "attribution_counts": attribution_counts,
        "zero_truth_count": zero_truth_count,
        "attribution_priority": list(ATTRIBUTION_PRIORITY),
    }
    return analysis


def build_group_distillation(
    *,
    layer: Layer,
    calibration: CalibrationSet,
    error_analysis: dict[str, Any],
    distill_cfg: DistillationConfig,
) -> dict[str, Any]:
    records = error_analysis["records"]
    raw_by_sample: dict[str, list[float]] = {}
    quant_by_sample: dict[str, list[float]] = {}
    for record in records:
        sample_id = str(record["sample_id"])
        raw_by_sample.setdefault(sample_id, []).append(float(record["raw_value"]))
        quant_by_sample.setdefault(sample_id, []).append(float(record["quantized_value"]))
    samples: list[dict[str, Any]] = []
    for sample in calibration.samples:
        raw = raw_by_sample[sample.sample_id]
        quantized = quant_by_sample[sample.sample_id]
        corrected, meta = distill_vectors(raw, quantized, distill_cfg)
        samples.append(
            {
                "sample_id": sample.sample_id,
                "raw_output": raw,
                "quantized_output": quantized,
                "corrected_output": corrected,
                "distillation": meta,
            }
        )
    return {
        "kind": "group_distillation",
        "layer": layer.name,
        "content_version": layer.content_version,
        "content_fingerprint": layer.content_fingerprint,
        "group_index": int(error_analysis["group_index"]),
        "calibration_sample_ids": list(calibration.sample_ids),
        "calibration_fingerprint": calibration.fingerprint,
        "tolerance": clean_float(distill_cfg.tolerance),
        "samples": samples,
    }
