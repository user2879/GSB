from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .config import EffectiveConfig
from .fixedpoint import FixedPoint
from .util import fingerprint


def _log_gaussian_contribution(
    target: tuple[float, ...], sample: tuple[float, ...], bandwidth: float
) -> float:
    distance_squared = 0.0
    for dimension_index in range(len(target)):
        difference = target[dimension_index] - sample[dimension_index]
        distance_squared += difference * difference
    return -(distance_squared / (2.0 * bandwidth * bandwidth))


def _kde_one(
    window_index: int,
    target: tuple[float, ...],
    samples: tuple[tuple[float, ...], ...],
    bandwidth: float,
) -> dict[str, Any]:
    contributions = [
        _log_gaussian_contribution(target, sample, bandwidth) for sample in samples
    ]
    maximum = max(contributions)
    scaled_sum = math.fsum(math.exp(value - maximum) for value in contributions)
    dimension = len(target)
    normalizer = (
        math.log(len(samples))
        + dimension * math.log(bandwidth)
        + 0.5 * dimension * math.log(2.0 * math.pi)
    )
    log_density = maximum + math.log(scaled_sum) - normalizer
    return {
        "window_index": window_index,
        "log_density": log_density,
        "sum_order": "window_index_ascending_then_dimension_ascending",
    }


def _score_payload(
    log_density: float,
    maximum_log_density: float,
    config: EffectiveConfig,
) -> dict[str, Any]:
    fixed = FixedPoint.from_config(config.fixed_point)
    raw_relative_permille = 1000.0 * math.exp(log_density - maximum_log_density)
    score_units = fixed.quantize(raw_relative_permille)
    threshold_units = int(config.threshold_permille * fixed.scale)
    half_step_permille = (fixed.quantization_units / fixed.scale) * 1000.0 / 2.0
    boundary = (
        abs(raw_relative_permille - config.threshold_permille) <= half_step_permille
    )
    reference_bits = fixed.fractional_bits + 1
    reference_fixed = FixedPoint(reference_bits, fixed.quantization_units)
    reference_score = reference_fixed.quantize(raw_relative_permille)
    reference_threshold = int(config.threshold_permille * reference_fixed.scale)
    density = math.exp(log_density) if log_density > -745.0 else 0.0
    return {
        "density": density,
        "log_density": log_density,
        "relative_permille": raw_relative_permille,
        "score_fixed_units": score_units,
        "score_permille": score_units / fixed.scale,
        "threshold_permille": config.threshold_permille,
        "threshold_fixed_units": threshold_units,
        "difference_to_threshold_units": score_units - threshold_units,
        "quantization_half_step_permille": half_step_permille,
        "boundary": boundary,
        "sentinel": "below_threshold" if score_units < threshold_units else "at_or_above_threshold",
        "reference": {
            "fractional_bits": reference_bits,
            "density_fixed_units": reference_score,
            "density": reference_score / reference_fixed.scale,
            "threshold_fixed_units": reference_threshold,
            "difference_to_threshold_units": reference_score - reference_threshold,
        },
    }


def run_kde(
    observations: dict[int, list[float]],
    config: EffectiveConfig,
    executor: ThreadPoolExecutor,
) -> dict[str, Any]:
    fixed = FixedPoint.from_config(config.fixed_point)
    ordered_indices = sorted(observations)
    quantized = {
        window_index: tuple(fixed.dequantize_vector(fixed.quantize_vector(observations[window_index])))
        for window_index in ordered_indices
    }
    samples = tuple(quantized[window_index] for window_index in ordered_indices)
    futures = []
    for window_index in ordered_indices:
        key = (window_index, "kde-density")
        future = executor.submit(
            _kde_one,
            window_index,
            quantized[window_index],
            samples,
            config.bandwidth,
        )
        futures.append((key, future))
    results = [(key, future.result()) for key, future in futures]
    results.sort(key=lambda item: item[0])
    density_rows = [result for _, result in results]
    maximum_log_density = max(float(row["log_density"]) for row in density_rows)
    scored: dict[int, dict[str, Any]] = {}
    for row in density_rows:
        window_index = int(row["window_index"])
        scored[window_index] = _score_payload(
            float(row["log_density"]), maximum_log_density, config
        )
        scored[window_index]["window_index"] = window_index
    window_rows = [scored[window_index] for window_index in ordered_indices]
    boundary_windows = [
        row["window_index"] for row in window_rows if bool(row["boundary"])
    ]
    below_threshold = [
        row["window_index"] for row in window_rows if row["sentinel"] == "below_threshold"
    ]
    payload = {
        "kernel": "gaussian",
        "bandwidth_label": config.bandwidth_label,
        "bandwidth": config.bandwidth,
        "threshold_permille": config.threshold_permille,
        "normalization": "relative_to_training_maximum_density_permille",
        "sum_order": "window_index_ascending_then_dimension_ascending",
        "maximum_log_density": maximum_log_density,
        "window_scores": window_rows,
        "boundary_windows": boundary_windows,
        "below_threshold_windows": below_threshold,
        "baseline_fingerprint": fingerprint(
            {
                "bandwidth": config.bandwidth,
                "threshold_permille": config.threshold_permille,
                "maximum_log_density": maximum_log_density,
                "scores": [row["score_fixed_units"] for row in window_rows],
            }
        ),
    }
    return payload
