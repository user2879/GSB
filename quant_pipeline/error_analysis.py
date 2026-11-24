"""Dequantization error analysis and deterministic source attribution."""

from __future__ import annotations

from typing import Any

import numpy as np

from .calibration import CalibrationResult
from .config import EffectiveLayerConfig, ToleranceConfig
from .errors import E_LAYER_QUANTIZATION, PipelineError
from .inputs import WeightLayer
from .quantization import QuantizationResult


ATTRIBUTION_PRIORITY = (
    "saturation_clipping",
    "percentile_discretization",
    "scaling_quantization",
    "dequantization_rounding",
)


def analyze_errors(
    layer: WeightLayer,
    quantization: QuantizationResult,
    calibration: CalibrationResult,
    config: EffectiveLayerConfig,
    tolerances: ToleranceConfig,
) -> dict[str, Any]:
    try:
        truth = np.asarray(layer.weights, dtype=np.float64).reshape(-1)
        actual = np.asarray(quantization.dequantized, dtype=np.float64).reshape(-1)
        base = np.asarray(quantization.base_dequantized, dtype=np.float64).reshape(-1)
        unclipped = np.asarray(quantization.unclipped_dequantized, dtype=np.float64).reshape(-1)
        clipped = np.asarray(quantization.clipped_mask, dtype=bool).reshape(-1)
        absolute_errors = np.abs(actual - truth)

        relative_errors: list[float | None] = []
        over_mask = np.zeros(truth.size, dtype=bool)
        relative_branches: list[str] = []
        absolute_threshold = float(tolerances.absolute)
        relative_threshold = float(tolerances.relative)
        for index, (true_value, abs_error) in enumerate(zip(truth, absolute_errors, strict=True)):
            if float(true_value) == 0.0:
                relative_errors.append(None)
                relative_branches.append("absolute_for_zero_truth")
                over_mask[index] = bool(abs_error > absolute_threshold)
            else:
                relative_value = float(abs_error / abs(float(true_value)))
                relative_errors.append(relative_value)
                relative_branches.append("relative_and_absolute")
                over_mask[index] = bool(
                    abs_error > absolute_threshold or relative_value > relative_threshold
                )

        # Percentile range metadata is expanded back to element order.
        low_by_element = np.empty(truth.size, dtype=np.float64)
        high_by_element = np.empty(truth.size, dtype=np.float64)
        group_id_by_element: list[str] = [""] * truth.size
        for group in calibration.data["groups"]:
            start = int(group["element_start"])
            end = int(group["element_end"])
            low_by_element[start:end] = float(group["percentile_low_value"])
            high_by_element[start:end] = float(group["percentile_high_value"])
            for index in range(start, end):
                group_id_by_element[index] = str(group["group_id"])

        over_elements: list[dict[str, Any]] = []
        distribution = {source: 0 for source in ATTRIBUTION_PRIORITY}
        for index in np.flatnonzero(over_mask).astype(int).tolist():
            percentile_value = float(
                np.clip(truth[index], low_by_element[index], high_by_element[index])
            )
            percentile_contribution = float(abs(truth[index] - percentile_value))
            scaling_contribution = float(abs(base[index] - percentile_value))
            saturation_contribution = (
                float(abs(actual[index] - unclipped[index])) if clipped[index] else 0.0
            )
            dequantization_contribution = float(abs(unclipped[index] - base[index]))
            contributions = {
                "saturation_clipping": saturation_contribution,
                "percentile_discretization": percentile_contribution,
                "scaling_quantization": scaling_contribution,
                "dequantization_rounding": dequantization_contribution,
            }
            maximum = max(contributions.values())
            primary = next(
                source
                for source in ATTRIBUTION_PRIORITY
                if contributions[source] == maximum
            )
            distribution[primary] += 1
            true_value = float(truth[index])
            over_elements.append(
                {
                    "element_index": index,
                    "group_id": group_id_by_element[index],
                    "true_value": true_value,
                    "dequantized_value": float(actual[index]),
                    "absolute_error": float(absolute_errors[index]),
                    "relative_error": relative_errors[index],
                    "relative_branch": relative_branches[index],
                    "over_tolerance": True,
                    "primary_source": primary,
                    "contributions": contributions,
                    "clipped": bool(clipped[index]),
                    "percentile_low_value": float(low_by_element[index]),
                    "percentile_high_value": float(high_by_element[index]),
                }
            )

        relative_values = [value for value in relative_errors if value is not None]
        stats = {
            "element_count": int(truth.size),
            "over_tolerance_count": int(np.count_nonzero(over_mask)),
            "within_tolerance_count": int(truth.size - np.count_nonzero(over_mask)),
            "zero_truth_count": int(np.count_nonzero(truth == 0.0)),
            "zero_truth_absolute_branch_count": int(
                sum(branch == "absolute_for_zero_truth" for branch in relative_branches)
            ),
            "max_absolute_error": float(np.max(absolute_errors)) if truth.size else 0.0,
            "mean_absolute_error": float(np.mean(absolute_errors)) if truth.size else 0.0,
            "root_mean_square_error": float(np.sqrt(np.mean(np.square(absolute_errors))))
            if truth.size
            else 0.0,
            "max_relative_error": float(max(relative_values)) if relative_values else 0.0,
            "mean_relative_error": float(np.mean(relative_values)) if relative_values else 0.0,
            "clipped_count": int(np.count_nonzero(clipped)),
        }
        return {
            "stage": "error_analysis",
            "layer": layer.name,
            "tolerances": tolerances.as_dict(),
            "over_tolerance_rule": "abs_error > absolute OR relative_error > relative (unless truth == 0)",
            "zero_truth_rule": "relative error is undefined at truth == 0; absolute error is used alone",
            "attribution_priority": list(ATTRIBUTION_PRIORITY),
            "attribution_tie_rule": "first source in attribution_priority wins exact ties",
            "statistics": stats,
            "attribution_distribution": distribution,
            "absolute_errors": absolute_errors.tolist(),
            "relative_errors": relative_errors,
            "relative_branches": relative_branches,
            "over_tolerance_elements": over_elements,
        }
    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError(
            E_LAYER_QUANTIZATION,
            f"error attribution failed for layer {layer.name!r}: {exc}",
            stage="error_analysis",
            layer=layer.name,
            exception_type=type(exc).__name__,
        ) from exc

