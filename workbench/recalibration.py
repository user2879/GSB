"""Fixed-point intent-confidence recalibration driven by slot completeness."""
from __future__ import annotations

import numpy as np

from .fixedpoint import (
    clamp_fixed,
    fixed_to_str,
    quantize_to_grid,
    scale_factor,
)


def recalibrate(raw_score_fixed: int, required_count: int, filled_count: int,
                suppression_count: int, config) -> dict:
    bits = config.scale_bits()
    factor = scale_factor(bits)
    knobs = config.recal_fixed()
    completeness = (filled_count * factor) // required_count if required_count else factor
    boost = (knobs["completeness_weight"] * completeness) // factor
    penalty = knobs["suppression_weight"] * int(suppression_count)
    floor = knobs["floor"]
    adjusted = raw_score_fixed + boost - penalty
    if adjusted < floor:
        adjusted = floor
    adjusted = clamp_fixed(adjusted, bits)

    grid = config.quant_grid()
    raw_quant = quantize_to_grid(raw_score_fixed, grid, bits)
    adj_quant = quantize_to_grid(adjusted, grid, bits)

    levels = {}
    for name, spec in config.quant_levels.items():
        level_grid = spec["grid"]
        levels[name] = {
            "grid": level_grid,
            "raw_quantized": fixed_to_str(quantize_to_grid(raw_score_fixed, level_grid, bits), bits),
            "recal_quantized": fixed_to_str(quantize_to_grid(adjusted, level_grid, bits), bits),
        }
    vector = np.asarray([raw_score_fixed, adjusted, raw_quant, adj_quant], dtype=np.int64)
    assert bool(np.all(vector >= 0)) and bool(np.all(vector <= factor))
    return {
        "stage": "recalibration",
        "completeness": fixed_to_str(completeness, bits),
        "suppression_count": int(suppression_count),
        "boost_fixed": boost,
        "penalty_fixed": penalty,
        "floor": config.recal["floor"],
        "confidence_before": fixed_to_str(raw_score_fixed, bits),
        "confidence_after": fixed_to_str(adjusted, bits),
        "quantization_level": config.active_quant_level,
        "grid": grid,
        "confidence_before_quantized": fixed_to_str(raw_quant, bits),
        "confidence_after_quantized": fixed_to_str(adj_quant, bits),
        "all_quantization_levels": levels,
    }
