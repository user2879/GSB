"""Fixed-point error attribution across two adjacent fractional bit widths.

The same input is fully evaluated at the configured width ``F`` and at the
reference width ``F+1``.  Four pipeline stages are compared on every pixel (no
sampling):

1. ``noise_overlay``  - wide Q16 accumulated height fields, per layer;
2. ``quantize``       - narrowed Q(F) layer fields;
3. ``material_blend`` - per-channel contributions before blending;
4. ``channel_compose``- composed Q(F) channel values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import CHANNELS, GenerationConfig, MaterialSpec
from .fixedpoint import rescale_array
from .pipeline import WORK_BITS, ChannelResult, LayerResult, MaterialResult, compute_layer, compose_material
from .loader import DecodedTexture

STAGES = ("noise_overlay", "quantize", "material_blend", "channel_compose")


@dataclass
class GearRun:
    f_bits: int
    layers: dict[str, LayerResult]
    materials: dict[str, MaterialResult]


def run_gear(config: GenerationConfig, materials: list[MaterialSpec],
             textures: dict[str, DecodedTexture], f_bits: int) -> GearRun:
    layers: dict[str, LayerResult] = {}
    previous_wide: np.ndarray | None = None
    previous_source_name: str | None = None
    for spec in config.noise_layers:
        texture = textures[spec.source_texture]
        if previous_wide is None or spec.source_texture != previous_source_name:
            previous_wide = None
        result = compute_layer(spec, texture, previous_wide, config,
                               f_bits=f_bits, quantize=config.quantize, rounding=config.rounding)
        layers[f"layer{spec.index:02d}"] = result
        previous_wide = result.accum
        previous_source_name = spec.source_texture
    material_results = {
        material.name: compose_material(material, layers, config, f_bits, config.rounding)
        for material in materials
    }
    return GearRun(f_bits=f_bits, layers=layers, materials=material_results)


def _max_abs(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.max(np.abs(a.astype(np.int64) - b.astype(np.int64)))) if a.size else 0


def _aligned(reference: np.ndarray, ref_f: int, target_f: int, rounding: str) -> np.ndarray:
    """Align the F+1 reference down to the target Q(F) scale for comparison."""
    return rescale_array(reference, ref_f, target_f, rounding)


def _first_breach_stage(stage_diffs: dict[str, np.ndarray],
                        tolerances: dict[str, float], height: int, width: int) -> np.ndarray:
    """Per-pixel first stage whose absolute diff exceeds its tolerance."""
    label = np.full((height, width), "", dtype=object)
    for stage in STAGES:
        diff = stage_diffs[stage]
        over = np.abs(diff) > tolerances[stage]
        unfilled = label == ""
        take = over & unfilled
        label[take] = stage
    return label


def _bin_index(value: np.ndarray, f_bits: int, bin_width: int) -> np.ndarray:
    """Decision bins are defined on the final uint8 output domain, so both
    gears are first converted to byte scale and then assigned to bins."""
    from .fixedpoint import to_uint8
    byte_values = to_uint8(value, f_bits, "truncate").astype(np.int64)
    return np.floor_divide(byte_values, bin_width)


def compare_gears(base: GearRun, reference: GearRun, config: GenerationConfig,
                  materials: list[MaterialSpec]) -> dict[str, Any]:
    if reference.f_bits != base.f_bits + 1:
        raise ValueError("reference gear must be exactly one fractional bit wider")
    height = width = config.active_tier
    rounding = config.rounding
    report: dict[str, Any] = {"stages": {}, "materials": {}, "tolerances": dict(config.tolerances)}

    overlay_diffs = []
    quant_diffs = []
    for spec in config.noise_layers:
        layer_id = f"layer{spec.index:02d}"
        base_layer = base.layers[layer_id]
        ref_layer = reference.layers[layer_id]
        ref_overlay = _aligned(ref_layer.accum, WORK_BITS, WORK_BITS, rounding)
        overlay_diffs.append(np.abs(base_layer.accum.astype(np.int64) - ref_overlay.astype(np.int64)))
        if config.quantize:
            ref_quant = _aligned(
                rescale_array(ref_layer.accum, WORK_BITS, reference.f_bits, rounding),
                reference.f_bits, base.f_bits, rounding)
            quant_diffs.append(np.abs(base_layer.quantized.astype(np.int64) - ref_quant.astype(np.int64)))
        else:
            quant_diffs.append(np.zeros((height, width), dtype=np.int64))
    report["stages"]["noise_overlay"] = {
        "max_abs_diff": int(max((d.max() for d in overlay_diffs), default=0)),
        "mean_abs_diff": float(np.mean([d.mean() for d in overlay_diffs])) if overlay_diffs else 0.0,
    }
    report["stages"]["quantize"] = {
        "max_abs_diff": int(max((d.max() for d in quant_diffs), default=0)),
        "mean_abs_diff": float(np.mean([d.mean() for d in quant_diffs])) if quant_diffs else 0.0,
    }

    blend_max = 0
    compose_max = 0
    boundary_sensitive: list[dict[str, Any]] = []
    for material in materials:
        base_material = base.materials[material.name]
        ref_material = reference.materials[material.name]
        per_channel: dict[str, Any] = {}
        for channel in CHANNELS:
            base_ch: ChannelResult = base_material.channels[channel]
            ref_ch: ChannelResult = ref_material.channels[channel]
            ref_contrib_aligned = _aligned(ref_ch.contributions, reference.f_bits, base.f_bits, rounding)
            blend_diff = np.abs(base_ch.contributions.astype(np.int64) - ref_contrib_aligned.astype(np.int64))
            blend_max = max(blend_max, int(blend_diff.max()) if blend_diff.size else 0)
            ref_values_aligned = _aligned(ref_ch.values, reference.f_bits, base.f_bits, rounding)
            compose_diff = base_ch.values.astype(np.int64) - ref_values_aligned.astype(np.int64)
            compose_max = max(compose_max, int(np.abs(compose_diff).max()) if compose_diff.size else 0)
            per_channel[channel] = {
                "max_abs_diff": int(np.abs(compose_diff).max()),
                "mean_abs_diff": float(np.abs(compose_diff).mean()),
            }

            from .fixedpoint import to_uint8
            base_bytes = to_uint8(base_ch.values, base.f_bits, rounding).astype(np.int64)
            ref_bytes = to_uint8(ref_ch.values, reference.f_bits, rounding).astype(np.int64)
            base_bins = np.floor_divide(base_bytes, config.bin_width)
            ref_bins = np.floor_divide(ref_bytes, config.bin_width)
            # Boundary-sensitive: the two gears either land in different
            # decision bins or round to a different output byte at the edge.
            different_side = (base_bins != ref_bins) | (base_bytes != ref_bytes)
            if different_side.any():
                stage_diffs = {
                    "noise_overlay": overlay_diffs[-1],
                    "quantize": quant_diffs[-1],
                    "material_blend": blend_diff.max(axis=0),
                    "channel_compose": np.abs(compose_diff),
                }
                first_stage = _first_breach_stage(stage_diffs, config.tolerances, height, width)
                rows, cols = np.nonzero(different_side)
                stage_strength = {stage: float(stage_diffs[stage][rows, cols].max()) for stage in STAGES}
                dominant = max(stage_strength, key=lambda key: (stage_strength[key], STAGES.index(key)))
                for row, col in zip(rows.tolist(), cols.tolist()):
                    boundary_sensitive.append({
                        "material": material.name,
                        "channel": channel,
                        "pixel": [row, col],
                        "base_bin": int(base_bins[row, col]),
                        "reference_bin": int(ref_bins[row, col]),
                        "dominant_error_stage": dominant,
                        "first_breach_stage": first_stage[row, col] or "within_tolerance",
                    })
        report["materials"][material.name] = per_channel
    report["stages"]["material_blend"] = {
        "max_abs_diff": blend_max,
        "mean_abs_diff": float(blend_max),
    }
    report["stages"]["channel_compose"] = {
        "max_abs_diff": compose_max,
        "mean_abs_diff": float(compose_max),
    }
    report["boundary_sensitive_pixels"] = boundary_sensitive
    report["boundary_sensitive_count"] = len(boundary_sensitive)
    report["gear_widths"] = {"configured": base.f_bits, "reference": reference.f_bits}
    return report
