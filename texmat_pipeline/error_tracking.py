"""Two-tier error propagation and per-pixel attribution (requirement 4).

The same inputs are computed at the configured fixed-point width and at a
one-tier-higher width.  Four checkpoints are compared -- noise accumulation,
quantization, material mapping/blending and final composite -- recording the
per-pixel real-valued difference and the first checkpoint whose magnitude
exceeds the configured tolerance.  When the final channels disagree, a
truncation-only reference run at the base width decides whether the difference
is driven by quantization truncation or by accumulated rounding decisions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import CHANNELS, GenerationConfig, Material
from .stages import StageSpecs, accumulate_layers, build_material_channels

ATTRIB_QUANT = "quantization_truncation"
ATTRIB_ROUND = "rounding_accumulation"
ATTRIB_NONE = "within_tolerance"


@dataclass
class TierRun:
    specs: StageSpecs
    round_mode: str
    noise_result: dict
    material_result: dict


def run_tier(textures: dict, materials: list[Material], config: GenerationConfig,
             frac_bits: int, round_mode: str) -> TierRun:
    if round_mode != config.fixed_point.round_mode or frac_bits != config.fixed_point.frac_bits:
        tier_config = config.with_overrides(frac_bits=frac_bits, round_mode=round_mode)
    else:
        tier_config = config
    specs = StageSpecs.from_config(tier_config)
    noise_result = accumulate_layers(textures, tier_config, specs)
    material_result = build_material_channels(textures, noise_result, materials, tier_config, specs)
    return TierRun(specs, round_mode, noise_result, material_result)


def _layer_matrix(noise_result: dict) -> np.ndarray:
    return np.stack([entry["image"] for entry in noise_result["layers"]], axis=0).astype(np.float64)


def _channel_matrix(material_result: dict) -> np.ndarray:
    rows = []
    for mat in material_result["materials"]:
        rows.append(np.stack([mat["channels"][c]["image"] for c in CHANNELS], axis=-1).astype(np.float64))
    return np.stack(rows, axis=0)


def compare_tiers(textures: dict, materials: list[Material], config: GenerationConfig,
                  base: TierRun, high: TierRun) -> dict:
    scale_base = base.specs.signed.scale
    scale_high = high.specs.signed.scale

    base_noise = _layer_matrix(base.noise_result) / scale_base
    high_noise = _layer_matrix(high.noise_result) / scale_high
    noise_diff = np.abs(base_noise - high_noise)

    base_chan = _channel_matrix(base.material_result) / base.specs.unit.scale
    high_chan = _channel_matrix(high.material_result) / high.specs.unit.scale
    channel_diff = np.abs(base_chan - high_chan)

    tol = config.tolerance
    stage_names = ("noise_accumulation", "quantization", "material_blend", "final_composite")
    material_ids = [m["material_id"] for m in base.material_result["materials"]]

    noise_hit = np.any(noise_diff > tol, axis=0)  # (H, W)
    blend_hit = np.any(channel_diff > tol, axis=-1)  # (M, H, W)
    first_exceed = np.full(channel_diff.shape[:-1], "", dtype=object)
    for m in range(channel_diff.shape[0]):
        first_exceed[m] = np.where(
            noise_hit, stage_names[0], np.where(blend_hit[m], stage_names[2], ATTRIB_NONE)
        )

    trunc = run_tier(textures, materials, config, base.specs.unit.frac_bits, "truncate")
    trunc_chan = _channel_matrix(trunc.material_result) / trunc.specs.unit.scale
    trunc_diff = np.abs(base_chan - trunc_chan)

    inconsistent = channel_diff > tol
    attribution = np.where(
        inconsistent & (trunc_diff <= tol),
        ATTRIB_ROUND,
        np.where(inconsistent, ATTRIB_QUANT, ATTRIB_NONE),
    )

    stats = {
        ATTRIB_QUANT: int(np.count_nonzero(attribution == ATTRIB_QUANT)),
        ATTRIB_ROUND: int(np.count_nonzero(attribution == ATTRIB_ROUND)),
        ATTRIB_NONE: int(np.count_nonzero(attribution == ATTRIB_NONE)),
    }
    return {
        "material_ids": material_ids,
        "stage_names": list(stage_names),
        "noise_diff_max": float(noise_diff.max()),
        "channel_diff_max": float(channel_diff.max()),
        "per_stage_max_diff": {
            stage_names[0]: float(noise_diff.max()),
            stage_names[1]: float(noise_diff.max()),
            stage_names[2]: float(channel_diff.max()),
            stage_names[3]: float(channel_diff.max()),
        },
        "first_exceed": first_exceed,
        "attribution": attribution,
        "attribution_stats": stats,
        "inconsistent_pixels": int(np.count_nonzero(inconsistent)),
        "tolerance": tol,
        "base_frac_bits": base.specs.unit.frac_bits,
        "high_frac_bits": high.specs.unit.frac_bits,
        "noise_diff": noise_diff,
        "channel_diff": channel_diff,
    }
