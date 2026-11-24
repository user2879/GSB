"""Cross-width error propagation and final-result attribution."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from .cache import CacheStore
from .engine import MaterialEngine
from .fixed import CHANNELS, fixed_to_unit
from .injection import InjectionRegistry
from .mixing import mix_channel
from .models import BuildResult, GenerationConfig, Material, Texture


STAGE_ORDER = (
    "noise_composition",
    "quantization",
    "material_mix",
    "channel_synthesis",
)
STAGE_CODES = {
    "none": 0,
    "noise_composition": 1,
    "quantization": 2,
    "material_mix": 3,
    "channel_synthesis": 4,
}


def high_precision_config(config: GenerationConfig) -> tuple[GenerationConfig, int]:
    higher = [width for width in config.bit_width_ladder if width > config.fixed_point.bit_width]
    if not higher:
        raise ValueError("bit_width_ladder has no higher precision width")
    high_width = min(higher)
    delta = high_width - config.fixed_point.bit_width
    high_fraction = min(high_width - 2, config.fixed_point.fraction_bits + delta)
    fixed = replace(
        config.fixed_point,
        bit_width=high_width,
        fraction_bits=high_fraction,
    )
    return replace(config, fixed_point=fixed), high_width


def _max_layer_diff(
    low: BuildResult,
    high: BuildResult,
    attribute: str,
) -> np.ndarray:
    height = low.config.resolution
    width = low.config.resolution
    diff = np.zeros((height, width), dtype=np.float64)
    low_layers = {layer.index: layer for layer in low.layers}
    high_layers = {layer.index: layer for layer in high.layers}
    for index in sorted(set(low_layers) & set(high_layers)):
        low_values = getattr(low_layers[index], attribute)
        high_values = getattr(high_layers[index], attribute)
        low_unit = fixed_to_unit(low_values, low_layers[index].scale)
        high_unit = fixed_to_unit(high_values, high_layers[index].scale)
        diff = np.maximum(diff, np.abs(low_unit - high_unit))
    return diff


def _max_material_diff(
    low: BuildResult,
    high: BuildResult,
    attribute: str,
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    height = low.config.resolution
    width = low.config.resolution
    diff = np.zeros((height, width), dtype=np.float64)
    per_channel: dict[str, dict[str, float]] = {}
    for material_id in sorted(set(low.material_artifacts) & set(high.material_artifacts)):
        low_material = low.material_artifacts[material_id]
        high_material = high.material_artifacts[material_id]
        per_channel[material_id] = {}
        for channel in CHANNELS:
            if channel not in low_material.channels or channel not in high_material.channels:
                continue
            low_values = getattr(low_material.channels[channel], attribute)
            high_values = getattr(high_material.channels[channel], attribute)
            channel_diff = np.abs(
                fixed_to_unit(low_values, low.scale) - fixed_to_unit(high_values, high.scale)
            )
            diff = np.maximum(diff, channel_diff)
            per_channel[material_id][channel] = float(np.max(channel_diff))
    return diff, per_channel


def _stage_summary(diff: np.ndarray, tolerance: float) -> dict[str, Any]:
    mask = diff > tolerance
    first = None
    if np.any(mask):
        row, col = np.argwhere(mask)[0]
        first = {
            "row": int(row),
            "col": int(col),
            "difference": float(diff[row, col]),
        }
    return {
        "tolerance": float(tolerance),
        "max_abs_diff": float(np.max(diff)) if diff.size else 0.0,
        "mean_abs_diff": float(np.mean(diff)) if diff.size else 0.0,
        "nonzero_pixels": int(np.count_nonzero(diff)),
        "over_tolerance_pixels": int(np.count_nonzero(mask)),
        "first_over_tolerance": first,
    }


def _reverse_order_effect(
    config: GenerationConfig,
    materials: tuple[Material, ...],
    high: BuildResult,
    tolerance: float,
) -> dict[str, Any]:
    maximum = 0.0
    per_channel: dict[str, dict[str, float]] = {}
    if config.material_mix_mode == "weighted":
        return {
            "supported": False,
            "max_abs_diff": 0.0,
            "per_channel": per_channel,
            "reason": "weighted mode is commutative",
        }
    layers = {layer.index: layer for layer in high.layers}
    for material in materials:
        if material.material_id not in high.material_artifacts:
            continue
        per_channel[material.material_id] = {}
        for channel in CHANNELS:
            canonical = high.material_artifacts[material.material_id].channels.get(channel)
            if canonical is None:
                continue
            reversed_artifact = mix_channel(
                material=material,
                channel=channel,
                layers=layers,
                key="counterfactual-reversed",
                scale=high.scale,
                weight_fraction_bits=config.fixed_point.weight_fraction_bits,
                mix_mode=config.material_mix_mode,
                rounding=config.fixed_point.rounding[channel],
                max_boundary_samples=0,
                reverse_layers=True,
            )
            channel_diff = np.abs(
                fixed_to_unit(canonical.final, high.scale)
                - fixed_to_unit(reversed_artifact.final, high.scale)
            )
            maximum = max(maximum, float(np.max(channel_diff)))
            per_channel[material.material_id][channel] = float(np.max(channel_diff))
    return {
        "supported": True,
        "max_abs_diff": maximum,
        "over_tolerance": maximum > tolerance,
        "per_channel": per_channel,
    }


def build_error_trace(
    *,
    config: GenerationConfig,
    materials: tuple[Material, ...],
    textures: dict[str, Texture],
    low_result: BuildResult,
    cache: CacheStore | None = None,
    injections: InjectionRegistry | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray], BuildResult]:
    high_config, high_width = high_precision_config(config)
    high_engine = MaterialEngine(
        cache=cache or CacheStore(),
        injections=InjectionRegistry(
            injections.injections if injections is not None else [], apply_delays=False
        ),
    )
    high_result = high_engine.build(high_config, materials, textures)

    diffs: dict[str, np.ndarray] = {
        "noise_composition": _max_layer_diff(low_result, high_result, "unquantized"),
        "quantization": _max_layer_diff(low_result, high_result, "quantized"),
    }
    material_mix_diff, mix_per_channel = _max_material_diff(low_result, high_result, "mixed")
    channel_diff, channel_per_channel = _max_material_diff(low_result, high_result, "final")
    diffs["material_mix"] = material_mix_diff
    diffs["channel_synthesis"] = channel_diff

    stage_summary = {
        stage: _stage_summary(diffs[stage], config.tolerances[stage]) for stage in STAGE_ORDER
    }
    height = width = config.resolution
    first_stage = np.zeros((height, width), dtype=np.int8)
    for stage in STAGE_ORDER:
        mask = diffs[stage] > config.tolerances[stage]
        first_stage[(first_stage == 0) & mask] = STAGE_CODES[stage]
    first_stage_name_by_code = {code: name for name, code in STAGE_CODES.items()}
    first_stage_names = np.vectorize(lambda code: first_stage_name_by_code[int(code)])(first_stage)

    first_exceeded_stage = "none"
    for stage in STAGE_ORDER:
        if stage_summary[stage]["over_tolerance_pixels"]:
            first_exceeded_stage = stage
            break

    final_mismatch_mask = channel_diff > config.final_consistency_tolerance
    final_mismatch_pixels = int(np.count_nonzero(final_mismatch_mask))
    reverse_effect = _reverse_order_effect(
        config,
        materials,
        high_result,
        config.final_consistency_tolerance,
    )
    boundary_mismatch_pixels = 0
    for material_id, artifact_group in low_result.material_artifacts.items():
        for channel, artifact in artifact_group.channels.items():
            high_channel = (
                high_result.material_artifacts.get(material_id, None).channels.get(channel)
                if material_id in high_result.material_artifacts
                else None
            )
            if high_channel is None:
                continue
            channel_mask = (
                np.abs(fixed_to_unit(artifact.final, low_result.scale) - fixed_to_unit(high_channel.final, high_result.scale))
                > config.final_consistency_tolerance
            )
            for sample in artifact.boundary_samples:
                if sample.get("scope") != "pixel":
                    continue
                row = int(sample["row"])
                col = int(sample["col"])
                if channel_mask[row, col]:
                    boundary_mismatch_pixels += 1

    quantization_over = stage_summary["quantization"]["over_tolerance_pixels"]
    if final_mismatch_pixels > 0:
        if reverse_effect["supported"] and reverse_effect["max_abs_diff"] > config.final_consistency_tolerance:
            category = "mix_order"
        elif boundary_mismatch_pixels > 0:
            category = "rounding_boundary"
        elif quantization_over > 0:
            category = "quantization_truncation"
        else:
            category = "quantization_truncation"
    else:
        category = "none"

    attribution = {
        "category": category,
        "final_mismatch_pixels": final_mismatch_pixels,
        "quantization_over_tolerance_pixels": quantization_over,
        "rounding_boundary_mismatch_pixels": boundary_mismatch_pixels,
        "mix_order": reverse_effect,
        "evidence": {
            "quantization_max_abs_diff": stage_summary["quantization"]["max_abs_diff"],
            "material_mix_max_abs_diff": stage_summary["material_mix"]["max_abs_diff"],
            "channel_synthesis_max_abs_diff": stage_summary["channel_synthesis"]["max_abs_diff"],
            "final_consistency_tolerance": config.final_consistency_tolerance,
            "reverse_order_max_abs_diff": reverse_effect["max_abs_diff"],
            "boundary_mismatch_pixels": boundary_mismatch_pixels,
        },
    }

    report: dict[str, Any] = {
        "configured_bit_width": config.fixed_point.bit_width,
        "configured_fraction_bits": config.fixed_point.fraction_bits,
        "high_bit_width": high_width,
        "high_fraction_bits": high_config.fixed_point.fraction_bits,
        "stage_order": list(STAGE_ORDER),
        "stages": stage_summary,
        "first_exceeded_stage": first_exceeded_stage,
        "final_consistent": final_mismatch_pixels == 0,
        "final_mismatch_pixels": final_mismatch_pixels,
        "per_material_channel": {
            "material_mix_max_abs_diff": mix_per_channel,
            "channel_synthesis_max_abs_diff": channel_per_channel,
        },
        "first_exceeded_stage_by_pixel": first_stage_names.tolist(),
        "attribution": attribution,
    }
    if config.embed_pixel_diffs:
        report["pixel_differences"] = {stage: diffs[stage].tolist() for stage in STAGE_ORDER}
    return report, diffs, high_result


def compare_full_rebuild(
    *,
    incremental: BuildResult,
    config: GenerationConfig,
    materials: tuple[Material, ...],
    textures: dict[str, Texture],
    injections: InjectionRegistry | None = None,
) -> dict[str, Any]:
    fresh_engine = MaterialEngine(
        cache=CacheStore(),
        injections=InjectionRegistry(
            injections.injections if injections is not None else [], apply_delays=False
        ),
    )
    fresh = fresh_engine.build(config, materials, textures)
    per_channel: dict[str, dict[str, bool]] = {}
    incremental_complete = all(
        len(artifact.channels) == len(CHANNELS) for artifact in incremental.material_artifacts.values()
    ) and len(incremental.material_artifacts) == len(materials)
    fresh_complete = all(
        len(artifact.channels) == len(CHANNELS) for artifact in fresh.material_artifacts.values()
    ) and len(fresh.material_artifacts) == len(materials)
    exact = incremental_complete and fresh_complete
    for material_id in sorted(set(incremental.material_artifacts) | set(fresh.material_artifacts)):
        per_channel[material_id] = {}
        incremental_material = incremental.material_artifacts.get(material_id)
        fresh_material = fresh.material_artifacts.get(material_id)
        for channel in CHANNELS:
            left = (
                incremental_material.channels[channel].final
                if incremental_material is not None and channel in incremental_material.channels
                else None
            )
            right = (
                fresh_material.channels[channel].final
                if fresh_material is not None and channel in fresh_material.channels
                else None
            )
            if left is None or right is None:
                equal = left is None and right is None
            else:
                equal = bool(np.array_equal(left, right))
            per_channel[material_id][channel] = equal
            exact = exact and equal
    return {
        "exact_match": exact,
        "incremental_complete": incremental_complete,
        "fresh_complete": fresh_complete,
        "incremental_digest": incremental.final_digest,
        "fresh_digest": fresh.final_digest,
        "per_material_channel": per_channel,
    }
