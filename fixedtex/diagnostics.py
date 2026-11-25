"""Error propagation, quantization comparison and diagnostic artifacts."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .fixed import (
    DECISION_TIE_DOWN,
    DECISION_TIE_UP,
    downscale_fixed,
    quantize_fixed,
    scale_for,
)
from .hashing import content_key
from .io_utils import atomic_write_json
from .material import MixResult, base_channel_arrays, compose_channels, mix_channel
from .models import LayerSpec, MaterialConfig, RenderConfig, TextureData
from .noise import compute_layer_chunk, layer_fingerprint


@dataclass
class PathTrace:
    config: RenderConfig
    frac_bits: int
    quant_step: int | None
    layers_raw: dict[int, np.ndarray]
    layers_value: dict[int, np.ndarray]
    layers_residual: dict[int, np.ndarray]
    layer_tie_masks: dict[int, np.ndarray]
    channels: dict[str, MixResult]
    composite: np.ndarray


STAGE_NAMES = (
    "layer_noise_synthesis",
    "layer_quantization",
    "material_mixing",
    "channel_composite",
)


def _config_with(config: RenderConfig, *, frac_bits: int, quant_step: int | None = None) -> RenderConfig:
    fixed = config.fixed.__class__(
        fraction_bits=int(frac_bits),
        quant_step=config.fixed.quant_step if quant_step is None else int(quant_step),
        tie_rule=config.fixed.tie_rule,
        guard_bits=config.fixed.guard_bits,
    )
    return config.clone(fixed=fixed)


def _tie_mask(decisions: np.ndarray) -> np.ndarray:
    return (decisions == DECISION_TIE_DOWN) | (decisions == DECISION_TIE_UP)


def compute_path(
    config: RenderConfig,
    material: MaterialConfig,
    texture: TextureData,
    *,
    frac_bits: int,
    quantize: bool,
    quant_step: int | None = None,
) -> PathTrace:
    bits = int(frac_bits)
    path_config = _config_with(config, frac_bits=bits, quant_step=quant_step)
    height, width = texture.height, texture.width
    layers_raw: dict[int, np.ndarray] = {}
    layers_value: dict[int, np.ndarray] = {}
    layers_residual: dict[int, np.ndarray] = {}
    layer_tie_masks: dict[int, np.ndarray] = {}
    for layer in sorted(path_config.layers, key=lambda item: item.index):
        raw = compute_layer_chunk(
            layer,
            path_config,
            texture,
            0,
            width * height,
            frac_bits=bits,
            fingerprint=layer_fingerprint(layer, path_config, texture),
        )
        layers_raw[layer.index] = raw
        if quantize:
            quantized, residual, decisions = quantize_fixed(
                raw, int(quant_step if quant_step is not None else path_config.fixed.quant_step), path_config.fixed.tie_rule
            )
            layers_value[layer.index] = quantized
            layers_residual[layer.index] = residual
            layer_tie_masks[layer.index] = _tie_mask(decisions)
        else:
            layers_value[layer.index] = raw.copy()
            layers_residual[layer.index] = np.zeros_like(raw)
            layer_tie_masks[layer.index] = np.zeros(raw.shape, dtype=bool)

    base_channels = base_channel_arrays(texture, path_config, bits)
    channels: dict[str, MixResult] = {}
    for channel in material.ordered_channels():
        referenced = {
            index: layers_value[index].reshape(height, width)
            for index in channel.layer_indices
        }
        channels[channel.name] = mix_channel(
            channel,
            base_channels[channel.name],
            referenced,
            path_config,
            material,
            bits,
        )
    ordered_names = [channel.name for channel in material.ordered_channels()]
    composite = compose_channels(
        {name: channels[name].array for name in ordered_names},
        path_config,
        bits,
    )
    return PathTrace(
        config=path_config,
        frac_bits=bits,
        quant_step=quant_step if quantize else None,
        layers_raw=layers_raw,
        layers_value=layers_value,
        layers_residual=layers_residual,
        layer_tie_masks=layer_tie_masks,
        channels=channels,
        composite=composite,
    )


def _stats(array: np.ndarray) -> dict[str, Any]:
    values = np.abs(np.asarray(array, dtype=np.int64))
    if values.size == 0:
        return {"count": 0, "max_abs": 0, "mean_abs_x1000": 0}
    return {
        "count": int(values.size),
        "max_abs": int(values.max()),
        "mean_abs_x1000": int((int(values.sum()) * 1000) // values.size),
    }


def _first_coordinate(mask: np.ndarray) -> list[int] | None:
    points = np.argwhere(mask)
    if points.size == 0:
        return None
    return [int(points[0, 0]), int(points[0, 1])]


def _save_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def build_error_report(
    config: RenderConfig,
    material: MaterialConfig,
    texture: TextureData,
    output_dir: str | Path,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    low = compute_path(
        config,
        material,
        texture,
        frac_bits=config.fixed.fraction_bits,
        quantize=True,
        quant_step=config.fixed.quant_step,
    )
    high = compute_path(
        config,
        material,
        texture,
        frac_bits=config.fixed.fraction_bits + 1,
        quantize=False,
    )
    height, width = texture.height, texture.width
    if config.error_tolerance_quant_steps > 0:
        tolerance = config.fixed.quant_step * config.error_tolerance_quant_steps
    else:
        tolerance = max(1, config.fixed.quant_step // 4)
    stage_arrays: dict[str, np.ndarray] = {}
    stage_stats: dict[str, Any] = {name: {"layers": {}, "channels": {}, "max_abs": 0, "exceed_count": 0, "first_coordinate": None} for name in STAGE_NAMES}
    pixel_stage_diffs: list[np.ndarray] = []

    synthesis_layers: dict[str, np.ndarray] = {}
    for index in sorted(low.layers_raw):
        high_down = downscale_fixed(high.layers_raw[index], high.frac_bits, low.frac_bits).reshape(height, width)
        diff = np.abs(low.layers_raw[index].reshape(height, width) - high_down)
        synthesis_layers[str(index)] = diff
        stage_arrays[f"synthesis_layer_{index}"] = diff
        stats = _stats(diff)
        stats["exceed_count"] = int(np.count_nonzero(diff > tolerance))
        stats["first_coordinate"] = _first_coordinate(diff > tolerance)
        stage_stats[STAGE_NAMES[0]]["layers"][str(index)] = stats
    if synthesis_layers:
        stage_max = np.maximum.reduce(list(synthesis_layers.values()))
    else:
        stage_max = np.zeros((height, width), dtype=np.int64)
    pixel_stage_diffs.append(stage_max)

    quantization_layers: dict[str, np.ndarray] = {}
    for index in sorted(low.layers_raw):
        diff = np.abs(low.layers_residual[index]).reshape(height, width)
        quantization_layers[str(index)] = diff
        stage_arrays[f"quantization_layer_{index}"] = diff
        stats = _stats(diff)
        stats["exceed_count"] = int(np.count_nonzero(diff > tolerance))
        stats["first_coordinate"] = _first_coordinate(diff > tolerance)
        stage_stats[STAGE_NAMES[1]]["layers"][str(index)] = stats
    if quantization_layers:
        stage_max = np.maximum.reduce(list(quantization_layers.values()))
    else:
        stage_max = np.zeros((height, width), dtype=np.int64)
    pixel_stage_diffs.append(stage_max)

    mix_channels: dict[str, np.ndarray] = {}
    for channel in material.ordered_channels():
        high_down = downscale_fixed(high.channels[channel.name].array, high.frac_bits, low.frac_bits)
        diff = np.abs(low.channels[channel.name].array - high_down)
        reduced = diff.max(axis=2)
        mix_channels[channel.name] = reduced
        stage_arrays[f"mix_{channel.name}"] = reduced
        stats = _stats(diff)
        stats["exceed_count"] = int(np.count_nonzero(reduced > tolerance))
        stats["first_coordinate"] = _first_coordinate(reduced > tolerance)
        stage_stats[STAGE_NAMES[2]]["channels"][channel.name] = stats
    if mix_channels:
        stage_max = np.maximum.reduce(list(mix_channels.values()))
    else:
        stage_max = np.zeros((height, width), dtype=np.int64)
    pixel_stage_diffs.append(stage_max)

    high_composite_down = downscale_fixed(high.composite, high.frac_bits, low.frac_bits)
    composite_diff = np.abs(low.composite - high_composite_down)
    reduced_composite = composite_diff.max(axis=2)
    stage_arrays["composite"] = reduced_composite
    stage_stats[STAGE_NAMES[3]]["channels"]["rgba"] = _stats(composite_diff)
    stage_stats[STAGE_NAMES[3]]["channels"]["rgba"]["exceed_count"] = int(np.count_nonzero(reduced_composite > tolerance))
    stage_stats[STAGE_NAMES[3]]["channels"]["rgba"]["first_coordinate"] = _first_coordinate(reduced_composite > tolerance)
    pixel_stage_diffs.append(reduced_composite)

    first_exceed = np.zeros((height, width), dtype=np.uint8)
    assigned = np.zeros((height, width), dtype=bool)
    for stage_index, diff in enumerate(pixel_stage_diffs, start=1):
        mask = (diff > tolerance) & ~assigned
        first_exceed[mask] = stage_index
        assigned |= mask
    stage_arrays["first_exceed_stage"] = first_exceed
    for stage_index, name in enumerate(STAGE_NAMES, start=1):
        mask = first_exceed == stage_index
        aggregate = pixel_stage_diffs[stage_index - 1]
        stage_stats[name]["max_abs"] = int(aggregate.max()) if aggregate.size else 0
        stage_stats[name]["exceed_count"] = int(np.count_nonzero(aggregate > tolerance))
        stage_stats[name]["first_coordinate"] = _first_coordinate(aggregate > tolerance)
        stage_stats[name]["first_exceed_count"] = int(np.count_nonzero(mask))
        stage_stats[name]["first_exceed_coordinate"] = _first_coordinate(mask)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _save_npz_atomic(out / "stage_diffs.npz", stage_arrays)
    first_stage_index = int(np.min(np.where(first_exceed > 0, first_exceed, 255))) if np.any(first_exceed) else 0
    report: dict[str, Any] = {
        "kind": "stage_error_propagation",
        "tolerance_fixed": tolerance,
        "tolerance_derivation": {
            "quant_step": config.fixed.quant_step,
            "error_tolerance_quant_steps": config.error_tolerance_quant_steps,
            "zero_steps_rule": "max(1, quant_step // 4)",
        },
        "reference": {
            "fraction_bits": high.frac_bits,
            "quantized": False,
        },
        "observed": {
            "fraction_bits": low.frac_bits,
            "quant_step": config.fixed.quant_step,
        },
        "stages": stage_stats,
        "first_exceed_stage": STAGE_NAMES[first_stage_index - 1] if first_stage_index else None,
        "first_exceed_count": int(np.count_nonzero(first_exceed)),
        "artifacts": {"stage_diffs": str(out / "stage_diffs.npz")},
    }
    if extra:
        report.update(extra)
    atomic_write_json(out / "error_report.json", report)
    return report


def _pixel_quant_change(trace_a: PathTrace, trace_b: PathTrace, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for index in trace_a.layers_value:
        mask |= (trace_a.layers_value[index].reshape(height, width) != trace_b.layers_value[index].reshape(height, width))
    return mask


def _pixel_mix_change(trace_a: PathTrace, trace_b: PathTrace) -> np.ndarray:
    first = next(iter(trace_a.channels))
    height, width = trace_a.channels[first].array.shape[:2]
    mask = np.zeros((height, width), dtype=bool)
    for name in trace_a.channels:
        mask |= np.any(trace_a.channels[name].array != trace_b.channels[name].array, axis=2)
    return mask


def _pixel_tie(trace_a: PathTrace, trace_b: PathTrace, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for index in trace_a.layer_tie_masks:
        mask |= trace_a.layer_tie_masks[index].reshape(height, width)
        mask |= trace_b.layer_tie_masks[index].reshape(height, width)
    for name in trace_a.channels:
        mask |= trace_a.channels[name].tie_mask.max(axis=2)
        mask |= trace_b.channels[name].tie_mask.max(axis=2)
    return mask


def compare_quant_steps(
    config: RenderConfig,
    material: MaterialConfig,
    texture: TextureData,
    quant_steps: tuple[int, int],
    output_dir: str | Path,
) -> dict[str, Any]:
    step_a, step_b = (int(quant_steps[0]), int(quant_steps[1]))
    trace_a = compute_path(
        config,
        material,
        texture,
        frac_bits=config.fixed.fraction_bits,
        quantize=True,
        quant_step=step_a,
    )
    trace_b = compute_path(
        config,
        material,
        texture,
        frac_bits=config.fixed.fraction_bits,
        quantize=True,
        quant_step=step_b,
    )
    height, width = texture.height, texture.width
    mismatch = np.any(trace_a.composite != trace_b.composite, axis=2)
    quant_change = _pixel_quant_change(trace_a, trace_b, height, width)
    mix_change = _pixel_mix_change(trace_a, trace_b)
    tie = _pixel_tie(trace_a, trace_b, height, width)
    classification = np.zeros((height, width), dtype=np.uint8)
    boundary = mismatch & tie
    quantization = mismatch & ~boundary & quant_change
    mixing = mismatch & ~boundary & ~quantization & mix_change
    composition = mismatch & ~boundary & ~quantization & ~mixing
    classification[boundary] = 1
    classification[quantization] = 2
    classification[mixing] = 3
    classification[composition] = 4
    class_names = {0: None, 1: "boundary_decision", 2: "quantization_rounding", 3: "mixing_rounding", 4: "channel_composition"}
    counts = {class_names[index]: int(np.count_nonzero(classification == index)) for index in range(1, 5)}

    synthesis_deviation = np.zeros((height, width), dtype=np.int64)
    for index in trace_a.layers_raw:
        synthesis_deviation = np.maximum(
            synthesis_deviation,
            np.abs(trace_a.layers_raw[index].reshape(height, width) - trace_b.layers_raw[index].reshape(height, width)),
        )
    quantization_deviation = np.zeros((height, width), dtype=np.int64)
    for index in trace_a.layers_value:
        quantization_deviation = np.maximum(
            quantization_deviation,
            np.abs(trace_a.layers_value[index].reshape(height, width) - trace_b.layers_value[index].reshape(height, width)),
        )
    mixing_deviation = np.zeros((height, width), dtype=np.int64)
    for name in trace_a.channels:
        mixing_deviation = np.maximum(
            mixing_deviation,
            np.abs(trace_a.channels[name].array - trace_b.channels[name].array).max(axis=2),
        )
    composite_deviation = np.abs(trace_a.composite - trace_b.composite).max(axis=2)

    samples: list[dict[str, Any]] = []
    for y, x in np.argwhere(mismatch)[:64]:
        synth_dev = max(
            (int(abs(int(trace_a.layers_raw[index][y * width + x]) - int(trace_b.layers_raw[index][y * width + x]))) for index in trace_a.layers_raw),
            default=0,
        )
        quant_dev = max(
            (int(abs(int(trace_a.layers_value[index][y * width + x]) - int(trace_b.layers_value[index][y * width + x]))) for index in trace_a.layers_value),
            default=0,
        )
        mix_dev = max(
            (int(np.abs(trace_a.channels[name].array[y, x] - trace_b.channels[name].array[y, x]).max()) for name in trace_a.channels),
            default=0,
        )
        compose_dev = int(np.abs(trace_a.composite[y, x] - trace_b.composite[y, x]).max())
        samples.append(
            {
                "x": int(x),
                "y": int(y),
                "classification": class_names[int(classification[y, x])],
                "synthesis_deviation": synth_dev,
                "quantization_deviation": quant_dev,
                "mixing_deviation": mix_dev,
                "composite_deviation": compose_dev,
                "quant_a": [int(value) for value in trace_a.composite[y, x]],
                "quant_b": [int(value) for value in trace_b.composite[y, x]],
            }
        )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _save_npz_atomic(
        out / "quant_compare.npz",
        {
            "classification": classification,
            "mismatch": mismatch.astype(np.uint8),
            "quantization_change": quant_change.astype(np.uint8),
            "mixing_change": mix_change.astype(np.uint8),
            "tie_mask": tie.astype(np.uint8),
            "quant_a": trace_a.composite,
            "quant_b": trace_b.composite,
            "synthesis_deviation": synthesis_deviation,
            "quantization_deviation": quantization_deviation,
            "mixing_deviation": mixing_deviation,
            "composite_deviation": composite_deviation,
        },
    )
    report = {
        "kind": "quant_step_comparison",
        "quant_steps": [step_a, step_b],
        "fraction_bits": config.fixed.fraction_bits,
        "mismatch_pixels": int(np.count_nonzero(mismatch)),
        "classifications": counts,
        "stage_deviation_summary": {
            "layer_synthesis": _stats(synthesis_deviation),
            "layer_quantization": _stats(quantization_deviation),
            "material_mixing": _stats(mixing_deviation),
            "channel_composition": _stats(composite_deviation),
        },
        "samples": samples,
        "artifacts": {"quant_compare": str(out / "quant_compare.npz")},
    }
    atomic_write_json(out / "quant_compare.json", report)
    return report


def build_diagnostics(
    config: RenderConfig,
    material: MaterialConfig,
    texture: TextureData,
    output_dir: str | Path,
    *,
    quant_steps: tuple[int, int] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = Path(output_dir)
    error_report = build_error_report(config, material, texture, out, extra=extra)
    comparison = None
    if quant_steps is not None:
        comparison = compare_quant_steps(config, material, texture, quant_steps, out)
    combined = {
        "ok": True,
        "config_name": config.name,
        "texture": texture.summary(),
        "error_propagation": error_report,
        "quant_comparison": comparison,
    }
    if extra:
        combined.update(extra)
    atomic_write_json(out / "diagnostics.json", combined)
    return combined
