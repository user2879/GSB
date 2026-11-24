"""Fixed-point material channel extraction, mixing, and boundary evidence."""

from __future__ import annotations

from typing import Any

import numpy as np

from .cache import CacheEntry
from .fixed import CHANNEL_LIMITS, CHANNELS, extract_channel, round_div
from .models import LayerArtifact, Material, MaterialChannelArtifact, MaterialLayer


def _weight_raw(weight: float, fraction_bits: int) -> int:
    scaled = weight * float(1 << fraction_bits)
    return int(np.floor(scaled + 0.5))


def _clamp_unit(values: np.ndarray, scale: int) -> np.ndarray:
    return np.clip(values, 0, scale)


def _unsigned(values: np.ndarray, scale: int, channel: str) -> np.ndarray:
    if channel == "normal_perturbation":
        return _clamp_unit((values + scale + 1) // 2, scale)
    return _clamp_unit(values, scale)


def _signed(values: np.ndarray, scale: int, channel: str) -> np.ndarray:
    if channel == "normal_perturbation":
        return np.clip(values * 2 - scale, -scale, scale)
    return _clamp_unit(values, scale)


def _combine(
    mode: str,
    current: np.ndarray,
    incoming: np.ndarray,
    weight: int,
    total: int,
    scale: int,
    rounding: str,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "multiply":
        combined = round_div(current * incoming, scale, rounding)
    elif mode == "screen":
        combined = scale - round_div((scale - current) * (scale - incoming), scale, rounding)
    elif mode == "overlay":
        low = round_div(2 * current * incoming, scale, rounding)
        high = scale - round_div(2 * (scale - current) * (scale - incoming), scale, rounding)
        combined = np.where(current * 2 < scale, low, high)
    else:
        raise ValueError(f"unsupported sequential mix mode: {mode}")
    if total <= 0:
        return combined, None  # type: ignore[return-value]
    numerator = current * (total - weight) + combined * weight
    return round_div(numerator, total, rounding), numerator


def mix_channel(
    *,
    material: Material,
    channel: str,
    layers: dict[int, LayerArtifact],
    key: str,
    scale: int,
    weight_fraction_bits: int,
    mix_mode: str,
    rounding: str,
    max_boundary_samples: int,
    reverse_layers: bool = False,
    row_start: int = 0,
    row_end: int | None = None,
    col_start: int = 0,
    col_end: int | None = None,
) -> MaterialChannelArtifact:
    candidates: list[MaterialLayer] = [
        layer
        for layer in sorted(material.layers, key=lambda row: row.layer_index)
        if layer.weights[channel] > 0
    ]
    if reverse_layers:
        candidates = list(reversed(candidates))
    if not candidates:
        raise ValueError(f"material {material.material_id} has no positive {channel} weight")

    weight_scale = 1 << weight_fraction_bits
    weight_values = [_weight_raw(layer.weights[channel], weight_fraction_bits) for layer in candidates]
    total_weight = int(sum(weight_values))
    if total_weight <= 0:
        raise ValueError(f"material {material.material_id} has zero fixed {channel} weight")

    if row_end is None:
        row_end = int(layers[candidates[0].layer_index].quantized.shape[0])
    if col_end is None:
        col_end = int(layers[candidates[0].layer_index].quantized.shape[1])
    if not (0 <= row_start < row_end <= layers[candidates[0].layer_index].quantized.shape[0]):
        raise ValueError("invalid row range for channel mixing")
    if not (0 <= col_start < col_end <= layers[candidates[0].layer_index].quantized.shape[1]):
        raise ValueError("invalid column range for channel mixing")

    boundary_records: list[tuple[str, np.ndarray, int, str]] = []
    contributions: dict[int, np.ndarray] = {}
    extracted: list[np.ndarray] = []
    weighted_values: list[np.ndarray] = []
    for layer, weight in zip(candidates, weight_values):
        field = layers[layer.layer_index].quantized[row_start:row_end, col_start:col_end]
        values = extract_channel(field, scale, channel, rounding)
        weighted = values * weight
        contribution = round_div(weighted, total_weight, rounding)
        boundary_records.append(
            ("contribution_rounding_half_boundary", weighted, total_weight, rounding)
        )
        contributions[layer.layer_index] = contribution
        extracted.append(values)
        weighted_values.append(weighted)

    if mix_mode == "weighted":
        numerator = np.zeros_like(weighted_values[0], dtype=np.int64)
        for weighted in weighted_values:
            numerator = numerator + weighted
        mixed = round_div(numerator, total_weight, rounding)
        boundary_records.append(("rounding_half_boundary", numerator, total_weight, rounding))
    else:
        unsigned_values = [_unsigned(values, scale, channel) for values in extracted]
        current = unsigned_values[0].copy()
        for index in range(1, len(unsigned_values)):
            current, blend_numerator = _combine(
                mix_mode,
                current,
                unsigned_values[index],
                weight_values[index],
                total_weight,
                scale,
                rounding,
            )
            if blend_numerator is not None:
                boundary_records.append(
                    ("sequential_rounding_half_boundary", blend_numerator, total_weight, rounding)
                )
        mixed = _signed(current, scale, channel)

    low, high = CHANNEL_LIMITS[channel]
    fixed_low = int(round(low * scale))
    fixed_high = int(round(high * scale))
    final = np.clip(mixed, fixed_low, fixed_high).astype(np.int64)

    samples: list[dict[str, Any]] = []
    boundary_total = 0
    max_weight = max(weight_values)
    tied_layers = [layer.layer_index for layer, weight in zip(candidates, weight_values) if weight == max_weight]
    if len(tied_layers) > 1 and row_start == 0 and col_start == 0:
        boundary_total += 1
        if len(samples) < max_boundary_samples:
            samples.append(
                {
                    "kind": "equal_weight_tiebreak",
                    "material": material.material_id,
                    "channel": channel,
                    "scope": "material",
                    "tied_layers": tied_layers,
                    "chosen_layer": min(tied_layers),
                    "weight_raw": max_weight,
                    "denominator": total_weight,
                }
            )

    for kind, boundary_numerator, denominator, boundary_rounding in boundary_records:
        remainder = np.abs(boundary_numerator) % denominator
        boundary_mask = (remainder != 0) & (2 * remainder == denominator)
        rows, cols = np.nonzero(boundary_mask)
        boundary_total += int(rows.size)
        remaining = max(0, max_boundary_samples - len(samples))
        for row, col in zip(rows[:remaining], cols[:remaining]):
            value = int(boundary_numerator[row, col])
            samples.append(
                {
                    "kind": kind,
                    "material": material.material_id,
                    "channel": channel,
                    "scope": "pixel",
                    "row": int(row + row_start),
                    "col": int(col + col_start),
                    "numerator": value,
                    "denominator": int(denominator),
                    "remainder": int(abs(value) % denominator),
                    "rounding": boundary_rounding,
                    "chosen_value": int(final[row, col]),
                }
            )

    return MaterialChannelArtifact(
        key=key,
        material_id=material.material_id,
        channel=channel,
        layer_keys=tuple(layers[layer.layer_index].key for layer in candidates),
        layer_indices=tuple(layer.layer_index for layer in candidates),
        mixed=mixed.astype(np.int64),
        final=final,
        contributions=contributions,
        boundary_samples=samples,
        boundary_total=boundary_total,
    )


def artifact_from_cache(
    entry: CacheEntry,
    *,
    material_id: str,
    channel: str,
) -> MaterialChannelArtifact:
    meta = entry.meta
    contributions = {
        int(index): np.asarray(entry.arrays[f"contribution_{index}"], dtype=np.int64)
        for index in meta["layer_indices"]
        if f"contribution_{index}" in entry.arrays
    }
    return MaterialChannelArtifact(
        key=entry.key,
        material_id=material_id,
        channel=channel,
        layer_keys=tuple(meta["layer_keys"]),
        layer_indices=tuple(int(item) for item in meta["layer_indices"]),
        mixed=np.asarray(entry.arrays["mixed"], dtype=np.int64),
        final=np.asarray(entry.arrays["final"], dtype=np.int64),
        contributions=contributions,
        boundary_samples=list(meta.get("boundary_samples", [])),
        boundary_total=int(meta.get("boundary_total", 0)),
        reused=True,
    )


def cache_payload(artifact: MaterialChannelArtifact) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    arrays = {
        "mixed": artifact.mixed,
        "final": artifact.final,
    }
    for index, values in artifact.contributions.items():
        arrays[f"contribution_{index}"] = values
    meta = {
        "material_id": artifact.material_id,
        "channel": artifact.channel,
        "layer_keys": list(artifact.layer_keys),
        "layer_indices": list(artifact.layer_indices),
        "boundary_samples": artifact.boundary_samples,
        "boundary_total": artifact.boundary_total,
    }
    return meta, arrays
