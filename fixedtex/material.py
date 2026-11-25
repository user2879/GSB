"""Material channel mapping, per-channel fixed-point mixing and composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from .fixed import (
    CHANNEL_ORDER,
    DECISION_TIE_DOWN,
    DECISION_TIE_UP,
    clamp_array,
    fixed_to_u8,
    round_ratio_array,
    scale_for,
    u8_to_fixed,
)
from .models import ChannelArtifact, ChannelSpec, MaterialConfig, RenderConfig, TextureData


@dataclass
class MixResult:
    array: np.ndarray
    tie_mask: np.ndarray
    remainder_mask: np.ndarray
    tie_count: int
    rounding_count: int
    equal_weight_count: int


def base_channel_arrays(
    texture: TextureData, config: RenderConfig, frac_bits: int | None = None
) -> dict[str, np.ndarray]:
    bits = config.fixed.fraction_bits if frac_bits is None else int(frac_bits)
    scale = scale_for(bits)
    rgb = u8_to_fixed(texture.rgb, bits)
    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]
    luma = (299 * r + 587 * g + 114 * b) // 1000
    roughness = clamp_array(scale - luma, 0, scale)[:, :, None]
    metallic = clamp_array(np.max(rgb, axis=2) - np.min(rgb, axis=2), 0, scale)[:, :, None]

    left = np.roll(luma, 1, axis=1)
    right = np.roll(luma, -1, axis=1)
    up = np.roll(luma, 1, axis=0)
    down = np.roll(luma, -1, axis=0)
    gradient_x = (right - left) // 2
    gradient_y = (down - up) // 2
    normal = np.stack(
        [clamp_array(gradient_x + scale // 2, 0, scale), clamp_array(gradient_y + scale // 2, 0, scale), np.full_like(luma, scale)],
        axis=2,
    )
    return {
        "diffuse": rgb.astype(np.int64, copy=False),
        "roughness": roughness.astype(np.int64, copy=False),
        "metallic": metallic.astype(np.int64, copy=False),
        "normal": normal.astype(np.int64, copy=False),
    }


def _component_gradient(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left = np.roll(values, 1, axis=1)
    right = np.roll(values, -1, axis=1)
    up = np.roll(values, 1, axis=0)
    down = np.roll(values, -1, axis=0)
    return (right - left) // 2, (down - up) // 2


def layer_channel_contribution(
    channel_name: str,
    base: np.ndarray,
    layer_value: np.ndarray,
    frac_bits: int,
) -> np.ndarray:
    scale = scale_for(frac_bits)
    layer = np.asarray(layer_value, dtype=np.int64)
    if channel_name == "diffuse":
        delta = layer[:, :, None] - scale // 2
        return clamp_array(base + delta, 0, scale)
    if channel_name in {"roughness", "metallic"}:
        return clamp_array(layer[:, :, None], 0, scale)
    if channel_name == "normal":
        gx, gy = _component_gradient(layer)
        return np.stack(
            [
                clamp_array(gx + scale // 2, 0, scale),
                clamp_array(gy + scale // 2, 0, scale),
                np.full_like(layer, scale),
            ],
            axis=2,
        )
    raise ValueError(f"unknown channel {channel_name}")


def _round_verbose(numerator: np.ndarray, denominator: int, tie_rule: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    number = np.asarray(numerator, dtype=np.int64)
    quotient = np.floor_divide(number, denominator)
    remainder = number - quotient * denominator
    rounded, decisions = round_ratio_array(number, denominator, tie_rule)
    return rounded, decisions, remainder != 0


def _tie_mask_from_decisions(decisions: np.ndarray) -> np.ndarray:
    return (decisions == DECISION_TIE_DOWN) | (decisions == DECISION_TIE_UP)


def mix_channel(
    channel: ChannelSpec,
    base: np.ndarray,
    referenced_layers: dict[int, np.ndarray],
    config: RenderConfig,
    material: MaterialConfig,
    frac_bits: int | None = None,
) -> MixResult:
    """Mix one channel using only its declared layer references.

    The function deliberately has no access to the material's other channels.
    """

    bits = config.fixed.fraction_bits if frac_bits is None else int(frac_bits)
    scale = scale_for(bits)
    base_array = np.asarray(base, dtype=np.int64)
    contributions: list[tuple[int, int, np.ndarray]] = [(-1, channel.base_weight_q, base_array)]
    equal_weight_count = 0
    active_weights = [item.weight_q for item in channel.layer_weights if item.weight_q > 0]
    if len(active_weights) != len(set(active_weights)):
        equal_weight_count = len(active_weights) - len(set(active_weights))
    for item in channel.layer_weights:
        if item.index not in referenced_layers:
            raise KeyError(f"channel {channel.name} missing referenced layer {item.index}")
        value = layer_channel_contribution(
            channel.name,
            base_array,
            referenced_layers[item.index],
            bits,
        )
        contributions.append((item.index, item.weight_q, value))

    total_weight = sum(item[1] for item in contributions)
    if total_weight <= 0:
        raise ValueError(f"channel {channel.name} has zero total weight")

    if config.blend_mode == "weighted":
        numerator = np.zeros_like(base_array, dtype=np.int64)
        for _, weight, value in contributions:
            numerator += value * weight
        result, decisions, remainder_mask = _round_verbose(numerator, total_weight, material.tie_rule)
        tie_mask = _tie_mask_from_decisions(decisions)
        rounding_count = int(np.count_nonzero(remainder_mask))
    elif config.blend_mode == "max":
        scaled_contributions: list[tuple[int, int, np.ndarray, np.ndarray]] = []
        all_tie = np.zeros(base_array.shape, dtype=bool)
        all_remainder = np.zeros(base_array.shape, dtype=bool)
        for layer_index, weight, value in contributions:
            if weight <= 0:
                continue
            scaled, scale_decisions, scale_remainder = _round_verbose(
                value * weight, 1000, material.tie_rule
            )
            scaled = clamp_array(scaled, 0, scale)
            scaled_contributions.append((layer_index, weight, scaled, value))
            all_tie |= _tie_mask_from_decisions(scale_decisions)
            all_remainder |= scale_remainder
        if not scaled_contributions:
            raise ValueError(f"channel {channel.name} has no active weights")
        result = scaled_contributions[0][2].copy()
        tie_mask = np.zeros(result.shape, dtype=bool)
        remainder_mask = all_remainder.copy()
        source_index = np.full(result.shape, scaled_contributions[0][0], dtype=np.int64)
        for layer_index, weight, value, _ in scaled_contributions[1:]:
            greater = value > result
            equal = value == result
            result = np.where(greater, value, result)
            source_index = np.where(greater, layer_index, source_index)
            tie_mask |= equal
            if material.equal_weight_rule == "upper_layer_index":
                take_equal = equal
            else:
                take_equal = np.zeros_like(equal)
            result = np.where(take_equal, value, result)
            source_index = np.where(take_equal, layer_index, source_index)
        tie_mask |= all_tie
        rounding_count = int(np.count_nonzero(remainder_mask))
    elif config.blend_mode == "layered_over":
        result = base_array.copy()
        tie_mask = np.zeros(result.shape, dtype=bool)
        remainder_mask = np.zeros(result.shape, dtype=bool)
        for _, weight, value in contributions[1:]:
            if weight <= 0:
                continue
            numerator = result * (total_weight - weight) + value * weight
            result, decisions, step_remainder = _round_verbose(numerator, total_weight, material.tie_rule)
            tie_mask |= _tie_mask_from_decisions(decisions)
            remainder_mask |= step_remainder
        rounding_count = int(np.count_nonzero(remainder_mask))
    else:
        raise ValueError(f"unsupported blend mode {config.blend_mode}")

    result = clamp_array(result, 0, scale)
    return MixResult(
        array=result,
        tie_mask=tie_mask,
        remainder_mask=remainder_mask,
        tie_count=int(np.count_nonzero(tie_mask)),
        rounding_count=rounding_count,
        equal_weight_count=equal_weight_count,
    )


def compose_channels(
    channels: dict[str, np.ndarray],
    config: RenderConfig,
    frac_bits: int | None = None,
) -> np.ndarray:
    bits = config.fixed.fraction_bits if frac_bits is None else int(frac_bits)
    scale = scale_for(bits)
    diffuse = np.asarray(channels["diffuse"], dtype=np.int64)
    roughness = np.asarray(channels["roughness"], dtype=np.int64)[:, :, 0]
    metallic = np.asarray(channels["metallic"], dtype=np.int64)[:, :, 0]
    normal = np.asarray(channels["normal"], dtype=np.int64)

    specular = ((scale - roughness) * metallic) // scale
    perturb_x = (normal[:, :, 0] - scale // 2) // 16
    perturb_y = (normal[:, :, 1] - scale // 2) // 16
    red = diffuse[:, :, 0] + perturb_x + specular // 4
    green = diffuse[:, :, 1] + perturb_y + specular // 2
    blue = diffuse[:, :, 2] - specular // 8
    alpha = (roughness + metallic) // 2
    rgba = np.stack(
        [
            clamp_array(red, 0, scale),
            clamp_array(green, 0, scale),
            clamp_array(blue, 0, scale),
            clamp_array(alpha, 0, scale),
        ],
        axis=2,
    )
    return rgba.astype(np.int64, copy=False)


def build_channel_artifact(
    channel: ChannelSpec,
    mix: MixResult,
    key: str,
    layer_keys: dict[int, str],
    config: RenderConfig,
    parameter_key: str = "",
) -> ChannelArtifact:
    return ChannelArtifact(
        name=channel.name,
        key=key,
        layer_keys=tuple(layer_keys[index] for index in channel.layer_indices),
        layer_indices=channel.layer_indices,
        array=mix.array,
        frac_bits=config.fixed.fraction_bits,
        quant_step=config.fixed.quant_step,
        tie_count=mix.tie_count,
        rounding_count=mix.rounding_count,
        parameter_key=parameter_key,
        blend_mode=config.blend_mode,
    )


def channel_png(array: np.ndarray, frac_bits: int) -> Image.Image:
    values = np.asarray(array, dtype=np.int64)
    if values.ndim == 2:
        values = values[:, :, None]
    if values.shape[2] == 1:
        values = np.repeat(values, 3, axis=2)
    if values.shape[2] >= 3:
        rgb = fixed_to_u8(values[:, :, :3], frac_bits)
    else:
        raise ValueError("channel array must have at least one component")
    return Image.fromarray(rgb, mode="RGB")


def composite_png(array: np.ndarray, frac_bits: int) -> Image.Image:
    rgba = fixed_to_u8(np.asarray(array, dtype=np.int64), frac_bits)
    return Image.fromarray(rgba, mode="RGBA")


def channel_summary(artifact: ChannelArtifact) -> dict[str, Any]:
    array = artifact.array
    return {
        "name": artifact.name,
        "key": artifact.key,
        "shape": list(array.shape),
        "min": int(array.min()) if array.size else 0,
        "max": int(array.max()) if array.size else 0,
        "mean_x1000": int((int(array.sum()) * 1000) // max(1, array.size)),
        "layer_indices": list(artifact.layer_indices),
        "layer_keys": list(artifact.layer_keys),
        "tie_count": artifact.tie_count,
        "rounding_count": artifact.rounding_count,
    }
