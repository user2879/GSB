"""Deterministic coordinate-hash value noise with integer-only sampling."""

from __future__ import annotations

import hashlib
from functools import lru_cache

import numpy as np

from .fixed import clamp_array, scale_for
from .hashing import canonical_json, content_key, layer_parameter_key, layer_parameter_payload
from .models import LayerSpec, RenderConfig, TextureData

U64 = np.uint64
MASK64 = U64(0xFFFFFFFFFFFFFFFF)
POS_BITS = 16
POS_SCALE = 1 << POS_BITS


def _mix64(values: np.ndarray | np.uint64) -> np.ndarray:
    value = np.asarray(values, dtype=np.uint64)
    value = (value + U64(0x9E3779B97F4A7C15)) & MASK64
    value = (value ^ (value >> U64(30))) * U64(0xBF58476D1CE4E5B9) & MASK64
    value = (value ^ (value >> U64(27))) * U64(0x94D049BB133111EB) & MASK64
    return (value ^ (value >> U64(31))) & MASK64


def layer_fingerprint(layer: LayerSpec, config: RenderConfig, texture: TextureData) -> str:
    return content_key(
        "layer-fingerprint",
        {
            "parameter_key": layer_parameter_key(layer, config, texture),
            "parameters": layer_parameter_payload(layer, config, texture),
        },
    )


def seed_from_fingerprint(fingerprint: str) -> int:
    return int(fingerprint[:16], 16) & ((1 << 63) - 1)


def _corner_values(
    cx: np.ndarray,
    cy: np.ndarray,
    seed: int,
    salt: int,
    frac_bits: int,
) -> np.ndarray:
    x = cx.astype(np.uint64, copy=False)
    y = cy.astype(np.uint64, copy=False)
    mixed = _mix64(
        U64(seed)
        ^ (x * U64(0xD6E8FEB86659FD93))
        ^ (y * U64(0xA5A3564E27F8862F))
        ^ U64(salt)
    )
    shift = U64(64 - frac_bits)
    return (mixed >> shift).astype(np.int64, copy=False)


def compute_layer_chunk(
    layer: LayerSpec,
    config: RenderConfig,
    texture: TextureData,
    start: int,
    end: int,
    *,
    frac_bits: int | None = None,
    fingerprint: str | None = None,
) -> np.ndarray:
    """Return raw fixed-point layer samples for flat indices ``[start, end)``."""

    bits = config.fixed.fraction_bits if frac_bits is None else int(frac_bits)
    scale = scale_for(bits)
    width = texture.width
    total = texture.width * texture.height
    if start < 0 or end < start or end > total:
        raise ValueError("invalid layer chunk bounds")
    if start == end:
        return np.empty(0, dtype=np.int64)

    flat = np.arange(start, end, dtype=np.int64)
    x = flat % width
    y = flat // width

    frequency_fp = max(1, (layer.frequency_milli * POS_SCALE) // 1000)
    x_fp = x * frequency_fp
    y_fp = y * frequency_fp
    cx = x_fp >> POS_BITS
    cy = y_fp >> POS_BITS
    fx = (x_fp & (POS_SCALE - 1)) * scale >> POS_BITS
    fy = (y_fp & (POS_SCALE - 1)) * scale >> POS_BITS

    fingerprint = fingerprint or layer_fingerprint(layer, config, texture)
    seed = seed_from_fingerprint(fingerprint)
    v00 = _corner_values(cx, cy, seed, 0, bits)
    v10 = _corner_values(cx + 1, cy, seed, 1, bits)
    v01 = _corner_values(cx, cy + 1, seed, 2, bits)
    v11 = _corner_values(cx + 1, cy + 1, seed, 3, bits)

    sx = scale - fx
    sy = scale - fy
    top = (v00 * sx + v10 * fx) >> bits
    bottom = (v01 * sx + v11 * fx) >> bits
    value = (top * sy + bottom * fy) >> bits

    amplitude_fp = max(1, (layer.amplitude_milli * POS_SCALE) // 1000)
    value = (value * amplitude_fp) >> POS_BITS
    return clamp_array(value, 0, scale)


def compute_layer_raw(
    layer: LayerSpec,
    config: RenderConfig,
    texture: TextureData,
    *,
    frac_bits: int | None = None,
    fingerprint: str | None = None,
) -> np.ndarray:
    return compute_layer_chunk(
        layer,
        config,
        texture,
        0,
        texture.width * texture.height,
        frac_bits=frac_bits,
        fingerprint=fingerprint,
    ).reshape(texture.height, texture.width)


def fingerprint_summary(layer: LayerSpec, config: RenderConfig, texture: TextureData) -> dict[str, object]:
    fingerprint = layer_fingerprint(layer, config, texture)
    return {
        "layer": layer.index,
        "fingerprint": fingerprint,
        "seed": seed_from_fingerprint(fingerprint),
        "parameters": layer.parameter_dict(),
        "texture_content_key": texture.content_key,
    }
