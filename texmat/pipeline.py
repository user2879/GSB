"""Deterministic noise stacking and four-channel material synthesis.

Float only appears at one explicit parameter-quantization point and in the
coordinate-derived raw noise (immediately quantized).  Every subsequent value
is integer fixed-point math.

Quantization switch semantics
-----------------------------
Layer accumulation always runs at a wide working segment ``Q(F_work)`` so the
intermediate fixed-point segment genuinely differs when quantization is off:

* ``quantize=true``  -> each layer is narrowed to the configured ``Q(F)`` and
  the next layer accumulates on that narrowed field (error is committed early);
* ``quantize=false`` -> the wide ``Q(F_work)`` field is carried untouched and
  narrowing to ``Q(F)`` only happens inside channel composition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CHANNELS, GenerationConfig, LayerSpec, MaterialSpec
from .fixedpoint import qdiv, qmul, quantize_float, rescale_array, saturate_i32, to_uint8
from .hashing import array_fingerprint, combine_fingerprints
from .loader import DecodedTexture, luminance
from .noise import value_noise

WORK_BITS = 16
ONE_CACHE: dict[int, int] = {}


def one_q(f_bits: int) -> int:
    if f_bits not in ONE_CACHE:
        ONE_CACHE[f_bits] = 1 << f_bits
    return ONE_CACHE[f_bits]


@dataclass
class LayerResult:
    spec: LayerSpec
    accum: np.ndarray          # Q(F_work) wide accumulated field after this layer
    quantized: np.ndarray      # Q(F) narrowed field (valid only when quantized)
    is_quantized: bool
    noise_q: np.ndarray        # Q(F_work) quantized noise contribution
    summary: int               # deterministic checksum of the stored accumulator
    fingerprint: int


@dataclass
class ChannelResult:
    material: str
    channel: str
    values: np.ndarray         # Q(F) composed channel pixels
    contributions: np.ndarray  # (layers, H, W) Q(F) per-layer contributions
    boundary_pixels: list[tuple[int, int]]
    tie_pixels: list[tuple[int, int]]
    fingerprint: int


@dataclass
class MaterialResult:
    material: str
    channels: dict[str, ChannelResult]
    rgba: np.ndarray           # (H, W, 4) uint8
    fingerprint: int


def base_height(texture: DecodedTexture, f_bits: int, rounding: str) -> np.ndarray:
    """Convert a uint8 base texture into a Q(F) height field in [0, 1]."""
    luma = luminance(texture.pixels).astype(np.int64)
    scaled = luma * np.int64(one_q(f_bits))
    return qdiv(scaled, 255, f_bits, rounding)


def layer_fingerprint(spec: LayerSpec, base_fp: int, f_bits: int, quantize: bool,
                      rounding: str, resolution: int = 0) -> int:
    return combine_fingerprints(
        "layer", spec.index, base_fp, spec.fingerprint, spec.frequency, spec.amplitude,
        f_bits, WORK_BITS, quantize, rounding, resolution,
    )


def compute_layer(spec: LayerSpec, texture: DecodedTexture, previous_wide: np.ndarray | None,
                  config: GenerationConfig, f_bits: int, quantize: bool,
                  rounding: str) -> LayerResult:
    if previous_wide is None:
        previous_wide = base_height(texture, WORK_BITS, rounding)
    raw_noise = value_noise(texture.width, texture.height, spec.frequency,
                            fingerprint=spec.fingerprint, seed=spec.seed)
    amplitude_q = quantize_float(spec.amplitude, WORK_BITS, rounding)
    noise_q = np.rint(raw_noise.astype(np.float64) * float(amplitude_q)).astype(np.int64)
    wide = saturate_i32(previous_wide.astype(np.int64) + noise_q)
    if quantize:
        narrow = saturate_i32(rescale_array(wide, WORK_BITS, f_bits, rounding))
        stored = rescale_array(narrow, f_bits, WORK_BITS, rounding)
        is_quantized = True
    else:
        narrow = np.empty(0, dtype=np.int64)
        stored = wide
        is_quantized = False
    checksum_source = narrow if is_quantized else stored
    checksum = int(np.bitwise_xor.reduce(checksum_source.astype(np.uint32).ravel(),
                                         initial=np.uint32(0)))
    fp = layer_fingerprint(spec, texture.content_fingerprint, f_bits, quantize,
                           rounding, texture.width)
    return LayerResult(spec=spec, accum=stored, quantized=narrow, is_quantized=is_quantized,
                       noise_q=noise_q, summary=checksum, fingerprint=fp)


def _weights_q(weights: tuple[float, ...], f_bits: int, rounding: str) -> np.ndarray:
    total = sum(weights)
    normalized = [w / total for w in weights]
    return np.array([quantize_float(w, f_bits, rounding) for w in normalized], dtype=np.int64)


def _rounding_boundary(numerator: np.ndarray, den: int, rounding: str) -> np.ndarray:
    """Boolean mask of pixels whose |num|/den division sits on a rounding edge."""
    rem = np.abs(numerator) % den
    if rounding == "truncate":
        return np.zeros(numerator.shape, dtype=bool)
    if rounding == "half_up":
        return rem * 2 >= den
    return rem * 2 == den


def _weighted_blend(contrib: np.ndarray, weights_q: np.ndarray, f_bits: int,
                    rounding: str) -> tuple[np.ndarray, np.ndarray]:
    scaled_sum = np.zeros(contrib.shape[1:], dtype=np.int64)
    weight_sum = int(weights_q.sum())
    for idx in range(contrib.shape[0]):
        scaled_sum += qmul(np.broadcast_to(weights_q[idx], contrib.shape[1:]),
                           contrib[idx], f_bits, rounding)
    if weight_sum <= 0:
        return np.zeros_like(scaled_sum), np.zeros(scaled_sum.shape, dtype=bool)
    result = qdiv(scaled_sum, weight_sum, f_bits, rounding)
    boundary = _rounding_boundary(scaled_sum, weight_sum, rounding)
    return result, boundary


def _additive_blend(contrib: np.ndarray, weights_q: np.ndarray, f_bits: int,
                    rounding: str) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros(contrib.shape[1:], dtype=np.int64)
    boundary = np.zeros(contrib.shape[1:], dtype=bool)
    for idx in range(contrib.shape[0]):
        product, edge = _qmul_with_boundary(
            np.broadcast_to(weights_q[idx], contrib.shape[1:]), contrib[idx], f_bits, rounding)
        total += product
        boundary |= edge
    return np.clip(total, 0, one_q(f_bits) - 1), boundary


def _qmul_with_boundary(a: np.ndarray, b: np.ndarray, f_bits: int,
                        rounding: str) -> tuple[np.ndarray, np.ndarray]:
    product = np.asarray(a, dtype=np.int64) * np.asarray(b, dtype=np.int64)
    den = 1 << f_bits
    result = qmul(a, b, f_bits, rounding)
    return result, _rounding_boundary(product, den, rounding)


def _multiply_blend(contrib: np.ndarray, f_bits: int,
                    rounding: str) -> tuple[np.ndarray, np.ndarray]:
    result = np.full(contrib.shape[1:], one_q(f_bits), dtype=np.int64)
    boundary = np.zeros(contrib.shape[1:], dtype=bool)
    for idx in range(contrib.shape[0]):
        product = result * contrib[idx]
        result = qmul(result, contrib[idx], f_bits, rounding)
        boundary |= _rounding_boundary(product, 1 << f_bits, rounding)
    return result, boundary


def compose_material(material: MaterialSpec, layer_results: dict[str, LayerResult],
                     config: GenerationConfig, f_bits: int, rounding: str,
                     blend_mode_override: str | None = None) -> MaterialResult:
    blend_mode = blend_mode_override or material.blend_mode or config.blend_mode
    weights_q = _weights_q(material.weights, f_bits, rounding)
    height, width = next(iter(layer_results.values())).accum.shape
    channel_results: dict[str, ChannelResult] = {}

    equal_weight = len(set(material.weights)) == 1
    tie_pixels = [(idx // width, idx % width) for idx in range(height * width)] if equal_weight else []

    for channel_name in CHANNELS:
        channel_spec = material.channels[channel_name]
        selected = [layer_results[layer_id] for layer_id in material.layers]
        contrib = np.empty((len(selected), height, width), dtype=np.int64)
        base_q = quantize_float(channel_spec.base, f_bits, rounding)
        scale_q = quantize_float(channel_spec.scale, f_bits, rounding)
        bias_q = quantize_float(channel_spec.bias, f_bits, rounding)
        for idx, layer_result in enumerate(selected):
            if layer_result.is_quantized:
                field_q = layer_result.quantized
            else:
                field_q = rescale_array(layer_result.accum, WORK_BITS, f_bits, rounding)
            term = base_q + qmul(np.broadcast_to(scale_q, field_q.shape),
                                 field_q, f_bits, rounding) + bias_q
            contrib[idx] = np.clip(term, 0, one_q(f_bits) - 1)
        if blend_mode == "weighted":
            values, boundary_mask = _weighted_blend(contrib, weights_q, f_bits, rounding)
        elif blend_mode == "additive":
            values, boundary_mask = _additive_blend(contrib, weights_q, f_bits, rounding)
        elif blend_mode == "multiply":
            values, boundary_mask = _multiply_blend(contrib, f_bits, rounding)
        else:  # validated earlier, defensive guard
            raise ValueError(f"unsupported blend mode: {blend_mode}")
        values = np.clip(values, 0, one_q(f_bits) - 1)
        rows, cols = np.nonzero(boundary_mask)
        boundary = sorted(zip(rows.tolist(), cols.tolist()))
        fp = combine_fingerprints("channel", material.name, channel_name, blend_mode,
                                  array_fingerprint(values), f_bits, rounding)
        channel_results[channel_name] = ChannelResult(
            material=material.name, channel=channel_name, values=values, contributions=contrib,
            boundary_pixels=boundary, tie_pixels=tie_pixels, fingerprint=fp,
        )

    rgba = np.stack([to_uint8(channel_results[c].values, f_bits, rounding) for c in CHANNELS], axis=-1)
    final_fp = combine_fingerprints(
        "final", material.name,
        *[channel_results[c].fingerprint for c in CHANNELS], blend_mode, f_bits,
    )
    return MaterialResult(material=material.name, channels=channel_results, rgba=rgba, fingerprint=final_fp)
