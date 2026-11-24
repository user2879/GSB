"""Deterministic fixed-point tile rendering and cross-level composition."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from .config import SceneConfig
from .kernels import evaluate_sh
from .util import hash_value

BACKGROUND = np.array([0.94, 0.955, 0.94], dtype=np.float64)


def quantize_values(values: np.ndarray | float, frac_bits: int, gear: str) -> np.ndarray:
    scale = float(1 << int(frac_bits))
    scaled = np.asarray(values, dtype=np.float64) * scale
    if gear == "NEAREST":
        result = np.floor(scaled + 0.5)
    elif gear == "FLOOR":
        result = np.floor(scaled)
    elif gear == "CEIL":
        result = np.ceil(scaled)
    else:  # pragma: no cover - config validation catches this
        raise ValueError(f"unsupported quantization gear: {gear}")
    return np.asarray(result, dtype=np.int64)


def _round_ratio(numerator: np.ndarray, denominator: np.ndarray, gear: str) -> np.ndarray:
    denominator = np.asarray(denominator, dtype=np.float64)
    safe = np.maximum(denominator, 1.0)
    ratio = np.asarray(numerator, dtype=np.float64) / safe
    if gear == "NEAREST":
        result = np.floor(ratio + 0.5)
    elif gear == "FLOOR":
        result = np.floor(ratio)
    else:
        result = np.ceil(ratio)
    return np.asarray(result, dtype=np.int64)


def _sample_offsets(sample_count: int) -> tuple[np.ndarray, np.ndarray]:
    result_x = np.zeros(sample_count, dtype=np.float64)
    result_y = np.zeros(sample_count, dtype=np.float64)
    for sample in range(sample_count):
        angle = 2.0 * math.pi * ((sample * 0.6180339887498949 + 0.17) % 1.0)
        radius = 0.28 * math.sqrt((sample + 0.5) / sample_count)
        result_x[sample] = radius * math.cos(angle)
        result_y[sample] = radius * math.sin(angle)
    return result_x, result_y


def _sh_direction(level: int, row: int, col: int) -> np.ndarray:
    direction = np.array(
        [
            math.sin(0.73 * (col + 1) + 0.41 * level),
            math.cos(1.07 * (row + 1) + 0.23 * level),
            0.85 + 0.05 * level,
        ],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return direction / norm


def _sh_direction_batch(level: int, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    directions = np.stack(
        [
            np.sin(0.73 * (cols.astype(np.float64) + 1.0) + 0.41 * level),
            np.cos(1.07 * (rows.astype(np.float64) + 1.0) + 0.23 * level),
            np.full(rows.shape, 0.85 + 0.05 * level, dtype=np.float64),
        ],
        axis=-1,
    )
    norms = np.linalg.norm(directions, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return directions / norms


def _evaluate_sh_batch(sh: np.ndarray, directions: np.ndarray, degree: int) -> np.ndarray:
    x = directions[:, 0]
    y = directions[:, 1]
    z = directions[:, 2]
    columns = [np.full_like(x, 0.28209479177387814)]
    if degree >= 1:
        columns.extend([0.4886025119029199 * y, 0.4886025119029199 * z, 0.4886025119029199 * x])
    if degree >= 2:
        columns.extend(
            [
                1.0925484305920792 * x * y,
                1.0925484305920792 * y * z,
                0.31539156525252005 * (3.0 * z * z - 1.0),
                1.0925484305920792 * x * z,
                0.5462742152960396 * (x * x - y * y),
            ]
        )
    if degree >= 3:
        columns.extend(
            [
                0.5900435899266435 * y * (3.0 * x * x - y * y),
                2.890611442640554 * x * y * z,
                0.4570457994644658 * y * (5.0 * z * z - 1.0),
                0.3731763325901154 * z * (5.0 * z * z - 3.0),
                0.4570457994644658 * x * (5.0 * z * z - 1.0),
                1.445305721320277 * z * (x * x - y * y),
                0.5900435899266435 * x * (x * x - 3.0 * y * y),
            ]
        )
    basis = np.stack(columns[: (degree + 1) ** 2], axis=-1)
    return np.clip(np.einsum("nk,nkc->nc", basis, sh), 0.0, 1.0)


def render_tile_arrays(
    config: SceneConfig,
    kernels: dict[str, np.ndarray],
    kernel_refs: Iterable[int],
    level: int,
    tile_row: int,
    tile_col: int,
    frac_bits: int | None = None,
    quant_gear: str | None = None,
) -> dict[str, np.ndarray]:
    frac_bits = int(config.frac_bits if frac_bits is None else frac_bits)
    quant_gear = str(config.quant_gear if quant_gear is None else quant_gear)
    reference_bits = frac_bits + 1
    scale = float(1 << frac_bits)
    reference_scale = float(1 << reference_bits)
    width, height = config.image_size
    tile_size = config.tile_size
    x0 = tile_col * tile_size
    y0 = tile_row * tile_size
    x1 = min(width, x0 + tile_size)
    y1 = min(height, y0 + tile_size)
    tile_h = max(0, y1 - y0)
    tile_w = max(0, x1 - x0)
    if tile_h == 0 or tile_w == 0:
        raise ValueError("tile is outside the image")
    yy, xx = np.mgrid[y0:y1, x0:x1]
    xx = xx.astype(np.float64) + 0.5
    yy = yy.astype(np.float64) + 0.5

    refs = np.asarray(list(kernel_refs), dtype=np.int64)
    if refs.size:
        depths = np.asarray(kernels["depth"])[refs]
        grid = np.asarray(kernels["grid"])[refs]
        order_local = np.lexsort((grid[:, 1], grid[:, 0], refs, -depths))
        refs = refs[order_local]
    else:
        refs = np.zeros((0,), dtype=np.int64)

    accum_num = np.zeros((tile_h, tile_w, 3), dtype=np.int64)
    accum_weight = np.zeros((tile_h, tile_w), dtype=np.int64)
    reference_num = np.zeros((tile_h, tile_w, 3), dtype=np.int64)
    reference_weight = np.zeros((tile_h, tile_w), dtype=np.int64)
    raw_num = np.zeros((tile_h, tile_w, 3), dtype=np.float64)
    raw_weight = np.zeros((tile_h, tile_w), dtype=np.float64)
    offsets_x, offsets_y = _sample_offsets(int(config.render_samples))
    coefficient_count = (config.sh_degree + 1) ** 2

    scales = np.asarray(kernels["scales"], dtype=np.float64)
    opacity = np.asarray(kernels["opacity"], dtype=np.float64)
    sh = np.asarray(kernels["sh"], dtype=np.float64)
    grid_all = np.asarray(kernels["grid"], dtype=np.int64)
    if sh.ndim != 3 or sh.shape[1:] != (coefficient_count, 3):
        raise ValueError("kernel SH shape does not match configured degree")

    grid_size = np.asarray(kernels.get("grid_size", np.array([np.max(grid_all[:, 1]) + 1, np.max(grid_all[:, 0]) + 1])))
    grid_w = int(grid_size[0])
    grid_h = int(grid_size[1])
    if refs.size:
        ref_rows = grid_all[refs, 0]
        ref_cols = grid_all[refs, 1]
        center_x = (ref_cols.astype(np.float64) + 0.5) * (float(width) / float(grid_w))
        center_y = (ref_rows.astype(np.float64) + 0.5) * (float(height) / float(grid_h))
        sigma_x = np.maximum(0.2, scales[refs, 0])[:, None, None, None]
        sigma_y = np.maximum(0.2, scales[refs, 1])[:, None, None, None]
        directions = _sh_direction_batch(level, ref_rows, ref_cols)
        colors = _evaluate_sh_batch(sh[refs], directions, config.sh_degree)
        dx = (xx[None, None, :, :] + offsets_x[None, :, None, None] - center_x[:, None, None, None]) / sigma_x
        dy = (yy[None, None, :, :] + offsets_y[None, :, None, None] - center_y[:, None, None, None]) / sigma_y
        weights = np.exp(-0.5 * (dx * dx + dy * dy))
        weights *= np.clip(opacity[refs], 0.0, 1.0)[:, None, None, None]
        weights /= float(config.render_samples)
        weights[weights <= 1e-10] = 0.0
        weight_q = quantize_values(weights, frac_bits, quant_gear)
        reference_weight_q = quantize_values(weights, reference_bits, quant_gear)
        color_q = quantize_values(colors, frac_bits, quant_gear)
        reference_color_q = quantize_values(colors, reference_bits, quant_gear)
        accum_weight = np.sum(weight_q, axis=(0, 1), dtype=np.int64)
        reference_weight = np.sum(reference_weight_q, axis=(0, 1), dtype=np.int64)
        accum_num = np.sum(weight_q[..., None] * color_q[:, None, None, None, :], axis=(0, 1), dtype=np.int64)
        reference_num = np.sum(
            reference_weight_q[..., None] * reference_color_q[:, None, None, None, :],
            axis=(0, 1),
            dtype=np.int64,
        )
        raw_weight = np.sum(weights, axis=(0, 1), dtype=np.float64)
        raw_num = np.sum(weights[..., None] * colors[:, None, None, None, :], axis=(0, 1), dtype=np.float64)

    background_q = quantize_values(BACKGROUND, frac_bits, quant_gear)
    background_reference_q = quantize_values(BACKGROUND, reference_bits, quant_gear)
    pixels_q = np.empty((tile_h, tile_w, 4), dtype=np.int64)
    reference_q = np.empty((tile_h, tile_w, 4), dtype=np.int64)
    raw_pixels = np.empty((tile_h, tile_w, 4), dtype=np.float64)
    has_weight = accum_weight > 0
    if np.any(has_weight):
        ratio_q = _round_ratio(accum_num[has_weight], accum_weight[has_weight, None], quant_gear)
        pixels_q[has_weight, :3] = np.clip(ratio_q, 0, int(scale))
        reference_ratio_q = _round_ratio(reference_num[has_weight], reference_weight[has_weight, None], quant_gear)
        reference_q[has_weight, :3] = np.clip(reference_ratio_q, 0, int(reference_scale))
        raw_pixels[has_weight, :3] = raw_num[has_weight] / raw_weight[has_weight, None]
    if np.any(~has_weight):
        pixels_q[~has_weight, :3] = background_q[None, :]
        reference_q[~has_weight, :3] = background_reference_q[None, :]
        raw_pixels[~has_weight, :3] = BACKGROUND[None, :]
    pixels_q[:, :, 3] = quantize_values(np.minimum(1.0, accum_weight / scale), frac_bits, quant_gear)
    reference_q[:, :, 3] = quantize_values(np.minimum(1.0, reference_weight / reference_scale), reference_bits, quant_gear)
    raw_pixels[:, :, 3] = np.minimum(1.0, raw_weight)
    pixels = pixels_q.astype(np.float64) / scale
    reference = reference_q.astype(np.float64) / reference_scale
    residual = pixels - raw_pixels
    round_dir = np.sign(residual).astype(np.int8)
    return {
        "pixels_q": pixels_q,
        "pixels": pixels,
        "reference_q": reference_q,
        "reference": reference,
        "residual": residual,
        "round_dir": round_dir,
        "weight_q": accum_weight,
        "reference_weight_q": reference_weight,
        "depth_order": refs.astype(np.int64),
    }


def tile_cache_key(
    config: SceneConfig,
    view_index: int,
    level: int,
    tile_row: int,
    tile_col: int,
    refs: Iterable[int],
    kernel_keys: Iterable[str],
) -> str:
    refs_list = list(int(item) for item in refs)
    keys_list = list(kernel_keys)
    payload = {
        "base_scale": config.base_scale,
        "frac_bits": config.frac_bits,
        "kernel_refs": refs_list,
        "kernel_keys": keys_list,
        "level": level,
        "quant_gear": config.quant_gear,
        "render_samples": config.render_samples,
        "sh_degree": config.sh_degree,
        "tile": [tile_row, tile_col],
        "tile_size": config.tile_size,
        "view": view_index,
    }
    return "tile:" + hash_value(payload)


def composite_cache_key(
    config: SceneConfig,
    view_index: int,
    level: int,
    tile_keys: Iterable[str],
    lower_composite_key: str | None,
) -> str:
    payload = {
        "blend": _blend_factor(config, level),
        "frac_bits": config.frac_bits,
        "level": level,
        "lower_composite_key": lower_composite_key,
        "quant_gear": config.quant_gear,
        "tile_keys": list(tile_keys),
        "view": view_index,
    }
    return "composite:" + hash_value(payload)


def _blend_factor(config: SceneConfig, level: int) -> float:
    if level == 0:
        return 1.0
    return float(min(0.85, max(0.05, config.effective_scales()[level] / config.base_scale)))


def build_composite_arrays(
    config: SceneConfig,
    level: int,
    tile_arrays: dict[tuple[int, int], dict[str, np.ndarray]],
    lower_arrays: dict[str, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    width, height = config.image_size
    tile_size = config.tile_size
    pixels_q = np.zeros((height, width, 4), dtype=np.int64)
    reference_q = np.zeros((height, width, 4), dtype=np.int64)
    weight_q = np.zeros((height, width), dtype=np.int64)
    reference_weight_q = np.zeros((height, width), dtype=np.int64)
    for (tile_row, tile_col), arrays in sorted(tile_arrays.items()):
        x0 = tile_col * tile_size
        y0 = tile_row * tile_size
        tile_h, tile_w = arrays["pixels_q"].shape[:2]
        pixels_q[y0 : y0 + tile_h, x0 : x0 + tile_w] = arrays["pixels_q"]
        reference_q[y0 : y0 + tile_h, x0 : x0 + tile_w] = arrays["reference_q"]
        weight_q[y0 : y0 + tile_h, x0 : x0 + tile_w] = arrays["weight_q"]
        reference_weight_q[y0 : y0 + tile_h, x0 : x0 + tile_w] = arrays["reference_weight_q"]
    scale = float(1 << config.frac_bits)
    reference_scale = float(1 << (config.frac_bits + 1))
    if level == 0 or lower_arrays is None:
        raw_pixels = pixels_q.astype(np.float64) / scale
        raw_reference = reference_q.astype(np.float64) / reference_scale
        residual = np.zeros_like(raw_pixels)
        round_dir = np.zeros_like(pixels_q, dtype=np.int8)
        return {
            "pixels_q": pixels_q,
            "pixels": raw_pixels,
            "reference_q": reference_q,
            "reference": raw_reference,
            "residual": residual,
            "round_dir": round_dir,
            "weight_q": weight_q,
            "reference_weight_q": reference_weight_q,
        }
    factor = _blend_factor(config, level)
    lower_pixels = lower_arrays["pixels"]
    lower_reference = lower_arrays["reference"]
    current_pixels = pixels_q.astype(np.float64) / scale
    current_reference = reference_q.astype(np.float64) / reference_scale
    raw_pixels = lower_pixels * (1.0 - factor) + current_pixels * factor
    raw_reference = lower_reference * (1.0 - factor) + current_reference * factor
    composed_q = quantize_values(raw_pixels, config.frac_bits, config.quant_gear)
    composed_reference_q = quantize_values(raw_reference, config.frac_bits + 1, config.quant_gear)
    composed = composed_q.astype(np.float64) / scale
    composed_reference = composed_reference_q.astype(np.float64) / reference_scale
    residual = composed - raw_pixels
    return {
        "pixels_q": composed_q,
        "pixels": composed,
        "reference_q": composed_reference_q,
        "reference": composed_reference,
        "residual": residual,
        "round_dir": np.sign(residual).astype(np.int8),
        "weight_q": weight_q,
        "reference_weight_q": reference_weight_q,
    }
