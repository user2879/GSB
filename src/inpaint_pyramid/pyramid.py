"""Deterministic block-mean downsampling and bilinear upsampling."""

from __future__ import annotations

import numpy as np


UPSAMPLE_METADATA = {
    "method": "bilinear_clamped_edge",
    "coordinate_mapping": "src=(dst+0.5)*src_size/dst_size-0.5",
    "tie_break": "half_up",
    "edge_policy": "clamp_to_edge",
}


def block_mean_downsample(values: np.ndarray, factor: int) -> np.ndarray:
    """Downsample an int64 image by an integer factor using block means."""
    if factor == 1:
        return values.copy()
    height, width = values.shape[:2]
    if height % factor or width % factor:
        raise ValueError("image dimensions are not divisible by downsample factor")
    blocks = values.reshape(height // factor, factor, width // factor, factor, values.shape[2])
    total = blocks.sum(axis=(1, 3), dtype=np.int64)
    count = factor * factor
    return (total + count // 2) // count


def _axis_mapping(src_size: int, dst_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    den = 2 * dst_size
    numerator = (2 * np.arange(dst_size, dtype=np.int64) + 1) * src_size - dst_size
    max_numerator = (src_size - 1) * den
    numerator = np.clip(numerator, 0, max_numerator)
    base = numerator // den
    remainder = numerator % den
    upper = np.minimum(base + 1, src_size - 1)
    return base, upper, remainder, den


def bilinear_upsample(values: np.ndarray, out_height: int, out_width: int) -> np.ndarray:
    """Fixed-point bilinear interpolation with clamped edges and half-up ties."""
    src_height, src_width = values.shape[:2]
    if src_height == out_height and src_width == out_width:
        return values.copy()
    y0, y1, ry, deny = _axis_mapping(src_height, out_height)
    x0, x1, rx, denx = _axis_mapping(src_width, out_width)

    top_left = values[np.ix_(y0, x0)]
    top_right = values[np.ix_(y0, x1)]
    bottom_left = values[np.ix_(y1, x0)]
    bottom_right = values[np.ix_(y1, x1)]

    wy0 = (deny - ry)[:, None, None]
    wy1 = ry[:, None, None]
    wx0 = (denx - rx)[None, :, None]
    wx1 = rx[None, :, None]
    denominator = np.int64(deny * denx)
    numerator = (
        wy0 * wx0 * top_left
        + wy0 * wx1 * top_right
        + wy1 * wx0 * bottom_left
        + wy1 * wx1 * bottom_right
    )
    return (numerator + denominator // 2) // denominator


def axis_support(src_size: int, dst_size: int, start: int, stop: int) -> tuple[int, int]:
    """Inclusive source index range required by bilinear output indices [start, stop)."""
    base, upper, _, _ = _axis_mapping(src_size, dst_size)
    if start >= stop:
        raise ValueError("empty destination range")
    return int(base[start:stop].min()), int(upper[start:stop].max())


def cumulative_dimensions(base_width: int, base_height: int, factors: list[int] | tuple[int, ...]) -> list[tuple[int, int]]:
    cumulative = 1
    sizes: list[tuple[int, int]] = []
    for factor in factors:
        cumulative *= int(factor)
        if base_width % cumulative or base_height % cumulative:
            raise ValueError("downsample factor product is incompatible with base dimensions")
        sizes.append((base_width // cumulative, base_height // cumulative))
    return sizes

