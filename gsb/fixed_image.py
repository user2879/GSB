"""Shared fixed-point block measurements used by synthesis and inference."""

from __future__ import annotations

from typing import Any

import numpy as np

from gsb.models import BBox
from gsb.util import fixed_to_rounded_int, quantize_fixed, ratio_fixed, residual_fixed, round_half_up, rounding_direction


def ink_mask(pixels: np.ndarray, threshold: int) -> np.ndarray:
    return pixels <= threshold


def ink_bbox(pixels: np.ndarray, threshold: int) -> BBox | None:
    mask = ink_mask(pixels, threshold)
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return BBox(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def weighted_centroid_fixed(
    pixels: np.ndarray,
    bbox: BBox,
    threshold: int,
    bits: int,
) -> tuple[int, int, int]:
    region = pixels[bbox.top : bbox.bottom, bbox.left : bbox.right]
    local_y, local_x = np.indices(region.shape)
    strength = np.maximum(1, threshold - region.astype(np.int64))
    strength = np.where(region <= threshold, strength, 0)
    total_weight = int(strength.sum())
    if total_weight <= 0:
        center_x = bbox.left + bbox.right
        center_y = bbox.top + bbox.bottom
        return ratio_fixed(center_x, 2, bits), ratio_fixed(center_y, 2, bits), 0
    sum_x = int((strength * (local_x + bbox.left)).sum())
    sum_y = int((strength * (local_y + bbox.top)).sum())
    return ratio_fixed(sum_x, total_weight, bits), ratio_fixed(sum_y, total_weight, bits), total_weight


def regress_center(
    pixels: np.ndarray,
    *,
    threshold: int,
    bits: int,
    bbox: BBox | None = None,
) -> dict[str, Any] | None:
    region_bbox = bbox if bbox is not None else ink_bbox(pixels, threshold)
    if region_bbox is None:
        return None
    centroid_x, centroid_y, total_weight = weighted_centroid_fixed(
        pixels,
        region_bbox,
        threshold,
        bits,
    )
    bbox_center_x = (region_bbox.left + region_bbox.right) << max(0, bits - 1)
    bbox_center_y = (region_bbox.top + region_bbox.bottom) << max(0, bits - 1)
    if bits == 0:
        bbox_center_x = round_half_up(region_bbox.left + region_bbox.right, 2)
        bbox_center_y = round_half_up(region_bbox.top + region_bbox.bottom, 2)
    raw_x = round_half_up(3 * centroid_x + bbox_center_x, 4)
    raw_y = round_half_up(3 * centroid_y + bbox_center_y, 4)
    quantized_x = quantize_fixed(raw_x, bits)
    quantized_y = quantize_fixed(raw_y, bits)
    return {
        "bbox": region_bbox.to_list(),
        "centroid_fixed": [centroid_x, centroid_y],
        "bbox_center_fixed": [bbox_center_x, bbox_center_y],
        "raw_fixed": [raw_x, raw_y],
        "quantized_fixed": [quantized_x, quantized_y],
        "x": fixed_to_rounded_int(quantized_x, bits),
        "y": fixed_to_rounded_int(quantized_y, bits),
        "rounding": {
            "x": rounding_direction(raw_x, quantized_x),
            "y": rounding_direction(raw_y, quantized_y),
        },
        "residual_fixed": residual_fixed(raw_x, quantized_x)
        + residual_fixed(raw_y, quantized_y),
        "total_weight": total_weight,
    }
