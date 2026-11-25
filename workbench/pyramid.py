from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .configuration import WorkbenchConfig
from .fixedpoint import round_half_up, scale_for_bits
from .hashing import array_fingerprint, canonical_hash
from .masks import downsample_mask_any, rasterize_rectangles

INTERPOLATION_METADATA = {
    "downsample": "integer_block_mean_round_half_up",
    "upsample": "center_mapped_bilinear_fixed_point",
    "tie_break": "equidistant_samples_get_equal_weight; output_rounds_half_up; boundary_clamps_to_top_left",
}


@dataclass(frozen=True)
class Pyramid:
    image_levels: tuple[np.ndarray, ...]
    mask_levels: tuple[np.ndarray, ...]
    layer_fingerprints: tuple[str, ...]
    image_fingerprint: str
    metadata: dict[str, Any]


def block_mean_downsample(pixels: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 0:
        raise ValueError("factor must be positive")
    height, width, channels = pixels.shape
    if height % factor or width % factor:
        raise ValueError("factor must divide image dimensions")
    blocks = pixels.reshape(height // factor, factor, width // factor, factor, channels)
    totals = blocks.sum(axis=(1, 3), dtype=np.int64)
    count = factor * factor
    return round_half_up(totals, count).astype(np.uint8)


def build_pyramid(config: WorkbenchConfig, base_pixels: np.ndarray, image_fingerprint: str,
                  rectangles: dict[str, Any]) -> Pyramid:
    fine_to_coarse_images = [base_pixels]
    current = base_pixels
    for factor in config.factors_fine_to_coarse:
        current = block_mean_downsample(current, factor)
        fine_to_coarse_images.append(current)
    image_levels = tuple(reversed(fine_to_coarse_images))

    base_mask = rasterize_rectangles(rectangles, config.width, config.height)
    fine_to_coarse_masks = [base_mask]
    current_mask = base_mask
    for factor in config.factors_fine_to_coarse:
        current_mask = downsample_mask_any(current_mask, factor)
        fine_to_coarse_masks.append(current_mask)
    mask_levels = tuple(reversed(fine_to_coarse_masks))

    fingerprints: list[str] = []
    for level, image_level in enumerate(image_levels):
        fingerprints.append(canonical_hash({
            "image_content_fingerprint": image_fingerprint,
            "level": level,
            "cumulative_downsample_factor": config.cumulative_downsample[level],
            "adjacent_factor_to_finer": (
                config.upsample_factors_coarse_to_fine[level]
                if level < len(config.upsample_factors_coarse_to_fine) else 1
            ),
            "pixels_fingerprint": array_fingerprint(image_level),
        }))

    metadata = {
        **INTERPOLATION_METADATA,
        "level_dimensions": [list(dim) for dim in config.level_dimensions],
        "tile_sizes": list(config.tile_sizes),
        "layer_fingerprints": fingerprints,
    }
    return Pyramid(
        image_levels=image_levels,
        mask_levels=mask_levels,
        layer_fingerprints=tuple(fingerprints),
        image_fingerprint=image_fingerprint,
        metadata=metadata,
    )


def tile_identity(level: int, row: int, col: int) -> str:
    return f"L{level:02d}R{row:03d}C{col:03d}"


def iter_tile_coordinates(config: WorkbenchConfig, level: int) -> list[tuple[int, int]]:
    rows, cols = config.tile_grid(level)
    return [(row, col) for row in range(rows) for col in range(cols)]


def tile_bounds(config: WorkbenchConfig, level: int, row: int, col: int) -> tuple[int, int, int, int]:
    size = config.tile_sizes[level]
    y0, x0 = row * size, col * size
    return y0, y0 + size, x0, x0 + size


def crop_tile(array: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = bounds
    return array[y0:y1, x0:x1]


def bilinear_index_vectors(length_out: int, length_in: int, factor: int, fractional_bits: int):
    if length_out != length_in * factor:
        raise ValueError("bilinear helper expects exact factor dimensions")
    scale = scale_for_bits(fractional_bits)
    index0 = np.zeros(length_out, dtype=np.int64)
    index1 = np.zeros(length_out, dtype=np.int64)
    weight1 = np.zeros(length_out, dtype=np.int64)
    denominator = 2 * factor

    for index in range(length_out):
        numerator = 2 * index + 1 - factor
        if numerator <= 0:
            continue
        source0 = numerator // denominator
        remainder = numerator % denominator
        if source0 >= length_in - 1:
            index0[index] = length_in - 1
            index1[index] = length_in - 1
            continue
        index0[index] = source0
        index1[index] = source0 + 1
        weight1[index] = int(round_half_up(remainder * scale, denominator))
    return index0, index1, scale - weight1, weight1


def upsample_uint8_guide(coarse_pixels: np.ndarray, factor: int, out_height: int,
                         out_width: int, fractional_bits: int) -> np.ndarray:
    """Bilinear upsample and return fixed-point int64 values."""
    height_in, width_in, channels = coarse_pixels.shape
    y0, y1, wy0, wy1 = bilinear_index_vectors(out_height, height_in, factor, fractional_bits)
    x0, x1, wx0, wx1 = bilinear_index_vectors(out_width, width_in, factor, fractional_bits)
    coarse_fp = coarse_pixels.astype(np.int64) * scale_for_bits(fractional_bits)

    v00 = coarse_fp[np.ix_(y0, x0, np.arange(channels))]
    v01 = coarse_fp[np.ix_(y0, x1, np.arange(channels))]
    v10 = coarse_fp[np.ix_(y1, x0, np.arange(channels))]
    v11 = coarse_fp[np.ix_(y1, x1, np.arange(channels))]

    wy0_b = wy0[:, None, None]
    wy1_b = wy1[:, None, None]
    wx0_b = wx0[None, :, None]
    wx1_b = wx1[None, :, None]
    denominator = scale_for_bits(fractional_bits) ** 2
    total = (
        v00 * wy0_b * wx0_b
        + v01 * wy0_b * wx1_b
        + v10 * wy1_b * wx0_b
        + v11 * wy1_b * wx1_b
    )
    return ((total * 2 + denominator) // (2 * denominator)).astype(np.int64)


def referenced_coarse_tiles(
    config: WorkbenchConfig,
    fine_level: int,
    row: int,
    col: int,
) -> list[str]:
    """Return exact coarse tile IDs touched by this fine tile's guide crop."""
    if fine_level == 0:
        return []
    coarse_level = fine_level - 1
    factor = config.upsample_factors_coarse_to_fine[coarse_level]
    fine_w, fine_h = config.level_dimensions[fine_level]
    coarse_w, coarse_h = config.level_dimensions[coarse_level]
    bits = config.fractional_bits[fine_level]
    y0_idx, y1_idx, _, _ = bilinear_index_vectors(fine_h, coarse_h, factor, bits)
    x0_idx, x1_idx, _, _ = bilinear_index_vectors(fine_w, coarse_w, factor, bits)

    size_fine = config.tile_sizes[fine_level]
    y_start, y_end = row * size_fine, (row + 1) * size_fine
    x_start, x_end = col * size_fine, (col + 1) * size_fine
    coarse_rows = set()
    coarse_cols = set()
    for y in range(y_start, y_end):
        coarse_rows.add(int(y0_idx[y] // config.tile_sizes[coarse_level]))
        coarse_rows.add(int(y1_idx[y] // config.tile_sizes[coarse_level]))
    for x in range(x_start, x_end):
        coarse_cols.add(int(x0_idx[x] // config.tile_sizes[coarse_level]))
        coarse_cols.add(int(x1_idx[x] // config.tile_sizes[coarse_level]))

    rows, cols = config.tile_grid(coarse_level)
    refs = []
    for coarse_row in sorted(coarse_rows):
        for coarse_col in sorted(coarse_cols):
            if 0 <= coarse_row < rows and 0 <= coarse_col < cols:
                refs.append(tile_identity(coarse_level, coarse_row, coarse_col))
    return refs
