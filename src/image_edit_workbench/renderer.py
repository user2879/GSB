from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import numpy as np

from .config import WorkbenchConfig
from .errors import E_STRUCTURE_INVALID, WorkbenchError
from .models import RegionEdit
from .util import clamp_int, dec_str, round_half_up


ALPHA_MAX = 255
TRIG_BITS = 20
TRIG_ONE = 1 << TRIG_BITS
COORD_BITS = 8
COORD_ONE = 1 << COORD_BITS


@dataclass(frozen=True)
class MaskResult:
    core: np.ndarray
    alpha: np.ndarray
    crop: tuple[int, int, int, int]
    affected_pixels: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RepairResult:
    filled: np.ndarray
    crop: tuple[int, int, int, int]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class AttributeResult:
    attributed: np.ndarray
    crop: tuple[int, int, int, int]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class StyleResult:
    styled: np.ndarray
    crop: tuple[int, int, int, int]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ComposeResult:
    tile: np.ndarray
    alpha: np.ndarray
    crop: tuple[int, int, int, int]
    affected_pixels: int
    metadata: dict[str, Any]


def _fixed_coord(value: Decimal) -> int:
    return round_half_up(value * Decimal(COORD_ONE))


def _floor_div(value: int, divisor: int) -> int:
    return value // divisor


def _ceil_div(value: int, divisor: int) -> int:
    return -((-value) // divisor)


def _rotation_constants(rotation_deg: Decimal) -> tuple[int, int, int]:
    milli = round_half_up(rotation_deg * Decimal(1000))
    radians = math.radians(milli / 1000.0)
    return milli, round(math.cos(radians) * TRIG_ONE), round(math.sin(radians) * TRIG_ONE)


def _rotated_bounds(
    x: Decimal,
    y: Decimal,
    w: Decimal,
    h: Decimal,
    feather: Decimal,
    cos_q: int,
    sin_q: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    cx = _fixed_coord(x + w / 2)
    cy = _fixed_coord(y + h / 2)
    hw = _fixed_coord(w / 2)
    hh = _fixed_coord(h / 2)
    corners: list[tuple[int, int]] = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            lx = sx * hw
            ly = sy * hh
            gx = cx + (cos_q * lx - sin_q * ly) // TRIG_ONE
            gy = cy + (sin_q * lx + cos_q * ly) // TRIG_ONE
            corners.append((gx, gy))
    feather_fixed = _fixed_coord(feather) + 2 * COORD_ONE
    min_x = min(point[0] for point in corners) - feather_fixed
    max_x = max(point[0] for point in corners) + feather_fixed
    min_y = min(point[1] for point in corners) - feather_fixed
    max_y = max(point[1] for point in corners) + feather_fixed
    x0 = max(0, _floor_div(min_x, COORD_ONE))
    y0 = max(0, _floor_div(min_y, COORD_ONE))
    x1 = min(image_width, _ceil_div(max_x, COORD_ONE))
    y1 = min(image_height, _ceil_div(max_y, COORD_ONE))
    return x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)


def generate_mask(region: RegionEdit, image_width: int, image_height: int, config: WorkbenchConfig) -> MaskResult:
    if region.feather_radius > config.feather_radius_max:
        raise WorkbenchError(
            E_STRUCTURE_INVALID,
            f"feather_radius {dec_str(region.feather_radius)} exceeds max {dec_str(config.feather_radius_max)}",
            stage="mask",
        )
    rotation_milli, cos_q, sin_q = _rotation_constants(region.rotation_deg)
    feather = region.feather_radius
    if rotation_milli == 0 and feather == 0:
        x0 = max(0, round_half_up(region.rect.x))
        y0 = max(0, round_half_up(region.rect.y))
        x1 = min(image_width, round_half_up(region.rect.x + region.rect.w))
        y1 = min(image_height, round_half_up(region.rect.y + region.rect.h))
        if x1 <= x0 or y1 <= y0:
            raise WorkbenchError(E_STRUCTURE_INVALID, "region mask became empty after pixel rounding", stage="mask")
        height = y1 - y0
        width = x1 - x0
        core = np.ones((height, width), dtype=np.uint8)
        alpha = np.full((height, width), ALPHA_MAX, dtype=np.uint8)
        return MaskResult(
            core=core,
            alpha=alpha,
            crop=(x0, y0, x1, y1),
            affected_pixels=int(width * height),
            metadata={
                "mode": "axis_aligned",
                "crop": [x0, y0, x1, y1],
                "rotation_millideg": 0,
                "feather_radius": dec_str(feather),
                "affected_pixels": int(width * height),
            },
        )

    x0, y0, x1, y1 = _rotated_bounds(
        region.rect.x,
        region.rect.y,
        region.rect.w,
        region.rect.h,
        feather,
        cos_q,
        sin_q,
        image_width,
        image_height,
    )
    width, height = x1 - x0, y1 - y0
    ys = np.arange(y0, y1, dtype=np.int64)[:, None]
    xs = np.arange(x0, x1, dtype=np.int64)[None, :]
    rel_x = xs * COORD_ONE + COORD_ONE // 2 - _fixed_coord(region.rect.x + region.rect.w / 2)
    rel_y = ys * COORD_ONE + COORD_ONE // 2 - _fixed_coord(region.rect.y + region.rect.h / 2)
    local_x = (cos_q * rel_x + sin_q * rel_y) // TRIG_ONE
    local_y = (-sin_q * rel_x + cos_q * rel_y) // TRIG_ONE
    half_w = _fixed_coord(region.rect.w / 2)
    half_h = _fixed_coord(region.rect.h / 2)
    dx_inside = half_w - np.abs(local_x)
    dy_inside = half_h - np.abs(local_y)
    inside = (dx_inside >= 0) & (dy_inside >= 0)
    signed_distance = np.where(inside, np.minimum(dx_inside, dy_inside), np.maximum(dx_inside, dy_inside))
    feather_fixed = _fixed_coord(feather)
    if feather_fixed > 0:
        numerator = (signed_distance + feather_fixed) * ALPHA_MAX
        alpha_values = (numerator + feather_fixed) // (2 * feather_fixed)
        alpha_values = np.clip(alpha_values, 0, ALPHA_MAX)
    else:
        alpha_values = np.where(inside, ALPHA_MAX, 0)
    core = inside.astype(np.uint8)
    alpha = alpha_values.astype(np.uint8)
    affected = int(np.count_nonzero(alpha))
    return MaskResult(
        core=core,
        alpha=alpha,
        crop=(x0, y0, x1, y1),
        affected_pixels=affected,
        metadata={
            "mode": "rotated_feathered",
            "crop": [x0, y0, x1, y1],
            "rotation_millideg": rotation_milli,
            "cos_q": cos_q,
            "sin_q": sin_q,
            "trig_bits": TRIG_BITS,
            "coord_bits": COORD_BITS,
            "feather_radius": dec_str(feather),
            "affected_pixels": affected,
        },
    )


def _expanded_crop(
    crop: tuple[int, int, int, int],
    pad: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = crop
    return max(0, x0 - pad), max(0, y0 - pad), min(image_width, x1 + pad), min(image_height, y1 + pad)


def _offset_weights(method: str, radius: int) -> list[tuple[int, int, int]]:
    offsets: list[tuple[int, int, int]] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            manhattan = abs(dx) + abs(dy)
            if manhattan == 0 or manhattan > radius:
                continue
            if method == "edge_extend" and dx != 0 and dy != 0:
                continue
            if method == "edge_extend":
                weight = max(1, (radius + 1 - manhattan) * 8)
            elif method == "noise_guard":
                weight = max(1, 4 - manhattan)
            else:
                weight = max(1, (radius + 2 - manhattan) * 4)
            offsets.append((dy, dx, weight))
    return sorted(offsets)


def repair_fill(
    base_rgb: np.ndarray,
    mask_core: np.ndarray,
    mask_crop: tuple[int, int, int, int],
    *,
    repair_method: str,
    radius: int,
) -> RepairResult:
    image_height, image_width = base_rgb.shape[:2]
    crop = _expanded_crop(mask_crop, radius + 1, image_width, image_height)
    x0, y0, x1, y1 = crop
    mx0, my0, _, _ = mask_crop
    original = base_rgb[y0:y1, x0:x1, :].astype(np.int64, copy=True)
    core = np.zeros(original.shape[:2], dtype=bool)
    core[my0 - y0 : my0 - y0 + mask_core.shape[0], mx0 - x0 : mx0 - x0 + mask_core.shape[1]] = mask_core.astype(bool)
    if not np.any(core):
        return RepairResult(
            filled=original.astype(np.uint8),
            crop=crop,
            metadata={"iterations": 0, "filled_pixels": 0, "method": repair_method, "radius": radius},
        )

    out = original.copy()
    known = ~core
    offsets = _offset_weights(repair_method, radius)
    height, width = core.shape
    iterations = 0
    filled_pixels = 0
    for _ in range(height + width + 2):
        candidates = core & ~known
        if not np.any(candidates):
            break
        sums = np.zeros((height, width, 3), dtype=np.int64)
        weight_sums = np.zeros((height, width), dtype=np.int64)
        min_values = np.full((height, width, 3), 255, dtype=np.int64)
        max_values = np.zeros((height, width, 3), dtype=np.int64)
        has_neighbor = np.zeros((height, width), dtype=bool)
        padded_out = np.pad(out, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
        padded_known = np.pad(known, ((radius, radius), (radius, radius)), mode="constant", constant_values=False)
        for dy, dx, weight in offsets:
            values = padded_out[radius + dy : radius + dy + height, radius + dx : radius + dx + width]
            source_known = padded_known[radius + dy : radius + dy + height, radius + dx : radius + dx + width]
            selected = candidates & source_known
            if not np.any(selected):
                continue
            weight_add = selected.astype(np.int64) * weight
            sums += values * weight_add[:, :, None]
            weight_sums += weight_add
            has_neighbor |= selected
            if repair_method == "noise_guard":
                selected_3d = selected[:, :, None]
                min_values = np.where(selected_3d, np.minimum(min_values, values), min_values)
                max_values = np.where(selected_3d, np.maximum(max_values, values), max_values)
        if not np.any(has_neighbor):
            break
        rounded = (sums + weight_sums[:, :, None] // 2) // np.maximum(weight_sums[:, :, None], 1)
        if repair_method == "noise_guard":
            rounded = np.clip(rounded, min_values, max_values)
        rounded = np.clip(rounded, 0, 255)
        fill_mask = candidates & has_neighbor
        out[fill_mask] = rounded[fill_mask]
        known[fill_mask] = True
        filled_pixels += int(np.count_nonzero(fill_mask))
        iterations += 1

    remaining = core & ~known
    if np.any(remaining):
        known_values = out[~core]
        fallback = (
            np.rint(known_values.mean(axis=0)).astype(np.int64)
            if known_values.size
            else np.array([128, 128, 128], dtype=np.int64)
        )
        out[remaining] = fallback
        filled_pixels += int(np.count_nonzero(remaining))
        iterations += 1
    return RepairResult(
        filled=np.clip(out, 0, 255).astype(np.uint8),
        crop=crop,
        metadata={
            "iterations": iterations,
            "filled_pixels": filled_pixels,
            "method": repair_method,
            "radius": radius,
            "outside_in": True,
            "rounding": "half_up_integer",
        },
    )


def attribute_replace(rgb: np.ndarray, active: np.ndarray, attribute: str) -> AttributeResult:
    source = rgb.astype(np.int64)
    result = source.copy()
    r = source[:, :, 0]
    g = source[:, :, 1]
    b = source[:, :, 2]
    if attribute == "warm":
        transformed = np.stack([(r * 2 + 230) // 3, (g * 2 + 100) // 3, (b * 2 + 55) // 3], axis=-1)
    elif attribute == "cool":
        transformed = np.stack([(r * 2 + 35) // 3, (g * 2 + 125) // 3, (b * 2 + 225) // 3], axis=-1)
    elif attribute == "vivid":
        gray = (77 * r + 150 * g + 29 * b + 128) >> 8
        transformed = np.stack(
            [
                gray + ((r - gray) * 5) // 3,
                gray + ((g - gray) * 5) // 3,
                gray + ((b - gray) * 5) // 3,
            ],
            axis=-1,
        )
    elif attribute == "mono":
        gray = (77 * r + 150 * g + 29 * b + 128) >> 8
        transformed = np.stack([gray, gray, gray], axis=-1)
    elif attribute == "solar":
        transformed = np.stack([255 - r, (255 - g + 40) // 2, 255 - b], axis=-1)
    else:  # pragma: no cover - validated upstream
        raise WorkbenchError(E_STRUCTURE_INVALID, f"unknown target attribute {attribute}", stage="attribute")
    transformed = np.clip(transformed, 0, 255)
    result[active] = transformed[active]
    return AttributeResult(
        attributed=np.clip(result, 0, 255).astype(np.uint8),
        crop=(0, 0, rgb.shape[1], rgb.shape[0]),
        metadata={"target_attribute": attribute, "active_pixels": int(np.count_nonzero(active))},
    )


def style_map(
    base_rgb: np.ndarray,
    attributed_rgb: np.ndarray,
    active: np.ndarray,
    *,
    style_strength: Decimal,
    levels: int,
    fraction_bits: int,
) -> StyleResult:
    if levels < 2:
        raise WorkbenchError(E_STRUCTURE_INVALID, "style quantization needs at least two levels", stage="style")
    scale = 1 << fraction_bits
    level = clamp_int(round_half_up(style_strength * Decimal(levels - 1)), 0, levels - 1)
    strength_q = clamp_int(round_half_up(style_strength * Decimal(scale)), 0, scale)
    mix_q = (level * scale + (levels - 1) // 2) // (levels - 1)

    base = base_rgb.astype(np.int64)
    attributed = attributed_rgb.astype(np.int64)
    gain_r = scale + (((level * 13) % 17) - 8) * scale // 512
    gain_g = scale + (((level * 7) % 15) - 7) * scale // 600
    gain_b = scale + (((level * 11) % 19) - 9) * scale // 480
    offset_r = ((level * 5) % 9) - 4
    offset_g = ((level * 3) % 7) - 3
    offset_b = ((level * 17) % 11) - 5
    styled_raw = np.stack(
        [
            (attributed[:, :, 0] * gain_r + scale // 2) // scale + offset_r,
            (attributed[:, :, 1] * gain_g + scale // 2) // scale + offset_g,
            (attributed[:, :, 2] * gain_b + scale // 2) // scale + offset_b,
        ],
        axis=-1,
    )
    styled_raw = np.clip(styled_raw, 0, 255)
    blended = (base * (scale - mix_q) + styled_raw * mix_q + scale // 2) // scale
    blended = np.clip(blended, 0, 255)
    result = base.copy()
    result[active] = blended[active]
    return StyleResult(
        styled=result.astype(np.uint8),
        crop=(0, 0, base_rgb.shape[1], base_rgb.shape[0]),
        metadata={
            "quantization_levels": levels,
            "quantized_level": level,
            "fixed_fraction_bits": fraction_bits,
            "style_strength": dec_str(style_strength),
            "style_strength_fixed": strength_q,
            "mix_fixed": mix_q,
            "rounding": "half_up",
            "gains_fixed": [gain_r, gain_g, gain_b],
            "offsets": [offset_r, offset_g, offset_b],
            "active_pixels": int(np.count_nonzero(active)),
        },
    )


def compose_tile(filled_rgb: np.ndarray, styled_rgb: np.ndarray, alpha: np.ndarray) -> ComposeResult:
    alpha3 = alpha[:, :, None].astype(np.int64)
    filled = filled_rgb.astype(np.int64)
    styled = styled_rgb.astype(np.int64)
    tile = (filled * (ALPHA_MAX - alpha3) + styled * alpha3 + ALPHA_MAX // 2) // ALPHA_MAX
    tile = np.clip(tile, 0, 255).astype(np.uint8)
    return ComposeResult(
        tile=tile,
        alpha=alpha.copy(),
        crop=(0, 0, filled_rgb.shape[1], filled_rgb.shape[0]),
        affected_pixels=int(np.count_nonzero(alpha)),
        metadata={
            "alpha_max": ALPHA_MAX,
            "alpha_nonzero": int(np.count_nonzero(alpha)),
            "rounding": "half_up_integer",
            "composition": "source_over_region_tile",
        },
    )
