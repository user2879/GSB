from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from .configuration import ALLOWED_FRACTIONAL_BITS, ALLOWED_QUANT_LEVELS
from .errors import WorkbenchError
from .fixedpoint import decimal_to_fixed, fixed_to_byte_array, scale_for_bits
from .hashing import array_fingerprint, canonical_hash

PARAMETER_SCALE_BITS = 16
PARAMETER_SCALE = 1 << PARAMETER_SCALE_BITS
STYLE_TONE = (14, 4, -12)


def ensure_finite_pixels(values: np.ndarray, target: str) -> None:
    if values.dtype.kind in "fc" and not np.all(np.isfinite(values)):
        bad = int(np.sum(~np.isfinite(values)))
        raise WorkbenchError(
            "E_NONFINITE_PIXELS",
            "non-finite pixel values were injected into a real kernel input",
            {"target": target, "nonfinite_count": bad},
        )


def validate_injected_rectangle(rect: dict[str, Any], width: int, height: int, target: str) -> None:
    try:
        x = rect["x"]
        y = rect["y"]
        w = rect["width"]
        h = rect["height"]
    except KeyError as exc:
        raise WorkbenchError("E_MASK_OUT_OF_RANGE", "injected rectangle is missing fields", {"target": target}) from exc
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (x, y, w, h)):
        raise WorkbenchError("E_MASK_OUT_OF_RANGE", "injected rectangle coordinates must be integers", {"target": target, "rect": rect})
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
        raise WorkbenchError(
            "E_MASK_OUT_OF_RANGE",
            "injected rectangle is outside tile/level bounds",
            {"target": target, "rect": rect, "width": width, "height": height},
        )


def validate_quantization(levels: int, fractional_bits: int, target: str) -> None:
    if levels not in ALLOWED_QUANT_LEVELS or fractional_bits not in ALLOWED_FRACTIONAL_BITS:
        raise WorkbenchError(
            "E_QUANT_OUT_OF_RANGE",
            "injected quantization tier or fixed-point width is outside the allowed enum",
            {"target": target, "levels": levels, "fractional_bits": fractional_bits,
             "allowed_levels": list(ALLOWED_QUANT_LEVELS), "allowed_bits": list(ALLOWED_FRACTIONAL_BITS)},
        )


def _neighbor_diffusion(current: np.ndarray) -> np.ndarray:
    """Weighted 8-neighbor mean with orthogonal weight 2 and diagonal weight 1."""
    height, width, channels = current.shape
    padded = np.zeros((height + 2, width + 2, channels), dtype=np.int64)
    padded[1:-1, 1:-1, :] = current
    valid_padded = np.zeros((height + 2, width + 2), dtype=np.int64)
    valid_padded[1:-1, 1:-1] = 1
    weight_sum = np.zeros((height, width), dtype=np.int64)
    total = np.zeros((height, width, channels), dtype=np.int64)

    terms = (
        (2, padded[:-2, 1:-1], valid_padded[:-2, 1:-1]),
        (2, padded[2:, 1:-1], valid_padded[2:, 1:-1]),
        (2, padded[1:-1, :-2], valid_padded[1:-1, :-2]),
        (2, padded[1:-1, 2:], valid_padded[1:-1, 2:]),
        (1, padded[:-2, :-2], valid_padded[:-2, :-2]),
        (1, padded[:-2, 2:], valid_padded[:-2, 2:]),
        (1, padded[2:, :-2], valid_padded[2:, :-2]),
        (1, padded[2:, 2:], valid_padded[2:, 2:]),
    )
    for weight, values, valid in terms:
        total += weight * values
        weight_sum += weight * valid

    safe_weights = np.maximum(weight_sum, 1)
    rounded = (total * 2 + safe_weights[..., None]) // (2 * safe_weights[..., None])
    return rounded.astype(np.int64)


def _blend_fixed(base: np.ndarray, guide: np.ndarray, strength: float, fractional_bits: int) -> np.ndarray:
    scale = scale_for_bits(fractional_bits)
    strength_num = decimal_to_fixed(strength, 16)
    strength_num = min(max(strength_num, 0), PARAMETER_SCALE)
    guide_weight = (strength_num * scale) // PARAMETER_SCALE
    base_weight = scale - guide_weight
    numerator = base * base_weight + guide * guide_weight
    return ((numerator * 2 + scale) // (2 * scale)).astype(np.int64)


def _outside_in_fill(initial: np.ndarray, mask: np.ndarray, iterations: int) -> np.ndarray:
    """Fill mask layers from known boundary inward, then run fixed relaxation."""
    height, width, _ = initial.shape
    distance = np.full((height, width), -1, dtype=np.int32)
    queue: deque[tuple[int, int]] = deque()
    for y, x in np.argwhere(~mask):
        distance[y, x] = 0
        queue.append((int(y), int(x)))

    neighbors = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
    while queue:
        y, x = queue.popleft()
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < height and 0 <= nx < width):
                continue
            if mask[ny, nx] and distance[ny, nx] == -1:
                distance[ny, nx] = distance[y, x] + 1
                queue.append((ny, nx))

    current = initial.copy()
    reachable = mask & (distance > 0)
    max_distance = int(distance[reachable].max()) if np.any(reachable) else 0
    for layer_distance in range(1, max_distance + 1):
        coords = np.argwhere(distance == layer_distance)
        for y_array, x_array in coords:
            y, x = int(y_array), int(x_array)
            total = np.zeros(3, dtype=np.int64)
            weight_sum = 0
            for dy, dx in neighbors:
                ny, nx = y + dy, x + dx
                if not (0 <= ny < height and 0 <= nx < width):
                    continue
                if 0 <= distance[ny, nx] < layer_distance:
                    weight = 2 if dy == 0 or dx == 0 else 1
                    total += weight * current[ny, nx]
                    weight_sum += weight
            if weight_sum > 0:
                current[y, x] = (total * 2 + weight_sum) // (2 * weight_sum)

    # Tiles containing a fully enclosed masked component still get a deterministic
    # diffusion smoothing of their own initial values; normal sample tiles have boundaries.
    for _ in range(max(1, iterations)):
        diffusion = _neighbor_diffusion(current)
        current[mask] = diffusion[mask]
    return current


def repair_tile(
    source_tile: np.ndarray,
    mask_tile: np.ndarray,
    *,
    level: int,
    iterations: int,
    guidance_strength: float,
    fractional_bits: int,
    guide_fixed: np.ndarray | None = None,
) -> np.ndarray:
    if source_tile.ndim != 3 or source_tile.shape[2] != 3:
        raise WorkbenchError("E_CONFIG_INVALID", "repair tile must be RGB", {"level": level})
    if mask_tile.shape != source_tile.shape[:2]:
        raise WorkbenchError("E_CONFIG_INVALID", "mask tile shape does not match image tile")
    ensure_finite_pixels(source_tile, f"repair:L{level}")
    scale = scale_for_bits(fractional_bits)
    initial = source_tile.astype(np.int64) * scale

    if guide_fixed is None:
        repaired_fp = _outside_in_fill(initial, mask_tile, iterations)
    else:
        ensure_finite_pixels(guide_fixed, f"guide:L{level}")
        if guide_fixed.shape != initial.shape:
            raise WorkbenchError("E_CONFIG_INVALID", "guide tile shape does not match repair tile")
        current = initial.copy()
        for step in range(max(1, iterations)):
            base = current if step == 0 else _neighbor_diffusion(current)
            blended = _blend_fixed(base, guide_fixed, guidance_strength, fractional_bits)
            current[mask_tile] = blended[mask_tile]
        repaired_fp = current

    repaired = fixed_to_byte_array(repaired_fp, fractional_bits)
    repaired[~mask_tile] = source_tile[~mask_tile]
    return repaired


def build_style_lut(
    *,
    quant_levels: int,
    fractional_bits: int,
    strength: float,
    weights: tuple[float, float, float],
    target: str = "style",
) -> tuple[np.ndarray, dict[str, Any]]:
    validate_quantization(quant_levels, fractional_bits, target)
    scale = scale_for_bits(fractional_bits)
    strength_num = decimal_to_fixed(strength, 16)
    lut_fixed = np.zeros((3, 256), dtype=np.int64)
    lut_bytes = np.zeros((3, 256), dtype=np.uint8)

    for channel in range(3):
        weight_num = decimal_to_fixed(weights[channel], 16)
        tone_fp = STYLE_TONE[channel] * scale
        for value in range(256):
            source_fp = value * scale
            weighted_fp = source_fp * weight_num // PARAMETER_SCALE + tone_fp
            target_fp = int(np.clip(weighted_fp, 0, 255 * scale))
            numerator = source_fp * (PARAMETER_SCALE - strength_num) + target_fp * strength_num
            styled_fp = (numerator * 2 + PARAMETER_SCALE) // (2 * PARAMETER_SCALE)
            byte_value = int((styled_fp * 2 + scale) // (2 * scale))
            byte_value = min(max(byte_value, 0), 255)
            if quant_levels == 256:
                quantized = byte_value
            else:
                quantum = quant_levels - 1
                code = (2 * byte_value * quantum + 255) // (2 * 255)
                quantized = (2 * code * 255 + quantum) // (2 * quantum)
            lut_bytes[channel, value] = quantized
            lut_fixed[channel, value] = quantized * scale

    metadata = {
        "quant_levels": quant_levels,
        "fractional_bits": fractional_bits,
        "strength": strength,
        "weights": list(weights),
        "lut_fingerprint": array_fingerprint(lut_fixed),
        "lut_bytes_fingerprint": array_fingerprint(lut_bytes),
    }
    metadata["lut_key"] = canonical_hash(metadata)
    return lut_fixed, metadata


def style_tile(
    repaired_tile: np.ndarray,
    outside_reference_tile: np.ndarray,
    mask_tile: np.ndarray,
    lut_bytes: np.ndarray,
) -> np.ndarray:
    styled = repaired_tile.copy()
    for channel in range(3):
        channel_values = repaired_tile[:, :, channel]
        styled[:, :, channel] = np.where(
            mask_tile,
            lut_bytes[channel, channel_values],
            outside_reference_tile[:, :, channel],
        )
    styled[~mask_tile] = outside_reference_tile[~mask_tile]
    return styled
