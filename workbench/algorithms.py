"""Deterministic stage algorithms.

Every numeric path uses integer or explicit fixed-point arithmetic so that
serial and parallel runs, and Windows and Linux, produce identical arrays.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .utils import Q, digest_bytes, fp_div, fp_mul, weight_to_q


# ---------------------------------------------------------------------------
# Stage 1/2: region parsing and mask generation
# ---------------------------------------------------------------------------

def _rect_mask(width: int, height: int, region: dict) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[region["y"] : region["y"] + region["h"], region["x"] : region["x"] + region["w"]] = 1
    return mask


def _region_seed(region: dict) -> tuple[int, int]:
    return (
        region["y"] + region["h"] // 2,
        region["x"] + region["w"] // 2,
    )


def mask_threshold_connected(
    image: np.ndarray, region: dict, threshold: int
) -> np.ndarray:
    """Flood fill pixels whose colour distance to the seed pixel is below threshold."""
    height, width = image.shape[:2]
    seed_y, seed_x = _region_seed(region)
    seed_color = image[seed_y, seed_x].astype(np.int16)
    distances = np.abs(image.astype(np.int16) - seed_color).sum(axis=2)
    eligible = distances <= int(threshold)

    mask = np.zeros((height, width), dtype=np.uint8)
    queue = deque([(seed_y, seed_x)])
    mask[seed_y, seed_x] = 1
    while queue:
        y, x = queue.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < height and 0 <= nx < width and not mask[ny, nx] and eligible[ny, nx]:
                mask[ny, nx] = 1
                queue.append((ny, nx))
    return mask


def mask_edge_trace(image: np.ndarray, region: dict, edge_threshold: int) -> np.ndarray:
    """Close the region rect along its gradient edges and fill the contour."""
    height, width = image.shape[:2]
    gray = image.astype(np.int16).sum(axis=2)
    grad_y = np.zeros_like(gray)
    grad_x = np.zeros_like(gray)
    grad_y[1:-1, :] = np.abs(gray[2:, :] - gray[:-2, :])
    grad_x[:, 1:-1] = np.abs(gray[:, 2:] - gray[:, :-2])
    edge = ((grad_y + grad_x) > int(edge_threshold)).astype(np.uint8)

    contour = np.zeros((height, width), dtype=np.uint8)
    x0, y0, w, h = region["x"], region["y"], region["w"], region["h"]
    x1, y1 = min(width - 1, x0 + w - 1), min(height - 1, y0 + h - 1)
    contour[y0, x0 : x1 + 1] = 1
    contour[y1, x0 : x1 + 1] = 1
    contour[y0 : y1 + 1, x0] = 1
    contour[y0 : y1 + 1, x1] = 1
    # Snap each border cell onto the nearest strong gradient inside the rect.
    x_span = np.arange(x0, x1 + 1)
    y_span = np.arange(y0, y1 + 1)
    sub_edge = edge[y0 : y1 + 1, x0 : x1 + 1]
    top_cols = np.where(sub_edge.any(axis=0), sub_edge.argmax(axis=0), 0)
    bottom_cols = np.where(
        sub_edge.any(axis=0), sub_edge.shape[0] - 1 - sub_edge[::-1, :].argmax(axis=0), sub_edge.shape[0] - 1
    )
    left_rows = np.where(sub_edge.any(axis=1), sub_edge.argmax(axis=1), 0)
    right_rows = np.where(
        sub_edge.any(axis=1), sub_edge.shape[1] - 1 - sub_edge[:, ::-1].argmax(axis=1), sub_edge.shape[1] - 1
    )
    for off, col in zip(top_cols, x_span):
        contour[y0 + off, col] = 1
    for off, col in zip(bottom_cols, x_span):
        contour[y0 + off, col] = 1
    for off, row in zip(left_rows, y_span):
        contour[row, x0 + off] = 1
    for off, row in zip(right_rows, y_span):
        contour[row, x0 + off] = 1

    # Flood fill outside the contour, then invert to obtain its interior.
    outside = np.zeros((height, width), dtype=np.uint8)
    starts = [(0, 0), (height - 1, 0), (0, width - 1), (height - 1, width - 1)]
    queue = deque()
    for sy, sx in starts:
        if not contour[sy, sx] and not outside[sy, sx]:
            outside[sy, sx] = 1
            queue.append((sy, sx))
    while queue:
        y, x = queue.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < height and 0 <= nx < width and not contour[ny, nx] and not outside[ny, nx]:
                outside[ny, nx] = 1
                queue.append((ny, nx))
    return (1 - outside).astype(np.uint8)


def generate_mask(path: str, image: np.ndarray, region: dict, params: dict) -> np.ndarray:
    if path == "threshold":
        return mask_threshold_connected(image, region, params["mask_threshold"])
    if path == "edge":
        return mask_edge_trace(image, region, params["edge_threshold"])
    raise ValueError(f"unknown mask path: {path}")


# ---------------------------------------------------------------------------
# Stage 3: inpainting (fixed point integer kernels)
# ---------------------------------------------------------------------------

def _global_unmasked_mean(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    keep = mask == 0
    if not keep.any():
        return np.zeros(3, dtype=np.int64)
    return image[keep].astype(np.int64).sum(axis=0) // int(keep.sum())


def inpaint_neighbor_mean(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Iterative 4-neighbour weighted mean, unit weights, integer division."""
    result = image.astype(np.int64).copy()
    fallback = _global_unmasked_mean(image, mask)
    if not mask.any():
        return result.astype(np.uint8)
    height, width = mask.shape
    # Fixed iteration count keeps runtime and output independent of data shape.
    for _ in range(12):
        updated = result.copy()
        ys, xs = np.nonzero(mask)
        for y, x in zip(ys.tolist(), xs.tolist()):
            acc = np.zeros(3, dtype=np.int64)
            count = 0
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and not mask[ny, nx]:
                    acc += result[ny, nx]
                    count += 1
            updated[y, x] = fp_div(acc, count) if count else fallback
        result = updated
    return np.clip(result, 0, 255).astype(np.uint8)


def _axis_interpolate(image: np.ndarray, mask: np.ndarray, axis: int, fallback: np.ndarray) -> np.ndarray:
    result = image.astype(np.int64).copy()
    height, width = mask.shape
    if axis == 1:  # horizontal, iterate rows
        for y in range(height):
            known = np.flatnonzero(mask[y] == 0)
            if known.size == 0:
                result[y][mask[y] == 1] = fallback
                continue
            for x in np.flatnonzero(mask[y] == 1).tolist():
                left = known[known < x]
                right = known[known > x]
                if left.size and right.size:
                    lx, rx = int(left[-1]), int(right[0])
                    dl, dr = x - lx, rx - x
                    num = result[y, lx] * dr + result[y, rx] * dl
                    result[y, x] = fp_div(num, dl + dr)
                else:
                    anchor = int(known[0]) if right.size else int(known[-1])
                    result[y, x] = result[y, anchor]
    else:  # vertical
        for x in range(width):
            known = np.flatnonzero(mask[:, x] == 0)
            if known.size == 0:
                result[mask[:, x] == 1, x] = fallback
                continue
            for y in np.flatnonzero(mask[:, x] == 1).tolist():
                up = known[known < y]
                down = known[known > y]
                if up.size and down.size:
                    uy, dy = int(up[-1]), int(down[0])
                    du, dd = y - uy, dy - y
                    num = result[uy, x] * dd + result[dy, x] * du
                    result[y, x] = fp_div(num, du + dd)
                else:
                    anchor = int(known[0]) if down.size else int(known[-1])
                    result[y, x] = result[anchor, x]
    return result


def inpaint_directional(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend horizontal and vertical 1-D interpolation with fixed 1:1 weights."""
    fallback = _global_unmasked_mean(image, mask)
    horizontal = _axis_interpolate(image, mask, axis=1, fallback=fallback)
    vertical = _axis_interpolate(image, mask, axis=0, fallback=fallback)
    blended = fp_div(horizontal + vertical, 2)
    return np.clip(blended, 0, 255).astype(np.uint8)


def inpaint(path: str, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if path == "neighbor_mean":
        return inpaint_neighbor_mean(image, mask)
    if path == "directional":
        return inpaint_directional(image, mask)
    raise ValueError(f"unknown inpaint path: {path}")


# ---------------------------------------------------------------------------
# Stage 4: deterministic recolouring
# ---------------------------------------------------------------------------

def recolor_lut(attribute_value: int, config_fingerprint: str) -> list:
    """256-entry per-channel LUT derived from attribute value + config fingerprint."""
    seed_digest = digest_bytes(
        f"recolor:{attribute_value}:{config_fingerprint}".encode("utf-8")
    )
    params = np.frombuffer(bytes.fromhex(seed_digest), dtype=np.uint8).astype(np.int64)
    shift, slope_q, bias = int(params[0]), int(params[1]) + 1, int(params[2])
    lut = []
    for value in range(256):
        channel_values = []
        for channel in range(3):
            offset = int(params[3 + channel])
            mapped = fp_mul((value + shift + offset) & 0xFF, slope_q) + bias
            channel_values.append(int(np.clip(mapped, 0, 255)))
        lut.append(channel_values)
    return lut


def apply_recolor(
    image: np.ndarray,
    mask: np.ndarray,
    attribute_value: int,
    style_strength: float,
    config_fingerprint: str,
) -> tuple[np.ndarray, dict]:
    lut = recolor_lut(attribute_value, config_fingerprint)
    strength_q = weight_to_q(style_strength)
    result = image.astype(np.int64).copy()
    ys, xs = np.nonzero(mask)
    changed = 0
    for y, x in zip(ys.tolist(), xs.tolist()):
        source = image[y, x].astype(np.int64)
        target = np.asarray(lut[int(source.mean())], dtype=np.int64)
        blended = fp_mul(source, Q - strength_q) + fp_mul(target, strength_q)
        blended = np.clip(blended, 0, 255)
        if not np.array_equal(blended, source):
            changed += 1
        result[y, x] = blended
    summary = {
        "lut_entries": len(lut),
        "lut_summary": digest_bytes(canonical_lut_bytes(lut))[:12],
        "strength_q": strength_q,
        "changed_pixels": int(changed),
        "mask_pixels": int(mask.sum()),
    }
    return result.astype(np.uint8), summary


def canonical_lut_bytes(lut: list) -> bytes:
    return np.asarray(lut, dtype=np.uint8).tobytes()


# ---------------------------------------------------------------------------
# Stage 5: style mapping
# ---------------------------------------------------------------------------

def style_lut(quant_levels: int, style_seed: int) -> list:
    """Quantized colour lookup table; one representative bucket per level."""
    if not 2 <= quant_levels <= 256:
        raise ValueError("quant_levels must be in [2, 256]")
    step = 256 / quant_levels
    digest = digest_bytes(f"style-lut:{quant_levels}:{style_seed}".encode("utf-8"))
    jitter = np.frombuffer(bytes.fromhex(digest), dtype=np.uint8).astype(np.int64)
    lut = []
    for value in range(256):
        bucket = min(quant_levels - 1, int(value / step))
        representative = int((bucket + 0.5) * step)
        channels = []
        for channel in range(3):
            channels.append(int(np.clip(representable(representative, jitter, channel), 0, 255)))
        lut.append(channels)
    return lut


def representable(representative: int, jitter: np.ndarray, channel: int) -> int:
    return representative + int(jitter[channel]) - 128


def style_linear_weights(style_seed: int) -> list:
    """Fixed per-channel weights in Q fixed point, derived deterministically."""
    digest = digest_bytes(f"style-linear:{style_seed}".encode("utf-8"))
    raw = np.frombuffer(bytes.fromhex(digest), dtype=np.uint8).astype(np.int64)[:3]
    weights = []
    for channel in range(3):
        # Keep weights in [0.75, 1.25] so output stays a mild colour transform.
        weights.append(Q + int(raw[channel]) // 2 - 64)
    return weights


def apply_style_lut(image: np.ndarray, quant_levels: int, style_seed: int) -> tuple[np.ndarray, dict]:
    table = np.asarray(style_lut(quant_levels, style_seed), dtype=np.uint8)
    # The table maps each input level to an RGB triple; apply each channel's
    # own column so an (H, W, 3) image stays an (H, W, 3) image.
    result = np.empty_like(image)
    for channel in range(3):
        channel_lut = table[:, channel]
        result[:, :, channel] = channel_lut[image[:, :, channel]]
    return result.astype(np.uint8), {
        "path": "lut",
        "quant_levels": quant_levels,
        "lut_summary": digest_bytes(table.tobytes())[:12],
    }


def apply_style_linear(image: np.ndarray, quant_levels: int, style_seed: int) -> tuple[np.ndarray, dict]:
    weights = style_linear_weights(style_seed)
    work = image.astype(np.int64)
    result = np.empty_like(work)
    for channel in range(3):
        result[:, :, channel] = np.clip(fp_mul(work[:, :, channel], weights[channel]), 0, 255)
    return result.astype(np.uint8), {
        "path": "linear",
        "quant_levels": quant_levels,
        "weights_q": weights,
    }


def apply_style(path: str, image: np.ndarray, quant_levels: int, style_seed: int):
    if path == "lut":
        return apply_style_lut(image, quant_levels, style_seed)
    if path == "linear":
        return apply_style_linear(image, quant_levels, style_seed)
    raise ValueError(f"unknown style path: {path}")


# ---------------------------------------------------------------------------
# Stage 6: composition
# ---------------------------------------------------------------------------

def composite(source: np.ndarray, edited: np.ndarray, union_mask: np.ndarray) -> np.ndarray:
    """Mask retention: edited pixels inside mask, source pixels outside."""
    keep_edit = union_mask.astype(bool)
    result = source.copy()
    result[keep_edit] = edited[keep_edit]
    return result


# ---------------------------------------------------------------------------
# Deterministic per-sample cost model used by the strategy selector
# ---------------------------------------------------------------------------

MASK_COST = {"threshold": 1.0, "edge": 1.7}
INPAINT_COST = {"neighbor_mean": 1.0, "directional": 1.9}
STYLE_COST = {"lut": 0.8, "linear": 1.3}


def estimate_cost_units(
    *,
    pixels: int,
    regions: int,
    instruction_length: int,
    mask_path: str,
    inpaint_path: str,
    style_path: str,
    quant_levels: int,
) -> float:
    mask = pixels * regions * MASK_COST[mask_path]
    inpaint = pixels * regions * INPAINT_COST[inpaint_path]
    recolor = pixels * instruction_length
    style = pixels * STYLE_COST[style_path] * (256.0 / quant_levels)
    composite = pixels
    return round(mask + inpaint + recolor + style + composite, 3)
