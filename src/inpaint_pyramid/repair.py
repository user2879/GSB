"""Deterministic fixed-point diffusion/inpainting kernel."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fixed import blend_fixed, round_div


NEIGHBOR_WEIGHTS = (
    (-1, -1, 1),
    (-1, 0, 2),
    (-1, 1, 1),
    (0, -1, 2),
    (0, 1, 2),
    (1, -1, 1),
    (1, 0, 2),
    (1, 1, 1),
)


REPAIR_ALGORITHM = {
    "name": "deterministic_inverse_square_diffusion",
    "neighbor_weights": "cardinal=2, diagonal=1 (equivalent to 1/d^2 scaled by 2)",
    "coarsest": "outside-in Jacobi diffusion from known pixels",
    "upper_layers": "diffusion candidate blended 1:1 with seed=(1-g)*current_init+g*upsampled_lower",
    "rounding": "half_up",
    "unresolved_fallback": "nearest-known diffusion, then seed/source",
}


@dataclass
class RepairResult:
    image: np.ndarray
    retained_source_mask: np.ndarray


def _neighbor_mean(values: np.ndarray, known: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width, _ = values.shape
    weighted_sum = np.zeros_like(values, dtype=np.int64)
    weight_sum = np.zeros((height, width), dtype=np.int64)
    for dy, dx, weight in NEIGHBOR_WEIGHTS:
        ty0 = max(0, -dy)
        ty1 = min(height, height - dy)
        tx0 = max(0, -dx)
        tx1 = min(width, width - dx)
        if ty0 >= ty1 or tx0 >= tx1:
            continue
        source_y = slice(ty0 + dy, ty1 + dy)
        source_x = slice(tx0 + dx, tx1 + dx)
        target_y = slice(ty0, ty1)
        target_x = slice(tx0, tx1)
        known_patch = known[source_y, source_x]
        weight_patch = known_patch.astype(np.int64) * weight
        weighted_sum[target_y, target_x] += values[source_y, source_x] * weight_patch[..., None]
        weight_sum[target_y, target_x] += weight_patch
    has_neighbor = weight_sum > 0
    safe_weight = np.where(has_neighbor, weight_sum, 1)
    candidate = round_div(weighted_sum, safe_weight[..., None])
    return candidate.astype(np.int64, copy=False), has_neighbor


def repair_tile(
    source: np.ndarray,
    guide: np.ndarray | None,
    mask: np.ndarray,
    iterations: int,
    guide_strength_q: int,
    scale: int,
    mask_fill: str,
) -> RepairResult:
    """Fill one tile. The tile has no cross-tile image dependency."""
    source = np.asarray(source, dtype=np.int64)
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return RepairResult(source.copy(), np.zeros_like(mask, dtype=bool))
    if mask_fill == "preserve_source":
        return RepairResult(source.copy(), mask.copy())

    if guide is None:
        seed = source.copy()
    else:
        guide = np.asarray(guide, dtype=np.int64)
        seed = blend_fixed(source, guide, guide_strength_q, scale).astype(np.int64, copy=False)

    values = source.copy()
    known = ~mask
    for _ in range(iterations):
        candidate, has_neighbor = _neighbor_mean(values, known)
        update = mask & has_neighbor
        if not update.any():
            continue
        if guide is None:
            values[update] = candidate[update]
        else:
            values[update] = (candidate[update] + seed[update] + 1) // 2
        known |= update

    unresolved = mask & ~known
    if unresolved.any() and guide is None:
        for _ in range(values.shape[0] + values.shape[1]):
            candidate, has_neighbor = _neighbor_mean(values, known)
            update = unresolved & has_neighbor
            if not update.any():
                break
            values[update] = candidate[update]
            known |= update
            unresolved &= ~update
    if unresolved.any():
        values[unresolved] = seed[unresolved]

    return RepairResult(values, np.zeros_like(mask, dtype=bool))

