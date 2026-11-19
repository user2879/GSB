"""Sliding-window feature extraction and deterministic pixel classification.

All feature inputs originate from uint8 bands. Indices use integer numerators
and denominators (scaled by 1000) so identical integer pixels always produce
identical integers. Window generation is fully deterministic: top-left starts
follow the configured stride and are edge-clamped so every pixel is covered.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CLASS_CODES, CLASS_NAMES, SegmentationConfig

# Band positions fixed by the index band order (red, green, nir, elevation).
RED, GREEN, NIR, ELEVATION = 0, 1, 2, 3
SCALE = 1000


@dataclass(frozen=True)
class Window:
    top: int
    left: int
    core_top: int
    core_left: int
    core_bottom: int
    core_right: int

    @property
    def key(self) -> tuple[int, int]:
        return (self.top, self.left)


def generate_windows(tile_size: int, cfg: SegmentationConfig) -> list[Window]:
    """Deterministic sliding windows.

    Core regions form an exact, non-overlapping partition of the tile; each
    window is expanded by ``context_overlap`` pixels on every side (clamped at
    tile borders) so feature extraction sees contextual pixels.
    """

    ws = cfg.window_size
    stride = cfg.stride
    context = cfg.context_overlap
    if ws > tile_size or stride <= 2 * context:
        raise ValueError("invalid window/stride/context combination")
    core_step = stride - 2 * context
    core_starts = list(range(0, tile_size, core_step))
    windows: list[Window] = []
    for core_top in core_starts:
        core_bottom = min(core_top + core_step, tile_size)
        if core_top >= tile_size:
            continue
        for core_left in core_starts:
            core_right = min(core_left + core_step, tile_size)
            if core_left >= tile_size:
                continue
            top = max(0, core_top - context)
            left = max(0, core_left - context)
            bottom = min(tile_size, core_bottom + context)
            right = min(tile_size, core_right + context)
            if bottom - top > ws or right - left > ws:
                raise ValueError("context window exceeds configured window_size")
            windows.append(Window(top, left, core_top, core_left, core_bottom, core_right))
    return windows


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    denom = np.where(denominator == 0, 1, denominator)
    ratio = (numerator * SCALE) // denom
    return ratio.astype(np.int64)


def extract_window_features(bands: np.ndarray, window: Window) -> dict[str, np.ndarray]:
    """Compute the four feature planes for one window core region."""

    top, bottom = window.core_top, window.core_bottom
    left, right = window.core_left, window.core_right
    red = bands[top:bottom, left:right, RED].astype(np.int64)
    green = bands[top:bottom, left:right, GREEN].astype(np.int64)
    nir = bands[top:bottom, left:right, NIR].astype(np.int64)
    elevation = bands[top:bottom, left:right, ELEVATION].astype(np.int64)

    ndvi = _safe_ratio(nir - red, nir + red + 1)
    ndwi = _safe_ratio(green - nir, green + nir + 1)

    padded_red = np.pad(red, 1, mode="edge")
    diffs = [
        np.abs(padded_red[1:-1, 2:] - red),
        np.abs(padded_red[1:-1, :-2] - red),
        np.abs(padded_red[2:, 1:-1] - red),
        np.abs(padded_red[:-2, 1:-1] - red),
    ]
    edge = (np.sum(np.stack(diffs, axis=0), axis=0) // 4)

    padded_elev = np.pad(elevation, 1, mode="edge")
    grad_x = (padded_elev[1:-1, 2:] - padded_elev[1:-1, :-2]) // 2
    grad_y = (padded_elev[2:, 1:-1] - padded_elev[:-2, 1:-1]) // 2
    slope = np.abs(grad_x) + np.abs(grad_y)
    return {"ndvi": ndvi, "ndwi": ndwi, "edge": edge, "slope": slope}


def classify_tile(bands: np.ndarray, cfg: SegmentationConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (class_map uint8, confidence uint8, score_map int64 KxHxW)."""

    tile_size = bands.shape[0]
    windows = generate_windows(tile_size, cfg)
    score_map = np.zeros((len(CLASS_CODES), tile_size, tile_size), dtype=np.int64)
    covered = np.zeros((tile_size, tile_size), dtype=bool)

    for window in windows:
        features = extract_window_features(bands, window)
        ct, cb = window.core_top, window.core_bottom
        cl, cr = window.core_left, window.core_right
        for class_index, code in enumerate(CLASS_CODES):
            weights = cfg.class_weights[CLASS_NAMES[code]]
            score = np.zeros((cb - ct, cr - cl), dtype=np.int64)
            for feature_name, feature_plane in features.items():
                score += int(round(weights[feature_name])) * feature_plane
                # Deterministic threshold gate: every threshold edit changes
                # scores for the crossing pixels and therefore the fingerprint.
                threshold = int(round(cfg.feature_thresholds[feature_name] * SCALE))
                score += 100 * (feature_plane >= threshold).astype(np.int64)
            score += int(round(weights.get("bias", 0)))
            score_map[class_index, ct:cb, cl:cr] = score
        covered[ct:cb, cl:cr] = True

    if not np.all(covered):  # pragma: no cover - window generation guarantees cover
        raise RuntimeError("sliding windows failed to cover tile")

    # argmax over axis 0 returns the first (lowest class code) maximum on ties.
    class_indices = np.argmax(score_map, axis=0).astype(np.uint8)
    class_map = (class_indices + 1).astype(np.uint8)
    ordered = np.sort(score_map, axis=0)
    gap = ordered[-1] - ordered[-2]
    confidence = np.clip(gap, 0, 255).astype(np.uint8)
    return class_map, confidence, score_map
