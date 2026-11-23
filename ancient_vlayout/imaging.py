"""Image binarization, projection and tight-border regression helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .util import sha256_bytes


def binarize(gray: np.ndarray, threshold: int) -> np.ndarray:
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("binarize expects a uint8 grayscale array")
    return gray < int(threshold)


def _close_false_gaps(active: np.ndarray, max_gap: int) -> np.ndarray:
    if max_gap <= 0 or active.size == 0:
        return active.copy()
    out = active.copy()
    index = 0
    while index < out.size:
        if out[index]:
            index += 1
            continue
        start = index
        while index < out.size and not out[index]:
            index += 1
        gap = index - start
        if start > 0 and index < out.size and gap <= max_gap:
            out[start:index] = True
    return out


def _runs(active: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open runs of true values."""

    runs: list[tuple[int, int]] = []
    index = 0
    while index < active.size:
        if not active[index]:
            index += 1
            continue
        start = index
        while index < active.size and active[index]:
            index += 1
        runs.append((start, index))
    return runs


@dataclass(frozen=True)
class BorderRegression:
    bbox: tuple[int, int, int, int]
    centroid_q8: tuple[int, int]
    trimmed_top: int
    trimmed_bottom: int
    trimmed_left: int
    trimmed_right: int
    density_floor: int
    opening_radius: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "bbox": list(self.bbox),
            "centroid_q8": list(self.centroid_q8),
            "trimmed_top": self.trimmed_top,
            "trimmed_bottom": self.trimmed_bottom,
            "trimmed_left": self.trimmed_left,
            "trimmed_right": self.trimmed_right,
            "density_floor": self.density_floor,
            "opening_radius": self.opening_radius,
        }


def _binary_open(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    if radius <= 0 or mask.size == 0:
        return mask.copy()
    if radius == 1:
        # A 2x2 opening removes one-pixel bridges/dust while preserving
        # two-pixel or wider rendered strokes.
        source = mask.astype(bool)
        padded = np.pad(source, ((0, 1), (0, 1)), mode="constant", constant_values=False)
        eroded = (
            padded[:-1, :-1]
            & padded[1:, :-1]
            & padded[:-1, 1:]
            & padded[1:, 1:]
        )
        dilated_pad = np.pad(eroded, ((1, 0), (1, 0)), mode="constant", constant_values=False)
        return (
            dilated_pad[0 : source.shape[0], 0 : source.shape[1]]
            | dilated_pad[1 : source.shape[0] + 1, 0 : source.shape[1]]
            | dilated_pad[0 : source.shape[0], 1 : source.shape[1] + 1]
            | dilated_pad[1 : source.shape[0] + 1, 1 : source.shape[1] + 1]
        )
    padded = np.pad(mask.astype(bool), radius, mode="constant", constant_values=False)
    eroded = np.ones_like(mask, dtype=bool)
    size = 2 * radius + 1
    for dy in range(size):
        for dx in range(size):
            eroded &= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    padded = np.pad(eroded, radius, mode="constant", constant_values=False)
    dilated = np.zeros_like(mask, dtype=bool)
    for dy in range(size):
        for dx in range(size):
            dilated |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return dilated


def detect_column_x_ranges(ink: np.ndarray, projection_window: int) -> list[dict[str, Any]]:
    """Find vertical column x runs from the horizontal ink projection.

    Density is the count of ink rows for each x.  A two-pixel floor removes
    isolated dust while retaining rotated one-pixel stroke edges once they
    participate in a real column.  Small internal gaps are closed by the
    configured projection window.
    """

    if ink.ndim != 2:
        raise ValueError("ink mask must be two-dimensional")
    if ink.shape[1] == 0:
        return []
    density = ink.sum(axis=0, dtype=np.int64)
    floor = 3
    max_runs = np.zeros(ink.shape[1], dtype=np.int64)
    current = np.zeros(ink.shape[1], dtype=np.int64)
    for row in ink:
        current = np.where(row, current + 1, 0)
        max_runs = np.maximum(max_runs, current)
    active = (density >= floor) & (max_runs >= 3)
    active = _close_false_gaps(active, max(1, projection_window))
    min_width = max(4, projection_window + 1)
    ranges: list[dict[str, Any]] = []
    for x0, x1 in _runs(active):
        if x1 - x0 < min_width:
            continue
        peak = int(density[x0:x1].max())
        if peak < max(4, floor):
            continue
        ranges.append(
            {
                "x0": int(x0),
                "x1": int(x1),
                "peak_density": peak,
                "ink_columns": int(np.count_nonzero(density[x0:x1])),
            }
        )
    ranges.sort(key=lambda item: (-item["x0"], -item["x1"]))
    for index, item in enumerate(ranges):
        item["column_index"] = index
    return ranges


def segment_vertical_runs(
    ink: np.ndarray,
    x0: int,
    x1: int,
    projection_window: int,
    min_height: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split a column into vertical character candidate runs.

    Returns ``(candidates, discarded)``.  Projection gaps no wider than the
    projection window are treated as stroke gaps inside one character.
    """

    if not (0 <= x0 < x1 <= ink.shape[1]):
        raise ValueError("column bounds outside image")
    counts = ink[:, x0:x1].sum(axis=1, dtype=np.int64)
    active = counts > 0
    active = _close_false_gaps(active, projection_window)
    candidates: list[dict[str, Any]] = []
    discarded: list[dict[str, Any]] = []
    for y0, y1 in _runs(active):
        height = y1 - y0
        if height < min_height:
            discarded.append(
                {
                    "y0": int(y0),
                    "y1": int(y1),
                    "height": int(height),
                    "code": "E_SEGMENT_TOO_SHORT",
                    "reason": "segment height below min_char_height",
                    "min_char_height": int(min_height),
                }
            )
            continue
        candidates.append(
            {
                "y0": int(y0),
                "y1": int(y1),
                "height": int(height),
                "peak_density": int(counts[y0:y1].max()),
            }
        )
    return candidates, discarded


def column_ink_fingerprint(ink: np.ndarray, x0: int, x1: int) -> str:
    packed = np.packbits(ink[:, x0:x1].astype(np.uint8), axis=1)
    height = ink.shape[0]
    return sha256_bytes(packed.tobytes() + height.to_bytes(4, "little"))


def regress_border(
    ink: np.ndarray,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    density_floor: int = 1,
    opening_radius: int = 1,
) -> BorderRegression:
    """Trim whitespace and one-pixel bridges, then compute the ink centroid.

    The returned box is the tight integer ink box after row/column density
    filtering.  The centroid is retained as regression evidence and to make
    downstream ordering diagnostics explainable.
    """

    if not (0 <= x0 < x1 <= ink.shape[1] and 0 <= y0 < y1 <= ink.shape[0]):
        raise ValueError("regression bounds outside image")
    raw_sub = ink[y0:y1, x0:x1]
    sub = _binary_open(raw_sub, opening_radius)
    if not sub.any():
        sub = raw_sub.copy()
    row_counts = sub.sum(axis=1, dtype=np.int64)
    row_active = row_counts >= density_floor
    row_runs = _runs(row_active)
    if not row_runs:
        row_active = row_counts >= 1
        row_runs = _runs(row_active)
    if not row_runs:
        raise ValueError("segment contains no ink")
    top_rel = row_runs[0][0]
    bottom_rel = row_runs[-1][1]

    middle = sub[top_rel:bottom_rel, :]
    col_counts = middle.sum(axis=0, dtype=np.int64)
    col_active = col_counts >= density_floor
    col_runs = _runs(col_active)
    if not col_runs:
        col_active = col_counts >= 1
        col_runs = _runs(col_active)
    left_rel = col_runs[0][0]
    right_rel = col_runs[-1][1]

    box_sub = sub[top_rel:bottom_rel, left_rel:right_rel]
    yy, xx = np.nonzero(box_sub)
    if yy.size:
        # Fixed point centroid (1/256 pixel) keeps diagnostics reproducible.
        cy = int(round(float(yy.mean() + top_rel + y0) * 256.0))
        cx = int(round(float(xx.mean() + left_rel + x0) * 256.0))
    else:  # pragma: no cover - guarded by row_runs above
        cy = (y0 + y1) * 128
        cx = (x0 + x1) * 128

    bbox = (int(x0 + left_rel), int(y0 + top_rel), int(x0 + right_rel), int(y0 + bottom_rel))
    return BorderRegression(
        bbox=bbox,
        centroid_q8=(cx, cy),
        trimmed_top=int(top_rel),
        trimmed_bottom=int(y1 - y0 - bottom_rel),
        trimmed_left=int(left_rel),
        trimmed_right=int(x1 - x0 - right_rel),
        density_floor=int(density_floor),
        opening_radius=int(opening_radius),
    )


def projection_profile(ink: np.ndarray, x0: int, x1: int) -> np.ndarray:
    return ink[:, x0:x1].sum(axis=1, dtype=np.int64)
