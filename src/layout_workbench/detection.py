from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from .config import RecognitionConfig


def _window_positions(length: int, size: int, step: int) -> list[int]:
    if size > length:
        return []
    positions = list(range(0, length - size + 1, step))
    last = length - size
    if not positions or positions[-1] != last:
        positions.append(last)
    return positions


def _integral(array: np.ndarray) -> np.ndarray:
    result = np.zeros((array.shape[0] + 1, array.shape[1] + 1), dtype=np.int64)
    result[1:, 1:] = array.cumsum(axis=0).cumsum(axis=1)
    return result


def _rect_sum(integral: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    return (
        integral[y1, x1]
        - integral[y0, x1]
        - integral[y1, x0]
        + integral[y0, x0]
    )


def detect_blocks(gray: np.ndarray, config: RecognitionConfig) -> dict[str, Any]:
    """Sliding-window stroke/edge detector.

    A candidate window must contain enough binarized ink and strong local
    gradients.  Window score is a deterministic mixture of stroke density,
    edge density and darkness.  Overlapping windows are retained as evidence;
    line aggregation turns them into line candidates.
    """
    if gray.dtype != np.uint8 or gray.ndim != 2:
        raise ValueError("detector expects a uint8 grayscale image")
    ink = (gray <= config.binarization_threshold).astype(np.int64)
    dx = np.zeros_like(gray, dtype=np.int16)
    dy = np.zeros_like(gray, dtype=np.int16)
    dx[:, 1:] = np.abs(gray[:, 1:].astype(np.int16) - gray[:, :-1].astype(np.int16))
    dy[1:, :] = np.abs(gray[1:, :].astype(np.int16) - gray[:-1, :].astype(np.int16))
    edges = ((dx >= config.edge_difference_threshold) | (dy >= config.edge_difference_threshold)).astype(np.int64)
    ink_integral = _integral(ink)
    edge_integral = _integral(edges)
    gray_integral = _integral(gray.astype(np.int64))

    blocks: list[dict[str, Any]] = []
    for scale in config.window_scales:
        area = float(scale * scale)
        xs = _window_positions(gray.shape[1], scale, config.window_step)
        ys = _window_positions(gray.shape[0], scale, config.window_step)
        if not xs or not ys:
            continue
        for y in ys:
            for x in xs:
                x1 = x + scale
                y1 = y + scale
                ink_count = int(_rect_sum(ink_integral, x, y, x1, y1))
                density = ink_count / area
                if density < config.stroke_density_threshold:
                    continue
                edge_count = int(_rect_sum(edge_integral, x, y, x1, y1))
                edge_strength = edge_count / area
                if edge_strength < config.edge_strength_threshold:
                    continue
                darkness = 1.0 - (float(_rect_sum(gray_integral, x, y, x1, y1)) / area) / 255.0
                stroke_component = min(1.0, density / 0.20)
                edge_component = min(1.0, edge_strength / 0.10)
                dark_component = min(1.0, max(0.0, darkness) / 0.75)
                score = 0.50 * stroke_component + 0.35 * edge_component + 0.15 * dark_component
                blocks.append(
                    {
                        "bbox": [x, y, x1, y1],
                        "scale": int(scale),
                        "ink_density": round(density, 6),
                        "edge_strength": round(edge_strength, 6),
                        "score": round(score, 6),
                    }
                )
    blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0], b["scale"], b["bbox"][3], b["bbox"][2]))
    return {
        "stage": "detection",
        "image_size": [int(gray.shape[1]), int(gray.shape[0])],
        "parameters": config.detection_params,
        "blocks": blocks,
        "candidate_block_count": len(blocks),
        "scales_seen": sorted({b["scale"] for b in blocks}),
    }


def _y_bands(blocks: list[dict[str, Any]], gap: int) -> list[list[dict[str, Any]]]:
    bands: list[list[dict[str, Any]]] = []
    for block in sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][3], b["bbox"][0], b["scale"])):
        y0, y1 = block["bbox"][1], block["bbox"][3]
        if not bands:
            bands.append([block])
            continue
        current_y1 = max(b["bbox"][3] for b in bands[-1])
        if y0 <= current_y1 + gap:
            bands[-1].append(block)
        else:
            bands.append([block])
    return bands


def _x_intervals(blocks: list[dict[str, Any]], gap: int) -> list[tuple[int, int]]:
    intervals = sorted((b["bbox"][0], b["bbox"][2]) for b in blocks)
    merged: list[list[int]] = []
    for x0, x1 in intervals:
        if not merged or x0 > merged[-1][1] + gap:
            merged.append([x0, x1])
        else:
            merged[-1][1] = max(merged[-1][1], x1)
    return [(int(x0), int(x1)) for x0, x1 in merged]


def _refine_bbox(
    gray: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    threshold: int,
    min_pixels: int,
) -> tuple[int, int, int, int]:
    ink = gray[y0:y1, x0:x1] <= threshold
    if ink.size == 0:
        return x0, y0, x1, y1
    col_counts = ink.sum(axis=0)
    row_counts = ink.sum(axis=1)
    cols = np.flatnonzero(col_counts >= max(1, min_pixels))
    rows = np.flatnonzero(row_counts >= max(1, min_pixels))
    if len(cols) == 0 or len(rows) == 0:
        return x0, y0, x1, y1
    return (
        x0 + int(cols[0]),
        y0 + int(rows[0]),
        x0 + int(cols[-1]) + 1,
        y0 + int(rows[-1]) + 1,
    )


def aggregate_line_candidates(
    gray: np.ndarray,
    detection: dict[str, Any],
    config: RecognitionConfig,
) -> dict[str, Any]:
    blocks: list[dict[str, Any]] = detection["blocks"]
    candidates: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    bands = _y_bands(blocks, config.line_merge_projection_gap)
    for band in bands:
        band_y0 = min(b["bbox"][1] for b in band)
        band_y1 = max(b["bbox"][3] for b in band)
        for ix0, ix1 in _x_intervals(band, config.line_merge_projection_gap):
            relevant = [
                b
                for b in band
                if b["bbox"][2] > ix0 and b["bbox"][0] < ix1
            ]
            if not relevant:
                continue
            best = max(relevant, key=lambda b: (b["score"], -b["bbox"][1], -b["bbox"][0], b["scale"]))
            x0, y0, x1, y1 = _refine_bbox(
                gray,
                ix0,
                band_y0,
                ix1,
                band_y1,
                config.binarization_threshold,
                config.line_connectivity_min_pixels,
            )
            if x1 <= x0 or y1 <= y0:
                continue
            internal_blocks = sorted(
                [b for b in relevant if b["bbox"][2] > x0 and b["bbox"][0] < x1],
                key=lambda b: (b["bbox"][1], b["bbox"][0], b["scale"]),
            )
            record = {
                "bbox": [int(x0), int(y0), int(x1), int(y1)],
                "score": float(best["score"]),
                "scale": int(best["scale"]),
                "blocks": internal_blocks,
            }
            reason: str | None = None
            if y1 - y0 < config.min_char_height:
                reason = "min_char_height"
            elif record["score"] < config.candidate_score_threshold:
                reason = "candidate_score_threshold"
            if reason is None:
                record["line_index"] = len(candidates)
                candidates.append(record)
            else:
                record["reason"] = reason
                suppressed.append(record)

    candidates.sort(key=lambda r: (r["bbox"][1], r["bbox"][0], r["bbox"][3], r["bbox"][2], r["scale"]))
    for index, line in enumerate(candidates):
        line["line_index"] = index
    suppressed.sort(key=lambda r: (r["bbox"][1], r["bbox"][0], r["bbox"][3], r["bbox"][2], r["scale"]))
    return {
        "stage": "line_aggregation",
        "parameters": config.line_params,
        "candidate_lines": candidates,
        "suppressed_lines": suppressed,
        "line_bands": [
            {
                "bbox": [
                    min(b["bbox"][0] for b in band),
                    min(b["bbox"][1] for b in band),
                    max(b["bbox"][2] for b in band),
                    max(b["bbox"][3] for b in band),
                ],
                "block_count": len(band),
            }
            for band in bands
        ],
    }

