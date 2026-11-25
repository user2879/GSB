from __future__ import annotations

from typing import Any

import numpy as np


def binarize(gray: np.ndarray, threshold: int) -> np.ndarray:
    return gray < int(threshold)


def smooth_projection(projection: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window == 1:
        return projection.astype(float)
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(projection.astype(float), kernel, mode="same")


def find_active_runs(active: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start = None
    for i, value in enumerate(active):
        if value and start is None:
            start = i
        if start is not None and (not value or i == len(active) - 1):
            end = i if not value else i + 1
            if end > start:
                runs.append((int(start), int(end)))
            start = None
    return runs


def merge_runs(runs: list[tuple[int, int]], max_gap: int, limit: int) -> list[tuple[int, int]]:
    if not runs:
        return []
    merged = [list(runs[0])]
    for start, end in runs[1:]:
        if start - merged[-1][1] <= max_gap and end - merged[-1][0] <= limit:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(int(a), int(b)) for a, b in merged]


def detect_columns(gray: np.ndarray, config: dict[str, Any]) -> list[dict[str, Any]]:
    ink = binarize(gray, config["binarize_threshold"])
    projection = ink.sum(axis=0)
    smoothed = smooth_projection(projection, config["column_projection_window"])
    peak = float(smoothed.max(initial=0.0))
    cutoff = max(1.2, peak * 0.10)
    runs = find_active_runs(smoothed >= cutoff)
    runs = [(a, b) for a, b in runs if b - a >= 18]
    columns: list[dict[str, Any]] = []
    for index, (x1, x2) in enumerate(runs):
        pad = max(4, int(config["column_projection_window"]) // 2)
        x1 = max(0, x1 - pad)
        x2 = min(gray.shape[1], x2 + pad)
        columns.append({
            "detected_index_ltr": index,
            "rtl_index": 0,
            "x1": int(x1),
            "x2": int(x2),
            "interval_key": f"x{x1:04d}-{x2:04d}",
            "projection_peak": round(peak, 6),
        })
    columns.sort(key=lambda c: -c["x2"])
    for rtl_index, column in enumerate(columns):
        column["rtl_index"] = rtl_index
        column["column_key"] = f"col-{rtl_index}-{column['interval_key']}"
    return columns


def _valley_cost(projection: np.ndarray, y: int, y1: int, y2: int) -> float:
    segment = projection[y1:y2]
    peak = float(segment.max(initial=1.0)) or 1.0
    value = float(projection[y])
    depth = max(0.0, min(1.0, (peak - value) / peak))
    return 1.0 - depth


def split_tall_segment(
    projection: np.ndarray,
    y1: int,
    y2: int,
    config: dict[str, Any],
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    min_h = int(config["char_min_height"])
    max_h = int(config["char_max_height"])
    penalty = float(config["split_penalty"])
    n = len(projection)
    y1 = max(0, y1)
    y2 = min(n, y2)
    if y2 - y1 <= max_h:
        return [(y1, y2)], {"split": False}

    infinity = float("inf")
    best_cost: dict[int, float] = {y1: 0.0}
    best_cuts: dict[int, tuple[int, ...]] = {y1: ()}
    for end in range(y1 + min_h, y2 + 1):
        candidate_cost = infinity
        candidate_cuts: tuple[int, ...] | None = None
        lower = max(y1, end - max_h)
        for start in range(lower, end - min_h + 1):
            if start not in best_cost:
                continue
            if start > y1:
                cut_cost = _valley_cost(projection, start, y1, y2) + penalty
                cuts = best_cuts[start] + (start,)
            else:
                cut_cost = 0.0
                cuts = ()
            total = best_cost[start] + cut_cost
            if candidate_cuts is None or total < candidate_cost - 1e-12 or (
                abs(total - candidate_cost) <= 1e-12 and cuts < candidate_cuts
            ):
                candidate_cost = total
                candidate_cuts = cuts
        if candidate_cuts is not None:
            best_cost[end] = candidate_cost
            best_cuts[end] = candidate_cuts

    if y2 not in best_cost:
        return [], {"split": True, "reason": "no_dp_partition", "cost": None}
    cuts = best_cuts[y2]
    boundaries = [y1, *cuts, y2]
    segments = list(zip(boundaries[:-1], boundaries[1:]))
    meta = {
        "split": True,
        "cuts": [int(c) for c in cuts],
        "cost": round(float(best_cost[y2]), 9),
        "tie_rule": "equal_cost_choose_upper_split",
    }
    return [(int(a), int(b)) for a, b in segments], meta


def segment_column(
    gray: np.ndarray,
    column: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    projection = build_projection_candidates(gray, column, config)
    candidates = projection["candidates"]
    blocks: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for candidate in candidates:
        processed = process_candidate(gray, column, candidate, config)
        blocks.extend(processed["blocks"])
        dropped.extend(processed["dropped"])
    blocks.sort(key=lambda b: (b["rtl_index"], b["bbox"][1]))
    for char_index, block in enumerate(blocks):
        block["detected_char_index"] = char_index
        block["stable_char_key"] = f"{column['column_key']}-ch{char_index}"
    return {
        "column_key": column["column_key"],
        "candidates": candidates,
        "blocks": blocks,
        "dropped": dropped,
        "projection_peak": projection["projection_peak"],
    }


def build_projection_candidates(
    gray: np.ndarray,
    column: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    ink = binarize(gray, config["binarize_threshold"])
    x1, x2 = int(column["x1"]), int(column["x2"])
    column_ink = ink[:, x1:x2]
    y_projection = column_ink.sum(axis=1)
    smoothed = smooth_projection(y_projection, config["projection_window"])
    peak = float(smoothed.max(initial=0.0))
    cutoff = max(0.8, peak * 0.10)
    raw_runs = find_active_runs(smoothed >= cutoff)
    merge_gap = max(10, int(config["projection_window"]) * 3)
    merged = merge_runs(raw_runs, max_gap=merge_gap, limit=int(config["char_max_height"]) + merge_gap)

    candidates: list[dict[str, Any]] = []
    for seg_index, (sy1, sy2) in enumerate(merged):
        candidate_id = f"{column['column_key']}-seg{seg_index}"
        height = sy2 - sy1
        candidate = {
            "candidate_id": candidate_id,
            "column_key": column["column_key"],
            "seg_index": seg_index,
            "y1": int(sy1),
            "y2": int(sy2),
            "height": int(height),
        }
        candidates.append(candidate)
    return {
        "column_key": column["column_key"],
        "candidates": candidates,
        "projection_peak": round(peak, 6),
    }


def process_candidate(
    gray: np.ndarray,
    column: dict[str, Any],
    candidate: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    ink = binarize(gray, config["binarize_threshold"])
    x1, x2 = int(column["x1"]), int(column["x2"])
    smoothed = smooth_projection(ink[:, x1:x2].sum(axis=1), config["projection_window"])
    candidate_id = candidate["candidate_id"]
    seg_index = candidate["seg_index"]
    sy1, sy2 = int(candidate["y1"]), int(candidate["y2"])
    height = int(candidate["height"])
    blocks: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    if height < int(config["char_min_height"]):
        return {"blocks": [], "dropped": [{**candidate, "reason": "height_below_min"}]}

    split_segments, split_meta = split_tall_segment(smoothed, sy1, sy2, config)
    if not split_segments:
        return {"blocks": [], "dropped": [{**candidate, "reason": split_meta.get("reason", "split_failed"), "split_meta": split_meta}]}
    for part_index, (py1, py2) in enumerate(split_segments):
        part_height = py2 - py1
        ink_bbox = ink_bbox_in_region(ink, x1, x2, py1, py2)
        if ink_bbox is None:
            dropped.append({**candidate, "part_index": part_index, "reason": "no_ink_after_split"})
            continue
        ix1, iy1, ix2, iy2 = ink_bbox
        ink_w = max(1, ix2 - ix1)
        width_ratio = ink_w / max(1, part_height)
        if not (float(config["width_ratio_min"]) <= width_ratio <= float(config["width_ratio_max"])):
            dropped.append({
                **candidate,
                "part_index": part_index,
                "y1": py1,
                "y2": py2,
                "reason": "width_ratio_out_of_bounds",
                "width_ratio": round(width_ratio, 6),
            })
            continue
        bbox, regression = regress_char_bbox(ink, x1, x2, py1, py2, ink_bbox, config)
        blocks.append({
            "block_key": f"{candidate_id}-part{part_index}",
            "candidate_id": candidate_id,
            "column_key": column["column_key"],
            "rtl_index": column["rtl_index"],
            "seg_index": seg_index,
            "part_index": part_index,
            "y1": py1,
            "y2": py2,
            "bbox": bbox,
            "width_ratio": round(width_ratio, 6),
            "split_meta": split_meta if split_meta.get("split") else {"split": False},
            "regression": regression,
        })
    return {
        "blocks": blocks,
        "dropped": dropped,
    }


def ink_bbox_in_region(
    ink: np.ndarray,
    x1: int,
    x2: int,
    y1: int,
    y2: int,
) -> tuple[int, int, int, int] | None:
    region = ink[y1:y2, x1:x2]
    ys, xs = np.where(region)
    if len(xs) == 0:
        return None
    return (
        int(x1 + xs.min()),
        int(y1 + ys.min()),
        int(x1 + xs.max() + 1),
        int(y1 + ys.max() + 1),
    )


def regress_char_bbox(
    ink: np.ndarray,
    col_x1: int,
    col_x2: int,
    seg_y1: int,
    seg_y2: int,
    ink_bbox: tuple[int, int, int, int],
    config: dict[str, Any],
) -> tuple[list[int], dict[str, Any]]:
    ix1, iy1, ix2, iy2 = ink_bbox
    region = ink[seg_y1:seg_y2, col_x1:col_x2]
    ys, xs = np.where(region)
    if len(xs):
        centroid_x = float(col_x1 + xs.mean())
        centroid_y = float(seg_y1 + ys.mean())
    else:
        centroid_x = (ix1 + ix2) / 2.0
        centroid_y = (iy1 + iy2) / 2.0

    ink_h = max(1, iy2 - iy1)
    ink_w = max(1, ix2 - ix1)
    top_whitespace = int(max(0, iy1 - seg_y1))
    bottom_whitespace = int(max(0, seg_y2 - iy2))
    optical_center_y = int(round(0.5 * centroid_y + 0.25 * (iy1 + iy2)))
    optical_center_x = int(round(0.5 * centroid_x + 0.25 * (ix1 + ix2)))
    target_h = int(min(config["char_max_height"], max(config["char_min_height"], round(ink_h * 1.22))))
    target_w = int(min(col_x2 - col_x1, max(8, round(ink_w * 1.16))))
    y1 = optical_center_y - target_h // 2
    y2 = y1 + target_h
    x1 = optical_center_x - target_w // 2
    x2 = x1 + target_w

    y1 = max(seg_y1, min(y1, iy1))
    y2 = min(seg_y2, max(y2, iy2))
    if y2 - y1 < config["char_min_height"] and seg_y2 - seg_y1 >= config["char_min_height"]:
        center = (iy1 + iy2) // 2
        y1 = max(seg_y1, center - config["char_min_height"] // 2)
        y2 = min(seg_y2, y1 + config["char_min_height"])
    x1 = max(col_x1, x1)
    x2 = min(col_x2, x2)
    if x2 - x1 < target_w:
        if x1 == col_x1:
            x2 = min(col_x2, x1 + target_w)
        if x2 == col_x2:
            x1 = max(col_x1, x2 - target_w)
    bbox = [int(x1), int(y1), int(x2), int(y2)]
    regression = {
        "ink_bbox": [int(v) for v in ink_bbox],
        "centroid": [round(centroid_x, 3), round(centroid_y, 3)],
        "top_whitespace": top_whitespace,
        "bottom_whitespace": bottom_whitespace,
        "optical_center": [optical_center_x, optical_center_y],
        "borrowed_cross_column_pixels": False,
    }
    return bbox, regression


def detect_layout(gray: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    columns = detect_columns(gray, config)
    column_results = [segment_column(gray, column, config) for column in columns]
    blocks: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for result in column_results:
        blocks.extend(result["blocks"])
        dropped.extend(result["dropped"])
    blocks.sort(key=lambda b: (b["rtl_index"], b["bbox"][1]))
    for index, block in enumerate(blocks):
        block["global_detected_index"] = index
    return {"columns": columns, "column_results": column_results, "blocks": blocks, "dropped": dropped}
