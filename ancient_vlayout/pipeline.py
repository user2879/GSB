"""Column detection, char segmentation, border regression and DP splitting."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import stage_param_fingerprint
from .imaging import (
    binarize,
    column_ink_fingerprint,
    detect_column_x_ranges,
    projection_profile,
    regress_border,
    segment_vertical_runs,
)
from .util import sha256_json


def make_columns_artifact(
    gray: np.ndarray,
    config: dict[str, Any],
    image_fingerprint: str,
    cache_key: str,
) -> dict[str, Any]:
    ink = binarize(gray, config["binarization_threshold"])
    columns = detect_column_x_ranges(ink, config["projection_window"])
    return {
        "kind": "columns",
        "key": cache_key,
        "image_fingerprint": image_fingerprint,
        "parameters_fingerprint": stage_param_fingerprint("columns", config),
        "columns": columns,
        "diagnostics": {
            "binarization_threshold": config["binarization_threshold"],
            "projection_window": config["projection_window"],
            "column_density_floor": 3,
            "column_min_vertical_run": 3,
        },
    }


def make_split_artifact(
    ink: np.ndarray,
    config: dict[str, Any],
    image_name: str,
    image_fingerprint: str,
    column_key: str,
    cache_key: str,
    column: dict[str, Any],
) -> dict[str, Any]:
    raw_candidates, discarded = segment_vertical_runs(
        ink,
        column["x0"],
        column["x1"],
        config["projection_window"],
        config["min_char_height"],
    )
    column_content_fingerprint = column_ink_fingerprint(ink, column["x0"], column["x1"])
    parameters_fingerprint = stage_param_fingerprint("split", config)
    candidates: list[dict[str, Any]] = []
    for ordinal, raw in enumerate(raw_candidates):
        segment_id = f"{image_name}:c{column['column_index']:04d}:s{ordinal:04d}"
        segment_fingerprint = sha256_json(
            {
                "stage": "segment-v1",
                "image_fingerprint": image_fingerprint,
                "column_content_fingerprint": column_content_fingerprint,
                "parameters_fingerprint": parameters_fingerprint,
                "x0": column["x0"],
                "x1": column["x1"],
                "y0": raw["y0"],
                "y1": raw["y1"],
            }
        )
        candidates.append(
            {
                "segment_id": segment_id,
                "segment_fingerprint": segment_fingerprint,
                "column_index": column["column_index"],
                "x0": column["x0"],
                "x1": column["x1"],
                "y0": raw["y0"],
                "y1": raw["y1"],
                "height": raw["height"],
                "peak_density": raw["peak_density"],
            }
        )
    return {
        "kind": "split",
        "key": cache_key,
        "image_name": image_name,
        "image_fingerprint": image_fingerprint,
        "column_key": column_key,
        "column_index": column["column_index"],
        "column_range": [column["x0"], column["x1"]],
        "column_content_fingerprint": column_content_fingerprint,
        "parameters_fingerprint": parameters_fingerprint,
        "candidates": candidates,
        "discarded": discarded,
        "diagnostics": {
            "discarded_count": len(discarded),
            "candidate_count": len(candidates),
        },
    }


def _cut_cost(profile: np.ndarray, cut: int, config: dict[str, Any]) -> tuple[int, int, int]:
    """Return (cost, valley_depth, local_peak) for a cut before row ``cut``."""

    radius = max(3, int(config["projection_window"]) * 3)
    start = max(0, cut - radius)
    stop = min(profile.size, cut + radius + 1)
    local_peak = int(profile[start:stop].max(initial=0)) if stop > start else 0
    local_value = int(profile[cut])
    depth = max(0, local_peak - local_value)
    # Lower projection means a deeper valley.  Keeping the score non-negative
    # also prevents extra splits from becoming cheaper merely by accumulating
    # valley rewards.
    cost = int(config["split_penalty"]) + local_value * int(config["split_valley_weight"])
    return cost, depth, local_peak


def _best_partition(profile: np.ndarray, config: dict[str, Any]) -> dict[str, Any] | None:
    height = int(profile.size)
    min_height = int(config["min_char_height"])
    max_height = int(config["max_char_height"])
    min_pieces = max(1, (height + max_height - 1) // max_height)
    max_pieces = max(1, height // min_height)
    if min_pieces < 1 or max_pieces < min_pieces:
        return None

    # dp[p][end] = (cost, cuts tuple) for p pieces covering rows [0, end).
    dp: dict[int, dict[int, tuple[int, tuple[int, ...]]]] = {1: {}}
    for end in range(min_height, min(max_height, height) + 1):
        dp[1][end] = (0, ())

    for pieces in range(2, max_pieces + 1):
        current: dict[int, tuple[int, tuple[int, ...]]] = {}
        previous = dp[pieces - 1]
        for start, (prev_cost, prev_cuts) in previous.items():
            low = start + min_height
            high = min(height, start + max_height)
            for end in range(low, high + 1):
                cut_cost, _, _ = _cut_cost(profile, start, config)
                candidate = (prev_cost + cut_cost, prev_cuts + (start,))
                existing = current.get(end)
                # Tuple comparison gives the required upper tie break because
                # the lexicographically smaller cut tuple is the upper choice.
                if existing is None or (candidate[0], candidate[1]) < (existing[0], existing[1]):
                    current[end] = candidate
        dp[pieces] = current

    solutions: list[tuple[int, tuple[int, ...], int]] = []
    for pieces in range(min_pieces, max_pieces + 1):
        hit = dp.get(pieces, {}).get(height)
        if hit is not None:
            solutions.append((hit[0], hit[1], pieces))
    if not solutions:
        return None
    cost, cuts, pieces = min(solutions, key=lambda item: (item[0], item[1], item[2]))
    return {"cost": int(cost), "cuts": tuple(cuts), "pieces": int(pieces)}


def _infeasible_cut_diagnostics(profile: np.ndarray, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Describe two-child candidate cuts rejected by size constraints.

    Two-child evaluations are the primary split hypothesis for a merged glyph
    pair.  Multi-child DP feasibility is still governed by the same bounds.
    """

    height = int(profile.size)
    min_height = int(config["min_char_height"])
    max_height = int(config["max_char_height"])
    result: list[dict[str, Any]] = []
    for cut in range(1, height):
        top = cut
        bottom = height - cut
        reasons: list[str] = []
        if top < min_height:
            reasons.append("top_below_min_height")
        if bottom < min_height:
            reasons.append("bottom_below_min_height")
        if top > max_height:
            reasons.append("top_above_max_height")
        if bottom > max_height:
            reasons.append("bottom_above_max_height")
        if reasons:
            result.append(
                {
                    "cut": int(cut),
                    "reasons": reasons,
                    "top_height": int(top),
                    "bottom_height": int(bottom),
                }
            )
    return result


def _tie_diagnostics(
    profile: np.ndarray, config: dict[str, Any], selected_cuts: tuple[int, ...]
) -> dict[str, Any]:
    height = int(profile.size)
    min_height = int(config["min_char_height"])
    max_height = int(config["max_char_height"])
    if len(selected_cuts) != 1:
        return {"tie": False, "tie_candidates": [], "tie_break": "upper"}
    valid: list[tuple[int, int]] = []
    for cut in range(min_height, height - min_height + 1):
        if cut > max_height or height - cut > max_height:
            continue
        cost, _, _ = _cut_cost(profile, cut, config)
        valid.append((cost, cut))
    if not valid:
        return {"tie": False, "tie_candidates": [], "tie_break": "upper"}
    best_cost = min(item[0] for item in valid)
    candidates = [cut for cost, cut in valid if cost == best_cost]
    return {
        "tie": len(candidates) > 1,
        "tie_candidates": [int(cut) for cut in candidates],
        "tie_break": "upper",
        "selected_cut": int(selected_cuts[0]),
    }


def make_border_artifact(
    ink: np.ndarray,
    config: dict[str, Any],
    cache_key: str,
    segment: dict[str, Any],
    column_key: str,
    split_key: str,
) -> dict[str, Any]:
    x0, x1 = int(segment["x0"]), int(segment["x1"])
    y0, y1 = int(segment["y0"]), int(segment["y1"])
    height = y1 - y0
    split_info: dict[str, Any] = {
        "applied": False,
        "reason": "height_within_max",
        "cuts": [],
        "cost": 0,
        "tie": False,
        "tie_candidates": [],
        "tie_break": "upper",
        "infeasible_cuts": [],
    }
    pieces: list[dict[str, Any]] = []
    if height > int(config["max_char_height"]):
        profile = projection_profile(ink, x0, x1)[y0:y1]
        best = _best_partition(profile, config)
        infeasible = _infeasible_cut_diagnostics(profile, config)
        if best is None:
            split_info.update(
                {
                    "applied": False,
                    "reason": "no_feasible_partition",
                    "diagnostic_code": "E_SPLIT_INFEASIBLE",
                    "infeasible_cuts": infeasible,
                }
            )
            bounds = [(y0, y1)]
        else:
            absolute_cuts = [y0 + int(cut) for cut in best["cuts"]]
            tie_info = _tie_diagnostics(profile, config, best["cuts"])
            cut_evidence = []
            for relative_cut in best["cuts"]:
                cut_cost, valley_depth, local_peak = _cut_cost(profile, relative_cut, config)
                cut_evidence.append(
                    {
                        "cut": int(y0 + relative_cut),
                        "projection": int(profile[relative_cut]),
                        "local_peak": int(local_peak),
                        "valley_depth": int(valley_depth),
                        "cut_cost": int(cut_cost),
                    }
                )
            split_info.update(
                {
                    "applied": True,
                    "reason": "height_above_max",
                    "cuts": absolute_cuts,
                    "cost": best["cost"],
                    "piece_count": best["pieces"],
                    "cut_evidence": cut_evidence,
                    "infeasible_cuts": infeasible,
                    **tie_info,
                }
            )
            bounds = []
            start = y0
            for cut in absolute_cuts:
                bounds.append((start, cut))
                start = cut
            bounds.append((start, y1))
    else:
        bounds = [(y0, y1)]

    for piece_index, (piece_y0, piece_y1) in enumerate(bounds):
        piece_id = f"{segment['segment_id']}:p{piece_index:04d}"
        try:
            regression = regress_border(ink, x0, x1, piece_y0, piece_y1)
            regression_dict = regression.as_dict()
            bbox = list(regression.bbox)
            error = None
        except ValueError as exc:
            regression_dict = {}
            bbox = [x0, piece_y0, x1, piece_y1]
            error = {"code": "E_BORDER_EMPTY", "message": str(exc)}
        pieces.append(
            {
                "piece_id": piece_id,
                "parent_segment_id": segment["segment_id"],
                "parent_segment_fingerprint": segment["segment_fingerprint"],
                "piece_index": piece_index,
                "raw_y0": int(piece_y0),
                "raw_y1": int(piece_y1),
                "bbox": [int(value) for value in bbox],
                "regression": regression_dict,
                "error": error,
            }
        )
    return {
        "kind": "border",
        "key": cache_key,
        "column_key": column_key,
        "split_key": split_key,
        "segment_id": segment["segment_id"],
        "segment_fingerprint": segment["segment_fingerprint"],
        "split": split_info,
        "pieces": pieces,
    }
