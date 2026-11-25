from __future__ import annotations

from typing import Any

import numpy as np

from . import (
    BBOX_RULE_VERSION,
    GRID_RULE_VERSION,
    INTENSITY_RULE_VERSION,
    STATS_RULE_VERSION,
)
from .config import fixed_scale, permille_to_fixed
from .errors import ErrorCode, make_issue


def half_up_ratio(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        return 0
    return int((2 * int(numerator) * 1000 + denominator) // (2 * denominator))


def nearest_rank_percentile(values: np.ndarray, percentile_permille: int) -> int:
    flat = np.sort(np.asarray(values, dtype=np.int32).reshape(-1))
    count = int(flat.size)
    if count == 0:
        return 0
    rank = (int(percentile_permille) * count + 999) // 1000
    rank = max(1, min(count, rank))
    return int(flat[rank - 1])


def largest_connected_component(mask: np.ndarray) -> tuple[int, list[tuple[int, int]]]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best_area = 0
    best_pixels: list[tuple[int, int]] = []
    seeds = np.argwhere(mask)
    for seed_y, seed_x in seeds:
        seed_y = int(seed_y)
        seed_x = int(seed_x)
        if visited[seed_y, seed_x] or not mask[seed_y, seed_x]:
            continue
        stack = [(seed_y, seed_x)]
        visited[seed_y, seed_x] = True
        pixels: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny = y + dy
                nx = x + dx
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if len(pixels) > best_area:
            best_area = len(pixels)
            best_pixels = pixels
    return best_area, best_pixels


def build_grid(image_data: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    width = int(image_data["width"])
    height = int(image_data["height"])
    rows = int(config["grid_rows"])
    cols = int(config["grid_cols"])
    cells: list[dict[str, Any]] = []
    for row in range(rows):
        y0 = (row * height) // rows
        y1 = ((row + 1) * height) // rows
        for col in range(cols):
            x0 = (col * width) // cols
            x1 = ((col + 1) * width) // cols
            component_id = row * cols + col + 1
            cells.append(
                {
                    "image": image_data["image"],
                    "cell_ref": f"{image_data['image']}#r{row:02d}c{col:02d}",
                    "component_id": component_id,
                    "row": row,
                    "col": col,
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                    "width_px": x1 - x0,
                    "height_px": y1 - y0,
                    "pixel_count": (x1 - x0) * (y1 - y0),
                    "interval_rule": GRID_RULE_VERSION,
                    "attribution_basis": "floor_boundaries;left/top closed and right/bottom open; boundary coordinate belongs to right/bottom cell",
                    "remainder_dimension": {
                        "width_is_long_cell": (x1 - x0) > width // cols,
                        "height_is_long_cell": (y1 - y0) > height // rows,
                    },
                }
            )
    return cells


def compute_cell_stats(
    heat: np.ndarray, cell: dict[str, Any], config: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    x0, y0, x1, y1 = cell["x0"], cell["y0"], cell["x1"], cell["y1"]
    sub = heat[y0:y1, x0:x1]
    pixel_count = int(sub.size)
    low_heat = nearest_rank_percentile(sub, int(config["percentile_low_permille"]))
    high_heat = nearest_rank_percentile(sub, int(config["percentile_high_permille"]))

    nonzero_pixels = int(np.count_nonzero(sub > 0))
    coverage_permille = half_up_ratio(nonzero_pixels, pixel_count)
    active_above_low_permille = half_up_ratio(int(np.count_nonzero(sub > low_heat)), pixel_count)
    hot_mask = sub >= high_heat
    hot_area, hot_pixels = largest_connected_component(hot_mask)
    hot_area_permille = half_up_ratio(hot_area, pixel_count)

    values_greater = int(np.count_nonzero(sub > high_heat))
    values_equal = int(np.count_nonzero(sub == high_heat))
    target_rank = max(1, min(pixel_count, (int(config["percentile_high_permille"]) * pixel_count + 999) // 1000))
    percentile_tie = values_greater < target_rank <= values_greater + values_equal

    min_area = int(config["min_hotspot_area_permille"])
    area_tie = hot_area_permille == min_area
    coverage_degraded = coverage_permille < int(config["min_coverage_permille"])
    silent_degraded = low_heat == 0 and high_heat == 0 and nonzero_pixels == 0

    stats: dict[str, Any] = {
        "stats_rule": STATS_RULE_VERSION,
        "pixel_count": pixel_count,
        "low_percentile_heat": low_heat,
        "high_percentile_heat": high_heat,
        "coverage_permille": coverage_permille,
        "active_above_low_permille": active_above_low_permille,
        "hot_area_pixels": hot_area,
        "hot_area_permille": hot_area_permille,
        "hot_pixel_count_ge_threshold": int(np.count_nonzero(hot_mask)),
        "percentile_tie_boundary": bool(percentile_tie),
        "percentile_tie_rule": "EQUAL_PERCENTILE_VALUE_INCLUDED_IN_HOT_MASK",
        "area_tie_boundary": bool(area_tie),
        "area_tie_rule": "AREA_RATIO_EQUAL_TO_MIN_AREA_IS_MARKED;MIN_AREA_TEST_IS_GE",
        "coverage_rule": "COVERAGE_IS_NONZERO_THERMAL_DATA_FRACTION",
        "active_above_low_rule": "EQUAL_TO_LOW_PERCENTILE_EXCLUDED_FROM_ACTIVE_FRACTION",
        "minimum_area_permille": min_area,
        "minimum_coverage_permille": int(config["min_coverage_permille"]),
        "degraded": bool(coverage_degraded or silent_degraded),
    }

    if hot_pixels:
        ys = np.asarray([p[0] for p in hot_pixels], dtype=np.int64)
        xs = np.asarray([p[1] for p in hot_pixels], dtype=np.int64)
        weights = sub[ys, xs].astype(np.float64)
        weight_sum = float(weights.sum())
        if weight_sum > 0:
            centroid_x = int(np.floor((xs * weights).sum() / weight_sum + 0.5))
            centroid_y = int(np.floor((ys * weights).sum() / weight_sum + 0.5))
        else:
            centroid_x = int(np.floor(xs.mean() + 0.5))
            centroid_y = int(np.floor(ys.mean() + 0.5))
        stats["hot_extent_local"] = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        stats["hot_centroid_local"] = [centroid_x, centroid_y]
    else:
        stats["hot_extent_local"] = None
        stats["hot_centroid_local"] = [(x1 - x0) // 2, (y1 - y0) // 2]

    issues: list[dict[str, Any]] = []
    if stats["degraded"]:
        issues.append(
            make_issue(
                ErrorCode.CELL_DEGRADED,
                "cell heat coverage is insufficient",
                image=cell["image"],
                cell_ref=cell["cell_ref"],
                component_id=cell["component_id"],
                coverage_permille=coverage_permille,
                minimum_coverage_permille=int(config["min_coverage_permille"]),
                hot_area_permille=hot_area_permille,
            )
        )
    return stats, issues


def regress_bbox(
    heat: np.ndarray, cell: dict[str, Any], stats: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    x0, y0, x1, y1 = cell["x0"], cell["y0"], cell["x1"], cell["y1"]
    cell_width = x1 - x0
    cell_height = y1 - y0
    padding = int(config["bbox_padding_px"])
    expand_x = max(padding, cell_width // 5)
    expand_y = max(padding, cell_height // 6)
    centroid_x, centroid_y = stats.get("hot_centroid_local") or [cell_width // 2, cell_height // 2]
    extent = stats.get("hot_extent_local")

    if stats.get("degraded") or extent is None:
        proposal = [
            x0 + max(2, cell_width // 8),
            y0 + max(2, cell_height // 8),
            x1 - max(2, cell_width // 8),
            y1 - max(2, cell_height // 8),
        ]
        fallback = True
    else:
        ex0, ey0, ex1, ey1 = extent
        proposal = [
            x0 + min(ex0, centroid_x) - expand_x - padding,
            y0 + min(ey0, centroid_y) - expand_y - padding,
            x0 + max(ex1, centroid_x + 1) + expand_x + padding,
            y0 + max(ey1, centroid_y + 1) + expand_y + padding,
        ]
        fallback = False

    clipped = [
        max(x0, proposal[0]),
        max(y0, proposal[1]),
        min(x1, proposal[2]),
        min(y1, proposal[3]),
    ]
    if clipped[2] <= clipped[0]:
        center_x = max(x0, min(x1 - 1, x0 + int(centroid_x)))
        clipped[0] = max(x0, center_x - max(1, cell_width // 4))
        clipped[2] = min(x1, center_x + max(1, cell_width // 4))
        if clipped[2] <= clipped[0]:
            clipped[2] = min(x1, clipped[0] + 1)
    if clipped[3] <= clipped[1]:
        center_y = max(y0, min(y1 - 1, y0 + int(centroid_y)))
        clipped[1] = max(y0, center_y - max(1, cell_height // 4))
        clipped[3] = min(y1, center_y + max(1, cell_height // 4))
        if clipped[3] <= clipped[1]:
            clipped[3] = min(y1, clipped[1] + 1)

    bbox_width = clipped[2] - clipped[0]
    bbox_height = clipped[3] - clipped[1]
    fill_permille = half_up_ratio(bbox_width * bbox_height, cell["pixel_count"])
    return {
        "bbox_rule": BBOX_RULE_VERSION,
        "bbox": [int(v) for v in clipped],
        "proposal_bbox": [int(v) for v in proposal],
        "clipped_to_cell": proposal != clipped,
        "fallback_centroid_bbox": fallback,
        "clip_rule": "CLIP_TO_EXACT_GRID_CELL;NO_CROSS_CELL_PIXEL_BORROW",
        "fill_permille": fill_permille,
        "centroid_global": [x0 + int(centroid_x), y0 + int(centroid_y)],
        "expansion_px": [expand_x + padding, expand_y + padding],
    }


def synthesize_intensity(
    stats: dict[str, Any], regression: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    thermal_span = max(0, int(stats["high_percentile_heat"]) - int(stats["low_percentile_heat"]))
    heat_term = thermal_span * 65
    area_term = int(stats["hot_area_permille"]) * 25
    fill_term = int(regression["fill_permille"]) * 10
    raw_permille = (heat_term + area_term + fill_term) // 100
    raw_permille = max(0, min(1000, raw_permille))

    decimals = int(config["fixed_decimals"])
    levels = int(config["quant_levels"])
    scale = fixed_scale(decimals)
    raw_fixed = permille_to_fixed(raw_permille, decimals)
    step = max(1, (scale + levels - 2) // max(1, levels - 1))
    quant_integer = min(levels - 1, raw_fixed // step)
    intensity_fixed = min(scale, int(quant_integer) * step)
    residual_fixed = raw_fixed - intensity_fixed
    intensity_permille = int(
        ((intensity_fixed * 1000) + (scale // 2 if scale > 1 else 0)) // scale
    ) if scale > 1 else intensity_fixed * 1000
    intensity_permille = max(0, min(1000, intensity_permille))

    return {
        "intensity_rule": INTENSITY_RULE_VERSION,
        "operation_order": [
            "thermal_span=max(0,high_percentile_heat-low_percentile_heat)",
            "sum=thermal_span*65+hot_area_permille*25+fill_permille*10",
            "raw_intensity_permille=sum//100",
            "quantize_to_configured_fixed_point",
        ],
        "thermal_span_permille_heat": thermal_span,
        "heat_term": heat_term,
        "area_term": area_term,
        "fill_term": fill_term,
        "raw_intensity_permille": raw_permille,
        "fixed_decimals": decimals,
        "fixed_scale": scale,
        "quant_levels": levels,
        "quant_scheme": f"PV_Q{decimals}_L{levels}_{config['quant_scheme']}",
        "quant_step_fixed": step,
        "raw_fixed": raw_fixed,
        "quant_integer": int(quant_integer),
        "intensity_fixed": intensity_fixed,
        "residual_fixed": residual_fixed,
        "intensity_permille": intensity_permille,
        "intensity_decimal": format(intensity_fixed / scale, f".{decimals}f") if decimals else str(intensity_fixed),
    }


GRADE_CODE_TO_NAME = {0: "NORMAL", 1: "WATCH", 2: "MINOR", 3: "MAJOR", 4: "CRITICAL", 9: "DEGRADED"}
GRADE_NAME_TO_CODE = {name: code for code, name in GRADE_CODE_TO_NAME.items()}


def determine_grade(intensity: dict[str, Any], config: dict[str, Any], degraded: bool) -> dict[str, Any]:
    if degraded:
        return {
            "grade_code": 9,
            "grade_name": GRADE_CODE_TO_NAME[9],
            "threshold_boundary": False,
            "boundary_threshold_permille": None,
            "comparison_rule": "DEGRADED_CELL_FORCED_CODE_9",
            "equality_rule": "EQUAL_TO_THRESHOLD_USES_HIGHER_GRADE",
        }
    value = int(intensity["intensity_permille"])
    thresholds = [int(v) for v in config["grade_thresholds_permille"]]
    code = 0
    boundary = False
    boundary_threshold = None
    for index, threshold in enumerate(thresholds, start=1):
        if value >= threshold:
            code = index
        if value == threshold:
            boundary = True
            boundary_threshold = threshold
    return {
        "grade_code": code,
        "grade_name": GRADE_CODE_TO_NAME[code],
        "threshold_boundary": boundary,
        "boundary_threshold_permille": boundary_threshold,
        "comparison_rule": "GRADE_IS_COUNT_OF_THRESHOLDS_LESS_OR_EQUAL_TO_INTENSITY",
        "equality_rule": "EQUAL_TO_THRESHOLD_USES_HIGHER_GRADE_AND_IS_MARKED",
    }


def intersection_over_union(bbox_a: list[int], bbox_b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0 else intersection / union
