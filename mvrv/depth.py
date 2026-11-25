from __future__ import annotations

from typing import Any

from .fixedpoint import SCALE, quantize_step
from .hashing import sha256_object


def depth_parameter_fingerprint(config: dict[str, Any]) -> str:
    return sha256_object(
        {
            "stage": "depth_parameters",
            "algorithm_version": config.get("algorithm_version", "mvrv-1"),
            "width": 64,
            "height": 48,
            "depth_quantization": int(config["depth_quantization"]) * SCALE,
        }
    )


def depth_ref_id(view_index: int, x: int, y: int) -> str:
    return f"D:{view_index:02d}:{x:02d},{y:02d}"


def pixel_cache_key(
    view_index: int,
    x: int,
    y: int,
    source_points: list[dict[str, Any]],
    parameter_fingerprint: str,
) -> str:
    sources = sorted((
        {"point_id": point["point_id"], "point_cache_key": point["cache_key"]}
        for point in source_points
    ), key=lambda item: item["point_id"])
    return sha256_object(
        {
            "stage": "depth_pixel",
            "depth_ref": depth_ref_id(view_index, x, y),
            "x": x,
            "y": y,
            "parameter_fingerprint": parameter_fingerprint,
            "sources": sources,
        }
    )


def compute_depth_pixel(
    view_index: int,
    x: int,
    y: int,
    source_points: list[dict[str, Any]],
    parameter_fingerprint: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    cache_key = pixel_cache_key(view_index, x, y, source_points, parameter_fingerprint)
    source_ids = sorted(point["point_id"] for point in source_points)
    if not source_points:
        return {
            "depth_ref": depth_ref_id(view_index, x, y),
            "view_index": view_index,
            "xy": [x, y],
            "depth_q": 0,
            "depth_level": 0,
            "depth_integer": 0,
            "rounding": "none",
            "boundary": False,
            "conflict": False,
            "source_count": 0,
            "source_points": [],
            "cache_key": cache_key,
            "sort_key": [view_index, y, x],
        }

    step_q = int(config["depth_quantization"]) * SCALE
    depths: list[tuple[int, str]] = []
    for point in source_points:
        raw_depth = int(point["view_depths"][str(view_index)])
        depths.append((raw_depth, point["point_id"]))
    raw_values = [value for value, _ in depths]
    spread_q = max(raw_values) - min(raw_values)
    half_step = step_q // 2
    boundary = len(depths) >= 2 and 0 < spread_q <= half_step
    conflict = len(depths) >= 2 and spread_q > half_step

    selected_depth, selected_id = min(depths, key=lambda item: (item[0], item[1]))
    depth_q, level, direction = quantize_step(selected_depth, step_q)
    return {
        "depth_ref": depth_ref_id(view_index, x, y),
        "view_index": view_index,
        "xy": [x, y],
        "raw_depth_q": selected_depth,
        "depth_q": depth_q,
        "depth_level": level,
        "depth_integer": level,
        "rounding": direction,
        "boundary": boundary,
        "conflict": conflict,
        "spread_q": spread_q,
        "half_step_q": half_step,
        "source_count": len(depths),
        "selected_point": selected_id,
        "source_points": source_ids,
        "cache_key": cache_key,
        "sort_key": [view_index, y, x],
    }
