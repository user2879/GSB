"""Quantized per-pixel depth maps with deterministic conflict resolution."""

from __future__ import annotations

from typing import Any

from .constants import ENGINE_VERSION
from .fixed import quantize, stable_hash
from .models import ReconstructionConfig, ValidatedView
from .triangulation import project_triangle


def depth_pixel_identifier(view_id: int, x: int, y: int) -> str:
    return f"d:{view_id:04d}:{y:04d}:{x:04d}"


def _candidate(triangle: dict[str, Any], depth_raw: int, config: ReconstructionConfig) -> dict[str, Any]:
    depth_q, direction, remainder = quantize(depth_raw, config.quant_step)
    return {
        "triangle_id": triangle["id"],
        "triangle_key": triangle["key"],
        "depth_raw": depth_raw,
        "depth_q": depth_q,
        "rounding": {
            "step": config.quant_step,
            "direction": direction,
            "remainder": remainder,
        },
    }


def collect_candidates(
    triangles: dict[str, dict[str, Any]],
    views: dict[int, ValidatedView],
    image_width: int,
    image_height: int,
    config: ReconstructionConfig,
) -> dict[int, dict[tuple[int, int], list[dict[str, Any]]]]:
    candidates: dict[int, dict[tuple[int, int], list[dict[str, Any]]]] = {
        view_id: {} for view_id in sorted(views)
    }
    for triangle_id in sorted(triangles):
        triangle = triangles[triangle_id]
        for view_id in sorted(views):
            projection = project_triangle(triangle, views[view_id].camera, config)
            if projection is None:
                continue
            x, y, depth_raw = projection
            if x < 0 or y < 0 or x >= image_width or y >= image_height:
                continue
            bucket = candidates[view_id].setdefault((x, y), [])
            bucket.append(_candidate(triangle, depth_raw, config))
    for view_id in candidates:
        for coordinate in candidates[view_id]:
            candidates[view_id][coordinate].sort(
                key=lambda item: (item["depth_q"], item["triangle_id"], item["triangle_key"])
            )
    return candidates


def _choose_candidate(candidates: list[dict[str, Any]], config: ReconstructionConfig) -> dict[str, Any]:
    ordered = sorted(
        candidates,
        key=lambda item: (item["depth_q"], item["depth_raw"], item["triangle_id"]),
    )
    selected = ordered[0]
    conflict = len(ordered) > 1
    within_half_step = False
    if conflict:
        delta = abs(ordered[1]["depth_raw"] - selected["depth_raw"])
        within_half_step = delta * 2 <= config.quant_step
    rule = (
        "HALF_STEP_MIN_DEPTH_THEN_TRIANGLE_ID"
        if within_half_step
        else "MIN_DEPTH_THEN_TRIANGLE_ID"
    )
    return {
        "selected": selected,
        "conflict": conflict,
        "within_half_step": within_half_step,
        "resolution_rule": rule,
        "candidate_count": len(ordered),
    }


def depth_pixel_key(
    view: ValidatedView,
    x: int,
    y: int,
    candidates: list[dict[str, Any]],
    config: ReconstructionConfig,
) -> str:
    dependency = [
        {
            "triangle_key": candidate["triangle_key"],
            "depth_raw": candidate["depth_raw"],
        }
        for candidate in candidates
    ]
    return stable_hash(
        {
            "engine_version": ENGINE_VERSION,
            "stage": "depth_pixel",
            "view_input_key": view.input_key,
            "bits": config.bits,
            "quant_level": config.quant_level,
            "quant_step": config.quant_step,
            "x": x,
            "y": y,
            "dependency": dependency,
        }
    )


def compute_pixel(
    view: ValidatedView,
    x: int,
    y: int,
    candidates: list[dict[str, Any]],
    config: ReconstructionConfig,
) -> dict[str, Any]:
    key = depth_pixel_key(view, x, y, candidates, config)
    if candidates:
        decision = _choose_candidate(candidates, config)
        selected = decision["selected"]
        record = {
            "id": depth_pixel_identifier(view.view_id, x, y),
            "key": key,
            "view_id": view.view_id,
            "x": x,
            "y": y,
            "depth_raw": selected["depth_raw"],
            "depth_q": selected["depth_q"],
            "rounding": selected["rounding"],
            "conflict": decision["conflict"],
            "within_half_step": decision["within_half_step"],
            "resolution_rule": decision["resolution_rule"],
            "candidate_count": decision["candidate_count"],
            "contributors": sorted(candidate["triangle_id"] for candidate in candidates),
            "contributor_keys": sorted(candidate["triangle_key"] for candidate in candidates),
            "empty": False,
        }
        return record
    return {
        "id": depth_pixel_identifier(view.view_id, x, y),
        "key": key,
        "view_id": view.view_id,
        "x": x,
        "y": y,
        "depth_raw": 0,
        "depth_q": 0,
        "rounding": {"step": config.quant_step, "direction": "none", "remainder": 0},
        "conflict": False,
        "within_half_step": False,
        "resolution_rule": "EMPTY",
        "candidate_count": 0,
        "contributors": [],
        "contributor_keys": [],
        "empty": True,
    }


def stage_view_record(
    view: ValidatedView,
    image_width: int,
    image_height: int,
    candidates: dict[tuple[int, int], list[dict[str, Any]]],
    pixels: list[dict[str, Any]],
) -> dict[str, Any]:
    conflict_pixels = sum(1 for pixel in pixels if pixel["conflict"])
    occupied_pixels = sum(1 for pixel in pixels if not pixel["empty"])
    key = stable_hash(
        {
            "engine_version": ENGINE_VERSION,
            "stage": "depth_view",
            "view_input_key": view.input_key,
            "pixel_keys": [pixel["key"] for pixel in pixels],
            "resolution": [image_width, image_height],
        }
    )
    return {
        "view_id": view.view_id,
        "key": key,
        "width": image_width,
        "height": image_height,
        "occupied_pixels": occupied_pixels,
        "conflict_pixels": conflict_pixels,
        "candidate_pixels": len(candidates),
        "pixels": pixels,
    }
