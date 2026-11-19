"""Tolerance/order sensitivity metrics and drift classification."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from .boolean import BooleanOp, ResolutionOrder, boolean_operation
from .errors import ErrorCode, GlyphError
from .geometry import Vec, hausdorff_distance, loop_perimeter, total_abs_area
from .raster import Viewport, coverage_field, gray_distance, rasterize_loops


def symmetric_difference_area(
    loops_a: Sequence[Sequence[Vec]],
    loops_b: Sequence[Sequence[Vec]],
    *,
    resolution_order: ResolutionOrder | str = ResolutionOrder.LEXICOGRAPHIC,
) -> float:
    result = boolean_operation(loops_a, loops_b, BooleanOp.XOR, resolution_order=resolution_order)
    return total_abs_area(result.loops)


def symmetric_difference_area_details(
    loops_a: Sequence[Sequence[Vec]],
    loops_b: Sequence[Sequence[Vec]],
    *,
    resolution_order: ResolutionOrder | str = ResolutionOrder.LEXICOGRAPHIC,
    fallback_size: int = 256,
    viewport: Viewport = Viewport(),
) -> tuple[float, dict[str, Any]]:
    """Prefer exact arrangement XOR, fall back to an exact coverage-field integral.

    The fallback is required when two independently flattened boundaries contain
    sub-quantum slivers.  Per-pixel clipping is still exact before 8-bit
    quantization, so the integrated area is not a binary pixel count.
    """

    try:
        area = symmetric_difference_area(loops_a, loops_b, resolution_order=resolution_order)
        return area, {
            "method": "arrangement_xor",
            "resolution": None,
            "upper_error": 0.0,
        }
    except GlyphError as exc:
        if exc.code is not ErrorCode.BOOLEAN_DEGENERATE:
            raise
    field_a = coverage_field(loops_a, fallback_size, viewport=viewport)
    field_b = coverage_field(loops_b, fallback_size, viewport=viewport)
    pixel_area = (viewport.width / fallback_size) * (viewport.height / fallback_size)
    area = float(np.abs(field_a - field_b).sum() * pixel_area)
    return area, {
        "method": "coverage_field_integral",
        "resolution": fallback_size,
        "upper_error": pixel_area * 1e-9,
        "fallback_reason": "E_BOOLEAN_DEGENERATE",
    }


def region_metrics(
    loops_a: Sequence[Sequence[Vec]],
    loops_b: Sequence[Sequence[Vec]],
    *,
    tolerance_a: float,
    tolerance_b: float,
    bitmap_size: int,
    viewport: Viewport = Viewport(),
    resolution_order: ResolutionOrder | str = ResolutionOrder.LEXICOGRAPHIC,
) -> dict[str, Any]:
    area_delta, area_method = symmetric_difference_area_details(
        loops_a,
        loops_b,
        resolution_order=resolution_order,
        viewport=viewport,
    )
    hausdorff = hausdorff_distance(loops_a, loops_b)
    bitmap_a = rasterize_loops(loops_a, bitmap_size, viewport=viewport)
    bitmap_b = rasterize_loops(loops_b, bitmap_size, viewport=viewport)
    bitmap_delta = gray_distance(bitmap_a, bitmap_b)

    perimeter_a = sum(loop_perimeter(loop) for loop in loops_a)
    perimeter_b = sum(loop_perimeter(loop) for loop in loops_b)
    tolerance_sum = tolerance_a + tolerance_b
    area_limit = tolerance_sum * (perimeter_a + perimeter_b) + 4.0 * tolerance_sum * tolerance_sum
    area_limit += float(area_method["upper_error"])
    hausdorff_limit = tolerance_sum + 1e-7
    pixel_size = viewport.width / bitmap_size
    bitmap_pixel_limit = int(
        math.ceil(tolerance_sum / pixel_size * (perimeter_a + perimeter_b) / max(pixel_size, 1e-12))
    )

    area_drift = area_delta > area_limit + 1e-9
    distance_drift = float(hausdorff["distance"]) > hausdorff_limit + 1e-9
    drift = area_drift or distance_drift
    classification = "E_CONSISTENCY_DRIFT" if drift else "WITHIN_DECLARED_TOLERANCE"
    if not drift and bitmap_delta["differing_pixels"] > 0:
        classification = "WITHIN_TOLERANCE_BITMAP_DIFFERENCE"

    report: dict[str, Any] = {
        "symmetric_difference_area": area_delta,
        "symmetric_difference_area_method": area_method,
        "hausdorff_distance": hausdorff["distance"],
        "hausdorff_direction": hausdorff["direction"],
        "hausdorff_details": hausdorff,
        "bitmap": bitmap_delta,
        "error_interval": {
            "tolerance_a": tolerance_a,
            "tolerance_b": tolerance_b,
            "tolerance_sum": tolerance_sum,
            "area_upper": area_limit,
            "hausdorff_upper": hausdorff_limit,
            "bitmap_pixel_upper": bitmap_pixel_limit,
        },
        "drift": drift,
        "classification": classification,
        "divergent_edges": _divergent_edges(hausdorff),
        "comparison_order": str(resolution_order),
    }
    if drift:
        raise GlyphError(
            ErrorCode.CONSISTENCY_DRIFT,
            "geometry difference exceeds the declared tolerance interval",
            report,
        )
    return report


def _divergent_edges(hausdorff: dict[str, object]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for key in ("forward", "backward"):
        detail = hausdorff.get(key)
        if not isinstance(detail, dict):
            continue
        result.append(
            {
                "direction": key,
                "distance": detail.get("distance"),
                "source_loop": detail.get("source_loop"),
                "source_edge": detail.get("source_edge"),
                "target_loop": detail.get("target_loop"),
                "target_edge": detail.get("target_edge"),
                "point": detail.get("point"),
            }
        )
    return result


def require_no_drift(report: dict[str, Any]) -> None:
    if report.get("drift"):
        raise GlyphError(
            ErrorCode.CONSISTENCY_DRIFT,
            "consistency drift outside declared tolerance",
            report,
        )
