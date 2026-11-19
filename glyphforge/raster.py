"""Unified 8-bit anti-aliased coverage rasterization and consistency checks."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .boolean import BooleanResult
from .errors import ErrorCode, GlyphError
from .geometry import Loop, Vec, clip_signed_area_to_box, ray_intersections, winding_number
from .stroke import StrokeResult


@dataclass(frozen=True, slots=True)
class Viewport:
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 64.0
    y1: float = 64.0

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass(frozen=True, slots=True)
class RasterBitmap:
    size_px: int
    viewport: Viewport
    data: bytes
    coverage_total: float
    coverage_max: int
    nonzero_pixels: int

    @property
    def bitmap_hash(self) -> str:
        return hashlib.sha256(self.data).hexdigest()

    def gray(self, row: int, col: int) -> int:
        return self.data[row * self.size_px + col]

    def as_array(self) -> np.ndarray:
        return np.frombuffer(self.data, dtype=np.uint8).reshape((self.size_px, self.size_px))

    def as_payload(self) -> dict[str, Any]:
        return {
            "size_px": self.size_px,
            "viewport": [self.viewport.x0, self.viewport.y0, self.viewport.x1, self.viewport.y1],
            "hash": self.bitmap_hash,
            "coverage_total": self.coverage_total,
            "coverage_max": self.coverage_max,
            "nonzero_pixels": self.nonzero_pixels,
        }


@dataclass(frozen=True, slots=True)
class BitmapConsistency:
    consistent: bool
    boundary_hash: str
    direct_hash: str
    differing_pixels: int
    first_divergence: dict[str, Any] | None
    declared_tolerance: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "consistent": self.consistent,
            "boundary_hash": self.boundary_hash,
            "direct_hash": self.direct_hash,
            "differing_pixels": self.differing_pixels,
            "first_divergence": self.first_divergence,
            "declared_tolerance": self.declared_tolerance,
        }


def rasterize_loops(
    loops: Sequence[Sequence[Vec]],
    size_px: int,
    *,
    viewport: Viewport = Viewport(),
) -> RasterBitmap:
    if size_px <= 0:
        raise GlyphError(ErrorCode.OUTLINE_INVALID, "bitmap size must be positive")
    if viewport.width <= 0.0 or viewport.height <= 0.0:
        raise GlyphError(ErrorCode.OUTLINE_INVALID, "viewport must have positive extent")
    coverage_accum = coverage_field(loops, size_px, viewport=viewport)
    image = np.floor(coverage_accum * 255.0 + 0.5).astype(np.uint8)

    data = image.tobytes(order="C")
    array = image.astype(np.float64) / 255.0
    return RasterBitmap(
        size_px,
        viewport,
        data,
        float(array.sum()),
        int(image.max(initial=0)),
        int(np.count_nonzero(image)),
    )


def coverage_field(
    loops: Sequence[Sequence[Vec]],
    size_px: int,
    *,
    viewport: Viewport = Viewport(),
) -> np.ndarray:
    """Return exact per-pixel polygon area coverage in [0, 1] before 8-bit quantization."""

    if size_px <= 0:
        raise GlyphError(ErrorCode.OUTLINE_INVALID, "bitmap size must be positive")
    if viewport.width <= 0.0 or viewport.height <= 0.0:
        raise GlyphError(ErrorCode.OUTLINE_INVALID, "viewport must have positive extent")
    pixel_x = size_px / viewport.width
    pixel_y = size_px / viewport.height
    pixel_loops: list[Loop] = []
    for loop in loops:
        if len(loop) < 3:
            continue
        transformed = [
            ((x - viewport.x0) * pixel_x, (viewport.y1 - y) * pixel_y) for x, y in loop
        ]
        pixel_loops.append(transformed)

    coverage_accum = np.zeros((size_px, size_px), dtype=np.float64)
    for loop in pixel_loops:
        xs = [point[0] for point in loop]
        ys = [point[1] for point in loop]
        col0 = max(0, int(math.floor(min(xs))) - 1)
        col1 = min(size_px - 1, int(math.ceil(max(xs))) + 1)
        row0 = max(0, int(math.floor(min(ys))) - 1)
        row1 = min(size_px - 1, int(math.ceil(max(ys))) + 1)
        for row in range(row0, row1 + 1):
            y0 = float(row)
            y1 = float(row + 1)
            for col in range(col0, col1 + 1):
                x0 = float(col)
                x1 = float(col + 1)
                # The y-axis flip into image coordinates reverses signed
                # orientation, so the CCW/CW convention is restored here.
                coverage = -clip_signed_area_to_box(loop, x0, y0, x1, y1)
                if coverage == 0.0:
                    continue
                coverage_accum[row, col] += coverage

    return np.clip(coverage_accum, 0.0, 1.0)


def rasterize_boolean(result: BooleanResult, size_px: int, *, viewport: Viewport = Viewport()) -> tuple[RasterBitmap, RasterBitmap]:
    return (
        rasterize_loops(result.loops, size_px, viewport=viewport),
        rasterize_loops(result.direct_loops, size_px, viewport=viewport),
    )


def rasterize_stroke(result: StrokeResult, size_px: int, *, viewport: Viewport = Viewport()) -> tuple[RasterBitmap, RasterBitmap]:
    return (
        rasterize_loops(result.loops, size_px, viewport=viewport),
        rasterize_loops(result.direct_loops, size_px, viewport=viewport),
    )


def check_bitmap_consistency(
    boundary: RasterBitmap,
    direct: RasterBitmap,
    *,
    boundary_loops: Sequence[Sequence[Vec]],
    direct_loops: Sequence[Sequence[Vec]],
    declared_tolerance: float,
    viewport: Viewport = Viewport(),
) -> BitmapConsistency:
    if boundary.size_px != direct.size_px or boundary.viewport != direct.viewport:
        raise GlyphError(ErrorCode.BITMAP_HASH, "bitmap dimensions or viewport differ")
    differing = 0
    first: dict[str, Any] | None = None
    size = boundary.size_px
    for index, (a, b) in enumerate(zip(boundary.data, direct.data)):
        if a == b:
            continue
        differing += 1
        if first is None:
            row, col = divmod(index, size)
            center = pixel_center(row, col, size, viewport)
            first = {
                "row": row,
                "col": col,
                "flat_index": index,
                "boundary_gray": int(a),
                "direct_gray": int(b),
                "boundary_winding": winding_number(center, boundary_loops),
                "direct_winding": winding_number(center, direct_loops),
                "tolerance": declared_tolerance,
                "pixel_center": {"x": center[0], "y": center[1]},
                "ray_intersections": ray_intersections(center, boundary_loops),
            }
    return BitmapConsistency(
        differing == 0,
        boundary.bitmap_hash,
        direct.bitmap_hash,
        differing,
        first,
        declared_tolerance,
    )


def require_bitmap_consistency(report: BitmapConsistency) -> None:
    if not report.consistent:
        raise GlyphError(
            ErrorCode.BITMAP_HASH,
            "boundary and direct rasterization disagree",
            report.as_dict(),
        )


def pixel_center(row: int, col: int, size_px: int, viewport: Viewport = Viewport()) -> Vec:
    x = viewport.x0 + (col + 0.5) * viewport.width / size_px
    y = viewport.y1 - (row + 0.5) * viewport.height / size_px
    return (x, y)


def gray_distance(a: RasterBitmap, b: RasterBitmap) -> dict[str, Any]:
    if a.size_px != b.size_px:
        raise GlyphError(ErrorCode.BITMAP_HASH, "cannot compare different bitmap sizes")
    total_abs = 0
    max_abs = 0
    differing = 0
    for left, right in zip(a.data, b.data):
        delta = abs(int(left) - int(right))
        total_abs += delta
        max_abs = max(max_abs, delta)
        differing += int(delta != 0)
    count = a.size_px * a.size_px
    return {
        "differing_pixels": differing,
        "mean_abs_gray": total_abs / count / 255.0,
        "max_abs_gray": max_abs,
        "boundary_hash": a.bitmap_hash,
        "other_hash": b.bitmap_hash,
    }
