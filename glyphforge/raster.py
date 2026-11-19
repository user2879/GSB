"""Deterministic scanline coverage rasterization to 8-bit grayscale."""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np

from .errors import bitmap_hash_error, budget_denied, fill_rule_error
from .geometry import (
    DEFAULT_PIXEL_TOLERANCE,
    DEFAULT_UNITS_PER_EM,
    FlattenedGlyph,
    flatten_glyph,
)
from .model import Glyph

FILL_RULES = ("nonzero", "evenodd")
_FILL_RULE_ALIASES = {
    "nonzero": "nonzero",
    "non-zero": "nonzero",
    "evenodd": "evenodd",
    "even-odd": "evenodd",
}


def normalize_fill_rule(fill_rule: str) -> str:
    if not isinstance(fill_rule, str):
        raise fill_rule_error("fill rule must be a string", fill_rule=fill_rule)
    normalized = _FILL_RULE_ALIASES.get(fill_rule.lower())
    if normalized is None:
        raise fill_rule_error(
            "unsupported fill rule",
            fill_rule=fill_rule,
            allowed=list(FILL_RULES),
        )
    return normalized


def bitmap_hash(width: int, height: int, pixels: bytes | bytearray | np.ndarray) -> str:
    """Return a domain-separated hash over dimensions and grayscale pixels."""

    if isinstance(pixels, np.ndarray):
        array = np.asarray(pixels, dtype=np.uint8)
        raw = array.tobytes(order="C")
    else:
        raw = bytes(pixels)
    digest = hashlib.sha256()
    digest.update(b"GLYPHFORGE-BITMAP\x00")
    digest.update(struct.pack(">II", int(width), int(height)))
    digest.update(raw)
    return digest.hexdigest()


def verify_bitmap_hash(result: RasterResult, expected_hash: str) -> None:
    actual = bitmap_hash(result.width, result.height, result.pixels)
    if actual != expected_hash:
        raise bitmap_hash_error(
            "bitmap content hash mismatch",
            glyph_id=result.glyph_id,
            size=result.size,
            fill_rule=result.fill_rule,
            expected=expected_hash,
            actual=actual,
        )


@dataclass(frozen=True)
class RasterConfig:
    units_per_em: float = DEFAULT_UNITS_PER_EM
    pixel_tolerance: float = DEFAULT_PIXEL_TOLERANCE
    quality: float = 1.0
    scanline_samples: int = 4
    padding: int = 1
    max_scanline_work_bytes: int = 4 * 1024 * 1024

    def __post_init__(self) -> None:
        if not math.isfinite(self.units_per_em) or self.units_per_em <= 0:
            raise budget_denied("units_per_em must be positive and finite")
        if not math.isfinite(self.pixel_tolerance) or self.pixel_tolerance <= 0:
            raise budget_denied("pixel_tolerance must be positive and finite")
        if not math.isfinite(self.quality) or self.quality <= 0:
            raise budget_denied("quality must be positive and finite")
        if isinstance(self.scanline_samples, bool) or not (
            1 <= self.scanline_samples <= 16
        ):
            raise budget_denied("scanline_samples must be in [1, 16]")
        if isinstance(self.padding, bool) or not (0 <= self.padding <= 16):
            raise budget_denied("padding must be in [0, 16]")
        if (
            isinstance(self.max_scanline_work_bytes, bool)
            or self.max_scanline_work_bytes <= 0
        ):
            raise budget_denied("max_scanline_work_bytes must be positive")


@dataclass(frozen=True)
class RasterResult:
    glyph_id: str
    size: int
    fill_rule: str
    width: int
    height: int
    pixels: bytes
    content_hash: str
    bearing_x: int
    bearing_y: int
    top: int
    advance: int
    flattened: FlattenedGlyph
    edge_count: int
    work_buffer_bytes: int
    is_empty_glyph: bool = False

    @property
    def bitmap(self) -> np.ndarray:
        array = np.frombuffer(self.pixels, dtype=np.uint8).reshape(
            (self.height, self.width)
        )
        array.setflags(write=False)
        return array

    @property
    def bitmap_hash(self) -> str:
        return self.content_hash

    @property
    def non_zero_pixels(self) -> int:
        return int(np.count_nonzero(self.bitmap))

    def to_dict(self, *, include_bitmap_bytes: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "glyph_id": self.glyph_id,
            "size": self.size,
            "fill_rule": self.fill_rule,
            "width": self.width,
            "height": self.height,
            "bearing_x": self.bearing_x,
            "bearing_y": self.bearing_y,
            "top": self.top,
            "advance": self.advance,
            "bitmap_hash": self.content_hash,
            "non_zero_pixels": self.non_zero_pixels,
            "edge_count": self.edge_count,
            "work_buffer_bytes": self.work_buffer_bytes,
            "flatten": {
                "tolerance_pixels": self.flattened.tolerance_pixels,
                "scale": self.flattened.scale,
                "vertex_count": self.flattened.vertex_count,
                "segments": [
                    segment.to_dict()
                    for contour in self.flattened.contours
                    for segment in contour.segments
                ],
            },
        }
        if include_bitmap_bytes:
            import base64

            result["pixels_base64"] = base64.b64encode(self.pixels).decode("ascii")
        return result


def _edge_list(
    flattened: FlattenedGlyph, left: int, top: int
) -> list[tuple[float, float, float, float, int]]:
    edges: list[tuple[float, float, float, float, int]] = []
    edge_index = 0
    for contour in flattened.contours:
        points = contour.points
        if not points:
            continue
        transformed = [(point.x - left, top - point.y) for point in points]
        for index, start in enumerate(transformed):
            end = transformed[(index + 1) % len(transformed)]
            x0, y0 = start
            x1, y1 = end
            if y0 == y1:
                edge_index += 1
                continue
            edges.append((x0, y0, x1, y1, edge_index))
            edge_index += 1
    return edges


def _span_intersections(
    edges: list[tuple[float, float, float, float, int]],
    sample_y: float,
    edge_min_y: list[float],
    edge_max_y: list[float],
) -> list[tuple[float, int, int]]:
    crossings: list[tuple[float, int, int]] = []
    for edge_index, (x0, y0, x1, y1, stable_index) in enumerate(edges):
        low = edge_min_y[edge_index]
        high = edge_max_y[edge_index]
        if sample_y < low or sample_y >= high:
            continue
        denominator = y1 - y0
        t = (sample_y - y0) / denominator
        x = x0 + t * (x1 - x0)
        direction = 1 if y1 > y0 else -1
        crossings.append((x, direction, stable_index))
    crossings.sort(key=lambda item: (item[0], item[1], item[2]))
    return crossings


def _spans(
    crossings: list[tuple[float, int, int]], fill_rule: str
) -> list[tuple[float, float]]:
    if not crossings:
        return []
    spans: list[tuple[float, float]] = []
    if fill_rule == "evenodd":
        index = 0
        while index + 1 < len(crossings):
            left = crossings[index][0]
            right = crossings[index + 1][0]
            if right > left:
                spans.append((left, right))
            index += 2
        return spans

    winding = 0
    span_start: float | None = None
    for x, delta, _stable in crossings:
        previous = winding
        winding += delta
        if previous == 0 and winding != 0:
            span_start = x
        elif previous != 0 and winding == 0 and span_start is not None:
            if x > span_start:
                spans.append((span_start, x))
            span_start = None
    return spans


def _estimate_work_buffer(
    width: int, height: int, edge_count: int, samples: int
) -> int:
    # float64 coverage + uint8 output + per-subscan intersection capacity.
    return int(width * height * 9 + max(width, edge_count) * samples * 32 + 4096)


def rasterize_glyph(
    glyph: Glyph,
    size: int,
    fill_rule: str = "nonzero",
    *,
    config: RasterConfig | None = None,
) -> RasterResult:
    """Rasterize one glyph at one pixel-size tier and fill rule.

    The coverage calculation uses ``scanline_samples`` horizontal rays per
    output row.  Each ray is a pixel-row sample; horizontal overlap is computed
    analytically and accumulated into an 8-bit grayscale coverage map.
    """

    if not isinstance(glyph, Glyph):
        raise budget_denied("rasterize_glyph requires a Glyph")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise budget_denied("pixel size must be a positive integer", size=size)
    normalized_rule = normalize_fill_rule(fill_rule)
    configuration = config or RasterConfig()
    flattened = flatten_glyph(
        glyph,
        size,
        units_per_em=configuration.units_per_em,
        pixel_tolerance=configuration.pixel_tolerance,
        quality=configuration.quality,
    )

    all_points = [point for contour in flattened.contours for point in contour.points]
    if not all_points:
        width = height = 1
        left = top = 0
        pixels = bytes(1)
        content_hash = bitmap_hash(width, height, pixels)
        return RasterResult(
            glyph_id=glyph.glyph_id,
            size=size,
            fill_rule=normalized_rule,
            width=width,
            height=height,
            pixels=pixels,
            content_hash=content_hash,
            bearing_x=0,
            bearing_y=0,
            top=0,
            advance=max(1, math.ceil(size * 0.5)),
            flattened=flattened,
            edge_count=0,
            work_buffer_bytes=_estimate_work_buffer(
                1, 1, 0, configuration.scanline_samples
            ),
            is_empty_glyph=True,
        )

    min_x = min(point.x for point in all_points)
    max_x = max(point.x for point in all_points)
    min_y = min(point.y for point in all_points)
    max_y = max(point.y for point in all_points)
    padding = configuration.padding
    left = math.floor(min_x) - padding
    right = math.ceil(max_x) + padding
    bottom = math.floor(min_y) - padding
    top = math.ceil(max_y) + padding
    width = max(1, right - left)
    height = max(1, top - bottom)

    edges = _edge_list(flattened, left, top)
    estimated_work = _estimate_work_buffer(
        width,
        height,
        len(edges),
        configuration.scanline_samples,
    )
    if estimated_work > configuration.max_scanline_work_bytes:
        raise budget_denied(
            "single glyph scanline work buffer exceeds budget",
            glyph_id=glyph.glyph_id,
            size=size,
            required_bytes=estimated_work,
            budget_bytes=configuration.max_scanline_work_bytes,
        )

    coverage = np.zeros((height, width), dtype=np.float64)
    edge_min_y = [min(edge[1], edge[3]) for edge in edges]
    edge_max_y = [max(edge[1], edge[3]) for edge in edges]
    inv_samples = 1.0 / float(configuration.scanline_samples)

    for row in range(height):
        for sample in range(configuration.scanline_samples):
            sample_y = row + (sample + 0.5) * inv_samples
            crossings = _span_intersections(edges, sample_y, edge_min_y, edge_max_y)
            for span_left, span_right in _spans(crossings, normalized_rule):
                clipped_left = max(0.0, span_left)
                clipped_right = min(float(width), span_right)
                if clipped_right <= clipped_left:
                    continue
                first_column = max(0, math.floor(clipped_left))
                last_column = min(width - 1, math.ceil(clipped_right) - 1)
                for column in range(first_column, last_column + 1):
                    overlap = min(clipped_right, column + 1.0) - max(
                        clipped_left, float(column)
                    )
                    if overlap > 0.0:
                        coverage[row, column] += overlap * inv_samples

    scaled = np.clip(coverage, 0.0, 1.0) * 255.0 + 0.5
    bitmap = np.floor(scaled).astype(np.uint8)
    bitmap.setflags(write=False)
    pixels = bitmap.tobytes(order="C")
    content_hash = bitmap_hash(width, height, pixels)
    advance = max(1, math.ceil(max_x + size * 0.04))
    return RasterResult(
        glyph_id=glyph.glyph_id,
        size=size,
        fill_rule=normalized_rule,
        width=width,
        height=height,
        pixels=pixels,
        content_hash=content_hash,
        bearing_x=left,
        bearing_y=bottom,
        top=top,
        advance=advance,
        flattened=flattened,
        edge_count=len(edges),
        work_buffer_bytes=estimated_work,
    )


def estimate_raster_work_bytes(
    glyph: Glyph, size: int, config: RasterConfig | None = None
) -> int:
    """Conservative task memory estimate used by the thread scheduler.

    This intentionally avoids flattening so scheduling itself has bounded
    overhead.  It assumes the glyph can occupy a slightly oversized em square.
    """

    configuration = config or RasterConfig()
    dimension = math.ceil(size * 1.5) + configuration.padding * 2 + 2
    segment_count = sum(len(contour.segments) for contour in glyph.contours)
    curve_fanout = 32 if segment_count else 1
    pixel_bytes = dimension * dimension * 9
    edge_bytes = segment_count * curve_fanout * 64
    intersection_bytes = dimension * configuration.scanline_samples * 48
    return max(4096, pixel_bytes + edge_bytes + intersection_bytes + 4096)
