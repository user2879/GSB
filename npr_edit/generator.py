"""Hash-stream based deterministic scene generator.

The generator intentionally does not use ``random`` so that the byte output is
stable across Python builds and platforms.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

from .model import SCHEMA_VERSION, Element, Layer, Point, Scene, Segment, StrokeSampling, StyleParams

DEFAULT_SEED = 20261120
DEFAULT_CANVAS = (640, 480)
LAYER_COUNT = 8
LAYER_NAMES = (
    "background-ink",
    "midground-hatch",
    "figure-ink",
    "figure-fill",
    "foreground-hatch",
    "accent-ink",
    "texture",
    "overlay",
)
TIER_TOLERANCE = {200: 0.45, 2000: 0.8, 20000: 1.5}
STYLE_TAGS = ("ink", "hatch", "toon", "pencil")
STROKE_WIDTHS = (0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
TINT_PALETTE = (
    (26, 32, 38),
    (42, 55, 66),
    (70, 82, 96),
    (33, 45, 58),
    (86, 69, 58),
    (48, 64, 72),
)


def _round6(value: float) -> float:
    result = round(value, 6)
    return 0.0 if result == 0.0 else result


class HashRng:
    """Counter-mode SHA-256 stream, stable on every platform."""

    def __init__(self, seed: int, stream: str) -> None:
        self._key = hashlib.sha256(f"{int(seed)}:{stream}".encode("ascii")).digest()
        self._counter = 0

    def next_u64(self) -> int:
        digest = hashlib.sha256(self._key + self._counter.to_bytes(8, "big")).digest()
        self._counter += 1
        return int.from_bytes(digest[:8], "big")

    def unit(self) -> float:
        return self.next_u64() / float(1 << 64)

    def randint(self, low: int, high: int) -> int:
        if high < low:
            raise ValueError("high must be >= low")
        return low + (self.next_u64() % (high - low + 1))

    def uniform(self, low: float, high: float) -> float:
        return low + (high - low) * self.unit()


@dataclass(slots=True)
class _ElementSpec:
    segments: int
    closed: bool
    radius: float
    center: Point
    phase: float
    layer_index: int


def _spec_for(index: int, seed: int, canvas: tuple[int, int]) -> _ElementSpec:
    rng = HashRng(seed, f"element-layout:{index:06d}")
    width, height = canvas
    low_y = 34.0 if index % 5 else 16.0
    center = Point(
        _round6(rng.uniform(18.0, width - 18.0)),
        _round6(rng.uniform(low_y, height - 18.0)),
    )
    return _ElementSpec(
        segments=3 + (rng.randint(0, 2)),
        closed=(rng.randint(0, 99) < 78),
        radius=_round6(rng.uniform(5.0, 25.0)),
        center=center,
        phase=rng.uniform(0.0, math.tau),
        layer_index=index % LAYER_COUNT,
    )


def _angle_point(center: Point, radius: float, angle: float, squash: float) -> Point:
    return Point(
        _round6(center.x + math.cos(angle) * radius),
        _round6(center.y + math.sin(angle) * radius * squash),
    )


def _non_degenerate(a: Point, b: Point) -> Point:
    if a.distance_to(b) >= 1e-4:
        return b
    return Point(_round6(b.x + 0.25), _round6(b.y + 0.25))


def _build_segment(rng: HashRng, kind: str, start: Point, end: Point, normal: tuple[float, float]) -> Segment:
    chord = max(start.distance_to(end), 0.5)
    bend = rng.uniform(-0.32, 0.32) * chord
    if kind == "line":
        end = _non_degenerate(start, end)
        return Segment(kind="line", points=(start, end))
    if kind == "quadratic":
        control = Point(
            _round6((start.x + end.x) * 0.5 + normal[0] * bend),
            _round6((start.y + end.y) * 0.5 + normal[1] * bend),
        )
        if control.distance_to(start) < 1e-4 and control.distance_to(end) < 1e-4:
            control = Point(_round6(control.x + 0.5), _round6(control.y + 0.5))
        return Segment(kind="quadratic", points=(start, control, end))
    first = Point(
        _round6(start.x + (end.x - start.x) / 3.0 + normal[0] * bend * 0.7),
        _round6(start.y + (end.y - start.y) / 3.0 + normal[1] * bend * 0.7),
    )
    second = Point(
        _round6(start.x + (end.x - start.x) * 2.0 / 3.0 + normal[0] * bend * 0.35),
        _round6(start.y + (end.y - start.y) * 2.0 / 3.0 + normal[1] * bend * 0.35),
    )
    if first.distance_to(start) < 1e-4:
        first = Point(_round6(first.x + 0.25), _round6(first.y + 0.25))
    if second.distance_to(end) < 1e-4:
        second = Point(_round6(second.x - 0.25), _round6(second.y - 0.25))
    return Segment(kind="cubic", points=(start, first, second, end))


def _make_element(index: int, seed: int, canvas: tuple[int, int], tolerance: float) -> Element:
    spec = _spec_for(index, seed, canvas)
    rng = HashRng(seed, f"element-content:{index:06d}")
    layer_id = f"L{spec.layer_index:02d}"
    angles: list[float] = []
    for segment_index in range(spec.segments + 1):
        step = math.tau / spec.segments
        angle = spec.phase + step * segment_index + rng.uniform(-0.16, 0.16) * step
        angles.append(angle)
    if spec.closed:
        angles[-1] = angles[0]
    squash = rng.uniform(0.68, 1.18)
    vertices = [_angle_point(spec.center, spec.radius * rng.uniform(0.84, 1.12), angle, squash) for angle in angles]
    segments: list[Segment] = []
    for segment_index in range(spec.segments):
        start = vertices[segment_index]
        end = vertices[segment_index + 1]
        if spec.closed and segment_index == spec.segments - 1:
            end = vertices[0]
        kind = ("line", "quadratic", "cubic")[rng.randint(0, 2)]
        dx, dy = end.x - start.x, end.y - start.y
        length = max(math.hypot(dx, dy), 1e-9)
        normal = (-dy / length, dx / length)
        segments.append(_build_segment(rng, kind, start, end, normal))
    tag = STYLE_TAGS[rng.randint(0, len(STYLE_TAGS) - 1)]
    tint = TINT_PALETTE[rng.randint(0, len(TINT_PALETTE) - 1)]
    element = Element(
        id=f"e{index:06d}",
        layer_id=layer_id,
        layer_order=spec.layer_index,
        stroke_width=STROKE_WIDTHS[rng.randint(0, len(STROKE_WIDTHS) - 1)],
        style=StyleParams(
            tag=tag,
            tone=_round6(rng.uniform(0.18, 0.78)),
            roughness=_round6(rng.uniform(0.05, 0.55)),
            tint=tint,
        ),
        sampling=StrokeSampling(
            mode="adaptive",
            tolerance_px=tolerance,
            min_subdivisions=1,
            max_subdivisions=18,
            width_scale=1.0,
        ),
        closed=spec.closed,
        segments=tuple(segments),
    )
    return element


def generate_scene(
    count: int,
    seed: int = DEFAULT_SEED,
    canvas: tuple[int, int] = DEFAULT_CANVAS,
) -> Scene:
    """Generate *count* elements, deterministically, for any tier."""

    if count < 1:
        raise ValueError("count must be positive")
    tolerance = TIER_TOLERANCE.get(count)
    if tolerance is None:
        tolerance = 1.0 if count <= 2000 else 1.5
    layers = tuple(
        Layer(id=f"L{index:02d}", name=name, order=index, visible=True)
        for index, name in enumerate(LAYER_NAMES)
    )
    elements = tuple(_make_element(index, seed, canvas, tolerance) for index in range(count))
    return Scene(
        schema_version=SCHEMA_VERSION,
        seed=seed,
        canvas_width=canvas[0],
        canvas_height=canvas[1],
        generator={
            "kind": "npr-deterministic-hash-v1",
            "seed": int(seed),
            "canvas": [int(canvas[0]), int(canvas[1])],
            "tier": int(count),
        },
        default_sampling=StrokeSampling(
            mode="adaptive",
            tolerance_px=tolerance,
            min_subdivisions=1,
            max_subdivisions=18,
            width_scale=1.0,
        ),
        layers=layers,
        elements=elements,
    )


def build_element_hash_manifest(scene: Scene, tier: int | None = None) -> dict[str, Any]:
    entries = [
        {
            "id": element.id,
            "layer_id": element.layer_id,
            "layer_order": element.layer_order,
            "content_hash": element.content_hash,
        }
        for element in scene.elements
    ]
    return {
        "format": "npr-element-hash-manifest-v1",
        "schema_version": scene.schema_version,
        "seed": scene.seed,
        "tier": int(tier if tier is not None else len(scene.elements)),
        "element_count": len(scene.elements),
        "layer_count": len(scene.layers),
        "scene_hash": scene.content_hash,
        "elements": entries,
    }
