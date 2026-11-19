"""Domain model for deterministic NPR scenes."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Iterable, Sequence

from .errors import E_FORMAT_INDEX, E_OUTLINE_DEGENERATE, E_SCHEMA_VERSION, E_STROKE_SAMPLE, NPRError
from .util import canonical_json_bytes, sha256_bytes, sha256_json

SCHEMA_VERSION = 3
SEGMENT_KINDS = ("line", "quadratic", "cubic")
SEGMENT_POINT_COUNTS = {"line": 2, "quadratic": 3, "cubic": 4}


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float

    def as_list(self) -> list[float]:
        return [float(self.x), float(self.y)]

    def distance_to(self, other: "Point") -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        return (dx * dx + dy * dy) ** 0.5


@dataclass(frozen=True, slots=True)
class StyleParams:
    tag: str
    tone: float
    roughness: float
    tint: tuple[int, int, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "tone": float(self.tone),
            "roughness": float(self.roughness),
            "tint": [int(self.tint[0]), int(self.tint[1]), int(self.tint[2])],
        }


@dataclass(frozen=True, slots=True)
class StrokeSampling:
    mode: str
    tolerance_px: float
    min_subdivisions: int
    max_subdivisions: int
    width_scale: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "tolerance_px": float(self.tolerance_px),
            "min_subdivisions": int(self.min_subdivisions),
            "max_subdivisions": int(self.max_subdivisions),
            "width_scale": float(self.width_scale),
        }


@dataclass(frozen=True, slots=True)
class Segment:
    kind: str
    points: tuple[Point, ...]

    def as_dict(self) -> dict[str, Any]:
        return {"type": self.kind, "points": [p.as_list() for p in self.points]}


@dataclass(frozen=True, slots=True)
class Element:
    id: str
    layer_id: str
    layer_order: int
    stroke_width: float
    style: StyleParams
    sampling: StrokeSampling
    closed: bool
    segments: tuple[Segment, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "layer_id": self.layer_id,
            "layer_order": int(self.layer_order),
            "stroke_width": float(self.stroke_width),
            "style": self.style.as_dict(),
            "sampling": self.sampling.as_dict(),
            "closed": bool(self.closed),
            "segments": [seg.as_dict() for seg in self.segments],
        }

    @property
    def content_hash(self) -> str:
        return _element_content_hash(self)


@lru_cache(maxsize=50000)
def _element_content_hash(element: Element) -> str:
    return sha256_json(element.as_dict())


@dataclass(frozen=True, slots=True)
class Layer:
    id: str
    name: str
    order: int
    visible: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "order": int(self.order),
            "visible": bool(self.visible),
        }


@dataclass(frozen=True, slots=True)
class Scene:
    schema_version: int
    seed: int
    canvas_width: int
    canvas_height: int
    generator: Any
    default_sampling: StrokeSampling
    layers: tuple[Layer, ...]
    elements: tuple[Element, ...]
    _project_bytes_cache: bytes | None = field(default=None, init=False, compare=False, repr=False)
    _content_hash_cache: str | None = field(default=None, init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.generator, MappingProxyType):
            object.__setattr__(self, "generator", MappingProxyType(dict(self.generator)))

    def as_project_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "generator": dict(self.generator),
            "canvas": {"width": int(self.canvas_width), "height": int(self.canvas_height)},
            "sampling": self.default_sampling.as_dict(),
            "layers": [layer.as_dict() for layer in self.layers],
            "elements": [element.as_dict() for element in self.elements],
        }

    @property
    def project_bytes(self) -> bytes:
        cached = self._project_bytes_cache
        if cached is None:
            cached = canonical_json_bytes(self.as_project_dict())
            object.__setattr__(self, "_project_bytes_cache", cached)
        return cached

    @property
    def content_hash(self) -> str:
        cached = self._content_hash_cache
        if cached is None:
            cached = sha256_bytes(self.project_bytes)
            object.__setattr__(self, "_content_hash_cache", cached)
        return cached

    def element_map(self) -> dict[str, Element]:
        return {element.id: element for element in self.elements}

    def layer_map(self) -> dict[str, Layer]:
        return {layer.id: layer for layer in self.layers}

    def with_element(self, element: Element) -> "Scene":
        found = False
        elements: list[Element] = []
        for current in self.elements:
            if current.id == element.id:
                elements.append(element)
                found = True
            else:
                elements.append(current)
        if not found:
            raise NPRError("E_EDIT_TARGET", "element not found", element_id=element.id)
        return replace(self, elements=tuple(elements))

    def element_hashes(self) -> dict[str, str]:
        return {element.id: element.content_hash for element in self.elements}


def point_from_list(value: Sequence[Any]) -> Point:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise NPRError(E_OUTLINE_DEGENERATE, "a point must contain exactly two coordinates")
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise NPRError(E_OUTLINE_DEGENERATE, "point coordinate is not numeric", value=value) from exc
    return Point(x, y)


def sampling_from_dict(raw: Any) -> StrokeSampling:
    if not isinstance(raw, dict):
        raise NPRError(E_STROKE_SAMPLE, "stroke sampling configuration must be an object")
    try:
        sampling = StrokeSampling(
            mode=str(raw.get("mode", "adaptive")),
            tolerance_px=float(raw.get("tolerance_px", 1.0)),
            min_subdivisions=int(raw.get("min_subdivisions", 1)),
            max_subdivisions=int(raw.get("max_subdivisions", 18)),
            width_scale=float(raw.get("width_scale", 1.0)),
        )
    except (TypeError, ValueError) as exc:
        raise NPRError(E_STROKE_SAMPLE, "invalid stroke sampling configuration", raw=raw) from exc
    if sampling.tolerance_px <= 0.0:
        raise NPRError(E_STROKE_SAMPLE, "tolerance_px must be > 0", tolerance_px=sampling.tolerance_px)
    if sampling.min_subdivisions < 1:
        raise NPRError(E_STROKE_SAMPLE, "min_subdivisions must be >= 1")
    if sampling.max_subdivisions < sampling.min_subdivisions or sampling.max_subdivisions > 24:
        raise NPRError(E_STROKE_SAMPLE, "max_subdivisions is outside the supported range")
    if sampling.width_scale <= 0.0:
        raise NPRError(E_STROKE_SAMPLE, "width_scale must be > 0")
    return sampling


def style_from_dict(raw: Any) -> StyleParams:
    if not isinstance(raw, dict):
        raise NPRError(E_SCHEMA_VERSION, "style parameters must be an object")
    try:
        tint_raw = raw.get("tint", [24, 31, 38])
        if not isinstance(tint_raw, (list, tuple)) or len(tint_raw) != 3:
            raise ValueError("tint must have three channels")
        tint = tuple(max(0, min(255, int(channel))) for channel in tint_raw)
        return StyleParams(
            tag=str(raw.get("tag", "ink")),
            tone=float(raw.get("tone", 0.4)),
            roughness=float(raw.get("roughness", 0.2)),
            tint=(int(tint[0]), int(tint[1]), int(tint[2])),
        )
    except (TypeError, ValueError) as exc:
        raise NPRError(E_SCHEMA_VERSION, "invalid style parameters", raw=raw) from exc


def segment_from_dict(raw: Any) -> Segment:
    if not isinstance(raw, dict):
        raise NPRError(E_OUTLINE_DEGENERATE, "segment must be an object")
    kind = str(raw.get("type", raw.get("kind", raw.get("segment_type", ""))))
    if kind not in SEGMENT_KINDS:
        raise NPRError(E_OUTLINE_DEGENERATE, "unsupported segment type", segment_type=kind)
    points_raw = raw.get("points", raw.get("control_points"))
    if not isinstance(points_raw, (list, tuple)):
        raise NPRError(E_OUTLINE_DEGENERATE, "segment points must be a list")
    points = tuple(point_from_list(point) for point in points_raw)
    if len(points) != SEGMENT_POINT_COUNTS[kind]:
        raise NPRError(
            E_OUTLINE_DEGENERATE,
            f"{kind} segment needs {SEGMENT_POINT_COUNTS[kind]} control points",
            got=len(points),
        )
    return Segment(kind=kind, points=points)


def element_from_dict(raw: Any) -> Element:
    if not isinstance(raw, dict):
        raise NPRError(E_SCHEMA_VERSION, "element must be an object")
    try:
        element_id = str(raw["id"])
        layer_id = str(raw.get("layer_id", raw.get("layer", "L00")))
        if "layer_order" in raw:
            layer_order = int(raw["layer_order"])
        elif "order" in raw:
            layer_order = int(raw["order"])
        else:
            layer_order = 0
        stroke_width = float(raw["stroke_width"])
        closed = bool(raw["closed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise NPRError(E_SCHEMA_VERSION, "element is missing required fields", element_id=raw.get("id")) from exc
    segments_raw = raw.get("segments")
    if not isinstance(segments_raw, (list, tuple)) or not segments_raw:
        raise NPRError(E_OUTLINE_DEGENERATE, "element must have at least one segment", element_id=element_id)
    segments = tuple(segment_from_dict(segment) for segment in segments_raw)
    if stroke_width <= 0.0:
        raise NPRError(E_STROKE_SAMPLE, "stroke width must be > 0", element_id=element_id)
    return Element(
        id=element_id,
        layer_id=layer_id,
        layer_order=layer_order,
        stroke_width=stroke_width,
        style=style_from_dict(raw.get("style", {})),
        sampling=sampling_from_dict(raw.get("sampling", {})),
        closed=closed,
        segments=segments,
    )


def layer_from_dict(raw: Any) -> Layer:
    if not isinstance(raw, dict):
        raise NPRError(E_SCHEMA_VERSION, "layer must be an object")
    try:
        return Layer(
            id=str(raw["id"]),
            name=str(raw.get("name", raw["id"])),
            order=int(raw.get("order", 0)),
            visible=bool(raw.get("visible", True)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise NPRError(E_SCHEMA_VERSION, "invalid layer record") from exc


def scene_from_project_dict(raw: Any) -> Scene:
    if not isinstance(raw, dict):
        raise NPRError(E_SCHEMA_VERSION, "project root must be an object")
    version = raw.get("schema_version")
    if version != SCHEMA_VERSION:
        raise NPRError(
            E_SCHEMA_VERSION,
            "project must be migrated to the current schema before loading",
            got=version,
            expected=SCHEMA_VERSION,
        )
    try:
        canvas = raw["canvas"]
        width = int(canvas["width"])
        height = int(canvas["height"])
        layers = tuple(layer_from_dict(layer) for layer in raw["layers"])
        elements = tuple(element_from_dict(element) for element in raw["elements"])
        seed = int(raw.get("generator", {}).get("seed", 0))
        generator = dict(raw.get("generator", {}))
        default_sampling = sampling_from_dict(raw.get("sampling", {}))
    except (KeyError, TypeError, ValueError) as exc:
        raise NPRError(E_SCHEMA_VERSION, "project root is missing required fields") from exc
    scene = Scene(
        schema_version=SCHEMA_VERSION,
        seed=seed,
        canvas_width=width,
        canvas_height=height,
        generator=generator,
        default_sampling=default_sampling,
        layers=tuple(layers),
        elements=tuple(elements),
    )
    validate_scene(scene)
    return scene


def validate_scene(scene: Scene) -> None:
    if scene.schema_version != SCHEMA_VERSION:
        raise NPRError(E_SCHEMA_VERSION, "unsupported scene schema", got=scene.schema_version)
    if scene.canvas_width <= 0 or scene.canvas_height <= 0:
        raise NPRError(E_SCHEMA_VERSION, "canvas dimensions must be positive")
    if not scene.layers:
        raise NPRError(E_SCHEMA_VERSION, "scene must contain at least one layer")
    layer_ids: set[str] = set()
    layer_orders: set[int] = set()
    for layer in scene.layers:
        if layer.id in layer_ids:
            raise NPRError(E_FORMAT_INDEX, "duplicate layer id", layer_id=layer.id)
        if layer.order in layer_orders:
            raise NPRError(E_FORMAT_INDEX, "duplicate layer order", order=layer.order)
        layer_ids.add(layer.id)
        layer_orders.add(layer.order)
    element_ids: set[str] = set()
    for element in scene.elements:
        if element.id in element_ids:
            raise NPRError(E_FORMAT_INDEX, "duplicate element id", element_id=element.id)
        element_ids.add(element.id)
        if element.layer_id not in layer_ids:
            raise NPRError(E_FORMAT_INDEX, "element references an unknown layer", element_id=element.id, layer_id=element.layer_id)
        if not element.segments:
            raise NPRError(E_OUTLINE_DEGENERATE, "element has no segments", element_id=element.id)


def layer_groups(scene: Scene) -> list[tuple[Layer, tuple[Element, ...]]]:
    group: dict[str, list[Element]] = {layer.id: [] for layer in scene.layers}
    for element in scene.elements:
        group.setdefault(element.layer_id, []).append(element)
    result: list[tuple[Layer, tuple[Element, ...]]] = []
    for layer in sorted(scene.layers, key=lambda item: (item.order, item.id)):
        result.append((layer, tuple(sorted(group.get(layer.id, ()), key=lambda item: item.id))))
    return result


def iter_points(element: Element) -> Iterable[Point]:
    for segment in element.segments:
        yield from segment.points
