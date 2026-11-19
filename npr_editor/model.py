"""Canonical in-memory model and NPR project JSON codec."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .errors import NprError, require

CURRENT_SCHEMA_VERSION = 3
SUPPORTED_SCHEMA_VERSIONS = (1, 2, 3)
SEGMENT_ARITY = {"line": 2, "quadratic": 3, "cubic": 4}

Color = tuple[int, int, int]
Point = tuple[float, float]

DEFAULT_STYLE: dict[str, Any] = {
    "line_color": [18, 18, 20],
    "paper_color": [247, 244, 236],
    "line_opacity": 1.0,
    "width_scale": 1.0,
    "sketch_ink": [72, 70, 68],
    "sketch_hatch_spacing": 7.0,
    "sketch_hatch_angle": 42.0,
    "sketch_shade_strength": 0.24,
    "toon_palette": {
        "skin": [244, 193, 153],
        "leaf": [92, 160, 96],
        "accent": [224, 113, 84],
        "neutral": [205, 199, 185],
    },
    "toon_default_color": [205, 199, 185],
    "toon_shadow_color": [112, 91, 104],
    "toon_light": [0.45, -0.8],
    "toon_light_strength": 0.22,
}

DEFAULT_STROKE_SAMPLE: dict[str, Any] = {
    "curve_steps": 12,
    "line_steps": 1,
    "width_scale": 1.0,
    "min_segment_px": 0.0,
}


def f32(value: Any, *, field_name: str = "number") -> float:
    """Quantize a JSON number exactly as it will be stored in a project."""

    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise NprError(
            "E_SCHEMA_VERSION", f"{field_name} must be numeric", field=field_name
        ) from exc
    if not math.isfinite(parsed):
        raise NprError(
            "E_SCHEMA_VERSION", f"{field_name} must be finite", field=field_name
        )
    return float(np.float32(parsed))


def _as_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise NprError(
            "E_SCHEMA_VERSION", f"{field_name} must be an integer", field=field_name
        )
    return value


def _as_color(value: Any, *, field_name: str) -> Color:
    require(
        isinstance(value, (list, tuple)) and len(value) == 3,
        "E_SCHEMA_VERSION",
        f"{field_name} must have three channels",
        field=field_name,
    )
    result: list[int] = []
    for channel in value:
        if isinstance(channel, bool) or not isinstance(channel, (int, float)):
            raise NprError(
                "E_SCHEMA_VERSION", f"{field_name} channels must be numeric"
            )
        channel_int = int(round(float(channel)))
        require(
            0 <= channel_int <= 255,
            "E_SCHEMA_VERSION",
            f"{field_name} channel outside 0..255",
            field=field_name,
        )
        result.append(channel_int)
    return (result[0], result[1], result[2])


def _finite_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NprError("E_SCHEMA_VERSION", f"{field_name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise NprError("E_SCHEMA_VERSION", f"{field_name} must be finite")
    return parsed


@dataclass
class Segment:
    type: str
    points: list[Point]

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "points": [[x, y] for x, y in self.points]}

    def geometry_dict(self) -> dict[str, Any]:
        return self.to_dict()


@dataclass
class Contour:
    closed: bool
    segments: list[Segment]

    def to_dict(self) -> dict[str, Any]:
        return {
            "closed": bool(self.closed),
            "segments": [segment.to_dict() for segment in self.segments],
        }


@dataclass
class Shape:
    id: str
    layer: str
    stroke_width: float
    style_tags: list[str]
    contours: list[Contour]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "layer": self.layer,
            "stroke_width": f32(self.stroke_width, field_name=f"{self.id}.stroke_width"),
            "style_tags": list(self.style_tags),
            "contours": [contour.to_dict() for contour in self.contours],
        }

    def geometry_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "contours": [contour.to_dict() for contour in self.contours],
        }

    def control_point_count(self) -> int:
        return sum(len(seg.points) for contour in self.contours for seg in contour.segments)


@dataclass
class Style:
    line_color: Color = (18, 18, 20)
    paper_color: Color = (247, 244, 236)
    line_opacity: float = 1.0
    width_scale: float = 1.0
    sketch_ink: Color = (72, 70, 68)
    sketch_hatch_spacing: float = 7.0
    sketch_hatch_angle: float = 42.0
    sketch_shade_strength: float = 0.24
    toon_palette: dict[str, Color] = field(
        default_factory=lambda: {
            "skin": (244, 193, 153),
            "leaf": (92, 160, 96),
            "accent": (224, 113, 84),
            "neutral": (205, 199, 185),
        }
    )
    toon_default_color: Color = (205, 199, 185)
    toon_shadow_color: Color = (112, 91, 104)
    toon_light: tuple[float, float] = (0.45, -0.8)
    toon_light_strength: float = 0.22

    def to_dict(self) -> dict[str, Any]:
        return {
            "line_color": list(self.line_color),
            "paper_color": list(self.paper_color),
            "line_opacity": float(self.line_opacity),
            "width_scale": float(self.width_scale),
            "sketch_ink": list(self.sketch_ink),
            "sketch_hatch_spacing": float(self.sketch_hatch_spacing),
            "sketch_hatch_angle": float(self.sketch_hatch_angle),
            "sketch_shade_strength": float(self.sketch_shade_strength),
            "toon_palette": {
                key: list(value) for key, value in sorted(self.toon_palette.items())
            },
            "toon_default_color": list(self.toon_default_color),
            "toon_shadow_color": list(self.toon_shadow_color),
            "toon_light": [float(self.toon_light[0]), float(self.toon_light[1])],
            "toon_light_strength": float(self.toon_light_strength),
        }


@dataclass
class StrokeSample:
    curve_steps: int = 12
    line_steps: int = 1
    width_scale: float = 1.0
    min_segment_px: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "curve_steps": int(self.curve_steps),
            "line_steps": int(self.line_steps),
            "width_scale": float(self.width_scale),
            "min_segment_px": float(self.min_segment_px),
        }

    def fingerprint(self) -> str:
        return canonical_hash(self.to_dict())


@dataclass
class Project:
    project_id: str
    version_stamp: str
    shapes: list[Shape]
    style: Style
    stroke_sample: StrokeSample
    layer_order: list[str]
    schema_version: int = CURRENT_SCHEMA_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "project_id": self.project_id,
            "version_stamp": self.version_stamp,
            "shapes": [shape.to_dict() for shape in self.shapes],
            "style": self.style.to_dict(),
            "stroke_sample": self.stroke_sample.to_dict(),
            "layer_order": list(self.layer_order),
            "metadata": copy.deepcopy(self.metadata),
        }

    def clone(self) -> "Project":
        return project_from_dict(self.to_dict())

    def content_hash(self) -> str:
        return canonical_hash(self.to_dict())

    def shape_by_id(self, shape_id: str) -> Shape:
        for shape in self.shapes:
            if shape.id == shape_id:
                return shape
        raise KeyError(shape_id)

    def ordered_shapes(self) -> list[Shape]:
        order = {layer: index for index, layer in enumerate(self.layer_order)}
        return sorted(
            self.shapes,
            key=lambda shape: (order.get(shape.layer, len(order)), shape.id),
        )

    def geometry_dict(self) -> dict[str, Any]:
        return {
            "stroke_sample": self.stroke_sample.to_dict(),
            "shapes": [
                shape.geometry_dict() for shape in self.ordered_shapes()
            ],
        }

    def geometry_hash(self) -> str:
        return canonical_hash(self.geometry_dict())


def canonical_json(data: Any) -> str:
    return json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_hash(data: Any) -> str:
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def bytes_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_segment(raw: Any, *, shape_id: str, contour_index: int) -> Segment:
    require(
        isinstance(raw, Mapping),
        "E_SCHEMA_VERSION",
        "segment must be an object",
        shape=shape_id,
    )
    segment_type = raw.get("type")
    require(
        segment_type in SEGMENT_ARITY,
        "E_SCHEMA_VERSION",
        "unsupported segment type",
        shape=shape_id,
        segment_type=segment_type,
    )
    raw_points = raw.get("points")
    expected = SEGMENT_ARITY[segment_type]
    require(
        isinstance(raw_points, list) and len(raw_points) == expected,
        "E_SCHEMA_VERSION",
        "segment control point count does not match its type",
        shape=shape_id,
        segment_type=segment_type,
        expected=expected,
    )
    points: list[Point] = []
    for point_index, point in enumerate(raw_points):
        require(
            isinstance(point, (list, tuple)) and len(point) == 2,
            "E_SCHEMA_VERSION",
            "control point must be a two-element array",
            shape=shape_id,
            contour=contour_index,
            segment=segment_type,
            point=point_index,
        )
        points.append(
            (
                f32(
                    point[0],
                    field_name=f"{shape_id}.contour[{contour_index}].point[{point_index}].x",
                ),
                f32(
                    point[1],
                    field_name=f"{shape_id}.contour[{contour_index}].point[{point_index}].y",
                ),
            )
        )
    return Segment(segment_type, points)


def _parse_contour(raw: Any, *, shape_id: str, contour_index: int) -> Contour:
    require(
        isinstance(raw, Mapping),
        "E_SCHEMA_VERSION",
        "contour must be an object",
        shape=shape_id,
    )
    raw_segments = raw.get("segments")
    require(
        isinstance(raw_segments, list) and bool(raw_segments),
        "E_SCHEMA_VERSION",
        "contour requires at least one segment",
        shape=shape_id,
        contour=contour_index,
    )
    closed = raw.get("closed")
    require(
        isinstance(closed, bool),
        "E_SCHEMA_VERSION",
        "contour.closed must be boolean",
        shape=shape_id,
        contour=contour_index,
    )
    return Contour(
        closed=closed,
        segments=[
            _parse_segment(segment, shape_id=shape_id, contour_index=contour_index)
            for segment in raw_segments
        ],
    )


def _parse_shape(raw: Any) -> Shape:
    require(
        isinstance(raw, Mapping),
        "E_SCHEMA_VERSION",
        "shape must be an object",
    )
    shape_id = raw.get("id")
    layer = raw.get("layer")
    require(
        isinstance(shape_id, str) and bool(shape_id),
        "E_SCHEMA_VERSION",
        "shape.id must be a non-empty string",
    )
    require(
        isinstance(layer, str) and bool(layer),
        "E_SCHEMA_VERSION",
        "shape.layer must be a non-empty string",
        shape=shape_id,
    )
    stroke_width = f32(raw.get("stroke_width"), field_name=f"{shape_id}.stroke_width")
    require(
        stroke_width > 0.0,
        "E_SCHEMA_VERSION",
        "stroke_width must be positive",
        shape=shape_id,
    )
    tags = raw.get("style_tags", [])
    require(
        isinstance(tags, list) and all(isinstance(tag, str) for tag in tags),
        "E_SCHEMA_VERSION",
        "style_tags must be a string array",
        shape=shape_id,
    )
    raw_contours = raw.get("contours")
    require(
        isinstance(raw_contours, list) and bool(raw_contours),
        "E_SCHEMA_VERSION",
        "shape requires at least one contour",
        shape=shape_id,
    )
    contours = [
        _parse_contour(contour, shape_id=shape_id, contour_index=index)
        for index, contour in enumerate(raw_contours)
    ]
    return Shape(shape_id, layer, stroke_width, list(tags), contours)


def _parse_style(raw: Any) -> Style:
    require(isinstance(raw, Mapping), "E_SCHEMA_VERSION", "style must be an object")
    missing = sorted(set(DEFAULT_STYLE) - set(raw))
    require(
        not missing,
        "E_SCHEMA_VERSION",
        "style is missing required fields",
        missing=missing,
    )
    palette_raw = raw["toon_palette"]
    require(
        isinstance(palette_raw, Mapping),
        "E_SCHEMA_VERSION",
        "toon_palette must be an object",
    )
    palette = {
        str(key): _as_color(value, field_name=f"toon_palette.{key}")
        for key, value in palette_raw.items()
    }
    require(bool(palette), "E_SCHEMA_VERSION", "toon_palette must not be empty")
    light = raw["toon_light"]
    require(
        isinstance(light, (list, tuple)) and len(light) == 2,
        "E_SCHEMA_VERSION",
        "toon_light must have two elements",
    )
    line_opacity = _finite_float(raw["line_opacity"], field_name="line_opacity")
    width_scale = _finite_float(raw["width_scale"], field_name="width_scale")
    hatch_spacing = _finite_float(
        raw["sketch_hatch_spacing"], field_name="sketch_hatch_spacing"
    )
    shade_strength = _finite_float(
        raw["sketch_shade_strength"], field_name="sketch_shade_strength"
    )
    light_strength = _finite_float(
        raw["toon_light_strength"], field_name="toon_light_strength"
    )
    require(0.0 <= line_opacity <= 1.0, "E_SCHEMA_VERSION", "line_opacity out of range")
    require(width_scale > 0.0, "E_SCHEMA_VERSION", "width_scale must be positive")
    require(hatch_spacing > 0.0, "E_SCHEMA_VERSION", "hatch spacing must be positive")
    require(0.0 <= shade_strength <= 1.0, "E_SCHEMA_VERSION", "shade strength out of range")
    require(0.0 <= light_strength <= 1.0, "E_SCHEMA_VERSION", "light strength out of range")
    return Style(
        line_color=_as_color(raw["line_color"], field_name="line_color"),
        paper_color=_as_color(raw["paper_color"], field_name="paper_color"),
        line_opacity=line_opacity,
        width_scale=width_scale,
        sketch_ink=_as_color(raw["sketch_ink"], field_name="sketch_ink"),
        sketch_hatch_spacing=hatch_spacing,
        sketch_hatch_angle=_finite_float(
            raw["sketch_hatch_angle"], field_name="sketch_hatch_angle"
        ),
        sketch_shade_strength=shade_strength,
        toon_palette=palette,
        toon_default_color=_as_color(
            raw["toon_default_color"], field_name="toon_default_color"
        ),
        toon_shadow_color=_as_color(
            raw["toon_shadow_color"], field_name="toon_shadow_color"
        ),
        toon_light=(
            _finite_float(light[0], field_name="toon_light[0]"),
            _finite_float(light[1], field_name="toon_light[1]"),
        ),
        toon_light_strength=light_strength,
    )


def parse_stroke_sample(raw: Any) -> StrokeSample:
    if not isinstance(raw, Mapping):
        raise NprError("E_STROKE_SAMPLE", "stroke_sample must be an object")
    expected = set(DEFAULT_STROKE_SAMPLE)
    missing = sorted(expected - set(raw))
    require(
        not missing,
        "E_STROKE_SAMPLE",
        "stroke_sample is missing fields",
        missing=missing,
    )
    curve_steps = raw["curve_steps"]
    line_steps = raw["line_steps"]
    require(
        isinstance(curve_steps, int) and not isinstance(curve_steps, bool),
        "E_STROKE_SAMPLE",
        "curve_steps must be an integer",
    )
    require(
        isinstance(line_steps, int) and not isinstance(line_steps, bool),
        "E_STROKE_SAMPLE",
        "line_steps must be an integer",
    )
    require(1 <= curve_steps <= 256, "E_STROKE_SAMPLE", "curve_steps out of range")
    require(1 <= line_steps <= 256, "E_STROKE_SAMPLE", "line_steps out of range")
    try:
        width_scale = float(raw["width_scale"])
        min_segment_px = float(raw["min_segment_px"])
    except (TypeError, ValueError) as exc:
        raise NprError("E_STROKE_SAMPLE", "stroke sample values must be numeric") from exc
    require(
        math.isfinite(width_scale) and 0.0 < width_scale <= 100.0,
        "E_STROKE_SAMPLE",
        "width_scale must be finite and in (0, 100]",
    )
    require(
        math.isfinite(min_segment_px) and 0.0 <= min_segment_px <= 64.0,
        "E_STROKE_SAMPLE",
        "min_segment_px must be finite and in [0, 64]",
    )
    return StrokeSample(curve_steps, line_steps, width_scale, min_segment_px)


def project_from_dict(raw: Any) -> Project:
    require(
        isinstance(raw, Mapping),
        "E_SCHEMA_VERSION",
        "project document must be an object",
    )
    schema_version = _as_int(raw.get("schema_version"), field_name="schema_version")
    require(
        schema_version == CURRENT_SCHEMA_VERSION,
        "E_SCHEMA_VERSION",
        "project must be migrated to the current schema before parsing",
        schema_version=schema_version,
        current=CURRENT_SCHEMA_VERSION,
    )
    missing = sorted(
        {"project_id", "version_stamp", "shapes", "style", "stroke_sample", "layer_order"}
        - set(raw)
    )
    require(
        not missing,
        "E_SCHEMA_VERSION",
        "current project is missing required fields",
        missing=missing,
    )
    project_id = raw["project_id"]
    version_stamp = raw["version_stamp"]
    require(
        isinstance(project_id, str) and bool(project_id),
        "E_SCHEMA_VERSION",
        "project_id must be a non-empty string",
    )
    require(
        isinstance(version_stamp, str) and version_stamp.isdigit(),
        "E_SCHEMA_VERSION",
        "version_stamp must be a decimal string",
    )
    require(
        isinstance(raw["shapes"], list) and bool(raw["shapes"]),
        "E_SCHEMA_VERSION",
        "shapes must be a non-empty array",
    )
    shapes = [_parse_shape(shape) for shape in raw["shapes"]]
    shape_ids = [shape.id for shape in shapes]
    require(
        len(shape_ids) == len(set(shape_ids)),
        "E_SCHEMA_VERSION",
        "shape ids must be unique",
    )
    layer_order = raw["layer_order"]
    require(
        isinstance(layer_order, list)
        and bool(layer_order)
        and all(isinstance(layer, str) and layer for layer in layer_order),
        "E_SCHEMA_VERSION",
        "layer_order must be a non-empty string array",
    )
    require(
        len(layer_order) == len(set(layer_order)),
        "E_SCHEMA_VERSION",
        "layer_order contains duplicate layers",
    )
    shape_layers = {shape.layer for shape in shapes}
    require(
        shape_layers.issubset(set(layer_order)),
        "E_SCHEMA_VERSION",
        "layer_order omits a shape layer",
        omitted=sorted(shape_layers - set(layer_order)),
    )
    metadata = raw.get("metadata", {})
    require(
        isinstance(metadata, Mapping),
        "E_SCHEMA_VERSION",
        "metadata must be an object",
    )
    return Project(
        project_id=project_id,
        version_stamp=version_stamp,
        shapes=shapes,
        style=_parse_style(raw["style"]),
        stroke_sample=parse_stroke_sample(raw["stroke_sample"]),
        layer_order=list(layer_order),
        schema_version=schema_version,
        metadata=copy.deepcopy(dict(metadata)),
    )


def project_to_dict(project: Project) -> dict[str, Any]:
    return project.to_dict()


def dump_project(project: Project, *, indent: int | None = 2) -> str:
    """Serialize deterministically; pretty output remains hash-stable."""

    return json.dumps(
        project.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        indent=indent,
        separators=(",", ":") if indent is None else None,
        allow_nan=False,
    ) + ("\n" if indent is not None else "")


def load_project(raw: Any) -> Project:
    """Load a current-schema project or raise E_SCHEMA_VERSION.

    Legacy migration lives in :mod:`npr_editor.migration`, keeping this codec
    strict and preventing accidental silent upgrades.
    """

    return project_from_dict(raw)


def parse_json_project(text: str | bytes) -> Project:
    try:
        raw = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise NprError("E_SCHEMA_VERSION", "invalid project JSON") from exc
    return load_project(raw)


def bump_version_stamp(stamp: str, width: int = 12) -> str:
    if not stamp.isdigit():
        raise NprError("E_SCHEMA_VERSION", "version_stamp must be decimal")
    return str(int(stamp) + 1).zfill(max(width, len(stamp)))


def version_stamp_key(stamp: str) -> int:
    if not isinstance(stamp, str) or not stamp.isdigit():
        raise NprError("E_SNAPSHOT_STALE", "invalid version stamp", stamp=stamp)
    return int(stamp)


def iter_segments(shape: Shape) -> Iterable[tuple[int, int, Segment]]:
    for contour_index, contour in enumerate(shape.contours):
        for segment_index, segment in enumerate(contour.segments):
            yield contour_index, segment_index, segment
