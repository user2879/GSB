"""Immutable glyph outline model with canonical JSON and validation."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from .errors import invalid_outline

SUPPORTED_SIZES = (12, 16, 24, 32, 48)
MAX_ABS_COORDINATE = 1_000_000.0
_SEGMENT_TYPES = ("line", "quadratic", "cubic")


def _number(
    value: Any, *, glyph_id: str | None, segment_index: int | None, name: str
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise invalid_outline(
            f"{name} must be numeric",
            glyph_id=glyph_id,
            segment_index=segment_index,
        )
    result = float(value)
    if not math.isfinite(result) or abs(result) > MAX_ABS_COORDINATE:
        raise invalid_outline(
            f"{name} must be finite and within +/-{MAX_ABS_COORDINATE:g}",
            glyph_id=glyph_id,
            segment_index=segment_index,
        )
    return 0.0 if result == 0.0 else result


@dataclass(frozen=True, order=True)
class Point:
    x: float
    y: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.x, bool)
            or not isinstance(self.x, (int, float))
            or isinstance(self.y, bool)
            or not isinstance(self.y, (int, float))
        ):
            raise invalid_outline("point coordinates must be numeric")
        object.__setattr__(self, "x", float(self.x))
        object.__setattr__(self, "y", float(self.y))
        if not math.isfinite(self.x) or not math.isfinite(self.y):
            raise invalid_outline("point coordinates must be finite")
        if abs(self.x) > MAX_ABS_COORDINATE or abs(self.y) > MAX_ABS_COORDINATE:
            raise invalid_outline("point coordinates exceed the supported range")

    def to_dict(self) -> dict[str, float]:
        x = 0.0 if self.x == 0.0 else self.x
        y = 0.0 if self.y == 0.0 else self.y
        return {"x": x, "y": y}

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        glyph_id: str | None = None,
        segment_index: int | None = None,
    ) -> Point:
        if not isinstance(value, Mapping):
            raise invalid_outline(
                "point must be an object with x and y",
                glyph_id=glyph_id,
                segment_index=segment_index,
            )
        return cls(
            _number(
                value.get("x"), glyph_id=glyph_id, segment_index=segment_index, name="x"
            ),
            _number(
                value.get("y"), glyph_id=glyph_id, segment_index=segment_index, name="y"
            ),
        )


@dataclass(frozen=True)
class Segment:
    """A segment ending at ``to``.

    The start point is the contour start for segment 0 and the previous
    segment's ``to`` point thereafter.  This implicit chaining makes malformed
    gaps impossible while retaining an explicit segment index in diagnostics.
    """

    kind: str
    to: Point
    control1: Point | None = None
    control2: Point | None = None

    def __post_init__(self) -> None:
        if self.kind not in _SEGMENT_TYPES:
            raise invalid_outline(f"invalid segment type {self.kind!r}")
        if self.kind == "line" and (
            self.control1 is not None or self.control2 is not None
        ):
            raise invalid_outline("line segment must not define control points")
        if self.kind == "quadratic" and (
            self.control1 is None or self.control2 is not None
        ):
            raise invalid_outline("quadratic segment requires exactly control1")
        if self.kind == "cubic" and (self.control1 is None or self.control2 is None):
            raise invalid_outline("cubic segment requires control1 and control2")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"type": self.kind, "to": self.to.to_dict()}
        if self.control1 is not None:
            result["control1"] = self.control1.to_dict()
        if self.control2 is not None:
            result["control2"] = self.control2.to_dict()
        return result

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        glyph_id: str,
        segment_index: int,
    ) -> Segment:
        if not isinstance(value, Mapping):
            raise invalid_outline(
                "segment must be an object",
                glyph_id=glyph_id,
                segment_index=segment_index,
            )
        kind = value.get("type", value.get("segment_type"))
        if kind not in _SEGMENT_TYPES:
            raise invalid_outline(
                f"invalid segment type {kind!r}",
                glyph_id=glyph_id,
                segment_index=segment_index,
                allowed_types=list(_SEGMENT_TYPES),
            )
        to = Point.from_mapping(
            value.get("to"), glyph_id=glyph_id, segment_index=segment_index
        )
        control1 = (
            Point.from_mapping(
                value.get("control1"), glyph_id=glyph_id, segment_index=segment_index
            )
            if "control1" in value
            else None
        )
        control2 = (
            Point.from_mapping(
                value.get("control2"), glyph_id=glyph_id, segment_index=segment_index
            )
            if "control2" in value
            else None
        )
        if kind == "line" and (control1 is not None or control2 is not None):
            raise invalid_outline(
                "line segment must not define control points",
                glyph_id=glyph_id,
                segment_index=segment_index,
            )
        if kind == "quadratic" and (control1 is None or control2 is not None):
            raise invalid_outline(
                "quadratic segment requires exactly control1",
                glyph_id=glyph_id,
                segment_index=segment_index,
            )
        if kind == "cubic" and (control1 is None or control2 is None):
            raise invalid_outline(
                "cubic segment requires control1 and control2",
                glyph_id=glyph_id,
                segment_index=segment_index,
            )
        return cls(kind=kind, to=to, control1=control1, control2=control2)

    def with_role(self, role: str, point: Point) -> Segment:
        if role == "to":
            return replace(self, to=point)
        if role == "control1":
            if self.control1 is None:
                raise invalid_outline(f"{self.kind} segment has no control1")
            return replace(self, control1=point)
        if role == "control2":
            if self.control2 is None:
                raise invalid_outline(f"{self.kind} segment has no control2")
            return replace(self, control2=point)
        raise invalid_outline(f"unknown point role {role!r}")


@dataclass(frozen=True)
class Contour:
    start: Point
    segments: tuple[Segment, ...]
    closed: bool = True

    def __post_init__(self) -> None:
        if not self.closed:
            raise invalid_outline("contour must be closed")
        if not self.segments:
            raise invalid_outline("contour must contain at least one segment")

    def to_dict(self) -> dict[str, Any]:
        return {
            "closed": self.closed,
            "start": self.start.to_dict(),
            "segments": [segment.to_dict() for segment in self.segments],
        }

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        glyph_id: str,
        contour_index: int,
    ) -> Contour:
        if not isinstance(value, Mapping):
            raise invalid_outline(
                "contour must be an object",
                glyph_id=glyph_id,
                details={"contour_index": contour_index},
            )
        if value.get("closed") is not True:
            raise invalid_outline(
                "contour must be explicitly closed",
                glyph_id=glyph_id,
                details={"contour_index": contour_index},
            )
        start = Point.from_mapping(value.get("start"), glyph_id=glyph_id)
        raw_segments = value.get("segments")
        if not isinstance(raw_segments, list) or not raw_segments:
            raise invalid_outline(
                "contour segments must be a non-empty array",
                glyph_id=glyph_id,
                details={"contour_index": contour_index},
            )
        segments = tuple(
            Segment.from_mapping(raw, glyph_id=glyph_id, segment_index=index)
            for index, raw in enumerate(raw_segments)
        )
        return cls(start=start, segments=segments, closed=True)


@dataclass(frozen=True)
class Glyph:
    glyph_id: str
    contours: tuple[Contour, ...]
    suggested_sizes: tuple[int, ...] = SUPPORTED_SIZES

    def __post_init__(self) -> None:
        if not isinstance(self.glyph_id, str) or not self.glyph_id.strip():
            raise invalid_outline("glyph id must be a non-empty string")
        if len(self.glyph_id) > 128:
            raise invalid_outline("glyph id is too long", glyph_id=self.glyph_id)
        if not all(
            isinstance(size, int) and not isinstance(size, bool)
            for size in self.suggested_sizes
        ):
            raise invalid_outline(
                "suggested sizes must be integers", glyph_id=self.glyph_id
            )
        invalid_sizes = [
            size for size in self.suggested_sizes if size not in SUPPORTED_SIZES
        ]
        if invalid_sizes:
            raise invalid_outline(
                "unsupported suggested pixel size",
                glyph_id=self.glyph_id,
                details={"sizes": invalid_sizes, "allowed": list(SUPPORTED_SIZES)},
            )
        if len(set(self.suggested_sizes)) != len(self.suggested_sizes):
            raise invalid_outline(
                "suggested sizes must be unique", glyph_id=self.glyph_id
            )

    @property
    def revision_hash_source(self) -> dict[str, Any]:
        return self.to_dict()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.glyph_id,
            "sizes": list(self.suggested_sizes),
            "contours": [contour.to_dict() for contour in self.contours],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Glyph:
        if not isinstance(value, Mapping):
            raise invalid_outline("glyph must be an object")
        glyph_id = value.get("id", value.get("glyph_id"))
        if not isinstance(glyph_id, str) or not glyph_id.strip():
            raise invalid_outline("glyph id must be a non-empty string")
        raw_contours = value.get("contours")
        if not isinstance(raw_contours, list):
            raise invalid_outline("glyph contours must be an array", glyph_id=glyph_id)
        contours = tuple(
            Contour.from_mapping(raw, glyph_id=glyph_id, contour_index=index)
            for index, raw in enumerate(raw_contours)
        )
        raw_sizes = value.get("sizes", list(SUPPORTED_SIZES))
        if not isinstance(raw_sizes, list):
            raise invalid_outline("glyph sizes must be an array", glyph_id=glyph_id)
        return cls(
            glyph_id=glyph_id, contours=contours, suggested_sizes=tuple(raw_sizes)
        )

    def with_point(
        self,
        *,
        contour_index: int,
        segment_index: int,
        role: str,
        point: Point,
    ) -> Glyph:
        if contour_index < 0 or contour_index >= len(self.contours):
            raise invalid_outline(
                "contour index out of range",
                glyph_id=self.glyph_id,
                details={"contour_index": contour_index},
            )
        contour = self.contours[contour_index]
        if role == "start":
            if segment_index != -1:
                raise invalid_outline(
                    "start point requires segment_index=-1",
                    glyph_id=self.glyph_id,
                    segment_index=segment_index,
                )
            new_contour = replace(contour, start=point)
        else:
            if segment_index < 0 or segment_index >= len(contour.segments):
                raise invalid_outline(
                    "segment index out of range",
                    glyph_id=self.glyph_id,
                    segment_index=segment_index,
                )
            segment = contour.segments[segment_index].with_role(role, point)
            segments = (
                contour.segments[:segment_index]
                + (segment,)
                + contour.segments[segment_index + 1 :]
            )
            new_contour = replace(contour, segments=segments)
        contours = (
            self.contours[:contour_index]
            + (new_contour,)
            + self.contours[contour_index + 1 :]
        )
        return replace(self, contours=contours)


@dataclass(frozen=True)
class GlyphDocument:
    glyphs: tuple[Glyph, ...]
    format_version: str = "glyphforge-glyph-v1"

    def __post_init__(self) -> None:
        if self.format_version != "glyphforge-glyph-v1":
            raise invalid_outline("unsupported glyph document format version")
        if not self.glyphs:
            raise invalid_outline("glyph document must contain at least one glyph")
        ids = [glyph.glyph_id for glyph in self.glyphs]
        if len(ids) != len(set(ids)):
            raise invalid_outline("glyph ids must be unique")

    @property
    def by_id(self) -> dict[str, Glyph]:
        return {glyph.glyph_id: glyph for glyph in self.glyphs}

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format_version,
            "glyphs": [
                glyph.to_dict()
                for glyph in sorted(self.glyphs, key=lambda item: item.glyph_id)
            ],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> GlyphDocument:
        if not isinstance(value, Mapping):
            raise invalid_outline("document must be an object")
        format_version = value.get("format", value.get("format_version"))
        if format_version != "glyphforge-glyph-v1":
            raise invalid_outline("unsupported glyph document format")
        raw_glyphs = value.get("glyphs")
        if not isinstance(raw_glyphs, list):
            raise invalid_outline("glyphs must be an array")
        glyphs = tuple(Glyph.from_mapping(raw) for raw in raw_glyphs)
        return cls(glyphs=glyphs, format_version=format_version)


def validate_glyphs(glyphs: Mapping[str, Glyph] | Iterable[Glyph]) -> dict[str, Glyph]:
    if isinstance(glyphs, Mapping):
        items = list(glyphs.items())
        result: dict[str, Glyph] = {}
        for key, glyph in items:
            if not isinstance(key, str) or not isinstance(glyph, Glyph):
                raise invalid_outline(
                    "glyph mapping must contain Glyph values keyed by id"
                )
            if key != glyph.glyph_id:
                raise invalid_outline(
                    "glyph mapping key does not match glyph id",
                    glyph_id=glyph.glyph_id,
                    details={"mapping_key": key},
                )
            result[key] = glyph
        return dict(sorted(result.items()))
    result = {}
    for glyph in glyphs:
        if not isinstance(glyph, Glyph):
            raise invalid_outline("glyph collection must contain Glyph values")
        if glyph.glyph_id in result:
            raise invalid_outline("glyph ids must be unique", glyph_id=glyph.glyph_id)
        result[glyph.glyph_id] = glyph
    return dict(sorted(result.items()))


def dumps_glyphs(glyphs: Mapping[str, Glyph] | Iterable[Glyph] | GlyphDocument) -> str:
    if isinstance(glyphs, GlyphDocument):
        document = glyphs
    else:
        validated = validate_glyphs(glyphs)
        document = GlyphDocument(tuple(validated.values()))
    return (
        json.dumps(
            document.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )


def loads_glyphs(text: str | bytes) -> GlyphDocument:
    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise invalid_outline("glyph document is not valid UTF-8") from exc
    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise invalid_outline(f"invalid glyph JSON: {exc}") from exc
    return GlyphDocument.from_mapping(value)
