"""Deterministic synthetic fill-scene generation and validation."""

from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable

from .errors import CoverageError, ErrorCode
from .hashing import canonical_json_bytes, pretty_json_text, sha256_json


SCENE_SCHEMA = "vgcover.scene.v1"
CATEGORIES = (
    "simple_convex",
    "self_intersecting",
    "multi_nested",
    "near_degenerate_strip",
)
VALID_FILL_RULES = ("nonzero", "evenodd")
VALID_ORIENTATIONS = ("ccw", "cw")
POINT_EPSILON = 1e-9


def _q(value: float) -> float:
    """Stable decimal representation for generated scene coordinates."""

    rounded = round(float(value), 6)
    return 0.0 if rounded == 0.0 else rounded


def _point(x: float, y: float) -> list[float]:
    return [_q(x), _q(y)]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return _is_number(value) and math.isfinite(float(value))


def _point_equal(a: Any, b: Any, epsilon: float = POINT_EPSILON) -> bool:
    if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)):
        return False
    if len(a) != 2 or len(b) != 2:
        return False
    if not all(_is_finite_number(v) for v in (a[0], a[1], b[0], b[1])):
        return False
    return abs(float(a[0]) - float(b[0])) <= epsilon and abs(float(a[1]) - float(b[1])) <= epsilon


def validate_fill_rule(fill_rule: Any, *, region_id: int | None = None) -> str:
    if fill_rule not in VALID_FILL_RULES:
        raise CoverageError(
            ErrorCode.FILL_RULE,
            f"unsupported fill rule {fill_rule!r}; expected one of {VALID_FILL_RULES}",
            region_id=region_id,
            details={"received": fill_rule, "allowed": list(VALID_FILL_RULES)},
        )
    return str(fill_rule)


def validate_alpha(alpha: Any, *, region_id: int | None = None) -> float:
    if not _is_finite_number(alpha) or not 0.0 <= float(alpha) <= 1.0:
        raise CoverageError(
            ErrorCode.ALPHA_RANGE,
            "alpha must be a finite value in the closed interval [0, 1]",
            region_id=region_id,
            details={"received": alpha, "minimum": 0.0, "maximum": 1.0},
        )
    return float(alpha)


def validate_tolerance(tolerance: Any, *, region_id: int | None = None) -> float:
    if not _is_finite_number(tolerance) or float(tolerance) <= 0.0:
        raise CoverageError(
            ErrorCode.TOLERANCE_RANGE,
            "tolerance must be a finite value greater than zero",
            region_id=region_id,
            details={"received": tolerance},
        )
    return float(tolerance)


def _validate_control_points(
    control_points: Any,
    *,
    expected_count: int,
    region_id: int | None,
    contour_index: int,
    segment_index: int,
) -> list[list[float]]:
    if not isinstance(control_points, (list, tuple)) or len(control_points) != expected_count:
        raise CoverageError(
            ErrorCode.SCENE_INVALID,
            f"segment requires exactly {expected_count} control points",
            region_id=region_id,
            contour_index=contour_index,
            segment_index=segment_index,
            details={"received_count": len(control_points) if hasattr(control_points, "__len__") else None},
        )
    points: list[list[float]] = []
    for point_index, point in enumerate(control_points):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise CoverageError(
                ErrorCode.SCENE_INVALID,
                "control point must be a two-element [x, y] pair",
                region_id=region_id,
                contour_index=contour_index,
                segment_index=segment_index,
                details={"point_index": point_index, "received": point},
            )
        if not all(_is_finite_number(value) for value in point):
            raise CoverageError(
                ErrorCode.SCENE_INVALID,
                "control point coordinates must be finite numbers",
                region_id=region_id,
                contour_index=contour_index,
                segment_index=segment_index,
                details={"point_index": point_index, "received": list(point)},
            )
        points.append([float(point[0]), float(point[1])])
    return points


def validate_region(region: Any) -> dict[str, Any]:
    if not isinstance(region, dict):
        raise CoverageError(ErrorCode.SCENE_INVALID, "region must be an object")
    region_id = region.get("region_id")
    if not isinstance(region_id, int) or isinstance(region_id, bool) or region_id < 0:
        raise CoverageError(
            ErrorCode.SCENE_INVALID,
            "region_id must be a non-negative integer",
            details={"received": region_id},
        )
    validate_fill_rule(region.get("fill_rule"), region_id=region_id)
    validate_tolerance(region.get("tolerance"), region_id=region_id)
    validate_alpha(region.get("alpha"), region_id=region_id)

    color = region.get("fill_color")
    if (
        not isinstance(color, (list, tuple))
        or len(color) != 3
        or not all(isinstance(c, int) and not isinstance(c, bool) and 0 <= c <= 255 for c in color)
    ):
        raise CoverageError(
            ErrorCode.SCENE_INVALID,
            "fill_color must be three integer RGB channels in [0, 255]",
            region_id=region_id,
            details={"received": color},
        )

    contours = region.get("contours")
    if not isinstance(contours, list) or not contours:
        raise CoverageError(
            ErrorCode.SCENE_INVALID,
            "region must contain at least one contour",
            region_id=region_id,
        )

    seen_contour_indices: set[int] = set()
    for list_index, contour in enumerate(contours):
        if not isinstance(contour, dict):
            raise CoverageError(
                ErrorCode.SCENE_INVALID,
                "contour must be an object",
                region_id=region_id,
                contour_index=list_index,
            )
        contour_index = contour.get("contour_index", list_index)
        if (
            not isinstance(contour_index, int)
            or isinstance(contour_index, bool)
            or contour_index < 0
        ):
            raise CoverageError(
                ErrorCode.SCENE_INVALID,
                "contour_index must be a non-negative integer",
                region_id=region_id,
                contour_index=list_index,
                details={"received": contour_index},
            )
        if contour_index in seen_contour_indices:
            raise CoverageError(
                ErrorCode.SCENE_INVALID,
                "contour_index values must be unique inside a region",
                region_id=region_id,
                contour_index=contour_index,
            )
        seen_contour_indices.add(contour_index)
        if contour.get("orientation") not in VALID_ORIENTATIONS:
            raise CoverageError(
                ErrorCode.SCENE_INVALID,
                f"orientation must be one of {VALID_ORIENTATIONS}",
                region_id=region_id,
                contour_index=contour_index,
                details={"received": contour.get("orientation")},
            )
        if contour.get("closed") is not True:
            raise CoverageError(
                ErrorCode.CONTOUR_UNCLOSED,
                "contour.closed must be true",
                region_id=region_id,
                contour_index=contour_index,
                segment_index=0,
            )
        segments = contour.get("segments")
        if not isinstance(segments, list) or not segments:
            raise CoverageError(
                ErrorCode.SCENE_INVALID,
                "contour must contain at least one segment",
                region_id=region_id,
                contour_index=contour_index,
            )

        previous_end: list[float] | None = None
        contour_start: list[float] | None = None
        for segment_list_index, segment in enumerate(segments):
            if not isinstance(segment, dict):
                raise CoverageError(
                    ErrorCode.SCENE_INVALID,
                    "segment must be an object",
                    region_id=region_id,
                    contour_index=contour_index,
                    segment_index=segment_list_index,
                )
            segment_index = segment.get("segment_index", segment_list_index)
            if (
                not isinstance(segment_index, int)
                or isinstance(segment_index, bool)
                or segment_index < 0
            ):
                raise CoverageError(
                    ErrorCode.SCENE_INVALID,
                    "segment_index must be a non-negative integer",
                    region_id=region_id,
                    contour_index=contour_index,
                    segment_index=segment_list_index,
                    details={"received": segment_index},
                )
            segment_type = segment.get("type")
            if segment_type == "line":
                expected_count = 2
            elif segment_type == "cubic":
                expected_count = 4
            else:
                raise CoverageError(
                    ErrorCode.SCENE_INVALID,
                    "segment type must be 'line' or 'cubic'",
                    region_id=region_id,
                    contour_index=contour_index,
                    segment_index=segment_index,
                    details={"received": segment_type},
                )
            points = _validate_control_points(
                segment.get("control_points"),
                expected_count=expected_count,
                region_id=region_id,
                contour_index=contour_index,
                segment_index=segment_index,
            )
            if previous_end is not None and not _point_equal(previous_end, points[0]):
                raise CoverageError(
                    ErrorCode.CONTOUR_UNCLOSED,
                    "segment start does not match the previous segment end",
                    region_id=region_id,
                    contour_index=contour_index,
                    segment_index=segment_index,
                    details={"expected_start": previous_end, "received_start": points[0]},
                )
            if contour_start is None:
                contour_start = points[0]
            previous_end = points[-1]

        if contour_start is None or previous_end is None or not _point_equal(previous_end, contour_start):
            raise CoverageError(
                ErrorCode.CONTOUR_UNCLOSED,
                "contour end does not close back to its start",
                region_id=region_id,
                contour_index=contour_index,
                segment_index=len(segments) - 1,
                details={"start": contour_start, "end": previous_end},
            )

    return region


def validate_scene(scene: Any) -> dict[str, Any]:
    if not isinstance(scene, dict):
        raise CoverageError(ErrorCode.SCENE_INVALID, "scene must be an object")
    width = scene.get("width")
    height = scene.get("height")
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        raise CoverageError(
            ErrorCode.SCENE_INVALID,
            "scene.width must be a positive integer",
            details={"received": width},
        )
    if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
        raise CoverageError(
            ErrorCode.SCENE_INVALID,
            "scene.height must be a positive integer",
            details={"received": height},
        )
    regions = scene.get("regions")
    if not isinstance(regions, list) or not regions:
        raise CoverageError(
            ErrorCode.SCENE_INVALID,
            "scene.regions must be a non-empty list",
        )
    seen_ids: set[int] = set()
    for region in regions:
        validate_region(region)
        region_id = int(region["region_id"])
        if region_id in seen_ids:
            raise CoverageError(
                ErrorCode.SCENE_INVALID,
                "region_id values must be unique",
                region_id=region_id,
            )
        seen_ids.add(region_id)
    return scene


def clone_scene(scene: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(scene)


def scene_by_id(scene: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(region["region_id"]): region for region in scene["regions"]}


def scene_hash(scene: dict[str, Any]) -> str:
    return sha256_json(scene)


def scene_content_hash(scene: dict[str, Any]) -> str:
    """Alias with an explicit business name for the canonical scene hash."""

    return scene_hash(scene)


def scene_json_bytes(scene: dict[str, Any]) -> bytes:
    return canonical_json_bytes(scene)


def scene_json_text(scene: dict[str, Any], *, pretty: bool = False) -> str:
    if pretty:
        return pretty_json_text(scene)
    return scene_json_bytes(scene).decode("utf-8")


def load_scene(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        scene = json.load(stream)
    validate_scene(scene)
    return scene


def save_scene(path: str | Path, scene: dict[str, Any], *, pretty: bool = True) -> str:
    validate_scene(scene)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(scene_json_text(scene, pretty=pretty), encoding="utf-8", newline="\n")
    return scene_hash(scene)


def _segment(segment_type: str, points: Iterable[list[float]], segment_index: int) -> dict[str, Any]:
    return {
        "segment_index": segment_index,
        "type": segment_type,
        "control_points": [list(point) for point in points],
    }


def _ellipse_contour(
    *,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    orientation: str,
    contour_index: int,
) -> dict[str, Any]:
    """Build a closed four-cubic ellipse.

    The orientation tag follows the signed shoelace convention in the scene's
    declared coordinate system (positive signed area is tagged ``ccw``).
    """

    if orientation not in VALID_ORIENTATIONS:
        raise ValueError(f"invalid orientation: {orientation}")
    delta = math.pi / 2.0 if orientation == "ccw" else -math.pi / 2.0
    start = 0.0
    segments: list[dict[str, Any]] = []
    for quarter in range(4):
        a0 = start + quarter * delta
        a1 = a0 + delta
        tangent_scale = (4.0 / 3.0) * math.tan(delta / 4.0)
        p0 = _point(cx + rx * math.cos(a0), cy + ry * math.sin(a0))
        p3 = _point(cx + rx * math.cos(a1), cy + ry * math.sin(a1))
        p1 = _point(
            p0[0] + tangent_scale * (-rx * math.sin(a0)),
            p0[1] + tangent_scale * (ry * math.cos(a0)),
        )
        p2 = _point(
            p3[0] - tangent_scale * (-rx * math.sin(a1)),
            p3[1] - tangent_scale * (ry * math.cos(a1)),
        )
        segments.append(_segment("cubic", [p0, p1, p2, p3], quarter))
    return {
        "contour_index": contour_index,
        "orientation": orientation,
        "closed": True,
        "segments": segments,
    }


def _signed_area_from_contour(contour: dict[str, Any]) -> float:
    points = [segment["control_points"][0] for segment in contour["segments"]]
    total = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        total += float(point[0]) * float(next_point[1]) - float(next_point[0]) * float(point[1])
    return 0.5 * total


def _star_contour(cx: float, cy: float, radius: float, contour_index: int) -> dict[str, Any]:
    points = []
    for step in range(5):
        angle = -math.pi / 2.0 + step * (4.0 * math.pi / 5.0)
        points.append(_point(cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    segments = []
    for index in range(5):
        segments.append(_segment("line", [points[index], points[(index + 1) % 5]], index))
    contour = {
        "contour_index": contour_index,
        "orientation": "ccw",
        "closed": True,
        "segments": segments,
    }
    contour["orientation"] = "ccw" if _signed_area_from_contour(contour) > 0.0 else "cw"
    return contour


def _rotated_strip_contour(
    *,
    cx: float,
    cy: float,
    length: float,
    width: float,
    angle_radians: float,
    contour_index: int,
) -> dict[str, Any]:
    ux, uy = math.cos(angle_radians), math.sin(angle_radians)
    nx, ny = -uy, ux
    half_l = length / 2.0
    half_w = width / 2.0
    points = [
        _point(cx - ux * half_l - nx * half_w, cy - uy * half_l - ny * half_w),
        _point(cx + ux * half_l - nx * half_w, cy + uy * half_l - ny * half_w),
        _point(cx + ux * half_l + nx * half_w, cy + uy * half_l + ny * half_w),
        _point(cx - ux * half_l + nx * half_w, cy - uy * half_l + ny * half_w),
    ]
    segments = []
    for index in range(4):
        segments.append(_segment("line", [points[index], points[(index + 1) % 4]], index))
    contour = {
        "contour_index": contour_index,
        "orientation": "ccw" if _signed_area_from_contour(
            {"segments": segments}
        ) > 0.0 else "cw",
        "closed": True,
        "segments": segments,
    }
    return contour


def _jitter_color(rng: random.Random, color: tuple[int, int, int], amount: int = 5) -> list[int]:
    return [
        max(0, min(255, int(channel + rng.randint(-amount, amount))))
        for channel in color
    ]


def generate_scene(
    seed: int = 20261119,
    width: int = 192,
    height: int = 128,
) -> dict[str, Any]:
    """Generate the four required fill categories from a fixed seed.

    The generator intentionally uses a local ``random.Random`` instance and
    rounds all stored coordinates. Two calls with the same arguments therefore
    produce byte-identical canonical JSON.
    """

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise CoverageError(ErrorCode.SCENE_INVALID, "seed must be an integer")
    if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
        raise CoverageError(
            ErrorCode.SCENE_INVALID,
            "generator width must be a positive integer",
            details={"width": width, "height": height},
        )
    if not isinstance(height, int) or isinstance(height, bool) or height <= 0:
        raise CoverageError(
            ErrorCode.SCENE_INVALID,
            "generator height must be a positive integer",
            details={"width": width, "height": height},
        )
    rng = random.Random(seed)

    simple_cx = width * 0.25 + rng.uniform(-2.0, 2.0)
    simple_cy = height * 0.30 + rng.uniform(-2.0, 2.0)
    simple_rx = min(width, height) * 0.20 + rng.uniform(-1.0, 1.5)
    simple_ry = min(width, height) * 0.19 + rng.uniform(-1.0, 1.5)

    star_cx = width * 0.73 + rng.uniform(-2.0, 2.0)
    star_cy = height * 0.30 + rng.uniform(-2.0, 2.0)
    star_radius = min(width, height) * 0.24 + rng.uniform(-1.0, 1.5)

    nest_cx = width * 0.26 + rng.uniform(-1.5, 1.5)
    nest_cy = height * 0.73 + rng.uniform(-1.5, 1.5)
    nest_radius = min(width, height) * 0.18 + rng.uniform(-0.5, 1.0)

    strip_cx = width * 0.78 + rng.uniform(-1.0, 1.0)
    strip_cy = height * 0.73 + rng.uniform(-1.0, 1.0)
    strip_length = height * 0.43 + rng.uniform(-2.0, 2.0)
    strip_width = 0.22

    colors = [
        _jitter_color(rng, (220, 74, 58)),
        _jitter_color(rng, (46, 136, 214)),
        _jitter_color(rng, (74, 180, 102)),
        _jitter_color(rng, (226, 164, 42)),
    ]

    simple_contour = _ellipse_contour(
        cx=simple_cx,
        cy=simple_cy,
        rx=simple_rx,
        ry=simple_ry,
        orientation="ccw",
        contour_index=0,
    )
    star_contour = _star_contour(star_cx, star_cy, star_radius, 0)
    nested_contours = [
        _ellipse_contour(
            cx=nest_cx,
            cy=nest_cy,
            rx=nest_radius,
            ry=nest_radius * 0.88,
            orientation="ccw",
            contour_index=0,
        ),
        _ellipse_contour(
            cx=nest_cx,
            cy=nest_cy,
            rx=nest_radius * 0.64,
            ry=nest_radius * 0.56,
            orientation="cw",
            contour_index=1,
        ),
        _ellipse_contour(
            cx=nest_cx,
            cy=nest_cy,
            rx=nest_radius * 0.34,
            ry=nest_radius * 0.30,
            orientation="ccw",
            contour_index=2,
        ),
    ]
    strip_contour = _rotated_strip_contour(
        cx=strip_cx,
        cy=strip_cy,
        length=strip_length,
        width=strip_width,
        angle_radians=math.radians(12.0 + rng.uniform(-1.0, 1.0)),
        contour_index=0,
    )

    def normal_region(
        region_id: int,
        category: str,
        contours: list[dict[str, Any]],
        fill_rule: str,
        tolerance: float,
        color: list[int],
        alpha: float,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        region: dict[str, Any] = {
            "region_id": region_id,
            "category": category,
            "fill_rule": fill_rule,
            "tolerance": _q(tolerance),
            "fill_color": color,
            "alpha": _q(alpha),
            "contours": contours,
            "degenerate": {
                "flagged": False,
                "reason": None,
                "minimum_feature_width_px": None,
                "threshold_px": 0.5,
                "retained": True,
            },
        }
        if extra:
            region.update(extra)
        return region

    regions = [
        normal_region(
            10,
            "simple_convex",
            [simple_contour],
            "nonzero",
            0.18,
            colors[0],
            0.78,
        ),
        normal_region(
            20,
            "self_intersecting",
            [star_contour],
            "evenodd",
            0.22,
            colors[1],
            0.72,
            extra={"self_intersecting": True},
        ),
        normal_region(
            30,
            "multi_nested",
            nested_contours,
            "nonzero",
            0.15,
            colors[2],
            0.68,
            extra={"nested_contour_count": len(nested_contours)},
        ),
        normal_region(
            40,
            "near_degenerate_strip",
            [strip_contour],
            "evenodd",
            0.08,
            colors[3],
            0.90,
            extra={
                "degenerate": {
                    "flagged": True,
                    "reason": "minimum_feature_width_below_threshold",
                    "minimum_feature_width_px": strip_width,
                    "threshold_px": 0.5,
                    "retained": True,
                }
            },
        ),
    ]

    scene: dict[str, Any] = {
        "schema": SCENE_SCHEMA,
        "generator": {
            "name": "vgcover.deterministic_synthetic_fill",
            "seed": seed,
            "width": width,
            "height": height,
            "categories": list(CATEGORIES),
        },
        "width": width,
        "height": height,
        "coordinate_system": {
            "origin": "top_left",
            "x_axis": "right",
            "y_axis": "down",
            "orientation_tag": "signed_shoelace_positive_is_ccw",
            "pixel_sampling": "pixel_center_horizontal_ray",
        },
        "regions": regions,
    }
    validate_scene(scene)
    return scene


def generate_category_scenes(
    seed: int = 20261119,
    width: int = 192,
    height: int = 128,
) -> dict[str, dict[str, Any]]:
    """Return one deterministic single-category scene per required category."""

    full_scene = generate_scene(seed=seed, width=width, height=height)
    result: dict[str, dict[str, Any]] = {}
    for category in CATEGORIES:
        scene = copy.deepcopy(full_scene)
        scene["generator"]["categories"] = [category]
        scene["generator"]["category_scene"] = category
        scene["regions"] = [r for r in scene["regions"] if r.get("category") == category]
        validate_scene(scene)
        result[category] = scene
    return result
