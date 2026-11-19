from __future__ import annotations

import copy
import math
from typing import Any

from .canonical import sha256_value
from .errors import BusinessError, ErrorCode, Location
from .rng import SeedAudit

SCENE_SCHEMA = "vgsynth.scene/v1"
ALLOWED_FILL_RULES = ("nonzero", "evenodd")
EPSILON = 1e-9


def _point(x: float, y: float) -> list[float]:
    return [round(float(x), 6), round(float(y), 6)]


def _line(p0: list[float], p1: list[float]) -> dict[str, Any]:
    return {"type": "line", "p0": list(p0), "p1": list(p1)}


def _cubic(
    p0: list[float],
    c1: list[float],
    c2: list[float],
    p1: list[float],
) -> dict[str, Any]:
    return {"type": "cubic", "p0": list(p0), "c1": list(c1), "c2": list(c2), "p1": list(p1)}


def _rounded_rect(cx: float, cy: float, width: float, height: float, radius: float) -> list[dict[str, Any]]:
    x0, x1 = cx - width / 2.0, cx + width / 2.0
    y0, y1 = cy - height / 2.0, cy + height / 2.0
    r = max(0.5, min(radius, width / 2.0, height / 2.0))
    k = 0.5522847498307936 * r
    start = _point(x0 + r, y0)
    segments: list[dict[str, Any]] = []
    segments.append(_line(start, _point(x1 - r, y0)))
    segments.append(
        _cubic(
            _point(x1 - r, y0),
            _point(x1 - r + k, y0),
            _point(x1, y0 + r - k),
            _point(x1, y0 + r),
        )
    )
    segments.append(_line(_point(x1, y0 + r), _point(x1, y1 - r)))
    segments.append(
        _cubic(
            _point(x1, y1 - r),
            _point(x1, y1 - r + k),
            _point(x1 - r + k, y1),
            _point(x1 - r, y1),
        )
    )
    segments.append(_line(_point(x1 - r, y1), _point(x0 + r, y1)))
    segments.append(
        _cubic(
            _point(x0 + r, y1),
            _point(x0 + r - k, y1),
            _point(x0, y1 - r + k),
            _point(x0, y1 - r),
        )
    )
    segments.append(_line(_point(x0, y1 - r), _point(x0, y0 + r)))
    segments.append(
        _cubic(
            _point(x0, y0 + r),
            _point(x0, y0 + r - k),
            _point(x0 + r - k, y0),
            _point(x0 + r, y0),
        )
    )
    return segments


def _ellipse(cx: float, cy: float, rx: float, ry: float) -> list[dict[str, Any]]:
    k = 0.5522847498307936
    p_top = _point(cx, cy - ry)
    p_right = _point(cx + rx, cy)
    p_bottom = _point(cx, cy + ry)
    p_left = _point(cx - rx, cy)
    return [
        _cubic(p_top, _point(cx + k * rx, cy - ry), _point(cx + rx, cy - k * ry), p_right),
        _cubic(p_right, _point(cx + rx, cy + k * ry), _point(cx + k * rx, cy + ry), p_bottom),
        _cubic(p_bottom, _point(cx - k * rx, cy + ry), _point(cx - rx, cy + k * ry), p_left),
        _cubic(p_left, _point(cx - rx, cy - k * ry), _point(cx - k * rx, cy - ry), p_top),
    ]


def _star(cx: float, cy: float, outer: float, inner: float) -> list[dict[str, Any]]:
    points: list[list[float]] = []
    for order in (0, 2, 4, 1, 3):
        angle = -math.pi / 2.0 + order * (2.0 * math.pi / 5.0)
        points.append(_point(cx + outer * math.cos(angle), cy + outer * math.sin(angle)))
    segments = [_line(points[index], points[(index + 1) % len(points)]) for index in range(len(points))]
    return segments


def _path(
    path_id: str,
    subpaths: list[dict[str, Any]],
    fill_rule: str,
    fill_color: list[int],
    stroke_color: list[int],
    stroke_width: float,
) -> dict[str, Any]:
    return {
        "path_id": path_id,
        "subpaths": subpaths,
        "fill_rule": fill_rule,
        "fill_color": {
            "r": int(fill_color[0]),
            "g": int(fill_color[1]),
            "b": int(fill_color[2]),
            "a": int(fill_color[3]),
        },
        "stroke": {
            "color": {
                "r": int(stroke_color[0]),
                "g": int(stroke_color[1]),
                "b": int(stroke_color[2]),
                "a": int(stroke_color[3]),
            },
            "width": round(float(stroke_width), 6),
        },
    }


def _subpath(subpath_id: str, segments: list[dict[str, Any]], closed: bool = True) -> dict[str, Any]:
    return {"subpath_id": subpath_id, "closed": bool(closed), "segments": segments}


def _jitter(
    audit: SeedAudit,
    scene_id: str,
    label: str,
    purpose: str,
    base: float,
    radius: float,
) -> float:
    return audit.draw_float(f"{scene_id}:{label}", purpose, base - radius, base + radius, digits=6)


def _build_simple_cross(scene_id: str, seed: int) -> tuple[list[dict[str, Any]], SeedAudit]:
    audit = SeedAudit(seed)
    c0x = _jitter(audit, scene_id, "p0.cx", "horizontal path center x", 88.0, 5.0)
    c0y = _jitter(audit, scene_id, "p0.cy", "horizontal path center y", 70.0, 4.0)
    c1x = _jitter(audit, scene_id, "p1.cx", "vertical path center x", 98.0, 5.0)
    c1y = _jitter(audit, scene_id, "p1.cy", "vertical path center y", 72.0, 4.0)
    paths = [
        _path(
            "p0",
            [_subpath("s0", _rounded_rect(c0x, c0y, 126.0, 48.0, 11.0))],
            "nonzero",
            [232, 108, 45, 220],
            [32, 68, 180, 255],
            2.4,
        ),
        _path(
            "p1",
            [_subpath("s0", _rounded_rect(c1x, c1y, 46.0, 112.0, 12.0))],
            "evenodd",
            [48, 168, 216, 188],
            [22, 48, 98, 230],
            2.0,
        ),
    ]
    return paths, audit


def _build_self_intersect(scene_id: str, seed: int) -> tuple[list[dict[str, Any]], SeedAudit]:
    audit = SeedAudit(seed)
    center_x = _jitter(audit, scene_id, "p0.cx", "star center x", 96.0, 4.0)
    center_y = _jitter(audit, scene_id, "p0.cy", "star center y", 72.0, 3.0)
    rotation = audit.draw_int(f"{scene_id}:p0.rotation", "star rotation selector", 5)
    outer = _jitter(audit, scene_id, "p0.outer", "star outer radius", 52.0, 3.0)
    inner = _jitter(audit, scene_id, "p0.inner", "star inner radius", 23.0, 2.0)
    star = _star(center_x, center_y, outer, inner)
    if rotation:
        angle = rotation * (2.0 * math.pi / 5.0) / 5.0
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        for segment in star:
            for key in ("p0", "p1"):
                x, y = segment[key]
                dx, dy = x - center_x, y - center_y
                segment[key] = _point(center_x + dx * cos_a - dy * sin_a, center_y + dx * sin_a + dy * cos_a)
    paths = [
        _path(
            "p0",
            [_subpath("s0", star)],
            "evenodd",
            [206, 42, 120, 205],
            [72, 16, 72, 255],
            2.6,
        ),
        _path(
            "p1",
            [_subpath("s0", _ellipse(96.0, 72.0, 16.0, 16.0))],
            "nonzero",
            [44, 206, 172, 150],
            [20, 80, 72, 210],
            1.5,
        ),
    ]
    return paths, audit


def _build_overlapping_large(scene_id: str, seed: int) -> tuple[list[dict[str, Any]], SeedAudit]:
    audit = SeedAudit(seed)
    jittered = []
    for index, base in enumerate(((66.0, 58.0), (112.0, 66.0), (88.0, 96.0))):
        x = _jitter(audit, scene_id, f"p{index}.cx", f"ellipse {index} center x", base[0], 4.0)
        y = _jitter(audit, scene_id, f"p{index}.cy", f"ellipse {index} center y", base[1], 4.0)
        rx = _jitter(audit, scene_id, f"p{index}.rx", f"ellipse {index} x radius", 52.0, 4.0)
        ry = _jitter(audit, scene_id, f"p{index}.ry", f"ellipse {index} y radius", 38.0, 3.0)
        jittered.append((x, y, rx, ry))
    colors = (
        ([212, 52, 52, 130], [120, 12, 16, 235], 2.2),
        ([48, 180, 72, 120], [12, 96, 34, 235], 2.2),
        ([48, 86, 216, 110], [18, 34, 118, 235], 2.2),
    )
    paths = []
    for index, ((x, y, rx, ry), (fill, stroke, width)) in enumerate(zip(jittered, colors)):
        paths.append(
            _path(
                f"p{index}",
                [_subpath("s0", _ellipse(x, y, rx, ry))],
                "nonzero" if index != 1 else "evenodd",
                fill,
                stroke,
                width,
            )
        )
    return paths, audit


def _build_degenerate(scene_id: str, seed: int) -> tuple[list[dict[str, Any]], SeedAudit]:
    audit = SeedAudit(seed)
    cx = _jitter(audit, scene_id, "p0.cx", "main rounded rect center x", 92.0, 4.0)
    cy = _jitter(audit, scene_id, "p0.cy", "main rounded rect center y", 68.0, 3.0)
    normal = _rounded_rect(cx, cy, 104.0, 74.0, 14.0)
    zero = _point(152.0, 112.0)
    zero_triangle = [
        _line(zero, _point(171.0, 124.0)),
        _line(_point(171.0, 124.0), _point(151.0, 127.0)),
        _line(_point(151.0, 127.0), zero),
        _line(zero, zero),
    ]
    collinear = [
        _cubic(_point(24.0, 122.0), _point(48.0, 122.0), _point(67.0, 122.0), _point(91.0, 122.0)),
        _line(_point(91.0, 122.0), _point(91.0, 126.0)),
        _line(_point(91.0, 126.0), _point(24.0, 126.0)),
        _line(_point(24.0, 126.0), _point(24.0, 122.0)),
    ]
    paths = [
        _path(
            "p0",
            [
                _subpath("s0_normal", normal),
                _subpath("s1_zero", zero_triangle),
                _subpath("s2_collinear", collinear),
            ],
            "nonzero",
            [240, 170, 35, 190],
            [102, 46, 18, 255],
            2.3,
        ),
        _path(
            "p1",
            [_subpath("s0", _ellipse(52.0, 50.0, 24.0, 17.0))],
            "evenodd",
            [56, 176, 166, 135],
            [18, 84, 82, 230],
            1.8,
        ),
    ]
    return paths, audit


def scene_payload_hash(scene: dict[str, Any]) -> str:
    payload = copy.deepcopy(scene)
    payload.pop("content_hash", None)
    return sha256_value(payload)


def finalize_scene(scene: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(scene)
    result.pop("content_hash", None)
    result["content_hash"] = scene_payload_hash(result)
    return result


def generate_scene(kind: str, seed: int, scene_id: str | None = None) -> dict[str, Any]:
    scene_id = scene_id or f"scene_{kind}_{seed:016x}"
    builders = {
        "simple_cross": _build_simple_cross,
        "self_intersect_loop": _build_self_intersect,
        "overlapping_large": _build_overlapping_large,
        "degenerate_segments": _build_degenerate,
    }
    if kind not in builders:
        raise ValueError(f"unknown scene kind: {kind}")
    paths, audit = builders[kind](scene_id, seed)
    scene = {
        "schema": SCENE_SCHEMA,
        "generator_version": "1.0.0",
        "scene_id": scene_id,
        "kind": kind,
        "seed": int(seed),
        "canvas": {"width": 192, "height": 144},
        "paths": paths,
        "seed_audit": audit.manifest(),
    }
    return finalize_scene(scene)


def layer_id_for_index(index: int) -> str:
    return f"L{index:04d}"


def fill_rule_used(path: dict[str, Any]) -> str:
    rule = path.get("fill_rule")
    return rule if rule in ALLOWED_FILL_RULES else "nonzero"


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _segment_points(segment: dict[str, Any]) -> list[tuple[str, list[float]]]:
    keys = ("p0", "c1", "c2", "p1") if segment.get("type") == "cubic" else ("p0", "p1")
    result = []
    for key in keys:
        value = segment.get(key)
        if isinstance(value, list) and len(value) == 2:
            result.append((key, value))
    return result


def _is_degenerate_segment(segment: dict[str, Any]) -> tuple[bool, str]:
    points = [value for _, value in _segment_points(segment)]
    if len(points) < 2:
        return True, "missing-points"
    if all(abs(points[0][0] - point[0]) <= EPSILON and abs(points[0][1] - point[1]) <= EPSILON for point in points[1:]):
        return True, "zero-length"
    if segment.get("type") == "cubic":
        p0, c1, c2, p1 = points
        vx, vy = p1[0] - p0[0], p1[1] - p0[1]
        norm = math.hypot(vx, vy)
        collinear = True
        for point in (c1, c2):
            cross = abs(vx * (point[1] - p0[1]) - vy * (point[0] - p0[0]))
            if cross > EPSILON * max(1.0, norm):
                collinear = False
                break
        if collinear:
            return True, "collinear-controls"
    return False, ""


def validate_tolerance(tolerance: float) -> list[BusinessError]:
    if not _numeric(tolerance) or float(tolerance) <= 0.0:
        return [
            BusinessError(
                ErrorCode.TOLERANCE_RANGE,
                "扁平化容差必须为有限正数",
                Location(field="render_params.tolerance"),
                details={"value": tolerance},
            )
        ]
    return []


def validate_scene(scene: dict[str, Any], *, check_content_hash: bool = True) -> list[BusinessError]:
    issues: list[BusinessError] = []
    canvas = scene.get("canvas", {})
    width = canvas.get("width")
    height = canvas.get("height")
    if not _numeric(width) or not _numeric(height) or float(width) <= 0 or float(height) <= 0:
        issues.append(BusinessError(ErrorCode.SCENE_INVALID, "画布尺寸无效", Location(field="canvas")))
        width, height = 192, 144
    width, height = float(width), float(height)

    if check_content_hash and scene.get("content_hash") and scene_payload_hash(scene) != scene.get("content_hash"):
        issues.append(BusinessError(ErrorCode.SCENE_INVALID, "场景内容哈希不匹配", Location(field="content_hash")))

    paths = scene.get("paths")
    if not isinstance(paths, list):
        issues.append(BusinessError(ErrorCode.SCENE_INVALID, "paths 必须为数组", Location(field="paths")))
        return issues

    seen_paths: set[str] = set()
    for path_index, path in enumerate(paths):
        layer_id = layer_id_for_index(path_index)
        path_id = path.get("path_id", f"invalid-{path_index}")
        if path_id in seen_paths:
            issues.append(
                BusinessError(
                    ErrorCode.SCENE_INVALID,
                    "路径 ID 重复",
                    Location(layer_id=layer_id, path_id=path_id, field="path_id"),
                )
            )
        seen_paths.add(path_id)
        rule = path.get("fill_rule")
        if rule not in ALLOWED_FILL_RULES:
            issues.append(
                BusinessError(
                    ErrorCode.FILL_RULE,
                    "填充规则必须为 nonzero 或 evenodd",
                    Location(layer_id=layer_id, path_id=path_id, field="fill_rule"),
                    recovery="fallback:nonzero",
                    details={"value": rule},
                )
            )
        for color_name in ("fill_color", "stroke"):
            color = path.get("fill_color") if color_name == "fill_color" else path.get("stroke", {}).get("color")
            if not isinstance(color, dict):
                issues.append(
                    BusinessError(
                        ErrorCode.SCENE_INVALID,
                        "颜色对象缺失",
                        Location(layer_id=layer_id, path_id=path_id, field=color_name),
                    )
                )
                continue
            for channel in ("r", "g", "b", "a"):
                value = color.get(channel)
                if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 255:
                    issues.append(
                        BusinessError(
                            ErrorCode.COLOR_RANGE,
                            "颜色分量必须为 0..255 整数",
                            Location(layer_id=layer_id, path_id=path_id, field=f"{color_name}.{channel}"),
                            recovery="clamp:0..255",
                            details={"value": value},
                        )
                    )
        stroke = path.get("stroke", {})
        stroke_width = stroke.get("width")
        if not _numeric(stroke_width) or float(stroke_width) <= 0.0:
            issues.append(
                BusinessError(
                    ErrorCode.PATH_DEGENERATE,
                    "描边宽度必须为正数，当前按零描边处理",
                    Location(layer_id=layer_id, path_id=path_id, field="stroke.width"),
                    recovery="skip:stroke",
                    details={"value": stroke_width},
                )
            )
        subpaths = path.get("subpaths")
        if not isinstance(subpaths, list) or not subpaths:
            issues.append(
                BusinessError(
                    ErrorCode.SCENE_INVALID,
                    "路径至少需要一个子路径",
                    Location(layer_id=layer_id, path_id=path_id, field="subpaths"),
                )
            )
            continue
        for subpath_index, subpath in enumerate(subpaths):
            segments = subpath.get("segments")
            if not isinstance(segments, list) or not segments:
                issues.append(
                    BusinessError(
                        ErrorCode.SCENE_INVALID,
                        "子路径至少需要一个段",
                        Location(
                            layer_id=layer_id,
                            path_id=path_id,
                            subpath_index=subpath_index,
                            field="segments",
                        ),
                    )
                )
                continue
            for segment_index, segment in enumerate(segments):
                if segment.get("type") not in ("line", "cubic"):
                    issues.append(
                        BusinessError(
                            ErrorCode.SCENE_INVALID,
                            "段类型必须为 line 或 cubic",
                            Location(
                                layer_id=layer_id,
                                path_id=path_id,
                                subpath_index=subpath_index,
                                segment_index=segment_index,
                                field="type",
                            ),
                        )
                    )
                    continue
                for point_name, point in _segment_points(segment):
                    if len(point) != 2 or not all(_numeric(component) for component in point):
                        issues.append(
                            BusinessError(
                                ErrorCode.SCENE_INVALID,
                                "控制点必须为有限二维坐标",
                                Location(
                                    layer_id=layer_id,
                                    path_id=path_id,
                                    subpath_index=subpath_index,
                                    segment_index=segment_index,
                                    field=point_name,
                                ),
                            )
                        )
                        continue
                    for axis, value, upper in zip(("x", "y"), point, (width, height)):
                        if float(value) < 0.0 or float(value) > upper:
                            issues.append(
                                BusinessError(
                                    ErrorCode.SCENE_INVALID,
                                    "坐标超出画布声明边界",
                                    Location(
                                        layer_id=layer_id,
                                        path_id=path_id,
                                        subpath_index=subpath_index,
                                        segment_index=segment_index,
                                        field=f"{point_name}.{axis}",
                                    ),
                                    recovery="clamp:canvas",
                                    details={"value": value, "min": 0.0, "max": upper},
                                )
                            )
                degenerate, reason = _is_degenerate_segment(segment)
                if degenerate:
                    issues.append(
                        BusinessError(
                            ErrorCode.PATH_DEGENERATE,
                            "段退化，按声明规则降级或跳过",
                            Location(
                                layer_id=layer_id,
                                path_id=path_id,
                                subpath_index=subpath_index,
                                segment_index=segment_index,
                                field="geometry",
                            ),
                            recovery="degenerate-policy",
                            details={"reason": reason, "segment_type": segment.get("type")},
                        )
                    )
    return issues


def normalized_scene_for_render(scene: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(scene)
    width = float(result.get("canvas", {}).get("width", 192))
    height = float(result.get("canvas", {}).get("height", 144))
    for path in result.get("paths", []):
        if path.get("fill_rule") not in ALLOWED_FILL_RULES:
            path["fill_rule"] = "nonzero"
        for color_name in ("fill_color", "stroke"):
            color = path.get("fill_color") if color_name == "fill_color" else path.get("stroke", {}).get("color")
            if not isinstance(color, dict):
                continue
            for channel in ("r", "g", "b", "a"):
                value = color.get(channel, 0)
                if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                    value = 0
                color[channel] = max(0, min(255, int(round(float(value)))))
        stroke = path.get("stroke", {})
        if not _numeric(stroke.get("width")) or float(stroke.get("width")) <= 0.0:
            stroke["width"] = 0.0
        for subpath in path.get("subpaths", []):
            for segment in subpath.get("segments", []):
                for _, point in _segment_points(segment):
                    if len(point) != 2:
                        continue
                    point[0] = round(max(0.0, min(width, float(point[0]))), 6)
                    point[1] = round(max(0.0, min(height, float(point[1]))), 6)
    return result


def segment_fragment(scene: dict[str, Any], path_id: str | None, subpath_index: int | None, segment_index: int | None) -> dict[str, Any] | None:
    if path_id is None:
        return None
    for path in scene.get("paths", []):
        if path.get("path_id") != path_id:
            continue
        for s_index, subpath in enumerate(path.get("subpaths", [])):
            if subpath_index is not None and s_index != subpath_index:
                continue
            for seg_index, segment in enumerate(subpath.get("segments", [])):
                if segment_index is not None and seg_index != segment_index:
                    continue
                return {
                    "path_id": path_id,
                    "subpath_index": s_index,
                    "segment_index": seg_index,
                    "segment": copy.deepcopy(segment),
                }
    return None
