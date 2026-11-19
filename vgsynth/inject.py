from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from .errors import INJECTION_ERROR_CODES, BusinessError, ErrorCode, Location
from .scene import ALLOWED_FILL_RULES, layer_id_for_index

INJECTION_SCHEMA = "vgsynth.injections/v1"


@dataclass
class InjectionResult:
    scene: dict[str, Any]
    render_params: dict[str, Any]
    applications: list[dict[str, Any]] = field(default_factory=list)
    issues: list[BusinessError] = field(default_factory=list)
    output_mutations: list[dict[str, Any]] = field(default_factory=list)

    def manifest(self, source_config: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema": INJECTION_SCHEMA,
            "source": source_config,
            "applications": self.applications,
            "issues": [issue.to_dict() for issue in self.issues],
            "output_mutations": self.output_mutations,
        }


def empty_injection_config() -> dict[str, Any]:
    return {"schema": INJECTION_SCHEMA, "operations": []}


def default_render_params() -> dict[str, Any]:
    return {"tolerance": 0.25, "subscanlines": 8}


def _find_path(scene: dict[str, Any], path_id: str) -> tuple[int, dict[str, Any]] | None:
    for index, path in enumerate(scene.get("paths", [])):
        if path.get("path_id") == path_id:
            return index, path
    return None


def _resolve_target(scene: dict[str, Any], target: dict[str, Any], operation_id: str) -> tuple[dict[str, Any], Location] | None:
    path_id = target.get("path_id")
    if path_id is None:
        return None
    found = _find_path(scene, str(path_id))
    if found is None:
        return None
    path_index, path = found
    layer_id = layer_id_for_index(path_index)
    subpath_index = target.get("subpath_index")
    segment_index = target.get("segment_index")
    if subpath_index is None and segment_index is None:
        return path, Location(layer_id=layer_id, path_id=str(path_id))
    if not isinstance(subpath_index, int) or not 0 <= subpath_index < len(path.get("subpaths", [])):
        return None
    subpath = path["subpaths"][subpath_index]
    if segment_index is None:
        return path, Location(layer_id=layer_id, path_id=str(path_id), subpath_index=subpath_index)
    if not isinstance(segment_index, int) or not 0 <= segment_index < len(subpath.get("segments", [])):
        return None
    return path, Location(
        layer_id=layer_id,
        path_id=str(path_id),
        subpath_index=subpath_index,
        segment_index=segment_index,
    )


def _conflict_key(operation: dict[str, Any]) -> tuple | None:
    kind = operation.get("type")
    target = operation.get("target", {}) or {}
    path_id = target.get("path_id")
    if kind == "coordinate_out_of_bounds":
        return (
            path_id,
            target.get("subpath_index", 0),
            target.get("segment_index", 0),
            str(target.get("point", "p0")),
        )
    if kind == "degenerate_segment":
        return (path_id, target.get("subpath_index", 0), target.get("segment_index", 0), "geometry")
    if kind in ("fill_rule_invalid", "fill_alpha_out_of_range", "stroke_width_zero"):
        return (path_id, kind)
    if kind == "stroke_alpha_out_of_range":
        return (path_id, kind)
    if kind == "tolerance_negative":
        return ("render_params", "tolerance")
    if kind in ("layer_pixel_mutation", "layer_hash_mutation"):
        return (target.get("layer_id") or path_id, kind)
    return None


def apply_injections(
    scene: dict[str, Any],
    config: dict[str, Any] | None,
    render_params: dict[str, Any] | None = None,
) -> InjectionResult:
    source_config = copy.deepcopy(config or empty_injection_config())
    params = copy.deepcopy(render_params or default_render_params())
    result = InjectionResult(scene=copy.deepcopy(scene), render_params=params)
    operations = source_config.get("operations", [])
    if not isinstance(operations, list):
        result.issues.append(
            BusinessError(
                ErrorCode.INJECTION_CONFLICT,
                "注入 operations 必须为数组",
                Location(field="operations"),
            )
        )
        return result
    seen_ids: set[str] = set()
    seen_keys: dict[tuple, str] = {}
    for index, operation in enumerate(operations):
        if not isinstance(operation, dict):
            result.issues.append(
                BusinessError(
                    ErrorCode.INJECTION_CONFLICT,
                    "注入操作必须为对象",
                    Location(field="operations"),
                    injection_id=f"inj-{index:04d}",
                )
            )
            continue
        operation_id = str(operation.get("id", f"inj-{index:04d}"))
        kind = operation.get("type")
        target = operation.get("target", {}) or {}
        if not isinstance(target, dict):
            result.issues.append(
                BusinessError(
                    ErrorCode.INJECTION_CONFLICT,
                    "注入 target 必须为对象",
                    Location(field="target"),
                    injection_id=operation_id,
                )
            )
            continue
        value = operation.get("value")
        application: dict[str, Any] = {
            "injection_id": operation_id,
            "type": kind,
            "target": copy.deepcopy(target),
            "value": copy.deepcopy(value),
            "applied": False,
        }
        result.applications.append(application)
        if operation_id in seen_ids:
            result.issues.append(
                BusinessError(
                    ErrorCode.INJECTION_CONFLICT,
                    "注入 ID 重复",
                    Location(field="injection_id"),
                    injection_id=operation_id,
                )
            )
            continue
        seen_ids.add(operation_id)
        if kind not in INJECTION_ERROR_CODES:
            result.issues.append(
                BusinessError(
                    ErrorCode.INJECTION_CONFLICT,
                    "未知注入类型",
                    Location(field="type"),
                    injection_id=operation_id,
                    details={"type": kind},
                )
            )
            continue
        key = _conflict_key(operation)
        if key is not None and key in seen_keys:
            result.issues.append(
                BusinessError(
                    ErrorCode.INJECTION_CONFLICT,
                    "两个注入操作写入同一字段",
                    Location(field="target"),
                    injection_id=operation_id,
                    details={"conflicts_with": seen_keys[key]},
                )
            )
            continue
        if key is not None:
            seen_keys[key] = operation_id

        if kind == "tolerance_negative":
            try:
                result.render_params["tolerance"] = float(value if value is not None else -1.0)
            except (TypeError, ValueError):
                result.render_params["tolerance"] = value
            application["applied"] = True
            application["location"] = {"field": "render_params.tolerance"}
            continue
        if kind in ("layer_pixel_mutation", "layer_hash_mutation"):
            layer_id = target.get("layer_id")
            if layer_id is None and target.get("path_id") is not None:
                found = _find_path(result.scene, str(target["path_id"]))
                if found is not None:
                    layer_id = layer_id_for_index(found[0])
            if layer_id is None:
                result.issues.append(
                    BusinessError(
                        ErrorCode.INJECTION_CONFLICT,
                        "输出变异注入无法解析目标层",
                        Location(field="target.layer_id"),
                        injection_id=operation_id,
                    )
                )
                continue
            normalized_value = copy.deepcopy(value)
            if kind == "layer_pixel_mutation":
                if not isinstance(normalized_value, dict):
                    result.issues.append(
                        BusinessError(
                            ErrorCode.INJECTION_CONFLICT,
                            "像素变异 value 必须为对象",
                            Location(field="value"),
                            injection_id=operation_id,
                        )
                    )
                    continue
                try:
                    normalized_value = {
                        "x": int(normalized_value.get("x", 0)),
                        "y": int(normalized_value.get("y", 0)),
                        "channel": int(normalized_value.get("channel", 0)),
                        "delta": int(normalized_value.get("delta", 1)),
                    }
                except (TypeError, ValueError):
                    result.issues.append(
                        BusinessError(
                            ErrorCode.INJECTION_CONFLICT,
                            "像素变异坐标或通道必须为整数",
                            Location(field="value"),
                            injection_id=operation_id,
                        )
                    )
                    continue
            mutation = {
                "injection_id": operation_id,
                "type": kind,
                "layer_id": str(layer_id),
                "value": normalized_value,
            }
            result.output_mutations.append(mutation)
            application["applied"] = True
            application["location"] = {"layer_id": str(layer_id)}
            continue

        resolved = _resolve_target(result.scene, target, operation_id)
        if resolved is None:
            result.issues.append(
                BusinessError(
                    ErrorCode.INJECTION_CONFLICT,
                    "注入目标无法解析",
                    Location(field="target"),
                    injection_id=operation_id,
                    details={"target": target},
                )
            )
            continue
        path, location = resolved
        if kind == "coordinate_out_of_bounds":
            subpath = path["subpaths"][int(target.get("subpath_index", 0))]
            segment = subpath["segments"][int(target.get("segment_index", 0))]
            point_name = target.get("point", "p0")
            if point_name not in ("p0", "c1", "c2", "p1") or point_name not in segment:
                result.issues.append(
                    BusinessError(
                        ErrorCode.INJECTION_CONFLICT,
                        "坐标注入点名称无效",
                        Location(field="target.point"),
                        injection_id=operation_id,
                    )
                )
                continue
            if isinstance(value, dict):
                point = [value.get("x", -64.0), value.get("y", -64.0)]
            elif isinstance(value, list) and len(value) == 2:
                point = value
            else:
                point = [-64.0, -64.0]
            try:
                segment[point_name] = [float(point[0]), float(point[1])]
            except (TypeError, ValueError, IndexError):
                result.issues.append(
                    BusinessError(
                        ErrorCode.INJECTION_CONFLICT,
                        "坐标注入值必须为二维数值",
                        Location(field="value"),
                        injection_id=operation_id,
                    )
                )
                continue
            application["location"] = location.to_dict() | {"field": point_name}
        elif kind == "degenerate_segment":
            subpath = path["subpaths"][int(target.get("subpath_index", 0))]
            segment = subpath["segments"][int(target.get("segment_index", 0))]
            mode = value if isinstance(value, str) else "zero_length"
            if mode == "collinear" and segment.get("type") == "cubic":
                p0, p1 = segment["p0"], segment["p1"]
                segment["c1"] = [p0[0] + (p1[0] - p0[0]) / 3.0, p0[1] + (p1[1] - p0[1]) / 3.0]
                segment["c2"] = [p0[0] + 2.0 * (p1[0] - p0[0]) / 3.0, p0[1] + 2.0 * (p1[1] - p0[1]) / 3.0]
            else:
                p0 = list(segment["p0"])
                for point_name in ("c1", "c2", "p1"):
                    if point_name in segment:
                        segment[point_name] = list(p0)
            application["location"] = location.to_dict() | {"field": "geometry"}
        elif kind == "fill_rule_invalid":
            path["fill_rule"] = value if value is not None else "winding"
            application["location"] = location.to_dict() | {"field": "fill_rule"}
        elif kind == "fill_alpha_out_of_range":
            path["fill_color"]["a"] = value if value is not None else 300
            application["location"] = location.to_dict() | {"field": "fill_color.a"}
        elif kind == "stroke_alpha_out_of_range":
            path["stroke"]["color"]["a"] = value if value is not None else 300
            application["location"] = location.to_dict() | {"field": "stroke.color.a"}
        elif kind == "stroke_width_zero":
            path["stroke"]["width"] = value if value is not None else 0
            application["location"] = location.to_dict() | {"field": "stroke.width"}
        else:
            result.issues.append(
                BusinessError(
                    ErrorCode.INJECTION_CONFLICT,
                    "注入类型未实现",
                    Location(field="type"),
                    injection_id=operation_id,
                )
            )
            continue
        application["applied"] = True
    return result


def attach_injection_ids(issues: list[BusinessError], applications: list[dict[str, Any]]) -> list[BusinessError]:
    for issue in issues:
        if issue.injection_id is not None:
            continue
        for application in applications:
            if not application.get("applied"):
                continue
            location = application.get("location", {})
            if issue.location.layer_id != location.get("layer_id"):
                continue
            if issue.location.path_id != location.get("path_id"):
                continue
            if location.get("subpath_index") is not None and issue.location.subpath_index != location.get("subpath_index"):
                continue
            if location.get("segment_index") is not None and issue.location.segment_index != location.get("segment_index"):
                continue
            issue.injection_id = application["injection_id"]
            break
    return issues


def validate_fill_rule_value(value: Any) -> bool:
    return value in ALLOWED_FILL_RULES
