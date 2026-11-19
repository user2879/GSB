"""逐像素路径报告、首个受影响像素定位与最小反例。

最小反例：在给定场景 + 注入配置下，单独重放该像素可复现同一错误码；
反例字段集按错误码最小化，移除任一必需字段即不再复现该错误码。
"""

from __future__ import annotations

from typing import Any, Optional

from .camera import build_camera_frame
from .errors import RefracError, classify_exception
from .injection import InjectionConfig, apply_injection, pixel_patch_lookup
from .scene import Scene, dumps_canonical
from .tracer import trace_pixel


def report_to_dict(trace) -> dict[str, Any]:
    return {
        "row": trace.row,
        "col": trace.col,
        "rgba": [float(v) for v in trace.color],
        "termination": trace.termination,
        "isolated": trace.isolated,
        "error_code": trace.error_code,
        "error_message": trace.error_message,
        "interface_crossings": trace.interface_crossings,
        "tir_count": trace.tir_count,
        "internal_length": trace.internal_length,
        "events": trace.events,
    }


def serialize_reports(reports) -> str:
    payload = [report_to_dict(trace) for trace in reports]
    return dumps_canonical(payload)


def first_affected_pixel(isolated: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """行优先、列升序的确定性遍历顺序下，首个被隔离像素。"""
    if not isolated:
        return None
    return min(isolated, key=lambda record: (record["row"], record["col"]))


# 各错误码的“触发字段”：移除对应注入覆盖后必须不再复现同一错误码。
# 其余字段（row/col/body_id/declared_range 等）为定位与说明字段。
TRIGGER_FIELDS = {
    "E_IOR_RANGE": ("ior",),
    "E_RAY_DEGENERATE": ("direction",),
    "E_DEPTH_BOUNDS": ("max_depth",),
    "E_ABSORB_NEGATIVE": ("absorption",),
    "E_INTERSECT_DEGENERATE": ("radius", "thickness"),
    "E_PIXEL_ISOLATED": (),
    "E_INJECT_CONFIG": (),
}


def _counterexample_fields(code: str, record: dict[str, Any], scene: Scene,
                           injection: Optional[InjectionConfig]) -> dict[str, Any]:
    """按错误码给出最小输入片段（触发字段 + 定位/说明字段）。"""
    row, col = record["row"], record["col"]
    ctx = record.get("context", {}) or {}
    base = {"row": row, "col": col}
    if code == "E_IOR_RANGE":
        body_id = ctx.get("body_id")
        body = next(item for item in scene.bodies if item.body_id == body_id)
        base.update({"body_id": body_id, "ior": float(body.ior),
                     "declared_range": [1.0, 3.0]})
    elif code == "E_RAY_DEGENERATE":
        patches = pixel_patch_lookup(injection) if injection else {}
        direction = patches.get((row, col), {}).get("direction")
        base.update({"direction": list(direction) if direction is not None else [0.0, 0.0, 0.0]})
    elif code == "E_DEPTH_BOUNDS":
        patches = pixel_patch_lookup(injection) if injection else {}
        patch = patches.get((row, col), {})
        base.update({"max_depth": int(patch.get("max_depth", scene.max_depth))})
        if "interface_crossings" in ctx:
            base["interface_crossings"] = int(ctx["interface_crossings"])
    elif code == "E_ABSORB_NEGATIVE":
        body_id = ctx.get("body_id")
        body = next(item for item in scene.bodies if item.body_id == body_id)
        base.update({"body_id": body_id, "absorption": [float(v) for v in body.absorption]})
    elif code == "E_INTERSECT_DEGENERATE":
        body_id = ctx.get("body_id")
        body = next(item for item in scene.bodies if item.body_id == body_id)
        base.update({"body_id": body_id})
        if body.kind == "sphere":
            base["radius"] = float(body.radius)
        else:
            base["thickness"] = float(body.thickness)
    else:
        base.update({"origin_code": record.get("origin_code", code)})
    return base


def build_minimal_counterexample(scene: Scene, injection: Optional[InjectionConfig],
                                 isolated: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    first = first_affected_pixel(isolated)
    if first is None:
        return None
    code = first["code"]
    fields = _counterexample_fields(code, first, scene, injection)
    return {
        "code": code,
        "pixel": [first["row"], first["col"]],
        "injection_name": injection.name if injection else None,
        "required_fields": sorted(fields.keys()),
        "trigger_fields": list(TRIGGER_FIELDS.get(code, ())),
        "fragment": fields,
    }


def replay_pixel(scene: Scene, injection: Optional[InjectionConfig],
                 row: int, col: int) -> dict[str, Any]:
    """用场景 + 注入配置单独重放一个像素，返回成功或归类后的错误码。"""
    effective = apply_injection(scene, injection) if injection is not None else scene
    frame = build_camera_frame(effective.camera)
    patches = pixel_patch_lookup(injection) if injection is not None else {}
    try:
        trace = trace_pixel(effective, frame, row, col, patches.get((row, col)))
        return {"ok": True, "termination": trace.termination, "code": None}
    except RefracError as exc:
        return {"ok": False, "termination": "isolated", "code": classify_exception(exc),
                "message": str(exc), "context": exc.context}


def _sanitize_injection(injection: InjectionConfig, code: str, field_name: str,
                        row: int, col: int) -> Optional[InjectionConfig]:
    """构造移除/合法化反例字段后的注入配置；无任何覆盖时返回 None。"""
    body_overrides = {key: dict(value) for key, value in injection.body_overrides.items()}
    pixel_entries = [dict(entry) for entry in injection.pixel_overrides]
    render_overrides = dict(injection.render_overrides)

    if field_name in ("ior", "radius", "absorption", "thickness"):
        for patch in body_overrides.values():
            patch.pop(field_name, None)
    elif field_name == "direction":
        for entry in pixel_entries:
            if entry["row"] == row and entry["col"] == col:
                patch = dict(entry["patch"])
                patch.pop("direction", None)
                entry["patch"] = patch
    elif field_name == "max_depth":
        render_overrides.pop("max_depth", None)
        for entry in pixel_entries:
            if entry["row"] == row and entry["col"] == col and "max_depth" in entry["patch"]:
                patch = dict(entry["patch"])
                patch.pop("max_depth", None)
                entry["patch"] = patch

    body_overrides = {key: value for key, value in body_overrides.items() if value}
    pixel_entries = [entry for entry in pixel_entries if entry.get("patch")]
    if not body_overrides and not pixel_entries and not render_overrides:
        return None
    return InjectionConfig(
        name=injection.name + f"#sanitized:{field_name}",
        body_overrides=body_overrides,
        pixel_overrides=pixel_entries,
        render_overrides=render_overrides,
    )


def verify_counterexample_minimality(scene: Scene, injection: Optional[InjectionConfig],
                                     counterexample: dict[str, Any]) -> dict[str, Any]:
    """验证最小反例：

    - 完整片段单独重放可复现同一错误码；
    - 移除任一触发字段（回退注入）或改变像素坐标后，均不再复现该错误码。
    row/col 为定位字段，body_id/declared_range 等为说明字段，不参与移除探针。
    """
    row, col = counterexample["pixel"]
    expected = counterexample["code"]
    replay_full = replay_pixel(scene, injection, row, col)
    field_results: dict[str, dict[str, Any]] = {}

    fragment = counterexample.get("fragment", {})
    probe_fields = [name for name in counterexample.get("trigger_fields", ()) if name in fragment]
    for field_name in probe_fields:
        if injection is None:
            probe = replay_pixel(scene, None, row, col)
        else:
            sanitized = _sanitize_injection(injection, expected, field_name, row, col)
            probe = replay_pixel(scene, sanitized, row, col)
        field_results[field_name] = {
            "reproduces_same_code": bool(probe["code"] == expected),
            "observed_code": probe["code"],
            "probe": "trigger_field_removed",
        }

    # 坐标定位探针：
    # - 像素级注入：换相邻像素后错误不再复现；
    # - 全局体/渲染级注入：同坐标移除触发字段后错误不再复现。
    pixel_scoped = bool(injection and injection.pixel_overrides)
    if pixel_scoped:
        probe_col = 0 if col != 0 else 1
        coordinate_probe = replay_pixel(scene, injection, row, probe_col)
        field_results["coordinate"] = {
            "reproduces_same_code": bool(coordinate_probe["code"] == expected),
            "observed_code": coordinate_probe["code"],
            "probe_pixel": [row, probe_col],
        }
    elif probe_fields and injection is not None:
        sanitized = _sanitize_injection(injection, expected, probe_fields[0], row, col)
        coordinate_probe = replay_pixel(scene, sanitized, row, col)
    else:
        coordinate_probe = replay_pixel(scene, None, row, col)
        field_results["coordinate"] = {
            "reproduces_same_code": bool(coordinate_probe["code"] == expected),
            "observed_code": coordinate_probe["code"],
            "probe": "same_pixel_after_trigger_removed",
        }

    return {
        "expected_code": expected,
        "full_replay_code": replay_full["code"],
        "full_replay_matches": replay_full["code"] == expected,
        "field_removal": field_results,
        "minimal": replay_full["code"] == expected
        and all(not item["reproduces_same_code"] for item in field_results.values()),
    }
