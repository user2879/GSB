"""异常注入机制。

注入只通过输入数据与参数生效（绝不修改源码）：
- body_overrides：按 body_id 覆盖透明体参数（ior/radius/absorption/thickness）；
- pixel_overrides：按 (row,col) 覆盖该像素的主光线方向或递归深度预算；
- render_overrides：覆盖渲染级参数（max_depth、sample_count）。
注入配置本身经严格 schema 校验，非法配置归类为 E_INJECT_CONFIG。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .scene import Scene, TransparentBody, dumps_canonical
from .errors import InjectConfigError

BODY_FIELDS = {"ior", "radius", "absorption", "thickness", "axis", "center"}
PIXEL_FIELDS = {"direction", "max_depth", "origin"}
RENDER_FIELDS = {"max_depth", "sample_count"}


@dataclass(frozen=True)
class InjectionConfig:
    name: str
    body_overrides: dict[str, dict[str, Any]]
    pixel_overrides: list[dict[str, Any]]
    render_overrides: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "body_overrides": self.body_overrides,
            "pixel_overrides": self.pixel_overrides,
            "render_overrides": self.render_overrides,
        }

    def to_json(self) -> str:
        return dumps_canonical(self.to_dict())


def load_injection(path: str) -> InjectionConfig:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return injection_from_dict(data)


def injection_from_dict(data: dict[str, Any]) -> InjectionConfig:
    if not isinstance(data, dict):
        raise InjectConfigError("injection config root must be an object", {"received": type(data).__name__})
    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise InjectConfigError("injection config requires a non-empty string 'name'", {"name": name})

    body_overrides = data.get("body_overrides", {})
    pixel_overrides = data.get("pixel_overrides", [])
    render_overrides = data.get("render_overrides", {})

    if not isinstance(body_overrides, dict):
        raise InjectConfigError("body_overrides must be an object keyed by body_id", {})
    for body_id, patch in body_overrides.items():
        if not isinstance(body_id, str) or not isinstance(patch, dict) or not patch:
            raise InjectConfigError("each body override must be a non-empty object", {"body_id": body_id})
        unknown = set(patch) - BODY_FIELDS
        if unknown:
            raise InjectConfigError("unknown body override field",
                                    {"body_id": body_id, "unknown": sorted(unknown)})

    if not isinstance(pixel_overrides, list):
        raise InjectConfigError("pixel_overrides must be a list", {})
    seen_pixels: set[tuple[int, int]] = set()
    for entry in pixel_overrides:
        if not isinstance(entry, dict):
            raise InjectConfigError("each pixel override must be an object", {})
        row, col = entry.get("row"), entry.get("col")
        if not isinstance(row, int) or not isinstance(col, int) or row < 0 or col < 0:
            raise InjectConfigError("pixel override requires non-negative integer row/col",
                                    {"row": row, "col": col})
        if (row, col) in seen_pixels:
            raise InjectConfigError("duplicate pixel override", {"row": row, "col": col})
        seen_pixels.add((row, col))
        patch = entry.get("patch", {})
        if not isinstance(patch, dict) or not patch:
            raise InjectConfigError("pixel override requires non-empty patch", {"row": row, "col": col})
        unknown = set(patch) - PIXEL_FIELDS
        if unknown:
            raise InjectConfigError("unknown pixel override field", {"unknown": sorted(unknown)})
        if "direction" in patch:
            direction = patch["direction"]
            if not (isinstance(direction, list) and len(direction) == 3
                    and all(isinstance(v, (int, float)) for v in direction)):
                raise InjectConfigError("pixel direction must be a length-3 number list",
                                        {"row": row, "col": col})

    if not isinstance(render_overrides, dict):
        raise InjectConfigError("render_overrides must be an object", {})
    unknown = set(render_overrides) - RENDER_FIELDS
    if unknown:
        raise InjectConfigError("unknown render override field", {"unknown": sorted(unknown)})

    return InjectionConfig(name, dict(body_overrides), list(pixel_overrides), dict(render_overrides))


def apply_injection(scene: Scene, injection: InjectionConfig) -> Scene:
    """把注入覆盖应用到场景副本，返回新 Scene；原始场景不被修改。"""
    bodies = list(scene.bodies)
    known_ids = {body.body_id for body in bodies}
    for body_id, patch in injection.body_overrides.items():
        if body_id not in known_ids:
            raise InjectConfigError("body override targets unknown body_id", {"body_id": body_id})
        for index, body in enumerate(bodies):
            if body.body_id != body_id:
                continue
            merged = {
                "body_id": body.body_id, "kind": body.kind, "ior": body.ior,
                "absorption": body.absorption, "center": body.center, "radius": body.radius,
                "axis": body.axis, "thickness": body.thickness, "half_extent": body.half_extent,
            }
            for key, value in patch.items():
                if key == "absorption" and isinstance(value, list):
                    merged[key] = tuple(value)
                elif key == "center" and isinstance(value, list):
                    merged[key] = tuple(value)
                else:
                    merged[key] = value
            bodies[index] = TransparentBody(**merged)

    max_depth = int(injection.render_overrides.get("max_depth", scene.max_depth))
    sample_count = int(injection.render_overrides.get("sample_count", scene.sample_count))
    return Scene(
        format_version=scene.format_version,
        seed=scene.seed,
        camera=scene.camera,
        light=scene.light,
        background=scene.background,
        bodies=tuple(bodies),
        sample_count=sample_count,
        max_depth=max_depth,
        generator_tag=scene.generator_tag,
    )


def pixel_patch_lookup(injection: InjectionConfig) -> dict[tuple[int, int], dict[str, Any]]:
    return {(int(entry["row"]), int(entry["col"])): entry["patch"] for entry in injection.pixel_overrides}
