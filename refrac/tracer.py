"""光线追踪器：折射合成、深度控制、吸收衰减与逐像素路径报告。

渲染流程（逐像素，行优先、列升序）：
1. 针孔相机生成主光线（可被 pixel_overrides 注入）；
2. 与最近透明面求交，按斯涅尔定律折射或全反射；
3. 介质内部线段按声明采样数做比尔—朗伯衰减；
4. 深度上限、完全吸收、射向背景三类终止原因分别记录；
5. 单像素异常被捕获为隔离记录，不终止整帧。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .absorption import integrate_transmittance, is_fully_absorbed
from .camera import build_camera_frame, primary_ray
from .errors import DepthBoundsError, RefracError, classify_exception
from .geometry import Ray, make_ray, nearest_transparent_hit, intersect_background
from .injection import InjectionConfig, apply_injection, pixel_patch_lookup
from .optics import refract_or_reflect, validate_ior
from .scene import Scene
from .vectors import RAY_EPS, normalize

AIR_IOR = 1.0
ISOLATED_COLOR = np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float32)


@dataclass
class PixelTrace:
    row: int
    col: int
    color: np.ndarray
    events: list[dict[str, Any]] = field(default_factory=list)
    interface_crossings: int = 0
    tir_count: int = 0
    internal_length: float = 0.0
    termination: str = "background"
    isolated: bool = False
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    error_context: Optional[dict[str, Any]] = None


def _validate_render_params(scene: Scene) -> None:
    if not isinstance(scene.max_depth, int) or scene.max_depth < 1:
        raise DepthBoundsError(
            "render max_depth must be a positive integer",
            {"max_depth": scene.max_depth},
        )
    if not isinstance(scene.sample_count, int) or scene.sample_count < 1:
        raise DepthBoundsError(
            "render sample_count must be a positive integer",
            {"sample_count": scene.sample_count},
        )


def _background_shading(scene: Scene, point: np.ndarray) -> np.ndarray:
    light_dir = -normalize(np.asarray(scene.light.direction, dtype=np.float64))
    normal = normalize(np.asarray(scene.background.normal, dtype=np.float64))
    shade = max(0.0, float(np.dot(light_dir, normal)))
    checker = int(math.floor(point[0] / scene.background.checker_size)) + int(
        math.floor(point[1] / scene.background.checker_size)
    )
    base = np.asarray(scene.background.base_color, dtype=np.float64)
    alt = np.asarray(scene.background.checker_color, dtype=np.float64)
    color = base if checker % 2 == 0 else alt
    intensity = np.float64(scene.light.intensity) * (0.25 + 0.75 * shade)
    return np.clip(color * intensity, 0.0, 1.0)


def trace_pixel(scene: Scene, frame, row: int, col: int,
                pixel_patch: Optional[dict[str, Any]] = None) -> PixelTrace:
    """追踪单个像素；异常向上抛出，由 render_frame 统一隔离。"""
    _validate_render_params(scene)
    base_ray = primary_ray(frame, scene.camera, row, col)
    if pixel_patch is not None and "direction" in pixel_patch:
        base_ray = make_ray(
            pixel_patch.get("origin", base_ray.origin),
            pixel_patch["direction"],
        )
    # 先构造主光线（光线退化优先归类 E_RAY_DEGENERATE），再处理深度预算。
    depth_budget = int(scene.max_depth)
    per_pixel_depth: int | None = None
    if pixel_patch is not None and "max_depth" in pixel_patch:
        per_pixel_depth = int(pixel_patch["max_depth"])

    bodies_by_id = {body.body_id: body for body in scene.bodies}
    ray = base_ray
    throughput = np.ones(3, dtype=np.float64)
    stack: list[str] = []
    trace = PixelTrace(row=row, col=col, color=np.zeros(4, dtype=np.float32))
    # 防御性迭代上限：深度控制以 refraction 次数为准，TIR 另设硬上限。
    hard_step_cap = scene.max_depth * 4 + 32
    steps = 0
    depth_budget_checked = False

    while True:
        if not depth_budget_checked:
            if per_pixel_depth is not None and per_pixel_depth < 1:
                raise DepthBoundsError(
                    "per-pixel max_depth override must be a positive integer",
                    {"row": row, "col": col, "max_depth": per_pixel_depth},
                )
            if per_pixel_depth is not None:
                depth_budget = per_pixel_depth
            depth_budget_checked = True

        steps += 1
        if steps > hard_step_cap:  # pragma: no cover - 防御性保护
            raise DepthBoundsError("hard step cap exceeded", {"row": row, "col": col, "steps": steps})

        hit = nearest_transparent_hit(ray, scene.bodies)
        background_hit = intersect_background(ray, scene.background)
        if hit is None or (background_hit is not None and background_hit.t < hit.t):
            # 射向不透明背景：线性空间着色后乘以累计透过率。
            if background_hit is None:
                trace.termination = "escaped"
                rgb = np.zeros(3, dtype=np.float64)
            else:
                trace.termination = "background"
                rgb = _background_shading(scene, background_hit.point) * throughput
            trace.color = np.array(
                [np.float32(rgb[0]), np.float32(rgb[1]), np.float32(rgb[2]), np.float32(1.0)],
                dtype=np.float32,
            )
            return trace

        body = bodies_by_id[hit.body_id]
        segment_length = float(hit.t)
        # 介质内部吸收：进入时压栈，离开时对刚走过的内部线段积分。
        if hit.side == "exit" and stack and stack[-1] == hit.body_id:
            trace.internal_length += segment_length
            sampled, analytic, deviation = integrate_transmittance(
                body.absorption, segment_length, scene.sample_count, body_id=hit.body_id
            )
            throughput = throughput * sampled
            trace.events.append({
                "seq": len(trace.events),
                "type": "absorption",
                "body_id": hit.body_id,
                "segment_length": segment_length,
                "sampled_transmittance": [float(v) for v in sampled],
                "analytic_transmittance": [float(v) for v in analytic],
                "absorption_deviation": deviation,
                "sample_count": scene.sample_count,
            })
            if is_fully_absorbed(throughput):
                trace.termination = "fully_absorbed"
                trace.color = np.array(
                    [np.float32(0.0), np.float32(0.0), np.float32(0.0), np.float32(1.0)],
                    dtype=np.float32,
                )
                return trace

        # 深度预算按折射界面穿越计数；耗尽即终止并归类 E_DEPTH_BOUNDS。
        if trace.interface_crossings >= depth_budget:
            raise DepthBoundsError(
                "refraction depth limit reached",
                {
                    "row": row,
                    "col": col,
                    "max_depth": depth_budget,
                    "interface_crossings": trace.interface_crossings,
                    "body_id": hit.body_id,
                },
            )

        # 法线统一朝向入射介质一侧：进入时几何外法线已朝入射侧，
        # 离开时翻转外法线使其指向介质内部（入射侧）。
        incident_normal = hit.normal if hit.side == "enter" else -hit.normal
        ior_incident = AIR_IOR if not stack else bodies_by_id[stack[-1]].ior
        ior_transmitted = AIR_IOR if hit.side == "exit" else body.ior
        validate_ior(ior_incident, body_id=hit.body_id, role="incident")
        validate_ior(ior_transmitted, body_id=hit.body_id, role="transmitted")
        result = refract_or_reflect(ray.direction, incident_normal, ior_incident, ior_transmitted)

        trace.events.append({
            "seq": len(trace.events),
            "type": "interface",
            "body_id": hit.body_id,
            "kind": hit.kind,
            "side": hit.side,
            "t": segment_length,
            "point": [float(v) for v in hit.point],
            "normal": [float(v) for v in incident_normal],
            "ior_incident": float(ior_incident),
            "ior_transmitted": float(ior_transmitted),
            "cos_theta_i": result.cos_theta_i,
            "sin2_theta_t": result.sin2_theta_t,
            "critical_angle": result.critical_angle,
            "classification": result.classification,
        })
        trace.interface_crossings += 1

        if result.classification == "TIR":
            trace.tir_count += 1
            trace.events[-1]["tir_index"] = trace.tir_count
            # 独立 TIR 弹跳预算：防止掠射陷获光线无限弹跳；
            # 超限时以 depth_limit 终止并保留当前累计颜色，光线不被静默丢弃。
            if trace.tir_count > int(scene.max_tir_bounces):
                trace.termination = "depth_limit"
                trace.events[-1]["budget_exceeded"] = "max_tir_bounces"
                rgb = np.clip(throughput * np.ones(3, dtype=np.float64), 0.0, 1.0)
                trace.color = np.array(
                    [np.float32(rgb[0]), np.float32(rgb[1]), np.float32(rgb[2]),
                     np.float32(1.0)], dtype=np.float32,
                )
                return trace
            # 沿反射后光线方向偏移，避免在同一交点自相交形成周期轨道。
            new_origin = hit.point + normalize(result.direction) * RAY_EPS
            ray = Ray(new_origin, normalize(result.direction))
            continue

        if hit.side == "enter":
            stack.append(hit.body_id)
        elif stack and stack[-1] == hit.body_id:
            stack.pop()
        new_origin = hit.point - incident_normal * RAY_EPS
        ray = Ray(new_origin, normalize(result.direction))


def render_frame(scene: Scene, injection: Optional[InjectionConfig] = None) -> dict[str, Any]:
    """渲染整帧，返回 float32 RGBA 缓冲、路径报告与隔离记录。"""
    effective_scene = apply_injection(scene, injection) if injection is not None else scene
    _validate_render_params(effective_scene)
    frame = build_camera_frame(effective_scene.camera)
    width, height = effective_scene.camera.width, effective_scene.camera.height
    pixel_patches = pixel_patch_lookup(injection) if injection is not None else {}

    rgba = np.zeros((height, width, 4), dtype=np.float32)
    reports: list[PixelTrace] = []
    isolated: list[dict[str, Any]] = []
    coverage_hits: dict[str, int] = {body.body_id: 0 for body in effective_scene.bodies}
    termination_counts: dict[str, int] = {}

    for row in range(height):
        for col in range(width):
            patch = pixel_patches.get((row, col))
            try:
                trace = trace_pixel(effective_scene, frame, row, col, patch)
            except RefracError as exc:
                record = exc.to_record()
                record.update({"row": row, "col": col, "code": classify_exception(exc)})
                isolated.append(record)
                trace = PixelTrace(
                    row=row, col=col, color=ISOLATED_COLOR.copy(),
                    termination="isolated", isolated=True,
                    error_code=record["code"], error_message=str(exc),
                    error_context=record.get("context"),
                )
            rgba[row, col, :] = trace.color
            reports.append(trace)
            termination_counts[trace.termination] = termination_counts.get(trace.termination, 0) + 1
            if not trace.isolated:
                touched = {event["body_id"] for event in trace.events if event["type"] == "interface"}
                for body_id in touched:
                    coverage_hits[body_id] += 1

    return {
        "rgba": rgba,
        "reports": reports,
        "isolated": isolated,
        "coverage_hits": coverage_hits,
        "termination_counts": termination_counts,
        "scene": effective_scene,
        "frame": frame,
    }
