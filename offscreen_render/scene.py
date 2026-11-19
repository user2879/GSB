"""场景描述的数据结构、JSON 解析与输入校验（业务要求 1）。

校验失败一律抛出 :class:`~offscreen_render.errors.RenderError`，
由上层保证在任何输出文件落盘之前完成全部校验。
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .errors import (
    E_BOX,
    E_CAMERA,
    E_GRID,
    E_JSON_PARSE,
    E_LIGHT,
    E_PATH,
    E_SCHEMA,
    RenderError,
)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VolumeBox:
    """轴对齐介质盒。min_corner/max_corner 为世界坐标对角点。"""

    index: int
    name: str
    min_corner: np.ndarray
    max_corner: np.ndarray
    sigma_t: float          # 消光系数（吸收+散射外散射），> 0
    albedo: float           # 散射反照率，[0, 1]
    ior: float              # 折射率，>= 1

    @property
    def size(self) -> np.ndarray:
        return self.max_corner - self.min_corner


@dataclass(frozen=True)
class PointLight:
    index: int
    position: np.ndarray
    intensity: float        # 辐射强度，> 0


@dataclass(frozen=True)
class Camera:
    position: np.ndarray
    forward: np.ndarray     # 单位朝向
    fov_v_deg: float        # 垂直视场角，(0, 180)
    aspect: float           # 宽/高，> 0


@dataclass(frozen=True)
class RenderConfig:
    width: int
    height: int
    supersample: int        # 每像素 N x N 条光线
    step_length: float      # 光线采样步长（世界单位），> 0
    max_steps: int          # 单条光线最大步数，正整数
    transmittance_cutoff: float       # 透射率截断阈值，(0, 1)
    fp32_mantissa_bits: int           # 单精度中间量保留的有效尾数位数
    transmittance_tolerance: float    # 双/单精度透射率容差，> 0
    tile_width: int
    tile_height: int
    concurrency: int
    memory_limit_bytes: int
    state_dir: str
    output_dir: str
    step_jitter: tuple[float, ...] = field(default=tuple)
    cutoff_jitter: tuple[float, ...] = field(default=tuple)
    light_radius: float = 1.0          # 点光源距离衰减的参考半径
    background: float = 0.0            # 无介质背景灰度贡献


@dataclass(frozen=True)
class Scene:
    volumes: tuple[VolumeBox, ...]
    lights: tuple[PointLight, ...]
    camera: Camera
    config: RenderConfig
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    source_path: str | None = None


# ---------------------------------------------------------------------------
# 基础取值工具
# ---------------------------------------------------------------------------
def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _require(obj: Any, key: str, ctx: str) -> Any:
    if not isinstance(obj, dict) or key not in obj:
        raise RenderError(E_SCHEMA, f"{ctx} 缺少必填字段 {key!r}", {"missing": key})
    return obj[key]


def _vec3(value: Any, ctx: str) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise RenderError(E_SCHEMA, f"{ctx} 必须是长度为 3 的数组")
    out = np.empty(3, dtype=np.float64)
    for k, comp in enumerate(value):
        if not _is_finite_number(comp):
            raise RenderError(E_SCHEMA, f"{ctx} 的第 {k} 个分量不是有限数")
        out[k] = float(comp)
    return out


# ---------------------------------------------------------------------------
# 各子结构解析
# ---------------------------------------------------------------------------
def _parse_volume(idx: int, raw: Any) -> VolumeBox:
    ctx = f"volumes[{idx}]"
    if not isinstance(raw, dict):
        raise RenderError(E_BOX, f"{ctx} 必须是对象")
    name = str(raw.get("name", f"box_{idx}"))
    lo = _vec3(_require(raw, "min_corner", ctx), f"{ctx}.min_corner")
    hi = _vec3(_require(raw, "max_corner", ctx), f"{ctx}.max_corner")
    if not np.all(hi > lo):
        raise RenderError(E_BOX, f"{ctx} 尺寸必须为有限正数（max_corner 须逐分量大于 min_corner）",
                          {"min": lo.tolist(), "max": hi.tolist()})
    sigma = _require(raw, "sigma_t", ctx)
    if not _is_finite_number(sigma) or float(sigma) <= 0.0:
        raise RenderError(E_BOX, f"{ctx}.sigma_t 必须为有限正数", {"got": sigma})
    albedo = _require(raw, "albedo", ctx)
    if not _is_finite_number(albedo) or not 0.0 <= float(albedo) <= 1.0:
        raise RenderError(E_BOX, f"{ctx}.albedo 必须落在 [0,1]", {"got": albedo})
    ior = _require(raw, "ior", ctx)
    if not _is_finite_number(ior) or float(ior) < 1.0:
        raise RenderError(E_BOX, f"{ctx}.ior 必须为不小于 1 的有限数", {"got": ior})
    return VolumeBox(idx, name, lo, hi, float(sigma), float(albedo), float(ior))


def _parse_light(idx: int, raw: Any, bbox_min: np.ndarray, bbox_max: np.ndarray) -> PointLight:
    ctx = f"lights[{idx}]"
    if not isinstance(raw, dict):
        raise RenderError(E_LIGHT, f"{ctx} 必须是对象")
    pos = _vec3(_require(raw, "position", ctx), f"{ctx}.position")
    if np.any(pos < bbox_min) or np.any(pos > bbox_max):
        raise RenderError(E_LIGHT, f"{ctx}.position 不在场景包围盒内",
                          {"position": pos.tolist(), "bbox_min": bbox_min.tolist(),
                           "bbox_max": bbox_max.tolist()})
    intensity = _require(raw, "intensity", ctx)
    if not _is_finite_number(intensity) or float(intensity) <= 0.0:
        raise RenderError(E_LIGHT, f"{ctx}.intensity 必须为正数", {"got": intensity})
    return PointLight(idx, pos, float(intensity))


def _parse_camera(raw: Any) -> Camera:
    ctx = "camera"
    if not isinstance(raw, dict):
        raise RenderError(E_CAMERA, f"{ctx} 必须是对象")
    pos = _vec3(_require(raw, "position", ctx), f"{ctx}.position")
    fwd = _vec3(_require(raw, "forward", ctx), f"{ctx}.forward")
    norm = float(np.linalg.norm(fwd))
    if norm == 0.0:
        raise RenderError(E_CAMERA, f"{ctx}.forward 不能为零向量")
    fwd = fwd / norm
    fov = _require(raw, "fov_v_deg", ctx)
    if not _is_finite_number(fov) or not 0.0 < float(fov) < 180.0:
        raise RenderError(E_CAMERA, f"{ctx}.fov_v_deg 必须落在 (0,180) 度内", {"got": fov})
    aspect = _require(raw, "aspect", ctx)
    if not _is_finite_number(aspect) or float(aspect) <= 0.0:
        raise RenderError(E_CAMERA, f"{ctx}.aspect 必须为正数", {"got": aspect})
    return Camera(pos, fwd, float(fov), float(aspect))


def _positive_int(value: Any, ctx: str, code: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise RenderError(code, f"{ctx} 必须为正整数", {"got": value})
    return int(value)


def _parse_config(raw: Any, grid_w: int, grid_h: int) -> RenderConfig:
    ctx = "render"
    if not isinstance(raw, dict):
        raise RenderError(E_SCHEMA, f"{ctx} 必须是对象")
    width = _positive_int(_require(raw, "width", ctx), f"{ctx}.width", E_GRID)
    height = _positive_int(_require(raw, "height", ctx), f"{ctx}.height", E_GRID)
    supersample = _positive_int(raw.get("supersample", 1), f"{ctx}.supersample", E_GRID)
    step = _require(raw, "step_length", ctx)
    if not _is_finite_number(step) or float(step) <= 0.0:
        raise RenderError(E_GRID, f"{ctx}.step_length 必须为有限正数", {"got": step})
    max_steps = _positive_int(_require(raw, "max_steps", ctx), f"{ctx}.max_steps", E_GRID)
    cutoff = _require(raw, "transmittance_cutoff", ctx)
    if not _is_finite_number(cutoff) or not 0.0 < float(cutoff) < 1.0:
        raise RenderError(E_GRID, f"{ctx}.transmittance_cutoff 必须落在 (0,1)", {"got": cutoff})
    mantissa = raw.get("fp32_mantissa_bits", 23)
    if not isinstance(mantissa, int) or isinstance(mantissa, bool) or not 1 <= mantissa <= 52:
        raise RenderError(E_GRID, f"{ctx}.fp32_mantissa_bits 必须是 [1,52] 的整数", {"got": mantissa})
    tol = _require(raw, "transmittance_tolerance", ctx)
    if not _is_finite_number(tol) or float(tol) <= 0.0:
        raise RenderError(E_GRID, f"{ctx}.transmittance_tolerance 必须为正数", {"got": tol})
    tile_w = _positive_int(_require(raw, "tile_width", ctx), f"{ctx}.tile_width", E_GRID)
    tile_h = _positive_int(_require(raw, "tile_height", ctx), f"{ctx}.tile_height", E_GRID)
    concurrency = _positive_int(raw.get("concurrency", 1), f"{ctx}.concurrency", E_GRID)
    mem = _positive_int(_require(raw, "memory_limit_bytes", ctx), f"{ctx}.memory_limit_bytes", E_GRID)
    state_dir = str(_require(raw, "state_dir", ctx))
    output_dir = str(_require(raw, "output_dir", ctx))
    for label, path in (("state_dir", state_dir), ("output_dir", output_dir)):
        if not os.path.isdir(path) or not os.access(path, os.W_OK):
            raise RenderError(E_PATH, f"{ctx}.{label} 不是可写目录", {"path": path})

    # 分块必须完整覆盖超采样网格
    ss_w, ss_h = width * supersample, height * supersample
    if tile_w > ss_w or tile_h > ss_h or ss_w % tile_w != 0 or ss_h % tile_h != 0:
        raise RenderError(E_GRID, "分块规格必须完整、无余数地覆盖超采样像素网格",
                          {"grid": [ss_w, ss_h], "tile": [tile_w, tile_h]})

    def _jitter(key: str) -> tuple[float, ...]:
        vals = raw.get(key, [])
        if not isinstance(vals, list):
            raise RenderError(E_GRID, f"{ctx}.{key} 必须是数组")
        out: list[float] = []
        for jv in vals:
            if not _is_finite_number(jv) or float(jv) <= 0:
                raise RenderError(E_GRID, f"{ctx}.{key} 中的值必须为有限正数", {"got": jv})
            out.append(float(jv))
        return tuple(out)

    step_jitter = _jitter("step_jitter")
    cutoff_jitter = _jitter("cutoff_jitter")
    for cv in cutoff_jitter:
        if not 0.0 < cv < 1.0:
            raise RenderError(E_GRID, f"{ctx}.cutoff_jitter 值必须落在 (0,1)", {"got": cv})

    light_radius = float(raw.get("light_radius", 1.0))
    if not _is_finite_number(light_radius) or light_radius <= 0:
        raise RenderError(E_GRID, f"{ctx}.light_radius 必须为正数", {"got": light_radius})
    background = float(raw.get("background", 0.0))
    if not _is_finite_number(background) or not 0.0 <= background <= 1.0:
        raise RenderError(E_GRID, f"{ctx}.background 必须落在 [0,1]", {"got": background})

    return RenderConfig(
        width=width, height=height, supersample=supersample,
        step_length=float(step), max_steps=max_steps,
        transmittance_cutoff=float(cutoff), fp32_mantissa_bits=int(mantissa),
        transmittance_tolerance=float(tol), tile_width=tile_w, tile_height=tile_h,
        concurrency=concurrency, memory_limit_bytes=mem, state_dir=state_dir,
        output_dir=output_dir, step_jitter=step_jitter, cutoff_jitter=cutoff_jitter,
        light_radius=light_radius, background=background,
    )


# ---------------------------------------------------------------------------
# 顶层入口
# ---------------------------------------------------------------------------
def load_scene(path: str) -> Scene:
    if not os.path.isfile(path):
        raise RenderError(E_PATH, "场景描述文件不存在", {"path": path})
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except json.JSONDecodeError as exc:
        raise RenderError(E_JSON_PARSE, f"JSON 解析失败: {exc.msg}",
                          {"line": exc.lineno, "column": exc.colno}) from exc
    if not isinstance(raw, dict):
        raise RenderError(E_SCHEMA, "场景描述顶层必须是对象")

    vols_raw = _require(raw, "volumes", "scene")
    if not isinstance(vols_raw, list) or not vols_raw:
        raise RenderError(E_BOX, "volumes 必须是非空数组")
    volumes = tuple(_parse_volume(i, v) for i, v in enumerate(vols_raw))

    bbox_min = np.min(np.stack([v.min_corner for v in volumes]), axis=0)
    bbox_max = np.max(np.stack([v.max_corner for v in volumes]), axis=0)

    lights_raw = _require(raw, "lights", "scene")
    if not isinstance(lights_raw, list) or not lights_raw:
        raise RenderError(E_LIGHT, "lights 必须是非空数组")
    lights = tuple(_parse_light(i, v, bbox_min, bbox_max) for i, v in enumerate(lights_raw))

    camera = _parse_camera(_require(raw, "camera", "scene"))
    cfg = _parse_config(_require(raw, "render", "scene"), 0, 0)
    return Scene(volumes=volumes, lights=lights, camera=camera, config=cfg,
                 bbox_min=bbox_min, bbox_max=bbox_max, source_path=os.path.abspath(path))
