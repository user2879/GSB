"""体渲染核心：指数透射率、散射合成与双/单精度误差对照（业务要求 3、4）。

渲染模型（确定性、无随机数）
---------------------------
对每条主光线建立多介质有序区间链，按固定步长把光线切成段，
段的光学厚度由 *段中点* 所属介质的消光系数决定（中点归属，规避端点歧义）：

* 透射率：``T <- T * exp(-sigma_t * ds)``，逐段累乘；
* 散射：在每个采样点（段中点）对每盏灯做阴影可见性（同样按指数衰减求
  光源透射率），并以当前剩余透射率加权累加入射散射；
* 终止：透射率低于截断阈值或达到最大步数；终止原因逐光线记录。

同一条光线并行维护两条累乘路径：float64 参考路径与每步尾数量化的
单精度路径；最终灰度只取 float64，单精度路径仅用于误差评估。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera import build_frame, generate_ray, subpixel_uv
from .geometry import build_interval_chain, interval_at
from .precision import quantize_mantissa
from .scene import PointLight, RenderConfig, Scene, VolumeBox

# 每条光线在途采样缓冲的保守记账常量（字节）：
# 区间链临时数组 + 两条精度路径 + 散射/误差记录，按最大步数线性估算。
PER_RAY_BUFFER_BYTES = 512
PER_STEP_BUFFER_BYTES = 64


@dataclass
class RayResult:
    radiance: float                  # float64 路径得到的线性灰度（未量化）
    steps: int                       # 实际步进数
    stop_reason: str                 # completed | cutoff | max_steps
    max_abs_diff: float              # 双/单精度透射率最大绝对差
    first_exceed: dict | None        # 首次超容差点记录
    transmittance64: float           # 末态 float64 透射率


@dataclass
class TileResult:
    tile_index: int
    x0: int
    y0: int
    width: int                       # 超采样网格坐标下的块宽/高
    height: int
    grayscale: np.ndarray            # shape (height, width) float64，超采样灰度
    diff_map: np.ndarray             # shape (height, width) float64
    steps_map: np.ndarray            # shape (height, width) int32
    reason_map: np.ndarray           # shape (height, width) "<U16"
    first_exceeds: list[dict]        # 块内首次超容差记录（限量保留）
    cutoff_count: int
    max_steps_count: int
    content_digest: str = ""

    def memory_bytes(self) -> int:
        n = self.width * self.height
        arrays = self.grayscale.nbytes + self.diff_map.nbytes + \
            self.steps_map.nbytes + self.reason_map.nbytes
        return arrays + n * PER_RAY_BUFFER_BYTES


def _segment_sigma(chain_arr: np.ndarray, t_mid: float) -> tuple[float, int]:
    """按段中点归属返回 (消光系数, 介质盒序号)；不在任何介质内返回 (0,-1)。"""
    t0 = chain_arr["t0"]
    t1 = chain_arr["t1"]
    inside = (t_mid >= t0) & (t_mid < t1)
    if not np.any(inside):
        return 0.0, -1
    idx = int(np.argmax(inside))  # 链已按声明归属排序，命中即唯一
    return float(chain_arr[idx]["sigma"]), int(chain_arr[idx]["box_index"])


def shadow_transmittance(origin: np.ndarray, direction: np.ndarray,
                         length: float, volumes: tuple[VolumeBox, ...]) -> float:
    """阴影光线在给定长度内的指数透射率（解析积分，分段不重不漏）。"""
    chain = build_interval_chain(origin, direction, volumes)
    tau = 0.0
    for seg in chain:
        lo = max(seg.t_enter, 0.0)
        hi = min(seg.t_exit, length)
        if hi > lo:
            tau += seg.volume.sigma_t * (hi - lo)
    return float(np.exp(-tau))


def _light_contribution(point: np.ndarray, lights: tuple[PointLight, ...],
                        volumes: tuple[VolumeBox, ...], cfg: RenderConfig) -> float:
    total = 0.0
    for light in lights:
        delta = light.position - point
        dist = float(np.linalg.norm(delta))
        if dist <= 1.0e-12:
            irradiance = light.intensity
            t_shadow = 1.0
        else:
            direction = delta / dist
            t_shadow = shadow_transmittance(point, direction, dist, volumes)
            # 平方反比衰减，参考半径 light_radius 避免近奇异。
            irradiance = light.intensity / (1.0 + (dist / cfg.light_radius) ** 2)
        total += irradiance * t_shadow
    return total


def render_ray(origin: np.ndarray, direction: np.ndarray, scene: Scene) -> RayResult:
    cfg = scene.config
    chain = build_interval_chain(origin, direction, scene.volumes)
    if chain:
        chain_arr = np.array(
            [(s.t_enter, s.t_exit, s.volume.sigma_t, s.volume.index) for s in chain],
            dtype=np.dtype([("t0", "f8"), ("t1", "f8"), ("sigma", "f8"),
                            ("box_index", "i4")]),
        )
        t_end = float(chain_arr["t1"].max())
    else:
        chain_arr = np.empty(0, dtype=np.dtype(
            [("t0", "f8"), ("t1", "f8"), ("sigma", "f8"), ("box_index", "i4")]))
        t_end = 0.0

    trans64 = 1.0
    trans32 = 1.0
    scatter = 0.0
    step = 0
    max_diff = 0.0
    first_exceed: dict | None = None
    stop_reason = "completed"
    ds = cfg.step_length
    mantissa = cfg.fp32_mantissa_bits

    while step < cfg.max_steps:
        t_mid = (step + 0.5) * ds
        if t_mid >= t_end:
            break
        sigma, box_idx = _segment_sigma(chain_arr, t_mid)
        # 实际段长不越过区间链末端，保证不把介质外路径计入。
        seg_end = min((step + 1) * ds, t_end)
        seg_len = seg_end - step * ds
        if seg_len <= 0.0:
            break

        # 两条精度路径分别累乘同一光学厚度。
        attenuation = float(np.exp(-sigma * seg_len))
        trans64 *= attenuation
        trans32 = quantize_mantissa(trans32 * attenuation, mantissa)
        step += 1

        diff = abs(trans64 - trans32)
        if diff > max_diff:
            max_diff = diff
        if first_exceed is None and diff > cfg.transmittance_tolerance:
            seg = interval_at(chain, t_mid)
            first_exceed = {
                "step": step,
                "t_mid": t_mid,
                "box_index": box_idx,
                "box_name": seg.volume.name if seg is not None else None,
                "transmittance64": trans64,
                "transmittance32": trans32,
                "abs_diff": diff,
            }

        if sigma > 0.0:
            point = origin + direction * t_mid
            incoming = _light_contribution(point, scene.lights, scene.volumes, cfg)
            # 散射反照率取自中点所属介质；介质外不产生体散射。
            albedo = 0.0
            if box_idx >= 0:
                albedo = scene.volumes[box_idx].albedo
            scatter += trans64 * albedo * sigma * incoming * seg_len

        if trans64 < cfg.transmittance_cutoff:
            stop_reason = "cutoff"
            break
    else:
        # while 条件耗尽而未 break：达到最大步数且仍在介质内。
        if step >= cfg.max_steps and (step * ds) < t_end:
            stop_reason = "max_steps"

    radiance = cfg.background * trans64 + scatter
    return RayResult(
        radiance=float(radiance), steps=step, stop_reason=stop_reason,
        max_abs_diff=float(max_diff), first_exceed=first_exceed,
        transmittance64=float(trans64),
    )


def tile_ray_memory(width: int, height: int, max_steps: int) -> int:
    """估算一个分块全部光线的在途采样缓冲上限（字节）。"""
    n = width * height
    return n * (PER_RAY_BUFFER_BYTES + PER_STEP_BUFFER_BYTES * max_steps)


def render_tile(scene: Scene, tile_index: int, x0: int, y0: int,
                tile_w: int, tile_h: int,
                pixel_xy: list[tuple[int, int]] | None = None) -> TileResult:
    """渲染超采样网格上的一个分块。块内结果完全私有，不依赖调度顺序。"""
    cfg = scene.config
    frame = build_frame(scene.camera)
    grid_w = cfg.width * cfg.supersample
    grid_h = cfg.height * cfg.supersample
    n = cfg.supersample

    gray = np.zeros((tile_h, tile_w), dtype=np.float64)
    diff = np.zeros((tile_h, tile_w), dtype=np.float64)
    steps = np.zeros((tile_h, tile_w), dtype=np.int32)
    reasons = np.full((tile_h, tile_w), "completed", dtype="<U16")
    exceeds: list[dict] = []
    cutoff_count = 0
    max_steps_count = 0

    for ly in range(tile_h):
        for lx in range(tile_w):
            gx, gy = x0 + lx, y0 + ly
            px, py = gx // n, gy // n
            sx, sy = gx % n, gy % n
            u, v = subpixel_uv(px, py, cfg.width, cfg.height, n, sx, sy)
            origin, direction = generate_ray(scene.camera, u, v, frame)
            result = render_ray(origin, direction, scene)
            gray[ly, lx] = result.radiance
            diff[ly, lx] = result.max_abs_diff
            steps[ly, lx] = result.steps
            reasons[ly, lx] = result.stop_reason
            if result.stop_reason == "cutoff":
                cutoff_count += 1
            elif result.stop_reason == "max_steps":
                max_steps_count += 1
            if result.first_exceed is not None and len(exceeds) < 256:
                record = dict(result.first_exceed)
                record.update({"pixel": [int(px), int(py)],
                               "subpixel": [int(sx), int(sy)],
                               "grid_xy": [int(gx), int(gy)]})
                exceeds.append(record)

    return TileResult(
        tile_index=tile_index, x0=x0, y0=y0, width=tile_w, height=tile_h,
        grayscale=gray, diff_map=diff, steps_map=steps, reason_map=reasons,
        first_exceeds=exceeds, cutoff_count=cutoff_count,
        max_steps_count=max_steps_count,
    )
