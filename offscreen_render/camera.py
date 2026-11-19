"""相机坐标系与光线生成（确定性，无随机采样）。"""
from __future__ import annotations

import numpy as np

from .scene import Camera


def build_frame(camera: Camera) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (forward, right, up) 正交单位基。

    上方向优先取世界 +Y；当朝向与 +Y 平行时退化为 +Z，保证确定性。
    """
    forward = camera.forward.astype(np.float64)
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(forward, world_up))) > 0.999:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    return forward, right, up


def generate_ray(camera: Camera, u: float, v: float,
                 frame: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
                 ) -> tuple[np.ndarray, np.ndarray]:
    """按归一化画面坐标 (u,v) in [-1,1] 生成光线原点与单位方向。

    u 沿宽度方向、v 沿高度方向；垂直视场角决定 v=±1 的张角。
    """
    forward, right, up = frame or build_frame(camera)
    half_h = np.tan(np.deg2rad(camera.fov_v_deg) * 0.5)
    half_w = half_h * camera.aspect
    direction = forward + right * (u * half_w) + up * (v * half_h)
    direction = direction / np.linalg.norm(direction)
    return camera.position.astype(np.float64).copy(), direction


def subpixel_uv(px: int, py: int, width: int, height: int,
                supersample: int, sx: int, sy: int) -> tuple[float, float]:
    """像素 (px,py) 的第 (sx,sy) 个超采样子像素的归一化画面坐标。

    子样本在像素内均匀居中，顺序固定，因此整体渲染完全确定。
    """
    n = max(1, supersample)
    # 子像素中心在超采样网格中的索引
    gx = px * n + sx
    gy = py * n + sy
    grid_w = width * n
    grid_h = height * n
    u = (2.0 * (gx + 0.5) / grid_w) - 1.0
    # 图像顶部对应 v=+1
    v = 1.0 - (2.0 * (gy + 0.5) / grid_h)
    return u, v
