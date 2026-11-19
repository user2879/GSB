"""程序化表面：确定性高度场、逐像素法向与反照率贴图。

仅依赖 NumPy 的 ``default_rng(seed)`` 与封闭形式的三角函数，
相同种子 + 相同尺寸在任何平台产生位级一致的数组。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Surface:
    """一张参数化表面（xy 网格，z=height，法向与反照率逐像素定义）。"""

    height: np.ndarray  # (H, W) float64
    normals: np.ndarray  # (H, W, 3) float64，单位法向
    albedo: np.ndarray  # (H, W) float64，[0,1] 标量反照率
    seed: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.height.shape


def build_height_field(size: int, seed: int) -> np.ndarray:
    """由固定种子构造平滑高度起伏（低频正弦 + 平滑随机凸包）。"""
    rng = np.random.default_rng(seed)
    axis = np.linspace(0.0, 1.0, size, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis)

    height = (
        0.18 * np.sin(2.0 * np.pi * 2.0 * xx) * np.cos(2.0 * np.pi * 3.0 * yy)
        + 0.10 * np.sin(2.0 * np.pi * 5.0 * (xx + 0.3 * yy))
        + 0.06 * np.cos(2.0 * np.pi * 7.0 * (yy - 0.2 * xx))
    )

    # 有限个随机中心的平滑高斯凸包，保证可复现且处处可微
    centers = rng.uniform(0.15, 0.85, size=(6, 2))
    amplitudes = rng.uniform(-0.12, 0.12, size=6)
    sigmas = rng.uniform(0.05, 0.12, size=6)
    for (cx, cy), amp, sigma in zip(centers, amplitudes, sigmas):
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        height += amp * np.exp(-d2 / (2.0 * sigma**2))
    return height


def normals_from_height(height: np.ndarray) -> np.ndarray:
    """中心差分计算高度场法向并 L2 归一化（np.gradient 采用等间距单位网格）。"""
    dy, dx = np.gradient(height)
    # 表面 z = h(x, y) 的切平面法向 (-dh/dx, -dh/dy, 1)
    normals = np.stack([-dx, -dy, np.ones_like(height)], axis=-1)
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals / norms


def build_albedo(size: int, seed: int) -> np.ndarray:
    """确定性反照率贴图：平滑条带 + 斑块，裁剪到 [0.08, 0.92]。"""
    rng = np.random.default_rng(seed + 101)
    axis = np.linspace(0.0, 1.0, size, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis)
    base = 0.5 + 0.25 * np.sin(2.0 * np.pi * 1.5 * xx) * np.cos(2.0 * np.pi * 2.0 * yy)
    centers = rng.uniform(0.1, 0.9, size=(8, 2))
    shades = rng.uniform(-0.2, 0.2, size=8)
    sigmas = rng.uniform(0.04, 0.1, size=8)
    for (cx, cy), shade, sigma in zip(centers, shades, sigmas):
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        base += shade * np.exp(-d2 / (2.0 * sigma**2))
    return np.clip(base, 0.08, 0.92)


def build_surface(size: int, seed: int) -> Surface:
    """生成完整的程序化表面。"""
    height = build_height_field(size, seed)
    normals = normals_from_height(height)
    albedo = build_albedo(size, seed)
    return Surface(height=height, normals=normals, albedo=albedo, seed=seed)
