"""固定种子的确定性合成场景与多视图渲染（仅依赖 numpy/Pillow）。"""
from __future__ import annotations

import math
import os

import numpy as np
from PIL import Image

from .camera import Camera, look_at

IMAGE_W, IMAGE_H = 128, 96
MARKER_RADIUS = 2          # 5x5 几何图元
MARKER_BASE = 180          # 纹理标记基准灰度（按编号线性编码）
BACKGROUND = 20
NOISE_AMPLITUDE = 4


def make_scene(seed: int = 20261124, n_landmarks: int = 16):
    """确定性三维几何图元（纹理标记）集合。"""
    rng = np.random.default_rng(seed)
    # 在 z=0 平面附近的规则网格上施加固定随机扰动
    side = int(math.ceil(math.sqrt(n_landmarks)))
    pts = []
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= n_landmarks:
                break
            x = (j - (side - 1) / 2.0) * 1.4
            y = (i - (side - 1) / 2.0) * 1.4
            z = float(rng.uniform(-0.6, 0.6))
            pts.append((idx, (float(x), float(y), z)))
            idx += 1
    return pts


def make_cameras(n_views: int = 6, distance: float = 8.5,
                 image_w: int = IMAGE_W, image_h: int = IMAGE_H) -> dict:
    """绕场景一周的相机环；俯仰固定、方位角均分。"""
    fx = fy = 130.0
    cx, cy = image_w / 2.0, image_h / 2.0
    cams = {}
    for view in range(n_views):
        ang = 2.0 * math.pi * view / n_views
        eye = (distance * math.cos(ang), distance * math.sin(ang), 1.2)
        rot, trans = look_at(eye, (0.0, 0.0, 0.0))
        cams[view] = Camera(view=view, fx=fx, fy=fy, cx=cx, cy=cy, rot=rot, trans=trans)
    return cams


def render_view(camera: Camera, scene, image_w: int = IMAGE_W,
                image_h: int = IMAGE_H, seed: int = 0) -> np.ndarray:
    """渲染单视图：暗背景 + 固定噪声 + 按编号编码灰度的 5x5 几何图元。"""
    rng = np.random.default_rng(seed + camera.view * 997)
    img = np.full((image_h, image_w), BACKGROUND, dtype=np.int16)
    img += rng.integers(-NOISE_AMPLITUDE, NOISE_AMPLITUDE + 1, size=(image_h, image_w))

    for lid, point in scene:
        (u, v), depth = camera.project(point)
        if depth <= 0:
            continue
        ui, vi = int(round(float(u))), int(round(float(v)))
        if not (MARKER_RADIUS <= ui < image_w - MARKER_RADIUS
                and MARKER_RADIUS <= vi < image_h - MARKER_RADIUS):
            continue
        level = MARKER_BASE + lid * 4          # 纹理标记：灰度即编号
        for dy in range(-MARKER_RADIUS, MARKER_RADIUS + 1):
            for dx in range(-MARKER_RADIUS, MARKER_RADIUS + 1):
                img[vi + dy, ui + dx] = level + int(abs(dx) + abs(dy) == 0)
    return np.clip(img, 0, 255).astype(np.uint8)


def save_png(path: str, image: np.ndarray) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    Image.fromarray(image, mode="L").save(path, format="PNG")


def load_png_gray(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
