"""Deterministic local multi-view sample generator.

The generator is deliberately independent of external data and uses integer
hash noise, integer matrices and fixed-point projection only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import default_config, save_config
from .fixed import round_div
from .pngio import PngWriter


PATCH_SIZE = 5


def _hash32(*values: int) -> int:
    value = 0x811C9DC5
    for item in values:
        value ^= item & 0xFFFFFFFF
        value = (value * 0x01000193) & 0xFFFFFFFF
        value ^= value >> 13
        value = (value * 0x85EBCA6B) & 0xFFFFFFFF
    return value & 0xFFFFFFFF


def scene_landmarks(scene_seed: int) -> list[dict[str, Any]]:
    xs = [-48, -16, 16, 48]
    ys = [-24, 0, 24]
    landmarks: list[dict[str, Any]] = []
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            index = row * len(xs) + col
            z = ((index * 29 + scene_seed) % 83) - 30
            color = [
                165 + ((index * 11 + scene_seed) % 75),
                30 + ((index * 37 + scene_seed // 3) % 195),
                25 + ((index * 71 + scene_seed // 7) % 205),
            ]
            landmarks.append(
                {
                    "id": f"L{index:02d}",
                    "world": [x, y, z],
                    "color": color,
                }
            )
    return landmarks


def _draw_background(rgb: np.ndarray, width: int, height: int, seed: int, view_index: int) -> None:
    for y in range(height):
        row = rgb[y]
        for x in range(width):
            noise = _hash32(x, y, seed, view_index)
            gray = 26 + noise % 29
            tint = (noise >> 8) % 5
            row[x] = (gray + tint // 2, gray, gray + tint)

    for primitive in range(3):
        left = 8 + (_hash32(seed, view_index, primitive * 3) % max(1, width - 36))
        top = 6 + (_hash32(seed, view_index, primitive * 3 + 1) % max(1, height - 28))
        box_w = 12 + (_hash32(seed, view_index, primitive * 3 + 2) % 21)
        box_h = 8 + (_hash32(seed, view_index, primitive * 3 + 4) % 16)
        shade = 43 + (_hash32(seed, primitive, view_index) % 20)
        for y in range(top, min(height, top + box_h)):
            for x in range(left, min(width, left + box_w)):
                edge = x == left or y == top or x == left + box_w - 1 or y == top + box_h - 1
                if edge or ((x + y + primitive) % 7 == 0):
                    rgb[y, x] = (shade, shade + 2, shade)

    line_y = 12 + (_hash32(seed, view_index, 91) % max(1, height - 24))
    for step in range(min(width, height) // 2):
        x = (step * 3 + 7) % width
        y = (line_y + step) % height
        rgb[y, x] = (78, 76, 72)


def _draw_landmarks(
    rgb: np.ndarray,
    config: dict[str, Any],
    view: dict[str, Any],
    scene_seed: int,
) -> list[dict[str, Any]]:
    width = int(config["resolution"]["width"])
    height = int(config["resolution"]["height"])
    fixed_bits = int(config.get("fixed_bits", 10))
    scale = 1 << fixed_bits
    intrinsics = view["intrinsics"]
    extrinsics = view["extrinsics"]
    projections: list[dict[str, Any]] = []
    for landmark in scene_landmarks(scene_seed):
        wx, wy, wz = (int(value) * scale for value in landmark["world"])
        xc = extrinsics[0][0] * wx + extrinsics[0][1] * wy + extrinsics[0][2] * wz + extrinsics[0][3] * scale
        yc = extrinsics[1][0] * wx + extrinsics[1][1] * wy + extrinsics[1][2] * wz + extrinsics[1][3] * scale
        zc = extrinsics[2][0] * wx + extrinsics[2][1] * wy + extrinsics[2][2] * wz + extrinsics[2][3] * scale
        if zc == 0:
            continue
        u_num = intrinsics[0][0] * xc + intrinsics[0][1] * yc + intrinsics[0][2] * zc
        v_num = intrinsics[1][0] * xc + intrinsics[1][1] * yc + intrinsics[1][2] * zc
        w = intrinsics[2][0] * xc + intrinsics[2][1] * yc + intrinsics[2][2] * zc
        if w == 0:
            continue
        u, u_dir = round_div(u_num, w)
        v, v_dir = round_div(v_num, w)
        visible = PATCH_SIZE <= u < width - PATCH_SIZE and PATCH_SIZE <= v < height - PATCH_SIZE
        projections.append(
            {
                "id": landmark["id"],
                "u": u,
                "v": v,
                "depth": zc,
                "rounding": {"u": u_dir, "v": v_dir},
                "visible": visible,
                "world": landmark["world"],
            }
        )
        if not visible:
            continue
        base = landmark["color"]
        half = PATCH_SIZE // 2
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                checker = (dx + dy + int(landmark["id"][1:])) % 3
                delta = (-18, 18, 49)[checker]
                pixel = [max(0, min(255, channel + delta)) for channel in base]
                rgb[v + dy, u + dx] = pixel
    return projections


def render_view(config: dict[str, Any], view: dict[str, Any]) -> np.ndarray:
    width = int(config["resolution"]["width"])
    height = int(config["resolution"]["height"])
    scene_seed = int(config.get("scene_seed", 0))
    view_index = sorted(item["id"] for item in config["views"]).index(view["id"])
    render_seed = int(view.get("render_seed", 0))
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    view_seed = _hash32(scene_seed, render_seed, view_index, 0xA17F)
    _draw_background(rgb, width, height, view_seed, view_index)
    _draw_landmarks(rgb, config, view, scene_seed)
    return rgb


def write_view(project_dir: str | Path, config: dict[str, Any], view: dict[str, Any]) -> np.ndarray:
    root = Path(project_dir)
    image = render_view(config, view)
    width = int(config["resolution"]["width"])
    height = int(config["resolution"]["height"])
    PngWriter(width, height).write(root / view["image"], image)
    return image


def initialize_project(project_dir: str | Path, *, config: dict[str, Any] | None = None) -> dict[str, Any]:
    root = Path(project_dir)
    root.mkdir(parents=True, exist_ok=True)
    effective = config if config is not None else default_config()
    save_config(root, effective)
    for view in effective["views"]:
        write_view(root, effective, view)
    return effective
