"""诊断可视化：特征覆盖、匹配连线、深度图、点云投影（Pillow）。"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Sequence

import numpy as np
from PIL import Image, ImageDraw

from .features import Feature, MatchRecord
from .geometry import CloudPoint, DepthMap


def _to_rgb(gray_or_color: np.ndarray) -> Image.Image:
    if gray_or_color.ndim == 3:
        return Image.fromarray(gray_or_color, mode="RGB")
    return Image.fromarray(gray_or_color, mode="L").convert("RGB")


def draw_features(image: np.ndarray, features: Sequence[Feature],
                  out_path: str) -> None:
    canvas = _to_rgb(image)
    draw = ImageDraw.Draw(canvas)
    for feature in features:
        x, y = feature.x, feature.y
        draw.ellipse([x - 2, y - 2, x + 2, y + 2],
                     outline=(255, 40, 40), width=1)
    canvas.save(out_path, format="PNG")


def draw_matches(image_a: np.ndarray, image_b: np.ndarray,
                 matches: Sequence[MatchRecord],
                 features_a: Sequence[Feature],
                 features_b: Sequence[Feature], out_path: str) -> None:
    rgb_a = _to_rgb(image_a)
    rgb_b = _to_rgb(image_b)
    gap = 6
    width = rgb_a.width + rgb_b.width + gap
    height = max(rgb_a.height, rgb_b.height)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    canvas.paste(rgb_a, (0, 0))
    canvas.paste(rgb_b, (rgb_a.width + gap, 0))
    draw = ImageDraw.Draw(canvas)
    map_a = {f.feature_id: f for f in features_a}
    map_b = {f.feature_id: f for f in features_b}
    offset = rgb_a.width + gap
    for index, match in enumerate(matches):
        fa = map_a[match.feature_a]
        fb = map_b[match.feature_b]
        color = (40, 140, 40) if not match.boundary else (220, 140, 0)
        draw.line(
            [(fa.x, fa.y), (fb.x + offset, fb.y)], fill=color, width=1
        )
    canvas.save(out_path, format="PNG")


def draw_depth_map(depth: DepthMap, out_path: str) -> None:
    levels = depth.levels
    normalized = np.zeros_like(levels, dtype=np.uint8)
    finite = levels >= 0
    if finite.any():
        span = max(1, int(levels[finite].max()))
        normalized[finite] = (
            levels[finite].astype(np.float64) / span * 255.0
        ).astype(np.uint8)
    image = Image.fromarray(normalized, mode="L").resize(
        (levels.shape[1] * 6, levels.shape[0] * 6), Image.NEAREST
    )
    image.save(out_path, format="PNG")


def draw_cloud_projection(cloud: Sequence[CloudPoint], out_path: str,
                          image_size: tuple[int, int] = (320, 240)) -> None:
    canvas = Image.new("RGB", image_size, (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    if not cloud:
        canvas.save(out_path, format="PNG")
        return
    coords = np.array([[p.x, p.y, p.z] for p in cloud], dtype=np.float64)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    span = np.maximum(maxs - mins, 1e-9)
    margin = 12
    for point in cloud:
        nx = (point.x - mins[0]) / span[0]
        nz = (point.z - mins[2]) / span[2]
        x = margin + nx * (image_size[0] - 2 * margin)
        y = image_size[1] - margin - nz * (image_size[1] - 2 * margin)
        shade = int(200 * np.clip(point.confidence, 0.0, 1.0))
        draw.ellipse(
            [x - 2, y - 2, x + 2, y + 2],
            fill=(30, 30 + shade // 4, 200 - shade // 3),
            outline=(0, 0, 0),
        )
    canvas.save(out_path, format="PNG")


def render_all(load_views: List[Any], result_by_precision: Dict[str, Any],
               out_dir: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    written: List[str] = []
    raw = result_by_precision["float32"]["raw"]
    view_map = {v.view: v for v in load_views}
    for view, features in sorted(raw.features.items()):
        path = os.path.join(out_dir, f"features_v{view:02d}.png")
        draw_features(view_map[view].color, features, path)
        written.append(path)
    for pair in raw.pairs:
        path = os.path.join(
            out_dir, f"matches_{pair.view_a:02d}_{pair.view_b:02d}.png"
        )
        draw_matches(
            view_map[pair.view_a].color, view_map[pair.view_b].color,
            pair.matches, raw.features[pair.view_a],
            raw.features[pair.view_b], path,
        )
        written.append(path)
        for depth in (pair.depth_a, pair.depth_b):
            dpath = os.path.join(out_dir, f"depth_v{depth.view:02d}.png")
            draw_depth_map(depth, dpath)
            written.append(dpath)
    cloud_path = os.path.join(out_dir, "cloud_projection.png")
    draw_cloud_projection(raw.cloud, cloud_path)
    written.append(cloud_path)
    return written
