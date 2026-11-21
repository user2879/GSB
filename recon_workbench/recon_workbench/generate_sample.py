"""固定种子多视图样本生成器。

场景由两块透视投影的棋盘格平面与若干几何图元构成，角点丰富、
渲染逐字节确定（无随机噪点，随机数仅用于可复现的标记点抖动）。
"""
from __future__ import annotations

import argparse
import math
import os
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .utils import atomic_write_json, canonical_bytes

SEED = 20261121
WIDTH = 240
HEIGHT = 180
N_VIEWS = 5
CAMERA_RADIUS = 6.0
LOOK_AT = np.array([0.0, 0.0, 1.0])


def _look_at(eye: np.ndarray, target: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    """世界到相机的 3x3 旋转矩阵（相机看向 -Z 方向）。"""
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, up_hint)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    # OpenCV 约定：相机看向 +Z，右 +X，下 +Y（up 取反）
    return np.vstack([right, -up, forward])


def camera_pose(index: int, n_views: int) -> Tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(-math.radians(28.0), math.radians(28.0), n_views)
    theta = float(angles[index])
    eye = np.array(
        [CAMERA_RADIUS * math.sin(theta), 1.4, CAMERA_RADIUS * math.cos(theta)]
    )
    rotation = _look_at(eye, LOOK_AT, np.array([0.0, 1.0, 0.0]))
    translation = -rotation @ eye
    return rotation, translation


def intrinsic_matrix() -> np.ndarray:
    fx = fy = 260.0
    return np.array(
        [[fx, 0.0, WIDTH / 2.0], [0.0, fy, HEIGHT / 2.0], [0.0, 0.0, 1.0]]
    )


def project(points_world: np.ndarray, intrinsic: np.ndarray, rotation: np.ndarray,
            translation: np.ndarray) -> np.ndarray:
    cam = points_world @ rotation.T + translation
    depth = cam[:, 2]
    pixels = cam @ intrinsic.T
    pixels = pixels[:, :2] / pixels[:, 2:3]
    out = np.column_stack([pixels, depth])
    return out


def _plane_cells(origin: np.ndarray, axis_u: np.ndarray, axis_v: np.ndarray,
                 cols: int, rows: int, size: float) -> List[Dict[str, object]]:
    cells: List[Dict[str, object]] = []
    for i in range(cols):
        for j in range(rows):
            p0 = origin + axis_u * (i * size) + axis_v * (j * size)
            corners = np.vstack(
                [
                    p0,
                    p0 + axis_u * size,
                    p0 + axis_u * size + axis_v * size,
                    p0 + axis_v * size,
                ]
            )
            cells.append({"i": i, "j": j, "corners": corners,
                          "dark": (i + j) % 2 == 0})
    return cells


def build_scene() -> Dict[str, object]:
    rng = np.random.default_rng(SEED)
    # 竖直棋盘格平面（z=2 附近）与水平棋盘格地面（y=-1.5）
    vertical = _plane_cells(
        origin=np.array([-2.0, -1.0, 2.0]),
        axis_u=np.array([1.0, 0.0, 0.0]),
        axis_v=np.array([0.0, 1.0, 0.0]),
        cols=8, rows=6, size=0.5,
    )
    horizontal = _plane_cells(
        origin=np.array([-2.0, -1.5, 0.5]),
        axis_u=np.array([1.0, 0.0, 0.0]),
        axis_v=np.array([0.0, 0.0, 1.0]),
        cols=8, rows=5, size=0.5,
    )
    # 确定性抖动的纹理标记点（三角图元顶点），可被角点检测拾取
    jitter = rng.uniform(-0.03, 0.03, size=(6, 3))
    markers = np.array(
        [[-1.6, 0.8, 2.02], [-0.4, 1.2, 2.02], [0.9, 0.4, 2.02],
         [-1.2, -0.6, 2.02], [0.3, -0.8, 2.02], [1.4, 1.0, 2.02]],
        dtype=np.float64,
    ) + jitter
    return {"vertical": vertical, "horizontal": horizontal, "markers": markers}


def render_view(view_index: int, scene: Dict[str, object],
                intrinsic: np.ndarray) -> Image.Image:
    rotation, translation = camera_pose(view_index, N_VIEWS)
    image = Image.new("RGB", (WIDTH, HEIGHT), (245, 245, 245))
    draw = ImageDraw.Draw(image)

    for plane_name in ("vertical", "horizontal"):
        for cell in scene[plane_name]:  # type: ignore[index]
            projected = project(cell["corners"], intrinsic, rotation, translation)
            if (projected[:, 2] <= 0).any():
                continue
            polygon = [(float(p[0]), float(p[1])) for p in projected]
            fill = (20, 20, 20) if cell["dark"] else (235, 235, 235)
            draw.polygon(polygon, fill=fill, outline=(110, 110, 110))

    # 几何图元：三角形与圆，作为额外纹理标记
    marker_px = project(scene["markers"], intrinsic, rotation, translation)
    for idx, marker in enumerate(marker_px):
        x, y = float(marker[0]), float(marker[1])
        if idx % 2 == 0:
            draw.polygon(
                [(x, y - 5), (x - 5, y + 4), (x + 5, y + 4)],
                fill=(180, 30, 30), outline=(0, 0, 0),
            )
        else:
            draw.ellipse([x - 4, y - 4, x + 4, y + 4],
                         fill=(30, 60, 180), outline=(0, 0, 0))
    return image


def generate(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    intrinsic = intrinsic_matrix()
    scene = build_scene()
    manifest_views: List[Dict[str, object]] = []
    for view in range(N_VIEWS):
        image = render_view(view, scene, intrinsic)
        image_path = os.path.join(out_dir, f"view_{view:02d}.png")
        image.save(image_path, format="PNG", compress_level=6)

        rotation, translation = camera_pose(view, N_VIEWS)
        fx = float(intrinsic[0, 0])
        fov_deg = 2.0 * math.degrees(math.atan(WIDTH / (2.0 * fx)))
        camera = {
            "view": view,
            "width": WIDTH,
            "height": HEIGHT,
            "intrinsic": intrinsic.tolist(),
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
            "fov_deg": fov_deg,
        }
        camera_path = os.path.join(out_dir, f"view_{view:02d}.camera.json")
        atomic_write_json(camera_path, camera)
        manifest_views.append(
            {"view": view, "image": os.path.basename(image_path),
             "camera": os.path.basename(camera_path)}
        )

    atomic_write_json(
        os.path.join(out_dir, "scene_ground_truth.json"),
        {"markers": np.asarray(scene["markers"]).tolist(),
         "look_at": LOOK_AT.tolist(), "seed": SEED},
    )
    atomic_write_json(
        os.path.join(out_dir, "manifest.json"),
        {"views": manifest_views, "width": WIDTH, "height": HEIGHT,
         "generator_fingerprint": __import__("hashlib").sha256(
             canonical_bytes({"seed": SEED, "width": WIDTH,
                              "height": HEIGHT, "views": N_VIEWS})
         ).hexdigest()[:16]},
    )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="生成确定性多视图样本")
    parser.add_argument("out_dir", help="样本输出目录")
    args = parser.parse_args()
    generate(args.out_dir)
    print(f"sample generated at {args.out_dir}")


if __name__ == "__main__":
    main()
