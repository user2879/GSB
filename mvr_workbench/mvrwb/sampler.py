"""固定种子多视图样本生成器。

生成一个确定性的合成场景：空间中若干带纹理编号的亮方块标记，
6 个环绕相机以相同内参观测，输出 PNG 灰度图与相机内外参 JSON。
所有随机数来自 numpy.default_rng(固定种子)，重复生成逐字节一致。
"""

from __future__ import annotations

import math
import os

import numpy as np
from PIL import Image

from .util import write_json_atomic

SEED = 20261121
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240
MARKER_RADIUS = 7
MARKER_VALUE = 235
BACKGROUND_VALUE = 12
MARKER_DARK_VALUE = 150
MARKER_GRAY_LEVELS = (120, 158, 196, 235)


def _camera_poses(view_count: int, radius: float, height: float):
    poses = []
    for index in range(view_count):
        angle = 2.0 * math.pi * index / view_count
        eye = np.array([radius * math.cos(angle), height, radius * math.sin(angle)])
        # 俯视目标点，保证环绕各视图（含轴点机位）都能看到场景标记
        target = np.array([0.0, -0.2, 0.0])
        up = np.array([0.0, 1.0, 0.0])
        forward = target - eye
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        true_up = np.cross(right, forward)
        # 世界 -> 相机（OpenCV 约定）：x=right, y=down, z=看向目标
        rotation = np.vstack([right, -true_up, forward])
        translation = -rotation @ eye
        poses.append((eye, rotation, translation))
    return poses


def generate_sample(output_dir: str, view_count: int = 6, marker_count: int = 28) -> str:
    rng = np.random.default_rng(SEED)
    views_dir = os.path.join(output_dir, "views")
    os.makedirs(views_dir, exist_ok=True)

    # 场景标记点：带确定性抖动的三维网格，保证多视图可见且有足够视差
    side = int(math.ceil(marker_count ** (1.0 / 3.0)))
    grid = np.linspace(-1.8, 1.8, side)
    markers = []
    for x in grid:
        for y in grid:
            for z in grid:
                if len(markers) >= marker_count:
                    break
                jitter = rng.uniform(-0.05, 0.05, size=3)
                markers.append(np.array([x, y, z]) + jitter)
    markers = markers[:marker_count]

    focal = 360.0
    intrinsics = np.array([
        [focal, 0.0, IMAGE_WIDTH / 2.0],
        [0.0, focal, IMAGE_HEIGHT / 2.0],
        [0.0, 0.0, 1.0],
    ])
    poses = _camera_poses(view_count, radius=5.5, height=2.2)

    for view_id, (eye, rotation, translation) in enumerate(poses, start=1):
        camera_points = (rotation @ np.asarray(markers).T).T + translation
        homogeneous = (intrinsics @ camera_points.T).T
        pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
        image = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), BACKGROUND_VALUE, dtype=np.uint8)
        # 确定性弱纹理：固定条纹，避免平坦区域产生伪角点
        for x in range(0, IMAGE_WIDTH, 16):
            image[:, x:x + 1] = BACKGROUND_VALUE + 6
        for index, (pixel, cam_point) in enumerate(zip(pixels, camera_points)):
            if (cam_point[2] <= 0.1 or pixel[0] <= 0 or pixel[1] <= 0
                    or pixel[0] >= IMAGE_WIDTH or pixel[1] >= IMAGE_HEIGHT):
                continue
            px, py = int(round(pixel[0])), int(round(pixel[1]))
            x0, x1 = max(0, px - MARKER_RADIUS), min(IMAGE_WIDTH, px + MARKER_RADIUS + 1)
            y0, y1 = max(0, py - MARKER_RADIUS), min(IMAGE_HEIGHT, py + MARKER_RADIUS + 1)
            if x1 > x0 and y1 > y0:
                # 唯一确定性纹理：按标记编号取独立随机流生成 15x15 棋盘位模式，
                # 标记间距足够大且描述子窗口位于块内部，跨视图描述子稳定可区分
                pattern_rng = np.random.default_rng(SEED + 1009 * (index + 1))
                pattern = pattern_rng.integers(0, len(MARKER_GRAY_LEVELS), size=(21, 21))
                tile = np.asarray(MARKER_GRAY_LEVELS, dtype=np.uint8)[pattern]
                image[y0:y1, x0:x1] = tile[: y1 - y0, : x1 - x0]
        Image.fromarray(image, mode="L").save(
            os.path.join(views_dir, f"view_{view_id:04d}.png"), format="PNG"
        )
        camera_payload = {
            "view_id": view_id,
            "intrinsics": intrinsics.tolist(),
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
            "width": IMAGE_WIDTH,
            "height": IMAGE_HEIGHT,
        }
        write_json_atomic(
            os.path.join(views_dir, f"view_{view_id:04d}.cam.json"), camera_payload
        )

    config = {
        "feature_threshold": 150.0,
        "match_min_inliers": 6,
        "triang_min_parallax_deg": 1.5,
        "depth_resolution": [60, 80],
        "depth_quant_levels": 32,
        "depth_smooth_passes": 2,
        "voxel_size": 0.35,
        "fixed_point_frac_bits": 12,
        "thread_workers": 4,
        "global_point_budget": 200,
        "per_view_quota": 40,
        "snapshot_commit_interval_views": 2,
        "view_pairs": [[i, i + 1] for i in range(1, view_count)],
    }
    write_json_atomic(os.path.join(output_dir, "reconstruction_config.json"), config)

    edit_stream = {
        "commands": [
            {"command": "replace_image", "view_id": 2, "source_image": "alt_view_0002.png"},
            {"command": "replace_camera", "view_id": 3, "source_camera": "alt_view_0003.cam.json"},
            {"command": "add_view", "view_id": 7,
             "source_image": "extras/view_0007.png",
             "source_camera": "extras/view_0007.cam.json"},
            {"command": "remove_view", "view_id": 7},
            {"command": "update_config",
             "parameters": {"feature_threshold": 170.0, "voxel_size": 0.28}},
        ]
    }
    write_json_atomic(os.path.join(output_dir, "edit_stream.json"), edit_stream)

    injections = {
        "injections": [
            {"id": "INJ-001", "type": "corrupt_png", "view_id": 5, "stage": "load"},
            {"id": "INJ-002", "type": "missing_camera", "view_id": 6, "stage": "load"},
            {"id": "INJ-003", "type": "parallax_out_of_range",
             "pair_id": "vp-0001-0002", "stage": "matching"},
            {"id": "INJ-004", "type": "non_finite_feature", "view_id": 4, "stage": "feature"},
        ]
    }
    write_json_atomic(os.path.join(output_dir, "injection_config.json"), injections)
    _build_alternative_assets(output_dir, view_count, intrinsics)
    return output_dir


def _build_alternative_assets(output_dir: str, view_count: int, intrinsics) -> None:
    """生成增量编辑所需的备选视图与额外视图资产（确定性，纳入固定种子）。"""
    alt_rng = np.random.default_rng(777)
    alt_image = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), BACKGROUND_VALUE + 10, dtype=np.uint8)
    for _ in range(18):
        px = int(alt_rng.integers(20, IMAGE_WIDTH - 20))
        py = int(alt_rng.integers(20, IMAGE_HEIGHT - 20))
        alt_image[py - 3:py + 4, px - 3:px + 4] = MARKER_VALUE
    Image.fromarray(alt_image, mode="L").save(
        os.path.join(output_dir, "alt_view_0002.png"), format="PNG"
    )

    alt_camera = read_json_safe(os.path.join(output_dir, "views", "view_0003.cam.json"))
    alt_camera["translation"] = [value + 0.05 for value in alt_camera["translation"]]
    write_json_atomic(os.path.join(output_dir, "alt_view_0003.cam.json"), alt_camera)

    extras = os.path.join(output_dir, "extras")
    os.makedirs(extras, exist_ok=True)
    eye, rotation, translation = _camera_poses(
        view_count + 1, radius=5.5, height=2.2
    )[-1]
    extra_image = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), BACKGROUND_VALUE, dtype=np.uint8)
    for _ in range(16):
        px = int(alt_rng.integers(18, IMAGE_WIDTH - 18))
        py = int(alt_rng.integers(18, IMAGE_HEIGHT - 18))
        extra_image[py - 3:py + 4, px - 3:px + 4] = MARKER_VALUE
    Image.fromarray(extra_image, mode="L").save(
        os.path.join(extras, "view_0007.png"), format="PNG"
    )
    write_json_atomic(
        os.path.join(extras, "view_0007.cam.json"),
        {
            "view_id": view_count + 1,
            "intrinsics": intrinsics.tolist(),
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
            "width": IMAGE_WIDTH,
            "height": IMAGE_HEIGHT,
        },
    )


def read_json_safe(path: str):
    import json
    with open(path, "rb") as handle:
        return json.loads(handle.read().decode("utf-8"))
