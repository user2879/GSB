from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from .fixedpoint import divide_round
from .png_codec import write_png_rgb


SEED = 20261126
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 48
BLOCK_SIZE = 4
FOCAL = 24
CX = 32
CY = 24
CAMERA_X = [-6, -2, 2, 6]
CAMERA_T_Y = [0, 1, -1, 0]

PALETTE = [
    (230, 40, 40), (40, 190, 70), (45, 95, 235), (245, 190, 30),
    (170, 60, 220), (20, 200, 210), (240, 105, 25), (120, 70, 30),
    (255, 120, 170), (90, 230, 120), (130, 160, 255), (210, 220, 40),
    (255, 90, 200), (0, 150, 160), (160, 90, 40), (90, 40, 160),
    (255, 160, 80), (180, 255, 120), (80, 120, 120), (220, 180, 255),
    (255, 230, 120), (120, 255, 220), (200, 60, 120), (80, 80, 220),
]


def _project_q(x: int, y: int, z: int, camera_x: int, camera_t_y: int = 0) -> tuple[int, int]:
    u, _ = divide_round(FOCAL * (x - camera_x), z)
    v, _ = divide_round(FOCAL * (y + camera_t_y), z)
    return CX + u, CY + v


def _snap_block(value: int, block_size: int = BLOCK_SIZE) -> int:
    snapped, _ = divide_round(value, block_size)
    return snapped * block_size


def _regular_points() -> list[dict[str, int]]:
    points: list[dict[str, int]] = []
    point_id = 0
    for x in (-6, -2, 2, 6):
        for y in (-4, 0, 4):
            points.append({"landmark": point_id, "x": x, "y": y, "z": 12})
            point_id += 1
    for x in (-2, 2):
        for y in (-4, 4):
            points.append({"landmark": point_id, "x": x, "y": y, "z": 8})
            point_id += 1
    for x in (-6, -2, 2, 6):
        for y in (-4, 4):
            points.append({"landmark": point_id, "x": x, "y": y, "z": 24})
            point_id += 1
    return points


def _irregular_points(existing: list[dict[str, int]]) -> list[dict[str, int]]:
    rng = random.Random(SEED + 77)
    points = list(existing)
    used_blocks: list[set[tuple[int, int]]] = [set() for _ in CAMERA_X]
    for point in existing:
        for view_index, camera_x in enumerate(CAMERA_X):
            u, v = _project_q(point["x"], point["y"], point["z"], camera_x, CAMERA_T_Y[view_index])
            bu, bv = _snap_block(u), _snap_block(v)
            used_blocks[view_index].add((bu, bv))

    next_id = max(p["landmark"] for p in existing) + 1
    accepted: list[dict[str, int]] = []
    attempts = 0
    while len(accepted) < 8 and attempts < 500:
        attempts += 1
        candidate = {
            "landmark": next_id,
            "x": rng.choice([-5, -4, -3, -1, 0, 1, 3, 4, 5]),
            "y": rng.choice([-3, -2, -1, 1, 2, 3]),
            "z": rng.choice([9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20]),
        }
        blocks = []
        valid = True
        for view_index, camera_x in enumerate(CAMERA_X):
            u, v = _project_q(candidate["x"], candidate["y"], candidate["z"], camera_x, CAMERA_T_Y[view_index])
            bu, bv = _snap_block(u), _snap_block(v)
            if not (0 <= bu <= IMAGE_WIDTH - BLOCK_SIZE and 0 <= bv <= IMAGE_HEIGHT - BLOCK_SIZE):
                valid = False
                break
            if (bu, bv) in used_blocks[view_index]:
                valid = False
                break
            blocks.append((bu, bv))
        if not valid:
            continue
        accepted.append(candidate)
        points.append(candidate)
        for view_index, block in enumerate(blocks):
            used_blocks[view_index].add(block)
        next_id += 1
    return accepted


def _background() -> np.ndarray:
    image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    for y in range(IMAGE_HEIGHT):
        for x in range(IMAGE_WIDTH):
            grid = 10 if (x % 8 == 0 or y % 8 == 0) else 0
            image[y, x] = (
                22 + x // 3 + grid,
                26 + y // 4 + grid // 2,
                42 + (x + y) // 8 + grid,
            )
    return image


def _paint_marker(image: np.ndarray, x0: int, y0: int, color: tuple[int, int, int], variant: int = 0) -> None:
    dark = tuple(max(0, c // 4) for c in color)
    pale = tuple(min(255, c + 55) for c in color)
    for dy in range(BLOCK_SIZE):
        for dx in range(BLOCK_SIZE):
            if variant == 1:
                selected = pale if (dx in (1, 2) or dy in (1, 2)) else dark
            elif (dx < 2 and dy < 2) or (dx >= 2 and dy >= 2):
                selected = color
            else:
                selected = dark if (dx + dy) % 2 else pale
            image[y0 + dy, x0 + dx] = selected


def build_scene() -> dict[str, Any]:
    regular = _regular_points()
    irregular = _irregular_points(regular)
    landmarks = regular + irregular
    view_blocks: list[dict[int, tuple[int, int]]] = [dict() for _ in CAMERA_X]
    for point in landmarks:
        for view_index, camera_x in enumerate(CAMERA_X):
            u, v = _project_q(point["x"], point["y"], point["z"], camera_x, CAMERA_T_Y[view_index])
            block = (_snap_block(u), _snap_block(v))
            view_blocks[view_index][point["landmark"]] = block
    return {
        "seed": SEED,
        "width": IMAGE_WIDTH,
        "height": IMAGE_HEIGHT,
        "block_size": BLOCK_SIZE,
        "focal": FOCAL,
        "principal_point": [CX, CY],
        "camera_x": CAMERA_X,
        "landmarks": landmarks,
        "irregular_ids": [p["landmark"] for p in irregular],
        "view_blocks": view_blocks,
    }


def render_views(scene: dict[str, Any]) -> list[np.ndarray]:
    views = []
    for view_index in range(len(CAMERA_X)):
        image = _background()
        blocks = scene["view_blocks"][view_index]
        for landmark_id, (x0, y0) in blocks.items():
            _paint_marker(image, x0, y0, PALETTE[landmark_id % len(PALETTE)])
        views.append(image)
    return views


def camera_manifest() -> dict[str, Any]:
    views = []
    for index, camera_x in enumerate(CAMERA_X):
        views.append(
            {
                "index": index,
                "view_id": f"view_{index:02d}",
                "image": f"views/view_{index:02d}.png",
                "K": [[FOCAL, 0, CX], [0, FOCAL, CY], [0, 0, 1]],
                "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "t": [-camera_x, CAMERA_T_Y[index], 0],
            }
        )
    return {"schema_version": "1.0", "fixed_point_scale": 1024, "views": views}


def default_config() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "image": {"width": IMAGE_WIDTH, "height": IMAGE_HEIGHT, "pixel_mode": "RGB"},
        "fixed_point_scale": 1024,
        "algorithm_version": "mvrv-1",
        "min_parallax": 2,
        "residual_threshold_q": 820,
        "voxel_size": 4,
        "depth_quantization": 2,
        "max_subdivision_depth": 2,
        "world_bounds": {"min": [-12, -8, 4], "max": [12, 8, 28]},
        "global_point_budget": 100000,
        "view_point_quota": 100000,
        "workers": 2,
    }


def edit_commands() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "commands": [
            {"id": "cmd-01-min-parallax", "op": "set_config", "key": "min_parallax", "value": 5},
            {
                "id": "cmd-02-extrinsic",
                "op": "set_extrinsic",
                "view_index": 2,
                "t": [-3, -1, 0],
            },
            {
                "id": "cmd-03-replace-view",
                "op": "replace_view",
                "view_index": 1,
                "source": "replacements/view_01.png",
            },
            {"id": "cmd-04-voxel-size", "op": "set_config", "key": "voxel_size", "value": 3},
            {
                "id": "cmd-05-residual-threshold",
                "op": "set_config",
                "key": "residual_threshold_q",
                "value": 650,
            },
        ],
    }


def generate_project(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    data_dir = root / "data"
    config_dir = root / "config"
    view_dir = data_dir / "views"
    replacement_dir = data_dir / "replacements"
    data_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    view_dir.mkdir(parents=True, exist_ok=True)
    replacement_dir.mkdir(parents=True, exist_ok=True)

    scene = build_scene()
    views = render_views(scene)
    for index, image in enumerate(views):
        write_png_rgb(view_dir / f"view_{index:02d}.png", image)

    replacement = views[1].copy()
    target_block = scene["view_blocks"][1][0]
    _paint_marker(replacement, target_block[0], target_block[1], (250, 250, 250), variant=1)
    write_png_rgb(replacement_dir / "view_01.png", replacement)

    cameras = camera_manifest()
    (data_dir / "cameras.json").write_text(
        json.dumps(cameras, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (data_dir / "scene_truth.json").write_text(
        json.dumps(scene, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    config = default_config()
    (config_dir / "reconstruction.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    commands = edit_commands()
    (config_dir / "edit_commands.json").write_text(
        json.dumps(commands, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"scene": scene, "config": config, "cameras": cameras, "commands": commands}
