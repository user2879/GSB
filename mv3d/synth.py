"""Deterministic local multi-view sample generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .constants import SCHEMA_VERSION
from .fixed import stable_hash
from .geometry import project_world
from .io import write_json_atomic
from .models import Camera, ProjectState, parse_camera

MARKER_COLORS = (
    (224, 38, 44),
    (35, 214, 72),
    (42, 74, 228),
    (232, 208, 35),
    (216, 46, 204),
    (35, 206, 205),
    (238, 124, 28),
    (139, 68, 230),
    (33, 164, 111),
    (238, 84, 134),
    (122, 214, 34),
    (45, 146, 238),
)

VIEW_ANGLES = (
    (4, -3, 5, 1, 0, 1),
    (99, -20, 101, 99, 20, 101),
    (99, 20, 101, 99, -20, 101),
    (4, 3, 5, 1, 0, 1),
)

MARKER_POINTS = (
    (-22, -14, -3),
    (0, -16, 2),
    (22, -12, -2),
    (-24, 1, 5),
    (24, 1, -5),
    (-18, 14, 3),
    (0, 15, -2),
    (18, 14, 1),
    (24, 0, -32),
    (12, 0, -16),
)

CAMERA_DISTANCE = 80
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 48
INTRINSIC = ((60, 0, 32), (0, 60, 24), (0, 0, 1))


def _rotation_numerator(view_id: int) -> tuple[tuple[tuple[int, int, int], ...], int]:
    index = (view_id - 1) % len(VIEW_ANGLES)
    cy, sy, dy, cp, sp, dp = VIEW_ANGLES[index]
    ry = ((cy, 0, sy), (0, dy, 0), (-sy, 0, cy))
    rx = ((dp, 0, 0), (0, cp, -sp), (0, sp, cp))
    rotation = tuple(
        tuple(sum(rx[row][k] * ry[k][column] for k in range(3)) for column in range(3))
        for row in range(3)
    )
    return rotation, dy * dp


def demo_camera(view_id: int) -> Camera:
    rotation, scale = _rotation_numerator(view_id)
    return Camera(
        intrinsic=INTRINSIC,
        rotation=rotation,
        rotation_scale=scale,
        translation=(0, 0, CAMERA_DISTANCE),
    )


def _next_seed(value: int) -> int:
    return (value * 1_103_515_245 + 12_345) & 0x7FFFFFFF


def _background(seed: int, width: int, height: int) -> np.ndarray:
    state = seed & 0x7FFFFFFF
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            state = _next_seed(state)
            tile = ((x // 4) * 7 + (y // 4) * 11) & 15
            noise = (state >> 16) & 7
            image[y, x] = (
                20 + tile + noise,
                25 + (tile // 2) + noise,
                31 + (tile // 3) + noise,
            )
    return image


def _draw_primitives(image: np.ndarray, camera: Camera, seed: int) -> np.ndarray:
    picture = Image.fromarray(image, mode="RGB")
    draw = ImageDraw.Draw(picture)
    shapes = (
        (((-31, -20, 8), (12, -32, -4), (28, -8, 13)), (38, 54, 61)),
        (((-27, 8, -7), (5, 27, 9), (31, 4, -11)), (44, 58, 52)),
        (((-8, -25, 12), (30, 17, 6), (9, 24, -9)), (35, 50, 64)),
    )
    for triangle, color in shapes:
        projected = [project_world(camera, *point) for point in triangle]
        if any(point[2] <= 0 for point in projected):
            continue
        points = [(point[0], point[1]) for point in projected]
        draw.polygon(points, fill=color)
    state = seed & 0x7FFFFFFF
    for _ in range(12):
        state = _next_seed(state)
        x0 = (state >> 8) % max(1, image.shape[1] - 5)
        y0 = (state >> 16) % max(1, image.shape[0] - 5)
        shade = 34 + (state & 15)
        draw.rectangle(
            (x0, y0, x0 + 3, y0 + 3),
            fill=(shade, shade + 3, shade + 7),
            outline=(shade - 8, shade - 5, shade - 1),
        )
    return np.array(picture, dtype=np.uint8, copy=True)


def _paint_marker(
    image: np.ndarray,
    center_x: int,
    center_y: int,
    color: tuple[int, int, int],
) -> None:
    height, width = image.shape[:2]
    radius = 4
    for dy in range(-radius, radius + 1):
        y = center_y + dy
        if y < 0 or y >= height:
            continue
        for dx in range(-radius, radius + 1):
            x = center_x + dx
            if x < 0 or x >= width:
                continue
            if abs(dx) == radius or abs(dy) == radius:
                image[y, x] = (14, 14, 14)
            else:
                delta = ((dx + dy) & 1) * 24 + dx * 3 + dy * 2
                image[y, x] = tuple(
                    min(255, max(0, channel + delta)) for channel in color
                )


def render_view(seed: int, camera: Camera) -> np.ndarray:
    image = _background(seed, IMAGE_WIDTH, IMAGE_HEIGHT)
    image = _draw_primitives(image, camera, seed)
    visible = []
    for index, point in enumerate(MARKER_POINTS):
        u, v, depth = project_world(camera, *point)
        if depth > 0 and 5 <= u < IMAGE_WIDTH - 5 and 5 <= v < IMAGE_HEIGHT - 5:
            visible.append((depth, index, point, u, v))
    for _, index, _, u, v in sorted(visible, reverse=True):
        _paint_marker(image, u, v, MARKER_COLORS[index])
    return image


def _camera_payload(camera: Camera) -> dict[str, Any]:
    return {
        "intrinsic": [list(row) for row in camera.intrinsic],
        "rotation": [list(row) for row in camera.rotation],
        "rotation_scale": camera.rotation_scale,
        "translation": list(camera.translation),
    }


def demo_project(seed: int = 7) -> dict[str, Any]:
    views = []
    for view_id in range(1, 5):
        camera = demo_camera(view_id)
        views.append(
            {
                "id": view_id,
                "image": f"views/view-{view_id:04d}.png",
                "camera": _camera_payload(camera),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "recipe_version": "mv3d-demo-1",
        "image": {"width": IMAGE_WIDTH, "height": IMAGE_HEIGHT, "mode": "RGB"},
        "fixed": {"bits": 16, "quant_level": 2},
        "reconstruction": {
            "min_disparity": 1,
            "voxel_size": 16,
            "residual_threshold": 1,
            "max_subdiv_depth": 2,
            "max_voxels_per_parent": 8,
            "world_bounds": [-32, -32, -32, 32, 32, 32],
            "global_point_budget": 160,
            "view_point_quota": 32,
            "workers": 4,
            "strategy_override": None,
            "orthogonality_tolerance_ppm": 10,
        },
        "pairs": [[1, 2], [1, 3], [2, 3], [2, 4], [3, 4]],
        "views": views,
    }


def demo_commands() -> list[dict[str, Any]]:
    return [
        {"op": "replace_view", "view_id": 2, "path": "replacements/view-0002-alt.png"},
        {"op": "update_camera_extrinsics", "view_id": 3, "translation": [0, 0, 86]},
        {"op": "set_min_disparity", "value": 2},
        {"op": "set_voxel_size", "value": 12},
        {"op": "set_quantization", "bits": 16, "quant_level": 4},
    ]


def generate_demo(root: Path, seed: int = 7, force: bool = False) -> Path:
    root = root.resolve()
    if root.exists() and any(root.iterdir()) and not force:
        raise FileExistsError(f"refusing to overwrite non-empty directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    views_dir = root / "views"
    replacement_dir = root / "replacements"
    views_dir.mkdir(exist_ok=True)
    replacement_dir.mkdir(exist_ok=True)
    for view_id in range(1, 5):
        camera = demo_camera(view_id)
        image_seed = seed * 1009 + view_id * 47
        pixels = render_view(image_seed, camera)
        path = views_dir / f"view-{view_id:04d}.png"
        Image.fromarray(pixels, mode="RGB").save(path, format="PNG", optimize=False)
        if view_id == 2:
            alternate = render_view(image_seed + 10_007, camera)
            Image.fromarray(alternate, mode="RGB").save(
                replacement_dir / "view-0002-alt.png",
                format="PNG",
                optimize=False,
            )
    project = demo_project(seed)
    write_json_atomic(root / "project.json", project)
    command_text = "\n".join(
        json.dumps(command, ensure_ascii=False, sort_keys=True) for command in demo_commands()
    )
    (root / "commands.jsonl").write_text(command_text + "\n", encoding="utf-8")
    marker_points = [
        {"index": index, "world": list(point), "color": list(MARKER_COLORS[index])}
        for index, point in enumerate(MARKER_POINTS)
    ]
    write_json_atomic(
        root / "scene.json",
        {
            "seed": seed,
            "resolution": [IMAGE_WIDTH, IMAGE_HEIGHT],
            "marker_points": marker_points,
            "scene_fingerprint": stable_hash(marker_points),
        },
    )
    return root


def load_project(path: Path) -> ProjectState:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("project root must be an object")
    return ProjectState(data=raw, base_dir=path.resolve().parent, project_path=path.resolve())


def generate_replacement(
    project: ProjectState,
    view_id: int,
    destination: Path,
    seed: int = 10_007,
) -> Path:
    view = next(item for item in project.data["views"] if int(item["id"]) == view_id)
    camera = parse_camera(view["camera"])
    pixels = render_view(seed + view_id * 997, camera)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="RGB").save(destination, format="PNG", optimize=False)
    return destination
