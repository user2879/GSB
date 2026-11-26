from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from .config import DEFAULT_CONFIG
from .utils import write_json


def default_cameras(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    width = int(config["width"])
    height = int(config["height"])
    focal = max(width, height) + 12
    cx = width // 2
    cy = height // 2
    baseline = 6
    cameras: dict[str, dict[str, Any]] = {}
    for sequence in range(4):
        view_id = f"v{sequence}"
        cameras[view_id] = {
            "K": [[focal, 0, cx], [0, focal, cy], [0, 0, 1]],
            "R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "t": [-baseline * sequence, 0, 0],
            "center": [baseline * sequence, 0, 0],
        }
    return cameras


def default_landmarks() -> list[dict[str, int]]:
    landmarks: list[dict[str, int]] = []
    xs = [-6, 4, 14, 24]
    ys = [-10, 0, 10]
    zs = [62, 96, 150, 190]
    index = 0
    for z in zs:
        for y in ys:
            for x in xs:
                landmarks.append({"point_id": f"lm{index:03d}", "x": x, "y": y, "z": z})
                index += 1
    return landmarks


def project_landmark(point: dict[str, int], camera: dict[str, Any]) -> tuple[int, int, int]:
    k = camera["K"]
    fx = int(k[0][0])
    fy = int(k[1][1])
    cx = int(k[0][2])
    cy = int(k[1][2])
    tx, ty, tz = (int(v) for v in camera["t"])
    xc = point["x"] + tx
    yc = point["y"] + ty
    zc = point["z"] + tz
    if zc <= 0:
        raise ValueError("landmark behind camera")
    u = cx + (fx * xc * 2 + zc) // (2 * zc)
    v = cy + (fy * yc * 2 + zc) // (2 * zc)
    return int(u), int(v), int(zc)


def _pattern(appearance: int) -> list[list[tuple[int, int, int]]]:
    rows: list[list[tuple[int, int, int]]] = []
    state = (appearance * 1103515245 + 12345) & 0x7FFFFFFF
    colors = [
        (24, 28, 34),
        (245, 197, 66),
        (224, 76, 76),
        (72, 149, 239),
        (82, 196, 126),
        (245, 245, 245),
    ]
    for _ in range(5):
        row: list[tuple[int, int, int]] = []
        for _ in range(5):
            state = (state * 1103515245 + 12345) & 0x7FFFFFFF
            row.append(colors[(state >> 16) % len(colors)])
        rows.append(row)
    rows[0][0] = (10, 10, 10)
    rows[0][4] = (10, 10, 10)
    rows[4][0] = (10, 10, 10)
    rows[4][4] = (10, 10, 10)
    rows[2][2] = (255, 255, 255)
    return rows


def render_view(
    path: str | Path,
    view_id: str,
    sequence: int,
    camera: dict[str, Any],
    landmarks: list[dict[str, int]],
    config: dict[str, Any],
    seed: int = 400,
    variant: int = 0,
) -> dict[str, Any]:
    width = int(config["width"])
    height = int(config["height"])
    rng = random.Random(int(seed) + sequence * 1009 + variant * 7919)
    image = Image.new("RGB", (width, height), (214, 222, 226))
    draw = ImageDraw.Draw(image)

    block = 12
    for by in range(0, height, block):
        for bx in range(0, width, block):
            delta = rng.randint(-10 - variant * 2, 10 + variant * 2)
            color = (
                max(0, min(255, 214 + delta)),
                max(0, min(255, 222 + delta)),
                max(0, min(255, 226 + delta)),
            )
            draw.rectangle([bx, by, min(width, bx + block - 1), min(height, by + block - 1)], fill=color)

    for index in range(7):
        x0 = rng.randint(0, max(0, width - 24))
        y0 = rng.randint(0, max(0, height - 24))
        x1 = x0 + rng.randint(10, 24)
        y1 = y0 + rng.randint(8, 22)
        color = (180 + rng.randint(-12, 12), 195 + rng.randint(-12, 12), 205 + rng.randint(-12, 12))
        if index % 3 == 0:
            draw.ellipse([x0, y0, x1, y1], outline=color, width=2)
        else:
            draw.rectangle([x0, y0, x1, y1], outline=color, width=2)

    rendered: list[dict[str, Any]] = []
    for index, landmark in enumerate(landmarks):
        u, v, depth = project_landmark(landmark, camera)
        if not (3 <= u < width - 3 and 3 <= v < height - 3):
            continue
        pattern = _pattern(index * 37 + 11)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                draw.point((u + dx, v + dy), fill=pattern[dy + 2][dx + 2])
        rendered.append({"point_id": landmark["point_id"], "u": u, "v": v, "depth": depth})

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", compress_level=9, optimize=False)
    return {"view_id": view_id, "rendered_landmarks": rendered, "variant": variant}


def generate_sample_set(
    target_dir: str | Path,
    config: dict[str, Any] | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    config = config or DEFAULT_CONFIG
    seed = int(config.get("seed", 400) if seed is None else seed)
    target = Path(target_dir)
    view_dir = target / "views"
    view_dir.mkdir(parents=True, exist_ok=True)

    cameras = default_cameras(config)
    landmarks = default_landmarks()
    truth: dict[str, Any] = {"seed": seed, "landmarks": landmarks, "views": {}}
    views: list[dict[str, Any]] = []
    for sequence in range(4):
        view_id = f"v{sequence}"
        render_info = render_view(
            view_dir / f"{view_id}.png",
            view_id,
            sequence,
            cameras[view_id],
            landmarks,
            config,
            seed=seed,
        )
        truth["views"][view_id] = render_info
        views.append({"view_id": view_id, "sequence": sequence, "image": f"views/{view_id}.png"})

    dataset = {
        "dataset_id": "deterministic-wsl-prototype",
        "seed": seed,
        "width": config["width"],
        "height": config["height"],
        "mode": config["mode"],
        "camera_file": "cameras.json",
        "views": views,
        "pairs": [["v0", "v1"], ["v1", "v2"], ["v2", "v3"], ["v0", "v2"]],
    }
    write_json(target / "dataset.json", dataset)
    write_json(target / "cameras.json", {"cameras": cameras})
    write_json(target / "scene_truth.json", truth)
    return {"dataset": dataset, "cameras": cameras, "truth": truth}
