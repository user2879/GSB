"""Deterministic offline sample generator."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .util import atomic_write_bytes, atomic_write_json


def _proper_rotation(rng: np.random.Generator) -> np.ndarray:
    matrix = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    q = q @ np.diag(signs)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q.astype(np.float64)


def _make_camera(view_index: int, width: int, height: int, rng: np.random.Generator) -> dict[str, Any]:
    focal = float(width) * (0.78 + 0.18 * rng.random())
    intrinsic = np.array(
        [[focal, 0.0, (width - 1) * 0.5], [0.0, focal, (height - 1) * 0.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    rotation = _proper_rotation(rng)
    translation = rng.uniform(-0.75, 0.75, size=3).astype(np.float64)
    return {
        "intrinsic": intrinsic.tolist(),
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "view_index": int(view_index),
    }


def make_view_payload(
    view_index: int,
    width: int,
    height: int,
    seed: int,
    variant: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Render a light-background RGB image and a matching depth texture."""

    rng = np.random.default_rng(seed + view_index * 1009 + variant * 104729)
    yy, xx = np.mgrid[0:height, 0:width]
    background = np.array([238.0, 242.0, 237.0], dtype=np.float64)
    gradient = np.stack(
        [
            background[0] - 7.0 * (yy / max(1, height - 1)),
            background[1] - 5.0 * (xx / max(1, width - 1)),
            background[2] - 4.0 * ((xx + yy) / max(1, width + height - 2)),
        ],
        axis=-1,
    )
    image = Image.fromarray(np.clip(np.rint(gradient), 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image)
    depth = (0.55 + 0.35 * np.sqrt((xx / width - 0.5) ** 2 + (yy / height - 0.5) ** 2)).astype(np.float64)
    palette = [
        (226, 103, 86),
        (58, 128, 188),
        (214, 164, 63),
        (83, 157, 109),
        (142, 105, 176),
    ]
    shape_count = 3 + int(rng.integers(0, 3))
    for shape_index in range(shape_count):
        color = palette[(view_index + shape_index + variant) % len(palette)]
        kind = shape_index % 3
        cx = int(rng.integers(width // 6, max(width // 6 + 1, width - width // 6)))
        cy = int(rng.integers(height // 6, max(height // 6 + 1, height - height // 6)))
        radius = int(rng.integers(max(3, width // 12), max(4, width // 4)))
        obj_depth = float(0.35 + 0.55 * rng.random() + 0.03 * variant)
        if kind == 0:
            box = [cx - radius, cy - radius, cx + radius, cy + radius]
            draw.ellipse(box, fill=color, outline=(30, 42, 38), width=1)
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
        elif kind == 1:
            box = [cx - radius, cy - radius // 2, cx + radius, cy + radius // 2]
            draw.rectangle(box, fill=color, outline=(30, 42, 38), width=1)
            mask = (
                (xx >= cx - radius)
                & (xx <= cx + radius)
                & (yy >= cy - radius // 2)
                & (yy <= cy + radius // 2)
            )
        else:
            points = [(cx, cy - radius), (cx - radius, cy + radius), (cx + radius, cy + radius)]
            draw.polygon(points, fill=color, outline=(30, 42, 38))
            mask = np.zeros((height, width), dtype=bool)
            v0 = np.array(points[1], dtype=np.float64) - np.array(points[0], dtype=np.float64)
            v1 = np.array(points[2], dtype=np.float64) - np.array(points[0], dtype=np.float64)
            v2 = np.stack([xx - points[0][0], yy - points[0][1]], axis=-1).astype(np.float64)
            denom = v0[0] * v1[1] - v0[1] * v1[0]
            if abs(denom) > 1e-12:
                a = (v2[..., 0] * v1[1] - v2[..., 1] * v1[0]) / denom
                b = (v0[0] * v2[..., 1] - v0[1] * v2[..., 0]) / denom
                mask = (a >= 0) & (b >= 0) & (a + b <= 1)
        depth[mask] = obj_depth
        # Add a deterministic high-frequency texture inside each primitive.
        texture = ((xx + 3 * yy + 7 * shape_index + view_index + variant) % 9) / 9.0
        depth[mask] += 0.012 * texture[mask]
    # Sparse deterministic speckles make image content replacement visible.
    speckle_count = 18 + view_index * 3 + variant * 5
    for index in range(speckle_count):
        sx = int((index * 17 + view_index * 11 + variant * 7 + seed) % width)
        sy = int((index * 29 + view_index * 13 + variant * 3 + seed) % height)
        shade = int(110 + (index * 23 + view_index * 19) % 80)
        draw.point((sx, sy), fill=(shade, min(255, shade + 25), max(0, shade - 20)))
    rgb = np.asarray(image, dtype=np.uint8).copy()
    return rgb, depth.astype(np.float32)


def _write_png(path: Path, rgb: np.ndarray) -> None:
    buffer = path.with_name(path.name + ".tmp")
    Image.fromarray(rgb, mode="RGB").save(buffer, format="PNG", optimize=False)
    os.replace(buffer, path)


def _write_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        with open(temp_name, "wb") as handle:
            np.save(handle, np.asarray(array), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _write_asset(
    asset_dir: Path,
    view_index: int,
    width: int,
    height: int,
    seed: int,
    camera: dict[str, Any],
    variant: int,
) -> None:
    asset_dir.mkdir(parents=True, exist_ok=True)
    rgb, depth = make_view_payload(view_index, width, height, seed, variant=variant)
    _write_png(asset_dir / f"view_{view_index:04d}.png", rgb)
    _write_npy(asset_dir / f"view_{view_index:04d}.npy", depth)
    atomic_write_json(asset_dir / "camera.json", camera)


def default_config(width: int = 64, height: int = 64, concurrency: int = 4) -> dict[str, Any]:
    return {
        "base_scale": 1.2,
        "concurrency": concurrency,
        "frac_bits": 12,
        "image_size": [int(width), int(height)],
        "kernel_budgets": [4096, 4096, 4096],
        "level_density": [1.0, 0.75, 0.5],
        "level_scale_overrides": [None, None, None],
        "levels": 3,
        "pixel_mode": "RGB",
        "point_budget": 20000,
        "quant_gear": "NEAREST",
        "render_samples": 4,
        "scale_decay": 0.6,
        "seed": 20240517,
        "sh_degree": 1,
        "snapshot_interval": 2,
        "tile_size": 16,
        "tolerance": {"composite": 0.02, "kernel": 0.001, "render": 0.02},
    }


def default_edits(view_count: int = 6) -> list[dict[str, Any]]:
    replace_view = 1 if view_count > 1 else 0
    tweak_view = 2 if view_count > 2 else replace_view
    remove_view = 3 if view_count > 3 else view_count - 1
    added_view = view_count
    return [
        {"params": {"asset_dir": "assets/replace_0001", "view_index": replace_view}, "seq": 0, "type": "view_replace"},
        {
            "params": {
                "rotation_delta": [[0.0, 0.01, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                "view_index": tweak_view,
            },
            "seq": 1,
            "type": "camera_tweak",
        },
        {"params": {"level": 1, "value": 0.55}, "seq": 2, "type": "level_scale"},
        {"params": {"level": 2, "value": 0.6}, "seq": 3, "type": "level_density"},
        {
            "params": {
                "channel": 0,
                "coefficient": 1,
                "delta": 0.05,
                "kernel_index": 0,
                "levels": [0, 1],
                "view_index": 0,
            },
            "seq": 4,
            "type": "sh_modify",
        },
        {"params": {"frac_bits": 10, "quant_gear": "FLOOR"}, "seq": 5, "type": "quant_switch"},
        {"params": {"asset_dir": f"assets/add_{added_view:04d}", "view_index": added_view}, "seq": 6, "type": "view_add"},
        {"params": {"view_index": remove_view}, "seq": 7, "type": "view_remove"},
    ]


def default_injections() -> list[dict[str, Any]]:
    return [
        {"injection_id": "INJ-001", "kind": "truncate_image", "view_index": 4},
        {"injection_id": "INJ-002", "kind": "missing_camera", "view_index": 5},
        {"injection_id": "INJ-003", "kind": "zero_scale_decay", "level": 1},
        {
            "injection_id": "INJ-004",
            "kind": "density_over_budget",
            "level": 2,
            "params": {"density": 9000.0},
            "view_index": 0,
        },
        {"injection_id": "INJ-005", "kind": "quant_gear_out_of_range", "params": {"gear": "BOGUS_GEAR"}},
    ]


def generate_sample(
    out_dir: Path,
    *,
    seed: int = 20240517,
    width: int = 64,
    height: int = 64,
    views: int = 6,
    concurrency: int = 4,
    force: bool = False,
) -> Path:
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        if not force:
            raise FileExistsError(f"{out_dir} is not empty; pass force=True or choose a new directory")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir()
    (out_dir / "depth").mkdir()
    (out_dir / "assets").mkdir()

    config = default_config(width, height, concurrency)
    config["seed"] = int(seed)
    atomic_write_json(out_dir / "config.json", config)

    rng = np.random.default_rng(seed)
    cameras: list[dict[str, Any]] = []
    for view_index in range(views):
        camera = _make_camera(view_index, width, height, rng)
        cameras.append(camera)
        rgb, depth = make_view_payload(view_index, width, height, seed)
        _write_png(out_dir / "images" / f"view_{view_index:04d}.png", rgb)
        _write_npy(out_dir / "depth" / f"view_{view_index:04d}.npy", depth)
    atomic_write_json(out_dir / "cameras.json", cameras)

    replace_view = 1 if views > 1 else 0
    _write_asset(
        out_dir / "assets" / "replace_0001",
        replace_view,
        width,
        height,
        seed,
        cameras[replace_view],
        variant=3,
    )
    add_camera = _make_camera(views, width, height, rng)
    _write_asset(out_dir / "assets" / f"add_{views:04d}", views, width, height, seed, add_camera, variant=5)

    with (out_dir / "edits.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for edit in default_edits(views):
            handle.write(json.dumps(edit, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    atomic_write_json(out_dir / "injections.json", default_injections())
    atomic_write_json(out_dir / "sh_mods.json", [])
    return out_dir


def clone_sample(source: Path, target: Path, *, force: bool = True) -> Path:
    source = Path(source)
    target = Path(target)
    if target.exists():
        if not force:
            raise FileExistsError(target)
        shutil.rmtree(target)
    shutil.copytree(source, target)
    return target
