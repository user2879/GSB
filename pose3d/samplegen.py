"""Deterministic synthetic sample generator."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from .hashing import sha256_file


SEED = 20261122
WIDTH = 800
HEIGHT = 600
VIEW_ANGLES_DEG = [0.0, 10.0, 15.0, 80.0, 170.0, 260.0]
TARGET = np.array([0.0, 0.0, 0.2], dtype=np.float64)
UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)
RADIUS = 3.2


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("cannot normalize zero vector")
    return vector / norm


def _look_at_pose(center: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z_axis = _normalize(target - center)
    x_axis = _normalize(np.cross(z_axis, UP))
    y_axis = np.cross(z_axis, x_axis)
    R = np.stack([x_axis, y_axis, z_axis], axis=0)
    t = -R @ center
    return R, t


def _camera_center(angle_deg: float, elevation: float = 0.45) -> np.ndarray:
    angle = math.radians(angle_deg)
    return np.array(
        [RADIUS * math.cos(angle), RADIUS * math.sin(angle), elevation],
        dtype=np.float64,
    )


def _intrinsics() -> np.ndarray:
    return np.array(
        [[760.0, 0.0, WIDTH / 2.0], [0.0, 760.0, HEIGHT / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _project(K: np.ndarray, R: np.ndarray, t: np.ndarray, point: np.ndarray) -> tuple[float, float]:
    P = K @ np.hstack((R, t.reshape(3, 1)))
    value = P @ np.array([point[0], point[1], point[2], 1.0], dtype=np.float64)
    if value[2] <= 1e-9:
        raise ValueError("sample point is behind a camera")
    return float(value[0] / value[2]), float(value[1] / value[2])


def _cube_corners(center: np.ndarray, half: float) -> list[np.ndarray]:
    offsets = [
        (-half, -half, -half),
        (half, -half, -half),
        (half, half, -half),
        (-half, half, -half),
        (-half, -half, half),
        (half, -half, half),
        (half, half, half),
        (-half, half, half),
    ]
    return [center + np.array(offset, dtype=np.float64) for offset in offsets]


def _point_manifest() -> tuple[list[tuple[str, np.ndarray, list[int]]], list[tuple[str, np.ndarray, list[int]]]]:
    """Return normal observations and the intentionally fragile group.

    The first six A-instance and six B-instance keypoints share base names but
    use distinct instance suffixes, exercising same-name disambiguation.
    """

    normal: list[tuple[str, np.ndarray, list[int]]] = []
    fragile: list[tuple[str, np.ndarray, list[int]]] = []
    centers = {
        "A": np.array([-0.45, 0.0, 0.2], dtype=np.float64),
        "B": np.array([0.45, 0.15, 0.2], dtype=np.float64),
    }
    half_sizes = {"A": 0.34, "B": 0.30}
    for instance in ("A", "B"):
        corners = _cube_corners(centers[instance], half_sizes[instance])
        for index in range(6):
            normal.append(
                (
                    f"rig/corner_{index:02d}@instance-{instance}",
                    corners[index],
                    list(range(1, 7)),
                )
            )
    normal.extend(
        [
            ("aux/marker@instance-A", np.array([0.0, 0.68, 0.62]), list(range(1, 7))),
            ("bounds/tip@instance-A", np.array([-0.12, 0.72, 0.18]), list(range(1, 7))),
        ]
    )
    fragile.extend(
        [
            (
                "fragile/tip@instance-A",
                np.array([0.08, -0.58, 0.55]),
                [3, 4, 5],
            ),
            (
                "low/parallax@instance-A",
                np.array([0.0, 0.72, 0.45]),
                [1, 2, 3],
            ),
        ]
    )
    normal.sort(key=lambda item: item[0])
    fragile.sort(key=lambda item: item[0])
    return normal, fragile


def _draw_view(
    gray: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    cube_corners: list[list[np.ndarray]],
    marker_uv: list[tuple[float, float]],
) -> None:
    image = Image.fromarray(gray, mode="L")
    draw = ImageDraw.Draw(image)
    # Background geometry: a few large deterministic polygons.
    draw.polygon([(40, 520), (250, 360), (420, 560)], fill=215, outline=90)
    draw.rectangle((540, 55, 760, 245), outline=80, width=3, fill=235)
    draw.ellipse((55, 55, 210, 210), outline=70, width=3)
    draw.line((0, HEIGHT - 1, WIDTH - 1, 0), fill=120, width=2)

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for corners in cube_corners:
        projected = [_project(K, R, t, corner) for corner in corners]
        for left, right in edges:
            draw.line(
                (projected[left][0], projected[left][1], projected[right][0], projected[right][1]),
                fill=55,
                width=2,
            )
        for u, v in projected:
            draw.ellipse((u - 3, v - 3, u + 3, v + 3), fill=30)
    for index, (u, v) in enumerate(marker_uv):
        radius = 5 if index % 2 == 0 else 4
        draw.ellipse(
            (u - radius, v - radius, u + radius, v + radius),
            outline=0 if index % 3 else 245,
            width=2,
        )
        draw.line((u - 7, v, u + 7, v), fill=255, width=1)
        draw.line((u, v - 7, u, v + 7), fill=255, width=1)
    gray[:] = np.asarray(image, dtype=np.uint8)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def generate_sample(output_dir: str | Path, *, overwrite: bool = True) -> dict[str, Any]:
    out = Path(output_dir)
    if out.exists() and overwrite:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "views").mkdir()
    (out / "cameras").mkdir()
    rng = np.random.default_rng(SEED)
    K = _intrinsics()
    cameras: list[dict[str, Any]] = []
    for view_index, angle in enumerate(VIEW_ANGLES_DEG, start=1):
        center = _camera_center(angle)
        R, t = _look_at_pose(center, TARGET)
        cameras.append(
            {
                "view_index": view_index,
                "image": f"views/view_{view_index:02d}.png",
                "width": WIDTH,
                "height": HEIGHT,
                "K": K.tolist(),
                "R": R.tolist(),
                "t": t.tolist(),
            }
        )
        _write_json(out / "cameras" / f"view_{view_index:02d}.json", cameras[-1])

    normal_points, fragile_points = _point_manifest()
    corner_sets = [
        _cube_corners(np.array([-0.45, 0.0, 0.2]), 0.34),
        _cube_corners(np.array([0.45, 0.15, 0.2]), 0.30),
    ]
    per_view_markers: dict[int, list[tuple[float, float]]] = {i: [] for i in range(1, 7)}
    for _, point, views in normal_points + fragile_points:
        for view_index in views:
            camera = cameras[view_index - 1]
            u, v = _project(
                K,
                np.asarray(camera["R"], dtype=np.float64),
                np.asarray(camera["t"], dtype=np.float64),
                point,
            )
            per_view_markers[view_index].append((u, v))
    for view_index in range(1, 7):
        camera = cameras[view_index - 1]
        gray = np.full((HEIGHT, WIDTH), 247, dtype=np.uint8)
        _draw_view(
            gray,
            K,
            np.asarray(camera["R"], dtype=np.float64),
            np.asarray(camera["t"], dtype=np.float64),
            corner_sets,
            per_view_markers[view_index],
        )
        Image.fromarray(gray, mode="L").save(out / "views" / f"view_{view_index:02d}.png")

    noise_scale = 0.11
    lines: list[dict[str, Any]] = []
    for keypoint_id, point, views in normal_points:
        observations = []
        for view_index in views:
            camera = cameras[view_index - 1]
            u, v = _project(
                K,
                np.asarray(camera["R"], dtype=np.float64),
                np.asarray(camera["t"], dtype=np.float64),
                point,
            )
            u += float(rng.uniform(-noise_scale, noise_scale))
            v += float(rng.uniform(-noise_scale, noise_scale))
            if keypoint_id == "rig/corner_02@instance-A" and view_index == 4:
                u += 31.0
                v -= 19.0
            observations.append({"view_index": view_index, "u": u, "v": v})
        if keypoint_id == "bounds/tip@instance-A":
            observations.append({"view_index": 6, "u": -3.5, "v": 224.0})
        lines.append({"keypoint_id": keypoint_id, "observations": observations})
    for keypoint_id, point, views in fragile_points:
        observations = []
        for view_index in views:
            camera = cameras[view_index - 1]
            u, v = _project(
                K,
                np.asarray(camera["R"], dtype=np.float64),
                np.asarray(camera["t"], dtype=np.float64),
                point,
            )
            u += float(rng.uniform(-noise_scale, noise_scale))
            v += float(rng.uniform(-noise_scale, noise_scale))
            if keypoint_id == "fragile/tip@instance-A" and view_index == 4:
                u += 43.0
                v += 31.0
            observations.append({"view_index": view_index, "u": u, "v": v})
        lines.append({"keypoint_id": keypoint_id, "observations": observations})
    lines.sort(key=lambda item: item["keypoint_id"])
    with (out / "observations.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for line in lines:
            handle.write(json.dumps(line, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")

    config = {
        "schema_version": 1,
        "inputs": {
            "image_dir": "views",
            "camera_dir": "cameras",
            "observations_file": "observations.jsonl",
            "version_file": "input_version.json",
        },
        "triangulation": {
            "min_views": 3,
            "min_disparity_deg": 20.0,
            "legal_min_disparity_max_deg": 90.0,
            "degenerate_relative_tolerance": 1e-10,
            "degenerate_policy": "fail",
        },
        "outlier_rejection": {
            "max_rounds": 2,
            "threshold_px": 1.5,
            "legal_threshold_max_px": 1000.0,
            "min_inliers": 3,
        },
        "quantization": {
            "levels": [14, 15],
            "rounding": "round_half_even",
            "error_tolerance": 0.005,
        },
        "budget": {
            "unit": "reprojection_evaluation",
            "global_residual_budget": 20000,
            "per_keypoint_quota": 2000,
        },
        "scheduler": {
            "max_workers": 4,
            "snapshot_commit_interval": 2,
            "report_precision_digits": 15,
        },
    }
    _write_json(out / "solver_config.json", config)
    injections = {
        "schema_version": 1,
        "injection_version": "inj-2026-11-22-v1",
        "items": [
            {
                "id": "INJ-001-NONFINITE-PIXEL",
                "keypoint_id": "rig/corner_00@instance-A",
                "stage": "load",
                "fault": "nonfinite_pixel",
                "view_index": 3,
                "field": "u",
                "value": "NaN",
                "business_code": "E_OBS_INVALID",
            },
            {
                "id": "INJ-002-NONORTHOGONAL-ROTATION",
                "keypoint_id": "rig/corner_01@instance-B",
                "stage": "load",
                "fault": "nonorthogonal_rotation",
                "view_index": 2,
                "value": 0.25,
                "business_code": "E_CAMERA_INVALID",
            },
            {
                "id": "INJ-003-DISPARITY-ABOVE-LIMIT",
                "keypoint_id": "rig/corner_01@instance-A",
                "stage": "triangulation",
                "fault": "min_disparity_above_limit",
                "value": 180.0,
                "business_code": "E_CFG_INVALID",
            },
            {
                "id": "INJ-004-ZERO-THRESHOLD",
                "keypoint_id": "rig/corner_03@instance-B",
                "stage": "rejection",
                "fault": "zero_outlier_threshold",
                "value": 0.0,
                "business_code": "E_CFG_INVALID",
            },
        ],
    }
    _write_json(out / "injections.json", injections)
    version_info = {
        "manifest_version": "obs-2026-11-22-v1",
        "config_version": "cfg-2026-11-22-v1",
        "injection_version": "inj-2026-11-22-v1",
        "seed": SEED,
        "generator": "pose3d.samplegen/1.0.0",
        "image_resolution": [WIDTH, HEIGHT],
        "observations_sha256": sha256_file(out / "observations.jsonl"),
        "config_sha256": sha256_file(out / "solver_config.json"),
        "injections_sha256": sha256_file(out / "injections.json"),
    }
    _write_json(out / "input_version.json", version_info)
    return {
        "output_dir": str(out),
        "views": len(cameras),
        "keypoints": len(lines),
        "manifest_version": version_info["manifest_version"],
        "observations_sha256": version_info["observations_sha256"],
    }
