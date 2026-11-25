"""Project configuration, deterministic hashes and load-time validation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .errors import E_PARAM_INVALID, E_VOXEL_ALIGN, diagnostic
from .fixed import digest
from .pngio import read_png_checked


DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": 1,
    "project_name": "mgirecon-demo",
    "scene_seed": 20261126,
    "resolution": {"width": 128, "height": 96},
    "fixed_bits": 10,
    "views": [
        {
            "id": "v01",
            "image": "views/v01.png",
            "render_seed": 1101,
            "intrinsics": [[140, 0, 64], [0, 140, 48], [0, 0, 1]],
            "extrinsics": [[1, 0, 0, -24], [0, 1, 0, -12], [0, 0, 1, 200]],
        },
        {
            "id": "v02",
            "image": "views/v02.png",
            "render_seed": 1102,
            "intrinsics": [[140, 0, 64], [0, 140, 48], [0, 0, 1]],
            "extrinsics": [[1, 0, 0, -8], [0, 1, 0, -4], [0, 0, 1, 200]],
        },
        {
            "id": "v03",
            "image": "views/v03.png",
            "render_seed": 1103,
            "intrinsics": [[140, 0, 64], [0, 140, 48], [0, 0, 1]],
            "extrinsics": [[1, 0, 0, 8], [0, 1, 0, 4], [0, 0, 1, 200]],
        },
        {
            "id": "v04",
            "image": "views/v04.png",
            "render_seed": 1104,
            "intrinsics": [[140, 0, 64], [0, 140, 48], [0, 0, 1]],
            "extrinsics": [[1, 0, 0, 24], [0, 1, 0, 12], [0, 0, 1, 200]],
        },
    ],
    "matching": {"min_inliers": 2, "max_pairs": 6},
    "reconstruction": {
        "min_disparity": 5,
        "residual_threshold": 30,
        "voxel_size": 32,
        "bounds": {"min": [-64, -48, -32], "max": [64, 48, 64]},
        "quantization_levels": [1, 2, 4],
        "quantization_level": 1,
        "voxel_max_depth": 4,
        "min_occupancy_points": 1,
    },
    "scheduler": {
        "max_workers": 4,
        "global_point_budget": 128,
        "per_view_point_quota": 48,
    },
}


def default_config() -> dict[str, Any]:
    return copy.deepcopy(DEFAULT_CONFIG)


def load_config(project_dir: str | Path, filename: str = "config.json") -> dict[str, Any]:
    path = Path(project_dir) / filename
    return json.loads(path.read_text(encoding="utf-8"))


def save_config(project_dir: str | Path, config: dict[str, Any], filename: str = "config.json") -> Path:
    path = Path(project_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def config_hash(config: dict[str, Any]) -> str:
    return digest(config)


def camera_hash(view: dict[str, Any]) -> str:
    return digest(
        {
            "view": view.get("id"),
            "intrinsics": view.get("intrinsics"),
            "extrinsics": view.get("extrinsics"),
        }
    )


def det3(matrix: list[list[int]]) -> int:
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _int_matrix(value: Any, rows: int, cols: int) -> list[list[int]] | None:
    if not isinstance(value, list) or len(value) != rows:
        return None
    result: list[list[int]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != cols or not all(_is_int(v) for v in row):
            return None
        result.append(list(row))
    return result


def rotation_is_orthogonal(rotation: list[list[int]]) -> bool:
    for i in range(3):
        for j in range(3):
            dot = sum(rotation[k][i] * rotation[k][j] for k in range(3))
            if i == j:
                if dot <= 0:
                    return False
            elif dot != 0:
                return False
    diagonal = [sum(rotation[k][i] * rotation[k][i] for k in range(3)) for i in range(3)]
    return diagonal[0] == diagonal[1] == diagonal[2]


def voxel_size_vector(value: Any) -> list[int] | None:
    if _is_int(value):
        if value <= 0:
            return None
        return [value, value, value]
    if isinstance(value, list) and len(value) == 3 and all(_is_int(v) and v > 0 for v in value):
        return list(value)
    return None


def validate_config(config: dict[str, Any], project_dir: str | Path) -> dict[str, Any]:
    """Validate all independent inputs while retaining usable views.

    Validation never raises for bad user data.  Errors are returned in a stable
    order: project-level errors first, then view-id order.
    """

    root = Path(project_dir)
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    valid_views: list[dict[str, Any]] = []
    image_cache: dict[str, dict[str, Any]] = {}

    resolution = config.get("resolution")
    width = height = 0
    if (
        not isinstance(resolution, dict)
        or not _is_int(resolution.get("width"))
        or not _is_int(resolution.get("height"))
        or resolution.get("width", 0) <= 0
        or resolution.get("height", 0) <= 0
    ):
        errors.append(diagnostic(E_PARAM_INVALID, "resolution must contain positive integer width and height", stage="load"))
    else:
        width = resolution["width"]
        height = resolution["height"]

    fixed_bits = config.get("fixed_bits")
    if not _is_int(fixed_bits) or not 4 <= fixed_bits <= 24:
        errors.append(diagnostic(E_PARAM_INVALID, "fixed_bits must be an integer in [4, 24]", stage="load"))
        fixed_bits = 10

    reconstruction = config.get("reconstruction")
    if not isinstance(reconstruction, dict):
        errors.append(diagnostic(E_PARAM_INVALID, "reconstruction object is required", stage="load"))
        reconstruction = {}

    min_disparity = reconstruction.get("min_disparity")
    if not _is_int(min_disparity) or min_disparity <= 0:
        errors.append(diagnostic(E_PARAM_INVALID, "min_disparity must be a positive integer", stage="load"))
        min_disparity = 1
    residual_threshold = reconstruction.get("residual_threshold")
    if not _is_int(residual_threshold) or residual_threshold <= 0:
        errors.append(diagnostic(E_PARAM_INVALID, "residual_threshold must be a positive integer", stage="load"))
        residual_threshold = 30

    quantization_levels = reconstruction.get("quantization_levels")
    if (
        not isinstance(quantization_levels, list)
        or not quantization_levels
        or not all(_is_int(v) and v > 0 for v in quantization_levels)
    ):
        errors.append(diagnostic(E_PARAM_INVALID, "quantization_levels must contain positive integers", stage="load"))
        quantization_levels = [1]
    quantization_level = reconstruction.get("quantization_level")
    if quantization_level not in quantization_levels:
        errors.append(diagnostic(E_PARAM_INVALID, "quantization_level must be one of quantization_levels", stage="load"))
        quantization_level = quantization_levels[0]

    bounds = reconstruction.get("bounds")
    bounds_ok = (
        isinstance(bounds, dict)
        and isinstance(bounds.get("min"), list)
        and isinstance(bounds.get("max"), list)
        and len(bounds["min"]) == 3
        and len(bounds["max"]) == 3
        and all(_is_int(v) for v in bounds["min"] + bounds["max"])
        and all(bounds["max"][i] > bounds["min"][i] for i in range(3))
    )
    if not bounds_ok:
        errors.append(diagnostic(E_PARAM_INVALID, "bounds must be integer min/max with positive extents", stage="load"))
        bounds = {"min": [-64, -48, -32], "max": [64, 48, 64]}

    voxel_size = voxel_size_vector(reconstruction.get("voxel_size"))
    if voxel_size is None:
        errors.append(diagnostic(E_PARAM_INVALID, "voxel_size must be a positive integer or three positive integers", stage="load"))
        voxel_size = [32, 32, 32]
    if bounds_ok:
        extents = [bounds["max"][i] - bounds["min"][i] for i in range(3)]
        if any(extents[i] % voxel_size[i] != 0 for i in range(3)):
            errors.append(
                diagnostic(
                    E_VOXEL_ALIGN,
                    "voxel_size must divide each reconstruction extent",
                    stage="voxel",
                    detail={"extents": extents, "voxel_size": voxel_size},
                )
            )

    views = config.get("views")
    if not isinstance(views, list) or not views:
        errors.append(diagnostic(E_PARAM_INVALID, "views must be a non-empty list", stage="load"))
        views = []

    seen_ids: set[str] = set()
    seen_images: set[str] = set()
    for raw_view in sorted(views, key=lambda item: str(item.get("id", "")) if isinstance(item, dict) else ""):
        if not isinstance(raw_view, dict):
            errors.append(diagnostic(E_PARAM_INVALID, "view entry must be an object", stage="load"))
            continue
        view_id = raw_view.get("id")
        if not isinstance(view_id, str) or not view_id:
            errors.append(diagnostic(E_PARAM_INVALID, "view id must be a non-empty string", stage="load"))
            continue
        if view_id in seen_ids:
            errors.append(diagnostic(E_PARAM_INVALID, "view id must be unique", stage="load", view=view_id))
            continue
        seen_ids.add(view_id)

        intrinsics = _int_matrix(raw_view.get("intrinsics"), 3, 3)
        if intrinsics is None or det3(intrinsics) == 0:
            errors.append(diagnostic(E_PARAM_INVALID, "intrinsic matrix must be an invertible 3x3 integer matrix", stage="load", view=view_id))
            continue

        extrinsics = _int_matrix(raw_view.get("extrinsics"), 3, 4)
        if extrinsics is None:
            errors.append(diagnostic(E_PARAM_INVALID, "extrinsic matrix must be a 3x4 integer matrix", stage="load", view=view_id))
            continue
        rotation = [row[:3] for row in extrinsics]
        if not rotation_is_orthogonal(rotation):
            errors.append(diagnostic(E_PARAM_INVALID, "extrinsic rotation must be orthogonal", stage="load", view=view_id))
            continue

        image_field = raw_view.get("image")
        if not isinstance(image_field, str) or not image_field:
            errors.append(diagnostic(E_PARAM_INVALID, "view image path is required", stage="load", view=view_id))
            continue
        image_path = (root / image_field).resolve()
        image_key = str(image_path)
        if image_key in seen_images:
            errors.append(diagnostic(E_PARAM_INVALID, "each view must reference a distinct image", stage="load", view=view_id))
            continue
        seen_images.add(image_key)

        image, image_error = read_png_checked(
            image_path,
            expected_width=width,
            expected_height=height,
            view_id=view_id,
        )
        if image_error:
            errors.append(image_error)
            continue
        assert image is not None
        image_cache[view_id] = {
            "path": str(image_path),
            "sha256": image.raw_sha256,
            "width": image.width,
            "height": image.height,
            "bit_depth": image.bit_depth,
            "color_type": image.color_type,
        }
        valid_views.append(
            {
                "id": view_id,
                "image": image_field,
                "image_path": str(image_path),
                "image_sha256": image.raw_sha256,
                "render_seed": raw_view.get("render_seed", 0),
                "intrinsics": intrinsics,
                "extrinsics": extrinsics,
                "camera_hash": camera_hash({**raw_view, "intrinsics": intrinsics, "extrinsics": extrinsics}),
            }
        )

    valid_views.sort(key=lambda view: view["id"])
    matching = config.get("matching") if isinstance(config.get("matching"), dict) else {}
    min_inliers = matching.get("min_inliers", 2)
    max_pairs = matching.get("max_pairs", 12)
    if not _is_int(min_inliers) or min_inliers <= 0:
        errors.append(diagnostic(E_PARAM_INVALID, "matching.min_inliers must be a positive integer", stage="load"))
        min_inliers = 1
    if not _is_int(max_pairs) or max_pairs <= 0:
        errors.append(diagnostic(E_PARAM_INVALID, "matching.max_pairs must be a positive integer", stage="load"))
        max_pairs = 12

    scheduler = config.get("scheduler") if isinstance(config.get("scheduler"), dict) else {}
    max_workers = scheduler.get("max_workers", 1)
    global_budget = scheduler.get("global_point_budget", 1000000)
    per_view_quota = scheduler.get("per_view_point_quota", 1000000)
    if not _is_int(max_workers) or max_workers <= 0:
        errors.append(diagnostic(E_PARAM_INVALID, "scheduler.max_workers must be a positive integer", stage="load"))
        max_workers = 1
    if not _is_int(global_budget) or global_budget < 0:
        errors.append(diagnostic(E_PARAM_INVALID, "scheduler.global_point_budget must be a non-negative integer", stage="load"))
        global_budget = 1000000
    if not _is_int(per_view_quota) or per_view_quota < 0:
        errors.append(diagnostic(E_PARAM_INVALID, "scheduler.per_view_point_quota must be a non-negative integer", stage="load"))
        per_view_quota = 1000000

    normalized = copy.deepcopy(config)
    normalized["resolution"] = {"width": width, "height": height}
    normalized["fixed_bits"] = fixed_bits
    normalized["matching"] = {"min_inliers": min_inliers, "max_pairs": max_pairs}
    normalized["reconstruction"] = {
        **reconstruction,
        "min_disparity": min_disparity,
        "residual_threshold": residual_threshold,
        "voxel_size": voxel_size,
        "bounds": bounds,
        "quantization_levels": quantization_levels,
        "quantization_level": quantization_level,
        "voxel_max_depth": reconstruction.get("voxel_max_depth", 4)
        if _is_int(reconstruction.get("voxel_max_depth")) and reconstruction.get("voxel_max_depth", 0) >= 0
        else 4,
        "min_occupancy_points": reconstruction.get("min_occupancy_points", 1)
        if _is_int(reconstruction.get("min_occupancy_points")) and reconstruction.get("min_occupancy_points", 0) >= 0
        else 1,
    }
    normalized["scheduler"] = {
        "max_workers": max_workers,
        "global_point_budget": global_budget,
        "per_view_point_quota": per_view_quota,
    }
    normalized["views"] = sorted(views, key=lambda item: str(item.get("id", ""))) if isinstance(views, list) else []

    return {
        "config": normalized,
        "valid_views": valid_views,
        "images": image_cache,
        "errors": errors,
        "warnings": warnings,
        "config_hash": config_hash(config),
        "camera_hash": digest([{view["id"]: view["camera_hash"]} for view in valid_views]),
        "voxel_aligned": not any(error["code"] == E_VOXEL_ALIGN for error in errors),
    }
